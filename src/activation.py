"""Replay saved generations and capture hidden states at named token positions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.generation import (
    REASONING_INSTRUCTION,
    initial_prompt_content,
    render_initial_prompt,
)
from src.model import LoadedModel


ACTIVATION_PROTOCOL_VERSION = 2
ACTIVATION_SCHEMA_VERSION = 1
CAPTURE_BACKEND = "transformers_hidden_states"
POSITION_PROTOCOL = "semantic_spans_v1"
STORAGE_DTYPE = "float16"
REPLAY_ATOL = 0.01
REPLAY_RTOL = 0.002


class ActivationError(RuntimeError):
    """A saved sequence cannot be replayed or captured safely."""


@dataclass(frozen=True)
class ReplayPlan:
    token_ids: tuple[int, ...]
    positions: dict[str, tuple[int, ...]]
    answer_token_ids: tuple[int, ...]
    answer_token_logprobs: tuple[float, ...]
    replay_sha256: str


@dataclass(frozen=True)
class ActivationResult:
    tensors: dict[str, Any]
    state_labels: tuple[str, ...]
    max_logprob_difference: float


def _token_ids(value: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise ActivationError(f"{name} must be a list of integer token IDs.")
    if nonempty and not value:
        raise ActivationError(f"{name} cannot be empty.")
    return tuple(value)


def _sequence_hash(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_replay_plan(
    generation: Mapping[str, object],
    semantic_positions: Mapping[str, Sequence[int]] | None = None,
) -> ReplayPlan:
    """Build exact replay tokens and default capture positions without retokenizing."""
    if generation.get("status") != "success":
        raise ActivationError("Activation capture requires a successful generation.")

    prompt = _token_ids(
        generation.get("formatted_prompt_token_ids"),
        "formatted_prompt_token_ids",
        nonempty=True,
    )
    reasoning = generation.get("reasoning")
    if reasoning is None:
        reasoning_ids: tuple[int, ...] = ()
    elif isinstance(reasoning, dict):
        reasoning_ids = _token_ids(reasoning.get("token_ids"), "reasoning.token_ids")
    else:
        raise ActivationError("reasoning must be an object or null.")

    control = generation.get("final_control")
    if not isinstance(control, dict):
        raise ActivationError("final_control must be an object.")
    control_ids = _token_ids(
        control.get("token_ids"), "final_control.token_ids", nonempty=True
    )
    marker_span = control.get("final_marker_span")
    if (
        not isinstance(marker_span, list)
        or len(marker_span) != 2
        or not all(type(item) is int for item in marker_span)
        or not 0 <= marker_span[0] < marker_span[1] <= len(control_ids)
    ):
        raise ActivationError("final_control.final_marker_span is invalid.")
    if marker_span[1] != len(control_ids):
        raise ActivationError("The final marker must end the final control tokens.")

    answer = generation.get("answer")
    if not isinstance(answer, dict):
        raise ActivationError("answer must be an object.")
    answer_ids = _token_ids(answer.get("token_ids"), "answer.token_ids", nonempty=True)
    raw_logprobs = answer.get("token_logprobs")
    if (
        not isinstance(raw_logprobs, list)
        or len(raw_logprobs) != len(answer_ids)
        or not all(
            type(item) in (int, float) and math.isfinite(item)
            for item in raw_logprobs
        )
    ):
        raise ActivationError("answer.token_logprobs must match answer.token_ids.")
    answer_logprobs = tuple(float(item) for item in raw_logprobs)

    control_start = len(prompt) + len(reasoning_ids)
    answer_start = control_start + len(control_ids)
    marker_end = control_start + marker_span[1] - 1
    token_ids = (*prompt, *reasoning_ids, *control_ids, *answer_ids)
    positions = {
        "prompt_end": (len(prompt) - 1,),
        "final_prompt_end": (marker_end,),
        "answer_tokens": tuple(range(answer_start, answer_start + len(answer_ids))),
    }
    for name, raw_positions in (semantic_positions or {}).items():
        selected = tuple(raw_positions)
        if (
            name in positions
            or not selected
            or any(type(position) is not int or not 0 <= position < len(prompt)
                   for position in selected)
            or tuple(sorted(set(selected))) != selected
        ):
            raise ActivationError(f"Semantic token positions for '{name}' are invalid.")
        positions[name] = selected
    return ReplayPlan(
        token_ids=token_ids,
        positions=positions,
        answer_token_ids=answer_ids,
        answer_token_logprobs=answer_logprobs,
        replay_sha256=_sequence_hash(token_ids),
    )


def map_semantic_spans(
    tokenizer: Any,
    example: Mapping[str, object],
    generation: Mapping[str, object],
) -> dict[str, tuple[int, ...]]:
    """Map input character spans to exact positions in the saved initial prompt."""
    spans = example.get("semantic_spans", {})
    if not spans:
        return {}
    example_id = example.get("id")
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ActivationError(
            f"Example ID {example_id}: semantic spans require a fast tokenizer."
        )
    task_input = example.get("input")
    control = generation.get("final_control")
    if not isinstance(task_input, str) or not isinstance(control, dict):
        raise ActivationError(
            f"Example ID {example_id}: semantic span mapping inputs are invalid."
        )
    reasoning = generation.get("reasoning") is not None
    instruction = (
        REASONING_INSTRUCTION if reasoning else control.get("answer_instruction")
    )
    if not isinstance(instruction, str):
        raise ActivationError(
            f"Example ID {example_id}: initial prompt instruction is unavailable."
        )
    content = initial_prompt_content(task_input, instruction)
    try:
        rendered = render_initial_prompt(
            tokenizer,
            task_input,
            instruction,
            reasoning=reasoning,
            tokenize=False,
        )
    except ValueError as exc:
        raise ActivationError(f"Example ID {example_id}: {exc}") from exc
    if rendered.count(content) != 1:
        raise ActivationError(
            f"Example ID {example_id}: the chat template must preserve the input "
            "exactly once for semantic span mapping."
        )
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
    except Exception as exc:
        raise ActivationError(
            f"Example ID {example_id}: tokenizer offset mapping failed: {exc}"
        ) from exc
    expected_ids = generation.get("formatted_prompt_token_ids")
    if token_ids != expected_ids:
        raise ActivationError(
            f"Example ID {example_id}: offset tokenization does not match the "
            "saved generation prompt."
        )
    if (
        not isinstance(offsets, list)
        or len(offsets) != len(token_ids)
        or not all(
            isinstance(offset, (list, tuple))
            and len(offset) == 2
            and all(type(value) is int for value in offset)
            for offset in offsets
        )
    ):
        raise ActivationError(
            f"Example ID {example_id}: tokenizer returned invalid character offsets."
        )

    input_start = rendered.index(content)
    input_end = input_start + len(task_input)
    input_offsets = []
    for start, end in offsets:
        overlap_start = max(start, input_start)
        overlap_end = min(end, input_end)
        input_offsets.append(
            (overlap_start - input_start, overlap_end - input_start)
            if overlap_start < overlap_end
            else None
        )

    mapped = {}
    for name in sorted(spans, key=lambda value: int(value.removeprefix("span_"))):
        span = spans[name]
        start, end = span["start_char"], span["end_char"]
        positions = tuple(
            index
            for index, offset in enumerate(input_offsets)
            if offset is not None and offset[0] < end and start < offset[1]
        )
        if not positions:
            raise ActivationError(
                f"Example ID {example_id}, semantic span '{name}': character "
                "range maps to no prompt tokens."
            )
        mapped[name] = positions
    return mapped


def _runtime():
    try:
        import torch
    except ImportError as exc:
        raise ActivationError(
            "Activation dependencies are unavailable; install requirements.txt."
        ) from exc
    return torch


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None:
        raise ActivationError("The loaded model does not expose an input device.")
    return device


def _validate_replay(output: Any, plan: ReplayPlan, torch: Any) -> float:
    logits = getattr(output, "logits", None)
    if logits is None or getattr(logits, "ndim", None) != 3:
        raise ActivationError("The model did not return valid causal-language-model logits.")
    answer_positions = plan.positions["answer_tokens"]
    prediction_positions = [position - 1 for position in answer_positions]
    if any(position < 0 for position in prediction_positions):
        raise ActivationError("An answer token has no preceding prediction position.")
    selected = logits[0, prediction_positions].float()
    targets = torch.tensor(
        plan.answer_token_ids, dtype=torch.long, device=selected.device
    ).unsqueeze(1)
    replayed = selected.log_softmax(dim=-1).gather(1, targets).squeeze(1).cpu().tolist()

    differences = []
    for actual, expected in zip(replayed, plan.answer_token_logprobs):
        difference = abs(float(actual) - expected)
        tolerance = REPLAY_ATOL + REPLAY_RTOL * abs(expected)
        if difference > tolerance:
            raise ActivationError(
                "Replay answer-token probability mismatch: "
                f"actual log probability {float(actual):.6g}, "
                f"saved {expected:.6g}, difference {difference:.6g}."
            )
        differences.append(difference)
    return max(differences, default=0.0)


def capture_hidden_states(loaded: LoadedModel, plan: ReplayPlan) -> ActivationResult:
    """Run one frozen replay and return selected hidden states on CPU."""
    torch = _runtime()
    model = loaded.model
    device = _model_device(model)
    input_ids = torch.tensor([plan.token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    except Exception as exc:
        raise ActivationError(f"Activation replay failed: {exc}") from exc

    hidden_states = getattr(output, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise ActivationError("The model did not return hidden states.")
    sequence_length = len(plan.token_ids)
    hidden_size = None
    for index, state in enumerate(hidden_states):
        shape = getattr(state, "shape", ())
        if len(shape) != 3 or shape[0] != 1 or shape[1] != sequence_length:
            raise ActivationError(
                f"hidden_states[{index}] has an unsupported shape {tuple(shape)}."
            )
        if hidden_size is None:
            hidden_size = shape[2]
        elif shape[2] != hidden_size:
            raise ActivationError("Hidden-state dimensions differ across layers.")

    max_difference = _validate_replay(output, plan, torch)
    tensors = {}
    for name, positions in plan.positions.items():
        tensors[name] = torch.stack(
            [
                state[0]
                .index_select(
                    0,
                    torch.tensor(positions, dtype=torch.long, device=state.device),
                )
                .to(dtype=torch.float16, device="cpu")
                for state in hidden_states
            ],
            dim=0,
        ).contiguous()
    labels = ("embedding",) + tuple(
        f"hidden_state_{index}" for index in range(1, len(hidden_states))
    )
    return ActivationResult(tensors, labels, max_difference)
