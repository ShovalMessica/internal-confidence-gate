"""Replay saved generations and capture hidden states at named token positions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from src.generation import (
    REASONING_INSTRUCTION,
    _pad_contexts,
    initial_prompt_content,
    render_initial_prompt,
)
from src.model import LoadedModel


ACTIVATION_PROTOCOL_VERSION = 5
ACTIVATION_SCHEMA_VERSION = 1
CAPTURE_BACKEND = "transformers_hidden_states"
POSITION_PROTOCOL = "semantic_spans_v1"
STORAGE_DTYPE = "float16"


class ActivationError(RuntimeError):
    """A saved sequence cannot be replayed or captured safely."""


@dataclass(frozen=True)
class ReplayPlan:
    token_ids: tuple[int, ...]
    positions: dict[str, tuple[int, ...]]
    answer_token_ids: tuple[int, ...]
    answer_token_logprobs: tuple[float, ...]
    replay_sha256: str
    batch_sha256: str = ""


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
    *,
    allow_empty_answer: bool = False,
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
    control_ids = _token_ids(control.get("token_ids"), "final_control.token_ids")
    marker_span = control.get("final_marker_span")
    if control_ids:
        if (
            not isinstance(marker_span, list)
            or len(marker_span) != 2
            or not all(type(item) is int for item in marker_span)
            or not 0 <= marker_span[0] < marker_span[1] <= len(control_ids)
        ):
            raise ActivationError("final_control.final_marker_span is invalid.")
        if marker_span[1] != len(control_ids):
            raise ActivationError("The final marker must end the final control tokens.")
    elif marker_span is not None:
        raise ActivationError(
            "final_control.final_marker_span must be null without final control tokens."
        )

    answer = generation.get("answer")
    if not isinstance(answer, dict):
        raise ActivationError("answer must be an object.")
    answer_ids = _token_ids(
        answer.get("token_ids"), "answer.token_ids", nonempty=not allow_empty_answer
    )
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
    token_ids = (*prompt, *reasoning_ids, *control_ids, *answer_ids)
    positions = {"prompt_end": (len(prompt) - 1,)}
    if marker_span is not None:
        positions["final_prompt_end"] = (control_start + marker_span[1] - 1,)
    positions["answer_tokens"] = tuple(
        range(answer_start, answer_start + len(answer_ids))
    )
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


def build_replay_batches(
    generations: Mapping[int, dict],
    selected: Mapping[int, ReplayPlan],
    reasoning_mode: str,
) -> tuple[dict[int, ReplayPlan], ...]:
    """Restore original batch membership, including ineligible companion rows."""
    batches = []
    covered = set()
    for example_id in selected:
        if example_id in covered:
            continue
        if reasoning_mode == "reasoning":
            ids = [example_id]
        else:
            batch = generations[example_id].get("generation_batch")
            ids = batch.get("ids") if isinstance(batch, dict) else None
            if (
                not isinstance(ids, list) or not ids
                or any(type(i) is not int for i in ids)
                or len(set(ids)) != len(ids) or example_id not in ids
            ):
                raise ActivationError(
                    f"Missing or invalid generation batch for example ID {example_id}."
                )
        plans = {}
        for member_id in ids:
            record = generations.get(member_id)
            if record is None:
                raise ActivationError(
                    f"Replay for example ID {example_id} requires original batch "
                    f"member ID {member_id}; use the original dataset or regenerate "
                    "this dataset with --force-recompute."
                )
            batch = record.get("generation_batch")
            if reasoning_mode == "direct" and (
                not isinstance(batch, dict) or batch.get("ids") != ids
            ):
                raise ActivationError(
                    f"Inconsistent generation batch for example ID {member_id}; "
                    "regenerate this dataset with --force-recompute."
                )
            if record.get("status") != "success":
                continue  # Failed prompt/context checks never entered the model batch.
            plans[member_id] = selected.get(member_id) or build_replay_plan(
                record, allow_empty_answer=True
            )
        # Companion content and answer lengths affect padding and replay numerics.
        payload = [(i, p.token_ids, p.positions["answer_tokens"]) for i, p in plans.items()]
        batch_hash = hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        batches.append({i: replace(p, batch_sha256=batch_hash) for i, p in plans.items()})
        covered.update(plans)
    return tuple(batches)


def map_semantic_spans(
    tokenizer: Any,
    example: Mapping[str, object],
    generation: Mapping[str, object],
    system_prompt: str | None = None,
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
    if instruction is not None and not isinstance(instruction, str):
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
            system_prompt=system_prompt,
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


def _replay_logprob_difference(
    output: Any, plan: ReplayPlan, torch: Any, row: int = 0, offset: int = 0
) -> float:
    logits = getattr(output, "logits", None)
    if logits is None or getattr(logits, "ndim", None) != 3:
        raise ActivationError("The model did not return valid causal-language-model logits.")
    answer_positions = plan.positions["answer_tokens"]
    prediction_positions = [position + offset - 1 for position in answer_positions]
    if any(position < 0 for position in prediction_positions):
        raise ActivationError("An answer token has no preceding prediction position.")
    selected = logits[row, prediction_positions].float()
    targets = torch.tensor(
        plan.answer_token_ids, dtype=torch.long, device=selected.device
    ).unsqueeze(1)
    replayed = selected.log_softmax(dim=-1).gather(1, targets).squeeze(1).cpu().tolist()

    differences = []
    for actual, expected in zip(replayed, plan.answer_token_logprobs):
        difference = abs(float(actual) - expected)
        if not math.isfinite(actual) or difference > 0.01 + 0.002 * abs(expected):
            raise ActivationError(
                "Replay probability mismatch: "
                f"saved log-probability {expected:.6g}, replayed {actual:.6g}. "
                "Capture stopped; activations from this batch were not saved."
            )
        differences.append(difference)
    return max(differences, default=0.0)


def capture_hidden_states(loaded: LoadedModel, plan: ReplayPlan) -> ActivationResult:
    """Capture a singleton generation batch (including reasoning-mode answers)."""
    return capture_hidden_state_batch(loaded, {0: plan})[0]


def capture_hidden_state_batch(
    loaded: LoadedModel, plans: Mapping[int, ReplayPlan]
) -> dict[int, ActivationResult]:
    """Replay a complete saved batch, preserving its original prompt padding."""
    if not plans:
        raise ActivationError("Cannot replay an empty generation batch.")
    torch = _runtime()
    model = loaded.model
    contexts = [
        list(plan.token_ids[:len(plan.token_ids) - len(plan.answer_token_ids)])
        for plan in plans.values()
    ]
    inputs = _pad_contexts(loaded.tokenizer, contexts, model)
    context_width = int(inputs["input_ids"].shape[1])
    answer_width = max(len(p.answer_token_ids) for p in plans.values())
    # Answers grow on the right during generation. Left-padding whole replay
    # sequences instead would move shorter answers' original prompt columns.
    answers = torch.full(
        (len(plans), answer_width), loaded.tokenizer.pad_token_id,
        dtype=torch.long, device=inputs["input_ids"].device,
    )
    answer_mask = torch.zeros_like(answers)
    for row, plan in enumerate(plans.values()):
        count = len(plan.answer_token_ids)
        if count:
            answers[row, :count] = torch.tensor(plan.answer_token_ids, device=answers.device)
            answer_mask[row, :count] = 1
    input_ids = torch.cat((inputs["input_ids"], answers), dim=1)
    attention_mask = torch.cat((inputs["attention_mask"], answer_mask), dim=1)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    except Exception as exc:
        raise ActivationError(f"Activation replay failed: {exc}") from exc

    hidden_states = getattr(output, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise ActivationError("The model did not return hidden states.")
    sequence_length = input_ids.shape[1]
    hidden_size = None
    for index, state in enumerate(hidden_states):
        shape = getattr(state, "shape", ())
        if len(shape) != 3 or shape[0] != len(plans) or shape[1] != sequence_length:
            raise ActivationError(
                f"hidden_states[{index}] has an unsupported shape {tuple(shape)}."
            )
        if hidden_size is None:
            hidden_size = shape[2]
        elif shape[2] != hidden_size:
            raise ActivationError("Hidden-state dimensions differ across layers.")

    results = {}
    for row, (example_id, plan) in enumerate(plans.items()):
        offset = context_width - len(contexts[row])
        try:
            difference = _replay_logprob_difference(output, plan, torch, row, offset)
            results[example_id] = _select_hidden_states(
                hidden_states, plan, row, offset, difference, torch
            )
        except ActivationError as exc:
            raise ActivationError(f"Example ID {example_id}: {exc}") from exc
    return results


def _select_hidden_states(
    hidden_states: Sequence[Any], plan: ReplayPlan, row: int, offset: int,
    max_difference: float, torch: Any,
) -> ActivationResult:
    tensors = {}
    for name, positions in plan.positions.items():
        selected_states = []
        for index, state in enumerate(hidden_states):
            selected = state[row].index_select(
                0,
                torch.tensor(
                    [p + offset for p in positions], dtype=torch.long, device=state.device
                ),
            )
            if not bool(torch.isfinite(selected).all()):
                raise ActivationError(
                    f"Activation '{name}' contains nonfinite values at state {index}."
                )
            stored = selected.to(dtype=torch.float16, device="cpu")
            if not bool(torch.isfinite(stored).all()):
                raise ActivationError(
                    f"Activation '{name}' overflows float16 at state {index}."
                )
            selected_states.append(stored)
        tensors[name] = torch.stack(selected_states, dim=0).contiguous()
    labels = ("embedding",) + tuple(
        f"hidden_state_{index}" for index in range(1, len(hidden_states))
    )
    return ActivationResult(tensors, labels, max_difference)
