"""Generate task answers in direct or bounded-reasoning mode."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Iterator, Sequence

from src.config import TaskConfig
from src.model import LoadedModel


GENERATION_PROTOCOL_VERSION = 3
GENERATION_RECORD_SCHEMA_VERSION = 1

REASONING_INSTRUCTION = (
    "Reason about the task first. A separate final-answer instruction will follow."
)
ANSWER_INSTRUCTION = (
    "Complete the FINAL: line with only the answer, without reasoning or explanation."
)
ALLOW_ABSTENTION_INSTRUCTION = (
    "If you cannot determine the answer, return UNKNOWN."
)
DISALLOW_ABSTENTION_INSTRUCTION = (
    "Provide your best answer. Do not return UNKNOWN."
)
FINAL_MARKER = "FINAL:"
THINKING_BOUNDARY = "</think>"


class GenerationError(RuntimeError):
    """Generation cannot continue safely."""


@dataclass(frozen=True)
class GenerationUnit:
    """One persistable group of generated records."""

    records: tuple[dict, ...]


def answer_instruction(allow_abstention: bool) -> str:
    suffix = (
        ALLOW_ABSTENTION_INSTRUCTION
        if allow_abstention
        else DISALLOW_ABSTENTION_INSTRUCTION
    )
    return f"{ANSWER_INSTRUCTION}\n{suffix}"


def _runtime():
    try:
        import torch
        from transformers import StoppingCriteriaList, set_seed
    except ImportError as exc:
        raise GenerationError(
            "Generation dependencies are unavailable; install requirements.txt."
        ) from exc
    return torch, StoppingCriteriaList, set_seed


def _token_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise GenerationError("Expected one tokenized prompt.")
        value = value[0]
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise GenerationError("Tokenizer returned an unsupported token structure.")
    return value


def _encode(tokenizer: Any, text: str) -> list[int]:
    return _token_list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def initial_prompt_content(task_input: str, instruction: str) -> str:
    return f"{task_input}\n\n{instruction}"


def render_initial_prompt(
    tokenizer: Any,
    task_input: str,
    instruction: str,
    *,
    reasoning: bool,
    tokenize: bool = True,
    system_prompt: str | None = None,
) -> list[int] | str:
    """Render the exact initial chat prompt used for generation or span mapping."""
    content = initial_prompt_content(task_input, instruction)
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True,
            enable_thinking=reasoning,
        )
    except Exception as exc:
        raise ValueError(f"Could not render prompt: {exc}") from exc
    if tokenize:
        return _token_list(rendered)
    if not isinstance(rendered, str):
        raise ValueError("Chat template did not return rendered prompt text.")
    return rendered


def _context_limit(model: Any, tokenizer: Any) -> int | None:
    candidates = (
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    )
    finite = [
        int(value)
        for value in candidates
        if type(value) is int and 0 < value < 1_000_000_000
    ]
    return min(finite) if finite else None


def _stable_seed(base_seed: int, example_id: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def _failure(example: dict, reason: str, message: str) -> dict:
    return {
        "schema_version": GENERATION_RECORD_SCHEMA_VERSION,
        "protocol_version": GENERATION_PROTOCOL_VERSION,
        "id": example["id"],
        "split": example["split"],
        "status": "failed",
        "failure": {"reason": reason, "message": message},
    }


def _base_record(example: dict, prompt_ids: list[int], control: dict) -> dict:
    return {
        "schema_version": GENERATION_RECORD_SCHEMA_VERSION,
        "protocol_version": GENERATION_PROTOCOL_VERSION,
        "id": example["id"],
        "split": example["split"],
        "status": "success",
        "formatted_prompt_token_ids": prompt_ids,
        "formatted_prompt_tokens": len(prompt_ids),
        "final_control": control,
    }


def _termination(
    generated: list[int], eos_ids: set[int], limit: int
) -> tuple[list[int], str]:
    for index, token_id in enumerate(generated):
        if token_id in eos_ids:
            return generated[:index], "eos"
    if len(generated) >= limit:
        return generated[:limit], "token_limit"
    return generated, "model_stop"


def _move_inputs(inputs: dict, model: Any) -> dict:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {name: value.to(device) for name, value in inputs.items()}


def _pad_contexts(tokenizer: Any, contexts: Sequence[Sequence[int]], model: Any) -> dict:
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer.pad(
            {"input_ids": [list(ids) for ids in contexts]},
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = previous_side
    return _move_inputs(inputs, model)


def _answer_outputs(
    loaded: LoadedModel,
    contexts: Sequence[Sequence[int]],
    max_new_tokens: int,
) -> list[dict]:
    torch, _, _ = _runtime()
    model, tokenizer = loaded.model, loaded.tokenizer
    inputs = _pad_contexts(tokenizer, contexts, model)
    prompt_length = int(inputs["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    except Exception as exc:
        raise GenerationError(f"Answer generation failed: {exc}") from exc

    scores = tuple(getattr(output, "scores", ()))
    try:
        transition_scores = model.compute_transition_scores(
            output.sequences,
            scores,
            getattr(output, "beam_indices", None),
            normalize_logits=True,
        )
    except Exception as exc:
        raise GenerationError(f"Could not compute answer token probabilities: {exc}") from exc

    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    results = []
    for index in range(len(contexts)):
        generated = [int(item) for item in output.sequences[index, prompt_length:].tolist()]
        token_ids, stop_reason = _termination(generated, eos_ids, max_new_tokens)
        logprobs = [
            float(value)
            for value in transition_scores[index, : len(token_ids)].tolist()
        ]
        results.append(
            {
                "text": _decode(tokenizer, token_ids),
                "token_ids": token_ids,
                "token_logprobs": logprobs,
                "tokens": len(token_ids),
                "stop_reason": stop_reason,
            }
        )
    return results


class _StopOnSuffix:
    def __init__(self, suffix: Sequence[int], prompt_length: int, torch: Any):
        self.suffix = tuple(suffix)
        self.prompt_length = prompt_length
        self.torch = torch

    def __call__(self, input_ids, scores, **kwargs):
        matches = []
        for row in input_ids:
            generated = row[self.prompt_length :].tolist()
            matches.append(
                len(generated) >= len(self.suffix)
                and tuple(generated[-len(self.suffix) :]) == self.suffix
            )
        return self.torch.tensor(matches, device=input_ids.device, dtype=self.torch.bool)


def _find_subsequence(tokens: Sequence[int], pattern: Sequence[int]) -> int | None:
    width = len(pattern)
    for index in range(len(tokens) - width + 1):
        if list(tokens[index : index + width]) == list(pattern):
            return index
    return None


def _reasoning_output(
    loaded: LoadedModel,
    initial_ids: list[int],
    boundary_ids: list[int],
    max_new_tokens: int,
) -> tuple[list[int], str, bool]:
    torch, stopping_list, _ = _runtime()
    model, tokenizer = loaded.model, loaded.tokenizer
    inputs = _pad_contexts(tokenizer, [initial_ids], model)
    prompt_length = int(inputs["input_ids"].shape[1])
    stopping = stopping_list([_StopOnSuffix(boundary_ids, prompt_length, torch)])
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping,
                return_dict_in_generate=True,
                output_scores=False,
            )
    except Exception as exc:
        raise GenerationError(f"Reasoning generation failed: {exc}") from exc

    generated = [int(item) for item in output.sequences[0, prompt_length:].tolist()]
    boundary_index = _find_subsequence(generated, boundary_ids)
    if boundary_index is not None:
        return generated[:boundary_index], "thinking_boundary", False

    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    reasoning_ids, stop_reason = _termination(generated, eos_ids, max_new_tokens)
    return reasoning_ids, stop_reason, True


def _direct_unit(
    loaded: LoadedModel,
    indexed_examples: Sequence[tuple[int, dict]],
    config: TaskConfig,
) -> GenerationUnit:
    _, _, set_seed = _runtime()
    tokenizer, model = loaded.tokenizer, loaded.model
    marker_ids = _encode(tokenizer, FINAL_MARKER)
    instruction = answer_instruction(config.allow_abstention)
    limit = _context_limit(model, tokenizer)
    prepared: list[tuple[int, dict, list[int], list[int], dict]] = []
    records: list[dict] = []

    for position, example in indexed_examples:
        try:
            chat_ids = render_initial_prompt(
                tokenizer,
                example["input"],
                instruction,
                reasoning=False,
                system_prompt=config.system_prompt,
            )
        except ValueError as exc:
            records.append(_failure(example, "prompt_rendering", str(exc)))
            continue
        answer_context = [*chat_ids, *marker_ids]
        if limit is not None and len(answer_context) + config.answer_max_new_tokens > limit:
            records.append(
                _failure(
                    example,
                    "context_length",
                    "Prompt and answer budget require "
                    f"{len(answer_context) + config.answer_max_new_tokens} "
                    f"tokens, exceeding the model limit of {limit}.",
                )
            )
            continue
        control = {
            "text": FINAL_MARKER,
            "token_ids": marker_ids,
            "answer_instruction": instruction,
            "answer_instruction_token_ids": _encode(tokenizer, instruction),
            "final_marker_token_ids": marker_ids,
            "final_marker_span": [0, len(marker_ids)],
        }
        prepared.append((position, example, chat_ids, answer_context, control))

    if prepared:
        set_seed(_stable_seed(config.generation_seed, indexed_examples[0][1]["id"]))
        answers = _answer_outputs(
            loaded,
            [row[3] for row in prepared],
            config.answer_max_new_tokens,
        )
        for (_, example, prompt_ids, _, control), answer in zip(prepared, answers):
            record = _base_record(example, prompt_ids, control)
            record["reasoning"] = None
            record["answer"] = answer
            records.append(record)

    order = {example["id"]: position for position, example in indexed_examples}
    records.sort(key=lambda record: order[record["id"]])
    return GenerationUnit(records=tuple(records))


def _reasoning_unit(
    loaded: LoadedModel,
    position: int,
    example: dict,
    config: TaskConfig,
) -> GenerationUnit:
    _, _, set_seed = _runtime()
    tokenizer, model = loaded.tokenizer, loaded.model
    instruction = answer_instruction(config.allow_abstention)
    boundary_ids = _encode(tokenizer, THINKING_BOUNDARY)
    final_instruction_ids = _encode(tokenizer, f"\n\n{instruction}\n")
    marker_ids = _encode(tokenizer, FINAL_MARKER)

    try:
        initial_ids = render_initial_prompt(
            tokenizer,
            example["input"],
            REASONING_INSTRUCTION,
            reasoning=True,
            system_prompt=config.system_prompt,
        )
    except ValueError as exc:
        return GenerationUnit(
            records=(_failure(example, "prompt_rendering", str(exc)),),
        )

    required = (
        len(initial_ids)
        + config.reasoning_max_new_tokens
        + len(boundary_ids)
        + len(final_instruction_ids)
        + len(marker_ids)
        + config.answer_max_new_tokens
    )
    limit = _context_limit(model, tokenizer)
    if limit is not None and required > limit:
        return GenerationUnit(
            records=(
                _failure(
                    example,
                    "context_length",
                    f"Prompt and generation budgets require {required} tokens, "
                    f"exceeding the model limit of {limit}.",
                ),
            ),
        )

    set_seed(_stable_seed(config.generation_seed, example["id"]))
    reasoning_ids, stop_reason, forced = _reasoning_output(
        loaded,
        initial_ids,
        boundary_ids,
        config.reasoning_max_new_tokens,
    )
    control_ids = [*boundary_ids, *final_instruction_ids, *marker_ids]
    marker_start = len(boundary_ids) + len(final_instruction_ids)
    final_context = [*initial_ids, *reasoning_ids, *control_ids]
    answer = _answer_outputs(loaded, [final_context], config.answer_max_new_tokens)[0]
    control = {
        "text": f"{THINKING_BOUNDARY}\n\n{instruction}\n{FINAL_MARKER}",
        "token_ids": control_ids,
        "answer_instruction": instruction,
        "answer_instruction_token_ids": final_instruction_ids,
        "final_marker_token_ids": marker_ids,
        "final_marker_span": [marker_start, marker_start + len(marker_ids)],
    }
    record = _base_record(example, initial_ids, control)
    record["reasoning"] = {
        "text": _decode(tokenizer, reasoning_ids),
        "token_ids": reasoning_ids,
        "tokens": len(reasoning_ids),
        "stop_reason": stop_reason,
        "thinking_boundary_forced": forced,
    }
    record["answer"] = answer
    return GenerationUnit(records=(record,))


def generation_units(
    loaded: LoadedModel,
    examples: Sequence[dict],
    config: TaskConfig,
    completed_ids: Iterable[int] = (),
) -> Iterator[GenerationUnit]:
    """Yield persistable generation units in dataset order."""
    completed = set(completed_ids)
    indexed = list(enumerate(examples))
    if config.reasoning_mode == "direct":
        size = config.direct_batch_size
        missing = [item for item in indexed if item[1]["id"] not in completed]
        generation_config = getattr(loaded.model, "generation_config", None)
        if bool(getattr(generation_config, "do_sample", False)):
            size = 1
        for start in range(0, len(missing), size):
            batch = missing[start : start + size]
            yield _direct_unit(loaded, batch, config)
        return

    for position, example in indexed:
        if example["id"] not in completed:
            yield _reasoning_unit(loaded, position, example, config)
