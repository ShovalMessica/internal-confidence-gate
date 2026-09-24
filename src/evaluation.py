"""Evaluate saved model answers against dataset targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence


EVALUATION_PROTOCOL_VERSION = 1
EVALUATION_RECORD_SCHEMA_VERSION = 1

_SPLITS = ("train", "validation", "test")
_OUTCOMES = ("correct", "incorrect", "abstained", "invalid")
_MINIMUMS = {
    "train": {"correct": 100, "incorrect": 100},
    "validation": {"correct": 50, "incorrect": 50},
    "test": {"correct": 50, "incorrect": 50},
}


class EvaluationError(ValueError):
    """Saved generations cannot be evaluated safely."""


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[dict, ...]
    summary: dict
    shortages: tuple[dict, ...]


def normalize_answer(value: str) -> str:
    """Apply the task-independent V1 answer normalization."""
    return " ".join(value.casefold().split())


def _invalid(example: dict, reason: str) -> dict:
    return {
        "schema_version": EVALUATION_RECORD_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "id": example["id"],
        "split": example["split"],
        "outcome": "invalid",
        "normalized_answer": None,
        "is_correct": None,
        "invalid_reason": reason,
    }


def _evaluate_one(
    example: dict, generation: dict, allow_abstention: bool
) -> dict:
    if generation["status"] == "failed":
        return _invalid(example, "generation_failed")

    answer = generation.get("answer")
    if not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
        raise EvaluationError(
            f"Generation for ID {example['id']} has no valid answer text."
        )
    raw_answer = answer["text"]
    normalized = normalize_answer(raw_answer)
    if not normalized:
        return _invalid(example, "empty_answer")
    if "final:" in raw_answer.casefold():
        return _invalid(example, "repeated_final_marker")
    if sum(bool(line.strip()) for line in raw_answer.splitlines()) > 1:
        return _invalid(example, "multiple_nonempty_lines")

    if allow_abstention and normalized == "unknown":
        outcome: Literal["correct", "incorrect", "abstained"] = "abstained"
        is_correct = None
    else:
        is_correct = normalized == normalize_answer(example["target_answer"])
        outcome = "correct" if is_correct else "incorrect"
    return {
        "schema_version": EVALUATION_RECORD_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "id": example["id"],
        "split": example["split"],
        "outcome": outcome,
        "normalized_answer": normalized,
        "is_correct": is_correct,
    }


def _metrics(records: Sequence[dict]) -> dict:
    counts = Counter(record["outcome"] for record in records)
    total = len(records)
    usable = counts["correct"] + counts["incorrect"]
    return {
        "total": total,
        **{outcome: counts[outcome] for outcome in _OUTCOMES},
        "answer_token_limit": sum(
            bool(record.get("_answer_token_limit")) for record in records
        ),
        "usable_accuracy": counts["correct"] / usable if usable else None,
        "abstention_rate": counts["abstained"] / total if total else None,
        "invalid_rate": counts["invalid"] / total if total else None,
    }


def _shortages(summary: dict) -> tuple[dict, ...]:
    missing = []
    for split in _SPLITS:
        for outcome, required in _MINIMUMS[split].items():
            actual = summary["by_split"][split][outcome]
            if actual < required:
                missing.append(
                    {
                        "split": split,
                        "outcome": outcome,
                        "required": required,
                        "actual": actual,
                        "missing": required - actual,
                    }
                )
    return tuple(missing)


def evaluate_answers(
    examples: Sequence[dict],
    generations: dict[int, dict],
    *,
    allow_abstention: bool,
) -> EvaluationResult:
    """Evaluate complete saved generations in dataset order."""
    expected_ids = {example["id"] for example in examples}
    if set(generations) != expected_ids:
        raise EvaluationError("Generation records do not match the dataset examples.")

    records = []
    metric_records = []
    for example in examples:
        generation = generations[example["id"]]
        record = _evaluate_one(example, generation, allow_abstention)
        records.append(record)
        metric_records.append(
            {
                **record,
                "_answer_token_limit": (
                    generation["status"] == "success"
                    and generation["answer"].get("stop_reason") == "token_limit"
                ),
            }
        )

    summary = {
        "overall": _metrics(metric_records),
        "by_split": {
            split: _metrics(
                [record for record in metric_records if record["split"] == split]
            )
            for split in _SPLITS
        },
    }
    shortages = _shortages(summary)
    summary["probe_ready"] = not shortages
    summary["shortages"] = list(shortages)
    return EvaluationResult(tuple(records), summary, shortages)
