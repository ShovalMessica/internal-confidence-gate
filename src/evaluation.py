"""Evaluate saved model answers against dataset targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable, Literal, Sequence


EVALUATION_PROTOCOL_VERSION = 2
EVALUATION_RECORD_SCHEMA_VERSION = 1
_BUILTIN_MATCHER_NAME = "normalized_exact_v1"
_SPLITS = ("train", "validation", "test")
_OUTCOMES = ("correct", "incorrect", "abstained", "invalid")
_MINIMUMS = {
    "train": {"correct": 100, "incorrect": 100},
    "validation": {"correct": 50, "incorrect": 50},
    "test": {"correct": 50, "incorrect": 50},
}


class EvaluationError(ValueError):
    """Saved generations or answer-matching code cannot be evaluated safely."""


@dataclass(frozen=True)
class AnswerMatcher:
    kind: Literal["builtin", "custom"]
    sha256: str
    source_path: Path | None
    function: Callable[[str, str], bool] = field(repr=False, compare=False)

    def match(self, prediction: str, target_answer: str) -> bool:
        try:
            result = self.function(prediction, target_answer)
        except Exception as exc:
            raise EvaluationError(
                f"answer_match raised {type(exc).__name__}: {exc}"
            ) from exc
        if type(result) is not bool:
            raise EvaluationError("answer_match must return true or false.")
        return result

    def metadata(self) -> dict:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "source_path": str(self.source_path) if self.source_path else None,
        }


@dataclass(frozen=True)
class EvaluationIdentity:
    evaluation_id: str
    fingerprint: str
    generation_sha256: str
    matcher_sha256: str


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[dict, ...]
    summary: dict
    shortages: tuple[dict, ...]


def normalize_answer(value: str) -> str:
    """Apply the task-independent V1 answer normalization."""
    return " ".join(value.casefold().split())


def _default_answer_match(prediction: str, target_answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(target_answer)


def load_answer_matcher(path: Path | None) -> AnswerMatcher:
    """Load the built-in matcher or a user-supplied answer_match function."""
    if path is None:
        digest = hashlib.sha256(_BUILTIN_MATCHER_NAME.encode("utf-8")).hexdigest()
        return AnswerMatcher("builtin", digest, None, _default_answer_match)

    try:
        source = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Cannot read answer matcher: {path}") from exc
    digest = hashlib.sha256(source).hexdigest()
    namespace = {"__file__": str(path), "__name__": f"_answer_matcher_{digest}"}
    try:
        code = compile(source, str(path), "exec")
        exec(code, namespace)
    except Exception as exc:
        raise EvaluationError(
            f"Cannot load answer matcher {path} ({type(exc).__name__}: {exc})."
        ) from exc

    function = namespace.get("answer_match")
    if not callable(function):
        raise EvaluationError(
            f"Answer matcher must define answer_match(prediction, target_answer): {path}"
        )
    try:
        inspect.signature(function).bind("prediction", "target_answer")
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            f"answer_match must accept prediction and target_answer: {path}"
        ) from exc
    return AnswerMatcher("custom", digest, path, function)


def build_evaluation_identity(
    generation_sha256: str, matcher: AnswerMatcher
) -> EvaluationIdentity:
    payload = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "generation_sha256": generation_sha256,
        "matcher_sha256": matcher.sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = hashlib.sha256(encoded).hexdigest()
    return EvaluationIdentity(
        fingerprint[:12], fingerprint, generation_sha256, matcher.sha256
    )


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
    example: dict,
    generation: dict,
    allow_abstention: bool,
    matcher: AnswerMatcher,
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
        try:
            is_correct = matcher.match(raw_answer, example["target_answer"])
        except EvaluationError as exc:
            raise EvaluationError(
                f"Answer matcher failed for example ID {example['id']}: {exc}"
            ) from exc
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
    token_limit = sum(bool(record.get("_answer_token_limit")) for record in records)

    def rate(count: int) -> float | None:
        return count / total if total else None

    return {
        "total": total,
        **{outcome: counts[outcome] for outcome in _OUTCOMES},
        "answer_token_limit": token_limit,
        "correct_prediction_rate": rate(counts["correct"]),
        "wrong_prediction_rate": rate(counts["incorrect"]),
        "missed_prediction_rate": rate(counts["abstained"]),
        "invalid_output_rate": rate(counts["invalid"]),
        "token_limit_rate": rate(token_limit),
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


def probe_shortages(
    records: Sequence[dict], excluded_answers: Sequence[str] = ()
) -> tuple[dict, ...]:
    """Report probe-class shortages after task-specific answer exclusions."""
    excluded = set(excluded_answers)
    counts = {
        split: Counter(
            record["outcome"]
            for record in records
            if record.get("split") == split
            and record.get("outcome") in ("correct", "incorrect")
            and record.get("normalized_answer") not in excluded
        )
        for split in _SPLITS
    }
    return _shortages({"by_split": counts})


def evaluate_answers(
    examples: Sequence[dict],
    generations: dict[int, dict],
    *,
    allow_abstention: bool,
    matcher: AnswerMatcher | None = None,
) -> EvaluationResult:
    """Evaluate complete saved generations in dataset order."""
    expected_ids = {example["id"] for example in examples}
    if set(generations) != expected_ids:
        raise EvaluationError("Generation records do not match the dataset examples.")
    matcher = matcher or load_answer_matcher(None)

    records = []
    metric_records = []
    for example in examples:
        generation = generations[example["id"]]
        record = _evaluate_one(example, generation, allow_abstention, matcher)
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
