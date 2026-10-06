"""Read, validate, and split task examples without running a model."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re

from src.config import SplitRatios


@dataclass
class DatasetResult:
    examples: list[dict]
    excluded: list[dict]
    content_sha256: str | None = None


class DatasetError(ValueError):
    """Dataset-level failures, with any exclusions recorded before failure."""

    def __init__(self, errors: list[str], excluded: list[dict]):
        self.errors = tuple(errors)
        self.excluded = excluded
        super().__init__("Dataset errors:\n" + "\n".join(f"- {error}" for error in errors))


_SPAN_KEY = re.compile(r"span_[1-9][0-9]*")
_SPLIT_NAMES = ("train", "validation", "test")
_MIN_EXAMPLES = 700
_MIN_SPLIT_SIZES = {"train": 200, "validation": 100, "test": 100}


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key '{key}'.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant '{value}'.")


def _validate_spans(value: object, text: object, errors: list[str]) -> dict:
    if not isinstance(value, dict):
        errors.append("semantic_spans must be an object; omit it or use {} for no spans.")
        return {}
    spans = {}
    for name, span in value.items():
        field = f"semantic_spans.{name}"
        if not _SPAN_KEY.fullmatch(name):
            errors.append(f"{field}: use keys span_1, span_2, etc., without leading zeros.")
        if not isinstance(span, dict):
            errors.append(f"{field} must contain start_char and end_char.")
            continue
        start, end = span.get("start_char"), span.get("end_char")
        if type(start) is not int or type(end) is not int:
            errors.append(f"{field}: start_char and end_char must be integers.")
        elif not 0 <= start < end or (isinstance(text, str) and end > len(text)):
            errors.append(f"{field}: require 0 <= start_char < end_char <= input length.")
        else:
            spans[name] = {"start_char": start, "end_char": end}
    return spans


def _validate_record(record: dict, errors: list[str], allow_abstention: bool) -> dict:
    example = {name: record.get(name) for name in ("id", "input", "target_answer")}
    if type(example["id"]) is not int:
        errors.append("id must be an integer, not a Boolean.")
    for name in ("input", "target_answer"):
        value = example[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a nonempty string.")
    target = example["target_answer"]
    if isinstance(target, str):
        if allow_abstention and " ".join(target.split()).casefold() == "unknown":
            errors.append(
                "target_answer cannot be UNKNOWN while abstention is enabled; "
                "UNKNOWN is reserved for model abstention."
            )
        if "final:" in target.casefold():
            errors.append("target_answer must not contain the FINAL: prefix.")
    if "split" in record:
        example["split"] = record["split"]
        if record["split"] not in _SPLIT_NAMES:
            errors.append("split must be 'train', 'validation', or 'test'.")
    if "semantic_spans" in record:
        example["semantic_spans"] = _validate_spans(
            record["semantic_spans"], example["input"], errors
        )
    if "metadata" in record:
        if not isinstance(record["metadata"], dict):
            errors.append("metadata must be an object when supplied.")
        else:
            example["metadata"] = record["metadata"]
    return example


def _split_state(examples: list[dict]) -> str:
    supplied = ["split" in example for example in examples]
    if all(supplied):
        return "provided"
    if any(supplied):
        return "partial"
    return "automatic"


def load_dataset(path: str | Path, *, allow_abstention: bool = True) -> DatasetResult:
    """Return valid examples and exclusions; raise DatasetError on dataset failure.

    Offsets use Python string indices into unchanged, decoded input. Consistency
    checks apply only to surviving records. No files are written or messages printed.
    """
    examples, excluded = [], []
    seen_ids = set()
    example_lines = []
    content_hash = hashlib.sha256()
    try:
        # Decode each line separately so a later encoding failure preserves earlier exclusions.
        with Path(path).open("rb") as source:
            for line_number, raw in enumerate(source, start=1):
                content_hash.update(raw)
                try:
                    line = raw.decode("utf-8")
                except UnicodeError as exc:
                    raise DatasetError(
                        [f"Dataset is not valid UTF-8 at line {line_number}."], excluded
                    ) from exc
                reasons = []
                record_id = None
                try:
                    if not line.strip():
                        raise ValueError("Blank line; expected a JSON object.")
                    record = json.loads(
                        line, object_pairs_hook=_unique_object, parse_constant=_reject_constant
                    )
                except ValueError as exc:
                    reasons.append(f"Invalid record: {exc}")
                else:
                    if not isinstance(record, dict):
                        reasons.append("Each record must be a JSON object.")
                    else:
                        if type(record.get("id")) is int:
                            record_id = record["id"]
                            if record_id in seen_ids:
                                reasons.append(
                                    f"Duplicate id {record_id}; the first occurrence "
                                    "reserves it."
                                )
                            seen_ids.add(record_id)
                        example = _validate_record(record, reasons, allow_abstention)
                        if not reasons:
                            examples.append(example)
                            example_lines.append(line_number)
                if reasons:
                    excluded.append({"line": line_number, "id": record_id, "reasons": reasons})
    except DatasetError:
        raise
    except (OSError, ValueError) as exc:
        raise DatasetError([f"Cannot read dataset ({type(exc).__name__})."], excluded) from exc

    errors = []
    if not examples:
        errors.append("No valid examples remain.")
    else:
        if _split_state(examples) == "partial":
            errors.append("Supply split for every valid example or omit it from all examples.")
        expected = set(examples[0].get("semantic_spans", {}))
        for line_number, example in zip(example_lines, examples):
            actual = set(example.get("semantic_spans", {}))
            if actual != expected:
                errors.append(
                    f"Semantic-span keys differ at line {line_number} (id {example['id']}): "
                    f"expected {sorted(expected)} from line {example_lines[0]}, "
                    f"got {sorted(actual)}."
                )
    if errors:
        raise DatasetError(errors, excluded)
    return DatasetResult(examples, excluded, content_hash.hexdigest())


def _automatic_split_sizes(total: int, ratios: SplitRatios) -> dict[str, int]:
    exact = {name: total * getattr(ratios, name) for name in _SPLIT_NAMES}
    sizes = {name: math.floor(exact[name]) for name in _SPLIT_NAMES}
    order = sorted(
        _SPLIT_NAMES,
        key=lambda name: (-(exact[name] - sizes[name]), _SPLIT_NAMES.index(name)),
    )
    for name in order[: total - sum(sizes.values())]:
        sizes[name] += 1
    return sizes


def _check_split_sizes(total: int, sizes: dict[str, int], excluded: list[dict]) -> None:
    errors = []
    if total < _MIN_EXAMPLES:
        noun = "example" if total == 1 else "examples"
        errors.append(
            f"Dataset has {total} valid {noun}; at least {_MIN_EXAMPLES} are "
            "required."
        )
    for name in _SPLIT_NAMES:
        minimum = _MIN_SPLIT_SIZES[name]
        actual = sizes.get(name, 0)
        if actual < minimum:
            noun = "example" if actual == 1 else "examples"
            errors.append(
                f"{name} split has {actual} {noun}; at least {minimum} are required."
            )
    if errors:
        raise DatasetError(errors, excluded)


def assign_splits(
    dataset: DatasetResult,
    ratios: SplitRatios,
    seed: int,
    *,
    enforce_minimums: bool = True,
) -> DatasetResult:
    """Preserve supplied splits or assign deterministic splits to valid examples."""
    examples = dataset.examples
    split_state = _split_state(examples)
    if split_state == "partial":
        raise DatasetError(
            ["Supply split for every valid example or omit it from all examples."],
            dataset.excluded,
        )

    if split_state == "provided":
        sizes = dict(Counter(example["split"] for example in examples))
        if enforce_minimums:
            _check_split_sizes(len(examples), sizes, dataset.excluded)
        return DatasetResult(
            [dict(example) for example in examples],
            list(dataset.excluded),
            dataset.content_sha256,
        )

    sizes = _automatic_split_sizes(len(examples), ratios)
    if enforce_minimums:
        _check_split_sizes(len(examples), sizes, dataset.excluded)

    shuffled = list(range(len(examples)))
    random.Random(seed).shuffle(shuffled)
    assignments = {}
    start = 0
    for name in _SPLIT_NAMES:
        end = start + sizes[name]
        assignments.update((index, name) for index in shuffled[start:end])
        start = end

    split_examples = [dict(example, split=assignments[index])
                      for index, example in enumerate(examples)]
    return DatasetResult(split_examples, list(dataset.excluded), dataset.content_sha256)
