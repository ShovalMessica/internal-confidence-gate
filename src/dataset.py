"""Read and validate task examples without splitting data or running a model."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass
class DatasetResult:
    examples: list[dict]
    excluded: list[dict]


class DatasetError(ValueError):
    """Dataset-level failures, with any exclusions recorded before failure."""

    def __init__(self, errors: list[str], excluded: list[dict]):
        self.errors = tuple(errors)
        self.excluded = excluded
        super().__init__("Dataset errors:\n" + "\n".join(f"- {error}" for error in errors))


_SPAN_KEY = re.compile(r"span_[1-9][0-9]*")


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


def _validate_record(record: dict, errors: list[str]) -> dict:
    example = {name: record.get(name) for name in ("id", "input", "target_answer")}
    if type(example["id"]) is not int:
        errors.append("id must be an integer, not a Boolean.")
    for name in ("input", "target_answer"):
        value = example[name]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a nonempty string.")
    target = example["target_answer"]
    if isinstance(target, str):
        if " ".join(target.split()).casefold() == "unknown":
            errors.append("target_answer cannot be UNKNOWN; it is reserved for abstention.")
        if "final:" in target.casefold():
            errors.append("target_answer must not contain the FINAL: prefix.")
    if "split" in record:
        example["split"] = record["split"]
        if record["split"] not in ("train", "validation", "test"):
            errors.append("split must be 'train', 'validation', or 'test'.")
    if "semantic_spans" in record:
        example["semantic_spans"] = _validate_spans(
            record["semantic_spans"], example["input"], errors
        )
    return example


def load_dataset(path: str | Path) -> DatasetResult:
    """Return valid examples and exclusions; raise DatasetError on dataset failure.

    Offsets use Python string indices into unchanged, decoded input. Consistency
    checks apply only to surviving records. No files are written or messages printed.
    """
    examples, excluded = [], []
    seen_ids = set()
    example_lines = []
    try:
        # Decode each line separately so a later encoding failure preserves earlier exclusions.
        with Path(path).open("rb") as source:
            for line_number, raw in enumerate(source, start=1):
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
                                reasons.append(f"Duplicate id {record_id}; the first occurrence reserves it.")
                            seen_ids.add(record_id)
                        example = _validate_record(record, reasons)
                        if not reasons:
                            examples.append(example)
                            example_lines.append(line_number)
                if reasons:
                    excluded.append({"line": line_number, "id": record_id, "reasons": reasons})
    except (OSError, ValueError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError([f"Cannot read dataset ({type(exc).__name__})."], excluded) from exc

    errors = []
    if not examples:
        errors.append("No valid examples remain.")
    else:
        has_split = ["split" in example for example in examples]
        if any(has_split) and not all(has_split):
            errors.append("Supply split for every valid example or omit it from all examples.")
        expected = set(examples[0].get("semantic_spans", {}))
        for line_number, example in zip(example_lines, examples):
            actual = set(example.get("semantic_spans", {}))
            if actual != expected:
                errors.append(
                    f"Semantic-span keys differ at line {line_number} (id {example['id']}): "
                    f"expected {sorted(expected)} from line {example_lines[0]}, got {sorted(actual)}."
                )
    if errors:
        raise DatasetError(errors, excluded)
    return DatasetResult(examples, excluded)
