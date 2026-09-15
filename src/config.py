"""Load and validate task configuration without running pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Literal

import yaml


class ConfigurationError(ValueError):
    """Configuration problems that the caller can display or log together."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


@dataclass(frozen=True)
class TaskConfig:
    model_name_or_path: str
    dataset_path: Path
    reasoning_mode: Literal["direct", "reasoning"]
    output_dir: Path
    reasoning_max_new_tokens: int = 1024
    answer_max_new_tokens: int = 64
    allow_abstention: bool = True
    split_ratios: SplitRatios = SplitRatios()
    split_seed: int = 42


_FIELDS = frozenset(TaskConfig.__dataclass_fields__)
_SPLIT_NAMES = ("train", "validation", "test")
_MODEL_ID = re.compile(r"[\w][\w.-]*(?:/[\w][\w.-]*)?", re.ASCII)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Do not silently discard duplicated YAML settings."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigurationError(["YAML field names must be strings."])
            if key in seen:
                line = key_node.start_mark.line + 1
                raise ConfigurationError([f"Duplicate YAML field '{key}' at line {line}."])
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _absolute_path(value: object, field: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a nonempty absolute path string.")
        return None
    if "\0" in value:
        errors.append(f"{field} contains a null character.")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{field} must be an absolute path.")
        return None
    return path


def _integer(value: object, field: str, minimum: int, errors: list[str]) -> None:
    # bool is a subclass of int in Python, but is not a valid token limit or seed.
    if type(value) is not int or value < minimum:
        errors.append(f"{field} must be an integer >= {minimum}.")


def _split_ratios(value: object, errors: list[str]) -> SplitRatios | None:
    if not isinstance(value, dict):
        errors.append("split_ratios must contain train, validation, and test fractions.")
        return None
    if set(value) != set(_SPLIT_NAMES):
        errors.append("split_ratios must contain exactly train, validation, and test.")
    fractions = {}
    for name in _SPLIT_NAMES:
        fraction = value.get(name)
        try:
            valid = type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction < 1
        except OverflowError:
            valid = False
        if not valid:
            errors.append(f"split_ratios.{name} must be a finite number between 0 and 1, exclusive.")
        else:
            fractions[name] = float(fraction)
    if len(fractions) != 3:
        return None
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        errors.append("split_ratios must sum to 1.")
    return SplitRatios(**fractions)


def load_config(path: str | Path) -> TaskConfig:
    """Read YAML, fill defaults, and return validated immutable settings.

    ``path`` locates the YAML file; filesystem values inside it must be absolute.
    Omitted optional fields get defaults, while explicit null values are errors.
    No data records or model weights are read, no network calls are made, and no
    directories are created. Displaying/logging ConfigurationError is the caller's job.
    """
    try:
        source = Path(path).resolve()
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ConfigurationError:
        raise
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigurationError([f"Invalid YAML{location}; check indentation and syntax."]) from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigurationError([f"Cannot read the YAML configuration ({type(exc).__name__})."]) from exc

    if not isinstance(raw, dict):
        raise ConfigurationError(["The YAML configuration must be a mapping of field names to values."])

    errors: list[str] = []
    for name in sorted(set(raw) - _FIELDS):
        errors.append(f"Unknown field '{name}'; remove it or correct its spelling.")

    model = raw.get("model_name_or_path")
    if not isinstance(model, str) or not model.strip():
        errors.append("model_name_or_path is required: supply a Hugging Face ID or absolute checkpoint path.")
    elif "\0" in model:
        errors.append("model_name_or_path contains a null character.")
    elif Path(model).is_absolute():
        if not Path(model).is_dir():
            errors.append("model_name_or_path must point to an existing checkpoint directory.")
    elif not _MODEL_ID.fullmatch(model) or ".." in model or "--" in model or model.endswith((".", "-")):
        errors.append("model_name_or_path must be a model ID (name or owner/name) or an absolute checkpoint path.")

    dataset = _absolute_path(raw.get("dataset_path"), "dataset_path", errors)
    if dataset is not None:
        if dataset.suffix.lower() != ".jsonl":
            errors.append("dataset_path must name a .jsonl file.")
        if not dataset.is_file():
            errors.append("dataset_path must point to an existing file.")

    mode = raw.get("reasoning_mode")
    if mode not in ("direct", "reasoning"):
        errors.append("reasoning_mode is required and must be 'direct' or 'reasoning'.")

    if "output_dir" in raw:
        output = _absolute_path(raw["output_dir"], "output_dir", errors)
    else:
        output = source.parent / "outputs"
    if output is not None and output.exists() and not output.is_dir():
        errors.append("output_dir points to a file; supply a directory path instead.")

    reasoning_limit = raw.get("reasoning_max_new_tokens", 1024)
    answer_limit = raw.get("answer_max_new_tokens", 64)
    seed = raw.get("split_seed", 42)
    _integer(reasoning_limit, "reasoning_max_new_tokens", 1, errors)
    _integer(answer_limit, "answer_max_new_tokens", 1, errors)
    _integer(seed, "split_seed", 0, errors)

    abstention = raw.get("allow_abstention", True)
    if type(abstention) is not bool:
        errors.append("allow_abstention must be true or false.")

    ratios = _split_ratios(raw.get("split_ratios", {"train": 0.70, "validation": 0.15, "test": 0.15}), errors)
    if errors:
        raise ConfigurationError(errors)

    return TaskConfig(
        model_name_or_path=model,
        dataset_path=dataset,
        reasoning_mode=mode,
        output_dir=output,
        reasoning_max_new_tokens=reasoning_limit,
        answer_max_new_tokens=answer_limit,
        allow_abstention=abstention,
        split_ratios=ratios,
        split_seed=seed,
    )
