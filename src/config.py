"""Load and validate task configuration without running pipeline stages."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
import hashlib
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
    model_revision: str | None = None
    system_prompt_path: Path | None = None
    device: str = "auto"
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    reasoning_max_new_tokens: int = 1024
    answer_max_new_tokens: int = 64
    allow_abstention: bool = True
    answer_matcher_path: Path | None = None
    custom_metrics_path: Path | None = None
    generation_seed: int = 42
    direct_batch_size: int = 8
    decoding_strategy: Literal["model_default", "greedy"] = "model_default"
    probe_seed: int = 42
    probe_positions: tuple[str, ...] | None = None
    probe_layers: tuple[int, ...] | None = None
    probe_regularization_c: float = 1.0
    probe_class_weight: Literal["balanced", "none"] = "balanced"
    target_tpr: float = 0.90
    split_ratios: SplitRatios = SplitRatios()
    split_seed: int = 42
    system_prompt: str | None = field(
        default=None, repr=False, metadata={"yaml": False}
    )
    system_prompt_sha256: str | None = field(
        default=None, metadata={"yaml": False}
    )


_YAML_FIELDS = tuple(
    item for item in fields(TaskConfig) if item.metadata.get("yaml", True)
)
_FIELDS = {item.name for item in _YAML_FIELDS}
_DEFAULTS = {
    item.name: item.default
    for item in _YAML_FIELDS
    if item.default is not MISSING
}
_SPLIT_NAMES = tuple(field.name for field in fields(SplitRatios))
_MODEL_ID = re.compile(r"[\w][\w.-]*(?:/[\w][\w.-]*)?", re.ASCII)
_DEVICE = re.compile(r"(?:auto|cpu|cuda(?::\d+)?)", re.ASCII)
_PROBE_POSITION = re.compile(r"(?:prompt_end|final_prompt_end|answer_tokens|span_[1-9]\d*)")


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


def _string_list(
    value: object, field: str, errors: list[str]
) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a nonempty list of strings.")
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        errors.append(f"{field} must contain only nonempty strings.")
        return None
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        errors.append(f"{field} must not contain duplicates.")
        return None
    return normalized


def _split_ratios(value: object, errors: list[str]) -> SplitRatios | None:
    if not isinstance(value, dict):
        errors.append(
            "split_ratios must contain train, validation, and test fractions."
        )
        return None
    if set(value) != set(_SPLIT_NAMES):
        errors.append("split_ratios must contain exactly train, validation, and test.")
    fractions = {}
    for name in _SPLIT_NAMES:
        fraction = value.get(name)
        # This range check also rejects NaN, infinities, and oversized integers.
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            errors.append(
                f"split_ratios.{name} must be a finite number between 0 and 1, "
                "exclusive."
            )
        else:
            fractions[name] = float(fraction)
    if len(fractions) != 3:
        return None
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        errors.append("split_ratios must sum to 1.")
    return SplitRatios(**fractions)


def _read_yaml(path: str | Path) -> tuple[Path, dict]:
    """Read a settings mapping, reporting file and YAML errors clearly."""
    try:
        source = Path(path).resolve()
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigurationError(
            [f"Cannot read the YAML configuration ({type(exc).__name__})."]
        ) from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigurationError(
            [f"Invalid YAML{location}; check indentation and syntax."]
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(
            ["The YAML configuration must be a mapping of field names to values."]
        )
    return source, raw


def load_config(path: str | Path) -> TaskConfig:
    """Read YAML, apply omitted defaults, and return validated immutable settings.

    Explicit nulls remain invalid. Raises ConfigurationError for the caller to
    display; does not read data records, load models, print, or create directories.
    """
    source, raw = _read_yaml(path)
    values = _DEFAULTS | raw
    errors = [f"Unknown field '{name}'; remove it or correct its spelling."
              for name in sorted(raw.keys() - _FIELDS)]

    model = raw.get("model_name_or_path")
    if not isinstance(model, str) or not model.strip():
        errors.append(
            "model_name_or_path is required: supply a Hugging Face ID or "
            "absolute checkpoint path."
        )
    elif "\0" in model:
        errors.append("model_name_or_path contains a null character.")
    elif Path(model).is_absolute():
        if not Path(model).is_dir():
            errors.append("model_name_or_path must point to an existing checkpoint directory.")
    elif (
        not _MODEL_ID.fullmatch(model)
        or ".." in model
        or "--" in model
        or model.endswith((".", "-"))
    ):
        errors.append(
            "model_name_or_path must be a model ID (name or owner/name) or an "
            "absolute checkpoint path."
        )

    revision = values["model_revision"]
    if "model_revision" in raw and (
        not isinstance(revision, str) or not revision.strip()
    ):
        errors.append("model_revision must be a nonempty string when supplied.")
    if isinstance(model, str) and Path(model).is_absolute() and revision is not None:
        errors.append("model_revision cannot be used with a local checkpoint path.")

    device = values["device"]
    if not isinstance(device, str) or not _DEVICE.fullmatch(device):
        errors.append("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")

    if values["dtype"] not in ("auto", "float16", "bfloat16", "float32"):
        errors.append("dtype must be 'auto', 'float16', 'bfloat16', or 'float32'.")

    dataset = _absolute_path(raw.get("dataset_path"), "dataset_path", errors)
    values["dataset_path"] = dataset
    if dataset is not None:
        if dataset.suffix.lower() != ".jsonl":
            errors.append("dataset_path must name a .jsonl file.")
        if not dataset.is_file():
            errors.append("dataset_path must point to an existing file.")

    if "system_prompt_path" in raw:
        prompt_path = _absolute_path(
            raw["system_prompt_path"], "system_prompt_path", errors
        )
        values["system_prompt_path"] = prompt_path
        if prompt_path is not None:
            try:
                prompt_bytes = prompt_path.read_bytes()
                prompt_text = prompt_bytes.decode("utf-8")
            except (OSError, UnicodeError):
                errors.append(
                    "system_prompt_path must point to a readable UTF-8 text file."
                )
            else:
                if not prompt_text.strip():
                    errors.append(
                        "system_prompt_path must not contain an empty prompt."
                    )
                else:
                    values["system_prompt"] = prompt_text
                    values["system_prompt_sha256"] = hashlib.sha256(
                        prompt_bytes
                    ).hexdigest()

    if raw.get("reasoning_mode") not in ("direct", "reasoning"):
        errors.append("reasoning_mode is required and must be 'direct' or 'reasoning'.")

    if values["decoding_strategy"] not in ("model_default", "greedy"):
        errors.append(
            "decoding_strategy must be 'model_default' or 'greedy'."
        )

    if "output_dir" in raw:
        output = _absolute_path(raw["output_dir"], "output_dir", errors)
    else:
        output = source.parent / "outputs"
    if output is not None and output.exists() and not output.is_dir():
        errors.append("output_dir points to a file; supply a directory path instead.")
    values["output_dir"] = output

    integer_fields = (
        ("reasoning_max_new_tokens", 1),
        ("answer_max_new_tokens", 1),
        ("generation_seed", 0),
        ("direct_batch_size", 1),
        ("probe_seed", 0),
        ("split_seed", 0),
    )
    for name, minimum in integer_fields:
        _integer(values[name], name, minimum, errors)

    if "probe_positions" in raw:
        positions = _string_list(raw["probe_positions"], "probe_positions", errors)
        values["probe_positions"] = positions
        if positions is not None and any(
            not _PROBE_POSITION.fullmatch(position) for position in positions
        ):
            errors.append(
                "probe_positions entries must be prompt_end, final_prompt_end, "
                "answer_tokens, or span_N."
            )

    if "probe_layers" in raw:
        layers = raw["probe_layers"]
        if (
            not isinstance(layers, list)
            or not layers
            or any(type(layer) is not int or layer < 0 for layer in layers)
        ):
            errors.append("probe_layers must be a nonempty list of integers >= 0.")
        elif len(set(layers)) != len(layers):
            errors.append("probe_layers must not contain duplicates.")
        else:
            values["probe_layers"] = tuple(layers)

    regularization = values["probe_regularization_c"]
    if (
        type(regularization) not in (int, float)
        or not math.isfinite(regularization)
        or regularization <= 0
    ):
        errors.append("probe_regularization_c must be a finite number greater than 0.")
    else:
        values["probe_regularization_c"] = float(regularization)

    if values["probe_class_weight"] not in ("balanced", "none"):
        errors.append("probe_class_weight must be 'balanced' or 'none'.")

    target_tpr = values["target_tpr"]
    if type(target_tpr) not in (int, float) or not 0 < target_tpr <= 1:
        errors.append("target_tpr must be a finite number greater than 0 and at most 1.")
    else:
        values["target_tpr"] = float(target_tpr)

    if type(values["allow_abstention"]) is not bool:
        errors.append("allow_abstention must be true or false.")

    for field_name in ("answer_matcher_path", "custom_metrics_path"):
        if field_name not in raw:
            continue
        path = _absolute_path(raw[field_name], field_name, errors)
        values[field_name] = path
        if path is not None:
            if path.suffix.lower() != ".py":
                errors.append(f"{field_name} must name a .py file.")
            if not path.is_file():
                errors.append(f"{field_name} must point to an existing file.")

    if "split_ratios" in raw:
        values["split_ratios"] = _split_ratios(raw["split_ratios"], errors)
    if errors:
        raise ConfigurationError(errors)

    return TaskConfig(**values)
