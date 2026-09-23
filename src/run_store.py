"""Create and reuse minimal records for validated runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from src.config import TaskConfig


IDENTITY_SCHEMA_VERSION = 1
_RUN_ID_LENGTH = 12
_RUN_RECORD = "run.json"


class RunStoreError(ValueError):
    """A run directory or record cannot be safely created or reused."""


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    fingerprint: str
    dataset_sha256: str
    effective_configuration: dict


@dataclass(frozen=True)
class RegisteredRun:
    directory: Path
    created: bool


def build_run_identity(
    config: TaskConfig,
    dataset_sha256: str | None,
    split_source: Literal["dataset", "automatic"],
) -> RunIdentity:
    """Hash the inputs that can affect the current pipeline behavior."""
    if not dataset_sha256:
        raise RunStoreError("Dataset content hash is unavailable.")
    effective = {
        "model_name_or_path": config.model_name_or_path,
        "reasoning_mode": config.reasoning_mode,
        "answer_max_new_tokens": config.answer_max_new_tokens,
        "allow_abstention": config.allow_abstention,
    }
    if config.reasoning_mode == "reasoning":
        effective["reasoning_max_new_tokens"] = config.reasoning_max_new_tokens
    if split_source == "automatic":
        effective["split_ratios"] = {
            "train": config.split_ratios.train,
            "validation": config.split_ratios.validation,
            "test": config.split_ratios.test,
        }
        effective["split_seed"] = config.split_seed

    payload = {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "effective_configuration": effective,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    return RunIdentity(
        fingerprint[:_RUN_ID_LENGTH], fingerprint, dataset_sha256, effective
    )


def _record(
    identity: RunIdentity,
    config: TaskConfig,
    config_path: str | Path,
    preparation: dict,
) -> dict:
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "run_id": identity.run_id,
        "fingerprint": identity.fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_sha256": identity.dataset_sha256,
        "effective_configuration": identity.effective_configuration,
        "source_paths": {
            "configuration": str(Path(config_path).resolve()),
            "dataset": str(config.dataset_path),
        },
        "completed_stages": ["preparation"],
        "preparation": preparation,
    }


def _load_record(path: Path) -> dict:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStoreError(f"Cannot read valid run record: {path}") from exc
    if not isinstance(record, dict):
        raise RunStoreError(f"Run record must contain a JSON object: {path}")
    return record


def _validate_record(record: dict, identity: RunIdentity, path: Path) -> None:
    if record.get("schema_version") != IDENTITY_SCHEMA_VERSION:
        raise RunStoreError(f"Run record has an unsupported schema version: {path}")
    if record.get("run_id") != identity.run_id:
        raise RunStoreError(f"Run record ID does not match its directory: {path}")
    if record.get("fingerprint") != identity.fingerprint:
        raise RunStoreError(f"Run ID collision or conflicting run record: {path}")
    if (
        record.get("dataset_sha256") != identity.dataset_sha256
        or record.get("effective_configuration") != identity.effective_configuration
    ):
        raise RunStoreError(f"Run record contents conflict with its fingerprint: {path}")
    stages = record.get("completed_stages")
    if (
        not isinstance(stages, list)
        or not all(isinstance(stage, str) for stage in stages)
        or "preparation" not in stages
    ):
        raise RunStoreError(f"Run record has invalid completed stages: {path}")


def _write_record(path: Path, record: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RunStoreError(f"Cannot write run record: {path}") from exc


def register_run(
    identity: RunIdentity,
    config: TaskConfig,
    config_path: str | Path,
    preparation: dict,
) -> RegisteredRun:
    """Create a run record or safely reuse an identical existing record."""
    directory = config.output_dir / identity.run_id
    record_path = directory / _RUN_RECORD

    if directory.exists() and not directory.is_dir():
        raise RunStoreError(f"Run path is not a directory: {directory}")
    if record_path.exists():
        record = _load_record(record_path)
        _validate_record(record, identity, record_path)
        if record.get("preparation") != preparation:
            raise RunStoreError(f"Run preparation summary conflicts with: {record_path}")
        return RegisteredRun(directory, created=False)

    try:
        if directory.exists() and any(directory.iterdir()):
            raise RunStoreError(
                f"Run directory is nonempty but has no run.json: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RunStoreError(f"Cannot create run directory: {directory}") from exc
    _write_record(record_path, _record(identity, config, config_path, preparation))
    return RegisteredRun(directory, created=True)
