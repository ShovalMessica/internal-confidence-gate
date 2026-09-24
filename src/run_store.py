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
from src.evaluation import (
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_RECORD_SCHEMA_VERSION,
)
from src.generation import (
    GENERATION_PROTOCOL_VERSION,
    GENERATION_RECORD_SCHEMA_VERSION,
)


IDENTITY_SCHEMA_VERSION = 3
_RUN_ID_LENGTH = 12
_RUN_RECORD = "run.json"
_GENERATION_ARTIFACT = "generations.jsonl"
_EVALUATION_ARTIFACT = "evaluations.jsonl"


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
        "device": config.device,
        "dtype": config.dtype,
        "reasoning_mode": config.reasoning_mode,
        "answer_max_new_tokens": config.answer_max_new_tokens,
        "allow_abstention": config.allow_abstention,
        "generation_seed": config.generation_seed,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
    }
    if config.model_revision is not None:
        effective["model_revision"] = config.model_revision
    if config.reasoning_mode == "reasoning":
        effective["reasoning_max_new_tokens"] = config.reasoning_max_new_tokens
    else:
        effective["direct_batch_size"] = config.direct_batch_size
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


def load_run_record(directory: Path) -> dict:
    """Load an already registered run record."""
    path = directory / _RUN_RECORD
    if not path.is_file():
        raise RunStoreError(f"Run record is missing: {path}")
    return _load_record(path)


def _save_run_record(directory: Path, record: dict) -> None:
    _write_record(directory / _RUN_RECORD, record)


def generation_artifact_path(directory: Path) -> Path:
    return directory / _GENERATION_ARTIFACT


def evaluation_artifact_path(directory: Path) -> Path:
    return directory / _EVALUATION_ARTIFACT


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RunStoreError(f"Cannot write JSONL artifact: {path}") from exc


def _validate_generation_record(record: object, expected: dict, line: int) -> dict:
    if not isinstance(record, dict):
        raise RunStoreError(f"Generation record at line {line} must be an object.")
    if record.get("schema_version") != GENERATION_RECORD_SCHEMA_VERSION:
        raise RunStoreError(
            f"Generation record at line {line} has an unsupported schema version."
        )
    if record.get("protocol_version") != GENERATION_PROTOCOL_VERSION:
        raise RunStoreError(
            f"Generation record at line {line} has an unsupported protocol version."
        )
    example_id = record.get("id")
    if type(example_id) is not int or example_id not in expected:
        raise RunStoreError(
            f"Generation record at line {line} has an unexpected example ID."
        )
    if record.get("split") != expected[example_id]["split"]:
        raise RunStoreError(
            f"Generation record at line {line} has the wrong split for ID {example_id}."
        )
    if record.get("status") not in ("success", "failed"):
        raise RunStoreError(
            f"Generation record at line {line} has an invalid status."
        )
    if record["status"] == "success" and not all(
        name in record
        for name in ("formatted_prompt_token_ids", "final_control", "answer")
    ):
        raise RunStoreError(
            f"Successful generation record at line {line} is incomplete."
        )
    if record["status"] == "failed" and "failure" not in record:
        raise RunStoreError(
            f"Failed generation record at line {line} has no failure reason."
        )
    return record


def load_generation_records(
    directory: Path, examples: list[dict]
) -> tuple[dict[int, dict], bool]:
    """Load resumable records and recover only a truncated final line."""
    path = generation_artifact_path(directory)
    if not path.exists():
        return {}, False
    if not path.is_file():
        raise RunStoreError(f"Generation artifact is not a file: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise RunStoreError(f"Cannot read generation artifact: {path}") from exc

    recovered = bool(content and not content.endswith(b"\n"))
    if recovered:
        content = content[: content.rfind(b"\n") + 1] if b"\n" in content else b""
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise RunStoreError(f"Generation artifact is not valid UTF-8: {path}") from exc

    expected = {example["id"]: example for example in examples}
    records: dict[int, dict] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunStoreError(
                f"Invalid generation JSON at line {line_number}: {path}"
            ) from exc
        record = _validate_generation_record(raw, expected, line_number)
        example_id = record["id"]
        if example_id in records:
            raise RunStoreError(
                f"Duplicate generation record for ID {example_id}: {path}"
            )
        records[example_id] = record

    if recovered:
        _write_jsonl_records(path, list(records.values()))
    return records, recovered


def reset_partial_direct_batches(
    directory: Path,
    records: dict[int, dict],
    examples: list[dict],
    batch_size: int,
) -> tuple[dict[int, dict], int]:
    """Drop incomplete fixed batches so resumed sampling uses the same grouping."""
    retained = dict(records)
    removed = 0
    for start in range(0, len(examples), batch_size):
        ids = [example["id"] for example in examples[start : start + batch_size]]
        present = [example_id in retained for example_id in ids]
        if any(present) and not all(present):
            for example_id in ids:
                removed += int(retained.pop(example_id, None) is not None)
    if removed:
        ordered = [retained[example["id"]] for example in examples if example["id"] in retained]
        _write_jsonl_records(generation_artifact_path(directory), ordered)
    return retained, removed


def append_generation_records(directory: Path, records: list[dict]) -> None:
    path = generation_artifact_path(directory)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise RunStoreError(f"Cannot append generation artifact: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunStoreError(f"Cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def finalize_generation_records(
    directory: Path, records: dict[int, dict], examples: list[dict]
) -> tuple[str, dict[str, int]]:
    if set(records) != {example["id"] for example in examples}:
        raise RunStoreError("Cannot finalize generation before every example is recorded.")
    ordered = [records[example["id"]] for example in examples]
    path = generation_artifact_path(directory)
    _write_jsonl_records(path, ordered)
    counts = {
        "total": len(ordered),
        "successful": sum(record["status"] == "success" for record in ordered),
        "failed": sum(record["status"] == "failed" for record in ordered),
    }
    return _sha256(path), counts


def _validate_evaluation_record(record: object, expected: dict, line: int) -> dict:
    if not isinstance(record, dict):
        raise RunStoreError(f"Evaluation record at line {line} must be an object.")
    if record.get("schema_version") != EVALUATION_RECORD_SCHEMA_VERSION:
        raise RunStoreError(
            f"Evaluation record at line {line} has an unsupported schema version."
        )
    if record.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise RunStoreError(
            f"Evaluation record at line {line} has an unsupported protocol version."
        )
    example_id = record.get("id")
    if type(example_id) is not int or example_id not in expected:
        raise RunStoreError(
            f"Evaluation record at line {line} has an unexpected example ID."
        )
    if record.get("split") != expected[example_id]["split"]:
        raise RunStoreError(
            f"Evaluation record at line {line} has the wrong split for ID {example_id}."
        )
    outcome = record.get("outcome")
    if outcome not in ("correct", "incorrect", "abstained", "invalid"):
        raise RunStoreError(
            f"Evaluation record at line {line} has an invalid outcome."
        )
    expected_correct = {"correct": True, "incorrect": False}.get(outcome)
    if (
        "is_correct" not in record
        or record.get("is_correct") is not expected_correct
    ):
        raise RunStoreError(
            f"Evaluation record at line {line} has inconsistent correctness."
        )
    normalized = record.get("normalized_answer")
    if outcome == "invalid":
        if normalized is not None or not isinstance(record.get("invalid_reason"), str):
            raise RunStoreError(
                f"Invalid evaluation record at line {line} is incomplete."
            )
    elif not isinstance(normalized, str) or not normalized:
        raise RunStoreError(
            f"Evaluation record at line {line} has no normalized answer."
        )
    return record


def load_evaluation_records(
    directory: Path, examples: list[dict]
) -> dict[int, dict]:
    """Load and validate a complete evaluation artifact."""
    path = evaluation_artifact_path(directory)
    if not path.is_file():
        raise RunStoreError(f"Evaluation artifact is missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RunStoreError(f"Cannot read evaluation artifact: {path}") from exc
    if text and not text.endswith("\n"):
        raise RunStoreError(f"Evaluation artifact is truncated: {path}")

    expected = {example["id"]: example for example in examples}
    records: dict[int, dict] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunStoreError(
                f"Invalid evaluation JSON at line {line_number}: {path}"
            ) from exc
        record = _validate_evaluation_record(raw, expected, line_number)
        example_id = record["id"]
        if example_id in records:
            raise RunStoreError(
                f"Duplicate evaluation record for ID {example_id}: {path}"
            )
        records[example_id] = record
    if set(records) != set(expected):
        raise RunStoreError("Evaluation artifact does not contain every example.")
    expected_order = [example["id"] for example in examples]
    if list(records) != expected_order:
        raise RunStoreError("Evaluation artifact is not in dataset order.")
    return records


def write_evaluation_records(directory: Path, records: list[dict]) -> str:
    path = evaluation_artifact_path(directory)
    _write_jsonl_records(path, records)
    return _sha256(path)


def complete_evaluation(
    directory: Path,
    artifact_sha256: str,
    generation_sha256: str,
    summary: dict,
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "evaluation" not in stages:
        stages.append("evaluation")
    record["evaluation"] = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "artifact": _EVALUATION_ARTIFACT,
        "artifact_sha256": artifact_sha256,
        "generation_sha256": generation_sha256,
        "summary": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _save_run_record(directory, record)


def validate_completed_evaluation(
    directory: Path, record: dict, generation_sha256: str
) -> bool:
    if "evaluation" not in record.get("completed_stages", []):
        return False
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RunStoreError("Completed evaluation has no valid run summary.")
    if evaluation.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise RunStoreError("Completed evaluation uses an unsupported protocol version.")
    if evaluation.get("generation_sha256") != generation_sha256:
        raise RunStoreError("Completed evaluation does not match saved generation.")
    path = evaluation_artifact_path(directory)
    if not path.is_file() or _sha256(path) != evaluation.get("artifact_sha256"):
        raise RunStoreError("Completed evaluation artifact is missing or has changed.")
    if not isinstance(evaluation.get("summary"), dict):
        raise RunStoreError("Completed evaluation summary is invalid.")
    return True


def save_model_metadata(
    directory: Path, metadata: dict, provenance: dict, generation_settings: dict
) -> None:
    record = load_run_record(directory)
    existing = record.get("model")
    if existing is not None and existing != metadata:
        raise RunStoreError(
            "Loaded model metadata differs from the interrupted run; use "
            "--force-recompute to restart generation."
        )
    if existing is None:
        record["model"] = metadata
        record["generation_settings"] = generation_settings
        record["generation_started_at_utc"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        record["execution"] = provenance
        _save_run_record(directory, record)


def complete_generation(
    directory: Path, artifact_sha256: str, counts: dict[str, int]
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "generation" not in stages:
        stages.append("generation")
    record["generation"] = {
        **counts,
        "artifact": _GENERATION_ARTIFACT,
        "artifact_sha256": artifact_sha256,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _save_run_record(directory, record)


def validate_completed_generation(directory: Path, record: dict) -> bool:
    if "generation" not in record.get("completed_stages", []):
        return False
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise RunStoreError("Completed generation has no valid run summary.")
    path = generation_artifact_path(directory)
    if not path.is_file() or _sha256(path) != generation.get("artifact_sha256"):
        raise RunStoreError("Completed generation artifact is missing or has changed.")
    return True


def reset_generation(directory: Path) -> str | None:
    """Remove generation state while returning its pinned Hub revision."""
    record = load_run_record(directory)
    model = record.pop("model", None)
    pinned = model.get("resolved_revision") if isinstance(model, dict) else None
    record.pop("generation", None)
    record.pop("generation_started_at_utc", None)
    record.pop("execution", None)
    record.pop("generation_settings", None)
    record.pop("evaluation", None)
    record["completed_stages"] = [
        stage
        for stage in record["completed_stages"]
        if stage not in ("generation", "evaluation")
    ]
    try:
        generation_artifact_path(directory).unlink(missing_ok=True)
        evaluation_artifact_path(directory).unlink(missing_ok=True)
    except OSError as exc:
        raise RunStoreError("Cannot remove generation or evaluation artifacts.") from exc
    _save_run_record(directory, record)
    return pinned


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
