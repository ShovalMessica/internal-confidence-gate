"""Create and reuse minimal records for validated runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from src.activation import (
    ACTIVATION_PROTOCOL_VERSION,
    CAPTURE_BACKEND,
    POSITION_PROTOCOL,
    STORAGE_DTYPE,
)
from src.activation_store import (
    ACTIVATION_FILE,
    ActivationIdentity,
    activation_file_path,
)
from src.config import TaskConfig
from src.evaluation import (
    AnswerMatcher,
    EvaluationIdentity,
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_RECORD_SCHEMA_VERSION,
)
from src.generation import (
    GENERATION_PROTOCOL_VERSION,
    GENERATION_RECORD_SCHEMA_VERSION,
)
from src.generation_cache import (
    CachedGeneration,
    GenerationContext,
    example_key,
    load_generation,
    store_generation,
)


IDENTITY_SCHEMA_VERSION = 4
_RUN_ID_LENGTH = 12
_RUN_RECORD = "run.json"
_GENERATION_ARTIFACT = "generation-manifest.jsonl"
_GENERATION_MANIFEST_SCHEMA_VERSION = 1
_EVALUATIONS_DIR = "evaluations"
_LEGACY_EVALUATION_ARTIFACT = "evaluations.jsonl"


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
        "activation_protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "position_protocol": POSITION_PROTOCOL,
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


def evaluation_artifact_path(directory: Path, evaluation_id: str) -> Path:
    return directory / _EVALUATIONS_DIR / f"{evaluation_id}.jsonl"


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


def _append_jsonl_records(path: Path, records: list[dict]) -> None:
    try:
        with path.open("a", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise RunStoreError(f"Cannot append JSONL artifact: {path}") from exc


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


def _manifest_entry(
    context: GenerationContext, example: dict, cached: CachedGeneration
) -> dict:
    return {
        "schema_version": _GENERATION_MANIFEST_SCHEMA_VERSION,
        "context_id": context.context_id,
        "id": example["id"],
        "split": example["split"],
        "example_key": cached.example_key,
        "artifact_sha256": cached.artifact_sha256,
    }


def _validate_manifest_entry(
    entry: object,
    context: GenerationContext,
    expected: dict[int, dict],
    line: int,
) -> dict:
    if not isinstance(entry, dict):
        raise RunStoreError(f"Generation manifest line {line} must be an object.")
    example_id = entry.get("id")
    example = expected.get(example_id) if type(example_id) is int else None
    if (
        entry.get("schema_version") != _GENERATION_MANIFEST_SCHEMA_VERSION
        or entry.get("context_id") != context.context_id
        or example is None
        or entry.get("split") != example["split"]
        or entry.get("example_key") != example_key(context, example)
    ):
        raise RunStoreError(f"Generation manifest line {line} is invalid.")
    artifact_hash = entry.get("artifact_sha256")
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64:
        raise RunStoreError(f"Generation manifest line {line} has an invalid hash.")
    return entry


def load_generation_records(
    directory: Path, context: GenerationContext, examples: list[dict]
) -> tuple[dict[int, dict], bool]:
    """Resolve a run manifest into generation records."""
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
    entries: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunStoreError(
                f"Invalid generation JSON at line {line_number}: {path}"
            ) from exc
        entry = _validate_manifest_entry(raw, context, expected, line_number)
        example_id = entry["id"]
        if example_id in records:
            raise RunStoreError(
                f"Duplicate generation manifest entry for ID {example_id}: {path}"
            )
        try:
            cached = load_generation(
                context, expected[example_id], entry["artifact_sha256"]
            )
        except ValueError as exc:
            raise RunStoreError(str(exc)) from exc
        if cached is None or cached.example_key != entry["example_key"]:
            raise RunStoreError(f"Cached generation is missing for ID {example_id}.")
        record = _validate_generation_record(cached.record, expected, line_number)
        records[example_id] = record
        entries.append(entry)

    if recovered:
        _write_jsonl_records(path, entries)
    return records, recovered


def _append_manifest_entries(directory: Path, entries: list[dict]) -> None:
    _append_jsonl_records(generation_artifact_path(directory), entries)


def append_generation_records(
    directory: Path,
    context: GenerationContext,
    examples: dict[int, dict],
    records: list[dict],
) -> None:
    entries = []
    for record in records:
        example = examples[record["id"]]
        try:
            cached = store_generation(context, example, record)
        except ValueError as exc:
            raise RunStoreError(str(exc)) from exc
        entries.append(_manifest_entry(context, example, cached))
    _append_manifest_entries(directory, entries)


def reuse_cached_generation_records(
    directory: Path,
    context: GenerationContext,
    examples: list[dict],
    completed_ids: set[int],
) -> dict[int, dict]:
    """Attach unchanged cross-run generations to the current run."""
    reused: dict[int, dict] = {}
    entries: list[dict] = []
    for example in examples:
        if example["id"] in completed_ids:
            continue
        try:
            cached = load_generation(context, example)
        except ValueError as exc:
            raise RunStoreError(str(exc)) from exc
        if cached is not None:
            record = _validate_generation_record(cached.record, {example["id"]: example}, 1)
            reused[example["id"]] = record
            entries.append(_manifest_entry(context, example, cached))
    if entries:
        _append_manifest_entries(directory, entries)
    return reused


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
    directory: Path,
    context: GenerationContext,
    records: dict[int, dict],
    examples: list[dict],
) -> tuple[str, dict[str, int]]:
    if set(records) != {example["id"] for example in examples}:
        raise RunStoreError("Cannot finalize generation before every example is recorded.")
    ordered = [records[example["id"]] for example in examples]
    entries = []
    for example, record in zip(examples, ordered):
        try:
            cached = store_generation(context, example, record)
        except ValueError as exc:
            raise RunStoreError(str(exc)) from exc
        entries.append(_manifest_entry(context, example, cached))
    path = generation_artifact_path(directory)
    _write_jsonl_records(path, entries)
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
    directory: Path, evaluation_id: str, examples: list[dict]
) -> dict[int, dict]:
    """Load and validate a complete evaluation artifact."""
    path = evaluation_artifact_path(directory, evaluation_id)
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


def write_evaluation_records(
    directory: Path, evaluation_id: str, records: list[dict]
) -> str:
    path = evaluation_artifact_path(directory, evaluation_id)
    try:
        path.parent.mkdir(exist_ok=True)
    except OSError as exc:
        raise RunStoreError(f"Cannot create evaluation directory: {path.parent}") from exc
    _write_jsonl_records(path, records)
    return _sha256(path)


def complete_evaluation(
    directory: Path,
    identity: EvaluationIdentity,
    matcher: AnswerMatcher,
    artifact_sha256: str,
    summary: dict,
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "evaluation" not in stages:
        stages.append("evaluation")
    evaluations = record.setdefault("evaluations", {})
    if not isinstance(evaluations, dict):
        raise RunStoreError("Run record has an invalid evaluation registry.")
    artifact = f"{_EVALUATIONS_DIR}/{identity.evaluation_id}.jsonl"
    entry = {
        "evaluation_id": identity.evaluation_id,
        "fingerprint": identity.fingerprint,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "artifact": artifact,
        "artifact_sha256": artifact_sha256,
        "generation_sha256": identity.generation_sha256,
        "matcher": matcher.metadata(),
        "summary": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = evaluations.get(identity.evaluation_id)
    if existing is not None and existing != entry:
        raise RunStoreError("Evaluation ID collision or conflicting run record.")
    evaluations[identity.evaluation_id] = entry
    _save_run_record(directory, record)


def validate_completed_evaluation(
    directory: Path, record: dict, identity: EvaluationIdentity
) -> bool:
    evaluations = record.get("evaluations")
    if evaluations is None:
        return False
    if not isinstance(evaluations, dict):
        raise RunStoreError("Run record has an invalid evaluation registry.")
    evaluation = evaluations.get(identity.evaluation_id)
    if evaluation is None:
        return False
    if "evaluation" not in record.get("completed_stages", []):
        raise RunStoreError("Saved evaluation is missing its completed stage.")
    if not isinstance(evaluation, dict):
        raise RunStoreError("Completed evaluation has no valid run summary.")
    if (
        evaluation.get("evaluation_id") != identity.evaluation_id
        or evaluation.get("fingerprint") != identity.fingerprint
    ):
        raise RunStoreError("Completed evaluation identity is invalid.")
    if evaluation.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise RunStoreError("Completed evaluation uses an unsupported protocol version.")
    if evaluation.get("generation_sha256") != identity.generation_sha256:
        raise RunStoreError("Completed evaluation does not match saved generation.")
    matcher = evaluation.get("matcher")
    if not isinstance(matcher, dict) or matcher.get("sha256") != identity.matcher_sha256:
        raise RunStoreError("Completed evaluation has invalid matcher provenance.")
    expected_artifact = f"{_EVALUATIONS_DIR}/{identity.evaluation_id}.jsonl"
    if evaluation.get("artifact") != expected_artifact:
        raise RunStoreError("Completed evaluation has an invalid artifact path.")
    path = evaluation_artifact_path(directory, identity.evaluation_id)
    if not path.is_file() or _sha256(path) != evaluation.get("artifact_sha256"):
        raise RunStoreError("Completed evaluation artifact is missing or has changed.")
    if not isinstance(evaluation.get("summary"), dict):
        raise RunStoreError("Completed evaluation summary is invalid.")
    return True


def complete_activation_capture(
    directory: Path,
    identity: ActivationIdentity,
    artifact_sha256: str,
    summary: dict,
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "activation_capture" not in stages:
        stages.append("activation_capture")
    entry = {
        "capture_id": identity.capture_id,
        "fingerprint": identity.fingerprint,
        "context_id": identity.context_id,
        "context_fingerprint": identity.context_fingerprint,
        "protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "backend": CAPTURE_BACKEND,
        "position_protocol": POSITION_PROTOCOL,
        "storage_dtype": STORAGE_DTYPE,
        "generation_sha256": identity.generation_sha256,
        "request_fingerprint": identity.request_fingerprint,
        "artifact": ACTIVATION_FILE,
        "artifact_sha256": artifact_sha256,
        "summary": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = record.get("activation_capture")
    if existing is not None and existing != entry:
        raise RunStoreError("Activation capture ID collision or conflicting run record.")
    record["activation_capture"] = entry
    _save_run_record(directory, record)


def validate_completed_activation_capture(
    directory: Path, record: dict, identity: ActivationIdentity
) -> bool:
    capture = record.get("activation_capture")
    if capture is None:
        return False
    if "activation_capture" not in record.get("completed_stages", []):
        raise RunStoreError("Saved activations are missing their completed stage.")
    if (
        not isinstance(capture, dict)
        or capture.get("capture_id") != identity.capture_id
        or capture.get("fingerprint") != identity.fingerprint
        or capture.get("context_id") != identity.context_id
        or capture.get("context_fingerprint") != identity.context_fingerprint
        or capture.get("protocol_version") != ACTIVATION_PROTOCOL_VERSION
        or capture.get("backend") != CAPTURE_BACKEND
        or capture.get("position_protocol") != POSITION_PROTOCOL
        or capture.get("storage_dtype") != STORAGE_DTYPE
        or capture.get("generation_sha256") != identity.generation_sha256
        or capture.get("request_fingerprint") != identity.request_fingerprint
        or capture.get("artifact") != ACTIVATION_FILE
        or not isinstance(capture.get("summary"), dict)
    ):
        raise RunStoreError("Completed activation capture is invalid.")
    path = activation_file_path(directory)
    if not path.is_file() or _sha256(path) != capture.get("artifact_sha256"):
        raise RunStoreError("Completed activation file is missing or has changed.")
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
    directory: Path,
    context: GenerationContext,
    artifact_sha256: str,
    counts: dict[str, int],
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "generation" not in stages:
        stages.append("generation")
    record["generation"] = {
        **counts,
        "context_id": context.context_id,
        "artifact": _GENERATION_ARTIFACT,
        "artifact_sha256": artifact_sha256,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _save_run_record(directory, record)


def validate_completed_generation(
    directory: Path, record: dict, context: GenerationContext
) -> bool:
    if "generation" not in record.get("completed_stages", []):
        return False
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise RunStoreError("Completed generation has no valid run summary.")
    if generation.get("context_id") != context.context_id:
        raise RunStoreError("Completed generation uses a different cache context.")
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
    record.pop("evaluations", None)
    record.pop("activation_capture", None)
    record.pop("activation_captures", None)
    record["completed_stages"] = [
        stage
        for stage in record["completed_stages"]
        if stage not in ("generation", "evaluation", "activation_capture")
    ]
    try:
        generation_artifact_path(directory).unlink(missing_ok=True)
        activation_file_path(directory).unlink(missing_ok=True)
        (directory / _LEGACY_EVALUATION_ARTIFACT).unlink(missing_ok=True)
        evaluations = directory / _EVALUATIONS_DIR
        if evaluations.exists():
            if not evaluations.is_dir():
                raise OSError(f"Not a directory: {evaluations}")
            artifacts = list(evaluations.iterdir())
            unexpected = next(
                (artifact for artifact in artifacts if not artifact.is_file()), None
            )
            if unexpected is not None:
                raise OSError(f"Unexpected entry: {unexpected}")
            for artifact in artifacts:
                artifact.unlink()
            evaluations.rmdir()
        activations = directory / "activations"
        if activations.exists():
            if not activations.is_dir():
                raise OSError(f"Not a directory: {activations}")
            artifacts = list(activations.iterdir())
            unexpected = next(
                (artifact for artifact in artifacts if not artifact.is_file()), None
            )
            if unexpected is not None:
                raise OSError(f"Unexpected entry: {unexpected}")
            for artifact in artifacts:
                artifact.unlink()
            activations.rmdir()
    except OSError as exc:
        raise RunStoreError(
            "Cannot remove generation, evaluation, or activation artifacts."
        ) from exc
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
