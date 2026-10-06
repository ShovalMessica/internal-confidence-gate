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
    CustomMetrics,
    CustomMetricsIdentity,
    CUSTOM_METRICS_PROTOCOL_VERSION,
    EvaluationError,
    EvaluationIdentity,
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_RECORD_SCHEMA_VERSION,
    validate_custom_metrics_results,
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
from src.probe import (
    PROBE_FILE,
    PROBE_PROTOCOL_VERSION,
    SELECTION_PROTOCOL_VERSION,
    TEST_EVALUATION_PROTOCOL_VERSION,
    ProbeIdentity,
    SelectionIdentity,
    TestEvaluationIdentity,
    probe_file_path,
    validate_probe_group,
    validate_selection_group,
    validate_test_evaluation_group,
)
from src.reporting import (
    REPORT_PROTOCOL_VERSION,
    REPORTS_DIR,
    ReportIdentity,
    validate_report,
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
        "decoding_strategy": config.decoding_strategy,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "activation_protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "position_protocol": POSITION_PROTOCOL,
    }
    if config.system_prompt_sha256 is not None:
        effective["system_prompt_sha256"] = config.system_prompt_sha256
    if config.probe_excluded_answers:
        effective["probe_excluded_answers"] = list(config.probe_excluded_answers)
    if config.model_revision is not None:
        effective["model_revision"] = config.model_revision
    if config.reasoning_mode == "reasoning":
        effective["reasoning_max_new_tokens"] = config.reasoning_max_new_tokens
    else:
        effective["direct_batch_size"] = config.direct_batch_size
        effective["direct_output_format"] = config.direct_output_format
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
            **(
                {"system_prompt": str(config.system_prompt_path)}
                if config.system_prompt_path is not None
                else {}
            ),
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


def complete_custom_metrics(
    directory: Path,
    identity: CustomMetricsIdentity,
    custom_metrics: CustomMetrics,
    results: dict,
) -> None:
    """Register a completed custom Model Behavior summary."""
    record = load_run_record(directory)
    registry = record.setdefault("custom_metrics", {})
    if not isinstance(registry, dict):
        raise RunStoreError("Run record has an invalid custom-metrics registry.")
    entry = {
        "metrics_id": identity.metrics_id,
        "fingerprint": identity.fingerprint,
        "protocol_version": CUSTOM_METRICS_PROTOCOL_VERSION,
        "dataset_sha256": identity.dataset_sha256,
        "generation_sha256": identity.generation_sha256,
        "evaluation_sha256": identity.evaluation_sha256,
        "implementation": custom_metrics.metadata(),
        "results": results,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = registry.get(identity.metrics_id)
    if existing is not None and existing != entry:
        raise RunStoreError("Custom metrics ID collision or conflicting run record.")
    registry[identity.metrics_id] = entry
    _save_run_record(directory, record)


def validate_completed_custom_metrics(
    record: dict, identity: CustomMetricsIdentity
) -> dict | None:
    """Return saved results when a matching custom summary is complete."""
    registry = record.get("custom_metrics")
    if registry is None:
        return None
    if not isinstance(registry, dict):
        raise RunStoreError("Run record has an invalid custom-metrics registry.")
    entry = registry.get(identity.metrics_id)
    if entry is None:
        return None
    if (
        not isinstance(entry, dict)
        or entry.get("metrics_id") != identity.metrics_id
        or entry.get("fingerprint") != identity.fingerprint
        or entry.get("protocol_version") != CUSTOM_METRICS_PROTOCOL_VERSION
        or entry.get("dataset_sha256") != identity.dataset_sha256
        or entry.get("generation_sha256") != identity.generation_sha256
        or entry.get("evaluation_sha256") != identity.evaluation_sha256
    ):
        raise RunStoreError("Completed custom metrics identity is invalid.")
    implementation = entry.get("implementation")
    if (
        not isinstance(implementation, dict)
        or implementation.get("sha256") != identity.implementation_sha256
    ):
        raise RunStoreError("Completed custom metrics provenance is invalid.")
    results = entry.get("results")
    try:
        return validate_custom_metrics_results(results)
    except EvaluationError as exc:
        raise RunStoreError("Completed custom metrics results are invalid.") from exc


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


def complete_probe_training(
    directory: Path,
    identity: ProbeIdentity,
    content_sha256: str,
    summary: dict,
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "probe_training" not in stages:
        stages.append("probe_training")
    trainings = record.setdefault("probe_trainings", {})
    if not isinstance(trainings, dict):
        raise RunStoreError("Run record has an invalid probe-training registry.")
    entry = {
        "probe_id": identity.probe_id,
        "fingerprint": identity.fingerprint,
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "seed": identity.seed,
        "activation_sha256": identity.activation_sha256,
        "evaluation_sha256": identity.evaluation_sha256,
        "artifact": PROBE_FILE,
        "group": f"trainings/{identity.probe_id}",
        "content_sha256": content_sha256,
        "summary": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = trainings.get(identity.probe_id)
    if existing is not None and existing != entry:
        raise RunStoreError("Probe ID collision or conflicting run record.")
    trainings[identity.probe_id] = entry
    _save_run_record(directory, record)


def validate_completed_probe_training(
    directory: Path, record: dict, identity: ProbeIdentity
) -> bool:
    trainings = record.get("probe_trainings")
    if trainings is None:
        return False
    if not isinstance(trainings, dict):
        raise RunStoreError("Run record has an invalid probe-training registry.")
    training = trainings.get(identity.probe_id)
    if training is None:
        return False
    if "probe_training" not in record.get("completed_stages", []):
        raise RunStoreError("Saved probes are missing their completed stage.")
    if (
        not isinstance(training, dict)
        or training.get("probe_id") != identity.probe_id
        or training.get("fingerprint") != identity.fingerprint
        or training.get("protocol_version") != PROBE_PROTOCOL_VERSION
        or training.get("seed") != identity.seed
        or training.get("activation_sha256") != identity.activation_sha256
        or training.get("evaluation_sha256") != identity.evaluation_sha256
        or training.get("artifact") != PROBE_FILE
        or training.get("group") != f"trainings/{identity.probe_id}"
        or not isinstance(training.get("summary"), dict)
    ):
        raise RunStoreError("Completed probe training is invalid.")
    _, summary = validate_probe_group(
        directory, identity, training.get("content_sha256")
    )
    if summary != training["summary"]:
        raise RunStoreError("Completed probe summary is invalid.")
    return True


def complete_probe_selection(
    directory: Path,
    identity: SelectionIdentity,
    content_sha256: str,
    summary: dict,
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "probe_selection" not in stages:
        stages.append("probe_selection")
    selections = record.setdefault("probe_selections", {})
    if not isinstance(selections, dict):
        raise RunStoreError("Run record has an invalid probe-selection registry.")
    entry = {
        "selection_id": identity.selection_id,
        "fingerprint": identity.fingerprint,
        "protocol_version": SELECTION_PROTOCOL_VERSION,
        "probe_id": identity.probe_id,
        "probe_sha256": identity.probe_sha256,
        "target_tpr": identity.target_tpr,
        "artifact": PROBE_FILE,
        "group": f"selections/{identity.selection_id}",
        "content_sha256": content_sha256,
        "selected": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = selections.get(identity.selection_id)
    if existing is not None and existing != entry:
        raise RunStoreError("Probe selection ID collision or conflicting run record.")
    selections[identity.selection_id] = entry
    _save_run_record(directory, record)


def validate_completed_probe_selection(
    directory: Path,
    record: dict,
    probe_identity: ProbeIdentity,
    identity: SelectionIdentity,
) -> bool:
    selections = record.get("probe_selections")
    if selections is None:
        return False
    if not isinstance(selections, dict):
        raise RunStoreError("Run record has an invalid probe-selection registry.")
    selection = selections.get(identity.selection_id)
    if selection is None:
        return False
    if "probe_selection" not in record.get("completed_stages", []):
        raise RunStoreError("Saved probe selection is missing its completed stage.")
    if (
        not isinstance(selection, dict)
        or selection.get("selection_id") != identity.selection_id
        or selection.get("fingerprint") != identity.fingerprint
        or selection.get("protocol_version") != SELECTION_PROTOCOL_VERSION
        or selection.get("probe_id") != identity.probe_id
        or selection.get("probe_sha256") != identity.probe_sha256
        or selection.get("target_tpr") != identity.target_tpr
        or selection.get("artifact") != PROBE_FILE
        or selection.get("group") != f"selections/{identity.selection_id}"
        or not isinstance(selection.get("selected"), dict)
    ):
        raise RunStoreError("Completed probe selection is invalid.")
    _, summary = validate_selection_group(
        directory,
        probe_identity,
        identity,
        selection.get("content_sha256"),
    )
    if summary != selection["selected"]:
        raise RunStoreError("Completed probe-selection summary is invalid.")
    return True


def complete_test_evaluation(
    directory: Path,
    identity: TestEvaluationIdentity,
    content_sha256: str,
    summary: dict,
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "test_evaluation" not in stages:
        stages.append("test_evaluation")
    evaluations = record.setdefault("test_evaluations", {})
    if not isinstance(evaluations, dict):
        raise RunStoreError("Run record has an invalid test-evaluation registry.")
    entry = {
        "test_id": identity.test_id,
        "fingerprint": identity.fingerprint,
        "protocol_version": TEST_EVALUATION_PROTOCOL_VERSION,
        "selection_id": identity.selection_id,
        "selection_sha256": identity.selection_sha256,
        "generation_sha256": identity.generation_sha256,
        "artifact": PROBE_FILE,
        "group": f"test_evaluations/{identity.test_id}",
        "content_sha256": content_sha256,
        "summary": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = evaluations.get(identity.test_id)
    if existing is not None and existing != entry:
        raise RunStoreError("Test evaluation ID collision or conflicting run record.")
    evaluations[identity.test_id] = entry
    _save_run_record(directory, record)


def validate_completed_test_evaluation(
    directory: Path,
    record: dict,
    probe_identity: ProbeIdentity,
    selection_identity: SelectionIdentity,
    identity: TestEvaluationIdentity,
) -> bool:
    evaluations = record.get("test_evaluations")
    if evaluations is None:
        return False
    if not isinstance(evaluations, dict):
        raise RunStoreError("Run record has an invalid test-evaluation registry.")
    evaluation = evaluations.get(identity.test_id)
    if evaluation is None:
        return False
    if "test_evaluation" not in record.get("completed_stages", []):
        raise RunStoreError("Saved test evaluation is missing its completed stage.")
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("test_id") != identity.test_id
        or evaluation.get("fingerprint") != identity.fingerprint
        or evaluation.get("protocol_version") != TEST_EVALUATION_PROTOCOL_VERSION
        or evaluation.get("selection_id") != identity.selection_id
        or evaluation.get("selection_sha256") != identity.selection_sha256
        or evaluation.get("generation_sha256") != identity.generation_sha256
        or evaluation.get("artifact") != PROBE_FILE
        or evaluation.get("group") != f"test_evaluations/{identity.test_id}"
        or not isinstance(evaluation.get("summary"), dict)
    ):
        raise RunStoreError("Completed test evaluation is invalid.")
    _, summary = validate_test_evaluation_group(
        directory,
        probe_identity,
        selection_identity,
        identity,
        evaluation.get("content_sha256"),
    )
    if summary != evaluation["summary"]:
        raise RunStoreError("Completed test-evaluation summary is invalid.")
    return True


def complete_report(
    directory: Path,
    identity: ReportIdentity,
    artifact_sha256: str,
    summary: dict,
    files: dict[str, str],
) -> None:
    record = load_run_record(directory)
    stages = record["completed_stages"]
    if "reporting" not in stages:
        stages.append("reporting")
    reports = record.setdefault("reports", {})
    if not isinstance(reports, dict):
        raise RunStoreError("Run record has an invalid report registry.")
    entry = {
        "report_id": identity.report_id,
        "fingerprint": identity.fingerprint,
        "protocol_version": REPORT_PROTOCOL_VERSION,
        "selection_id": identity.selection_id,
        "selection_sha256": identity.selection_sha256,
        "test_id": identity.test_id,
        "test_sha256": identity.test_sha256,
        "directory": f"{REPORTS_DIR}/{identity.report_id}",
        "artifact_sha256": artifact_sha256,
        "files": files,
        "summary": summary,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    existing = reports.get(identity.report_id)
    if existing is not None and existing != entry:
        raise RunStoreError("Report ID collision or conflicting run record.")
    reports[identity.report_id] = entry
    _save_run_record(directory, record)


def validate_completed_report(
    directory: Path, record: dict, identity: ReportIdentity
) -> bool:
    reports = record.get("reports")
    if reports is None:
        return False
    if not isinstance(reports, dict):
        raise RunStoreError("Run record has an invalid report registry.")
    report = reports.get(identity.report_id)
    if report is None:
        return False
    if "reporting" not in record.get("completed_stages", []):
        raise RunStoreError("Saved report is missing its completed stage.")
    if (
        not isinstance(report, dict)
        or report.get("report_id") != identity.report_id
        or report.get("fingerprint") != identity.fingerprint
        or report.get("protocol_version") != REPORT_PROTOCOL_VERSION
        or report.get("selection_id") != identity.selection_id
        or report.get("selection_sha256") != identity.selection_sha256
        or report.get("test_id") != identity.test_id
        or report.get("test_sha256") != identity.test_sha256
        or report.get("directory") != f"{REPORTS_DIR}/{identity.report_id}"
        or not isinstance(report.get("summary"), dict)
        or not isinstance(report.get("files"), dict)
    ):
        raise RunStoreError("Completed report is invalid.")
    result = validate_report(directory, identity, report.get("artifact_sha256"))
    if result.summary != report["summary"] or result.files != report["files"]:
        raise RunStoreError("Completed report summary is invalid.")
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
    record.pop("custom_metrics", None)
    record.pop("activation_capture", None)
    record.pop("activation_captures", None)
    record.pop("probe_training", None)
    record.pop("probe_trainings", None)
    record.pop("probe_selections", None)
    record.pop("test_evaluations", None)
    record.pop("reports", None)
    record["completed_stages"] = [
        stage
        for stage in record["completed_stages"]
        if stage not in (
            "generation",
            "evaluation",
            "activation_capture",
            "probe_training",
            "probe_selection",
            "test_evaluation",
            "reporting",
        )
    ]
    try:
        generation_artifact_path(directory).unlink(missing_ok=True)
        activation_file_path(directory).unlink(missing_ok=True)
        probe_file_path(directory).unlink(missing_ok=True)
        reports = directory / REPORTS_DIR
        if reports.exists():
            if not reports.is_dir():
                raise OSError(f"Not a directory: {reports}")
            for report in reports.iterdir():
                if not report.is_dir():
                    raise OSError(f"Unexpected entry: {report}")
                for artifact in report.iterdir():
                    if not artifact.is_file():
                        raise OSError(f"Unexpected entry: {artifact}")
                    artifact.unlink()
                report.rmdir()
            reports.rmdir()
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
            "Cannot remove generation, evaluation, activation, probe, or report artifacts."
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
