"""Store every activation capture for a run in one resumable HDF5 file."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from src.activation import (
    ACTIVATION_PROTOCOL_VERSION,
    CAPTURE_BACKEND,
    POSITION_PROTOCOL,
    STORAGE_DTYPE,
    ActivationResult,
    ReplayPlan,
)
from src.config import TaskConfig
from src.generation_cache import GenerationContext


ACTIVATION_STORE_SCHEMA_VERSION = 1
ACTIVATION_FILE = "activations.h5"


class ActivationStoreError(ValueError):
    """An activation file is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class ActivationContext:
    context_id: str
    fingerprint: str
    settings: dict


@dataclass(frozen=True)
class ActivationIdentity:
    capture_id: str
    fingerprint: str
    context_id: str
    context_fingerprint: str
    generation_sha256: str


@dataclass(frozen=True)
class StoredActivation:
    metadata: dict[str, str]


ExpectedActivations = dict[int, tuple[dict, str, ReplayPlan]]


def _encoded(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ActivationStoreError(f"Cannot hash activation file: {path}") from exc
    return digest.hexdigest()


def _h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ActivationStoreError(
            "Activation storage is unavailable; install requirements.txt."
        ) from exc
    return h5py


def activation_file_path(run_directory: Path) -> Path:
    return run_directory / ACTIVATION_FILE


def build_activation_context(
    config: TaskConfig,
    generation_context: GenerationContext,
    model_metadata: Mapping[str, object],
) -> ActivationContext:
    model_keys = (
        "identifier",
        "requested_revision",
        "resolved_revision",
        "model_class",
        "tokenizer_class",
        "dtype",
        "device",
        "torch_version",
        "transformers_version",
    )
    settings = {
        "activation_protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "backend": CAPTURE_BACKEND,
        "position_protocol": POSITION_PROTOCOL,
        "storage_dtype": STORAGE_DTYPE,
        "generation_context_fingerprint": generation_context.fingerprint,
        "model": {key: model_metadata.get(key) for key in model_keys},
    }
    fingerprint = hashlib.sha256(
        _encoded(
            {
                "schema_version": ACTIVATION_STORE_SCHEMA_VERSION,
                "settings": settings,
            }
        )
    ).hexdigest()
    return ActivationContext(fingerprint[:12], fingerprint, settings)


def build_activation_identity(
    context: ActivationContext, generation_sha256: str
) -> ActivationIdentity:
    fingerprint = hashlib.sha256(
        _encoded(
            {
                "schema_version": ACTIVATION_STORE_SCHEMA_VERSION,
                "context_fingerprint": context.fingerprint,
                "generation_sha256": generation_sha256,
            }
        )
    ).hexdigest()
    return ActivationIdentity(
        fingerprint[:12],
        fingerprint,
        context.context_id,
        context.fingerprint,
        generation_sha256,
    )


def _root_metadata(
    identity: ActivationIdentity, context: ActivationContext
) -> dict[str, object]:
    return {
        "schema_version": ACTIVATION_STORE_SCHEMA_VERSION,
        "protocol_version": ACTIVATION_PROTOCOL_VERSION,
        "capture_id": identity.capture_id,
        "capture_fingerprint": identity.fingerprint,
        "context_id": context.context_id,
        "context_fingerprint": context.fingerprint,
        "context_settings": json.dumps(
            context.settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "generation_sha256": identity.generation_sha256,
        "backend": CAPTURE_BACKEND,
        "position_protocol": POSITION_PROTOCOL,
        "storage_dtype": STORAGE_DTYPE,
    }


def _initialize(source, identity: ActivationIdentity, context: ActivationContext) -> None:
    expected = _root_metadata(identity, context)
    if not source.attrs:
        for name, value in expected.items():
            source.attrs[name] = value
        source.attrs["completed"] = False
        source.attrs["state_labels"] = ""
        source.attrs["state_count"] = 0
        source.attrs["hidden_size"] = 0
        source.create_group("examples")
        source.create_group("_pending")
        source.flush()
        return
    for name, value in expected.items():
        if source.attrs.get(name) != value:
            raise ActivationStoreError(
                f"Activation file has incompatible {name}: {source.filename}"
            )
    if "examples" not in source or "_pending" not in source:
        raise ActivationStoreError(
            f"Activation file has an invalid structure: {source.filename}"
        )


def _state_metadata(source) -> tuple[tuple[str, ...], int]:
    raw_labels = source.attrs.get("state_labels", "")
    hidden_size = int(source.attrs.get("hidden_size", 0))
    if not raw_labels:
        return (), hidden_size
    try:
        labels = json.loads(raw_labels)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ActivationStoreError(
            f"Activation state labels are invalid: {source.filename}"
        ) from exc
    if (
        not isinstance(labels, list)
        or not labels
        or int(source.attrs.get("state_count", 0)) != len(labels)
        or hidden_size <= 0
    ):
        raise ActivationStoreError(
            f"Activation state metadata is invalid: {source.filename}"
        )
    return tuple(labels), hidden_size


def _positions_json(plan: ReplayPlan) -> str:
    return json.dumps(plan.positions, sort_keys=True, separators=(",", ":"))


def _stored_metadata(source, group) -> dict[str, str]:
    labels, hidden_size = _state_metadata(source)
    return {
        "state_labels": json.dumps(labels, separators=(",", ":")),
        "state_count": str(len(labels)),
        "hidden_size": str(hidden_size),
        "max_logprob_difference": repr(float(group.attrs["max_logprob_difference"])),
    }


def _validate_example(
    source,
    key: str,
    expected: tuple[dict, str, ReplayPlan],
    *,
    require_split: bool = True,
) -> StoredActivation:
    example, generation_sha256, plan = expected
    group = source["examples"][key]
    if (
        int(group.attrs.get("id", -1)) != example["id"]
        or group.attrs.get("generation_sha256") != generation_sha256
        or group.attrs.get("replay_sha256") != plan.replay_sha256
        or group.attrs.get("positions") != _positions_json(plan)
        or (require_split and group.attrs.get("split") != example["split"])
    ):
        raise ActivationStoreError(
            f"Activation data is invalid for example ID {example['id']}."
        )
    labels, hidden_size = _state_metadata(source)
    if set(group.keys()) != set(plan.positions):
        raise ActivationStoreError(
            f"Activation tensors are incomplete for example ID {example['id']}."
        )
    for name, positions in plan.positions.items():
        dataset = group[name]
        if (
            dataset.dtype.name != "float16"
            or dataset.shape != (len(labels), len(positions), hidden_size)
        ):
            raise ActivationStoreError(
                f"Activation tensor '{name}' is invalid for example ID {example['id']}."
            )
    return StoredActivation(_stored_metadata(source, group))


def _open(path: Path, mode: str):
    h5py = _h5py()
    try:
        return h5py.File(path, mode)
    except (OSError, ValueError) as exc:
        raise ActivationStoreError(f"Cannot open activation file: {path}") from exc


def load_activation_records(
    run_directory: Path,
    identity: ActivationIdentity,
    context: ActivationContext,
    expected: ExpectedActivations,
) -> tuple[dict[int, StoredActivation], bool]:
    """Create or validate the run's activation file and recover pending writes."""
    path = activation_file_path(run_directory)
    if path.exists() and path.stat().st_size == 0:
        path.unlink()
    records = {}
    recovered = False
    try:
        with _open(path, "a") as source:
            _initialize(source, identity, context)
            pending = source["_pending"]
            if len(pending):
                if bool(source.attrs.get("completed", False)):
                    raise ActivationStoreError(
                        f"Completed activation file contains pending data: {path}"
                    )
                for key in list(pending.keys()):
                    del pending[key]
                source.flush()
                recovered = True
            for key in source["examples"]:
                try:
                    example_id = int(key)
                except ValueError as exc:
                    raise ActivationStoreError(
                        f"Activation file contains an invalid example key: {key}"
                    ) from exc
                item = expected.get(example_id)
                if item is None:
                    raise ActivationStoreError(
                        f"Activation file contains unexpected example ID {example_id}."
                    )
                records[example_id] = _validate_example(source, key, item)
    except ActivationStoreError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ActivationStoreError(f"Activation file is invalid: {path}") from exc
    return records, recovered


def _set_state_metadata(source, result: ActivationResult) -> None:
    first = next(iter(result.tensors.values()))
    labels, hidden_size = _state_metadata(source)
    if not labels:
        source.attrs["state_labels"] = json.dumps(
            result.state_labels, separators=(",", ":")
        )
        source.attrs["state_count"] = len(result.state_labels)
        source.attrs["hidden_size"] = int(first.shape[2])
    elif labels != result.state_labels or hidden_size != int(first.shape[2]):
        raise ActivationStoreError("Captured hidden-state dimensions are inconsistent.")


def append_activation_record(
    run_directory: Path,
    identity: ActivationIdentity,
    context: ActivationContext,
    example: dict,
    generation_sha256: str,
    plan: ReplayPlan,
    result: ActivationResult,
) -> StoredActivation:
    """Write one complete example and make it visible only after flushing."""
    path = activation_file_path(run_directory)
    key = str(example["id"])
    try:
        with _open(path, "a") as source:
            _initialize(source, identity, context)
            if bool(source.attrs.get("completed", False)):
                raise ActivationStoreError("Cannot append to a completed activation file.")
            if key in source["examples"]:
                raise ActivationStoreError(
                    f"Activation already exists for example ID {example['id']}."
                )
            _set_state_metadata(source, result)
            pending = source["_pending"]
            if key in pending:
                del pending[key]
            group = pending.create_group(key)
            group.attrs["id"] = example["id"]
            group.attrs["split"] = example["split"]
            group.attrs["generation_sha256"] = generation_sha256
            group.attrs["replay_sha256"] = plan.replay_sha256
            group.attrs["positions"] = _positions_json(plan)
            group.attrs["max_logprob_difference"] = result.max_logprob_difference
            if set(result.tensors) != set(plan.positions):
                raise ActivationStoreError("Captured activation positions are incomplete.")
            for name, positions in plan.positions.items():
                tensor = result.tensors[name]
                expected_shape = (
                    len(result.state_labels),
                    len(positions),
                    int(tensor.shape[2]),
                )
                if tuple(tensor.shape) != expected_shape or str(tensor.dtype) != "torch.float16":
                    raise ActivationStoreError(
                        f"Captured activation tensor '{name}' has an invalid shape or dtype."
                    )
                group.create_dataset(name, data=tensor.numpy())
            source.flush()
            source.move(f"_pending/{key}", f"examples/{key}")
            source.flush()
            return _validate_example(
                source,
                key,
                (example, generation_sha256, plan),
            )
    except ActivationStoreError:
        raise
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ActivationStoreError(
            f"Cannot store activation for example ID {example['id']}."
        ) from exc


def reuse_activation_records(
    output_directory: Path,
    run_directory: Path,
    identity: ActivationIdentity,
    context: ActivationContext,
    expected: ExpectedActivations,
    completed_ids: set[int],
) -> dict[int, StoredActivation]:
    """Copy unchanged examples from compatible completed run files."""
    missing = set(expected) - completed_ids
    if not missing or not output_directory.is_dir():
        return {}
    target_path = activation_file_path(run_directory)
    reused = {}
    for candidate in sorted(output_directory.iterdir()):
        if not missing or candidate == run_directory or not candidate.is_dir():
            continue
        run_record = candidate / "run.json"
        source_path = activation_file_path(candidate)
        if not run_record.is_file() or not source_path.is_file():
            continue
        try:
            record = json.loads(run_record.read_text(encoding="utf-8"))
            capture = record.get("activation_capture")
            if (
                "activation_capture" not in record.get("completed_stages", [])
                or not isinstance(capture, dict)
                or capture.get("context_fingerprint") != context.fingerprint
                or capture.get("artifact") != ACTIVATION_FILE
                or capture.get("artifact_sha256") != _sha256(source_path)
            ):
                continue
            with _open(source_path, "r") as source, _open(target_path, "a") as target:
                _initialize(target, identity, context)
                if not bool(source.attrs.get("completed", False)):
                    continue
                source_labels, source_hidden = _state_metadata(source)
                target_labels, target_hidden = _state_metadata(target)
                if target_labels and (
                    target_labels != source_labels or target_hidden != source_hidden
                ):
                    raise ActivationStoreError(
                        "Reusable activation files have incompatible hidden states."
                    )
                if not target_labels:
                    target.attrs["state_labels"] = source.attrs["state_labels"]
                    target.attrs["state_count"] = source.attrs["state_count"]
                    target.attrs["hidden_size"] = source.attrs["hidden_size"]
                for example_id in list(missing):
                    key = str(example_id)
                    if key not in source["examples"]:
                        continue
                    _validate_example(
                        source, key, expected[example_id], require_split=False
                    )
                    source.copy(source["examples"][key], target["_pending"], name=key)
                    target["_pending"][key].attrs["split"] = expected[example_id][0]["split"]
                    target.move(f"_pending/{key}", f"examples/{key}")
                    reused[example_id] = _validate_example(
                        target, key, expected[example_id]
                    )
                    missing.remove(example_id)
                target.flush()
        except (ActivationStoreError, OSError, UnicodeError, json.JSONDecodeError):
            continue
    return reused


def finalize_activation_file(
    run_directory: Path,
    identity: ActivationIdentity,
    context: ActivationContext,
    expected: ExpectedActivations,
    records: Mapping[int, StoredActivation],
) -> str:
    if set(records) != set(expected):
        raise ActivationStoreError(
            "Cannot finalize activations before every example is captured."
        )
    path = activation_file_path(run_directory)
    loaded, _ = load_activation_records(run_directory, identity, context, expected)
    if set(loaded) != set(expected):
        raise ActivationStoreError("Activation file does not contain every example.")
    try:
        with _open(path, "a") as source:
            source.attrs["completed"] = True
            source.flush()
    except OSError as exc:
        raise ActivationStoreError(f"Cannot finalize activation file: {path}") from exc
    return _sha256(path)
