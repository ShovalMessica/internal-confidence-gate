"""Content-addressed storage for captured activation tensors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from src.activation import (
    ACTIVATION_PROTOCOL_VERSION,
    ACTIVATION_SCHEMA_VERSION,
    CAPTURE_BACKEND,
    POSITION_PROTOCOL,
    STORAGE_DTYPE,
    ActivationResult,
    ReplayPlan,
)
from src.config import TaskConfig
from src.generation_cache import GenerationContext


ACTIVATION_CACHE_SCHEMA_VERSION = 1
ACTIVATION_MANIFEST_SCHEMA_VERSION = 1
_CACHE_ROOT = ".cache/activations"
_CONTEXT_RECORD = "context.json"


class ActivationCacheError(ValueError):
    """An activation cache or manifest is missing, corrupt, or conflicting."""


@dataclass(frozen=True)
class ActivationContext:
    context_id: str
    fingerprint: str
    settings: dict
    directory: Path


@dataclass(frozen=True)
class ActivationIdentity:
    capture_id: str
    fingerprint: str
    context_id: str
    generation_sha256: str


@dataclass(frozen=True)
class CachedActivation:
    example_key: str
    artifact_sha256: str
    path: Path
    metadata: dict[str, str]


def _encoded(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ActivationCacheError(f"Cannot hash activation artifact: {path}") from exc
    return digest.hexdigest()


def build_activation_context(
    config: TaskConfig,
    generation_context: GenerationContext,
    model_metadata: Mapping[str, object],
) -> ActivationContext:
    """Identify reusable capture work independently of a dataset run."""
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
                "schema_version": ACTIVATION_CACHE_SCHEMA_VERSION,
                "settings": settings,
            }
        )
    ).hexdigest()
    return ActivationContext(
        fingerprint[:12],
        fingerprint,
        settings,
        config.output_dir / _CACHE_ROOT / fingerprint[:12],
    )


def build_activation_identity(
    context: ActivationContext, generation_sha256: str
) -> ActivationIdentity:
    fingerprint = hashlib.sha256(
        _encoded(
            {
                "schema_version": ACTIVATION_MANIFEST_SCHEMA_VERSION,
                "context_fingerprint": context.fingerprint,
                "generation_sha256": generation_sha256,
            }
        )
    ).hexdigest()
    return ActivationIdentity(
        fingerprint[:12], fingerprint, context.context_id, generation_sha256
    )


def save_context(context: ActivationContext) -> None:
    path = context.directory / _CONTEXT_RECORD
    payload = {
        "schema_version": ACTIVATION_CACHE_SCHEMA_VERSION,
        "context_id": context.context_id,
        "fingerprint": context.fingerprint,
        "settings": context.settings,
    }
    content = _encoded(payload)
    temporary = path.with_suffix(".json.tmp")
    try:
        context.directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise ActivationCacheError(
                    f"Activation cache context conflicts with: {path}"
                )
            return
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ActivationCacheError(f"Cannot save activation context: {path}") from exc


def example_key(
    context: ActivationContext, generation_sha256: str, plan: ReplayPlan
) -> str:
    return hashlib.sha256(
        _encoded(
            {
                "context_fingerprint": context.fingerprint,
                "generation_sha256": generation_sha256,
                "replay_sha256": plan.replay_sha256,
                "positions": plan.positions,
            }
        )
    ).hexdigest()


def _artifact_directory(context: ActivationContext, key: str) -> Path:
    return context.directory / "records" / key


def _metadata(
    context: ActivationContext,
    example_id: int,
    generation_sha256: str,
    plan: ReplayPlan,
    result: ActivationResult,
) -> dict[str, str]:
    first_tensor = next(iter(result.tensors.values()))
    return {
        "schema_version": str(ACTIVATION_SCHEMA_VERSION),
        "protocol_version": str(ACTIVATION_PROTOCOL_VERSION),
        "context_id": context.context_id,
        "example_id": str(example_id),
        "generation_sha256": generation_sha256,
        "replay_sha256": plan.replay_sha256,
        "positions": json.dumps(plan.positions, sort_keys=True, separators=(",", ":")),
        "state_labels": json.dumps(result.state_labels, separators=(",", ":")),
        "state_count": str(len(result.state_labels)),
        "hidden_size": str(first_tensor.shape[2]),
        "storage_dtype": STORAGE_DTYPE,
        "max_logprob_difference": repr(result.max_logprob_difference),
    }


def store_activation(
    context: ActivationContext,
    example_id: int,
    generation_sha256: str,
    plan: ReplayPlan,
    result: ActivationResult,
) -> CachedActivation:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ActivationCacheError(
            "Activation storage is unavailable; install requirements.txt."
        ) from exc

    save_context(context)
    key = example_key(context, generation_sha256, plan)
    directory = _artifact_directory(context, key)
    temporary = directory / "artifact.safetensors.tmp"
    metadata = _metadata(context, example_id, generation_sha256, plan, result)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        save_file(result.tensors, str(temporary), metadata=metadata)
        artifact_sha256 = _sha256(temporary)
        path = directory / f"{artifact_sha256}.safetensors"
        if path.exists():
            if not path.is_file() or path.read_bytes() != temporary.read_bytes():
                raise ActivationCacheError(f"Cached activation conflicts with: {path}")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    except (OSError, RuntimeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, ActivationCacheError):
            raise
        raise ActivationCacheError(
            f"Cannot store activation for example ID {example_id}."
        ) from exc
    return CachedActivation(key, artifact_sha256, path, metadata)


def _inspect(path: Path) -> tuple[dict[str, str], dict[str, tuple[list[int], str]]]:
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            tensors = {
                name: (
                    source.get_slice(name).get_shape(),
                    source.get_slice(name).get_dtype(),
                )
                for name in source.keys()
            }
    except Exception as exc:
        raise ActivationCacheError(f"Cannot read cached activation: {path}") from exc
    return metadata, tensors


def load_activation(
    context: ActivationContext,
    example_id: int,
    generation_sha256: str,
    plan: ReplayPlan,
    artifact_sha256: str | None = None,
) -> CachedActivation | None:
    key = example_key(context, generation_sha256, plan)
    directory = _artifact_directory(context, key)
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise ActivationCacheError(
            f"Activation cache entry is not a directory: {directory}"
        )
    candidates = sorted(directory.glob("*.safetensors"))
    if artifact_sha256 is None:
        if not candidates:
            return None
        path = candidates[0]
        artifact_sha256 = path.stem
    else:
        path = directory / f"{artifact_sha256}.safetensors"
    if not path.is_file() or _sha256(path) != artifact_sha256:
        raise ActivationCacheError(f"Cached activation is missing or changed: {path}")

    metadata, tensors = _inspect(path)
    expected_metadata = {
        "schema_version": str(ACTIVATION_SCHEMA_VERSION),
        "protocol_version": str(ACTIVATION_PROTOCOL_VERSION),
        "context_id": context.context_id,
        "example_id": str(example_id),
        "generation_sha256": generation_sha256,
        "replay_sha256": plan.replay_sha256,
        "positions": json.dumps(plan.positions, sort_keys=True, separators=(",", ":")),
        "storage_dtype": STORAGE_DTYPE,
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ActivationCacheError(f"Cached activation metadata is invalid: {path}")
    try:
        labels = json.loads(metadata["state_labels"])
        state_count = int(metadata["state_count"])
        hidden_size_value = int(metadata["hidden_size"])
        float(metadata["max_logprob_difference"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ActivationCacheError(
            f"Cached activation metadata is incomplete: {path}"
        ) from exc
    if (
        not isinstance(labels, list)
        or not labels
        or state_count != len(labels)
        or hidden_size_value <= 0
    ):
        raise ActivationCacheError(f"Cached activation state labels are invalid: {path}")
    if set(tensors) != set(plan.positions):
        raise ActivationCacheError(f"Cached activation tensors are incomplete: {path}")
    hidden_size = None
    for name, positions in plan.positions.items():
        shape, dtype = tensors[name]
        if (
            len(shape) != 3
            or shape[0] != len(labels)
            or shape[1] != len(positions)
            or dtype != "F16"
        ):
            raise ActivationCacheError(
                f"Cached activation tensor '{name}' is invalid: {path}"
            )
        hidden_size = shape[2] if hidden_size is None else hidden_size
        if shape[2] != hidden_size:
            raise ActivationCacheError(f"Cached activation hidden sizes differ: {path}")
    if hidden_size != hidden_size_value:
        raise ActivationCacheError(f"Cached activation hidden size is invalid: {path}")
    return CachedActivation(key, artifact_sha256, path, metadata)
