"""Content-addressed, cross-run storage for model generations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from src.config import TaskConfig
from src.generation import GENERATION_PROTOCOL_VERSION


CACHE_SCHEMA_VERSION = 1
_CACHE_ROOT = ".cache/generations"
_CONTEXT_RECORD = "context.json"


class GenerationCacheError(ValueError):
    """The shared generation cache is missing, corrupt, or conflicting."""


@dataclass(frozen=True)
class GenerationContext:
    context_id: str
    fingerprint: str
    settings: dict
    directory: Path


@dataclass(frozen=True)
class CachedGeneration:
    record: dict
    example_key: str
    artifact_sha256: str


def _encoded(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def build_generation_context(config: TaskConfig) -> GenerationContext:
    """Identify model work independently of any particular dataset version."""
    settings = {
        "model_name_or_path": config.model_name_or_path,
        "model_revision": config.model_revision,
        "device": config.device,
        "dtype": config.dtype,
        "reasoning_mode": config.reasoning_mode,
        "answer_max_new_tokens": config.answer_max_new_tokens,
        "allow_abstention": config.allow_abstention,
        "generation_seed": config.generation_seed,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
    }
    if config.reasoning_mode == "reasoning":
        settings["reasoning_max_new_tokens"] = config.reasoning_max_new_tokens
    else:
        settings["direct_batch_size"] = config.direct_batch_size
    fingerprint = hashlib.sha256(
        _encoded({"schema_version": CACHE_SCHEMA_VERSION, "settings": settings})
    ).hexdigest()
    directory = config.output_dir / _CACHE_ROOT / fingerprint[:12]
    return GenerationContext(fingerprint[:12], fingerprint, settings, directory)


def _context_payload(context: GenerationContext, model: dict) -> dict:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "context_id": context.context_id,
        "fingerprint": context.fingerprint,
        "settings": context.settings,
        "model": model,
    }


def load_context_model(context: GenerationContext) -> dict | None:
    """Return the model pin and provenance already attached to this cache."""
    path = context.directory / _CONTEXT_RECORD
    if not path.exists():
        return None
    if not path.is_file():
        raise GenerationCacheError(f"Generation cache context is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationCacheError(f"Cannot read generation cache context: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("context_id") != context.context_id
        or payload.get("fingerprint") != context.fingerprint
        or payload.get("settings") != context.settings
        or not isinstance(payload.get("model"), dict)
    ):
        raise GenerationCacheError(f"Generation cache context is invalid: {path}")
    return payload["model"]


def save_context_model(context: GenerationContext, model: dict) -> None:
    """Pin the resolved model metadata for all runs sharing this context."""
    existing = load_context_model(context)
    if existing is not None:
        if existing != model:
            raise GenerationCacheError(
                "Loaded model metadata conflicts with the shared generation cache."
            )
        return
    path = context.directory / _CONTEXT_RECORD
    temporary = path.with_suffix(".json.tmp")
    try:
        context.directory.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(_encoded(_context_payload(context, model)))
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise GenerationCacheError(f"Cannot write generation cache context: {path}") from exc


def example_key(context: GenerationContext, example: dict) -> str:
    payload = {
        "context_fingerprint": context.fingerprint,
        "id": example["id"],
        "input": example["input"],
    }
    return hashlib.sha256(_encoded(payload)).hexdigest()


def _artifact_directory(context: GenerationContext, key: str) -> Path:
    return context.directory / "records" / key


def _cache_record(record: dict) -> dict:
    cached = dict(record)
    cached.pop("split", None)
    return cached


def store_generation(
    context: GenerationContext, example: dict, record: dict
) -> CachedGeneration:
    """Store an immutable generation and return its manifest reference."""
    if record.get("id") != example["id"] or record.get("split") != example["split"]:
        raise GenerationCacheError("Generation record does not match its example.")
    key = example_key(context, example)
    cached = _cache_record(record)
    content = _encoded(cached)
    artifact_sha256 = hashlib.sha256(content).hexdigest()
    directory = _artifact_directory(context, key)
    path = directory / f"{artifact_sha256}.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise GenerationCacheError(f"Cached generation conflicts with: {path}")
        else:
            temporary.write_bytes(content)
            os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise GenerationCacheError(f"Cannot store cached generation: {path}") from exc
    restored = {**cached, "split": example["split"]}
    return CachedGeneration(restored, key, artifact_sha256)


def load_generation(
    context: GenerationContext,
    example: dict,
    artifact_sha256: str | None = None,
) -> CachedGeneration | None:
    """Load a specific cached result, or the canonical result for an example."""
    key = example_key(context, example)
    directory = _artifact_directory(context, key)
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise GenerationCacheError(f"Generation cache entry is not a directory: {directory}")
    if artifact_sha256 is None:
        candidates = sorted(directory.glob("*.json"))
        if not candidates:
            return None
        path = candidates[0]
        artifact_sha256 = path.stem
    else:
        path = directory / f"{artifact_sha256}.json"
    if not path.is_file():
        raise GenerationCacheError(f"Cached generation is missing: {path}")
    try:
        content = path.read_bytes()
        cached = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationCacheError(f"Cannot read cached generation: {path}") from exc
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != artifact_sha256 or not isinstance(cached, dict):
        raise GenerationCacheError(f"Cached generation is invalid: {path}")
    if cached.get("id") != example["id"] or "split" in cached:
        raise GenerationCacheError(f"Cached generation has invalid example metadata: {path}")
    restored = {**cached, "split": example["split"]}
    return CachedGeneration(restored, key, artifact_sha256)
