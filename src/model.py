"""Load supported Hugging Face models without running generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import TaskConfig


_CHAT_TEMPLATE_ERROR = (
    "The tokenizer has no usable chat template; v1 requires an instruction/chat model."
)


class ModelLoadError(RuntimeError):
    """The configured model cannot be loaded by the toolkit."""


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    resolved_device: str | dict[str, str]
    resolved_dtype: str
    resolved_revision: str | None


def _dependencies():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModelLoadError(
            "Model dependencies are unavailable; install requirements.txt."
        ) from exc
    return torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _validate_device(torch: Any, device: str) -> None:
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise ModelLoadError(f"Requested device '{device}', but CUDA is unavailable.")
    if ":" in device:
        index = int(device.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise ModelLoadError(
                f"Requested device '{device}', but only {torch.cuda.device_count()} "
                "CUDA device(s) are available."
            )


def _validate_chat_template(tokenizer: Any) -> None:
    try:
        template = tokenizer.get_chat_template()
    except (AttributeError, ValueError) as exc:
        raise ModelLoadError(_CHAT_TEMPLATE_ERROR) from exc
    if not isinstance(template, str) or not template.strip():
        raise ModelLoadError(_CHAT_TEMPLATE_ERROR)


def _resolved_device(model: Any, requested: str) -> str | dict[str, str]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        return {str(name): str(device) for name, device in device_map.items()}
    model_device = getattr(model, "device", None)
    return str(model_device) if model_device is not None else requested


def _resolved_dtype(model: Any, requested: str) -> str:
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        return requested
    return str(model_dtype).removeprefix("torch.")


def _resolve_model(
    config: TaskConfig, pinned_revision: str | None, auto_config: Any
) -> tuple[Any, str | None, dict[str, object]]:
    source = config.model_name_or_path
    requested_revision = pinned_revision or config.model_revision
    revision_kwargs = {"revision": requested_revision} if requested_revision else {}
    common = {"trust_remote_code": False, **revision_kwargs}

    try:
        model_config = auto_config.from_pretrained(source, **common)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load model configuration for '{source}': {exc}"
        ) from exc

    if getattr(model_config, "is_encoder_decoder", False):
        raise ModelLoadError(
            "The model is encoder-decoder; v1 requires a decoder-only model."
        )
    if getattr(model_config, "vision_config", None) is not None:
        raise ModelLoadError("The model is multimodal; v1 supports text-only models.")

    is_local = Path(source).is_absolute()
    resolved_revision = None if is_local else getattr(model_config, "_commit_hash", None)
    if not is_local and (
        not isinstance(resolved_revision, str) or not resolved_revision.strip()
    ):
        raise ModelLoadError(
            f"Could not resolve an exact Hub revision for '{source}'."
        )
    if pinned_revision is not None and resolved_revision != pinned_revision:
        raise ModelLoadError(
            f"Loaded revision '{resolved_revision}' does not match the pinned "
            f"revision '{pinned_revision}'."
        )
    load_common: dict[str, object] = {"trust_remote_code": False}
    if resolved_revision:
        load_common["revision"] = resolved_revision
    return model_config, resolved_revision, load_common


def _prepare_tokenizer(tokenizer: Any) -> Any:
    _validate_chat_template(tokenizer)
    if tokenizer.eos_token_id is None:
        raise ModelLoadError("The tokenizer has no EOS token; generation is unsupported.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _load_tokenizer(source: str, load_common: dict[str, object], factory: Any) -> Any:
    try:
        return _prepare_tokenizer(factory.from_pretrained(source, **load_common))
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load tokenizer for '{source}': {exc}") from exc


def load_tokenizer(
    config: TaskConfig, *, pinned_revision: str | None = None
) -> Any:
    """Load only the exact tokenizer needed for prompt-to-token mapping."""
    _, auto_config, _, auto_tokenizer = _dependencies()
    _, _, load_common = _resolve_model(config, pinned_revision, auto_config)
    return _load_tokenizer(config.model_name_or_path, load_common, auto_tokenizer)


def load_model(
    config: TaskConfig,
    *,
    pinned_revision: str | None = None,
    tokenizer: Any | None = None,
) -> LoadedModel:
    """Load and validate a supported model and its tokenizer.

    This function performs no generation and does not update run records.
    """
    torch, auto_config, auto_model, auto_tokenizer = _dependencies()
    _validate_device(torch, config.device)
    model_config, resolved_revision, load_common = _resolve_model(
        config, pinned_revision, auto_config
    )

    source = config.model_name_or_path
    tokenizer = (
        _load_tokenizer(source, load_common, auto_tokenizer)
        if tokenizer is None
        else _prepare_tokenizer(tokenizer)
    )

    dtype = "auto" if config.dtype == "auto" else getattr(torch, config.dtype)
    try:
        model = auto_model.from_pretrained(
            source,
            config=model_config,
            torch_dtype=dtype,
            device_map=config.device,
            **load_common,
        )
    except Exception as exc:
        raise ModelLoadError(f"Could not load model weights for '{source}': {exc}") from exc

    can_generate = getattr(model, "can_generate", None)
    if not callable(can_generate) or not can_generate():
        raise ModelLoadError("The loaded model does not support text generation.")
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        resolved_device=_resolved_device(model, config.device),
        resolved_dtype=_resolved_dtype(model, config.dtype),
        resolved_revision=resolved_revision,
    )


def describe_model(loaded: LoadedModel) -> dict:
    """Return JSON-compatible model and runtime provenance."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ModelLoadError(
            "Model dependencies are unavailable; install requirements.txt."
        ) from exc

    generation_config = getattr(loaded.model, "generation_config", None)
    serialized_generation = (
        generation_config.to_dict()
        if generation_config is not None and hasattr(generation_config, "to_dict")
        else {}
    )
    return {
        "resolved_revision": loaded.resolved_revision,
        "model_class": type(loaded.model).__name__,
        "tokenizer_class": type(loaded.tokenizer).__name__,
        "dtype": loaded.resolved_dtype,
        "device": loaded.resolved_device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "generation_config": serialized_generation,
    }
