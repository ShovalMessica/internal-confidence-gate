"""Load supported Hugging Face models without running generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import TaskConfig


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
                f"Requested device '{device}', but only {torch.cuda.device_count()} CUDA device(s) are available."
            )


def _chat_template(tokenizer: Any) -> str:
    try:
        template = tokenizer.get_chat_template()
    except (AttributeError, ValueError) as exc:
        raise ModelLoadError(
            "The tokenizer has no usable chat template; v1 requires an instruction/chat model."
        ) from exc
    if not isinstance(template, str) or not template.strip():
        raise ModelLoadError(
            "The tokenizer has no usable chat template; v1 requires an instruction/chat model."
        )
    return template


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


def load_model(config: TaskConfig) -> LoadedModel:
    """Load and validate a supported model and its tokenizer.

    This function performs no generation and does not update run records.
    """
    torch, auto_config, auto_model, auto_tokenizer = _dependencies()
    _validate_device(torch, config.device)

    source = config.model_name_or_path
    requested_revision = config.model_revision
    revision_kwargs = {"revision": requested_revision} if requested_revision else {}
    common = {"trust_remote_code": False, **revision_kwargs}

    try:
        model_config = auto_config.from_pretrained(source, **common)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load model configuration for '{source}': {exc}"
        ) from exc

    if getattr(model_config, "is_encoder_decoder", False):
        raise ModelLoadError("The model is encoder-decoder; v1 requires a decoder-only model.")
    if getattr(model_config, "vision_config", None) is not None:
        raise ModelLoadError("The model is multimodal; v1 supports text-only models.")

    is_local = Path(source).is_absolute()
    resolved_revision = None if is_local else getattr(model_config, "_commit_hash", None)
    load_revision = resolved_revision or requested_revision
    load_common = {"trust_remote_code": False}
    if load_revision:
        load_common["revision"] = load_revision

    try:
        tokenizer = auto_tokenizer.from_pretrained(source, **load_common)
        _chat_template(tokenizer)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load tokenizer for '{source}': {exc}") from exc

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
        resolved_revision=resolved_revision or requested_revision,
    )
