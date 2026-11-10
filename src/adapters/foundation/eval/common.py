"""Shared model loading and finite-context handling for foundation evaluations."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def clean_context(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("evaluation context must be non-empty")
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    return np.interp(np.arange(values.size), np.flatnonzero(finite), values[finite]).astype(np.float32)


def build_checkpoint_adapter(model_key: str, checkpoint: Path, device: str):
    """Load a merged local model checkpoint behind its project adapter."""
    import torch
    from adapters.foundation import build_adapter
    adapter = build_adapter(model_key, device=device)
    if model_key == "timerxl-84m":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
    elif model_key == "timesfm-2.5":
        from transformers import TimesFm2_5ModelForPrediction
        model = TimesFm2_5ModelForPrediction.from_pretrained(
            str(checkpoint), local_files_only=True, torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError("unsupported foundation evaluation model: " + model_key)
    adapter._model = model.to(device)
    adapter._model.eval()
    return adapter


def tiled_quantiles(point_forecasts: np.ndarray, levels: list[float]) -> np.ndarray:
    point = np.asarray(point_forecasts, dtype=np.float32)
    if point.ndim != 2 or not np.isfinite(point).all():
        raise ValueError("foundation point forecast must be a finite rank-2 array")
    return np.repeat(point[:, None, :], len(levels), axis=1)
