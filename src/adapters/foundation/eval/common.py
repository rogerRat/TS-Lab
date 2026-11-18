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
    elif model_key == "chronos-2-small":
        from dataclasses import replace
        adapter.spec = replace(adapter.spec, model_id=str(checkpoint))
        model = adapter.load()
    else:
        raise ValueError("unsupported foundation evaluation model: " + model_key)
    adapter._model = model.to(device)
    adapter._model.eval()
    return adapter


def predict_quantiles(adapter, contexts: np.ndarray, prediction_length: int,
                      levels: list[float]) -> np.ndarray:
    if hasattr(adapter, "forecast_quantiles"):
        native, native_levels = adapter.forecast_quantiles(
            contexts, prediction_length=prediction_length)
        native = np.asarray(native, dtype=np.float32)
        native_levels = np.asarray(native_levels, dtype=np.float32)
        if native.ndim != 3 or native.shape[1] != len(native_levels):
            raise ValueError("Chronos quantile output has invalid shape")
        if (len(native_levels) < 2 or np.any(np.diff(native_levels) <= 0)
                or not np.isfinite(native).all()):
            raise ValueError("Chronos quantile output is invalid")
        result = np.empty((native.shape[0], len(levels), native.shape[2]),
                          dtype=np.float32)
        for row in range(native.shape[0]):
            for step in range(native.shape[2]):
                result[row, :, step] = np.interp(
                    np.asarray(levels, dtype=np.float32), native_levels,
                    native[row, :, step])
        return result
    return tiled_quantiles(adapter.forecast(
        contexts, prediction_length=prediction_length), levels)


def tiled_quantiles(point_forecasts: np.ndarray, levels: list[float]) -> np.ndarray:
    point = np.asarray(point_forecasts, dtype=np.float32)
    if point.ndim != 2 or not np.isfinite(point).all():
        raise ValueError("foundation point forecast must be a finite rank-2 array")
    return np.repeat(point[:, None, :], len(levels), axis=1)
