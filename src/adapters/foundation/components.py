"""Bottom-layer array validation and fixed-window operations."""

from __future__ import annotations

import numpy as np


def as_finite_batch(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("%s must have shape [batch, time]" % name)
    if not np.isfinite(result).all():
        raise ValueError("%s must contain only finite values" % name)
    return result


def split_static_windows(
    windows: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split materialized TS3 windows without resampling or normalization."""
    values = as_finite_batch(windows, name="windows")
    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length and prediction_length must be positive")
    expected = context_length + prediction_length
    if values.shape[1] != expected:
        raise ValueError("window length is %d, expected %d" % (values.shape[1], expected))
    return values[:, :context_length], values[:, context_length:]


def median_forecast(samples: np.ndarray, *, prediction_length: int) -> np.ndarray:
    """Normalize [B,S,H] samples or [B,H] point forecasts to [B,H]."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 3:
        values = np.median(values, axis=1)
    if values.ndim != 2 or values.shape[1] < prediction_length:
        raise ValueError("forecast output must provide [batch, horizon] values")
    return values[:, :prediction_length]


def shifted_windows(
    series: np.ndarray,
    target_starts: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
) -> np.ndarray:
    """Materialize fixed long-context windows at explicit prediction events.

    The caller chooses target events; this function never samples, pads, or
    changes their order. It is therefore suitable for both W2048 individual
    scoring and S2048 grouped mean-gradient scoring.
    """
    values = np.asarray(series, dtype=np.float32)
    starts = np.asarray(target_starts, dtype=np.int64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("series must be a non-empty finite vector")
    if starts.ndim != 1 or not len(starts):
        raise ValueError("target_starts must be a non-empty vector")
    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length and prediction_length must be positive")
    if np.any(starts < context_length) or np.any(starts + prediction_length > len(values)):
        raise ValueError("target start cannot satisfy the requested context/horizon")
    return np.stack([
        values[int(start) - context_length:int(start) + prediction_length]
        for start in starts
    ])


def grouped_mean_sketches(sketches: np.ndarray, group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average event sketches within stable, first-seen group order."""
    values = np.asarray(sketches, dtype=np.float32)
    groups = np.asarray(group_ids)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("sketches must be a non-empty finite matrix")
    if groups.ndim != 1 or len(groups) != len(values):
        raise ValueError("group_ids must align with sketch rows")
    order = list(dict.fromkeys(groups.tolist()))
    means = np.stack([values[groups == group].mean(axis=0) for group in order])
    return np.asarray(order), means.astype(np.float32, copy=False)


def cosine_to_reference(sketches: np.ndarray, reference_sketches: np.ndarray) -> np.ndarray:
    """Cosine score each row against the mean normalized reference direction."""
    source = as_finite_batch(sketches, name="sketches")
    reference = as_finite_batch(reference_sketches, name="reference_sketches")
    if source.shape[1] != reference.shape[1]:
        raise ValueError("source and reference sketch dimensions must match")
    source_norm = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-8)
    reference_norm = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), 1e-8,
    )
    return (source_norm @ reference_norm.mean(axis=0)).astype(np.float32)


__all__ = [
    "as_finite_batch", "cosine_to_reference", "grouped_mean_sketches", "median_forecast",
    "shifted_windows", "split_static_windows",
]
