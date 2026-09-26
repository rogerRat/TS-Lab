"""MASE and SQL forecast metrics for Chronos-style quantile models.

The model outputs a categorical distribution over value bins per horizon
step. Quantile forecasts are read off the bin CDF (piecewise-constant
inverse); both metrics are scaled by the seasonal-naive inaccuracy on the
same window, following the FEV benchmark convention:

    s = (1/H) * sum_h |y_h - y_{h-m}|                     (seasonality m)
    MASE = (1/H) * sum_h |y_h - q_0.5,h| / s
    SQL  = (2 / (|T| H)) * sum_h sum_t rho_t(y_h - q_t,h) / s

with rho_t(u) = (t - 1{u<0}) u and quantile levels T = 0.1 .. 0.9.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Seasonality per reference dataset, strictly the FEV-native table (domain E1):
# boomlet_1282/619 are 1-minute series (daily period 1440), boomlet_1676 is
# 30-minute (48), ETT_15T 96, ETT_1H 24, jena_weather_1H 24.
DATASET_SEASONALITY: dict[str, int] = {
    "ETT_15T": 96,
    "ETT_1H": 24,
    "bizitobs_l2c_5T": 288,
    "jena_weather_1H": 24,
    "boomlet_1282": 1440,
    "boomlet_619": 1440,
    "boomlet_1676": 48,
}
DEFAULT_SEASONALITY = 24


def seasonality_for(entity_id: str) -> int:
    for name, value in DATASET_SEASONALITY.items():
        if name in entity_id:
            return value
    return DEFAULT_SEASONALITY


@torch.no_grad()
def quantile_forecasts(engine, windows: np.ndarray, *, batch_size: int = 128):
    """Quantile forecasts and tokenizer scales for a stack of windows.

    Returns:
        quantiles: float32 [N, H, len(QUANTILE_LEVELS)] in original units
        scales:    float32 [N] per-window tokenizer scales

    Each horizon CDF is conditional on the preceding greedily generated tokens;
    these are recursive conditional quantiles, not sampled marginal quantiles.
    True future labels never enter generation.
    """
    tokenizer = engine.tokenizer
    levels = torch.tensor(QUANTILE_LEVELS, dtype=torch.float32, device=engine.device)

    windows = np.asarray(windows, dtype=np.float32)
    context_length = engine.config.context_length
    quantile_chunks = []
    scale_chunks = []
    for begin in range(0, len(windows), batch_size):
        chunk = torch.from_numpy(windows[begin : begin + batch_size])
        contexts = chunk[:, :context_length]
        input_ids, attention_mask, scale = tokenizer.context_input_transform(contexts)
        # Forecast from history only. Passing true labels here teacher-forces
        # previous future values into the decoder and is not a forecast exam.
        from rags.generation import generate_tokens
        _tokens, scores = generate_tokens(engine, input_ids.to(engine.device),
                                          attention_mask.to(engine.device), output_scores=True)
        probabilities = torch.softmax(scores, dim=-1)
        cdf = torch.cumsum(probabilities, dim=-1).contiguous()  # [B, H, V]
        batch, horizon = cdf.shape[0], cdf.shape[1]
        # piecewise-constant inverse CDF: first bin whose cumulative mass
        # reaches the requested level
        indices = torch.searchsorted(
            cdf,  # [B, H, V]
            levels.view(1, 1, -1).expand(batch, horizon, -1).contiguous(),  # [B, H, T]
        ).clamp_max(probabilities.shape[-1] - 1)  # [B, H, T]
        # Use the official tokenizer's token-to-bin offset and inverse scaling.
        quantiles = tokenizer.output_transform(indices.cpu(), scale)  # [B, H, T]
        quantile_chunks.append(quantiles.cpu())  # [B, H, T]
        scale_chunks.append(scale.reshape(-1).cpu())
    return torch.cat(quantile_chunks).numpy(), torch.cat(scale_chunks).numpy()


def seasonal_scale(windows: np.ndarray, context_length: int, seasonality: int) -> np.ndarray:
    """Per-window seasonal-naive inaccuracy on the forecast horizon."""
    windows = np.asarray(windows, dtype=np.float64)
    futures = windows[:, context_length:]
    naive_source_start = context_length - seasonality
    if naive_source_start < 0:
        raise ValueError("seasonality %d exceeds context length %d" % (seasonality, context_length))
    naive = windows[:, naive_source_start : naive_source_start + futures.shape[1]]
    return np.abs(futures - naive).mean(axis=1)


def history_seasonal_scale(history: np.ndarray, seasonality: int) -> float:
    """Seasonal-naive inaccuracy over a series' FULL history (domain E1).

    s = mean_t |y_t - y_{t-m}| for t = m..L-1.  1440-period series exceed the
    512-point window, so the scale must come from the whole series, not the
    evaluation window.  Returns 0.0 for constant series (reported separately).
    """
    values = np.asarray(history, dtype=np.float64)
    if len(values) <= seasonality:
        return 0.0
    return float(np.abs(values[seasonality:] - values[:-seasonality]).mean())


def point_metrics(windows, predictions, entities, *, context_length, scales=None):
    """MASE only across students; constant-series MAE is a separate diagnostic.

    First average windows within each series, then series equally. A zero
    seasonal scale has no defined MASE and must not become an epsilon-scaled
    artificial score. Formal tasks pass full safe-history scales explicitly.
    """
    values = np.asarray(windows, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    futures = values[:, context_length:]
    if predictions.shape != futures.shape or len(entities) != len(values):
        raise ValueError("forecast shape / entity count does not match exam")
    if not np.isfinite(predictions).all():
        raise ValueError("non-finite point forecast")
    if scales is None:
        scales = np.asarray([seasonal_scale(values[i:i+1], context_length, seasonality_for(e))[0]
                             for i, e in enumerate(entities)])
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (len(values),) or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("invalid full-history seasonal scales")
    errors = np.abs(futures - predictions).mean(axis=1)
    by_series, constants = {}, {}
    for entity in dict.fromkeys(entities):
        rows = np.asarray([i for i, e in enumerate(entities) if e == entity])
        valid = scales[rows] > 0
        if valid.any():
            by_series[entity] = float(np.mean(errors[rows[valid]] / scales[rows[valid]]))
        if (~valid).any():
            constants[entity] = float(errors[rows[~valid]].mean())
    return {"mase": float(np.mean(list(by_series.values()))) if by_series else None,
            "num_series": len(set(entities)), "num_windows": len(values),
            "per_series": {"mase": by_series}, "constant_series_mae": constants}


def mase_sql(
    windows: np.ndarray,
    quantiles: np.ndarray,
    *,
    context_length: int,
    seasonality: int,
    scales: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> dict:
    """MASE (median forecast) and SQL (mean over the 9 FEV quantile levels).

    ``scales`` gives each window's seasonal-naive denominator (full-history
    scale per series, domain E1); when omitted the per-window context-based
    scale is used (legacy behaviour).
    """
    windows = np.asarray(windows, dtype=np.float64)
    futures = windows[:, context_length :]
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if scales is None:
        scale = seasonal_scale(windows, context_length, seasonality)
    else:
        scale = np.asarray(scales, dtype=np.float64)
    safe_scale = np.maximum(scale, epsilon)

    median = quantiles[:, :, QUANTILE_LEVELS.index(0.5)]
    mase = np.abs(futures - median).mean(axis=1) / safe_scale

    errors = futures[:, :, None] - quantiles  # [N, H, T]
    levels = np.asarray(QUANTILE_LEVELS)[None, None, :]
    pinball = (levels - (errors < 0).astype(np.float64)) * errors
    sql = 2.0 * pinball.mean(axis=(1, 2)) / safe_scale
    return {
        "mase": float(mase.mean()),
        "sql": float(sql.mean()),
    }


__all__ = [
    "QUANTILE_LEVELS",
    "DATASET_SEASONALITY",
    "seasonality_for",
    "quantile_forecasts",
    "seasonal_scale",
    "history_seasonal_scale",
    "mase_sql",
]
