"""Epoch/exposure accounting shared by all downstream experiment arms."""
from __future__ import annotations

import math


def exposure_checkpoints(n_windows: int, batch_size: int, epochs) -> list[dict]:
    """Requested half/integer epochs and actual full-batch exposure accounting.

    Different requested epochs can share one step on tiny development pools;
    keep both labels instead of silently dropping an observation.
    """
    if n_windows < 1 or batch_size < 1:
        raise ValueError("Pool N and batch size must be positive")
    result = []
    for epoch in epochs:
        if not math.isfinite(epoch) or epoch <= 0:
            raise ValueError("Epochs must be finite and positive")
        steps = math.ceil(epoch * n_windows / batch_size)
        draws = steps * batch_size
        result.append(dict(epoch=float(epoch), step=steps, draws=draws,
                           actual_epoch=draws / n_windows,
                           rounding_draws=draws - epoch * n_windows))
    return result


def epoch_checkpoints(n_windows: int, batch_size: int, epochs: int = 4) -> list[tuple[int, int]]:
    """(epoch, cumulative optimizer step), rounded up only at each boundary.

    Sampling is with replacement: an epoch is N sample exposures, not a claim
    that every selected window is visited once. A small N may share boundaries.
    """
    if min(n_windows, batch_size, epochs) < 1:
        raise ValueError("window count, batch size and epochs must be positive")
    return [(epoch, math.ceil(epoch * n_windows / batch_size))
            for epoch in range(1, epochs + 1)]
