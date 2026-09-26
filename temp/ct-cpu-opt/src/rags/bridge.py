"""Bridge between the ChronosTiny gradient engine and the RAGS pipeline.

The RAGS core is numpy-only and receives gradients through a callable
``gradient_computer(window, context_length, prediction_length) -> np.ndarray``.
This module adapts ``gradient.ChronosTinyGradientEngine`` (torch) to that
contract, optionally batching candidate windows for throughput.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


class ChronosGradientComputer:
    """Per-window projected gradients from a ChronosTinyGradientEngine."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.window_calls = 0

    @property
    def dimension(self) -> int:
        return int(self.engine.config.projection_dimension)

    def __call__(self, window: np.ndarray, context_length: int, prediction_length: int) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32).reshape(1, -1)
        _losses, sketches = self.engine.batch_gradient_sketch(window, batch_size=1)
        self.window_calls += 1
        return np.asarray(sketches, dtype=np.float32).reshape(-1)

    def compute_batch(self, windows: np.ndarray) -> np.ndarray:
        """Projected gradients for a stack of windows [N, window_length]."""
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 2:
            raise ValueError("compute_batch expects a 2-D window stack")
        _losses, sketches = self.engine.batch_gradient_sketch(windows)
        self.window_calls += len(windows)
        return np.asarray(sketches, dtype=np.float32)


def batched_candidate_computer(computer: ChronosGradientComputer, batch_size: int = 8) -> Callable:
    """Wrap a ChronosGradientComputer into RAGS's candidate-computer contract.

    RAGS calls the candidate computer one window at a time; this wrapper
    keeps that contract while accumulating windows into engine-sized batches
    internally is unnecessary because the engine already chunks internally.
    Kept as an explicit alias point for future batching experiments.
    """

    def candidate_computer(window: np.ndarray, context_length: int, prediction_length: int) -> np.ndarray:
        return computer(window, context_length, prediction_length)

    return candidate_computer


__all__ = ["ChronosGradientComputer", "batched_candidate_computer"]
