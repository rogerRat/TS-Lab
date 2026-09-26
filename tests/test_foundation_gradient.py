"""Projected native-loss gradient engine without external model weights."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from adapters.foundation.components import split_static_windows
from adapters.foundation.spec import get_model_spec
from gradient.foundation import FoundationGradientConfig, FoundationGradientEngine


class _FakeAdapter:
    def __init__(self) -> None:
        self.spec = get_model_spec("timesfm-2.5")
        self.model = nn.Linear(4, 2, bias=False)

    def load(self):
        return self.model

    def training_loss(self, windows, *, context_length, prediction_length):
        context, future = split_static_windows(
            windows, context_length=context_length, prediction_length=prediction_length,
        )
        prediction = self.model(torch.from_numpy(context))
        return nn.functional.mse_loss(prediction, torch.from_numpy(future))


def test_foundation_gradient_engine_is_deterministic_and_finite() -> None:
    torch.manual_seed(4)
    adapter = _FakeAdapter()
    config = FoundationGradientConfig(
        context_length=4,
        prediction_length=2,
        layer="weight",
        projection_dimension=7,
        projection_seed=11,
    )
    engine = FoundationGradientEngine(config, device="cpu", adapter=adapter)
    windows = np.arange(18, dtype=np.float32).reshape(3, 6) / 10
    losses, sketches = engine.batch_gradient_sketch(windows)
    assert losses.shape == (3,)
    assert sketches.shape == (3, 7)
    assert np.isfinite(losses).all() and np.isfinite(sketches).all()
    repeat_losses, repeat_sketches = engine.batch_gradient_sketch(windows)
    np.testing.assert_allclose(losses, repeat_losses)
    np.testing.assert_allclose(sketches, repeat_sketches)
