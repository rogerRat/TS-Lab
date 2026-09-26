"""Tests for the model-agnostic foundation fine-tuning middle layer."""

from __future__ import annotations

import numpy as np
import pytest

from adapters.foundation import get_model_spec
from adapters.foundation.base import FoundationModelAdapter
from adapters.foundation.training import FineTuneConfig, FoundationFineTuner


class _TinyTrainableAdapter(FoundationModelAdapter):
    def __init__(self):
        super().__init__(get_model_spec("timesfm-2.5"), device="cpu")

    def load(self):
        import torch
        if self._model is None:
            self._model = torch.nn.Linear(3, 1)
        return self._model

    def forecast(self, contexts, *, prediction_length):
        raise NotImplementedError

    def training_loss(self, windows, *, context_length, prediction_length):
        import torch
        values = torch.as_tensor(windows, dtype=torch.float32)
        return torch.nn.functional.mse_loss(
            self.load()(values[:, :context_length]), values[:, context_length:],
        )


def test_fine_tuner_runs_requested_micro_and_optimizer_steps() -> None:
    adapter = _TinyTrainableAdapter()
    config = FineTuneConfig(
        context_length=3, prediction_length=1, steps=5, batch_size=2,
        gradient_accumulation=2,
    )
    result = FoundationFineTuner(adapter, config).fit(
        np.arange(32, dtype=np.float32).reshape(8, 4),
    )
    assert len(result.losses) == 5
    assert result.optimizer_steps == 3
    assert result.examples_seen == 10
    assert np.isfinite(result.losses).all()


def test_fine_tuner_rejects_inference_only_models_before_load() -> None:
    class InferenceOnly(_TinyTrainableAdapter):
        def __init__(self):
            FoundationModelAdapter.__init__(
                self, get_model_spec("timesfm-3.0"), device="cpu",
            )

    with pytest.raises(NotImplementedError, match="fine_tune"):
        FoundationFineTuner(
            InferenceOnly(), FineTuneConfig(context_length=3, prediction_length=1),
        )
