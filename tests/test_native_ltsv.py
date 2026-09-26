from types import SimpleNamespace

import numpy as np
import torch

from baselines.ltsv.runner import _ltsv_reference_mse, _ltsv_training_loss


def test_native_ltsv_forecasts_only_history_and_delegates_native_loss():
    seen = []
    def forecast(contexts):
        seen.append(contexts.copy())
        return np.repeat(contexts[:, -1:], 2, axis=1)
    model = torch.nn.Linear(4, 1)
    engine = SimpleNamespace(config=SimpleNamespace(context_length=2, prediction_length=2),
        forecast=forecast, training_loss=lambda w: model(torch.from_numpy(w)).square().mean())
    windows = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _ltsv_reference_mse(engine, windows, batch_size=2) == 2.5
    np.testing.assert_array_equal(np.concatenate(seen), windows[:, :2])
    loss = _ltsv_training_loss(engine, windows)
    expected = engine.training_loss(windows)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.weight.grad is not None
