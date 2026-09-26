from types import SimpleNamespace
import numpy as np
import torch
import pytest
from gradient.base import NativeProbe


class BatchedProbe(NativeProbe):
    def per_window_losses(self, windows):
        values = torch.as_tensor(windows, dtype=self.model.weight.dtype)
        return self.model(values).square().mean(dim=1)


def make_probe():
    torch.manual_seed(13)
    model = torch.nn.Linear(3, 2)
    model.unused = torch.nn.Parameter(torch.ones(2))
    engine = SimpleNamespace(model=model, device="cpu",
        training_loss=lambda windows: model(torch.as_tensor(windows)).square().mean())
    return BatchedProbe(engine, model_key="test", projection_dimension=32)


def test_execution_policy_preserves_existing_positional_arguments():
    from rags.execution import SelectionExecution
    policy = SelectionExecution(16, 64, .7, 4, "eager")
    assert policy.inference_batch == 64
    assert policy.gradient_workers == 2
    assert policy.generation == "eager"


def test_batched_vjp_is_each_windows_gradient_not_batch_mean():
    probe = make_probe()
    windows = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
    expected_losses, expected = probe.batch_gradient_sketch(windows, batch_size=1)
    losses, actual = probe.batch_gradient_sketch(windows, batch_size=4)
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert not np.allclose(actual[0], actual[-1])
    assert all(parameter.grad is None for parameter in probe.model.parameters())


def test_batched_vjp_reduces_batch_after_oom_without_losing_rows(monkeypatch):
    probe = make_probe()
    windows = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
    expected_losses, expected = probe.batch_gradient_sketch(windows, batch_size=1)
    original = probe._batched_gradient_sketch
    def limited(values):
        if len(values) > 2:
            raise torch.cuda.OutOfMemoryError("fixture allocation limit")
        return original(values)
    monkeypatch.setattr(probe, "_batched_gradient_sketch", limited)
    losses, actual = probe.batch_gradient_sketch(windows, batch_size=4)
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert probe.last_gradient_execution["oom_retries"] == 1
    assert probe.last_gradient_execution["batch_size"] == 2


def test_family_batch_limit_and_safe_tail_shapes(monkeypatch):
    probe = make_probe()
    probe.max_gradient_batch = 4
    probe.power_of_two_batches = True
    windows = np.arange(21, dtype=np.float32).reshape(7, 3) / 10
    expected_losses, expected = probe.batch_gradient_sketch(windows, batch_size=1)
    original = probe._batched_gradient_sketch
    sizes = []
    def observe(values):
        sizes.append(len(values))
        return original(values)
    monkeypatch.setattr(probe, "_batched_gradient_sketch", observe)
    losses, actual = probe.batch_gradient_sketch(windows, batch_size=32)
    assert sizes == [4, 2]
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="positive"):
        probe.batch_gradient_sketch(windows, batch_size=-1)


def test_bf16_chronos_uses_independent_workers_and_worker_calls_stay_serial(monkeypatch):
    from gradient.chronos import ChronosT5Engine, ChronosT5Config
    from rags.execution import SelectionExecution
    probe = ChronosT5Engine.__new__(ChronosT5Engine)
    probe.model = torch.nn.Linear(1, 1).bfloat16()
    probe.device = torch.device("cuda")
    probe.config = ChronosT5Config(context_length=2, prediction_length=1)
    probe.execution = SelectionExecution(gradient_batch=8, gradient_workers=4)
    probe.parameter_count = 201374976
    calls = []
    def processes(values, size):
        calls.append(size)
        return "workers"
    monkeypatch.setattr(probe, "_process_gradients", processes)
    monkeypatch.setattr(probe, "gradient_sketch", lambda row: (float(row.sum()), torch.ones(4096)))
    def forbidden(*args, **kwargs):
        raise AssertionError("BF16 vmap is not the native-window cache protocol")
    monkeypatch.setattr(probe, "_vmap_batch_gradient_sketch", forbidden)
    windows = np.ones((4, 3), dtype=np.float32)
    assert probe.batch_gradient_sketch(windows) == "workers"
    losses, gradients = probe.batch_gradient_sketch(windows, batch_size=1)
    assert calls == [4]
    np.testing.assert_array_equal(losses, np.full(4, 3.))
    assert gradients.shape == (4, 4096)
