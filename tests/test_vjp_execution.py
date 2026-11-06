import numpy as np
import pytest
from test_batched_gradient import make_probe
from rage.execution import SelectionExecution


@pytest.mark.parametrize("vjp", [1, 2, 4, 8, 16])
def test_runtime_vjp_preserves_independent_gradients(vjp):
    probe = make_probe()
    windows = np.arange(21, dtype=np.float32).reshape(7, 3) / 10
    expected_losses, expected = probe.batch_gradient_sketch(windows, batch_size=1)
    probe.execution = SelectionExecution(gradient_vjp_batch=vjp)
    losses, actual = probe.batch_gradient_sketch(windows, batch_size=7)
    np.testing.assert_allclose(losses, expected_losses, rtol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert probe.last_gradient_execution["vjp_batch"] == vjp


def test_negative_vjp_is_rejected():
    with pytest.raises(ValueError):
        SelectionExecution(gradient_vjp_batch=-1)
