import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adapters.foundation.timerxl import native_v6_training_batch, v6_single_patch_loss


def test_v6_uses_history_only_for_normalization_and_shapes():
    values = np.arange(2 * 576, dtype=np.float32).reshape(2, 576)
    values[:, 480:] += 20000  # target must not affect history statistics
    batch = native_v6_training_batch(values, device="cpu", dtype=torch.float32)
    assert tuple(batch["input_ids"].shape) == (2, 480)
    assert tuple(batch["labels"].shape) == (2, 96)
    assert tuple(batch["loss_masks"].shape) == (2, 1)
    expected = (values[:, :480] - values[:, :480].mean(1, keepdims=True)) / values[:, :480].std(1, keepdims=True)
    np.testing.assert_allclose(batch["input_ids"].numpy(), expected, rtol=0, atol=1e-6)
    assert float(batch["labels"].mean()) > 100


class _Core(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        # Five patch tokens; the final token depends on the full history.
        hidden = input_ids.reshape(len(input_ids), 5, 96).mean(-1, keepdim=True)
        return type("Output", (), {"last_hidden_state": hidden})()


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(96, 1))

    def forward(self, hidden):
        return hidden @ self.weight.T


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Core()
        self.lm_heads = torch.nn.ModuleList([_Head()])
        self.output_token_len_map = {96: 0}


def test_v6_loss_is_one_next_patch_and_differentiable():
    model = _Model()
    values = np.random.default_rng(1).normal(size=(3, 576)).astype(np.float32)
    batch = native_v6_training_batch(values, device="cpu", dtype=torch.float32)
    loss = v6_single_patch_loss(model, batch)
    assert tuple(loss.shape) == (3,)
    assert torch.isfinite(loss).all() and loss.requires_grad
    loss.mean().backward()
    assert model.lm_heads[0].weight.grad is not None
