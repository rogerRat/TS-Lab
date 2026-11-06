import numpy as np
import torch
from types import SimpleNamespace

from rage.experiments.native_probe import NativeProbe
from rage.execution import encoder_embeddings


def test_native_embedding_dispatch_excludes_future_and_preserves_rows():
    probe = NativeProbe.__new__(NativeProbe)
    probe.config = SimpleNamespace(context_length=3)
    probe.engine = SimpleNamespace(context_embeddings=lambda contexts: contexts.mean(axis=1, keepdims=True))
    windows = np.arange(20, dtype=np.float32).reshape(4, 5)
    expected = windows[:, :3].mean(axis=1, keepdims=True)
    np.testing.assert_array_equal(encoder_embeddings(probe, windows, batch_size=2), expected)
    windows[:, 3:] = 999
    np.testing.assert_array_equal(encoder_embeddings(probe, windows, batch_size=3), expected)


def test_native_full_gradient_projection_finite_difference_and_restore():
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    engine = SimpleNamespace(model=model, device="cpu")
    engine.training_loss = lambda w: (model(torch.tensor(w[:, :3])) - torch.tensor(w[:, 3:])).square().mean()
    probe = NativeProbe(engine, model_key="fixture", projection_dimension=8)
    x = np.array([[.1, .2, .3, .5], [.3, .2, .1, .2]], dtype=np.float32)
    base = {k: v.clone() for k, v in model.state_dict().items()}
    losses, vectors = probe.batch_gradient_sketch(x)
    expected = []
    for w in x:
        loss = engine.training_loss(w[None])
        gradients = torch.autograd.grad(loss, list(model.parameters()))
        expected.append(probe.projector.project(gradients).numpy())
    np.testing.assert_array_equal(vectors, expected)
    audit = probe.finite_difference(x[0])
    assert max(r['relative_error'] for r in audit['finite_differences']) < .001
    for k in base:
        torch.testing.assert_close(model.state_dict()[k], base[k], rtol=0, atol=0)
    np.testing.assert_array_equal(probe.batch_gradient_sketch(x)[1], vectors)
    calls = []
    def observe():
        calls.append(True)
        return len(calls)
    head_audit = probe.finite_difference(x[0], parameter_prefix="bias", forward_observer=observe)
    assert len(calls) == 7
    assert head_audit["initial_forward_observation"] == 1
    assert head_audit["finite_differences"][0]["forward_observations_minus_plus"] == [2, 3]
    assert max(r["relative_error"] for r in head_audit["finite_differences"]) < .001
    for k in base:
        torch.testing.assert_close(model.state_dict()[k], base[k], rtol=0, atol=0)
