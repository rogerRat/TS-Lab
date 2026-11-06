from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_resampling import distribution
from rage.experiments.train_eval import fine_tune


@pytest.mark.parametrize("kind", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_low_memory_adam_has_same_update(kind, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(7)
    left = [torch.nn.Parameter(torch.randn(17, 23, device=device)) for _ in range(3)]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    a = kind(left, lr=1e-5, weight_decay=.01, foreach=False)
    b = kind(right, lr=1e-5, weight_decay=.01, foreach=True)
    for _ in range(8):
        for p, q in zip(left, right):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        a.step()
        b.step()
    for p, q in zip(left, right):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def engine():
    torch.manual_seed(91)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
    seen = []
    def loss(windows):
        seen.append(windows.copy())
        return model(torch.from_numpy(windows)).square().mean()
    return SimpleNamespace(model=model, device="cpu", training_loss=loss, seen=seen)


def test_optimizer_rng_and_exposure_resume_exactly(tmp_path):
    windows = distribution()
    kwargs = dict(steps=6, batch_size=3, learning_rate=1e-5, seed=7)
    uninterrupted = engine()
    expected = fine_tune(uninterrupted, windows, **kwargs)
    interrupted = engine()
    path = tmp_path / "resume.pt"
    def interrupt(step):
        if step == 3:
            raise RuntimeError("simulated process interruption")
    with pytest.raises(RuntimeError, match="interruption"):
        fine_tune(interrupted, windows, **kwargs, on_step=interrupt,
                  resume_path=path, checkpoint_steps=[2, 4, 6])
    restarted = engine()
    actual = fine_tune(restarted, windows, **kwargs, resume_path=path, checkpoint_steps=[2, 4, 6])
    assert actual["resumed_from_step"] == 2
    for key in ("final_loss", "tail_mean", "sample_exposures", "unique_window_exposures", "repeat_fraction"):
        assert actual[key] == expected[key]
    for name, value in restarted.model.state_dict().items():
        torch.testing.assert_close(value, uninterrupted.model.state_dict()[name], rtol=0, atol=0)
    np.testing.assert_array_equal(np.stack(restarted.seen), np.stack(uninterrupted.seen[2:]))
    completed = engine()
    log = fine_tune(completed, windows, **kwargs, resume_path=path)
    assert log["resumed_from_step"] == 6 and completed.seen == []


def test_resume_rejects_changed_schedule_or_distribution(tmp_path):
    windows = distribution()
    kwargs = dict(steps=1, batch_size=2, learning_rate=1e-6, seed=0, resume_path=tmp_path / "state.pt")
    fine_tune(engine(), windows, **kwargs)
    with pytest.raises(ValueError, match="contract changed"):
        fine_tune(engine(), windows, **dict(kwargs, learning_rate=1e-4))
    windows.probabilities = np.array([.3, .7])
    with pytest.raises(ValueError, match="probabilities changed"):
        fine_tune(engine(), windows, **kwargs)


def test_full_pool_epoch_checkpoints_survive_reentry(tmp_path):
    from rage.experiments.train_eval import run_arm_training
    windows = distribution()
    model = torch.nn.Linear(4, 2)
    def loss(w):
        data = torch.from_numpy(w)
        return (model(data[:, :4]) - data[:, 4:]).square().mean()
    def forecast(contexts):
        with torch.no_grad():
            return model(torch.from_numpy(contexts)).numpy()
    native = SimpleNamespace(model=model, training_loss=loss, forecast=forecast, device="cpu")
    kwargs = dict(source_windows=windows, source_epochs=2, batch_size=2, learning_rate=1e-7, seed=3,
        checkpoint_dir=tmp_path, test_windows=windows[np.array([0, 5])],
        test_entities=["a", "b"], context_length=4, test_scales=np.ones(2))
    first = run_arm_training(native, **kwargs)
    again = run_arm_training(native, **kwargs)
    assert [r["epoch"] for r in again["checkpoints"]] == [.5, 1., 1.5, 2.]
    assert [r["step"] for r in again["checkpoints"]] == [1, 2, 3, 3]
    assert [r["metrics"] for r in first["checkpoints"]] == [r["metrics"] for r in again["checkpoints"]]
    assert again["train"]["source"]["resumed_from_step"] == 3
    assert first["mase"] == again["mase"]
