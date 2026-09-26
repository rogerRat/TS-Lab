import numpy as np
import torch
import pytest

from rags.core import TimeSeriesFragment
from rags.experiments.calibration import online_curve, validation_from_history
from rags.experiments.train_eval import fine_tune


def test_online_evaluation_does_not_change_training_rng_or_optimizer():
    from types import SimpleNamespace
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.4), torch.nn.Linear(8, 1))
    engine = SimpleNamespace(model=model, device="cpu")
    engine.training_loss = lambda w: (model(torch.tensor(w[:, :3])) - torch.tensor(w[:, 3:])).square().mean()
    windows = np.arange(48, dtype=np.float32).reshape(12, 4) / 100
    base = {k: v.clone() for k, v in model.state_dict().items()}
    fine_tune(engine, windows, steps=8, batch_size=3, learning_rate=1e-3, seed=7)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    captured = []
    def assess():
        model.eval()
        torch.rand(100)  # Deliberately consume RNG; it must be restored.
        captured.append({k: v.clone() for k, v in model.state_dict().items()})
        return {"mase": 1.}
    result = online_curve(engine, windows, base_state=base, learning_rate=1e-3, seed=7,
        checkpoints=[(.5, 2), (1, 4), (2, 8)], batch_size=3, evaluate_fn=assess)
    assert [r['step'] for r in result['curve']] == [2, 4, 8]
    for k in base:
        torch.testing.assert_close(captured[-1][k], expected[k], rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict()[k], base[k], rtol=0, atol=0)


def test_history_validation_reserves_full_windows_and_uses_past_scales():
    fragment = TimeSeriesFragment(np.arange(40, dtype=np.float32), entity_id="test", fragment_id=1)
    validation, bounds = validation_from_history([fragment], context_length=4, prediction_length=2, count=3)
    np.testing.assert_array_equal(validation['test_windows'][:, 0], [30, 32, 34])
    assert bounds[0]['reference_stop_exclusive'] == 30
    assert bounds[0]['validation_stop_exclusive'] == 40
    assert len(validation['test_entities']) == 3
    earlier, earlier_bounds = validation_from_history([fragment], context_length=4,
        prediction_length=2, count=3, gap_windows=5)
    np.testing.assert_array_equal(earlier['test_windows'][:, 0], [20, 22, 24])
    assert earlier_bounds[0]['validation_stop_exclusive'] == bounds[0]['validation_first_window']
    assert earlier_bounds[0]['reference_stop_exclusive'] == 20


def test_online_curve_resume_retains_observations_and_final_weights(tmp_path):
    from types import SimpleNamespace
    torch.manual_seed(9)
    model = torch.nn.Linear(3, 1)
    engine = SimpleNamespace(model=model, device="cpu")
    engine.training_loss = lambda w: (model(torch.tensor(w[:, :3])) - torch.tensor(w[:, 3:])).square().mean()
    windows = np.arange(48, dtype=np.float32).reshape(12, 4) / 100
    base = {k: v.clone() for k, v in model.state_dict().items()}
    captured = []
    def assess():
        captured.append({k: v.clone() for k, v in model.state_dict().items()})
        return {"mase": float(model.weight.detach().sum())}
    kwargs = dict(base_state=base, learning_rate=1e-3, seed=7,
        checkpoints=[(.5, 2), (1, 4), (1.5, 6)], batch_size=3, evaluate_fn=assess)
    expected = online_curve(engine, windows, **kwargs)
    final = captured[-1]
    def interrupt(record):
        if record["epoch"] == 1:
            raise RuntimeError("interrupted before optimizer checkpoint")
    with pytest.raises(RuntimeError, match="interrupted"):
        online_curve(engine, windows, **kwargs, on_record=interrupt,
            resume_path=tmp_path / "resume.pt", observations_path=tmp_path / "curve.json")
    result = online_curve(engine, windows, **kwargs,
        resume_path=tmp_path / "resume.pt", observations_path=tmp_path / "curve.json")
    assert [r["epoch"] for r in result["curve"]] == [.5, 1, 1.5]
    assert [r["metrics"] for r in result["curve"]] == [r["metrics"] for r in expected["curve"]]
    for k in final:
        torch.testing.assert_close(captured[-1][k], final[k], rtol=0, atol=0)
    captured.clear()
    complete = online_curve(engine, windows, **kwargs,
        resume_path=tmp_path / "resume.pt", observations_path=tmp_path / "curve.json")
    assert not captured
    assert complete["curve"] == result["curve"]
