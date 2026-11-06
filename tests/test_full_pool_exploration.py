import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rage.core import TimeSeriesFragment
from rage.experiments import full_pool_exploration as module
from rage.experiments.development_inputs import save_fragments
from rage.experiments.students import atomic_json


@pytest.mark.parametrize("training_data", ["source_random", "safe_history_random", "source_probability"])
@pytest.mark.parametrize("retention", ["keep", "completed_cell_cleanup"])
def test_exploration_separates_history_and_exam_and_resumes(tmp_path, monkeypatch, training_data, retention):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    refs = [TimeSeriesFragment(np.arange(60, dtype=np.float32) / 100,
        entity_id="test", fragment_id=0)]
    pool = [TimeSeriesFragment(np.arange(40, dtype=np.float32) / 100,
        entity_id="source", fragment_id=1)]
    save_fragments(inputs / "test__reference.npz", refs)
    save_fragments(inputs / "pool.npz", pool)
    atomic_json(inputs / "pool_manifest.json", dict(total_equivalents=6))
    model = torch.nn.Linear(4, 2)
    engine = SimpleNamespace(model=model, device="cpu")
    engine.training_loss = lambda w: (model(torch.tensor(w[:, :4])) - torch.tensor(w[:, 4:])).square().mean()
    evaluations = []
    def evaluate(engine, **kwargs):
        evaluations.append(kwargs["test_windows"].copy())
        return dict(mase=float(model.weight.detach().square().mean()))
    monkeypatch.setattr(module, "evaluate", evaluate)
    config = dict(inputs=str(inputs), datasets=["test"], student="chronos-base",
        training_data=training_data, epochs=[.5, 1, 1.5], learning_rates=[1e-4, 1e-3],
        seeds=[0, 1], batch_size=2, context_length=4, prediction_length=2, validation_windows=3,
        checkpoint_retention=retention)
    identity = dict(model_id="fixture", revision="v1", projection_dimension=3, projection_seed=42,
                    precision_policy="cpu-test", ckpt="fixture")
    scored = []
    if training_data == "source_probability":
        from rage.experiments.arms import ArmOutput, SelectionCost
        from rage.experiments.resampling import random_distribution
        def select(arm, probe, **kwargs):
            scored.append(kwargs)
            assert len(kwargs["reference_fragments"][0].data) == 50
            assert np.max(kwargs["reference_fragments"][0].data) < .5
            windows = random_distribution(kwargs["index"], pool_manifest=kwargs["pool_manifest"])
            from rage.experiments.full_pool import COSINE_NUMERICS
            windows.metadata["cosine_numerics"] = COSINE_NUMERICS
            return ArmOutput(arm, windows, [], SelectionCost(), {})
        monkeypatch.setattr("rage.experiments.full_pool.select_full_pool", select)
        epsilon = tmp_path / "epsilon.json"
        atomic_json(epsilon, dict(model=identity, epsilon=.01, percentile=10, count=10, distribution={"10":.01}))
        config.update(arm="rage", temperature=.1, epsilon_calibration=str(epsilon), crb={})
    out = tmp_path / "out"
    bundle = (engine, engine, identity)
    report = module.run_exploration(config, out, model_bundle=bundle)
    assert len(report["trials"]) == 4
    assert len(list(out.glob("lr-*/resume.pt"))) == (4 if retention == "keep" else 0)
    assert len(list(out.glob("lr-*/curve.json"))) == 4
    assert not report["formal_exam_used"]
    bounds = json.loads((out / "validation_boundaries.json").read_text())["test"][0]
    assert bounds["reference_stop_exclusive"] == 50
    assert bounds["validation_first_window"] == 50
    assert all(np.min(w) >= .5 for w in evaluations)
    sampling = json.loads((out / "sampling.json").read_text())
    assert sampling["pool_n"] == (8 if training_data == "safe_history_random" else 6)
    assert all([row["epoch"] for row in trial["curve"]] == [.5, 1, 1.5] for trial in report["trials"])
    evaluations.clear()
    assert module.run_exploration(config, out, model_bundle=bundle) == report
    assert not evaluations
    assert len(scored) == (1 if training_data == "source_probability" else 0)
    with pytest.raises(ValueError, match="configuration changed"):
        module.run_exploration(dict(config, seeds=[9]), out, model_bundle=bundle)
