from types import SimpleNamespace
import numpy as np
import torch

from rags.core import TimeSeriesFragment
from rags.experiments.arms import WindowIndex, streaming_random_select, RANDOM_SAMPLING_REVISION
from rags.experiments.train_eval import fine_tune


def test_random_training_draws_from_supplied_pool_with_replacement():
    pool = [TimeSeriesFragment(np.arange(n,dtype=np.float32)+i*100,
        entity_id=f'feasible::{i}',fragment_id=i) for i,n in enumerate([10,12])]
    index = WindowIndex(pool, context_length=4,prediction_length=2,stride=1)
    arm = streaming_random_select(index,n_select=2)
    assert len(arm.selected)==12 and arm.selected.budget_windows==2
    assert arm.diagnostics['random_sampling_revision']==RANDOM_SAMPLING_REVISION
    model = torch.nn.Linear(6,1)
    seen=[]
    def loss(windows):
        seen.append(windows.copy())
        return model(torch.from_numpy(windows)).square().mean()
    engine=SimpleNamespace(model=model,device='cpu',training_loss=loss)
    fine_tune(engine,arm.selected,steps=3,batch_size=32,learning_rate=1e-6,seed=7)
    rng=np.random.default_rng(7)
    expected=np.vstack([np.lib.stride_tricks.sliding_window_view(f.data,6) for f in pool])
    for batch in seen:
        draws=rng.integers(0,12,size=32)
        np.testing.assert_array_equal(batch,expected[draws])
        assert len(np.unique(draws))<len(draws)
    assert len(np.unique(np.concatenate(seen)[:,0]))>2


def test_formal_random_artifact_retains_full_pool_and_training_budget(tmp_path):
    from rags.experiments import main_table
    from rags.experiments.regions import load_selection, save_selection
    from rags.experiments.schedule import epoch_checkpoints
    pool = [TimeSeriesFragment(np.arange(n, dtype=np.float32),
            entity_id=str(i), fragment_id=i) for i, n in enumerate([10, 12])]
    index = WindowIndex(pool, context_length=4, prediction_length=2, stride=3)
    arm = main_table.streaming_random_select(index, n_select=2)
    path = tmp_path / 'random.npz'
    save_selection(path, arm)
    restored = load_selection(path)
    assert len(restored.selected) == 12  # All legal starts, not the selector grid.
    assert restored.selected.budget_windows == 2
    assert epoch_checkpoints(restored.selected.budget_windows, 2, 4)[-1] == (4, 4)
    np.testing.assert_array_equal(restored.selected[np.arange(12)], arm.selected[np.arange(12)])


def test_formal_resume_rejects_old_random(tmp_path):
    import json
    import pytest
    from rags.experiments.main_table import parse_args, check_baseline_resume
    from rags.experiments.regions import REVISION
    from baselines.trajectory import REVISION as BASELINE_REVISION
    args = parse_args([])
    protocol = dict(vars(args), delivery_revision=REVISION,
                    language_baseline_revision=BASELINE_REVISION)
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps({'protocol': protocol}))
    with pytest.raises(ValueError, match='Random'):
        check_baseline_resume(tmp_path, args)
    protocol['random_sampling_revision'] = RANDOM_SAMPLING_REVISION
    path.write_text(json.dumps({'protocol': protocol}))
    check_baseline_resume(tmp_path, args)
