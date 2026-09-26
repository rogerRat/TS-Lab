from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from baselines.ltsv.fast import (BlockScorer, block_rng, block_starts, score_fragment,
    reference_mse, optimized_select, REVISION)
from baselines.ltsv.runner import _ltsv_reference_mse
from rags.execution import SelectionExecution
from rags.core import TimeSeriesFragment
from rags.experiments.arms import WindowIndex


def engine():
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 2))
    def forecast(x):
        return model(torch.tensor(x)).detach().numpy()
    def loss(w):
        return (model(torch.tensor(w[:, :3])) - torch.tensor(w[:, 3:])).square().mean()
    return SimpleNamespace(model=model, forecast=forecast, training_loss=loss,
        config=SimpleNamespace(context_length=3, prediction_length=2),
        execution=SelectionExecution(), device='cpu')


def test_native_batching_preserves_original_reduction():
    probe = engine()
    probe.model.eval()
    val = np.arange(350, dtype=np.float32).reshape(70, 5)/300
    assert reference_mse(probe, val, 64) == _ltsv_reference_mse(probe, val)


def test_tail_block_and_long_block_subwindows():
    np.testing.assert_array_equal(block_starts(12, 5, 2), [0, 2, 4, 6, 7])
    np.testing.assert_array_equal(block_starts(4, 5, 2), [])
    probe = engine()
    seen = []
    original = probe.training_loss
    def record(w):
        seen.append(w.shape)
        return original(w)
    probe.training_loss = record
    scorer = BlockScorer(probe, np.ones((2,5),dtype=np.float32))
    score_fragment(scorer, (0, np.arange(9,dtype=np.float32), 0, None), width=7, stride=2)
    assert seen == [(3,5), (3,5)]
    scorer.close()


def test_blocks_match_fresh_adam_and_preserve_rng_and_weights():
    probe = engine()
    initial = {k:v.clone() for k,v in probe.model.state_dict().items()}
    val = np.arange(20,dtype=np.float32).reshape(4,5)/10
    window = val[:1]
    scorer = BlockScorer(probe, val, seed=24)
    state = torch.get_rng_state().clone()
    score = scorer.score(window, 7)
    assert torch.equal(state, torch.get_rng_state())
    for k,v in probe.model.state_dict().items():
        assert torch.equal(v, initial[k])
    with block_rng('cpu',24,7):
        probe.model.train()
        optimizer = torch.optim.Adam(probe.model.parameters(), lr=1e-5,
            foreach=False, betas=(.9,.999), eps=1e-8, weight_decay=.1)
        optimizer.zero_grad(set_to_none=True)
        probe.training_loss(window).backward()
        torch.nn.utils.clip_grad_norm_(probe.model.parameters(), 1.)
        optimizer.step()
        probe.model.eval()
        expected = scorer.base_loss - _ltsv_reference_mse(probe, val)
    assert score == expected
    scorer.close()


def test_resume_matches_uninterrupted_and_handles_failure(tmp_path):
    probe = engine()
    scorer = BlockScorer(probe, np.ones((2,5), dtype=np.float32))
    task = (5, np.arange(14,dtype=np.float32)/10, 12, str(tmp_path/'fragment-5.npz'))
    original = scorer.score
    calls = []
    def failing(window, ordinal):
        calls.append(ordinal)
        if len(calls)==4:
            raise RuntimeError('interrupted')
        return original(window,ordinal)
    scorer.score = failing
    with pytest.raises(RuntimeError, match='interrupted'):
        score_fragment(scorer,task,width=5,stride=2,checkpoint_every=2)
    scorer.score = original
    resumed = score_fragment(scorer,task,width=5,stride=2,checkpoint_every=2)
    fresh = score_fragment(scorer,(*task[:3],None),width=5,stride=2)
    np.testing.assert_array_equal(resumed['values'],fresh['values'])
    assert resumed['resumed_blocks']==2
    cached = score_fragment(scorer,task,width=5,stride=2)
    assert cached['computed_blocks']==0
    scorer.close()


def test_failure_in_training_restores_and_clears_gradients():
    probe = engine()
    scorer = BlockScorer(probe,np.ones((2,5),dtype=np.float32))
    initial = [p.clone() for p in probe.model.parameters()]
    def broken(w):
        with torch.no_grad():
            next(probe.model.parameters()).add_(1)
        raise RuntimeError('training failed')
    probe.training_loss = broken
    with pytest.raises(RuntimeError,match='training failed'):
        scorer.score(np.ones((1,5),dtype=np.float32),0)
    assert all(torch.equal(a,b) for a,b in zip(initial,probe.model.parameters()))
    assert all(p.grad is None for p in probe.model.parameters())
    scorer.close()


def test_full_selection_resumes_and_rejects_mismatched_identity(tmp_path):
    probe = engine()
    pool = [TimeSeriesFragment(np.linspace(-2,3,12,dtype=np.float32),fragment_id=1)]
    refs = [TimeSeriesFragment(np.linspace(0,2,10,dtype=np.float32),fragment_id=9)]
    index = WindowIndex(pool,context_length=3,prediction_length=2)
    execution = dict(schema=REVISION,model={},workers=1,inference_batch=64,
        generation='eager',block_seed=20260916,baseline_seconds_per_block=1.)
    args = dict(reference_fragments=refs,index=index,context_length=3,prediction_length=2,
        cache_dir=tmp_path,pool_manifest={'total_equivalents':2},dataset_tag='test',seed=0,
        resampling={'model':{},'temperature':1.},execution=execution)
    result = optimized_select(probe,**args)
    assert result.cost.gradient_computations==5
    assert result.cost.notes['runtime_adjustment'] is not None
    second = optimized_select(probe,**args)
    assert second.cost.gradient_computations==0
    assert second.cost.notes['scores_cached']
    np.testing.assert_array_equal(result.selected.probabilities,second.selected.probabilities)
    execution['block_seed']+=1
    with pytest.raises(ValueError,match='cache fields differ'):
        optimized_select(probe,**args)


def test_budget_preserves_coverage_and_accounts_original_block_count(tmp_path):
    probe = engine()
    pool = [TimeSeriesFragment(np.linspace(-2,3,12,dtype=np.float32),fragment_id=1)]
    refs = [TimeSeriesFragment(np.linspace(0,2,50,dtype=np.float32),fragment_id=9)]
    index = WindowIndex(pool,context_length=3,prediction_length=2)
    execution = dict(schema=REVISION,model={},workers=1,inference_batch=2,
        generation='eager',block_seed=20260916,baseline_seconds_per_block=1.,
        context_windows=2,block_stride=5)
    result = optimized_select(probe,reference_fragments=refs,index=index,
        context_length=3,prediction_length=2,cache_dir=tmp_path,
        pool_manifest={'total_equivalents':2},dataset_tag='budget',seed=0,
        resampling={'model':{},'temperature':1.},execution=execution)
    assert result.cost.gradient_computations==3
    assert np.isfinite(result.selected.native_scores).all()
    assert result.cost.notes['original_blocks']==5
    assert result.cost.notes['budget_blocks']==3
    adjusted = result.cost.wall_seconds + result.cost.notes['runtime_adjustment']['add_back_seconds']
    assert adjusted == pytest.approx(5.)
    manifest=json.loads((tmp_path/REVISION/'budget/cache_manifest.json').read_text())
    assert len(manifest['reference_ids'])==2
    assert manifest['stride']==5


def test_block_schedule_does_not_change_scores():
    probe = engine()
    val=np.arange(20,dtype=np.float32).reshape(4,5)/10
    scorer=BlockScorer(probe,val,seed=24)
    forward={i:scorer.score(val[i:i+1],i) for i in range(4)}
    reverse={i:scorer.score(val[i:i+1],i) for i in reversed(range(4))}
    assert forward==reverse
    scorer.close()


def test_runner_dispatch_uses_explicit_execution_config_without_legacy_hash(tmp_path):
    from baselines.ltsv.runner import ltsv_select
    probe=engine()
    index=WindowIndex([TimeSeriesFragment(np.linspace(-2,3,12,dtype=np.float32),fragment_id=1)],
        context_length=3,prediction_length=2)
    refs=[TimeSeriesFragment(np.linspace(0,2,50,dtype=np.float32),fragment_id=9)]
    execution=dict(schema=REVISION,model={},workers=1,inference_batch=2,
        generation='eager',block_seed=20260916,context_windows=2,block_stride=5)
    (tmp_path/'ltsv_execution.json').write_text(json.dumps(execution))
    result=ltsv_select(probe,reference_fragments=refs,index=index,
        context_length=3,prediction_length=2,cache_dir=tmp_path,
        pool_manifest={'total_equivalents':2},dataset_tag='dispatch',
        resampling={'model':{},'temperature':1.})
    assert result.cost.gradient_computations==3
    assert (tmp_path/REVISION/'dispatch/cache_manifest.json').exists()
    assert not list(tmp_path.glob('ltsv_scores*'))


def test_report_labels_budget_accuracy_and_original_counterfactual(tmp_path):
    from scripts.exp.report_selection_runtime import report
    (tmp_path/'protocol.json').write_text(json.dumps(dict(students=['model'],datasets=['task'],arms=['ltsv'])))
    folder=tmp_path/'students/model/selections'
    folder.mkdir(parents=True)
    np.savez(folder/'task__ltsv.npz',cost=json.dumps(dict(wall_seconds=100,notes=dict(
        scores_cached=False,execution={'label':'LTSV-Budget'},runtime_adjustment=dict(
            report_basis='estimated_original_full_budget_serial_implementation',add_back_seconds=200)))))
    row=report(tmp_path)['rows'][0]
    assert row['method_label']=='LTSV-Budget'
    assert row['quality_protocol']=='budget_variant'
    assert row['original_cost_is_counterfactual']
    assert row['actual_execution_seconds']==100
    assert row['original_full_budget_seconds_estimate']==300
