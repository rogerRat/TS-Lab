from types import SimpleNamespace
import json

import numpy as np
import pytest

from rage.experiments.reference_geometry import summarize_trajectory, summarize_task, alignment_summary
from rage.experiments.reference_census import safe_references, fragment_request, collect_references, inventory
from rage.experiments.reference_associations import association, fdr_bh, paired_effect, read_cells, analyze, write_report


def summary(g, **kwargs):
    g = np.asarray(g,dtype=float)
    return summarize_trajectory(lambda a,b:g[a:b],len(g),**kwargs)


def test_constant_gradient_is_smooth_and_aligned():
    r = summary(np.tile([3.,4.],(20,1)),chunk=3,lags=(1,6))
    assert r['cancellation'] == pytest.approx(0)
    assert r['relative_raw_variance'] == pytest.approx(0)
    assert r['lag1_direction_change']['mean'] == pytest.approx(0)
    assert r['relative_curvature']['mean'] == pytest.approx(0)
    assert r['early_late_cosine'] == pytest.approx(1)


def test_alternating_gradients_cancel_with_sharp_turns():
    r = summary(np.tile([[1.,0.],[-1.,0.]],(10,1)),chunk=3,lags=(1,2))
    assert r['cancellation'] == pytest.approx(1)
    assert r['lag1_direction_change']['mean'] == pytest.approx(2)
    assert r['lag2_direction_change']['mean'] == pytest.approx(0)
    assert r['lag1_negative_fraction'] == 1
    assert r['relative_curvature']['mean'] == pytest.approx(1)


def test_zero_gradients_and_singleton_are_not_fake_directions():
    r = summary([[0.,0.]])
    assert r['cancellation'] is None
    assert r['early_late_cosine'] is None
    assert r['lag1_direction_change']['mean'] is None
    assert r['zero_fraction'] == 1
    assert r['nonzero_directional_variance'] is None


def test_chunk_boundary_and_scale_invariance():
    g = np.random.default_rng(1).normal(size=(37,5))
    a, b = summary(g,chunk=3,lags=(1,7)), summary(g*100,chunk=19,lags=(1,7))
    for key in ('cancellation','norm_cv','relative_raw_variance','early_late_relative_shift'):
        assert a[key] == pytest.approx(b[key])
    for key in ('lag1_direction_change','lag7_direction_change','relative_curvature'):
        assert a[key]['mean'] == pytest.approx(b[key]['mean'])


def test_linear_reconstruction_and_anchor_bias():
    g = np.column_stack((np.arange(11),np.ones(11)))
    r = summary(g,chunk=4,anchors={0:g[0],10:g[10]})
    assert r['crb']['nonanchor_windows'] == 9
    assert r['crb']['relative_rmse'] == pytest.approx(0,abs=1e-15)
    assert r['crb']['anchor_mean_relative_bias'] == pytest.approx(0)
    g[5] += 100
    r = summary(g,anchors={0:g[0],10:g[10]})
    assert r['crb']['nonanchor_relative_rmse'] > .8


def test_directional_within_between_decomposition():
    a = summary(np.tile([1.,0.],(3,1)))
    b = summary(np.tile([-1.,0.],(30,1)))
    r = summarize_task([a,b])
    assert r['within_directional_variance'] == pytest.approx(0)
    assert r['between_directional_variance'] == pytest.approx(1)
    assert r['mean_raw'] == pytest.approx([0,0])  # Equal fragments, not lengths.


def test_exact_alignment_matches_explicit_pairwise():
    ref = np.array([[1.,0.],[0.,1.],[0.,0.]])
    source = np.array([[1.,1.],[-1.,0.],[0.,0.]])
    u = source/np.maximum(np.linalg.norm(source,axis=1,keepdims=True),1e-30)
    reference = summarize_task([summary(ref)])
    result = alignment_summary(source,reference)
    assert result['mean_pair_cosine'] == pytest.approx((u@ref.T).mean())


def write_inputs(path, *, unsafe=False):
    from rage.core import TimeSeriesFragment
    from rage.experiments.development_inputs import save_fragments
    f = TimeSeriesFragment(np.arange(14),entity_id='e',fragment_id=9)
    save_fragments(path/'task__reference.npz',[f])
    manifest = dict(dataset='task',full_reference_history=True, formal_exam_excluded_from_reference=True,
        source_training_only=True,context_length=4,prediction_length=2,
        series=[dict(entity='e',exam_boundary=13 if unsafe else 14,
                     reference_segments=[dict(fragment_id=9,start=0,stop=14)])])
    (path/'task__manifest.json').write_text(json.dumps(manifest))
    return f


def test_safe_history_boundaries_and_complete_starts(tmp_path):
    write_inputs(tmp_path)
    refs,_ = safe_references(tmp_path,'task',4,2)
    request = fragment_request(refs[0],'reference:task',0,9,4,2)
    assert request.ids.tolist() == [[9,i] for i in range(1,10)]
    np.testing.assert_array_equal(request.windows(np.array([0,8])),[[0,1,2,3,4,5],[8,9,10,11,12,13]])
    write_inputs(tmp_path,unsafe=True)
    with pytest.raises(ValueError,match='unsafe'):
        safe_references(tmp_path,'task',4,2)


def test_manifest_cannot_claim_sampled_history(tmp_path):
    write_inputs(tmp_path)
    p = tmp_path/'task__manifest.json'; m = json.loads(p.read_text()); m['full_reference_history']=False
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='full-history'):
        safe_references(tmp_path,'task',4,2)


def test_all_window_cache_restart_and_crb_replay_without_bp(tmp_path):
    from assets.gradient.cache import GradientCache
    from rage.execution import SelectionExecution
    from rage.crb import CRBConfig
    class Probe:
        device='cpu'
        execution=SelectionExecution(gradient_batch=3)
        config=SimpleNamespace(projection_dimension=2,projection_seed=1)
        count=0
        def batch_gradient_sketch(self,windows):
            self.count += len(windows)
            return windows.mean(1), windows[:,[0,-1]]
    probe=Probe(); f=write_inputs(tmp_path)
    def cache():
        return GradientCache(tmp_path,probe,model_identity={'model_id':'fixture'},data_fields={},context_length=4,prediction_length=2)
    first=cache()
    first.prepare([fragment_request(f,'reference:task',0,3,4,2)])
    first.close()
    second=cache()
    result=collect_references(second,[f],'task',4,2,CRBConfig(context_length=4,prediction_length=2),chunk=4)
    assert probe.count == 9
    assert result['coverage'] == dict(expected=9,actual=9,complete=True,stride=1)
    again=collect_references(second,[f],'task',4,2,CRBConfig(context_length=4,prediction_length=2),chunk=2)
    assert probe.count == 9
    assert again['summary']['mean_raw'] == pytest.approx(result['summary']['mean_raw'])
    second.close()


def test_pairing_requires_all_seeds_and_uses_ratio_of_means():
    records={('m','t','a',s):v for s,v in enumerate([1.,2.,3.])}
    records.update({('m','t','b',s):v for s,v in enumerate([2.,4.,6.])})
    r=paired_effect(records,'m','t','a','b',[0,1,2])
    assert r['relative_gain_percent'] == 50
    assert r['seed_wins'] == 3
    del records[('m','t','b',1)]
    assert paired_effect(records,'m','t','a','b',[0,1,2]) is None


def test_permutation_sign_null_and_reproducibility():
    a=association(range(9),range(9),permutations=999,bootstraps=99)
    b=association(range(9),list(range(9))[::-1],permutations=999,bootstraps=99)
    assert a['rho'] == pytest.approx(1)
    assert b['rho'] == pytest.approx(-1)
    assert a['p'] <= .01
    assert a == association(range(9),range(9),permutations=999,bootstraps=99)
    assert association([1]*9,range(9))['p'] is None
    assert association([1,2],[1,2])['status'].startswith('insufficient')


def test_fdr_includes_all_valid_tests_and_preserves_missing():
    assert fdr_bh([.01,.04,None,.03]) == pytest.approx([.03,.04,None,.04])


def test_inventory_deduplicates_but_rejects_conflicting_inputs(tmp_path):
    inputs=tmp_path/'inputs'; inputs.mkdir(); write_inputs(inputs)
    runs=[]
    for name in ('r1','r2'):
        run=tmp_path/name;run.mkdir();runs.append(run)
        (run/'protocol.json').write_text(json.dumps(dict(inputs=str(inputs),context_length=4,prediction_length=2,
                                                       datasets=['task'],students=['m'])))
    result=inventory(runs)
    assert result['unique_reference_windows'] == 9
    assert result['unique_model_task_pairs'] == 1


def test_end_to_end_association_report_and_missing_cell(tmp_path):
    run=tmp_path/'run'; run.mkdir(); census=tmp_path/'census'; (census/'m').mkdir(parents=True)
    tasks=[f't{i}' for i in range(6)]
    config=dict(students=['m'],datasets=tasks,arms=['rand','rage-block-linearSum','rage-traj-linearSum','tsds'],
                seeds=[0,1,2],experiment_version='v5')
    (run/'protocol.json').write_text(json.dumps(config))
    for i,t in enumerate(tasks):
        (census/'m'/f'{t}__geometry.json').write_text(json.dumps(dict(student='m',dataset=t,coverage={'complete':True},
            summary=dict(within_directional_variance=i/10,windows=100+i,trajectories=2))))
        for seed in config['seeds']:
            folder=run/'rounds'/f'seed{seed}'/'students'/'m'/'cells';folder.mkdir(parents=True,exist_ok=True)
            for arm in config['arms']:
                value=1-i*.03 if arm=='rage-block-linearSum' else 1
                (folder/f'{t}__{arm}__seed{seed}.json').write_text(json.dumps(dict(student='m',dataset=t,arm=arm,
                    seed=seed,experiment_version='v5',mase=value)))
    result=analyze([run],census,permutations=99,bootstraps=29)
    assert any(t['rho']==pytest.approx(1) for t in result['tests'] if t['feature']=='within_directional_variance')
    assert all(t['n_tasks']<=6 for t in result['tests'])
    assert len(result['missing']) == 6
    assert all(r['kind']=='alignment' for r in result['missing'])
    write_report(result,tmp_path/'report')
    assert (tmp_path/'report'/'REPORT.md').exists()
    cell=run/'rounds'/'seed2'/'students'/'m'/'cells'/'t0__tsds__seed2.json'
    cell.unlink()
    assert len(read_cells(run)[2]) == 1


def test_compute_entrypoint_and_actual_selection_sampling(tmp_path):
    from assets.gradient.cache import GradientCache
    from rage.execution import SelectionExecution
    from rage.core import TimeSeriesFragment
    from rage.experiments.arms import WindowIndex, ArmOutput, SelectionCost
    from rage.experiments.resampling import ResampledWindows
    from rage.experiments.regions import save_selection
    from rage.experiments.reference_census import compute
    inputs=tmp_path/'inputs';inputs.mkdir();write_inputs(inputs)
    manifest={'total_equivalents':4}
    (inputs/'pool_manifest.json').write_text(json.dumps(manifest))
    run=tmp_path/'run';run.mkdir()
    config=dict(inputs=str(inputs),context_length=4,prediction_length=2,crb={},students=['m'],datasets=['task'],
        arms=['rand','tsds'],selection_execution={'gradient_batch':3},settings={'m':{'temperature':1.}})
    (run/'protocol.json').write_text(json.dumps(config))
    pool=[TimeSeriesFragment(np.arange(12)+i*100,fragment_id=i) for i in range(2)]
    index=WindowIndex(pool,context_length=4,prediction_length=2)
    for arm,p in [('rand',[.5,.5]),('tsds',[0.,1.])]:
        selected=ResampledWindows.from_index(index,list(index.positions_by_fragment().values()),p,
                                             pool_manifest=manifest,metadata={})
        path=run/'rounds'/'seed0'/'students'/'m'/'selections'/f'task__{arm}.npz'
        save_selection(path,ArmOutput(arm,selected,[],SelectionCost(),{}))
    class Probe:
        device='cpu'; execution=SelectionExecution(gradient_batch=3)
        config=SimpleNamespace(projection_dimension=2,projection_seed=1)
        count=0
        def batch_gradient_sketch(self,windows):
            self.count += len(windows)
            return windows.mean(1),windows[:,[0,-1]]
    probe=Probe();identity={'model_id':'fixture','projection_dimension':2}
    args=SimpleNamespace(run=str(run),student='m',out=str(tmp_path/'out'),gradient_batch=None,
                         chunk=4,stage='references',alignment_samples=128,seed=42)
    compute(args,model_bundle=(probe,probe,identity))
    assert probe.count == 9+6  # Three batched/single native-check windows.
    args.stage='alignment'
    compute(args,model_bundle=(probe,probe,identity))
    assert probe.count == 9+6+len(index)
    result=json.loads((tmp_path/'out'/'m'/'task__alignment__run.json').read_text())
    tsds=result['alignment']['tsds']
    assert all(fid==1 for fid,start in tsds['sampled_window_ids'])
    assert tsds['metrics']['window_ess_fraction'] == pytest.approx(.5)
    assert tsds['metrics']['unique_sampled_windows'] == 7
    before=probe.count
    compute(args,model_bundle=(probe,probe,identity))
    assert probe.count == before


def test_oom_halves_batch_and_retries_only_missing(tmp_path):
    import torch
    from assets.gradient.cache import GradientCache
    from rage.execution import SelectionExecution
    from rage.experiments.reference_census import prepare_with_oom
    class Probe:
        device='cpu';execution=SelectionExecution(gradient_batch=8)
        config=SimpleNamespace(projection_dimension=2,projection_seed=1)
        seen=[]
        def batch_gradient_sketch(self,windows):
            self.seen.append(len(windows))
            if len(windows)>2:
                raise torch.cuda.OutOfMemoryError('synthetic')
            return windows.mean(1), windows[:,[0,-1]]
    f=write_inputs(tmp_path);probe=Probe()
    cache=GradientCache(tmp_path,probe,model_identity={'model_id':'fixture'},data_fields={},context_length=4,prediction_length=2)
    prepare_with_oom(cache,fragment_request(f,'reference:task',0,9,4,2))
    assert probe.execution.gradient_batch==2
    assert cache.computed_windows==9
    cache.close()


def test_native_probe_check_rejects_batch_mean_as_independent_gradients():
    from rage.experiments.reference_census import validate_probe
    class BadProbe:
        def batch_gradient_sketch(self,windows):
            return windows.mean(1),np.tile(windows.mean(0),(len(windows),1))
    assert not validate_probe(BadProbe(),np.asarray([[1.,2.],[10.,20.]]))['passed']
