"""No threshold, relative cancellation, rightmost ties, exact capped budget."""
from dataclasses import replace
import numpy as np
import pytest
from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine, CRBEngineMulti, _Interval


def cfg(n, cover=3, error=0, split=5):
    return CRBConfig(context_length=2, prediction_length=1,
        cover_density=cover/max(n,1), error_density=error/max(n,1), split_density=split/max(n,1),
        split_policy="min_kappa", epsilon_policy="none", priority_tie_policy="right")


def run(config, gradients, fragment_id=0):
    f=TimeSeriesFragment(np.arange(len(gradients)+2,dtype=np.float32),fragment_id=fragment_id)
    observed=[]
    def field(_f,starts):
        ids=np.asarray(starts)-1
        observed.extend(ids.tolist())
        return gradients[ids]
    return CRBEngine(config,field).process_fragment(f),observed


@pytest.mark.parametrize('left,right,expected', [
    ([1.,0.],[-1.,0.],0.), ([1.,0.],[1.,0.],1.),
    ([1.,0.],[0.,0.],1.), ([0.,0.],[0.,0.],1.),
    ([1.,0.],[-.9,0.],1/19), ([100.,0.],[-1.,0.],99/101),
])
def test_relative_cancellation_score(left,right,expected):
    value=CRBEngine._interval_kappa(_Interval(0,2),{0:np.array(left),2:np.array(right)})
    assert value == pytest.approx(expected)


def test_kappa_is_scale_invariant_and_uses_raw_norms():
    interval=_Interval(0,2)
    for scale in (1e-30,1.,1e30):
        assert CRBEngine._interval_kappa(interval,
            {0:scale*np.array([2.,0.]),2:scale*np.array([-1.,0.])}) == pytest.approx(1/3)


@pytest.mark.parametrize('n',[0,1,2,3,9,17,33])
@pytest.mark.parametrize('value',[0.,1.])
def test_zero_and_flat_fields_exhaust_budget_without_quantile_or_rng(n,value,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No P10 or RNG allowed')
    monkeypatch.setattr(np,'quantile',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    config=cfg(n,cover=3,split=n)
    result,observed=run(config,np.full((n,2),value,dtype=np.float32))
    assert result.split_computations == config.budgets(n)[2]
    assert result.epsilon_audit is None
    assert result.split_audit['split_unused'] == 0
    assert result.split_audit['tie_rng_enabled'] is False
    assert len(observed)==len(set(observed))==sum(config.budgets(n))
    assert result.discarded_no_signal==result.discarded_antagonistic==result.banned_budget==0
    if n:
        assert np.concatenate([b.start_positions for b in result.blocks]).tolist()==result.valid_starts.tolist()


def test_small_absolute_midpoint_does_not_beat_true_cancellation():
    gradients=np.ones((17,1),dtype=np.float32)
    gradients[[0,8,16],0]=[.001,.001,-.9]
    result,observed=run(cfg(17,split=1),gradients)
    assert observed==[0,16,8,12]
    assert result.split_computations==1


@pytest.mark.parametrize('seed',[0,1,2,3])
def test_matches_independent_recursive_oracle_and_multi(seed):
    n=33
    gradients=np.random.default_rng(seed).normal(size=(n,3)).astype(np.float32)
    config=cfg(n,cover=6,error=3,split=20)
    result,observed=run(config,gradients)
    prefix=result.cover_computations+result.error_computations
    anchors=sorted(observed[:prefix]); intervals=list(zip(anchors[:-1],anchors[1:]))
    expected=[]
    while len(expected)<config.budgets(n)[2]:
        candidates=[]
        for l,r in intervals:
            if r-l<=1:continue
            a,b=gradients[l].astype(np.float64),gradients[r].astype(np.float64)
            denominator=np.linalg.norm(a)+np.linalg.norm(b)
            kappa=min(1.,float(np.linalg.norm(a+b)/denominator)) if denominator else 1.
            candidates.append((kappa,-l,l,r))
        _,_,l,r=min(candidates)
        m=(l+r)//2;expected.append(m)
        intervals.remove((l,r));intervals.extend([(l,m),(m,r)])
    assert observed[prefix:]==expected
    assert [(b.left_index,b.right_index) for b in result.blocks]==sorted(intervals)
    fragments=[TimeSeriesFragment(np.arange(n+2,dtype=np.float32),fragment_id=i) for i in (1,0)]
    class Many:
        def compute_many(self,requests):
            return np.concatenate([gradients[np.asarray(s)-1] for f,s in requests])
    parallel=CRBEngineMulti(config,Many()).process_fragments(fragments)
    for other in parallel:
        assert sorted(other.anchors)==sorted(result.anchors)
        assert other.split_computations==result.split_computations
        assert other.epsilon_audit is None


def test_protocol_rejects_p10_or_random_ties():
    config=cfg(17)
    for changes in ({'epsilon_policy':'trajectory_p10'},{'priority_tie_policy':'random'},
                    {'initial_placement':'random'}):
        with pytest.raises(ValueError):replace(config,**changes)


def test_worker_without_calibration_emits_split_audit_not_p10(tmp_path):
    import json
    from test_full_pool_worker import fixture
    from assets.gradient.cache import atomic_json
    from rage.experiments.full_pool_worker import run_worker
    config,bundle=fixture(tmp_path)
    config['arms']=['rage-block-linearSum']
    config['crb']=dict(split_policy='min_kappa',epsilon_policy='none',
        priority_tie_policy='right',cover_density=.3,error_density=.1,split_density=.1)
    from rage.experiments.protocol import method_protocols
    config['method_protocols']=method_protocols(config['arms'])
    atomic_json(tmp_path/'protocol.json',config)
    out=run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    audits=list(tmp_path.rglob('crb-splits-tiny.json'))
    assert len(audits)==1
    report=json.loads(audits[0].read_text())
    assert report['schema']=='min-kappa-split-v5'
    assert not list(tmp_path.rglob('crb-thresholds-*.json'))
    for row in report['source']+report['reference']:
        assert row['split_used']==row['split_budget'] and row['split_unused']==0
