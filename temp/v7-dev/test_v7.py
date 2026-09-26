import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig
from rags.experiments.arms import WindowIndex, ArmOutput, SelectionCost
from rags.experiments.v7_pools import training_pool
from rags.experiments.v7_protocol import validate, make_config, exposure_audit, crb_for, NEW_ARMS
from rags.experiments.v7_scores import uniform_anchors, reference_mean, score_trajectory, deploy
from rags.experiments.v7_cache import V7Cache
from rags.experiments.v7_assets import prepare
from rags.experiments.v7_run import freeze, gate, cell_identity, marker
from rags.experiments import v7_io
from assets.gradient.cache import WindowRequest

def pool_fixture():
    counts = [1,2,3,4,5,6,7,8,9,10,11,12]
    pool = [TimeSeriesFragment(np.arange(n*576,dtype=np.float32),fragment_id=100+i,
        entity_id=f'native-{i}') for i,n in enumerate(counts)]
    return pool,dict(rows=list(range(len(pool))),total_equivalents=sum(counts),
        source_load='native-root',window_length=576,pool_protocol='exact-complete-trajectories-v1')

def test_exact_whole_trajectory_seed_pools_stable_ids_and_overlap():
    feasible,m = pool_fixture()
    selections = []
    for seed in range(3):
        pool,manifest = training_pool(feasible,m,seed,target=19)
        assert manifest['total_equivalents']==19
        assert sum(f.length//576 for f in pool)==19
        assert all(any(f is old for old in feasible) for f in pool)
        assert manifest==training_pool(feasible,m,seed,target=19)[1]
        selections.append(manifest)
    assert len({tuple(m['fragment_ids']) for m in selections})==3
    assert set(selections[0]['fragment_ids']) & set(selections[2]['fragment_ids'])
    with pytest.raises(ValueError,match='unreachable'):
        training_pool(feasible[:1],dict(m,rows=[0],total_equivalents=1),0,target=2)

def test_protocol_and_exposure():
    c=make_config(json.loads(Path('configs/rags/main.json').read_text()),
                  json.loads(Path('configs/rags/main_v6_timerxl.json').read_text()))
    validate(c)
    assert len(c['arms'])==20
    for row in exposure_audit(c).values():
        assert row['pool_n']==16284 and row['per_epoch_exposure_ratio_vs_v6']==1
        assert row['schedule_steps']==3054
    assert exposure_audit(c)['chronos-base']['max_steps']==1527
    assert exposure_audit(c)['timerxl-84m']['max_steps']==2036
    for k,v in [('training_pool_equivalents',65136),('feasible_pool_equivalents',48852)]:
        with pytest.raises(ValueError):validate(dict(c,**{k:v}))

@pytest.mark.parametrize('n', [1,2,3,10,99,100,101,1000,10001])
def test_uniform_matches_capped_crb(n):
    cfg=crb_for('timerxl-84m')
    k=sum(cfg.budgets(n))
    p=uniform_anchors(n,k)
    assert len(np.unique(p))==k and p[0]==0
    assert p[-1]==n-1
    assert all(b>=0 for b in cfg.budgets(n)) and k<=n
    assert uniform_anchors(n,1).tolist()==[0]

def test_scores_match_dense_reference_and_signed_zero_behavior():
    p=np.array([0,3,6]);g=np.array([[1.,-2.],[3.,4.],[-1.,1.]])
    dense=np.stack([np.interp(np.arange(7),p,g[:,j]) for j in range(2)],axis=1)
    target=np.array([-1.,2.])
    scores=score_trajectory(7,p,g,target,chunk=2)
    np.testing.assert_allclose(scores['grad-dot'],dense@target,rtol=0,atol=1e-14)
    np.testing.assert_allclose(scores['grand-static'],np.linalg.norm(dense,axis=1))
    np.testing.assert_allclose(scores['grad-cos'],dense@target/np.linalg.norm(dense,axis=1)/np.linalg.norm(target))
    assert (scores['grad-dot']<0).any()
    target_mean=reference_mean([(7,p,g),(1,[0],np.array([[11.,-2.]]))])
    np.testing.assert_allclose(target_mean,(dense.mean(axis=0)+[11.,-2.])/2)
    assert not np.allclose(target_mean,(dense.sum(axis=0)+[11.,-2.])/8)
    assert np.all(score_trajectory(7,p,np.zeros_like(g),target)['grad-cos']==0)
    assert np.all(score_trajectory(7,p,g,[0.,0.])['grad-cos']==0)
    with pytest.raises(ValueError):score_trajectory(7,p,g,[np.nan,0])
    with pytest.raises(ValueError):score_trajectory(7,p,g*np.nan,target)

def test_window_rank_ties_compact_io_and_seed_identity(tmp_path):
    pool,m=pool_fixture();pool,m=training_pool(pool,m,0,target=19)
    idx=WindowIndex(pool,context_length=480,prediction_length=96)
    scores=np.arange(len(idx),dtype=float)//3
    selected=deploy(idx,scores,arm='grad-dot',manifest=m,identity={'seed':0})
    from rags.experiments.ranking import rank_mass
    np.testing.assert_array_equal(selected.probabilities,rank_mass(scores,np.ones(len(idx)))[1])
    assert np.isclose(selected.probabilities.sum(),1,atol=1e-12) and selected.probabilities.min()>0
    assert selected.probabilities[0]==selected.probabilities[1]
    assert selected.pool_n==19
    rng=np.random.default_rng(0);sample=selected.sample_indices(rng,100000)
    assert set(selected.window_ids[sample,0])<=set(m['fragment_ids'])
    output=ArmOutput('grad-dot',selected,[],SelectionCost(),{})
    path=tmp_path/'s.npz';identity={'seed':0,'pool':m}
    v7_io.save(path,output,identity)
    recovered=v7_io.load(path,idx,m,identity)
    np.testing.assert_array_equal(recovered.selected.native_scores,scores)
    np.testing.assert_array_equal(recovered.selected.probabilities,selected.probabilities)
    with np.load(path) as data:
        assert 'values' not in data and 'window_ids' not in data
        assert len(data['starts'])==0
    with pytest.raises(ValueError,match='identity'):
        v7_io.load(path,idx,m,dict(identity,seed=1))

class Probe:
    config=SimpleNamespace(layer='all',projection_dimension=4096,projection_seed=42,projection_repeats=1)
    device='cpu'
    def __init__(self):self.count=0
    def batch_gradient_sketch(self,windows):
        self.count+=len(windows)
        x=np.asarray(windows,dtype=np.float32)
        g=np.tile(np.stack((x.mean(1),x.std(1),x[:,0],x[:,-1]),axis=1),(1,1024))
        return np.square(x).mean(1),g

def cache(root,probe,**kw):
    return V7Cache(root,probe,model_identity={'model_id':'fixture','revision':'r'},
        data_fields={'pool':{'seed':0}},context_length=4,prediction_length=2,**kw)

def request(fid,start):
    return WindowRequest('source',[[fid,start]],lambda rows:np.arange(6,dtype=np.float32)[None,:])

def test_cross_seed_cache_dedup_and_readonly_missing_rejection(tmp_path,monkeypatch):
    monkeypatch.delenv('TS3_GRADIENT_CACHE_ROOT',raising=False)
    p=Probe();shared={'feasible':{'N':40},'seed_manifests':{'0':'a','1':'b'}}
    first=cache(tmp_path,p,shared_data=shared)
    first.get(request(100,1));first.freeze()
    before=p.count
    second=cache(tmp_path,p,shared_data=shared,readonly=True)
    assert second.path==first.path
    second.get(request(100,1));assert p.count==before
    with pytest.raises(RuntimeError,match='zero-BP'):
        second.get(request(101,1))
    second.close()

def test_donor_migration_maps_physical_ids_and_preserves_bits(tmp_path,monkeypatch):
    monkeypatch.delenv('TS3_GRADIENT_CACHE_ROOT',raising=False)
    p=Probe();old=cache(tmp_path/'old',p)
    expected=old.get(request(3,1));old.freeze()
    before=p.count
    new=cache(tmp_path/'new',p,shared_data={'different_pool':True},donors=[(old.path,{3:100},[])])
    np.testing.assert_array_equal(new.get(request(100,1)),expected)
    assert p.count==before and new.imported_windows==1
    assert ('source',3,1) not in new.rows
    new.freeze()
    bad=V7Cache(tmp_path/'bad',p,model_identity={'model_id':'changed','revision':'r'},
        data_fields={'pool':0},context_length=4,prediction_length=2,donors=[(old.path,{3:100},[])])
    assert not bad.donor_rows and bad.reuse_audit[0]['differing_fields']
    bad.close()

def test_assets_sparse_union_readonly_reconstruction(tmp_path,monkeypatch):
    monkeypatch.delenv('TS3_GRADIENT_CACHE_ROOT',raising=False)
    pool=[TimeSeriesFragment(np.sin(np.arange(150)*.1).astype(np.float32),fragment_id=100,entity_id='source')]
    refs=[TimeSeriesFragment(np.cos(np.arange(160)*.1).astype(np.float32),fragment_id=0,entity_id='ref')]
    idx=WindowIndex(pool,context_length=4,prediction_length=2)
    cfg=CRBConfig(context_length=4,prediction_length=2,cover_density=.005,error_density=.005,split_density=.005)
    p=Probe();kw=dict(probe=p,run_dir=tmp_path,pool=pool,index=idx,refs=refs,
        manifest={'total_equivalents':25},identity={'model_id':'fixture','revision':'r'},crb=cfg,task='test')
    assets,ri,audit=prepare(**kw)
    assert audit['complete'] and audit['new_bp']==p.count
    assert p.count < len(idx)+len(ri)
    for side in ['source','reference']:
        for row in audit['plans'][side]:
            assert row['gradients']==row['uniform_actual_bp']==sum(cfg.budgets(row['legal_windows']))
    count=p.count
    again,_,after=prepare(**kw,readonly=True)
    assert p.count==count and after['new_bp']==0

def test_barrier_and_checkpoint_selection(tmp_path):
    from rags.experiments.v7_train import choose_checkpoint
    rows=[dict(epoch=0,mase=3),dict(epoch=.5,mase=1),dict(epoch=1,mase=1),dict(epoch=4,mase=2)]
    assert choose_checkpoint(rows)['epoch']==.5
    with pytest.raises(ValueError):choose_checkpoint([dict(epoch=0,mase=float('nan'))])
    c={'seeds':[0],'datasets':['t']}
    freeze(tmp_path/'inputs'/'seed0_manifest.json',{'seed':0})
    with pytest.raises(RuntimeError,match='barrier'):
        gate(c,tmp_path,'assets')
    freeze(tmp_path/'f.json',{'seed':0})
    with pytest.raises(ValueError,match='identity'):
        freeze(tmp_path/'f.json',{'seed':1})

def test_analytic_scores_constant_ties_and_antipodal_cancellation():
    rng=np.random.default_rng(3)
    g=rng.normal(size=(1,64))
    constant=score_trajectory(301,[0,300],np.repeat(g,2,axis=0),g[0])
    for values in constant.values():
        assert len(np.unique(values))==1
    opposed=score_trajectory(301,[0,300],np.concatenate([g,-g]),g[0])
    assert opposed['grand-static'][150]==0 and opposed['grad-cos'][150]==0
    p=uniform_anchors(1001,17);g=rng.normal(size=(17,64));target=rng.normal(size=64)
    from rags.experiments.v7_scores import interpolated_chunks
    dense=np.concatenate([v for _,v in interpolated_chunks(1001,p,g)])
    scores=score_trajectory(1001,p,g,target)
    np.testing.assert_allclose(scores['grad-dot'],dense@target,atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(scores['grand-static'],np.linalg.norm(dense,axis=1),atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(scores['grad-cos'],dense@target/np.linalg.norm(dense,axis=1)/np.linalg.norm(target),atol=2e-13,rtol=2e-13)
