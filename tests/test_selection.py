from dataclasses import replace,asdict
import json
import numpy as np
import pytest
from test_full_pool_resampling import Probe,inputs
from assets.gradient.prepare import prepare_gradients
from rage.experiments.full_pool import select_full_pool,ARM_BASE,validate_cached_selection
from rage.experiments.ranking import rank_mass,DEFAULT_TEMPERATURE,WINDOW_UNIFORMIZATION_REVISION
from rage.experiments.reference_opinions import OPINION_ARMS, UNI_ARMS, WIN_UNI_ARMS
from rage.experiments.regions import save_selection,load_selection

METHODS=('less-main',)+OPINION_ARMS

@pytest.fixture
def prepared(tmp_path):
    kw=inputs();kw['crb_config']=replace(kw['crb_config'],cover_density=.3,error_density=.1,split_density=.2)
    probe=Probe()
    assets=prepare_gradients(['less','rage','rage_traj'],probe,run_dir=tmp_path,
        reference_fragments=kw['reference_fragments'],pool=kw['pool'],index=kw['index'],
        pool_manifest=kw['pool_manifest'],model_identity=kw['model_identity'],crb_config=kw['crb_config'],dataset_tag='tiny')
    def forbidden(*args,**kwargs):raise AssertionError('No BP after preparation')
    probe.batch_gradient_sketch=forbidden
    return kw,probe,assets

@pytest.mark.parametrize('name',METHODS)
def test_selection_scores_rank_volume_save_draw_and_no_extra_bp(name,prepared,tmp_path):
    kw,probe,assets=prepared
    out=select_full_pool(name,probe,**kw,gradient_assets=assets,dataset_tag='tiny')
    selected=out.selected;meta=selected.metadata
    if name in UNI_ARMS:
        block_trajectories=np.array([int(np.unique(
            kw['index'].fragment_ids[np.asarray(b.start_positions,dtype=np.int64)])[0])
            for b in assets.source_blocks])
        _,inverse=np.unique(block_trajectories,return_inverse=True)
        if name in WIN_UNI_ARMS:
            block_counts=np.asarray([len(block.start_positions) for block in assets.source_blocks])
            _rank,block_mass,_audit=rank_mass(meta['native_scores'],block_counts)
            expected=np.bincount(inverse,weights=block_mass)
        else:
            counts=np.diff(selected.unit_offsets)
            _rank,expected,_audit=rank_mass(meta['native_scores'],counts)
            totals=np.bincount(inverse,weights=expected);sizes=np.bincount(inverse)
            expected=totals[inverse]/sizes[inverse]
        assert selected.metadata['intra_trajectory_uniformization']==(WINDOW_UNIFORMIZATION_REVISION if name in WIN_UNI_ARMS else 'rage-intra-trajectory-uniform-v1')
    else:
        counts=np.diff(selected.unit_offsets)
        _rank,expected,_audit=rank_mass(meta['native_scores'],counts)
    np.testing.assert_allclose(selected.probabilities,expected,atol=1e-15)
    assert len(selected.unit_positions)==len(kw['index'])
    assert len(np.unique(selected.unit_positions))==len(kw['index'])
    assert out.cost.gradient_computations==0
    assert meta['frozen_rank_protocol']['post_rank_softmax'] is False
    validate_cached_selection(meta,arm=name,identity=kw['model_identity'],manifest=kw['pool_manifest'],crb=kw['crb_config'],temperature=DEFAULT_TEMPERATURE)
    path=tmp_path/(name+'.npz');save_selection(path,out);restored=load_selection(path).selected
    np.testing.assert_array_equal(restored.probabilities,selected.probabilities)
    np.testing.assert_array_equal(restored.sample_indices(np.random.default_rng(3),2000),selected.sample_indices(np.random.default_rng(3),2000))
    assert selected.probabilities.sum()==pytest.approx(1.)
    if name.startswith('rage-block'):
        weights=meta['reference_audit']['reference_weights'];refs=assets.reference_blocks
        for result in assets.reference_results:
            assert sum(w for w,b in zip(weights,refs) if b.fragment_id==result.fragment_id)==pytest.approx(1/len(assets.reference_results))

@pytest.mark.parametrize('name',METHODS)
def test_missing_assets_error_without_bp(name):
    kw=inputs();probe=Probe()
    def forbidden(*args,**kwargs):raise AssertionError('No hidden BP')
    probe.batch_gradient_sketch=forbidden
    with pytest.raises(ValueError,match='Prepared gradients required'):select_full_pool(name,probe,**kw)

@pytest.mark.parametrize('key,value',[('source_weighting','equal_block'),('crb',{}),('frozen_rank_protocol',None),('opinion_aggregation','expSum')])
def test_resume_rejects_old_or_incompatible_probabilities(key,value,prepared):
    kw,probe,assets=prepared;name='rage-block-maxSum'
    meta=select_full_pool(name,probe,**kw,gradient_assets=assets).selected.metadata.copy();meta[key]=value
    with pytest.raises(ValueError):validate_cached_selection(meta,arm=name,identity=kw['model_identity'],manifest=kw['pool_manifest'],crb=kw['crb_config'],temperature=DEFAULT_TEMPERATURE)


def test_all_gradient_arms_worker_training_evaluation_and_completed_resume(tmp_path,monkeypatch):
    from test_full_pool_worker import fixture
    from rage.experiments.full_pool_worker import run_worker
    from rage.experiments.protocol import method_protocols
    from rage.experiments.students import atomic_json
    config,bundle=fixture(tmp_path)
    config.update(arms=list(METHODS),seeds=[0],method_protocols=method_protocols(METHODS))
    config['crb']=dict(cover_density=.3,error_density=.1,split_density=.1)
    atomic_json(tmp_path/'protocol.json',config)
    out=run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    paths=sorted((out/'cells').glob('*.json'));assert len(paths)==len(METHODS)+1
    before={p.name:p.read_bytes() for p in paths}
    def forbidden(*args,**kwargs):raise AssertionError('Resume must reuse completed cells and prepared distributions')
    bundle[1].batch_gradient_sketch=forbidden
    monkeypatch.setattr('rage.experiments.full_pool_worker.run_arm_training',forbidden)
    run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    assert before=={p.name:p.read_bytes() for p in paths}
    for p in (out/'selections').glob('*.npz'):
        selected=load_selection(p).selected
        assert selected.metadata['frozen_rank_protocol']['additional_bp_allowed'] is False
        assert selected.metadata['source_weighting']=='legal_window_counts'


def test_worker_rejects_copied_historical_completed_cell(tmp_path):
    from test_full_pool_worker import fixture
    from rage.experiments.full_pool_worker import run_worker
    from rage.experiments.students import atomic_json
    config,bundle=fixture(tmp_path);out=run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    path=out/'cells'/'tiny__rand__seed0.json';record=json.loads(path.read_text())
    record['experiment_version']='v4';atomic_json(path,record)
    with pytest.raises(ValueError,match='Completed cell identity differs'):
        run_worker(tmp_path,'chronos-base',model_bundle=bundle)
