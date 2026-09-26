import json
import numpy as np
import pytest

from test_full_pool_worker import fixture
from rags.experiments.full_pool_assets import run_assets
from rags.experiments.full_pool_worker import run_worker
from rags.experiments.students import atomic_json
from assets.gradient.prepare import prepare_gradients
from rags.crb import CRBConfig


def test_reference_assets_never_evaluate_or_train_and_resume(tmp_path,monkeypatch):
    config,bundle=fixture(tmp_path)
    (tmp_path/'inputs'/'tiny__exam.npz').unlink()
    def forbidden(*args,**kwargs):
        raise AssertionError('assets-only must never evaluate/train')
    monkeypatch.setattr('rags.experiments.full_pool_worker.evaluate',forbidden)
    monkeypatch.setattr('rags.experiments.full_pool_worker.run_arm_training',forbidden)
    before={k:v.clone() for k,v in bundle[0].model.state_dict().items()}
    out=run_assets(tmp_path,'chronos-base',model_bundle=bundle,reference_batch=2)
    root=out/'reference_assets'/'tiny'
    assert len(list(root.glob('trajectory-*.npz')))==8
    assert json.loads((root/'complete.json').read_text())['complete']
    assert not (out/'cells').exists()
    assert not (out/'selections').exists()
    assert all((before[k]==v).all() for k,v in bundle[0].model.state_dict().items())
    records={p.name:p.read_bytes() for p in root.glob('trajectory-*.npz')}
    def no_gradient(*args,**kwargs):
        raise AssertionError('completed reference assets should be reused')
    bundle[1].batch_gradient_sketch=no_gradient
    run_assets(tmp_path,'chronos-base',model_bundle=bundle,reference_batch=3)
    assert records=={p.name:p.read_bytes() for p in root.glob('trajectory-*.npz')}
    assert json.loads((root/'complete.json').read_text())['new_windows']==0


def test_execution_control_blocks_full_worker_before_loading(tmp_path,monkeypatch):
    config,bundle=fixture(tmp_path)
    atomic_json(tmp_path/'execution_control.json',dict(mode='assets_only'))
    def forbidden(*args,**kwargs):
        raise AssertionError('paused worker cannot load model')
    monkeypatch.setattr('rags.experiments.full_pool_worker.build_model',forbidden)
    with pytest.raises(RuntimeError,match='training and evaluation are paused'):
        run_worker(tmp_path,'chronos-base')


def test_batched_reference_union_matches_production_requirements(tmp_path):
    from rags.experiments.development_inputs import load_fragments
    from rags.experiments.arms import WindowIndex
    from assets.gradient.cache import GradientCache
    config,bundle=fixture(tmp_path)
    out=run_assets(tmp_path,'chronos-base',model_bundle=bundle,reference_batch=2)
    inputs=tmp_path/'inputs'
    manifest=json.loads((inputs/'pool_manifest.json').read_text())
    pool=load_fragments(inputs/'pool.npz')
    refs=load_fragments(inputs/'tiny__reference.npz')
    cache=GradientCache(tmp_path,bundle[1],model_identity=bundle[2],data_fields=dict(pool=manifest),
                        context_length=4,prediction_length=2)
    expected={key for key in cache.rows if key[0]=='reference:tiny'}
    cache.close()
    assets=prepare_gradients(['rags','less'],bundle[1],run_dir=tmp_path/'comparison',
        reference_fragments=refs,pool=pool,index=WindowIndex(pool,context_length=4,prediction_length=2),
        pool_manifest=manifest,model_identity=bundle[2],
        crb_config=CRBConfig(context_length=4,prediction_length=2,**config['crb']),
        dataset_tag='tiny')
    actual={key for key in assets.cache.rows if key[0]=='reference:tiny'}
    assert expected==actual
    for result in assets.reference_results:
        with np.load(out/'reference_assets'/'tiny'/f'trajectory-{result.fragment_id}.npz') as saved:
            np.testing.assert_array_equal(saved['valid_starts'],result.valid_starts)
            np.testing.assert_allclose(saved['block_directions'],np.stack([b.direction for b in result.blocks]))
