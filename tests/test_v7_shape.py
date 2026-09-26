import numpy as np
import pytest

from rags.experiments.v7_shape import (
    ARMS, endpoint_positions, feature_cache, integrated_squared_distance,
    reference_weights, scores_many, shape_rows,
)
from rags.crb import GradientBlock


def test_integrated_squared_distance_matches_quadrature():
    left = np.array([[1., 2.], [2., -1.]])
    right = np.array([[3., 1.], [-2., 3.]])
    features = shape_rows(left, right)
    energy = np.sum(((left+right)/2)**2, axis=1) + np.sum((right-left)**2, axis=1)/12
    u = np.linspace(-.5, .5, 100001)
    curves = [(left[i]+right[i])/2 + u[:, None]*(right[i]-left[i]) for i in range(2)]
    normalized = [curves[i]/np.sqrt(energy[i]) for i in range(2)]
    numeric = np.trapezoid(np.sum((normalized[0]-normalized[1])**2, axis=1), u)
    assert integrated_squared_distance(features[0], features[1]) == pytest.approx(numeric, abs=1e-6)


def test_shape_reduces_to_midpoint_cosine_and_preserves_slope():
    constant = shape_rows(np.array([[1., 0.], [0., 1.]]),
                          np.array([[1., 0.], [0., 1.]]))
    assert constant[0] @ constant[1] == pytest.approx(0)
    rising = shape_rows(np.array([[-1., 0.], [1., 0.]]),
                        np.array([[1., 0.], [-1., 0.]]))
    assert rising[0] @ rising[1] == pytest.approx(-1)
    assert integrated_squared_distance(rising[0], rising[1]) == pytest.approx(4)
    zero = shape_rows(np.zeros((1,2)), np.zeros((1,2)))
    assert np.array_equal(zero, np.zeros((1,4)))


def test_arm_priority_is_exp_then_linear_then_controls():
    assert ARMS == ('rags-shape-expSum', 'rags-shape-linearSum',
                    'rags-shape-zeroClipSum', 'rags-shape-maxSum')


def test_filtered_shape_worker_still_rejects_unknown_arm(tmp_path):
    from rags.experiments.v7_shape_backfill import run_model
    with pytest.raises(ValueError, match='Unknown shape arm'):
        run_model(tmp_path, [], [], 'chronos-base', 'wrong',
                  seed_filter=1, task_filter={'time_coastal_ts_h'})


def test_first_receipts_create_missing_sidecar_parents(tmp_path):
    from rags.experiments.v7_shape_backfill import publish_receipt
    import json
    for section in ('shape_select', 'shape_complete', 'arm_phases'):
        path = tmp_path/'fresh_run'/section/'first.json'
        assert not path.parent.exists()
        publish_receipt(path, {'complete':True, 'section':section})
        assert json.loads(path.read_text()) == {'complete':True, 'section':section}


def test_global_arm_barrier_requires_every_model_seed_task(tmp_path,monkeypatch):
    import json
    from scripts import run_v7_shape_backfill as queue
    from rags.experiments.v7_protocol import MODELS
    from rags.experiments.v7_run import artifact_identity
    monkeypatch.setattr(queue,'report_view',lambda raw,expected,config,model: raw
                        if raw.get('identity') == expected else (_ for _ in ()).throw(ValueError()))
    parent=tmp_path/'run'
    config={'datasets':['task']}
    side_config={'asset_protocol':{'experiment_version':'v7'},
                 'training_protocol':{'version':'reduced'},'shape_amendment':{'version':'shape'}}
    sidecar=parent/'shape_backfill_20260925'
    sidecar.mkdir(parents=True)
    (sidecar/'shape_config.json').write_text(json.dumps(side_config))
    for model in MODELS:
        path=parent/'models'/f'{model}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'model':model}))
        for seed in (0,1):
            manifest={'total_equivalents':16284,'seed':seed}
            path=parent/'inputs'/f'seed{seed}_manifest.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(manifest))
            expected=dict(artifact_identity(side_config,model,seed,'task',manifest,{'model':model}),
                          training_protocol=side_config['training_protocol'])
            path=sidecar/'seeds'/f'seed{seed}'/model/'cells'/'task__rags-shape-expSum.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(identity=expected,arm='rags-shape-expSum',
                                            seed=seed,task='task')))
    assert queue.arm_complete([parent],[config],ARMS[0])
    receipt = queue.accept_arm_phase(parent,[parent],[config],ARMS[0])
    assert receipt['cells'] == 6
    marker = parent/'shape_backfill_20260925'/'arm_phases'/f'{ARMS[0]}.json'
    assert json.loads(marker.read_text())['complete'] is True
    assert queue.accept_arm_phase(parent,[parent],[config],ARMS[0]) == receipt
    assert not queue.arm_complete([parent],[config],ARMS[1])
    path.unlink()
    assert not queue.arm_complete([parent],[config],ARMS[0])


def test_reference_weights_and_physical_endpoints():
    class Index:
        def bounds(self, fid):
            return (0, 4) if fid == 7 else (4, 6)
    class Result:
        def __init__(self, fid, n):
            self.fragment_id = fid
            self.valid_starts = np.arange(n)
    def block(fid, left, right, owned):
        return GradientBlock(fid, left, right, np.arange(owned), np.zeros(2), np.zeros(2))
    blocks = [block(7,0,2,2), block(7,2,3,2), block(8,0,1,2)]
    assert np.array_equal(endpoint_positions(blocks, Index()), np.array([[0,2],[2,3],[4,5]]))
    assert np.allclose(reference_weights(blocks,[Result(7,4),Result(8,2)]), [.25,.25,.5])


def test_shape_cache_reuses_exact_endpoint_rows_without_bp(tmp_path):
    class Index:
        def bounds(self, fid):
            assert fid == 7
            return 0, 2
        def fragment_arrays(self, positions):
            return np.full(len(positions),7), np.asarray(positions)+1
    class Cache:
        manifest = {'model':{'projection_dimension':4096}}
        def __init__(self):
            self.reads = 0
        def get(self, request):
            self.reads += 1
            rows=np.zeros((len(request.ids),4096),dtype=np.float32)
            for i, (fid,start) in enumerate(request.ids):
                rows[i,0 if start == 1 else 1]=1.
            return rows
    block = GradientBlock(7,0,1,np.array([0,1]),np.zeros(2),np.zeros(2))
    cache = Cache()
    first, receipt = feature_cache(tmp_path/'features',cache=cache,dataset='source',
                                   index=Index(),blocks=[block],fields={'pool':'stable'})
    assert receipt['cached'] is False and receipt['new_bp'] == 0
    assert np.asarray(first).shape == (1,8192)
    assert cache.reads == 2
    second, receipt = feature_cache(tmp_path/'features',cache=cache,dataset='source',
                                    index=Index(),blocks=[block],fields={'pool':'stable'})
    assert receipt['cached'] is True and cache.reads == 2
    assert np.array_equal(first,second)
    with pytest.raises(ValueError,match='manifest'):
        feature_cache(tmp_path/'features',cache=cache,dataset='source',
                      index=Index(),blocks=[block],fields={'pool':'changed'})


def test_four_aggregations_keep_existing_reference_weight_placement():
    refs = np.array([[1.,0.],[0.,1.]],dtype=np.float32)
    source = np.array([[1.,0.]],dtype=np.float32)
    results = scores_many(refs,source,np.array([.75,.25]))
    assert results['linearSum'][0][0] == pytest.approx(.75)
    assert results['zeroClipSum'][0][0] == pytest.approx(.75)
    assert results['expSum'][0][0] == pytest.approx(.75*np.e+.25)
    assert results['maxSum'][0][0] == pytest.approx(.75)


def test_shape_sidecar_identity_is_new_and_parent_untouched(monkeypatch,tmp_path):
    from rags.experiments import v7_shape_backfill, v7_run
    monkeypatch.setattr(v7_shape_backfill,'validate',lambda config:None)
    parent={'asset_protocol':{'experiment_version':'v7'},'seeds':[0,1,2],
            'arms':['orig','rand','rags-block-expSum'],'datasets':['task'],
            'training_protocol':{'version':'v7-seven-tasks-chronos3-other4-20260924'}}
    shape=v7_shape_backfill.sidecar_config(parent,tmp_path)
    assert parent['seeds'] == [0,1,2]
    assert shape['seeds'] == [0,1]
    assert shape['arms'][2] == 'rags-shape-expSum'
    parent_identity=v7_run.artifact_identity(parent,'chronos-base',0,'task',{},'model')
    shape_identity=v7_run.artifact_identity(shape,'chronos-base',0,'task',{},'model')
    assert 'shape_amendment' not in parent_identity
    assert shape_identity['shape_amendment']['version'] == 'v7-rags-block-shape-integral-v1'


def test_shape_queue_requires_all_original_and_extension_cells(tmp_path):
    import json
    from scripts.run_v7_shape_backfill import complete
    from rags.experiments.v7_protocol import MODELS
    runs=[tmp_path]+[tmp_path/'extensions'/name for name in ('wind','sg','coastal')]
    configs=[{'datasets':['task'],'arms':['orig','rand']}]
    assert not complete(tmp_path,runs,configs)
    for model in MODELS:
        for seed in (0,1):
            root=tmp_path/f'seeds/seed{seed}'/model/'cells'
            root.mkdir(parents=True)
            for arm in ('orig','rand'):
                (root/f'task__{arm}.json').write_text('{}')
    assert not complete(tmp_path,runs,configs)
    for run in runs[1:]:
        run.mkdir(parents=True)
        (run/'task_queue_complete.json').write_text(json.dumps(
            dict(complete=True,cells=120,seeds=[0,1])))
    assert complete(tmp_path,runs,configs)
    (runs[-1]/'task_queue_complete.json').write_text(json.dumps(
        dict(complete=True,cells=119,seeds=[0,1])))
    assert not complete(tmp_path,runs,configs)


def test_baseline_reuse_relabels_verified_result_without_inference(tmp_path):
    import json
    from rags.experiments import v7_shape_backfill, v7_run
    parent=tmp_path/'parent'; sidecar=tmp_path/'shape'
    model='chronos-base'; task='task'; manifest={'total_equivalents':16284}; identity={'model_id':'test'}
    training={'version':'v7-seven-tasks-chronos3-other4-20260924'}
    base={'asset_protocol':{'experiment_version':'v7'},'training_protocol':training,
          'settings':{model:{'epochs':3}},'datasets':[task],'arms':['orig','rand']}
    shape=dict(base,shape_amendment={'version':'new'})
    old=dict(v7_run.artifact_identity(base,model,0,task,manifest,identity),training_protocol=training)
    root=parent/'seeds/seed0'/model/'cells'; root.mkdir(parents=True)
    (root/'task__orig.json').write_text(json.dumps(dict(identity=old,arm='orig',mase=2.,mae=2.,mse=2.)))
    curve=[dict(epoch=i/2,step=i,metrics={'mase':1.,'mae':1.,'mse':1.}) for i in range(1,7)]
    (root/'task__rand.json').write_text(json.dumps(dict(identity=old,arm='rand',mase=1.,final_epoch=3,
        checkpoints=curve,test_curve=[dict(epoch=0.,step=0,metrics={'mase':2.})]+curve)))
    v7_shape_backfill.copy_baselines(parent,sidecar,base,shape,model,0,task,manifest,identity)
    copied=json.loads((sidecar/'seeds/seed0'/model/'cells/task__rand.json').read_text())
    assert copied['mase'] == 1.
    assert copied['reused_from'] == str(root/'task__rand.json')
    assert copied['identity']['shape_amendment'] == {'version':'new'}
