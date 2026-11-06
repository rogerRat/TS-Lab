import numpy as np
import pytest
from rage.experiments.v7_run import native_cache_directory
from rage.experiments.arms import WindowIndex
from baselines.trajectory import trajectory_features, budget_config
from test_trajectory_baselines import Probe, fragments


def test_cross_task_source_feature_reuse_without_forward(tmp_path, monkeypatch):
    index = WindowIndex(fragments(), context_length=4, prediction_length=2)
    config = budget_config(index, .2, .1, .05)
    engine = Probe()
    manifest = {'seed': 0, 'total_points': 126}
    out = tmp_path/'seed0'/'model'
    tasks = ['first', 'second']
    first = native_cache_directory(out, tasks[0], 'tsds', tasks)
    monkeypatch.setattr('rage.execution.encoder_embeddings', lambda e, w, **kw: e.features(w))
    expected, cold = trajectory_features(engine,index,config,kind='tsds',
        cache_dir=first,pool_manifest=manifest)
    assert cold['executed_windows'] > 0
    def forbidden(*a, **kw):
        raise AssertionError('reused source features must not execute forward')
    monkeypatch.setattr('rage.execution.encoder_embeddings', forbidden)
    second = native_cache_directory(out, tasks[1], 'tsds', tasks)
    assert second == first
    actual, warm = trajectory_features(engine,index,config,kind='tsds',
        cache_dir=second,pool_manifest=manifest)
    np.testing.assert_array_equal(actual, expected)
    assert warm['pool_cached'] and warm['executed_windows'] == 0
    assert native_cache_directory(out,'second','less-main',tasks) != first
    assert native_cache_directory(tmp_path/'seed1'/'model','second','tsds',tasks) != first
    assert native_cache_directory(tmp_path/'seed0'/'other','second','tsds',tasks) != first
    with pytest.raises(ValueError, match='cache fields differ'):
        trajectory_features(engine,index,config,kind='tsds',cache_dir=second,
                            pool_manifest=dict(manifest,seed=1))
