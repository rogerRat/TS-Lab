from types import SimpleNamespace

import numpy as np
import pytest
import torch

from assets.gradient import GradientCache, WindowRequest, FragmentGradients, GradientDataset
from assets.gradient.prepare import prepare_gradients
from rags.crb import CRBConfig, CRBEngineMulti
from rags.core import TimeSeriesFragment
from rags.execution import SelectionExecution
from rags.experiments.arms import WindowIndex
from rags.experiments.full_pool import select_full_pool


class Probe:
    device = "cpu"
    execution = SelectionExecution(gradient_batch=8)
    config = SimpleNamespace(projection_dimension=2, projection_seed=42)

    def __init__(self):
        self.count = 0

    def batch_gradient_sketch(self, windows):
        self.count += len(windows)
        return windows.mean(axis=1), np.column_stack((windows[:, 0], windows[:, -1]))


def cache_at(path, probe, **model):
    return GradientCache(path, probe, model_identity=dict(model_id="fixture", ckpt="step-1", **model),
        data_fields={"pool": "v1"}, context_length=4, prediction_length=2)


def test_union_order_restart_and_bf16_storage(tmp_path):
    probe = Probe()
    windows = np.arange(30, dtype=np.float32).reshape(5, 6) / 7
    def request(rows, dataset="source"):
        rows = np.array(rows)
        return WindowRequest(dataset, np.column_stack([np.zeros(len(rows)), rows + 1]),
                             lambda positions: windows[rows[positions]])
    cache = cache_at(tmp_path, probe)
    cache.prepare([request([0, 1, 2]), request([2, 3, 1])])
    assert probe.count == 4
    expected = torch.from_numpy(windows[[3, 1, 3]][:, [0, -1]].copy()).bfloat16().float().numpy()
    np.testing.assert_array_equal(cache.get(request([3, 1, 3])), expected)
    cache.prepare([request([0], "reference:task")])
    assert probe.count == 5
    with np.load(cache.shards[0]) as data:
        assert data["gradients_bf16"].dtype == np.uint16
        assert data["window_ids"].shape == (4, 2)
    cache.freeze()
    dataset = GradientDataset(cache.path)
    np.testing.assert_array_equal(dataset.lookup("source", [[0, 4], [0, 2], [0, 4]]), expected)
    assert dataset.manifest["model"]["ckpt"] == "step-1"
    assert len(dataset) == 5
    with pytest.raises(RuntimeError, match="unprepared"):
        cache.get(request([4]))
    second = cache_at(tmp_path, probe)
    second.prepare([request([0, 4])])
    assert probe.count == 6
    assert second.manifest["model"]["ckpt"] == "step-1"
    assert second.path.parent.name == "grad_cache"
    second.close()


def test_field_isolation_and_writer_lock(tmp_path):
    cache = cache_at(tmp_path, Probe(), precision_policy="bf16")
    with pytest.raises(RuntimeError, match="another process"):
        cache_at(tmp_path, Probe(), precision_policy="bf16")
    cache.close()
    with pytest.warns(UserWarning, match="fields differ"):
        other = cache_at(tmp_path, Probe(), precision_policy="fp32")
    assert other.path != cache.path
    other.close()


def test_crb_keeps_real_fragment_ids_and_reuses_all_probes(tmp_path):
    fragments = [TimeSeriesFragment(np.arange(24, dtype=np.float32) + i + 1,
                                    fragment_id=i + 7) for i in range(3)]
    config = CRBConfig(context_length=4, prediction_length=2)
    probe = Probe()
    cache = cache_at(tmp_path, probe)
    crb = CRBEngineMulti(config, FragmentGradients(cache, "source"))
    first = crb.process_fragments(fragments)
    count = probe.count
    assert {key[1] for key in cache.rows} == {7, 8, 9}
    second = crb.process_fragments(fragments)
    assert count == probe.count
    for left, right in zip(first, second):
        for a, b in zip(left.blocks, right.blocks):
            np.testing.assert_array_equal(a.direction, b.direction)
    cache.close()


def test_all_gradient_methods_score_without_computation_and_reuse_new_temperature(tmp_path):
    pool = [TimeSeriesFragment(np.linspace(1, 3, 30, dtype=np.float32), fragment_id=1)]
    refs = [TimeSeriesFragment(np.linspace(2, 4, 30, dtype=np.float32), fragment_id=1)]
    index = WindowIndex(pool, context_length=4, prediction_length=2)
    config = CRBConfig(context_length=4, prediction_length=2)
    probe = Probe()
    kwargs = dict(reference_fragments=refs, pool=pool, index=index, pool_manifest={"total_equivalents": 5},
                  model_identity={"model_id": "fixture", "ckpt": "base"}, crb_config=config, dataset_tag="task")
    names = ["rags", "less", "rags_traj"]
    assets = prepare_gradients(names, probe, run_dir=tmp_path, **kwargs)
    count = probe.count
    from rags.experiments.ranking import DEFAULT_TEMPERATURE
    for name in ("less-main","rags-block-maxSum","rags-traj-maxSum"):
        for temperature in (DEFAULT_TEMPERATURE,):
            output = select_full_pool(name, None, run_dir=tmp_path, gradient_assets=assets,
                                      temperature=temperature, **kwargs)
            assert output.cost.gradient_computations == 0
            assert np.isclose(output.selected.window_probabilities().sum(), 1)
    assert probe.count == count
    other = prepare_gradients(names, probe, run_dir=tmp_path, **kwargs)
    assert probe.count == count
    assert other.cache.computed_windows == 0


def test_bf16_count_sketch_uses_bounded_float_accumulation():
    from gradient.count_sketch import add_count_sketch
    values = torch.arange(30, dtype=torch.bfloat16).reshape(2, 15)
    hashes = torch.arange(15) % 4
    signs = torch.tensor([1, -1, 1] * 5, dtype=torch.int8)
    actual = torch.zeros(2, 4)
    expected = torch.zeros_like(actual)
    expected.index_add_(1, hashes, values.float() * signs)
    add_count_sketch(actual, values, hashes, signs, fused=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_later_method_only_computes_delta_and_publishes_union(tmp_path):
    windows = np.arange(36, dtype=np.float32).reshape(6, 6)
    calls = []
    def requirement(rows):
        rows = np.asarray(rows)
        def load(selected):
            calls.append(len(selected))
            return windows[rows[selected]]
        return WindowRequest("source", np.column_stack([np.ones(len(rows)), rows + 1]), load)
    probe = Probe()
    first = cache_at(tmp_path, probe)
    first.prepare([requirement([0, 1, 2])])
    first.freeze()
    before = GradientDataset(first.path).lookup("source", [[1, 1], [1, 2], [1, 3]])
    second = cache_at(tmp_path, probe)
    second.prepare([requirement([2, 3, 4, 3])])
    assert second.path == first.path
    assert second.computed_windows == 2
    second.freeze()
    merged = GradientDataset(second.path)
    assert len(merged) == 5
    np.testing.assert_array_equal(merged.lookup("source", [[1, 1], [1, 2], [1, 3]]), before)
    np.testing.assert_array_equal(merged.lookup("source", [[1, 5], [1, 4]]), windows[[4, 3]][:, [0, -1]])
    # Vectorized loading calls the materializer once per requirement batch.
    assert calls == [3, 2]


def test_reopen_recovers_shards_after_last_published_index(tmp_path):
    probe = Probe()
    windows = np.arange(18, dtype=np.float32).reshape(3, 6)
    def request(rows):
        rows = np.asarray(rows)
        return WindowRequest("source", np.column_stack([np.ones(len(rows)), rows + 1]),
                             lambda positions: windows[rows[positions]])
    cache = cache_at(tmp_path, probe)
    cache.prepare([request([0])])
    cache._publish_index()
    snapshot = (cache.path / "window_index.npz").read_bytes()
    cache.prepare([request([1])])
    cache.close()
    # Simulate a completed atomic shard write followed by a crash before index publication.
    (cache.path / "window_index.npz").write_bytes(snapshot)
    recovered = cache_at(tmp_path, probe)
    recovered.prepare([request([0, 1, 2])])
    assert recovered.computed_windows == 1
    recovered.freeze()
    assert len(GradientDataset(recovered.path)) == 3


def test_v2_shared_cache_reuse_and_refuse_fresh_cache(tmp_path):
    from assets.gradient.cache import atomic_json
    old = tmp_path/'v1'
    new = tmp_path/'v2'
    new.mkdir()
    fragments = [TimeSeriesFragment(np.arange(24,dtype=np.float32),fragment_id=7)]
    index = WindowIndex(fragments,context_length=4,prediction_length=2)
    probe = Probe()
    kwargs = dict(reference_fragments=fragments,pool=fragments,index=index,
        pool_manifest={'seed':0},model_identity={'model_id':'fixture','ckpt':'step-1'},
        crb_config=CRBConfig(context_length=4,prediction_length=2),dataset_tag='tiny')
    prepared = prepare_gradients(['rags'],probe,run_dir=old,**kwargs)
    path = prepared.cache.path.resolve()
    prepared.cache.close()
    before = probe.count
    (new/'grad_cache').symlink_to(old/'grad_cache',target_is_directory=True)
    atomic_json(new/'gradient_reuse.json',{'experiment_version':'v2'})
    atomic_json(new/'warm_cache_verification.json',{'reports':[{'cache':str(path)}]})
    warm = prepare_gradients(['rags'],probe,run_dir=new,**kwargs)
    assert probe.count == before and warm.cache.computed_windows == 0
    warm.cache.close()
    kwargs['model_identity'] = {'model_id':'fixture','ckpt':'different'}
    with pytest.warns(UserWarning,match='fields differ'):
        with pytest.raises(ValueError,match='refusing full recomputation'):
            prepare_gradients(['rags'],probe,run_dir=new,**kwargs)
    assert probe.count == before
