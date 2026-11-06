import json
import numpy as np
import pytest

from assets.gradient import WindowRequest, GradientDataset
from test_gradient_assets import Probe, cache_at
from rage.experiments.reference_opinions import reference_scores_many, AGGREGATIONS


def request(row, dataset='source'):
    return WindowRequest(dataset, [[7, row + 1]],
        lambda positions: np.full((len(positions), 6), row / 7, dtype=np.float32))


def test_sparse_shards_durable_coalesced_and_snapshot_sealed(tmp_path):
    probe = Probe()
    probe.last_gradient_execution = dict(backend='sequential', batch_size=1, oom_retries=0)
    cache = cache_at(tmp_path, probe)
    expected = []
    for row in range(130):
        result = cache.get(request(row))
        expected.append(result[0])
        # A reader of the latest atomic shard sees every completed round.
        path, location = cache.rows[('source', 7, row + 1)]
        with np.load(path) as data:
            assert tuple(data['window_ids'][location]) == (7, row + 1)
    assert len(cache.shards) == 3
    np.testing.assert_array_equal(cache.get(request(0))[0], expected[0])
    cache._publish_index()
    snapshot = GradientDataset(cache.path)
    cache.get(request(130))
    assert len(cache.shards) == 4  # a published tail is never overwritten
    assert len(snapshot) == 130
    np.testing.assert_array_equal(snapshot.lookup('source', [[7, 130]])[0], expected[129])
    cache.get(request(131, 'reference:x'))
    assert len(cache.shards) == 5
    # Simulate a crash without publishing the new index: recovery must find
    # all new rows from durable shards, even when several calls shared a tail.
    cache._lock.close()
    resumed = cache_at(tmp_path, probe)
    count = probe.count
    resumed.prepare([request(i) for i in range(131)])
    assert probe.count == count
    np.testing.assert_array_equal(resumed.get(request(129))[0], expected[129])
    resumed.close()


def test_failed_tail_publication_preserves_previous_rows(tmp_path, monkeypatch):
    cache = cache_at(tmp_path, Probe())
    cache.get(request(0))
    path = cache.shards[0]
    from pathlib import Path
    original = Path.replace
    def fail(self, target):
        if target == path:
            raise OSError('fixture atomic publication failure')
        return original(self, target)
    monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError, match='publication'):
        cache.prepare([request(1)])
    assert ('source', 7, 2) not in cache.rows
    with np.load(path) as data:
        np.testing.assert_array_equal(data['window_ids'], [[7, 1]])
    monkeypatch.setattr(Path, 'replace', original)
    cache.close()


def test_requested_and_effective_batches_are_separate(tmp_path):
    probe = Probe()
    probe.last_gradient_execution = dict(backend='sequential', batch_size=1, oom_retries=0)
    cache = cache_at(tmp_path, probe)
    cache.prepare([request(i) for i in range(8)])
    timing = cache.gradient_benchmark()
    assert timing['batch_sizes'] == {'8': 1}
    label = next(iter(timing['execution_batches']))
    assert json.loads(label)['batch_size'] == 1
    cache.close()


def test_native_oom_reduction_recovers_on_next_request(monkeypatch):
    import torch
    from test_batched_gradient import make_probe
    probe = make_probe()
    original = probe._batched_gradient_sketch
    failed = False
    def once(windows):
        nonlocal failed
        if len(windows) == 4 and not failed:
            failed = True
            raise torch.cuda.OutOfMemoryError('transient fixture')
        return original(windows)
    monkeypatch.setattr(probe, '_batched_gradient_sketch', once)
    windows = np.arange(12, dtype=np.float32).reshape(4, 3)
    probe.batch_gradient_sketch(windows, batch_size=4)
    assert probe.last_gradient_execution['batch_size'] == 2
    probe.batch_gradient_sketch(windows, batch_size=4)
    assert probe.last_gradient_execution['batch_size'] == 4
    assert probe.last_gradient_execution['oom_retries'] == 0


@pytest.mark.parametrize('modes', [AGGREGATIONS, ('expSum', 'linearSum'), ('maxSum',)])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_shared_cosines_bitwise_equal_legacy_tiled_reductions(modes, device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    rng = np.random.default_rng(713)
    refs = rng.normal(size=(37, 64))
    src = rng.normal(size=(103, 64))
    refs /= np.linalg.norm(refs, axis=1)[:, None]
    src /= np.linalg.norm(src, axis=1)[:, None]
    src[10] = src[9]  # exact rank ties must survive
    w = rng.uniform(.01, 1, len(refs)); w /= w.sum()
    source_gpu = torch.as_tensor(src, device='cuda') if device == 'cuda' else None
    expected = {}
    for mode in modes:
        scores = np.full(len(src), -np.inf if mode == 'maxSum' else 0., dtype=np.float64)
        for rb in range(0, len(refs), 7):
            for sb in range(0, len(src), 19):
                values = (refs[rb:rb+7] @ src[sb:sb+19].T if source_gpu is None else
                          (torch.as_tensor(refs[rb:rb+7], device='cuda') @ source_gpu[sb:sb+19].T).cpu().numpy())
                if mode == 'maxSum':
                    scores[sb:sb+19] = np.maximum(scores[sb:sb+19],
                        (w[rb:rb+7, None] * values).max(axis=0))
                else:
                    if mode == 'zeroClipSum': values = np.maximum(values, 0.)
                    elif mode == 'expSum': values = np.exp(values)
                    scores[sb:sb+19] += w[rb:rb+7] @ values
        expected[mode] = scores
    actual = reference_scores_many(refs, src, w, aggregations=modes,
                                   reference_batch=7, source_batch=19, device=device)
    assert set(actual) == set(modes)
    from rage.experiments.ranking import rank_mass
    counts = np.arange(len(src)) + 1
    for mode in modes:
        np.testing.assert_array_equal(actual[mode][0], expected[mode])
        np.testing.assert_array_equal(rank_mass(actual[mode][0], counts)[1],
                                      rank_mass(expected[mode], counts)[1])


def test_shared_arms_use_one_score_pass_and_preserve_distribution(tmp_path, monkeypatch):
    from assets.gradient.prepare import prepare_gradients
    from rage.core import TimeSeriesFragment
    from rage.crb import CRBConfig
    from rage.experiments.arms import WindowIndex
    from rage.experiments import full_pool
    pool = [TimeSeriesFragment(np.linspace(1, 3, 60, dtype=np.float32), fragment_id=1)]
    refs = [TimeSeriesFragment(np.linspace(2, 4, 60, dtype=np.float32), fragment_id=2)]
    index = WindowIndex(pool, context_length=4, prediction_length=2)
    kwargs = dict(reference_fragments=refs, pool=pool, index=index,
                  pool_manifest={'total_equivalents': 10}, model_identity={'model_id':'fixture'},
                  crb_config=CRBConfig(context_length=4, prediction_length=2), dataset_tag='one')
    assets = prepare_gradients(['rage'], Probe(), run_dir=tmp_path, **kwargs)
    arms = ['rage-block-' + mode for mode in AGGREGATIONS]
    expected = [full_pool.select_full_pool(arm, None, gradient_assets=assets, **kwargs) for arm in arms]
    assets.selection_aggregations['rage'] = list(AGGREGATIONS)
    calls = []
    implementation = full_pool.reference_scores_many
    def observed(*args, **kwargs):
        calls.append(kwargs['aggregations'])
        return implementation(*args, **kwargs)
    monkeypatch.setattr(full_pool, 'reference_scores_many', observed)
    for arm, original in zip(arms, expected):
        output = full_pool.select_full_pool(arm, None, gradient_assets=assets, **kwargs)
        np.testing.assert_array_equal(output.selected.probabilities, original.selected.probabilities)
        assert output.selected.metadata == original.selected.metadata
    assert len(calls) == 1
