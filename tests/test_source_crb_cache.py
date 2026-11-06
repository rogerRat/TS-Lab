from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest

from assets.gradient.blocks import compute_pool_blocks
from assets.gradient.source_crb_cache import source_cache_directory
from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig
from rage.experiments.arms import WindowIndex


def same(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.fragment_id == b.fragment_id
        for key in ("start_positions", "mean_gradient", "direction"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))


@pytest.mark.parametrize("stride", [1, 3])
def test_roundtrip_restart_and_field_isolation(tmp_path, stride):
    config = CRBConfig(context_length=8, prediction_length=4,
        window_stride=stride, cover_density=.1, error_density=.05, split_density=.05)
    pool = [TimeSeriesFragment(np.sin(np.arange(600 + i * 70, dtype=np.float32)),
                               fragment_id=i + 4) for i in range(3)]
    pool.append(TimeSeriesFragment(np.ones(5, dtype=np.float32), fragment_id=99))
    index = WindowIndex(pool, context_length=8, prediction_length=4, stride=stride)
    calls = []
    def gradients(fragment, starts):
        calls.extend(starts.tolist())
        return np.stack([fragment.extract_window(int(p), 8, 4)[:7] for p in starts])
    cache = SimpleNamespace(path=tmp_path, manifest={"model": "fixture", "window": [8, 4]})
    directory = source_cache_directory(cache, config, pool, index)
    args = dict(config=config, gradient_computer=gradients, log_every=0, trajectory_batch_size=2)
    original_traj = dict(trajectories=[], counts=[])
    original, counters = compute_pool_blocks(pool, index, trajectory_assets=original_traj, **args)
    first_traj = dict(trajectories=[], counts=[])
    first, first_counters = compute_pool_blocks(pool, index, trajectory_assets=first_traj,
                                               source_cache_dir=directory, **args)
    same(original, first)
    same(original_traj["trajectories"], first_traj["trajectories"])
    assert counters == first_counters
    calls.clear()
    def forbidden(*args):
        pytest.fail("warm source cache must never request gradients")
    warm_traj = dict(trajectories=[], counts=[])
    warm, warm_counters = compute_pool_blocks(pool, index, trajectory_assets=warm_traj,
        source_cache_dir=directory, **dict(args, gradient_computer=forbidden))
    same(original, warm)
    same(original_traj["trajectories"], warm_traj["trajectories"])
    assert counters == warm_counters
    assert original_traj["counts"] == warm_traj["counts"]
    # An interrupted preparation leaves completed trajectories reusable.
    (directory / "trajectory-5.npz").unlink()
    partial, partial_counters = compute_pool_blocks(pool, index, source_cache_dir=directory, **args)
    same(original, partial)
    assert counters == partial_counters and calls
    assert source_cache_directory(cache, config, pool, index) == directory
    assert source_cache_directory(cache, replace(config, error_density=.06), pool, index) != directory
    assert source_cache_directory(SimpleNamespace(path=tmp_path, manifest={"model": "other"}),
                                   config, pool, index) != directory


def test_benchmark_warm_replay(tmp_path):
    config = CRBConfig(context_length=8, prediction_length=4,
                      cover_density=.02, error_density=.01, split_density=.01)
    pool = [TimeSeriesFragment(np.sin(np.arange(30000, dtype=np.float32)), fragment_id=i)
            for i in range(4)]
    index = WindowIndex(pool, context_length=8, prediction_length=4)
    directory = source_cache_directory(SimpleNamespace(path=tmp_path, manifest={}), config, pool, index)
    def gradients(fragment, starts):
        return np.stack([fragment.extract_window(int(p), 8, 4) for p in starts])
    args = dict(config=config, gradient_computer=gradients, log_every=0)
    start = time.perf_counter()
    baseline, counts = compute_pool_blocks(pool, index, **args)
    replay = time.perf_counter() - start
    compute_pool_blocks(pool, index, source_cache_dir=directory, **args)
    start = time.perf_counter()
    warm, warm_counts = compute_pool_blocks(pool, index, source_cache_dir=directory, **args)
    loaded = time.perf_counter() - start
    same(baseline, warm)
    assert counts == warm_counts
    print(f"source_crb CPU fixture replay={replay:.4f}s warm={loaded:.4f}s speedup={replay/loaded:.2f}x")
