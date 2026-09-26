from types import SimpleNamespace
import numpy as np
import pytest

from rags.experiments.exact_pool import ExactEquivalents, build_exact_pool


def test_exact_not_greedy_overshoot_or_trajectory_reuse():
    search = ExactEquivalents(10)
    assert not search.add(0, 8)
    assert not search.add(1, 6)
    assert search.add(2, 4)
    assert search.selected_rows() == [1, 2]
    search = ExactEquivalents(8)
    assert not search.add(0, 4)  # one length-4 row cannot be used twice
    with pytest.raises(ValueError, match="greatest reachable N=4"):
        search.selected_rows()


def test_formal_16284_not_16384():
    search = ExactEquivalents(16284)
    assert not search.add(0, 16384)
    assert not search.add(1, 16000)
    assert search.add(2, 284)
    assert search.selected_rows() == [1, 2]


def test_real_builder_manifest_counts_complete_trajectories(monkeypatch, tmp_path):
    from rags.experiments import data
    arrays = [np.arange(n, dtype=np.float32) for n in [8 * 6 + 3, 6 * 6 + 2, 4 * 6 + 1]]
    bundle = SimpleNamespace(series_count=lambda: 3, meta={"series_id": ["a", "b", "c"]})
    monkeypatch.setattr(data, "load_bundle_index", lambda path: bundle)
    monkeypatch.setattr(data, "_read_series", lambda bundle, row: arrays[row])
    pool, manifest = build_exact_pool(load_dir=tmp_path, equivalents=10, window_length=6)
    assert manifest["total_equivalents"] == 10
    assert set(manifest["rows"]) == {1, 2}
    assert sorted(f.length for f in pool) == [25, 38]
    assert manifest["total_points"] == 63
    assert manifest["total_legal_windows"] == 53
    for fragment, row in zip(pool, manifest["rows"]):
        np.testing.assert_array_equal(fragment.data, arrays[row])
