"""Independent point-budget, region identity and training-sampler oracles."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rags.core import TimeSeriesFragment
from rags.experiments.arms import ArmOutput, SelectionCost, WindowIndex, trajectory_level_select
from rags.experiments.regions import (RegionWindows, block_regions, deliver_regions,
                                     deliver_trajectories, load_selection, save_selection)


def index_fixture():
    return WindowIndex([TimeSeriesFragment(np.arange(30, dtype=np.float32), fragment_id=10),
                        TimeSeriesFragment(np.arange(23, dtype=np.float32) + 100, fragment_id=20)],
                       context_length=4, prediction_length=2)


def test_block_envelopes_budget_overlap_and_no_cross_boundary(tmp_path):
    index = index_fixture()
    blocks = [np.arange(1, 4), np.arange(3, 6), np.arange(25, 29)]
    regions = block_regions(index, blocks)
    assert regions == [(10, 1, 9), (10, 3, 11), (20, 0, 9)]
    selected = deliver_regions(index, regions, n=2)
    assert selected.regions.tolist() == [[10, 1, 9], [10, 3, 11]]
    stats = selected.coverage()
    assert stats == dict(budget_windows=2, budget_points=12, delivered_units=2,
        delivered_points=16, unique_points=10, repeated_points=6, overflow_points=4,
        overflow_ratio=4 / 12, legal_window_occurrences=6)
    oracle = np.stack([index.by_fragment[fid].data[start: start + 6]
        for fid, first, stop in selected.regions for start in range(first, stop - 6 + 1)])
    np.testing.assert_array_equal(selected[:], oracle)
    np.testing.assert_array_equal(selected[np.array([5, 0, 5])], oracle[[5, 0, 5]])
    small = selected.prefix(1)
    assert small.coverage()["delivered_points"] == 8
    assert small.budget_windows == 1
    path = tmp_path / "rags.npz"
    save_selection(path, ArmOutput("rags", selected, [], SelectionCost(), {"delivery": stats}))
    restored = load_selection(path)
    assert restored.diagnostics["delivery"] == stats
    np.testing.assert_array_equal(restored.selected[:], oracle)


def test_repeated_whole_trajectories_preserve_occurrence_weight():
    index = index_fixture()
    selected = deliver_trajectories(index, [10, 10, 20], n=13)
    assert selected.coverage()["delivered_points"] == 83
    assert selected.coverage()["unique_points"] == 53
    assert selected.coverage()["repeated_points"] == 30
    assert len(selected) == 25 + 25 + 18
    owners, starts = selected.identities(np.arange(len(selected)))
    assert np.bincount(owners).tolist() == [25, 25, 18]
    assert starts.max() == 24
    np.testing.assert_array_equal(selected[:25], selected[25:50])
    with pytest.raises(ValueError, match="point budget"):
        deliver_trajectories(index, [10], n=6)


def test_formal_width_crb_envelope():
    index = WindowIndex([TimeSeriesFragment(np.arange(1000, dtype=np.float32), fragment_id=4)],
                        context_length=512, prediction_length=64)
    assert block_regions(index, [np.arange(99, 120)]) == [(4, 99, 695)]
    selected = deliver_regions(index, [(4, 99, 695)], n=1)
    assert len(selected) == 21
    np.testing.assert_array_equal(selected[0], np.arange(99, 675))
    np.testing.assert_array_equal(selected[20], np.arange(119, 695))


def test_curve_region_cache_and_whole_unit_prefix(tmp_path, monkeypatch):
    from rags.experiments import curves
    index = index_fixture()
    selected = deliver_trajectories(index, [10, 20], n=8)
    arm = ArmOutput("random", selected, [], SelectionCost(), {})
    monkeypatch.setattr(curves, "streaming_random_select", lambda *a, **kw: arm)
    kwargs = dict(engine=None, task=None, pool=[], index=index, pool_manifest={},
                  args=SimpleNamespace(), n_max=8)
    curves._load_or_compute_selection(tmp_path, "random", **kwargs)
    monkeypatch.setattr(curves, "streaming_random_select", lambda *a, **kw: pytest.fail("cache missed"))
    restored = curves._load_or_compute_selection(tmp_path, "random", **kwargs)
    assert restored.selected.prefix(4).regions.tolist() == [[10, 0, 30]]
    assert restored.selected.prefix(6).regions.tolist() == [[10, 0, 30], [20, 0, 23]]
    assert restored.selected.prefix(6).budget_windows == 6


def test_retired_raw_reference_mean_delivery_is_rejected():
    index=index_fixture()
    with pytest.raises(RuntimeError,match="Retired hard trajectory delivery"):
        trajectory_level_select(list(index.by_fragment.values()),list(index.by_fragment.values()),
            index,engine=None,n_select=1,context_length=4,prediction_length=2)


@pytest.mark.parametrize("student", ["chronos-small", "timesfm-2.5", "timemoe-200m"])
def test_region_worker_sampling_and_epoch_budget(tmp_path, student):
    from test_multistudent import TinyAdapter, make_manifest
    from rags.experiments.students import adapter_engine, atomic_json, run_student_task
    path, manifest = make_manifest(tmp_path)
    # Two copies plus a second trajectory, all longer than one training window.
    values = np.r_[np.arange(12), np.arange(12), np.arange(10) + 20].astype(np.float32) / 100
    regions = RegionWindows(values, [0, 12, 24, 34], [[0, 0, 12], [0, 0, 12], [1, 0, 10]],
                            window_length=4, budget_windows=8)
    save_selection(tmp_path / "selection.npz", ArmOutput("tsds", regions, [], SelectionCost(), {}))
    manifest["selections"] = {"tsds": manifest["selections"]["random"]}
    atomic_json(path, manifest)
    engine = adapter_engine(TinyAdapter("timesfm-2.5"), 3, 1)
    seen = []
    loss = engine.training_loss
    def capture(windows):
        seen.append(windows.copy())
        return loss(windows)
    engine.training_loss = capture
    out = tmp_path / student
    run_student_task(student, path, out, engine=engine)
    result = json.loads((out / "cells" / "ETT_1H__tsds__seed0.json").read_text())
    assert [r["step"] for r in result["checkpoints"]] == [3, 6, 8, 11]
    assert result["selection_coverage"] == regions.coverage()
    assert result["train"]["source"]["sample_exposures"] == 33
    rng = np.random.default_rng(0)
    for batch in seen:
        np.testing.assert_array_equal(batch, regions[rng.integers(0, len(regions), size=3)])
    assert len(seen) == 11


def test_rags_max_score_block_delivery_matches_ablation_full():
    from rags.experiments.arms import rags_crb_select
    from rags.experiments.ablation import select_variant
    index = index_fixture()
    refs = [TimeSeriesFragment(np.arange(22, dtype=np.float32), fragment_id=77)]
    def field(fragment, starts):
        return np.stack([np.ones(len(starts)), np.asarray(starts) / 10], axis=1).astype(np.float32)
    args = SimpleNamespace(context_length=4, prediction_length=2, pool_stride=1,
        cover_density=.2, error_density=.1, split_density=.05, mean_epsilon=.001,
        endpoint_epsilon=.001, curvature_cap=1., block_budget=8, n_windows=4, window_entry_budget=None)
    pool = list(index.by_fragment.values())
    actual = rags_crb_select(refs, pool, index, gradient_computer=field, context_length=4,
        prediction_length=2, window_stride=1, cover_density=.2, error_density=.1,
        split_density=.05, block_budget=8, n_select=4, mean_epsilon=.001,
        endpoint_epsilon=.001, tie_seed=5, device="cpu")
    expected = select_variant("full", refs, pool, index, gradient_computer=field, args=args)
    np.testing.assert_array_equal(actual.selected.regions, expected.selected.regions)
    assert actual.selected.coverage() == expected.selected.coverage()
    assert actual.cost.gradient_computations == expected.cost.gradient_computations
