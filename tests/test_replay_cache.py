"""Replay contract tests: cached-store reads, exact-budget layouts, metrics."""

import json
from pathlib import Path

import numpy as np
import pytest

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine
from rage.experiments.v7_scores import uniform_anchors
from rage.replay_cache import (
    CacheGradientComputer,
    ProjectedGradientStore,
    anchor_budget,
    bisection_plan,
    compare_layouts,
    reconstruction_metrics,
    uniform_plan,
)


def make_store(root: Path, rows: int = 153, dimensions: int = 8, seed: int = 0) -> ProjectedGradientStore:
    root.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(seed).normal(size=(rows, dimensions)).astype(np.float16)
    np.save(root / "gradient_sketches.npy", values)
    (root / "manifest.json").write_text(json.dumps({
        "window_start_count": rows, "subset": "synthetic", "series_id": "S0",
    }), encoding="ascii")
    (root / ".complete").write_text("complete\n", encoding="ascii")
    return ProjectedGradientStore(root)


def test_store_requires_completion_marker(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    np.save(root / "gradient_sketches.npy", np.zeros((4, 2), dtype=np.float16))
    (root / "manifest.json").write_text(json.dumps({"window_start_count": 4}), encoding="ascii")
    with pytest.raises(FileNotFoundError):
        ProjectedGradientStore(root)


def test_store_rejects_manifest_count_mismatch(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    np.save(root / "gradient_sketches.npy", np.zeros((4, 2), dtype=np.float16))
    (root / "manifest.json").write_text(json.dumps({"window_start_count": 5}), encoding="ascii")
    (root / ".complete").write_text("complete\n", encoding="ascii")
    with pytest.raises(ValueError):
        ProjectedGradientStore(root)


def test_store_reads_exact_rows_only(tmp_path):
    store = make_store(tmp_path / "store")
    rows = np.array([0, 5, 152])
    values = store.read(rows)
    np.testing.assert_allclose(values, np.asarray(store.gradients[rows], dtype=np.float32))
    with pytest.raises(KeyError):
        store.read(np.array([153]))
    with pytest.raises(KeyError):
        store.read(np.array([-1]))


def test_anchor_budget_matches_equivalent_spacing():
    assert anchor_budget(230161, 4) == 57541
    assert anchor_budget(230161, 32) == 7193
    assert anchor_budget(153, 128) == 2
    with pytest.raises(ValueError):
        anchor_budget(10, 0)
    with pytest.raises(ValueError):
        anchor_budget(10, float("nan"))


def test_bisection_plan_equals_crb_coverage_stage(tmp_path):
    store = make_store(tmp_path / "store", rows=3201, dimensions=4)
    anchors, served = bisection_plan(store, 32, context_length=8, prediction_length=4)
    budget = anchor_budget(store.rows, 32)
    # Reference: the coverage stage alone with a synthetic 1-based start domain.
    gradients = np.asarray(store.gradients, dtype=np.float32)
    starts = np.arange(1, store.rows + 1, dtype=np.int64)
    fragment = TimeSeriesFragment(np.zeros(store.rows + 11, dtype=np.float32), fragment_id=0)

    def computer(_fragment, physical):
        return gradients[np.asarray(physical, dtype=np.int64) - 1]

    config = CRBConfig(cover_density=float(budget) / store.rows, error_density=0.0,
                       split_density=0.0, detection="none", context_length=8,
                       prediction_length=4)
    expected = np.asarray(sorted(CRBEngine(config, computer).process_fragment(fragment).anchors),
                          dtype=np.int64)
    assert np.array_equal(anchors, expected)
    assert served == budget
    assert len(np.unique(anchors)) == budget
    assert anchors[0] == 0 and anchors[-1] == store.rows - 1


def test_uniform_plan_matches_frozen_linspace_rule(tmp_path):
    store = make_store(tmp_path / "store", rows=230161, dimensions=2)
    anchors = uniform_plan(store, 32)
    budget = anchor_budget(store.rows, 32)
    assert np.array_equal(anchors, uniform_anchors(store.rows, budget))
    assert len(anchors) == budget


def test_reconstruction_is_exact_at_anchor_rows(tmp_path):
    store = make_store(tmp_path / "store", rows=257, dimensions=6, seed=3)
    anchors, _ = bisection_plan(store, 16, context_length=8, prediction_length=4)
    result = reconstruction_metrics(store, anchors, chunk_rows=64)
    np.testing.assert_allclose(result["cosine"][anchors], 1.0, atol=1e-6)
    np.testing.assert_allclose(result["relative_l2"][anchors], 0.0, atol=1e-6)
    summary = result["summary"]
    assert summary["anchor_count"] == anchor_budget(store.rows, 16)
    assert summary["windows"] == 257
    assert summary["non_anchor_windows"] == 257 - summary["anchor_count"]
    assert 0.0 <= summary["fraction_cosine_ge_threshold"] <= 1.0


def test_bisection_halves_gaps_evenly_on_power_of_two_length(tmp_path):
    store = make_store(tmp_path / "store", rows=65, dimensions=3)
    anchors, _ = bisection_plan(store, 8, context_length=8, prediction_length=4)
    assert anchor_budget(65, 8) == 9
    assert anchors.tolist() == [0, 8, 16, 24, 32, 40, 48, 56, 64]
    result = reconstruction_metrics(store, anchors)
    assert result["summary"]["gap_min"] == 8 and result["summary"]["gap_max"] == 8


def test_cache_computer_serves_requested_starts(tmp_path):
    store = make_store(tmp_path / "store", rows=20, dimensions=3)
    computer = CacheGradientComputer(store)
    fragment = TimeSeriesFragment(np.zeros(20 + 11, dtype=np.float32), fragment_id=0)
    values = computer(fragment, np.array([1, 4, 20], dtype=np.int64))
    np.testing.assert_allclose(values, np.asarray(store.gradients[[0, 3, 19]], dtype=np.float32))
    assert computer.served_rows == 3
    with pytest.raises(KeyError):
        computer(fragment, np.array([21], dtype=np.int64))


def test_compare_layouts_reports_paired_deltas(tmp_path):
    store = make_store(tmp_path / "store", rows=1025, dimensions=6, seed=7)
    bisect = reconstruction_metrics(store, bisection_plan(store, 32, context_length=8,
                                                          prediction_length=4)[0])
    uniform = reconstruction_metrics(store, uniform_plan(store, 32))
    comparison = compare_layouts(bisect, uniform)
    assert comparison["window_cosine_wins_bisect"] + \
        comparison["window_cosine_wins_uniform"] + comparison["window_cosine_ties"] == 1025
    assert comparison["cosine_delta_bisect_minus_uniform"] == pytest.approx(
        comparison["mean_cosine_bisect"] - comparison["mean_cosine_uniform"])
    assert comparison["anchor_count"] == anchor_budget(1025, 32)
