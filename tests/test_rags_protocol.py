"""Tests for the doc-frozen protocol pieces: native exams (A4), full-history
scales (E1), disjoint deployment (A6/C2), trajectory objects (F2), budget
shares (C1), and the numpy WindowIndex."""

import numpy as np
import pytest

from rags.core import TimeSeriesFragment
from rags.experiments.data import native_test_rows
from rags.experiments.data import test_window_count as fev_test_window_count
from rags.experiments.deploy import cut_disjoint, disjoint_slots
from rags.experiments.metrics import (
    DATASET_SEASONALITY,
    history_seasonal_scale,
    mase_sql,
)
from rags.experiments.arms import (
    WindowIndex,
    random_select,
    trajectory_level_select,
)
from rags.experiments.mechanism1 import DEFAULT_SHARES


# ---------------------------------------------------------------------------
# A4: FEV-native exam windows
# ---------------------------------------------------------------------------

def test_native_test_rows_disjoint_tail_coverage():
    values = np.arange(2000, dtype=np.float64)
    rows, starts = native_test_rows(values, dataset="ETT_1H",
                                    context_length=512, prediction_length=64)
    assert len(rows) == 20 and len(starts) == 20
    # head-to-tail: each row's targets follow the previous row's targets
    for i in range(1, len(rows)):
        assert np.array_equal(rows[i - 1][512:] + 64, rows[i][512:])
    # targets cover exactly the last 20*64 points
    assert np.array_equal(rows[-1][512:], values[-64:])
    assert np.array_equal(rows[0][512:], values[-1280:-1216])
    # each row = [context | targets] contiguous slice of the series
    for row, start in zip(rows, starts):
        assert np.array_equal(np.concatenate([row[:512], row[512:]]),
                              values[start:start + 576])


def test_native_test_window_counts_per_dataset():
    assert fev_test_window_count("jena_weather_1H") == 10
    assert fev_test_window_count("boomlet_1676") == 20
    assert fev_test_window_count("ETT_15T") == 20
    assert fev_test_window_count("unknown_dataset") == 20


# ---------------------------------------------------------------------------
# E1: full-history seasonal scales and the FEV seasonality table
# ---------------------------------------------------------------------------

def test_seasonality_table_matches_fev_native():
    assert DATASET_SEASONALITY["boomlet_1282"] == 1440
    assert DATASET_SEASONALITY["boomlet_619"] == 1440
    assert DATASET_SEASONALITY["boomlet_1676"] == 48
    assert DATASET_SEASONALITY["ETT_15T"] == 96
    assert DATASET_SEASONALITY["ETT_1H"] == 24
    assert DATASET_SEASONALITY["jena_weather_1H"] == 24


def test_history_seasonal_scale_constant_and_sine():
    assert history_seasonal_scale(np.ones(100), 24) == 0.0
    t = np.arange(3000, dtype=np.float64)
    sine = np.sin(2 * np.pi * t / 24)
    assert history_seasonal_scale(sine, 24) < 1e-3
    # a pure trend has constant seasonal difference 24
    assert abs(history_seasonal_scale(t, 24) - 24.0) < 1e-9


def test_mase_sql_accepts_external_scales():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(4, 512 + 64))
    quantiles = np.repeat(windows[:, 512:, None], 9, axis=2)  # degenerate forecasts
    result = mase_sql(windows, quantiles, context_length=512,
                      seasonality=24, scales=np.full(4, 2.0))
    assert np.isfinite(result["mase"])
    assert result["sql"] >= 0.0


# ---------------------------------------------------------------------------
# A6/C2: disjoint deployment
# ---------------------------------------------------------------------------

def test_cut_disjoint_respects_interval_and_order():
    fids = np.asarray([1, 1, 1, 1, 2])
    starts = np.asarray([1, 2, 577, 578, 1])
    scores = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    chosen = cut_disjoint(fids, starts, scores, window_length=576, n=3)
    # start 2 blocked by start 1; start 578 blocked by 577; fragment 2 free
    assert list(chosen) == [0, 2, 4]


def test_cut_disjoint_fills_in_score_order_with_early_stop():
    fids = np.full(2000, 7)
    starts = np.arange(1, 2001)
    scores = np.linspace(0, 1, 2000)  # ascending: last is best
    chosen = cut_disjoint(fids, starts, scores, window_length=576, n=3)
    picked = sorted(starts[chosen].tolist())
    assert picked[-1] == 2000
    assert all(b - a >= 576 for a, b in zip(picked, picked[1:]))
    assert len(chosen) == 3


def test_disjoint_slots_grid():
    lengths = {1: 1200, 2: 576}
    fids, starts = disjoint_slots(lengths, window_length=576)
    assert list(fids) == [1, 1, 1, 2]
    assert list(starts) == [1, 577, 1153, 1]


# ---------------------------------------------------------------------------
# WindowIndex and F2 trajectory objects
# ---------------------------------------------------------------------------

def _pool(count=3, length=1200):
    return [TimeSeriesFragment(np.arange(length, dtype=np.float32) + i * 10_000,
                               fragment_id=i + 1) for i in range(count)]


def test_window_index_locate_and_extract_roundtrip():
    index = WindowIndex(_pool(), context_length=512, prediction_length=64,
                        stride=1)
    assert len(index) == 3 * (1200 - 575)
    position = len(index) // 2
    fid, start = index.fragment_arrays(np.asarray([position]))
    fid, start = int(fid[0]), int(start[0])
    assert index.locate(fid, start) == position
    window = index.windows_at(np.asarray([position]))[0]
    fragment = index.by_fragment[fid]
    assert np.array_equal(window, fragment.extract_window(
        start, 512, 64))


def test_random_select_returns_disjoint_windows():
    index = WindowIndex(_pool(), context_length=512, prediction_length=64)
    out = random_select(index, n_select=5, seed=0)
    assert len(out.selected) == 5
    fids, starts = index.fragment_arrays(
        np.asarray([index.locate(f, s) for f, s in out.window_ids]))
    per_fragment = {}
    for f, s in zip(fids.tolist(), starts.tolist()):
        per_fragment.setdefault(f, []).append(s)
    for rows in per_fragment.values():
        rows.sort()
        assert all(b - a >= 576 for a, b in zip(rows, rows[1:]))


def test_trajectory_object_spans():
    from types import SimpleNamespace
    from rags.experiments.ablation import select_uniform_segments
    fragment = TimeSeriesFragment(np.arange(1730, dtype=np.float32),
                                  fragment_id=1)  # 3 full 576 segments + 2 tail
    index = WindowIndex([fragment], context_length=512, prediction_length=64)
    out = select_uniform_segments([fragment], [fragment], index,
        gradient_computer=lambda f, s: np.ones((len(s), 4), dtype=np.float32),
        args=SimpleNamespace(context_length=512, prediction_length=64, n_windows=3))
    assert sorted(out.selected.regions.tolist()) == [[1, 0, 576], [1, 576, 1152], [1, 1152, 1728]]
    assert out.cost.gradient_computations == 6

def test_trajectory_level_select_with_stub_engine():
    class StubEngine:
        device = "cpu"
        class config:
            projection_dimension = 4

        def batch_gradient_sketch(self, windows):
            values = np.asarray(windows, dtype=np.float32)
            sketches = np.stack([values.mean(axis=1)] * 4, axis=1)
            return np.zeros(len(values)), sketches

    pool = _pool(count=2, length=1200)
    reference = [TimeSeriesFragment(np.arange(1200, dtype=np.float32) + 99_999,
                                    fragment_id=50)]
    index = WindowIndex(pool, context_length=512, prediction_length=64)
    with pytest.raises(RuntimeError,match="Retired hard trajectory delivery"):
        trajectory_level_select(reference, pool, index,
                                  engine=StubEngine(), n_select=3,
                                  context_length=512, prediction_length=64)


# ---------------------------------------------------------------------------
# C1: frozen 4:4:1 shares
# ---------------------------------------------------------------------------

def test_default_shares_are_4_4_1():
    assert pytest.approx(sum(DEFAULT_SHARES)) == 1.0
    assert pytest.approx(DEFAULT_SHARES[0]) == 4 / 9
    assert pytest.approx(DEFAULT_SHARES[1]) == 4 / 9
    assert pytest.approx(DEFAULT_SHARES[2]) == 1 / 9
    # reproduces the frozen densities at the 4.5% total density
    total = 0.045
    assert pytest.approx(total * DEFAULT_SHARES[0], abs=1e-9) == 0.02
    assert pytest.approx(total * DEFAULT_SHARES[2], abs=1e-9) == 0.005
