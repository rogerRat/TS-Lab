import pytest
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine
from rage.experiments.ablation import (
    RecallUnit,
    _streamed_top_b,
    equal_length_units,
    reconstruct_gradients,
    select_variant,
    variant_config,
    window_units,
)


def fragment_with_positions(count, *, context=2, horizon=1, stride=1, fragment_id=7):
    length = context + horizon + (count - 1) * stride
    return TimeSeriesFragment(np.arange(length, dtype=np.float32), fragment_id=fragment_id)


class Field:
    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, fragment, physical_starts):
        self.calls += len(physical_starts)
        return np.stack([np.asarray(self.function(int(v)), dtype=np.float32)
                         for v in physical_starts])


BASE = CRBConfig(context_length=2, prediction_length=1, window_stride=1,
                 cover_density=0.4, error_density=0.1, split_density=0.05,
                 mean_epsilon=1e-8, endpoint_epsilon=1e-8)


def test_variant_config_budget_and_flag_overrides():
    full = variant_config(BASE, "full")
    assert full == BASE

    no_split = variant_config(BASE, "no_split_detect")
    assert no_split.detection == "none"
    assert no_split.split_density == 0.0
    assert no_split.error_density == BASE.error_density + BASE.split_density

    drop_only = variant_config(BASE, "drop_only")
    assert drop_only.detection == "drop_only"
    assert drop_only.error_density == BASE.error_density + BASE.split_density

    assert variant_config(BASE, "no_error").error_placement == "uniform"
    with pytest.raises(ValueError,match="rightmost ties"):
        variant_config(BASE,"random_cover")
    assert variant_config(BASE, "fixed_blocks") == BASE
    assert variant_config(BASE, "grid_unif_nodet").detection == "none"
    assert variant_config(BASE, "grid_unif_det").error_placement == "uniform"


def test_uniform_error_placement_uses_longest_interval():
    fragment = fragment_with_positions(33)
    field = Field(lambda position: [float(position), 1.0])
    config = replace(BASE, error_placement="uniform", error_density=6 / 33,
                     split_density=0.0, mean_epsilon=0.0)
    result = CRBEngine(config, field).process_fragment(fragment)
    # cover 13 + error 6 anchors, all placed by longest-interval bisection
    anchors = sorted(result.anchors)
    assert result.error_computations == 6
    assert len(anchors) == 19
    assert max(b - a for a, b in zip(anchors, anchors[1:])) <= 4


def test_reconstruct_gradients_matches_anchor_values_and_midpoints():
    fragment = fragment_with_positions(9)
    field = Field(lambda position: [float(position), 2.0])
    config = replace(BASE, cover_density=4 / 9, error_density=0, split_density=0)
    result = CRBEngine(config, field).process_fragment(fragment)
    gradients = reconstruct_gradients(result)
    positions = sorted(result.anchors)
    assert gradients.shape == (9, 2)
    for row, position in enumerate(positions):
        assert np.allclose(gradients[position], result.anchors[position])
    # midpoint of two anchors is their average
    a, b = positions[0], positions[1]
    mid = (a + b) // 2
    expected = 0.5 * (result.anchors[a] + result.anchors[b])
    assert np.allclose(gradients[mid], expected)


def test_equal_length_units_block_count_and_detection():
    fragment = fragment_with_positions(17)
    field = Field(lambda position: [1.0, 1.0])
    config = replace(BASE, cover_density=5 / 17, error_density=0, split_density=0)
    result = CRBEngine(config, field).process_fragment(fragment)
    units, dropped = equal_length_units(result, mean_epsilon=1e-8,
                                         endpoint_epsilon=1e-8)
    assert len(units) + dropped == max(len(result.anchors) - 1, 1)
    covered = np.concatenate([u.starts for u in units]) if units else np.array([])
    assert len(set(covered.tolist())) == len(covered)  # disjoint

    # cancelling endpoints: mean ~ 0 -> dropped even though endpoints strong
    field_cancel = Field(lambda position: [1.0 if position < 8 else -1.0, 0.0])
    result_cancel = CRBEngine(config, field_cancel).process_fragment(fragment)
    _units, dropped_cancel = equal_length_units(
        result_cancel, mean_epsilon=0.5, endpoint_epsilon=1e-8)
    assert dropped_cancel >= 1


def test_window_units_skip_near_zero_windows():
    fragment = fragment_with_positions(9)

    def gradient(position):
        # a zero-gradient region between two nonzero regions
        return [0.0, 0.0] if 3 <= position <= 6 else [1.0, float(position)]

    field = Field(gradient)
    config = replace(BASE, cover_density=7 / 9, error_density=0, split_density=0)
    result = CRBEngine(config, field).process_fragment(fragment)
    units = window_units(result, mean_epsilon=1e-3)
    starts = {int(u.starts[0]) for u in units}
    # interior windows of the zero region are reconstructed as exact zeros
    assert len(units) < 9
    assert len(starts) == len(units)
    for unit in units:
        assert abs(float(np.linalg.norm(unit.direction)) - 1.0) < 1e-5


def test_streamed_top_b_matches_dense_result():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(6, 16)).astype(np.float32)
    units = [RecallUnit(i, np.array([i]), v)
             for i, v in enumerate(rng.normal(size=(500, 16)).astype(np.float32))]
    dense = np.stack([u.direction for u in units])
    take = 5
    values, indices = _streamed_top_b(reference, units, take, chunk=64)
    expected = np.argsort(-(reference @ dense.T), axis=1, kind="stable")[:, :take]
    assert np.array_equal(indices, expected)
    scores = np.take_along_axis(reference @ dense.T, expected, axis=1)
    assert np.allclose(values, scores, atol=1e-5)


def _fake_args(**overrides):
    values = dict(
        context_length=2, prediction_length=1, pool_stride=1,
        cover_density=0.4, error_density=0.1, split_density=0.05,
        mean_epsilon=1e-8, endpoint_epsilon=1e-8, curvature_cap=1.0,
        block_budget=3, window_entry_budget=None, n_windows=4,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_select_variant_window_return_and_full_shapes():
    from rage.experiments.arms import WindowIndex

    pool = [fragment_with_positions(9, fragment_id=1),
            fragment_with_positions(9, fragment_id=2)]
    reference = [fragment_with_positions(9, fragment_id=3)]
    index = WindowIndex(pool, context_length=2, prediction_length=1, stride=1)
    field = Field(lambda position: [1.0, float(position)])

    args = _fake_args()
    out = select_variant("full", reference, pool, index,
                         gradient_computer=field, args=args)
    assert out.selected.coverage()["delivered_points"] >= 12
    assert out.selected.coverage()["budget_windows"] == 4
    assert out.cost.gradient_computations == field.calls
    assert out.cost.index_bytes > 0

    field_w = Field(lambda position: [1.0, float(position)])
    out_w = select_variant("window_return", reference, pool, index,
                           gradient_computer=field_w, args=args)
    # window-return keeps every window as its own index entry
    assert out_w.cost.notes["candidate_units"] == 18
    assert "adjacent_duplicate_fraction" in out_w.cost.notes
    # deployment fills mutually non-overlapping windows (may fall short of N
    # when the returned entries cluster on one trajectory)
    assert 1 <= len(out_w.selected) <= 4
    pool_starts = {}
    for fid, start in out_w.window_ids:
        pool_starts.setdefault(fid, []).append(start)
    for rows in pool_starts.values():
        assert all(b - a >= 3 for a, b in zip(sorted(rows), sorted(rows)[1:]))


def test_select_variant_reuses_shared_crb_results():
    from rage.experiments.arms import WindowIndex

    pool = [fragment_with_positions(9, fragment_id=1)]
    reference = [fragment_with_positions(9, fragment_id=3)]
    index = WindowIndex(pool, context_length=2, prediction_length=1, stride=1)

    calls = {"n": 0}

    def computer(fragment, physical_starts):
        calls["n"] += len(physical_starts)
        return np.stack([np.asarray([1.0, float(v)], dtype=np.float32)
                         for v in physical_starts])

    args = _fake_args()
    cache: dict = {}
    select_variant("full", reference, pool, index, gradient_computer=computer,
                   args=args, crb_cache=cache)
    first = calls["n"]
    # fixed_blocks shares the full-method configuration -> no recomputation
    select_variant("fixed_blocks", reference, pool, index,
                   gradient_computer=computer, args=args, crb_cache=cache)
    assert calls["n"] == first
