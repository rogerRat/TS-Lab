"""Coverage-then-cancellation selection contract tests."""
import numpy as np
import pytest

from rage.ads import uniform_anchor_positions
from rage.crb_select import (
    CRBSplitConfig,
    interval_kappa,
    select_crb_positions,
    selection_audit,
)


def test_interval_kappa_is_scale_invariant():
    left = np.array([2.0, 0.0])
    right = np.array([-1.0, 0.0])
    for scale in (1e-30, 1.0, 1e30):
        assert interval_kappa(left * scale, right * scale) == pytest.approx(1 / 3)


@pytest.mark.parametrize("left,right,expected", [
    ([1.0, 0.0], [-1.0, 0.0], 0.0),
    ([1.0, 0.0], [1.0, 0.0], 1.0),
    ([1.0, 0.0], [0.0, 0.0], 1.0),
    ([0.0, 0.0], [0.0, 0.0], 1.0),
    ([100.0, 0.0], [-1.0, 0.0], 99 / 101),
])
def test_interval_kappa_values(left, right, expected):
    assert interval_kappa(np.array(left), np.array(right)) == pytest.approx(expected)


def test_cover_positions_follow_uniform_anchor_rule():
    gradients = np.zeros((101, 4), dtype=np.float32)
    config = CRBSplitConfig(q=8, cover_spacing_ratio=1.5, split_fraction=0.5)
    result = select_crb_positions(gradients, config)
    assert result.cover_spacing == pytest.approx(12.0)
    assert result.cover_positions[0] == 0
    assert result.cover_positions[-1] == 100
    assert len(result.cover_positions) == len(uniform_anchor_positions(101, 12.0))


def test_split_budget_is_capped_and_uses_strongest_intervals():
    count = 97
    gradients = np.zeros((count, 2), dtype=np.float32)
    gradients[:, 0] = 1.0
    config = CRBSplitConfig(q=8, cover_spacing_ratio=1.5, split_fraction=0.5)
    result = select_crb_positions(gradients, config)
    assert result.split_budget == int(np.floor(0.5 * result.interval_count))
    assert result.split_used == result.split_budget
    split_kappas = [record.kappa for record in result.intervals if record.split]
    kept_kappas = [record.kappa for record in result.intervals if not record.split]
    if split_kappas and kept_kappas:
        assert max(split_kappas) <= min(kept_kappas) + 1e-12


def test_opposing_endpoints_rank_before_agreeing_endpoints():
    count = 481
    gradients = np.full((count, 2), [1.0, 0.0], dtype=np.float32)
    config = CRBSplitConfig(q=16, cover_spacing_ratio=1.5, split_fraction=0.5)
    result = select_crb_positions(gradients, config)
    agree = select_crb_positions(gradients, config)
    assert np.array_equal(result.anchor_positions, agree.anchor_positions)
    assert result.split_budget >= 1
    for record in result.intervals:
        assert record.kappa == pytest.approx(1.0)

    cancelling = gradients.copy()
    for index in range(0, count, 2):
        cancelling[index] = np.array([-1.0, 0.0], dtype=np.float32)
    mixed = select_crb_positions(cancelling, config)
    assert mixed.split_budget == result.split_budget
    records = sorted(mixed.intervals, key=lambda record: record.rank)
    split_records = [record for record in records if record.split]
    assert split_records
    kept_records = [record for record in records if not record.split]
    assert kept_records
    assert max(record.kappa for record in split_records) <= min(
        record.kappa for record in kept_records) + 1e-12
    assert max(record.kappa for record in split_records) < 1.0


def test_deterministic_selection_and_tie_break_rule():
    gradients = np.full((65, 2), [1.0, 0.0], dtype=np.float32)
    config = CRBSplitConfig(q=4, cover_spacing_ratio=1.5, split_fraction=0.5)
    first = select_crb_positions(gradients, config)
    second = select_crb_positions(gradients, config)
    assert np.array_equal(first.anchor_positions, second.anchor_positions)
    assert sorted(record.rank for record in first.intervals) == list(range(first.interval_count))
    ordered = sorted(first.intervals, key=lambda record: record.rank)
    for previous, current in zip(ordered, ordered[1:]):
        assert previous.width >= current.width
        if previous.width == current.width:
            assert previous.left <= current.left


def test_anchors_are_unique_and_no_interval_split_twice():
    gradients = np.random.default_rng(0).normal(size=(201, 5)).astype(np.float32)
    config = CRBSplitConfig(q=8, cover_spacing_ratio=1.5, split_fraction=0.5)
    result = select_crb_positions(gradients, config)
    anchors = result.anchor_positions
    assert len(anchors) == len(np.unique(anchors))
    assert np.array_equal(anchors, np.sort(anchors))
    assert len(result.split_positions) == len(np.unique(result.split_positions))
    covered = {int(value) for value in result.cover_positions}
    for value in result.split_positions:
        assert int(value) not in covered
    split_intervals = [record for record in result.intervals if record.split]
    assert len(split_intervals) == result.split_used
    for record in split_intervals:
        assert record.left < record.midpoint < record.right
    for record in result.intervals:
        if not record.split:
            assert record.midpoint is None


def test_equivalent_spacing_tracks_target_q():
    gradients = np.random.default_rng(1).normal(size=(3201, 8)).astype(np.float32)
    config = CRBSplitConfig(q=32, cover_spacing_ratio=1.5, split_fraction=0.5)
    result = select_crb_positions(gradients, config)
    assert result.equivalent_spacing == pytest.approx(32.0, rel=0.06)


def test_audit_is_json_safe_and_ranks_cancellation():
    gradients = np.random.default_rng(2).normal(size=(513, 4)).astype(np.float32)
    config = CRBSplitConfig(q=8)
    result = select_crb_positions(gradients, config)
    audit = selection_audit(result, config)
    assert audit["cover_anchor_count"] == len(result.cover_positions)
    assert audit["split_used"] == len(result.split_positions)
    assert audit["total_anchor_count"] == len(result.anchor_positions)
    assert len(audit["interval_rank_order"]) == result.interval_count
    assert audit["mean_kappa_split"] <= audit["mean_kappa_kept"]


def test_single_anchor_trajectory():
    gradients = np.array([[1.0, 0.0]], dtype=np.float32)
    result = select_crb_positions(gradients, CRBSplitConfig(q=8))
    assert result.total_anchor_count == 1
    assert result.split_used == 0
    assert result.anchor_positions.tolist() == [0]


def test_unsplittable_interval_is_never_charged():
    gradients = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    config = CRBSplitConfig(q=1, cover_spacing_ratio=1.5, split_fraction=1.0)
    result = select_crb_positions(gradients, config)
    assert result.interval_count == 1
    assert result.split_budget == 1
    assert result.split_used == 0
    assert result.skipped_unsplittable == 1
    assert result.anchor_positions.tolist() == [0, 1]


@pytest.mark.parametrize("ratio,fraction", [(1.5, 0.5), (2.0, 1 / 3), (2.0, 0.5)])
def test_budget_semantics_are_reported(ratio, fraction):
    gradients = np.random.default_rng(3).normal(size=(1025, 6)).astype(np.float32)
    config = CRBSplitConfig(q=32, cover_spacing_ratio=ratio, split_fraction=fraction)
    result = select_crb_positions(gradients, config)
    assert result.cover_spacing == pytest.approx(ratio * 32)
    assert result.split_budget == min(result.interval_count,
                                      int(np.floor(fraction * result.interval_count)))
