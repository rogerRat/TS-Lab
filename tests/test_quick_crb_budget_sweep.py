import pytest

from scripts.quick_crb_budget_sweep import (
    constrained_phase_allocation, feasible_allocations, reduced_ratio,
)


@pytest.mark.parametrize("total,expected", [
    (4, (2, 1, 1)),
    (9, (7, 1, 1)),
    (18, (15, 2, 1)),
])
def test_coverage_heavy_ratio_preserves_all_three_phases(total, expected):
    result = constrained_phase_allocation(total, (9, 1, 1))
    assert result == expected
    assert sum(result) == total
    assert result in feasible_allocations(total)


def test_standard_ratios_keep_original_integer_allocation():
    assert [constrained_phase_allocation(k, (1, 1, 1)) for k in (4, 9, 18)] == [
        (2, 1, 1), (3, 3, 3), (6, 6, 6)]
    assert [constrained_phase_allocation(k, (3, 1, 1)) for k in (4, 9, 18)] == [
        (2, 1, 1), (5, 2, 2), (11, 4, 3)]


def test_ratio_reduction_and_invalid_ratio():
    assert reduced_ratio((18, 2, 2)) == (9, 1, 1)
    with pytest.raises(ValueError):
        constrained_phase_allocation(9, (1, 0, 1))
