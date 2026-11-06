import numpy as np
import pytest

from scripts.quick_crb_ratio_ref import TruthComputer, phase_allocation
from rage.core import TimeSeriesFragment
from rage.experiments.arms import WindowIndex


@pytest.mark.parametrize("total,expected", [
    (4, (2, 1, 1)),
    (9, (5, 2, 2)),
    (18, (11, 4, 3)),
])
def test_phase_allocation_keeps_exact_total(total, expected):
    actual = phase_allocation(total)
    assert actual == expected
    assert sum(actual) == total


def test_truth_computer_returns_only_requested_physical_starts():
    fragment = TimeSeriesFragment(np.arange(8, dtype=np.float32),
                                  entity_id="synthetic", fragment_id=7)
    index = WindowIndex([fragment], context_length=2, prediction_length=1)
    truth = np.arange(len(index) * 3, dtype=np.float32).reshape(len(index), 3)
    computer = TruthComputer(index, truth)
    starts = fragment.valid_starts(2, 1, 1)
    np.testing.assert_array_equal(computer(fragment, starts[[0, -1]]), truth[[0, -1]])
    with pytest.raises(KeyError):
        computer(fragment, np.array([int(starts[-1]) + 1]))
