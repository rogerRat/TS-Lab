import numpy as np
import pytest

from rage.static_selection import selection_size, select_round_robin, select_topk


def test_selection_size_uses_ceiling_for_static_ratio():
    assert selection_size(11, ratio=0.2) == 3
    assert selection_size(11, count=4) == 4


def test_topk_is_stable_and_descending():
    np.testing.assert_array_equal(select_topk([2, 4, 4, 1], 3), [1, 2, 0])


def test_round_robin_balances_first_seen_groups():
    np.testing.assert_array_equal(select_round_robin(["a", "a", "b", "c", "b"], 4, [0.5, 0.9, 0.6, 0.1, 0.8]), [1, 4, 3, 0])


def test_selection_rejects_bad_inputs():
    with pytest.raises(ValueError):
        selection_size(10, ratio=0)
    with pytest.raises(ValueError):
        select_topk([1, np.nan], 1)
    with pytest.raises(ValueError):
        select_round_robin([0, 1], 1, [1])
