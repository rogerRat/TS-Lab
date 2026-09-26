import numpy as np
import pytest

from rags.experiments.probe_audit import balanced_reference_rows


def test_reference_sampling_balances_tasks_without_replacement():
    ids = [[f"a::{i}", 0, 576] for i in range(12)] + [[f"b::{i}", 0, 576] for i in range(4)]
    rows = balanced_reference_rows(ids, 8, np.random.default_rng(1))
    assert len(set(rows)) == 8
    assert sum(ids[i][0].startswith("a::") for i in rows) == 4
    np.testing.assert_array_equal(rows, balanced_reference_rows(ids, 8, np.random.default_rng(1)))
    with pytest.raises(ValueError, match="divisible"):
        balanced_reference_rows(ids, 7, np.random.default_rng(1))
