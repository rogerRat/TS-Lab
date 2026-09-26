"""Independent reference checks for the integer-budget fast cosine search."""
from math import comb

import numpy as np
import pytest

from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine
from rags.experiments.anchor_budget_simulate import evaluate_reconstruction
from rags.experiments.anchor_budget_integer import mean_block_cosine, positive_integer_budgets


def compare(truth, **options):
    truth = np.asarray(truth, dtype=np.float32).astype(np.float64)
    fragment = TimeSeriesFragment(np.zeros(len(truth) + 5, dtype=np.float32),
                                  entity_id="synthetic", fragment_id=0)
    defaults = dict(context_length=4, prediction_length=2, cover_density=.15,
                    error_density=.05, split_density=.05)
    defaults.update(options)
    config = CRBConfig(**defaults)
    result = CRBEngine(config, lambda fragment, starts:
                      truth[np.asarray(starts, dtype=int) - 1].astype(np.float32)).process_fragment(fragment)
    norms = np.linalg.norm(truth, axis=1)
    expected = evaluate_reconstruction(truth, norms, result)
    actual = mean_block_cosine(truth, norms, result)
    assert actual == pytest.approx(float(expected["block_cos"].mean()), abs=2e-12)
    return actual, expected, result


@pytest.mark.parametrize("seed", range(5))
def test_random_gradients_match_reference(seed):
    truth = np.random.default_rng(seed).normal(size=(71, 7))
    compare(truth)


def test_smooth_directions_and_nonuniform_norms_match_reference():
    x = np.arange(83)
    truth = np.column_stack((np.ones(len(x)), np.sin(x / 20), np.cos(x / 15)))
    truth *= np.geomspace(.01, 100, len(x))[:, None]
    actual, metrics, result = compare(truth)
    assert result.blocks
    assert actual > 0


def test_strong_cancellation_keeps_all_windows():
    truth = np.tile([1., 0., 0.], (40, 1))
    truth[20:] *= -1
    actual, metrics, result = compare(truth, cover_density=0, error_density=0,
                                      split_density=0, detection="drop_only")
    assert metrics["dropped_windows"] == 0
    assert actual == 0


def test_zero_truth_rows_are_not_removed_from_average():
    truth = np.tile([1., 2., 3.], (73, 1))
    truth[1::3] = 0
    actual, metrics, result = compare(truth, detection="none")
    assert metrics["zero_truth"] == 24
    assert actual <= (len(truth) - 24) / len(truth) + 1e-12


def test_zero_reconstruction_directions_match_reference():
    truth = np.tile([1., 0.], (40, 1))
    truth[-1] = [-1., 0.]
    actual, metrics, result = compare(truth, cover_density=0, error_density=0,
                                      split_density=0, detection="none")
    assert actual == 0


@pytest.mark.parametrize("truth,expected", [([[1., -2., 3.]], 1.), ([[0., 0., 0.]], 0.)])
def test_single_window_uses_its_anchor(truth, expected):
    actual, metrics, result = compare(truth)
    assert actual == pytest.approx(expected)
    assert metrics["single_window"]
    assert len(result.blocks)==1


def test_all_positive_integer_allocations_through_30_are_complete():
    count = 0
    for total in range(3, 31):
        allocations = list(positive_integer_budgets(total))
        expected = {(cover, error, total - cover - error)
                    for cover in range(1, total - 1)
                    for error in range(1, total - cover)}
        assert all(all(isinstance(value, int) and value >= 1 for value in row)
                   and sum(row) == total for row in allocations)
        assert len(allocations) == len(set(allocations)) == comb(total - 1, 2)
        assert set(allocations) == expected
        count += len(allocations)
    assert count == comb(30, 3) == 4060
