"""Budget cap and recommendation decisions require no model or GPU."""
from decimal import Decimal
from itertools import product

import pytest

from rage.experiments.anchor_budget_simulate import (
    GRID, budget_combinations, choose_budget, run_self_test,
)


def rows(point, means, anchors, split=0):
    return [dict(cover=point[0], error=point[1], split=point[2], task=f"task{i}",
        mean_cos_block=mean, pass_block=mean >= .90, windows=100,
        anchors_total=anchors, split_used_total=split) for i, mean in enumerate(means)]


def test_grid_preserves_low_points_and_adds_finite_boundary_points():
    combos = budget_combinations(GRID)
    points = {tuple(combo[name] for name in ("cover", "error", "split")) for combo in combos}
    old = set(product(GRID["cover"], GRID["error"], GRID["split"]))
    assert old <= points
    assert len(points - old) == 9
    totals = [sum(Decimal(str(value)) for value in point) for point in points]
    assert max(totals) == Decimal(".30")
    assert len(budget_combinations(dict(cover=[.1], error=[.1], split=[.1]))) == 1


@pytest.mark.parametrize("point", [(.3, .001, 0), (.30000001, 0, 0),
    (-.01, 0, 0), (float("nan"), 0, 0), (float("inf"), 0, 0)])
def test_invalid_budget_is_rejected(point):
    with pytest.raises(ValueError):
        budget_combinations(dict(zip(("cover", "error", "split"), ([v] for v in point))))


def test_passing_choice_minimizes_actual_anchors_not_nominal_density():
    candidates = rows((.1, 0, 0), [.95] * 5, 15) + rows((.15, 0, 0), [.91] * 5, 12)
    chosen, best = choose_budget(candidates, [f"task{i}" for i in range(5)])
    assert chosen["cover"] == .15
    assert chosen["total_unique_anchors"] == 60
    assert chosen["actual_anchor_ratio"] == .12
    assert chosen["pass_block"] is True
    assert chosen["split_identifiable"] is False
    assert best is None


def test_best_effort_prioritizes_worst_task_then_macro_mean_then_actual_cost():
    candidates = rows((.1, 0, 0), [.79, .99, .99, .99, .99], 5)
    candidates += rows((.15, 0, 0), [.8, .81, .81, .81, .81], 5)
    candidates += rows((.2, 0, 0), [.8, .82, .82, .82, .82], 20)
    candidates += rows((.25, 0, 0), [.8, .82, .82, .82, .82], 18)
    chosen, best = choose_budget(candidates, [f"task{i}" for i in range(5)])
    assert chosen is None
    assert best["cover"] == .25
    assert best["pass_block"] is False
    assert best["worst_task_mean_cosine"] == .8
    assert len(best["task_mean_cosines"]) == 5


def test_partial_candidate_cannot_receive_recommendation():
    assert choose_budget(rows((.1, 0, 0), [.99] * 4, 10),
                         [f"task{i}" for i in range(5)]) == (None, None)


def test_rounding_over_actual_cap_is_not_recommended_even_if_cosine_passes():
    candidates = rows((.30, 0, 0), [.99] * 5, 31)
    candidates += rows((.25, 0, 0), [.85] * 5, 25)
    chosen, best = choose_budget(candidates, [f"task{i}" for i in range(5)])
    assert chosen is None
    assert best["cover"] == .25
    assert best["actual_budget_usable"]


def test_actual_budget_boundary_is_allowed():
    chosen, best = choose_budget(rows((.30, 0, 0), [.99] * 5, 30),
                                [f"task{i}" for i in range(5)])
    assert chosen["actual_anchor_ratio"] == .30
    assert best is None


def test_retired_replay_protocol_is_rejected(tmp_path):
    import pytest
    with pytest.raises(RuntimeError,match='Retired block-constant budget protocol'):
        run_self_test(tmp_path,log=lambda *args:None)
