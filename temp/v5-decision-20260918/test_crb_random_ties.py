"""Random choices affect exact priority ties, remain reproducible and batched."""
from dataclasses import replace
import numpy as np
import pytest
from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine, CRBEngineMulti


def simulate(config, gradients, fragment_id=0):
    n = len(gradients)
    fragment = TimeSeriesFragment(np.arange(n+2, dtype=np.float32), fragment_id=fragment_id)
    observed = []
    def field(_fragment, starts):
        ids = np.asarray(starts)-1
        observed.extend(ids.tolist())
        return gradients[ids]
    return CRBEngine(config, field).process_fragment(fragment), observed


def cfg(cover=3, error=0, split=0, seed=0):
    return CRBConfig(context_length=2, prediction_length=1,
                     cover_density=cover/17, error_density=error/17,
                     split_density=split/17, split_policy="closest_zero_v5",
                     priority_tie_policy="random", placement_seed=seed)


@pytest.mark.parametrize("stage", ["cover", "error", "split"])
def test_uniform_seeded_draw_from_current_exact_tie(stage):
    gradients = (np.arange(17, dtype=np.float32)[:, None]/16)**2
    config = cfg(cover=4) if stage == "cover" else cfg(error=1)
    stream = {"cover": 0, "error": 1, "split": 2}[stage]
    if stage == "split":
        config = cfg(split=1)
        gradients = np.full((17, 1), 4., dtype=np.float32)
        gradients[[0, 8, 16], 0] = [2., -1., 2.]
    selected = set()
    for seed in range(20):
        result, observed = simulate(replace(config, placement_seed=seed), gradients)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, stream]))
        expected = [4, 12][int(rng.choice(2, size=1, replace=False)[0])]
        assert observed == [0, 16, 8, expected]
        assert result.epsilon_audit["priority_tie_policy"] == "random"
        selected.add(observed[-1])
    assert selected == {4, 12}


def test_partial_coverage_frontier_is_subset_without_replacement():
    n = 33
    gradients = np.zeros((n, 1), dtype=np.float32)
    config = replace(cfg(), cover_density=8/n)
    selected = set()
    for seed in range(12):
        result, observed = simulate(replace(config, placement_seed=seed), gradients)
        assert len(observed) == len(set(observed)) == 8
        assert sorted(observed[:5]) == [0, 8, 16, 24, 32]
        assert set(observed[5:]).issubset({4, 12, 20, 28})
        selected.update(observed[5:])
    assert selected == {4, 12, 20, 28}


@pytest.mark.parametrize("seed", [0, 1, 9])
def test_priority_order_is_preserved_replay_and_multi_independent_of_order(seed):
    n = 65
    gradients = np.random.default_rng(42).normal(size=(n, 3)).astype(np.float32)
    config = replace(cfg(seed=seed), cover_density=8/n,
                     error_density=5/n, split_density=25/n)
    fragments = [TimeSeriesFragment(np.arange(n+2, dtype=np.float32), fragment_id=i)
                 for i in (2, 0, 1)]
    sequential = {f.fragment_id: simulate(config, gradients, f.fragment_id)[0]
                  for f in fragments}
    class Many:
        def compute_many(self, requests):
            return np.concatenate([gradients[np.asarray(s)-1] for f, s in requests])
    for ordered in (fragments, list(reversed(fragments))):
        results = CRBEngineMulti(config, Many()).process_fragments(ordered)
        for result in results:
            expected = sequential[result.fragment_id]
            assert sorted(result.anchors) == sorted(expected.anchors)
            assert result.epsilon_audit == expected.epsilon_audit
            assert np.concatenate([b.start_positions for b in result.blocks]).tolist() == result.valid_starts.tolist()
    first, ids1 = simulate(config, gradients)
    second, ids2 = simulate(config, gradients)
    assert ids1 == ids2
    # Independent replay of split decisions: only actual minimum-norm, eligible
    # intervals can be picked, irrespective of their tie ordering.
    phase12 = first.cover_computations+first.error_computations
    anchors = sorted(ids1[:phase12])
    intervals = list(zip(anchors[:-1], anchors[1:]))
    threshold = first.epsilon_audit["epsilon"]
    for midpoint in ids1[phase12:]:
        candidates = [(float(np.linalg.norm((gradients[l].astype(np.float64)+gradients[r])/2)), l, r)
                      for l, r in intervals if r-l > 1 and
                      max(np.linalg.norm(gradients[l].astype(np.float64)),
                          np.linalg.norm(gradients[r].astype(np.float64))) > threshold]
        minimum = min(x[0] for x in candidates)
        matching = [(l, r) for value, l, r in candidates
                    if value == minimum and (l+r)//2 == midpoint]
        assert len(matching) == 1
        l, r = matching[0]
        intervals.remove((l, r)); intervals.extend([(l, midpoint), (midpoint, r)])


def test_random_unknown_and_zero_risk_ties_do_not_fail():
    config = replace(cfg(cover=6, error=6), initial_placement="random")
    gradients = np.ones((17, 2), dtype=np.float32)
    result, observed = simulate(config, gradients)
    assert result.error_computations == 6
    assert len(observed) == len(set(observed)) == 12
    assert result.discarded_no_signal == result.banned_budget == 0


def test_right_policy_is_explicit_and_random_policy_validates():
    gradients = np.zeros((17, 1), dtype=np.float32)
    _, right = simulate(replace(cfg(cover=4), priority_tie_policy="right"), gradients)
    _, left = simulate(replace(cfg(cover=4), priority_tie_policy="left"), gradients)
    assert right == [0, 16, 8, 12]
    assert left == [0, 16, 8, 4]
    with pytest.raises(ValueError):
        replace(cfg(), priority_tie_policy="unknown")
