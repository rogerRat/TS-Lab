"""Strict frozen-P10 recursive splitting; existing revisions stay unchanged."""
import numpy as np
import pytest
from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine, CRBEngineMulti


def run(gradients, cover=3, split=5, many=False):
    n = len(gradients)
    cfg = CRBConfig(context_length=2, prediction_length=1,
                    cover_density=cover/n, error_density=0, split_density=split/n,
                    split_policy="closest_zero_v5", epsilon_policy="fixed")
    fragments = [TimeSeriesFragment(np.arange(n+2, dtype=np.float32), fragment_id=i)
                 for i in range(2 if many else 1)]
    observed = []
    def field(fragment, starts):
        ids = np.asarray(starts)-1
        observed.extend((fragment.fragment_id, int(x)) for x in ids)
        return gradients[ids]
    class Many:
        def compute_many(self, requests):
            return np.concatenate([field(f, s) for f, s in requests])
    results = (CRBEngineMulti(cfg, Many()).process_fragments(fragments) if many
               else [CRBEngine(cfg, field).process_fragment(fragments[0])])
    return cfg, results, observed


@pytest.mark.parametrize("constant", [0., 2.])
def test_equal_endpoint_norms_are_excluded_strictly(constant):
    _, [result], observed = run(np.full((17, 2), constant, dtype=np.float32))
    assert result.split_computations == 0
    assert len(observed) == 3
    assert result.epsilon_audit["split_unused"] == 5
    assert result.epsilon_audit["split_stop_reason"] == "eligible_pool_exhausted"
    assert result.epsilon_audit["endpoint_comparator"] == "strict_gt"
    assert result.discarded_no_signal == result.discarded_antagonistic == result.banned_budget == 0


def test_recurses_freezes_threshold_and_spends_budget():
    gradients = np.full((17, 1), 4., dtype=np.float32)
    gradients[[0, 8, 16], 0] = [2., -1., 2.]
    cfg, [result], observed = run(gradients)
    assert result.split_computations == cfg.budgets(17)[2] == 5
    assert result.epsilon_audit["epsilon"] == pytest.approx(1.2)
    assert result.epsilon_audit["count"] == 3
    assert result.epsilon_audit["split_stop_reason"] == "budget_exhausted"
    assert len(observed) == len(set(observed)) == len(result.anchors) == 8
    # A new child competes immediately; the old equal-minimum sibling can wait.
    assert [index for _, index in observed[:5]] == [0, 16, 8, 4, 12]
    assert any(i not in (0, 16, 8, 4, 12) for _, i in observed)


def test_new_child_preempts_old_tied_frontier():
    gradients = np.full((17, 1), 3., dtype=np.float32)
    gradients[[0, 8, 16, 4], 0] = [2., -1., 2., 1.4]
    _, [result], observed = run(gradients, split=2)
    assert [index for _, index in observed] == [0, 16, 8, 4, 6]
    assert result.split_computations == 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_independent_current_minimum_oracle_and_multi(seed):
    n = 33
    gradients = np.random.default_rng(seed).normal(size=(n, 3)).astype(np.float32)
    cfg, [result], observed = run(gradients, cover=6, split=20)
    intervals = [(0, n-1)]
    ids = [0, n-1]
    cover_budget, _, split_budget = cfg.budgets(n)
    while len(ids) < cover_budget:
        candidates = [iv for iv in intervals if iv[1]-iv[0] > 1]
        widest = max(r-l for l, r in candidates)
        batch = sorted(iv for iv in candidates if iv[1]-iv[0] == widest)[:cover_budget-len(ids)]
        for l, r in batch:
            m = (l+r)//2
            ids.append(m)
            intervals.remove((l, r))
            intervals.extend([(l, m), (m, r)])
    epsilon = float(np.quantile(np.linalg.norm(gradients[ids].astype(np.float64), axis=1), .1))
    split_ids = []
    while len(split_ids) < split_budget:
        candidates = [(float(np.linalg.norm((gradients[l].astype(np.float64)+gradients[r])/2)), l, r)
                      for l, r in intervals if r-l > 1 and
                      max(np.linalg.norm(gradients[l].astype(np.float64)),
                          np.linalg.norm(gradients[r].astype(np.float64))) > epsilon]
        if not candidates:
            break
        _, l, r = min(candidates)
        m = (l+r)//2
        split_ids.append(m)
        intervals.remove((l, r))
        intervals.extend([(l, m), (m, r)])
    assert [index for _, index in observed] == ids+split_ids
    assert result.split_computations == len(split_ids)
    assert [(b.left_index, b.right_index) for b in result.blocks] == sorted(intervals)
    assert np.concatenate([b.start_positions for b in result.blocks]).tolist() == result.valid_starts.tolist()
    _, parallel, _ = run(gradients, cover=6, split=20, many=True)
    for other in parallel:
        assert sorted(other.anchors) == sorted(result.anchors)
        assert other.epsilon_audit["epsilon"] == result.epsilon_audit["epsilon"]
        assert other.split_computations == result.split_computations


def test_new_zero_children_are_not_reenqueued():
    gradients = np.zeros((17, 1), dtype=np.float32)
    gradients[16] = 2.
    _, [result], _ = run(gradients, split=10)
    assert result.epsilon_audit["epsilon"] == 0
    assert sorted(result.anchors) == [0, 8, 12, 14, 15, 16]
    assert result.split_computations == 3
    assert result.epsilon_audit["split_unused"] == 7
    assert result.epsilon_audit["split_stop_reason"] == "eligible_pool_exhausted"
