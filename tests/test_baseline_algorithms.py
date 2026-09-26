"""Parity tests for kernels transcribed from the locked upstream baselines."""

from __future__ import annotations

import heapq

import numpy as np

from baselines.algorithms import (
    less_adam_direction,
    less_influence_scores,
    lp_mirror_scores,
    tsds_probability_assignment,
)


def test_less_adam_direction_matches_locked_upstream_equations() -> None:
    gradient = np.asarray([2.0, -1.0, 0.5], dtype=np.float32)
    first = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    second = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)
    expected = (0.9 * first + 0.1 * gradient) / np.sqrt(
        0.999 * second + 0.001 * gradient ** 2 + 1e-8
    )
    np.testing.assert_allclose(less_adam_direction(gradient, first, second), expected)


def test_less_multi_checkpoint_matches_upstream_normalize_matmul_weighting() -> None:
    source = np.asarray([[[1, 2], [3, 1]], [[2, -1], [1, 4]]], dtype=np.float32)
    target = np.asarray([[[2, 1], [1, 3]], [[4, 1], [-1, 2]]], dtype=np.float32)
    weights = np.asarray([2.0, 1.0])
    expected = np.zeros(2)
    for checkpoint, weight in enumerate(weights / weights.sum()):
        train = source[checkpoint] / np.linalg.norm(source[checkpoint], axis=1, keepdims=True)
        validation = target[checkpoint] / np.linalg.norm(target[checkpoint], axis=1, keepdims=True)
        expected += weight * (train @ validation.T).sum(axis=1)
    # Norm accumulation is now float64 before returning FP32 unit vectors;
    # retain the upstream formula oracle with a few FP32 rounding ulps.
    np.testing.assert_allclose(less_influence_scores(source, target, weights), expected,
                               rtol=3e-7, atol=1e-7)


def _upstream_tsds_reference(distances, indices, kdes, candidate_count, alpha, c_param):
    """Literal small-array version of locked TSDS ``tsds.py``."""
    m_queries = indices.shape[0]
    max_k = indices.shape[1]
    last_k = [0] * m_queries
    heap = [(1.0 / kdes[j][0], 0, j) for j in range(m_queries)]
    heapq.heapify(heap)
    weighted = [distances[j][0] / kdes[j][0] for j in range(m_queries)]
    cost = np.zeros(m_queries)
    total_cost = 0.0
    while heap:
        count, current_k, current_j = heapq.heappop(heap)
        scale = count
        total_cost -= cost[current_j]
        cost[current_j] = distances[current_j][current_k + 1] * count - weighted[current_j]
        total_cost += cost[current_j]
        if alpha / c_param * total_cost >= (1 - alpha) * m_queries:
            break
        last_k[current_j] = current_k
        if current_k < max_k - 2:
            count += 1.0 / kdes[current_j][current_k + 1]
            heapq.heappush(heap, (count, current_k + 1, current_j))
            weighted[current_j] += distances[current_j][current_k + 1] / kdes[current_j][current_k + 1]
    result = np.zeros(candidate_count)
    for query in range(m_queries):
        assigned = 0.0
        for neighbour in range(last_k[query] + 1):
            mass = 1 / m_queries / scale / kdes[query][neighbour]
            result[indices[query][neighbour]] += mass
            assigned += mass
        result[indices[query][last_k[query] + 1]] += max(1 / m_queries - assigned, 0)
    return result / result.sum()


def test_tsds_assignment_matches_locked_upstream_loop() -> None:
    distances = np.asarray([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]])
    indices = np.asarray([[0, 1, 2], [2, 3, 1]])
    kdes = np.asarray([[1.0, 1.5, 2.0], [1.2, 1.1, 1.7]])
    actual = tsds_probability_assignment(
        distances, indices, kdes, candidate_count=4, alpha=0.1, c_param=5.0,
    )
    expected = _upstream_tsds_reference(distances, indices, kdes, 4, 0.1, 5.0)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(), 1.0)


def test_lp_mirror_score_is_locked_forward_inf_loss_drop() -> None:
    original = np.asarray([3.0, 1.0, 2.0])
    adapted = np.asarray([1.0, 1.5, 0.5])
    np.testing.assert_array_equal(lp_mirror_scores(original, adapted), [2.0, -0.5, 1.5])
