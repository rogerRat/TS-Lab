"""tsds numerical kernels; independent of model and dataset code."""
from __future__ import annotations
import heapq
import numpy as np

def tsds_probability_assignment(
    top_distances: np.ndarray,
    top_indices: np.ndarray,
    top_kdes: np.ndarray,
    *,
    candidate_count: int,
    alpha: float,
    c_param: float,
) -> np.ndarray:
    """The probability-assignment loop in locked upstream ``tsds.py``."""
    distances = np.asarray(top_distances, dtype=np.float64)
    indices = np.asarray(top_indices, dtype=np.int64)
    kdes = np.asarray(top_kdes, dtype=np.float64)
    if distances.shape != indices.shape or distances.shape != kdes.shape or distances.ndim != 2:
        raise ValueError("TSDS neighbour arrays must share shape [queries, K]")
    queries, neighbours = indices.shape
    if queries == 0 or neighbours < 2:
        raise ValueError("TSDS requires at least one query and two neighbours")
    if candidate_count <= 0 or np.any(indices < 0) or np.any(indices >= candidate_count):
        raise ValueError("TSDS candidate indices are out of range")
    if np.any(kdes <= 0) or not np.isfinite(kdes).all():
        raise ValueError("TSDS KDE values must be positive and finite")

    last_k = [0] * queries
    heap = [(1.0 / kdes[j][0], 0, j) for j in range(queries)]
    heapq.heapify(heap)
    weighted_distance = [distances[j][0] / kdes[j][0] for j in range(queries)]
    cost = np.zeros(queries, dtype=np.float64)
    total_cost = 0.0
    scale = 0.0
    while heap:
        count, current_k, current_query = heapq.heappop(heap)
        scale = count
        total_cost -= cost[current_query]
        cost[current_query] = (
            distances[current_query][current_k + 1] * count
            - weighted_distance[current_query]
        )
        total_cost += cost[current_query]
        if alpha / c_param * total_cost >= (1.0 - alpha) * queries:
            break
        last_k[current_query] = current_k
        if current_k < neighbours - 2:
            count += 1.0 / kdes[current_query][current_k + 1]
            heapq.heappush(heap, (count, current_k + 1, current_query))
            weighted_distance[current_query] += (
                distances[current_query][current_k + 1]
                / kdes[current_query][current_k + 1]
            )

    probabilities = np.zeros(candidate_count, dtype=np.float64)
    for query in range(queries):
        assigned = 0.0
        for neighbour in range(last_k[query] + 1):
            mass = 1.0 / queries / scale / kdes[query][neighbour]
            probabilities[indices[query][neighbour]] += mass
            assigned += mass
        remainder = max(1.0 / queries - assigned, 0.0)
        probabilities[indices[query][last_k[query] + 1]] += remainder
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("TSDS produced an invalid probability assignment")
    return probabilities / total
