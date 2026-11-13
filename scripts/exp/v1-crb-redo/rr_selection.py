"""RR selection over ref-to-source representative scores."""
from __future__ import annotations

import numpy as np


def round_robin_units(scores, source_window_counts, window_budget):
    """Select source units in ref order until their windows reach a fixed budget.

    Each score row belongs to one ref representative. Each column belongs to
    one source representative. A selected source unit is never selected again.
    The last selected unit may contribute only a prefix of its windows.
    """
    values = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(source_window_counts, dtype=np.int64)
    if (values.ndim != 2 or not values.shape[0] or not values.shape[1]
            or counts.shape != (values.shape[1],) or np.any(counts <= 0)
            or not np.isfinite(values).all() or window_budget <= 0
            or window_budget > int(counts.sum())):
        raise ValueError("Invalid RR score matrix, unit counts, or window budget")
    orders = np.argsort(-values, axis=1, kind="stable")
    cursors = np.zeros(values.shape[0], dtype=np.int64)
    selected = np.zeros(values.shape[1], dtype=bool)
    picked = []
    picked_windows = []
    used_windows = 0
    while used_windows < window_budget:
        progressed = False
        for ref_index in range(values.shape[0]):
            order = orders[ref_index]
            while cursors[ref_index] < len(order) and selected[order[cursors[ref_index]]]:
                cursors[ref_index] += 1
            if cursors[ref_index] >= len(order):
                continue
            source_index = int(order[cursors[ref_index]])
            cursors[ref_index] += 1
            selected[source_index] = True
            remaining = window_budget - used_windows
            take = min(int(counts[source_index]), remaining)
            picked.append(source_index)
            picked_windows.append(take)
            used_windows += take
            progressed = True
            if used_windows == window_budget:
                break
        if not progressed:
            raise RuntimeError("RR exhausted source units before reaching window budget")
    return np.asarray(picked, dtype=np.int64), np.asarray(picked_windows, dtype=np.int64)
