"""Static selection methods over prepared window assets."""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


def selection_size(total: int, *, ratio: float | None = None,
                   count: int | None = None) -> int:
    if total < 0:
        raise ValueError("total must be non-negative")
    if (ratio is None) == (count is None):
        raise ValueError("provide exactly one of ratio or count")
    if ratio is not None:
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        return min(total, max(1, int(np.ceil(total * ratio)))) if total else 0
    if count is None or count <= 0:
        raise ValueError("count must be positive")
    return min(total, count)


def select_topk(scores: Sequence[float], k: int) -> np.ndarray:
    """Return stable descending score indices; ties keep input order."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    if k < 0 or k > len(values):
        raise ValueError("k must be between zero and the candidate count")
    return np.argsort(-values, kind="stable")[:k].astype(np.int64)


def select_round_robin(groups: Sequence[object], k: int,
                       scores: Sequence[float] | None = None) -> np.ndarray:
    """Cycle over groups, greedily taking each group's highest score first."""
    labels = np.asarray(groups)
    if labels.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if k < 0 or k > len(labels):
        raise ValueError("k must be between zero and the candidate count")
    values = np.arange(len(labels), dtype=np.float64) if scores is None else np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(labels) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite and match groups")
    members: dict[object, list[int]] = {}
    for index, value in enumerate(labels.tolist()):
        key = value.item() if isinstance(value, np.generic) else value
        members.setdefault(key, []).append(index)
    queues = {
        key: deque(sorted(indices, key=lambda index: (-values[index], index)))
        for key, indices in members.items()
    }
    selected: list[int] = []
    active = list(queues)
    while len(selected) < k:
        remaining = []
        for key in active:
            if queues[key] and len(selected) < k:
                selected.append(queues[key].popleft())
            if queues[key]:
                remaining.append(key)
        active = remaining
    return np.asarray(selected, dtype=np.int64)



__all__ = ["selection_size", "select_topk", "select_round_robin"]
