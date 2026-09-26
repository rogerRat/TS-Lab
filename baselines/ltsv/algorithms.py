"""ltsv numerical kernels; independent of model and dataset code."""
from __future__ import annotations
import heapq
import numpy as np

def ltsv_window_scores(sums: np.ndarray, counts: np.ndarray,
                       starts: np.ndarray, window_length: int) -> np.ndarray:
    """Paper Eqs. 10--11; starts are zero-based, all points must be covered."""
    points = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float64),
                       where=counts > 0)
    prefix = np.concatenate(([0.0], np.cumsum(points)))
    covered = np.concatenate(([0], np.cumsum(counts > 0)))
    stops = starts + window_length
    values = (prefix[stops] - prefix[starts]) / window_length
    return np.where(covered[stops] - covered[starts] == window_length, values, -np.inf)
