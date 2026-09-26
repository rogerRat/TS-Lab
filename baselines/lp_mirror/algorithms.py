"""lp_mirror numerical kernels; independent of model and dataset code."""
from __future__ import annotations
import heapq
import numpy as np

def lp_mirror_scores(original_losses: np.ndarray, adapted_losses: np.ndarray) -> np.ndarray:
    """ForwardINF score: source loss before target adaptation minus after."""
    original = np.asarray(original_losses, dtype=np.float32)
    adapted = np.asarray(adapted_losses, dtype=np.float32)
    if original.shape != adapted.shape:
        raise ValueError("LP-Mirror loss arrays must have identical shapes")
    return original - adapted
