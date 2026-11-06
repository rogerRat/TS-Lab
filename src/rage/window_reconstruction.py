"""Window-start gradients, before any block classification or aggregation."""
import numpy as np


def interpolate_window_gradients(result, indices=None):
    """Interpolate raw anchor vectors on legal physical starts; never extrapolate.

    Block rejection affects recall, not the existence of a window estimate.
    Optional indices bound memory when evaluating a long trajectory.
    """
    starts = np.asarray(result.valid_starts)
    anchors = np.asarray(sorted(result.anchors), dtype=np.int64)
    if not len(starts) or not len(anchors):
        raise ValueError('Reconstruction requires legal starts and anchors')
    if anchors[0] != 0 or anchors[-1] != len(starts)-1 or np.any(np.diff(starts) <= 0):
        raise ValueError('Anchors must cover the ordered legal window domain')
    rows = np.arange(len(starts)) if indices is None else np.asarray(indices, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= len(starts)):
        raise ValueError('Cannot reconstruct an illegal window start')
    positions = starts[anchors]
    vectors = np.stack([result.anchors[int(a)] for a in anchors]).astype(np.float64)
    if len(anchors) == 1:
        return np.repeat(vectors, len(rows), axis=0)
    right = np.clip(np.searchsorted(positions, starts[rows], side='right'), 1, len(anchors)-1)
    left = right-1
    alpha = (starts[rows]-positions[left])/(positions[right]-positions[left])
    return (1-alpha[:, None])*vectors[left] + alpha[:, None]*vectors[right]
