"""less numerical kernels; independent of model and dataset code."""
from __future__ import annotations
import heapq
import numpy as np

def unit_rows(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """L2-normalize nonzero gradients; native models can be below 1e-8.

    ``epsilon`` remains a legacy call argument, not a cosine denominator floor.
    This does not change the separate Adam numerical epsilon.
    """
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0).astype(np.float32)


def less_adam_direction(
    gradient: np.ndarray,
    exp_avg: np.ndarray,
    exp_avg_sq: np.ndarray,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Official LESS optimizer-aware source direction before projection.

    LESS intentionally does not apply bias correction here.  ``exp_avg`` and
    ``exp_avg_sq`` are the checkpoint's saved Adam states; the candidate
    example supplies the hypothetical next gradient.
    """
    grad = np.asarray(gradient, dtype=np.float32)
    first = np.asarray(exp_avg, dtype=np.float32)
    second = np.asarray(exp_avg_sq, dtype=np.float32)
    if grad.shape != first.shape or grad.shape != second.shape:
        raise ValueError("gradient and Adam moments must have identical shapes")
    updated_first = beta1 * first + (1.0 - beta1) * grad
    updated_second = beta2 * second + (1.0 - beta2) * np.square(grad)
    return updated_first / np.sqrt(updated_second + epsilon)


def less_influence_scores(
    source_features: np.ndarray,
    reference_features: np.ndarray,
    checkpoint_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate normalized LESS influence across reference rows/checkpoints.

    Inputs may be ``[N,D]``/``[M,D]`` for one checkpoint or
    ``[K,N,D]``/``[K,M,D]`` for the official multi-checkpoint computation.
    The sum over references is rank-equivalent to their mean for one target
    task and matches upstream matrix multiplication followed by aggregation.
    """
    source = np.asarray(source_features, dtype=np.float32)
    reference = np.asarray(reference_features, dtype=np.float32)
    if source.ndim == 2:
        source = source[None, ...]
    if reference.ndim == 2:
        reference = reference[None, ...]
    if source.ndim != 3 or reference.ndim != 3:
        raise ValueError("LESS features must be rank 2 or 3")
    if source.shape[0] != reference.shape[0] or source.shape[2] != reference.shape[2]:
        raise ValueError("source/reference checkpoint and feature dimensions must match")
    checkpoints = source.shape[0]
    if checkpoint_weights is None:
        weights = np.full(checkpoints, 1.0 / checkpoints, dtype=np.float64)
    else:
        weights = np.asarray(checkpoint_weights, dtype=np.float64)
        if weights.shape != (checkpoints,) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("checkpoint_weights must be finite positive-sum [K]")
        weights = weights / weights.sum()
    scores = np.zeros(source.shape[1], dtype=np.float64)
    for checkpoint, weight in enumerate(weights):
        train = unit_rows(source[checkpoint])
        validation = unit_rows(reference[checkpoint])
        scores += weight * (train @ validation.sum(axis=0))
    return scores.astype(np.float32)
