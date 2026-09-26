"""Compact RAGS ablation assets; never materialize interpolated window vectors."""
from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryGradient:
    fragment_id: int
    start_positions: np.ndarray
    mean_gradient: np.ndarray

    @property
    def direction(self):
        norm = np.linalg.norm(self.mean_gradient)
        return self.mean_gradient / norm if norm else np.zeros_like(self.mean_gradient)


def aggregate_retained(result):
    """Average raw linear reconstruction over the union of retained windows.

    Block ownership is respected, including half-open shared endpoints. Each
    anchor receives the sum of its interpolation coefficients, independent of
    how often the cache was read. Only one vector per trajectory is returned.
    """
    if not result.blocks:
        return None
    positions = np.unique(np.concatenate([b.start_positions for b in result.blocks]))
    weights = {}
    seen = set()
    for block in result.blocks:
        owned = np.asarray([p for p in block.start_positions if int(p) not in seen])
        seen.update(map(int, owned))
        if not len(owned):
            continue
        left, right = block.left_index, block.right_index
        lo, hi = result.valid_starts[left], result.valid_starts[right]
        right_weight = float(np.sum((owned - lo) / (hi - lo))) if hi != lo else 0.
        weights[left] = weights.get(left, 0.) + len(owned) - right_weight
        weights[right] = weights.get(right, 0.) + right_weight
    mean = sum(weight * np.asarray(result.anchors[key], dtype=np.float64)
               for key, weight in weights.items()) / len(positions)
    return TrajectoryGradient(result.fragment_id, positions, mean)


def trajectory_counts(result):
    return dict(fragment_id=int(result.fragment_id), gradients=int(result.computed_gradients),
                cover=int(result.cover_computations), error=int(result.error_computations),
                split=int(result.split_computations), legal_windows=len(result.valid_starts),
                retained_windows=sum(len(b.start_positions) for b in result.blocks))


def pair_counts(rags_source, uniform_source, rags_reference, uniform_reference):
    """Frozen pairing: identical configured budgets, exact K for every trajectory.

    A mismatch is diagnostic only. No density search, repeated probes or hidden
    top-up computes are used to manufacture an equal-cost comparison.
    """
    rows = []
    for side, base, variant in (("source", rags_source, uniform_source),
                                 ("reference", rags_reference, uniform_reference)):
        base = {r['fragment_id']: r for r in base}
        variant = {r['fragment_id']: r for r in variant}
        for fid in sorted(base.keys() | variant.keys()):
            a, b = base.get(fid), variant.get(fid)
            matched = a is not None and b is not None and a['gradients'] == b['gradients']
            rows.append(dict(side=side, fragment_id=fid, rags=a, uniform=b, matched=matched))
    return dict(rule='same-phase-budgets-exact-K-per-side-per-trajectory-v1',
                eligible_same_compute=bool(rows) and all(r['matched'] for r in rows), rows=rows)


def aggregate_anchors(result, *, expected_count=None):
    """Same FP32 arithmetic sampled-gradient mean as adapted LESS."""
    if not result.anchors:
        return None
    keys = sorted(result.anchors)
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError('RAGS-Traj distinct anchor count differs from configured budget')
    if sum(len(b.start_positions) for b in result.blocks) != len(result.valid_starts):
        raise ValueError('RAGS-Traj requires all legal starts, without filtering')
    mean = np.zeros_like(np.asarray(result.anchors[keys[0]]), dtype=np.float32)
    for key in keys:
        mean += np.asarray(result.anchors[key], dtype=np.float32)
    mean /= np.float32(len(keys))
    return TrajectoryGradient(result.fragment_id, result.valid_starts.copy(), mean)
