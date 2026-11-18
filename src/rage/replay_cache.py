"""Model-free replay of cached per-window projected gradients.

The store is a run directory holding ``gradient_sketches.npy`` plus
``manifest.json`` and ``.complete`` (the layout written by
``scripts/exp/main-ads-v0/compute_grad_cache_parallel.py``). Replay never
computes gradients: anchors are served from the memory-mapped cache, and every
remaining window is rebuilt by linear interpolation between its bracketing
anchors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import TimeSeriesFragment
from .crb import CRBConfig, CRBEngine
from .experiments.v7_scores import uniform_anchors
from .window_reconstruction import interpolate_window_gradients


class ProjectedGradientStore:
    """Memory-mapped reader for one trajectory's cached projected gradients."""

    def __init__(self, cache_dir: str | Path, *, manifest_name: str = "manifest.json"):
        self.root = Path(cache_dir)
        for name in ("gradient_sketches.npy", manifest_name, ".complete"):
            if not (self.root / name).is_file():
                raise FileNotFoundError("gradient cache is incomplete: " + str(self.root / name))
        self.gradients = np.load(self.root / "gradient_sketches.npy", mmap_mode="r")
        self.manifest = json.loads((self.root / manifest_name).read_text(encoding="ascii"))
        if self.gradients.ndim != 2 or self.gradients.shape[0] < 2 or self.gradients.shape[1] < 1:
            raise ValueError("gradient cache must be a two dimensional matrix with at least two rows")
        if not np.issubdtype(self.gradients.dtype, np.floating):
            raise ValueError("gradient cache must store floating point values")
        declared = self.manifest.get("window_start_count")
        if declared is not None and int(declared) != int(self.gradients.shape[0]):
            raise ValueError("manifest window_start_count differs from the cached matrix")
        self.rows = int(self.gradients.shape[0])
        self.dimensions = int(self.gradients.shape[1])

    @property
    def window_starts(self) -> np.ndarray:
        """One-based physical window starts matching the cached row order."""
        return np.arange(1, self.rows + 1, dtype=np.int32)

    def read(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 1 or not len(values):
            raise ValueError("row request must be a non-empty one dimensional array")
        if int(values.min()) < 0 or int(values.max()) >= self.rows:
            raise KeyError("row request lies outside the cached trajectory")
        return np.asarray(self.gradients[values], dtype=np.float32)


class CacheGradientComputer:
    """``BatchGradientComputer`` over one-based starts served from the cache."""

    def __init__(self, store: ProjectedGradientStore):
        self.store = store
        self.served_rows = 0

    def __call__(self, fragment: TimeSeriesFragment, starts: np.ndarray) -> np.ndarray:
        positions = np.asarray(starts, dtype=np.int64)
        if positions.ndim != 1:
            raise ValueError("gradient request must be one dimensional")
        rows = positions - 1
        if np.any(rows < 0) or np.any(rows >= self.store.rows):
            raise KeyError("gradient request lies outside the cached trajectory")
        self.served_rows += len(rows)
        return self.store.read(rows)


def anchor_budget(rows: int, q: float) -> int:
    """Exact anchor count for an equivalent spacing ``q``."""
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be finite and positive")
    return max(2, int(math.ceil(float(rows) / float(q))))


def bisection_plan(store: ProjectedGradientStore, q: float, *, context_length: int,
                   prediction_length: int, window_stride: int = 1):
    """Longest-gap bisection anchors via the CRB coverage stage.

    Only the coverage stage runs: ``cover_density`` is the exact anchor budget
    over the legal window count, and the error/split stages are disabled, so the
    placement is the pure repeated widest-gap midpoint bisection.
    """
    budget = anchor_budget(store.rows, q)
    density = float(budget) / float(store.rows)
    config = CRBConfig(cover_density=density, error_density=0.0, split_density=0.0,
                       detection="none", context_length=context_length,
                       prediction_length=prediction_length, window_stride=window_stride)
    computer = CacheGradientComputer(store)
    fragment = TimeSeriesFragment(
        np.zeros(store.rows + context_length + prediction_length - 1, dtype=np.float32),
        fragment_id=0)
    result = CRBEngine(config, computer).process_fragment(fragment)
    anchors = np.asarray(sorted(result.anchors), dtype=np.int64)
    if len(anchors) != budget:
        raise RuntimeError("coverage bisection produced %d anchors, expected %d"
                           % (len(anchors), budget))
    if anchors[0] != 0 or anchors[-1] != store.rows - 1:
        raise RuntimeError("coverage bisection must include both endpoints")
    if len(np.unique(anchors)) != budget:
        raise RuntimeError("coverage bisection produced duplicate anchors")
    return anchors, computer.served_rows


def uniform_plan(store: ProjectedGradientStore, q: float) -> np.ndarray:
    """Budget-matched uniform anchors using the frozen linspace rule."""
    return uniform_anchors(store.rows, anchor_budget(store.rows, q))


def _interpolation_view(store: ProjectedGradientStore, anchors: np.ndarray) -> SimpleNamespace:
    anchor_rows = np.asarray(anchors, dtype=np.int64)
    vectors = store.read(anchor_rows)
    return SimpleNamespace(valid_starts=store.window_starts,
                           anchors={int(row): vector for row, vector in zip(anchor_rows, vectors)})


def reconstruction_metrics(store: ProjectedGradientStore, anchors: np.ndarray, *,
                           chunk_rows: int = 4096, cosine_ge: float = 0.9) -> dict:
    """Linear-interpolation reconstruction metrics for one anchor layout."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    anchor_rows = np.sort(np.asarray(anchors, dtype=np.int64))
    if len(anchor_rows) != len(np.unique(anchor_rows)):
        raise ValueError("anchors must be unique")
    view = _interpolation_view(store, anchor_rows)
    is_anchor = np.zeros(store.rows, dtype=bool)
    is_anchor[anchor_rows] = True
    relative = np.empty(store.rows, dtype=np.float32)
    cosine = np.empty(store.rows, dtype=np.float32)
    l2_sum = 0.0
    for begin in range(0, store.rows, chunk_rows):
        indices = np.arange(begin, min(begin + chunk_rows, store.rows), dtype=np.int64)
        truth = store.read(indices)
        estimate = interpolate_window_gradients(view, indices).astype(np.float32)
        truth_norm = np.linalg.norm(truth, axis=1)
        estimate_norm = np.linalg.norm(estimate, axis=1)
        difference = np.linalg.norm(estimate - truth, axis=1)
        denominator = truth_norm * estimate_norm
        chunk_cosine = np.divide(
            np.einsum("ij,ij->i", estimate, truth, dtype=np.float32),
            denominator, out=np.zeros(len(indices), dtype=np.float32),
            where=denominator > 0)
        chunk_cosine = np.clip(chunk_cosine, np.float32(-1.0), np.float32(1.0))
        chunk_relative = np.divide(
            difference, truth_norm, out=np.zeros(len(indices), dtype=np.float32),
            where=truth_norm > 0)
        relative[indices] = chunk_relative
        cosine[indices] = chunk_cosine
        l2_sum += float(difference.sum(dtype=np.float64))
    non_anchor = ~is_anchor
    if not np.any(non_anchor):
        raise ValueError("a reconstruction needs at least one non-anchor window")
    gaps = np.diff(anchor_rows)
    summary = {
        "anchor_count": int(len(anchor_rows)),
        "equivalent_spacing": float(store.rows) / float(len(anchor_rows)),
        "gap_min": int(gaps.min()),
        "gap_max": int(gaps.max()),
        "gap_mean": float(gaps.mean()),
        "gap_coefficient_of_variation": float(gaps.std(ddof=0) / gaps.mean()) if gaps.mean() > 0 else 0.0,
        "windows": int(store.rows),
        "non_anchor_windows": int(np.count_nonzero(non_anchor)),
        "mean_l2": l2_sum / float(store.rows),
        "mean_relative_l2": float(relative.mean(dtype=np.float64)),
        "mean_relative_l2_non_anchor": float(relative[non_anchor].mean(dtype=np.float64)),
        "mean_cosine": float(cosine.mean(dtype=np.float64)),
        "mean_cosine_non_anchor": float(cosine[non_anchor].mean(dtype=np.float64)),
        "fraction_cosine_ge_threshold": float(np.count_nonzero(cosine >= cosine_ge)) / float(store.rows),
        "cosine_threshold": float(cosine_ge),
        "median_cosine": float(np.median(cosine)),
        "p10_cosine": float(np.percentile(cosine, 10)),
        "min_cosine": float(cosine.min()),
        "max_cosine": float(cosine.max()),
    }
    return {"summary": summary, "relative_l2": relative, "cosine": cosine}


def compare_layouts(bisect: dict, uniform: dict) -> dict:
    """Paired summary of the two layouts at one budget."""
    cosine_delta = bisect["cosine"] - uniform["cosine"]
    relative_delta = bisect["relative_l2"] - uniform["relative_l2"]
    summary = {
        "anchor_count": bisect["summary"]["anchor_count"],
        "equivalent_spacing": bisect["summary"]["equivalent_spacing"],
        "mean_relative_l2_bisect": bisect["summary"]["mean_relative_l2"],
        "mean_relative_l2_uniform": uniform["summary"]["mean_relative_l2"],
        "relative_l2_delta_bisect_minus_uniform":
            bisect["summary"]["mean_relative_l2"] - uniform["summary"]["mean_relative_l2"],
        "mean_cosine_bisect": bisect["summary"]["mean_cosine"],
        "mean_cosine_uniform": uniform["summary"]["mean_cosine"],
        "cosine_delta_bisect_minus_uniform":
            bisect["summary"]["mean_cosine"] - uniform["summary"]["mean_cosine"],
        "mean_relative_l2_non_anchor_bisect": bisect["summary"]["mean_relative_l2_non_anchor"],
        "mean_relative_l2_non_anchor_uniform": uniform["summary"]["mean_relative_l2_non_anchor"],
        "mean_cosine_non_anchor_bisect": bisect["summary"]["mean_cosine_non_anchor"],
        "mean_cosine_non_anchor_uniform": uniform["summary"]["mean_cosine_non_anchor"],
        "cosine_delta_non_anchor_bisect_minus_uniform":
            bisect["summary"]["mean_cosine_non_anchor"] - uniform["summary"]["mean_cosine_non_anchor"],
        "window_cosine_wins_bisect": int(np.count_nonzero(cosine_delta > 1e-9)),
        "window_cosine_wins_uniform": int(np.count_nonzero(cosine_delta < -1e-9)),
        "window_cosine_ties": int(np.count_nonzero(np.abs(cosine_delta) <= 1e-9)),
        "window_cosine_delta_mean": float(cosine_delta.mean(dtype=np.float64)),
        "window_relative_l2_delta_mean": float(relative_delta.mean(dtype=np.float64)),
    }
    return summary


__all__ = [
    "CacheGradientComputer",
    "ProjectedGradientStore",
    "anchor_budget",
    "bisection_plan",
    "compare_layouts",
    "reconstruction_metrics",
    "uniform_plan",
]
