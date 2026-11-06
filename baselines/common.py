"""Shared static-pool and reference-window adapters."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rage.core import TimeSeriesFragment
from rage.experiments.arms import ArmOutput, SelectionCost, WindowIndex

from rage.experiments.deploy import cut_disjoint

def _pool_fingerprint(pool_manifest: dict | None) -> str:
    if pool_manifest is None:
        return "pool"
    return "seed-%s--N-%s--points-%s" % (
        pool_manifest.get("seed"), pool_manifest.get("equivalents_target"),
        pool_manifest.get("total_points"))


def reference_windows(reference_fragments: list[TimeSeriesFragment], *,
                      context_length: int, prediction_length: int,
                      stride: int, max_windows: int | None = None,
                      seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Legal reference windows (the validation set of gradient baselines)."""
    starts_by_fragment = [fragment.valid_starts(context_length, prediction_length, stride)
                          for fragment in reference_fragments]
    offsets = np.cumsum([0] + [len(starts) for starts in starts_by_fragment])
    total = int(offsets[-1])
    if max_windows is not None and total > max_windows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(total, size=max_windows, replace=False)
        keep.sort()
    else:
        keep = np.arange(total)
    # Sample the same global row IDs before materialising windows. No unused
    # reference windows need to be extracted or retained as Python objects.
    rows = np.empty((len(keep), context_length + prediction_length), dtype=np.float32)
    ids: list[tuple[int, int]] = []
    owners = np.searchsorted(offsets[1:], keep, side="right")
    width = np.arange(context_length + prediction_length)
    for owner in np.unique(owners):
        local = np.flatnonzero(owners == owner)
        starts = starts_by_fragment[owner][keep[local] - offsets[owner]]
        fragment = reference_fragments[owner]
        rows[local] = fragment.data[starts[:, None] - 1 + width]
        ids.extend((fragment.fragment_id, int(start)) for start in starts)
    return rows, ids


def _deploy_scores(index: WindowIndex, positions: np.ndarray, scores: np.ndarray,
                   *, n: int, window_length: int, tie_seed: int = 5,
                   ) -> tuple[list[tuple[int, int]], np.ndarray]:
    fids, sts = index.fragment_arrays(positions)
    rng = np.random.default_rng(tie_seed)
    chosen_local = cut_disjoint(fids, sts, scores, window_length=window_length,
                                n=n, tiebreak=rng.random(len(scores)))
    chosen = positions[chosen_local]
    return ([(int(f), int(s)) for f, s in zip(*index.fragment_arrays(chosen))],
            chosen)
