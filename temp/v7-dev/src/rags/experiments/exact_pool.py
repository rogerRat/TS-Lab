"""Exact window-equivalent pool from complete, eligible source trajectories."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from rags.core import TimeSeriesFragment


class ExactEquivalents:
    """Incremental 0/1 subset sum: O(target) witnesses, bitset reachability.

    A witness is assigned only the first time a sum becomes reachable, so
    backtracking cannot reuse a trajectory from the same update.
    """
    def __init__(self, target):
        if target < 1:
            raise ValueError("Pool equivalent target must be positive")
        self.target = int(target)
        self.reachable = 1
        self.mask = (1 << (self.target + 1)) - 1
        self.previous = np.full(self.target + 1, -1, dtype=np.int64)
        self.owner = np.full(self.target + 1, -1, dtype=np.int64)

    def add(self, row, equivalents):
        equivalents = int(equivalents)
        if not 0 < equivalents <= self.target:
            return False
        shifted = (self.reachable << equivalents) & self.mask
        new = shifted & ~self.reachable
        while new:
            bit = new & -new
            total = bit.bit_length() - 1
            self.previous[total], self.owner[total] = total - equivalents, row
            new ^= bit
        self.reachable |= shifted
        return bool(self.reachable & (1 << self.target))

    def selected_rows(self):
        if not self.reachable & (1 << self.target):
            raise ValueError(f"Exact N={self.target} unreachable; greatest reachable N="
                             f"{self.reachable.bit_length() - 1}")
        rows, total = [], self.target
        while total:
            rows.append(int(self.owner[total]))
            total = int(self.previous[total])
        return rows[::-1]


def build_exact_pool(*, load_dir: Path, equivalents=16284, window_length=576, seed=0):
    from .data import load_bundle_index, _read_series
    bundle = load_bundle_index(load_dir)
    order = np.random.default_rng(seed).permutation(bundle.series_count())
    search = ExactEquivalents(equivalents)
    inspected = eligible = 0
    for row in order:
        inspected += 1
        values = _read_series(bundle, int(row))
        if len(values) < window_length or not np.isfinite(values).all() or float(np.std(values)) == 0:
            continue
        eligible += 1
        if search.add(int(row), len(values) // window_length):
            break
    rows = search.selected_rows()
    # Only retain selected complete trajectories in memory. Read once for
    # feasibility, once for delivery; never trim or join trajectory endpoints.
    fragments = [TimeSeriesFragment(np.asarray(_read_series(bundle, row), dtype=np.float32),
        entity_id=str(bundle.meta["series_id"][row])[:160], fragment_id=i)
        for i, row in enumerate(rows)]
    lengths = [f.length for f in fragments]
    manifest = dict(seed=seed, window_length=window_length, equivalents_target=equivalents,
        total_equivalents=sum(length // window_length for length in lengths),
        total_points=sum(lengths), total_legal_windows=sum(length - window_length + 1 for length in lengths),
        rows=rows, lengths=lengths, source_load=str(load_dir),
        pool_protocol="exact-complete-trajectories-v1", inspected_trajectories=inspected,
        eligible_trajectories=eligible)
    return fragments, manifest
