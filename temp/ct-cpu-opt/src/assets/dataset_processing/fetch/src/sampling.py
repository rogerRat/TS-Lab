"""Helpers for explicit raw-file trajectory sampling during fetch.

The default fetch path never imports this module.  Fetch adapters use it only
when a positive sample probability is explicitly requested.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


log = logging.getLogger("fetch")


def parallel_map(function, items, num_workers):
    """Apply one independent Fetch operation concurrently.

    @input: function accepts one item; items is a finite iterable of independent
            download/parse jobs; num_workers is a positive integer.
    @output: List of results in the same order as items.
    @scenario: Parallelize independent source files while keeping manifest and
               completion-marker writes in the caller thread.
    """
    values = list(items)
    if len(values) <= 1 or num_workers <= 1:
        return [function(item) for item in values]
    with ThreadPoolExecutor(max_workers=min(num_workers, len(values))) as pool:
        return list(pool.map(function, values))


def validate_sample_probability(value: float | None) -> float | None:
    """Validate an optional Bernoulli trajectory sampling probability.

    @input: value is None or a finite float in the closed interval (0, 1].
    @output: The validated float or None.
    @scenario: Reject ambiguous sampling configuration before any download.
    """
    if value is None:
        return None
    probability = float(value)
    if not np.isfinite(probability) or not 0.0 < probability <= 1.0:
        raise ValueError("sample_probability must be finite and in (0, 1]")
    return probability


def file_rng(seed: int, relative_path: str) -> np.random.Generator:
    """Create a deterministic random generator scoped to one source file.

    @input: seed is the user seed; relative_path is the source-relative file path.
    @output: NumPy Generator with stable cross-run initialization.
    @scenario: Make each file's sampling reproducible without shared state.
    """
    return np.random.default_rng([int(seed), *relative_path.encode("utf-8")])


class FileTrajectorySampler:
    """Apply Bernoulli sampling while forcing the first candidate to remain.

    @input: probability is the retention probability; rng is file-scoped.
    @output: Boolean decisions and a selected-count property.
    @scenario: Retain complete source records and guarantee one per source file.
    """

    def __init__(self, probability: float, rng: np.random.Generator) -> None:
        self.probability = validate_sample_probability(probability)
        if self.probability is None:
            raise ValueError("probability is required")
        self.rng = rng
        self.candidates = 0
        self.selected = 0

    def keep(self) -> bool:
        """Return whether the next complete trajectory record is retained.

        @input: No arguments; each call represents one complete source record.
        @output: True when the record must be written to the sampled file.
        @scenario: Keep at least one record while otherwise using Bernoulli sampling.
        """
        self.candidates += 1
        decision = self.candidates == 1 or bool(self.rng.random() < self.probability)
        if decision:
            self.selected += 1
        return decision


def cache_path(cache_root: Path, relative_path: str) -> Path:
    """Return the raw cache path preserving the upstream relative file tree.

    @input: cache_root is the raw ``.cache`` directory; relative_path is safe
            repository-relative source path text.
    @output: Cache file path.
    @scenario: Keep one downloaded source file outside the published raw tree.
    """
    path = cache_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_replace(source: Path, destination: Path) -> None:
    """Atomically publish one sampled file and preserve failure recoverability.

    @input: source is a completed temporary output; destination is final raw path.
    @output: None; destination is replaced after source validation by the caller.
    @scenario: Prevent consumers from observing a partially written sampled file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def remove_empty_cache(cache_root: Path) -> None:
    """Remove empty cache directories without touching non-cache raw files.

    @input: cache_root is the raw ``.cache`` directory.
    @output: None.
    @scenario: Finish a successful sampling fetch with no stale cache tree.
    """
    if not cache_root.is_dir():
        return
    for path in sorted(cache_root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        cache_root.rmdir()
    except OSError:
        pass


def discard_cache_file(path: Path, cache_root: Path) -> None:
    """Delete one cache file after its sampled replacement is published.

    @input: path is a downloaded cache file; cache_root is the cache boundary.
    @output: None; only the specified cache file is removed.
    @scenario: Bound disk usage to one source file during concurrent sampling
               without removing parent directories used by other workers.
    """
    path.unlink(missing_ok=True)
