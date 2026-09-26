"""Window coverage, Random's slot grid and diagnostic disjoint-window cutting.

Whole-unit arms use regions.py. LP-Mirror/LTSV retain native overlapping top-k.
"""

from __future__ import annotations


def selection_coverage(window_ids, window_length: int) -> dict:
    """Report native selection cardinality separately from unique point coverage."""
    unique = sorted(set(window_ids))
    covered = 0
    last_fid, stop = None, 0
    for fid, start in unique:
        if fid != last_fid:
            stop = start
        covered += max(0, start + window_length - max(start, stop))
        stop = max(stop, start + window_length)
        last_fid = fid
    return {"selected_windows": len(window_ids), "unique_windows": len(unique),
            "budget_windows": len(window_ids), "budget_points": len(window_ids) * window_length,
            "delivered_units": len(window_ids), "delivered_points": len(window_ids) * window_length,
            "repeated_points": len(window_ids) * window_length - covered,
            "overflow_points": 0, "overflow_ratio": 0., "legal_window_occurrences": len(window_ids),
            "unique_points": covered,
            "disjoint_window_equivalents": covered / window_length}

import numpy as np


def cut_disjoint(
    fragment_ids: np.ndarray,
    starts: np.ndarray,
    scores: np.ndarray,
    *,
    window_length: int,
    n: int,
    tiebreak: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy score-descending cut of mutually non-overlapping windows.

    ``fragment_ids``/``starts``/``scores`` are aligned arrays over scored
    candidates (starts are 1-indexed legal starts).  Candidates are taken in
    descending score order (ties broken by ``tiebreak``, then by input order)
    and accepted only if no already-accepted window on the same fragment lies
    within ``window_length - 1`` start units.  Returns the accepted candidate
    indices (length <= n).
    """
    fragment_ids = np.asarray(fragment_ids)
    starts = np.asarray(starts, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0 or n <= 0:
        return np.zeros(0, dtype=np.int64)
    if tiebreak is None:
        tiebreak = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((tiebreak, -scores))
    occupancy: dict[int, np.ndarray] = {}
    span = window_length - 1
    accepted: list[int] = []
    for candidate in order:
        fid = int(fragment_ids[candidate])
        start = int(starts[candidate])
        row = occupancy.get(fid)
        if row is None:
            row = np.zeros(start + span + 1, dtype=bool)
            occupancy[fid] = row
        elif len(row) <= start:
            grown = np.zeros(start + span + 1, dtype=bool)
            grown[: len(row)] = row
            occupancy[fid] = grown
            row = grown
        # accepting s marks every future start within window_length-1 of s;
        # a candidate is admissible iff its own position is unmarked (two
        # starts exactly window_length apart are already disjoint)
        if row[start]:
            continue
        row[max(0, start - span) : start + span + 1] = True
        accepted.append(int(candidate))
        if len(accepted) >= n:
            break
    return np.asarray(sorted(accepted), dtype=np.int64)


def disjoint_slots(
    lengths_by_fragment: dict[int, int],
    *,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical disjoint window grid per fragment (start = 1 + j*W).

    Returns (fragment_ids, starts) arrays of every grid slot with
    ``start <= T``; Random samples uniformly from these slots.
    """
    fids: list[int] = []
    starts: list[int] = []
    for fid in sorted(lengths_by_fragment):
        total = lengths_by_fragment[fid]
        slots = max(0, (total - 1) // window_length + 1) if total > 0 else 0
        for start in range(1, total + 1, window_length):
            fids.append(fid)
            starts.append(start)
    return (np.asarray(fids, dtype=np.int64),
            np.asarray(starts, dtype=np.int64))


__all__ = ["cut_disjoint", "disjoint_slots"]
