"""Mechanism experiment 1: does CRB keep the full-scan ranking on a budget?

Ground truth: every legal window's true gradient on an experiment subset
(reference and source sides), computed once with the same pretrained probe
and cached for reuse (mechanism experiment 2 reads the same cache).

Comparisons at matched actual gradient count K:

  gradient reconstruction   CRB vs uniform anchors vs random anchors
                            (both baselines include the endpoints);
  block recall              RAGE vs uniform+same detection vs random+same
                            detection vs fixed-length chunking (block count
                            aligned, representatives from the same
                            reconstructed gradients).

Metrics (per §3.2): relative gradient error, direction error, top-M rank
overlap (M = 50/100/500), high-score window coverage, boundary margin, and
the full cost table.  Also the one-sided replacement study: one side true,
the other estimated.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine, CRBResult

from .tasks import build_task_data, build_task_specs
from .train_eval import crb_gradient_computer


# ---------------------------------------------------------------------------
# Ground-truth full scan
# ---------------------------------------------------------------------------

@dataclass
class SideScan:
    fragments: list[TimeSeriesFragment]
    gradients: list[np.ndarray]     # [T_i, d] per fragment (true, per window)
    positions: list[np.ndarray]     # physical starts, aligned with rows
    window_losses: list[np.ndarray]


def full_scan(engine, fragments: list[TimeSeriesFragment], *,
              context_length: int, prediction_length: int,
              stride: int = 1, log_every: int = 5000) -> SideScan:
    scanner = crb_gradient_computer(engine, context_length, prediction_length)
    gradients: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    windows_done = 0
    for fragment in fragments:
        starts = fragment.valid_starts(context_length, prediction_length, stride)
        windows = np.stack([
            fragment.extract_window(int(s), context_length, prediction_length)
            for s in starts
        ])
        fragment_losses = np.empty(len(windows), dtype=np.float64)
        sketches = np.empty((len(windows), engine.config.projection_dimension),
                            dtype=np.float32)
        batch = 16
        for begin in range(0, len(windows), batch):
            stop = min(len(windows), begin + batch)
            loss, sketch = engine.batch_gradient_sketch(windows[begin:stop])
            fragment_losses[begin:stop] = loss
            sketches[begin:stop] = sketch
        gradients.append(sketches)
        positions.append(np.asarray(starts, dtype=np.int64))
        losses.append(fragment_losses)
        windows_done += len(windows)
        if log_every and windows_done // log_every != (windows_done - len(windows)) // log_every:
            print("  [full scan] %d windows" % windows_done, flush=True)
    return SideScan(fragments, gradients, positions, losses)


def save_scan(path: Path, scan: SideScan, *, side: str) -> None:
    payload = {
        ("gradients_%d" % i): value for i, value in enumerate(scan.gradients)
    }
    payload.update({
        ("positions_%d" % i): value for i, value in enumerate(scan.positions)
    })
    payload.update({
        ("data_%d" % i): fragment.data for i, fragment in enumerate(scan.fragments)
    })
    payload["count"] = np.asarray([len(scan.fragments)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_scan(path: Path, *, context_length: int, prediction_length: int,
               stride: int = 1) -> SideScan:
    with np.load(path) as payload:
        count = int(payload["count"][0])
        fragments = [
            TimeSeriesFragment(
                data=payload["data_%d" % i].astype(np.float32),
                entity_id="scan-%d" % i, fragment_id=10_000 + i,
            )
            for i in range(count)
        ]
        gradients = [payload["gradients_%d" % i] for i in range(count)]
        positions = [payload["positions_%d" % i] for i in range(count)]
    losses = [np.zeros(len(g)) for g in gradients]
    return SideScan(fragments, gradients, positions, losses)


# ---------------------------------------------------------------------------
# Anchor strategies and reconstruction
# ---------------------------------------------------------------------------

class CountingCache:
    """Budget simulation: serve gradients from a precomputed scan, count calls."""

    def __init__(self, scan: SideScan):
        self.table = {
            (scan.fragments[i].fragment_id, int(start)): scan.gradients[i][row]
            for i in range(len(scan.fragments))
            for row, start in enumerate(scan.positions[i])
        }
        self.calls = 0

    def computer(self, fragment: TimeSeriesFragment, starts: np.ndarray):
        rows = []
        for start in starts:
            key = (fragment.fragment_id, int(start))
            if key not in self.table:
                raise KeyError("scan cache miss for %s" % (key,))
            rows.append(self.table[key])
        self.calls += len(rows)
        return np.stack(rows).astype(np.float32)


def uniform_anchors(total: int, k: int, rng: np.random.Generator | None = None,
                    ) -> np.ndarray:
    """Anchor positions (0-indexed over [0, total)); endpoints always included."""
    k = max(min(k, total), 2)
    if rng is None:
        positions = np.linspace(0, total - 1, num=k).round().astype(int)
    else:
        positions = np.concatenate((
            [0, total - 1],
            rng.choice(np.arange(1, total - 1), size=k - 2, replace=False),
        ))
    return np.unique(positions)


def interpolate_from_anchors(anchors: np.ndarray, anchor_grads: np.ndarray,
                             total: int) -> np.ndarray:
    """Linear interpolation of gradients over positions [0, total)."""
    anchors = np.asarray(anchors, dtype=np.int64)
    order = np.argsort(anchors)
    anchors = anchors[order]
    anchor_grads = anchor_grads[order]
    estimated = np.empty((total, anchor_grads.shape[1]), dtype=np.float32)
    for pos in range(total):
        idx = np.searchsorted(anchors, pos)
        if idx <= 0:
            estimated[pos] = anchor_grads[0]
        elif idx >= len(anchors):
            estimated[pos] = anchor_grads[-1]
        else:
            left, right = anchors[idx - 1], anchors[idx]
            alpha = (pos - left) / (right - left)
            estimated[pos] = (1 - alpha) * anchor_grads[idx - 1] + alpha * anchor_grads[idx]
    return estimated


def crb_result_for(scan: SideScan, cache: CountingCache, *, total_density: float,
                   cover_share: float, error_share: float, split_share: float,
                   context_length: int, prediction_length: int,
                   stride: int, mean_epsilon: float, endpoint_epsilon: float,
                   curvature_cap: float) -> tuple[CRBResult, np.ndarray]:
    """Run CRB on every fragment of one side with a given budget split.

    Returns the per-fragment results plus the reconstructed gradients (for
    positions without an anchor, the linear interpolation between anchors).
    """
    config = CRBConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        window_stride=stride,
        cover_density=total_density * cover_share,
        error_density=total_density * error_share,
        split_density=total_density * split_share,
        mean_epsilon=mean_epsilon,
        endpoint_epsilon=endpoint_epsilon,
        curvature_cap=curvature_cap,
    )
    engine = CRBEngine(config, cache.computer)
    results = [engine.process_fragment(fragment) for fragment in scan.fragments]
    reconstructed = []
    for fragment, result in zip(scan.fragments, results):
        total = len(result.valid_starts)
        anchors = np.asarray(sorted(result.anchors), dtype=np.int64)
        grads = np.stack([result.anchors[int(a)] for a in anchors])
        reconstructed.append(interpolate_from_anchors(anchors, grads, total))
    return results, reconstructed


def detect_blocks(anchors: np.ndarray, anchor_grads: np.ndarray, total: int, *,
                  mean_epsilon: float, endpoint_epsilon: float,
                  split_budget: int, gradient_lookup) -> tuple[dict, int]:
    """Standalone cancellation detection over given anchors (crb.py §3.3).

    ``gradient_lookup(pos_index)`` must return the true gradient at a
    position index (used for the one-shot split) or raise KeyError; each
    successful lookup is counted as one extra gradient computation.

    Returns (info dict with passed half-open intervals [left, right) — the
    last one includes its right endpoint per §4.1 — plus discard counters,
    extra gradient calls).
    """
    order = np.argsort(anchors)
    anchors, anchor_grads = anchors[order], anchor_grads[order]
    anchor_set = {int(a): anchor_grads[i] for i, a in enumerate(anchors)}

    def interval_state(left: int, right: int) -> str:
        mean = (anchor_set[left] + anchor_set[right]) * 0.5
        if float(np.linalg.norm(mean)) >= mean_epsilon:
            return "pass"
        left_weak = float(np.linalg.norm(anchor_set[left])) < endpoint_epsilon
        right_weak = float(np.linalg.norm(anchor_set[right])) < endpoint_epsilon
        return "no_signal" if left_weak and right_weak else "antagonistic"

    intervals = [(int(anchors[i]), int(anchors[i + 1]))
                 for i in range(len(anchors) - 1)]
    passed: list[tuple[int, int]] = []
    flagged: list[tuple[int, int]] = []
    no_signal = 0
    for interval in intervals:
        state = interval_state(*interval)
        if state == "pass":
            passed.append(interval)
        elif state == "no_signal":
            no_signal += 1
        else:
            flagged.append(interval)

    antagonistic = 0
    extra_calls = 0
    for left, right in flagged[:split_budget]:
        mid = (left + right) // 2
        if not left < mid < right:
            antagonistic += 1
            continue
        try:
            mid_grad = gradient_lookup(mid)
        except KeyError:
            antagonistic += 1
            continue
        extra_calls += 1
        anchor_set[mid] = mid_grad
        for child in ((left, mid), (mid, right)):
            if interval_state(*child) == "pass":
                passed.append(child)
            else:
                antagonistic += 1
    banned = len(flagged) - min(len(flagged), split_budget)
    passed = sorted(set(passed))
    if passed and passed[-1][1] == int(anchors[-1]):
        passed[-1] = (passed[-1][0], min(total, passed[-1][1] + 1))
    info = {
        "blocks": passed,
        "no_signal": no_signal,
        "antagonistic": antagonistic,
        "banned": banned,
    }
    return info, extra_calls


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reconstruction_metrics(true: np.ndarray, estimated: np.ndarray, *,
                           direction_norm_floor: float = 1e-6) -> dict:
    """Relative error, direction error, near-zero counts."""
    num = float(((estimated - true) ** 2).sum())
    den = float((true ** 2).sum())
    true_norms = np.linalg.norm(true, axis=1)
    est_norms = np.linalg.norm(estimated, axis=1)
    valid = (true_norms > direction_norm_floor) & (est_norms > 0)
    if valid.sum() > 0:
        cosines = (estimated[valid] * true[valid]).sum(axis=1) / (
            est_norms[valid] * true_norms[valid])
        direction_error = float((1.0 - cosines).mean())
    else:
        direction_error = float("nan")
    return {
        "relative_error": num / max(den, 1e-12),
        "direction_error": direction_error,
        "near_zero_windows": int((true_norms <= direction_norm_floor).sum()),
        "windows": int(len(true)),
    }


def ranking_metrics(true_scores: np.ndarray, est_scores: np.ndarray,
                    m_values: tuple[int, ...] = (50, 100, 500)) -> dict:
    """Top-M overlap and the boundary margin around each M."""
    true_order = np.argsort(-true_scores)
    est_order = np.argsort(-est_scores)
    out = {}
    for m in m_values:
        m = min(m, len(true_scores))
        top_true = set(true_order[:m].tolist())
        top_est = set(est_order[:m].tolist())
        out["overlap@%d" % m] = len(top_true & top_est) / m
        if m < len(true_scores):
            out["margin@%d" % m] = float(
                true_scores[true_order[m - 1]] - true_scores[true_order[m]])
    out["spearman"] = float(_spearman(true_scores, est_scores))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats

    return stats.spearmanr(a, b).statistic


def coverage_metrics(true_top: set[int], returned_windows: set[int],
                     m: int) -> dict:
    return {
        "coverage@%d" % m: len(true_top & returned_windows) / max(m, 1),
        "returned_windows": len(returned_windows),
    }


__all__ = [
    "SideScan",
    "full_scan",
    "save_scan",
    "load_scan",
    "CountingCache",
    "uniform_anchors",
    "interpolate_from_anchors",
    "crb_result_for",
    "detect_blocks",
    "reconstruction_metrics",
    "ranking_metrics",
    "coverage_metrics",
    "run_mechanism1",
    "main",
]


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def _block_windows(blocks: list[tuple[int, int]], positions: np.ndarray,
                   ) -> set[int]:
    """Window row indices covered by half-open position intervals."""
    returned: set[int] = set()
    for left, right in blocks:
        for row in range(len(positions)):
            if left <= row < right:
                returned.add(row)
    return returned


def _block_representatives(blocks, estimated: np.ndarray) -> np.ndarray:
    reps = []
    for left, right in blocks:
        span = estimated[left:min(right, len(estimated))]
        if len(span) == 0:
            span = estimated[left:left + 1]
        reps.append(span.mean(axis=0))
    return np.stack(reps)


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def _crb_blocks_and_directions(result: CRBResult):
    """Passed blocks of one CRB fragment as half-open intervals + directions."""
    total = len(result.valid_starts)
    intervals = []
    directions = []
    for block in result.blocks:
        right = block.right_index + (1 if block.right_index == total - 1 else 0)
        intervals.append((block.left_index, right))
        directions.append(block.direction)
    return intervals, directions


def _window_assignment(total: int, intervals: list[tuple[int, int]]) -> list[int]:
    """Map each window position to its block index (-1 for dropped)."""
    assignment = [-1] * total
    for index, (left, right) in enumerate(intervals):
        for pos in range(left, right):
            assignment[pos] = index
    return assignment


def _expand_blocks(blocks_per_fragment: list[list[tuple[int, int]]],
                   ) -> list[set[int]]:
    """Global window-row sets for each fragment's blocks."""
    expanded: list[set[int]] = []
    offset = 0
    for blocks in blocks_per_fragment:
        for left, right in blocks:
            expanded.append(set(range(offset + left, offset + right)))
        offset += max((right for _l, right in blocks), default=0)
    return expanded


# Frozen budget ratio 4:4:1 (C1: 2% : 2% : 0.5% of 4.5% total density).
DEFAULT_SHARES = (4.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0)


def gradient_norm_stats(reference_scan: SideScan, source_scan: SideScan) -> dict:
    """Norm distribution of all saved real gradients (C3 epsilon provenance)."""
    norms = np.concatenate([
        np.linalg.norm(np.concatenate(scan.gradients, axis=0).astype(np.float32), axis=1)
        for scan in (reference_scan, source_scan)
    ])
    return {
        "count": int(len(norms)),
        "percentiles": {str(p): float(np.percentile(norms, p))
                        for p in (1, 5, 10, 25, 50, 75, 90, 99)},
        "mean": float(norms.mean()),
        "epsilon_calibration_note": (
            "epsilon_mu = epsilon_g = p10 over both sides' real sketch norms "
            "(C3); Count-Sketch norms are unbiased across projection widths"),
    }


def trajectory_aggregation_divergence(source_scan: SideScan, reference_scan: SideScan,
                           *,
                           m_values: tuple[int, ...] = (50, 100, 500)) -> dict:
    """Aggregated-object ranking vs full-window ranking (S3.2 row 6).

    Object gradients are recomputed directly from the saved full-scan
    gradients in contiguous legal-start groups spanning 576 points;
    windows inherit their object's score.  The query is the normalized mean
    reference gradient; the reference ranking is the same query against the
    true per-window gradients.
    """
    ref_mean = np.concatenate(reference_scan.gradients, axis=0).mean(axis=0)
    query = ref_mean / max(float(np.linalg.norm(ref_mean)), 1e-12)

    def ranking_from_scores(scores: np.ndarray, m: int) -> set[int]:
        m = min(m, len(scores))
        return set(np.argsort(-scores)[:m].tolist())

    out: dict = {"variants": {}}
    src_sizes = [len(g) for g in source_scan.gradients]
    inherited = np.zeros(sum(src_sizes), dtype=np.float64)
    cursor = 0
    for index, size in enumerate(src_sizes):
        grads = source_scan.gradients[index]
        for begin in range(0, size, 576):
            stop = min(begin + 576, size)
            mean = grads[begin:stop].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            score = float(query @ mean / norm) if norm > 1e-12 else -1.0
            inherited[cursor + begin:cursor + stop] = score
        cursor += size
    unit = np.concatenate(source_scan.gradients, axis=0)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    full_scores = unit @ query
    record = {}
    for m in m_values:
        top_full = ranking_from_scores(full_scores, m)
        top_agg = ranking_from_scores(inherited, m)
        record["overlap@%d" % m] = len(top_full & top_agg) / min(m, len(full_scores))
    out["variants"]["trajectory"] = record
    return out


def single_segment_share(source_scan: SideScan, *, top_fraction: float = 0.1,
                         span_fraction: float = 0.25) -> dict:
    """Share of trajectories whose high-score windows sit in one segment.

    Definition (fixed): a trajectory's high-score windows are its top
    ceil(top_fraction * T) windows by mean-reference-query score; the
    trajectory is "single-segment driven" when their position span is at
    most span_fraction of the trajectory length.
    """
    concentrated = 0
    total = 0
    spans = []
    for index, grads in enumerate(source_scan.gradients):
        size = len(grads)
        unit = grads / np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-12)
        order = np.argsort(-unit.mean(axis=0) @ unit.T)
        k = max(1, int(np.ceil(top_fraction * size)))
        top_positions = np.sort(order[:k])
        span = float(top_positions[-1] - top_positions[0]) / max(size - 1, 1)
        spans.append(span)
        if span <= span_fraction:
            concentrated += 1
        total += 1
    return {
        "definition": ("top ceil(%.0f%%*T) windows; concentrated iff position span "
                       "<= %.0f%% of trajectory" % (100 * top_fraction,
                                                    100 * span_fraction)),
        "trajectories": total,
        "concentrated": concentrated,
        "share": concentrated / max(total, 1),
        "median_span": float(np.median(spans)) if spans else None,
    }


def run_mechanism1(*, reference_scan: SideScan, source_scan: SideScan,
                   densities: list[float], b_values: list[int],
                   m_values: tuple[int, ...] = (50, 100, 500),
                   cover_share: float = DEFAULT_SHARES[0],
                   error_share: float = DEFAULT_SHARES[1],
                   split_share: float = DEFAULT_SHARES[2],
                   mean_epsilon: float = 1e-8,
                   endpoint_epsilon: float = 1e-8,
                   curvature_cap: float = 1.0,
                   context_length: int = 512,
                   prediction_length: int = 64,
                   stride: int = 1,
                   n_fixed: int | None = None,
                   seed: int = 0) -> dict:
    """Sweep densities: reconstruction variants and block-recall variants.

    Reference queries are reference *windows*; each inherits the result of
    its reference block (dropped blocks give an empty result).  True per-query
    rankings come from the full scans on both sides.
    """
    rng = np.random.default_rng(seed)
    ref_true = np.concatenate(reference_scan.gradients, axis=0)
    src_true = np.concatenate(source_scan.gradients, axis=0)
    ref_unit, src_unit = _unit_rows(ref_true), _unit_rows(src_true)
    true_scores = ref_unit @ src_unit.T  # [Q, S], computed once
    query_count, source_count = true_scores.shape
    true_top: dict[int, np.ndarray] = {}
    for m in m_values:
        m_eff = min(m, source_count)
        true_top[m] = np.argsort(-true_scores, axis=1)[:, :m_eff]

    ref_sizes = [len(g) for g in reference_scan.gradients]
    src_sizes = [len(g) for g in source_scan.gradients]
    ref_offsets = np.cumsum([0] + ref_sizes)
    src_offsets = np.cumsum([0] + src_sizes)
    if n_fixed is None:
        n_fixed = max(1, source_count // 20)

    def _unit_or_zero(rows: np.ndarray) -> np.ndarray:
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)

    def _side_crb(scan: SideScan, density: float):
        cache = CountingCache(scan)
        results, reconstructed = crb_result_for(
            scan, cache, total_density=density,
            cover_share=cover_share, error_share=error_share,
            split_share=split_share, context_length=context_length,
            prediction_length=prediction_length, stride=stride,
            mean_epsilon=mean_epsilon, endpoint_epsilon=endpoint_epsilon,
            curvature_cap=curvature_cap)
        return results, reconstructed, cache.calls

    def _variant_blocks(scan: SideScan, results, reconstructed, method: str):
        """Blocks + per-fragment reconstructed reps for a recall variant."""
        if method == "rage":
            blocks, directions = [], []
            for index, result in enumerate(results):
                intervals, dirs = _crb_blocks_and_directions(result)
                blocks.append(intervals)
                directions.append(np.stack(dirs) if dirs else np.zeros((0, reconstructed[index].shape[1])))
            extra_k = 0
            return blocks, directions, extra_k
        if method in ("uniform_detect", "random_detect"):
            blocks, directions = [], []
            extra_k = 0
            for index, result in enumerate(results):
                total = len(reconstructed[index])
                # match CRB's anchor count minus its split spend
                n_anchors = max(2, result.cover_computations + result.error_computations)
                anchors = uniform_anchors(total, n_anchors,
                                          rng=rng if method == "random_detect" else None)
                anchor_grads = np.stack([scan.gradients[index][a] for a in anchors])

                def lookup(pos, index=index):
                    return scan.gradients[index][pos]

                info, calls = detect_blocks(
                    anchors, anchor_grads, total,
                    mean_epsilon=mean_epsilon, endpoint_epsilon=endpoint_epsilon,
                    split_budget=result.split_computations,
                    gradient_lookup=lookup)
                blocks.append(info["blocks"])
                reps = _block_representatives(info["blocks"], reconstructed[index])
                directions.append(_unit_or_zero(reps))
                extra_k += calls
            return blocks, directions, extra_k
        if method == "fixed_length":
            blocks, directions = [], []
            crb_block_count = sum(len(r.blocks) for r in results)
            for index, result in enumerate(results):
                total = len(reconstructed[index])
                count = max(1, int(round(crb_block_count * total / sum(src_sizes))))
                edges = np.linspace(0, total, count + 1).round().astype(int)
                intervals = [(int(edges[i]), int(edges[i + 1]))
                             for i in range(count) if edges[i + 1] > edges[i]]
                if intervals and intervals[-1][1] < total:
                    intervals[-1] = (intervals[-1][0], total)
                blocks.append(intervals)
                reps = _block_representatives(intervals, reconstructed[index])
                directions.append(_unit_or_zero(reps))
            return blocks, directions, 0
        raise ValueError("unknown variant %s" % method)

    def _coverage(records, ref_blocks, ref_assignment, query_block_of,
                  src_directions, src_blocks, b_values, label, extra_k):
        """Per-query and union coverage for one source-side variant."""
        directions = np.concatenate(src_directions, axis=0) if src_directions else np.zeros((0, 0))
        if len(directions) == 0:
            return
        ref_dirs = np.concatenate(
            [np.stack(d) if len(d) else np.zeros((0, directions.shape[1]))
             for d in ref_blocks_directions], axis=0)
        scores = ref_dirs @ directions.T if len(ref_dirs) else np.zeros((0, len(directions)))
        for b in b_values:
            take = min(b, len(directions))
            top = np.argsort(-scores, axis=1, kind="stable")[:, :take] if len(scores) else []
            per_query = {m: [] for m in m_values}
            union: set[int] = set()
            window_best: dict[int, float] = {}
            for query in range(query_count):
                block_index = query_block_of[query]
                if block_index < 0:
                    for m in m_values:
                        per_query[m].append(0.0)
                    continue
                returned: set[int] = set()
                row = scores[block_index]
                for block_col in top[block_index]:
                    # map source block column -> fragment/interval
                    frag, interval, col_base = src_block_map[int(block_col)]
                    for row_index in range(src_offsets[frag] + interval[0],
                                           src_offsets[frag] + interval[1]):
                        returned.add(row_index)
                    score = float(row[block_col])
                    for row_index in returned:
                        if score > window_best.get(row_index, -np.inf):
                            window_best[row_index] = score
                union |= returned
                for m in m_values:
                    hits = len(true_top[m][query][np.isin(true_top[m][query],
                                                          list(returned))]) if returned else 0
                    per_query[m].append(hits / len(true_top[m][query]))
            record = {
                "variant": label, "B": b, "blocks": int(len(directions)),
                "extra_gradients": int(extra_k),
                "returned_union": len(union),
                "mean_query_coverage": {},
                "union_coverage": {},
            }
            for m in m_values:
                record["mean_query_coverage"]["@%d" % m] = float(np.mean(per_query[m]))
                m_eff = min(m, source_count)
                top_union = set(np.argsort(-true_scores.mean(axis=0))[:m_eff].tolist())
                record["union_coverage"]["@%d" % m] = len(top_union & union) / m_eff
            # fixed-N selection coverage (§1.3: best cosine over returning queries)
            ranked = sorted(window_best, key=lambda w: (-window_best[w], w))[:n_fixed]
            selected = set(ranked)
            record["fixed_n"] = n_fixed
            record["fixed_n_coverage"] = {}
            for m in m_values:
                m_eff = min(m, source_count)
                top_union = set(np.argsort(-true_scores.mean(axis=0))[:m_eff].tolist())
                record["fixed_n_coverage"]["@%d" % m] = len(top_union & selected) / m_eff
            records.append(record)

    results = {"protocol": {
        "cover_share": cover_share, "error_share": error_share,
        "split_share": split_share, "stride": stride, "seed": seed,
        "queries": query_count, "source_windows": source_count,
        "ref_sizes": ref_sizes, "src_sizes": src_sizes,
    }, "densities": []}

    for density in densities:
        density_record = {"density": density, "reconstruction": [], "recall": []}

        # ---- reconstruction: CRB vs uniform vs random at matched anchor K ----
        ref_results, ref_reconstructed, ref_k = _side_crb(reference_scan, density)
        src_results, src_reconstructed, src_k = _side_crb(source_scan, density)
        for side_name, scan, results_side, reconstructed, k_actual in (
                ("reference", reference_scan, ref_results, ref_reconstructed, ref_k),
                ("source", source_scan, src_results, src_reconstructed, src_k)):
            true_all = np.concatenate(scan.gradients, axis=0)
            est_all = np.concatenate(reconstructed, axis=0)
            record = {"side": side_name, "method": "crb", "k": k_actual,
                      "positions": int(len(true_all))}
            record.update(reconstruction_metrics(true_all, est_all))
            density_record["reconstruction"].append(record)
            for method in ("uniform", "random"):
                per_fragment_k = [r.computed_gradients for r in results_side]
                estimated_side = []
                k_total = 0
                for index, fragment in enumerate(scan.fragments):
                    total = len(scan.gradients[index])
                    anchors = uniform_anchors(total, per_fragment_k[index],
                                              rng=rng if method == "random" else None)
                    anchor_grads = np.stack([scan.gradients[index][a] for a in anchors])
                    k_total += len(anchors)
                    estimated_side.append(interpolate_from_anchors(
                        anchors, anchor_grads, total))
                record = {"side": side_name, "method": method, "k": k_total,
                          "positions": int(len(true_all))}
                record.update(reconstruction_metrics(
                    true_all, np.concatenate(estimated_side, axis=0)))
                density_record["reconstruction"].append(record)

        # ---- block recall --------------------------------------------------
        ref_blocks_intervals: list[list[tuple[int, int]]] = []
        ref_blocks_directions: list[np.ndarray] = []
        query_block_of: list[int] = []
        for index, result in enumerate(ref_results):
            intervals, dirs = _crb_blocks_and_directions(result)
            ref_blocks_intervals.append(intervals)
            ref_blocks_directions.append(
                np.stack(dirs) if dirs
                else np.zeros((0, np.concatenate(ref_reconstructed, axis=0).shape[1])))
            assignment = _window_assignment(len(result.valid_starts), intervals)
            query_block_of.extend(
                ref_offsets[index] + offset + assignment[offset]
                for offset in range(len(assignment)))
        # recompute as block ids with global cursor
        query_block_of = []
        cursor = 0
        for index, result in enumerate(ref_results):
            intervals, _dirs = _crb_blocks_and_directions(result)
            assignment = _window_assignment(len(result.valid_starts), intervals)
            for offset in range(len(assignment)):
                query_block_of.append(cursor + assignment[offset] if assignment[offset] >= 0 else -1)
            cursor += len(intervals)

        for variant in ("rage", "uniform_detect", "random_detect", "fixed_length"):
            blocks, directions, extra_k = _variant_blocks(
                source_scan, src_results, src_reconstructed, variant)
            # map source block column -> (fragment, interval)
            src_block_map = []
            for frag, intervals in enumerate(blocks):
                for interval in intervals:
                    src_block_map.append((frag, interval, 0))
            _coverage(density_record["recall"], ref_blocks_intervals,
                      None, query_block_of, directions, blocks, b_values,
                      variant, extra_k)

        results["densities"].append(density_record)
        print("[mech1] density %.3f done (ref K=%d, src K=%d)" % (
            density, ref_k, src_k), flush=True)

    # ---- one-sided replacement at the median density ------------------------
    median = densities[len(densities) // 2]
    _ref_results, ref_recon, _k = _side_crb(reference_scan, median)
    _src_results, src_recon, _k2 = _side_crb(source_scan, median)
    replacement = []
    for name, query_matrix, source_matrix in (
            ("ref_est__src_true",
             _unit_rows(np.concatenate(ref_recon, axis=0)) @ src_unit.T, true_scores),
            ("ref_true__src_est", None, None)):
        if query_matrix is None:
            query_matrix = ref_unit @ _unit_rows(np.concatenate(src_recon, axis=0)).T
        record = {"case": name, "density": median}
        for m in m_values:
            m_eff = min(m, source_count)
            overlap = []
            for query in range(query_count):
                est_top = set(np.argsort(-query_matrix[query])[:m_eff].tolist())
                true_top_set = set(true_top[m][query].tolist())
                overlap.append(len(est_top & true_top_set) / m_eff)
            record["overlap@%d" % m] = float(np.mean(overlap))
        replacement.append(record)
    results["replacement"] = replacement

    # boundary margins of the true rankings
    margins = {}
    for m in m_values:
        m_eff = min(m, source_count)
        if m_eff < source_count:
            row = true_scores[0]
            order = np.argsort(-row)
            margins["margin@%d" % m] = float(row[order[m_eff - 1]] - row[order[m_eff]])
    results["full_ranking"] = {"margins_sample_q0": margins}

    # S3.2 rows 6-7: trajectory aggregation divergence, single-segment concentration,
    # and the C3 norm table (epsilon calibration provenance).
    results["aggregation_divergence"] = trajectory_aggregation_divergence(
        source_scan, reference_scan, m_values=tuple(m_values))
    results["single_segment_share"] = single_segment_share(source_scan)
    results["gradient_norms"] = gradient_norm_stats(reference_scan, source_scan)
    return results


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETT_1H")
    parser.add_argument("--series-cap", type=int, default=2)
    parser.add_argument("--slice-length", type=int, default=4096)
    parser.add_argument("--pool-series", type=int, default=4)
    parser.add_argument("--pool-slice", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--projection-dimension", type=int, default=1024)
    parser.add_argument("--cover-density", type=float, default=None,
                        help="total density derived from shared densities is "
                             "ignored here; the sweep uses --densities")
    parser.add_argument("--densities", nargs="+", type=float,
                        default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--b-values", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--m-values", nargs="+", type=int, default=[50, 100, 500])
    parser.add_argument("--mean-epsilon", type=float, default=1e-8,
                        help="epsilon_mu (C3 calibrated value, frozen)")
    parser.add_argument("--endpoint-epsilon", type=float, default=1e-8,
                        help="epsilon_g (same calibrated value)")
    parser.add_argument("--curvature-cap", type=float, default=1.0)
    parser.add_argument("--cover-share", type=float, default=DEFAULT_SHARES[0])
    parser.add_argument("--error-share", type=float, default=DEFAULT_SHARES[1])
    parser.add_argument("--split-share", type=float, default=DEFAULT_SHARES[2])
    parser.add_argument("--out", default="outputs/mechanism1")
    parser.add_argument("--rescan", action="store_true",
                        help="recompute the full-scan cache even if it exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from rage.engines import ChronosT5Config, ChronosT5Engine
    engine = ChronosT5Engine(ChronosT5Config(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        projection_dimension=args.projection_dimension), device="cuda")

    ref_cache_path = out_dir / "scan_reference.npz"
    src_cache_path = out_dir / "scan_source.npz"
    if args.rescan or not ref_cache_path.exists():
        specs = build_task_specs()
        spec = next(s for s in specs if s.dataset == args.dataset)
        task = build_task_data(
            spec, context_length=args.context_length,
            prediction_length=args.prediction_length,
            series_cap=args.series_cap, slice_length=args.slice_length,
            pool_fragments=[])
        print("full scan: reference side (%d series)" % len(task.reference_fragments),
              flush=True)
        reference_scan = full_scan(engine, task.reference_fragments,
                                   context_length=args.context_length,
                                   prediction_length=args.prediction_length,
                                   stride=args.stride)
        save_scan(ref_cache_path, reference_scan, side="reference")
    else:
        reference_scan = load_scan(ref_cache_path, context_length=args.context_length,
                                   prediction_length=args.prediction_length,
                                   stride=args.stride)
        print("loaded reference scan cache: %s" % ref_cache_path, flush=True)

    if args.rescan or not src_cache_path.exists():
        from .data import build_candidate_fragments
        pool = build_candidate_fragments(
            window_length=args.context_length + args.prediction_length,
            slice_length=args.pool_slice,
            num_series=args.pool_series, seed=0)
        print("full scan: source side (%d series)" % len(pool), flush=True)
        source_scan = full_scan(engine, pool,
                                context_length=args.context_length,
                                prediction_length=args.prediction_length,
                                stride=args.stride)
        save_scan(src_cache_path, source_scan, side="source")
    else:
        source_scan = load_scan(src_cache_path, context_length=args.context_length,
                                prediction_length=args.prediction_length,
                                stride=args.stride)
        print("loaded source scan cache: %s" % src_cache_path, flush=True)

    started = time.time()
    results = run_mechanism1(
        reference_scan=reference_scan, source_scan=source_scan,
        densities=args.densities, b_values=args.b_values,
        m_values=tuple(args.m_values), stride=args.stride,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        mean_epsilon=args.mean_epsilon,
        endpoint_epsilon=args.endpoint_epsilon,
        curvature_cap=args.curvature_cap,
        cover_share=args.cover_share, error_share=args.error_share,
        split_share=args.split_share)
    results["protocol_cli"] = vars(args)
    results["elapsed_minutes"] = round((time.time() - started) / 60, 1)
    destination = out_dir / "mechanism1.json"
    destination.write_text(json.dumps(results, indent=2))
    print("MECHANISM1 DONE -> %s (%.1f min)" % (destination, results["elapsed_minutes"]),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
