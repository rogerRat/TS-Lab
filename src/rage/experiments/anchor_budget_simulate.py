"""Cache-only CRB grid replay for the TimeMoE anchor-budget mini-experiment.

Reads finished per-window gradients through the model-free
``assets.gradient.dataset.GradientDataset`` (NumPy only: no Torch, no GPU, no
cache writes, no locks) and replays the real CRB pipeline from ``rage.crb``
for every cover/error/split grid point.  The scan owned by main Codex
(``anchor_budget_scan.py``, tmux ``anchor-budget-scan``) computes and caches
every legal window's independent gradient; this module only replays.

Inputs follow the scan interface (SCAN_INTERFACE.md):
- ``sample_manifest.json``: flat ``samples`` list, each with task, dataset
  (``reference:<task>``), fragment_id, entity_id, rank (0/1/2 = wave),
  length, windows, input_path (reference npz), row.
- ``cache_location.json``: ``path`` = the standard GradientDataset directory.
- ``scan_status.json``: live progress; ``completed_samples`` lists finished
  trajectories, so preliminary replays can run before the full scan ends.

Metrics are reported under two semantics per the confirmed UPDATE:
1. block-constant (production): every retained window receives its block's
   ``mean_gradient`` (mean of the two endpoint anchors, per ``_make_block``);
2. endpoint linear interpolation (diagnostic only): alpha/beta weights with
   the same convention as ``CRBEngine._observed_curvature``.

Pass rule: per task, the window-weighted mean per-window cosine >= .90 on
production semantics, each of the five tasks separately.  Dropped and
zero-norm windows enter cosine means at 0 and are reported separately;
negative cosines are kept.  Window-weighted means are primary; trajectory
macro-averages and the per-trajectory mean-gradient cosine are reported
alongside and never substitute for the window mean.  A final recommendation
is emitted only in ``--mode final`` (every manifest sample complete);
preliminary replays are labelled as such.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

GRID = {
    "cover": (0.005, 0.01, 0.02, 0.04, 0.08),
    "error": (0.0, 0.0025, 0.005, 0.01, 0.02),
    "split": (0.0, 0.0025, 0.005, 0.01, 0.02),
}
SMALL_GRID = {
    "cover": (0.01, 0.08),
    "error": (0.0, 0.01),
    "split": (0.0, 0.01),
}
# Non-budget CRB parameters frozen at crb.py defaults; reported, not tuned.
FROZEN = dict(curvature_cap=1.0, mean_epsilon=1e-8, endpoint_epsilon=1e-8,
              score_epsilon=1e-8, initial_placement="longest",
              error_placement="risk", detection="standard", placement_seed=0)
AUDIT_WINDOWS = {
    "time_nem_demand_5t": 116643,
    "time_azure_i_cpu_5t": 21578,
    "time_darwin_water_15t": 34360,
    "time_finland_15t": 31585,
    "time_eweld_15t": 55875,
}
PASS_COSINE = 0.90
MAX_TOTAL_DENSITY = 0.30
# Bounded extension of the existing low-budget grid, not another Cartesian grid.
HIGH_BUDGET_POINTS = (
    (0.14, 0.02, 0.00),  # Five rank-0 trajectories pass at this 16% candidate.
    (0.12, 0.02, 0.01), (0.16, 0.03, 0.01), (0.20, 0.04, 0.01),
    (0.30, 0.00, 0.00), (0.25, 0.05, 0.00), (0.25, 0.04, 0.01),
    (0.20, 0.08, 0.02), (0.15, 0.10, 0.05),
)
DEFAULT_RUN_ROOT = Path("/root/autodl-tmp/rage/outputs/anchor-budget-20260914")


def budget_combinations(grid):
    points = list(product(grid["cover"], grid["error"], grid["split"]))
    if grid == GRID:
        points.extend(HIGH_BUDGET_POINTS)
    accepted = []
    for point in dict.fromkeys(points):
        values = [Decimal(str(value)) for value in point]
        if any(not value.is_finite() or value < 0 for value in values):
            raise ValueError("Anchor densities must be finite and nonnegative")
        if sum(values) > Decimal(str(MAX_TOTAL_DENSITY)):
            raise ValueError("cover_density + error_density + split_density must be <= 0.30")
        accepted.append(dict(zip(("cover", "error", "split"), point)))
    if not accepted:
        raise ValueError("Budget grid must not be empty")
    return accepted


def choose_budget(rows, task_names):
    """Rank complete candidates; best effort never becomes a passing result."""
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in ("cover", "error", "split"))
        grouped.setdefault(key, []).append(row)
    candidates = []
    for key, group in grouped.items():
        if {row["task"] for row in group} != set(task_names) or len(group) != len(task_names):
            continue
        anchors = sum(row["anchors_total"] for row in group)
        windows = sum(row["windows"] for row in group)
        # Endpoint minima / integer rounding must not produce an over-cap
        # recommendation even when its nominal allocation is legal.
        if anchors * 10 > windows * 3:
            continue
        means = [row["mean_cos_block"] for row in group]
        if not all(np.isfinite(value) for value in means):
            continue
        candidates.append(dict(cover=key[0], error=key[1], split=key[2],
            nominal_total_density=float(sum(Decimal(str(x)) for x in key)),
            total_unique_anchors=anchors, actual_anchor_ratio=anchors / windows,
            actual_budget_usable=True,
            worst_task_mean_cosine=min(means), macro_task_mean_cosine=float(np.mean(means)),
            task_mean_cosines={row["task"]: row["mean_cos_block"] for row in group},
            pass_block=all(row["pass_block"] for row in group),
            split_identifiable=any(row["split_used_total"] > 0 for row in group)))
    tie = lambda row: (row["total_unique_anchors"], row["cover"], row["error"], row["split"])
    passing = [row for row in candidates if row["pass_block"]]
    if passing:
        return dict(min(passing, key=tie), criterion="each task mean cosine >= .90; minimal actual unique anchors"), None
    if not candidates:
        return None, None
    best = min(candidates, key=lambda row: (-row["worst_task_mean_cosine"],
        -row["macro_task_mean_cosine"], *tie(row)))
    return None, dict(best, criterion="best effort only: maximize worst task mean, then task macro mean, then minimize actual anchors")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write(path, value) -> None:
    import tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        temporary = Path(stream.name)
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scan interface: sample manifest, cache location, live status
# ---------------------------------------------------------------------------

def load_sample_manifest(run_root) -> dict:
    """Group the flat manifest into per-task entries ordered by (rank, task)."""
    run_root = Path(run_root)
    data = json.loads((run_root / "sample_manifest.json").read_text())
    tasks, order = {}, []
    for sample in data["samples"]:
        task = sample["task"]
        if task not in tasks:
            tasks[task] = dict(task=task, dataset=sample["dataset"],
                               input_paths=[], fragments=[])
            order.append(task)
        entry = dict(task=task, dataset=sample["dataset"],
                     fragment_id=int(sample["fragment_id"]),
                     entity_id=str(sample.get("entity_id", "")),
                     row=int(sample.get("row", -1)),
                     length=int(sample.get("length", 0)),
                     windows=int(sample.get("windows", 0)),
                     wave=int(sample.get("rank", 0)),
                     input_path=sample["input_path"])
        tasks[task]["fragments"].append(entry)
        if sample["input_path"] not in tasks[task]["input_paths"]:
            tasks[task]["input_paths"].append(sample["input_path"])
    cache = None
    location = read_json(run_root / "cache_location.json")
    if location and location.get("path"):
        cache = str(location["path"])
    elif data.get("gradient_cache"):
        cache = str(data["gradient_cache"])
    if not cache:
        raise FileNotFoundError("no gradient cache path in cache_location.json")
    return dict(run_root=str(run_root), source=str(run_root / "sample_manifest.json"),
                seed=int(data.get("seed", 20260914)),
                total_windows=int(data.get("total_windows", 0)),
                formal_exam_used=bool(data.get("formal_exam_used", False)),
                cache=cache, tasks=[tasks[name] for name in order])


def resolve_cache(run_root) -> str:
    location = read_json(Path(run_root) / "cache_location.json")
    if not location or not location.get("path"):
        raise FileNotFoundError("cache_location.json missing or has no path")
    return str(location["path"])


def load_entry_fragment(entry: dict):
    from rage.experiments.development_inputs import load_fragments
    fragments = load_fragments(entry["input_path"])
    for fragment in fragments:
        if int(fragment.fragment_id) == entry["fragment_id"]:
            return fragment
    row = entry.get("row", -1)
    if 0 <= row < len(fragments):
        return fragments[row]
    raise KeyError(f"fragment {entry['fragment_id']} missing in {entry['input_path']}")


def manifest_entries(manifest: dict, waves=None) -> list[dict]:
    entries = []
    for task_index, task in enumerate(manifest["tasks"]):
        for fragment in task["fragments"]:
            if waves is not None and int(fragment["wave"]) not in waves:
                continue
            entries.append(dict(task_index=task_index, **fragment))
    entries.sort(key=lambda e: (e["wave"], e["task_index"]))
    return entries


def completed_keys(run_root) -> set[tuple]:
    status = read_json(Path(run_root) / "scan_status.json")
    if not status:
        return set()
    return {(s.get("task"), int(s.get("fragment_id", -1)), int(s.get("rank", 0)))
            for s in (status.get("completed_samples") or [])}


# ---------------------------------------------------------------------------
# Read-only CRB gradient computer over GradientDataset
# ---------------------------------------------------------------------------

class DatasetGradientComputer:
    """Serve CRB anchor requests from finished cache rows; zero model calls."""

    def __init__(self, dataset, dataset_name: str):
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.served_rows = 0

    def __call__(self, fragment, starts):
        starts = np.asarray(starts, dtype=np.int64)
        ids = np.column_stack([np.full(len(starts), int(fragment.fragment_id)), starts])
        gradients = self.dataset.lookup(self.dataset_name, ids)
        self.served_rows += len(ids)
        return gradients

    def compute_many(self, entries):
        return np.concatenate([self(fragment, starts) for fragment, starts in entries])


class ArrayGradientComputer:
    """Serve exact cached rows already loaded for evaluation, without shard rereads."""

    def __init__(self, starts, gradients, fragment_id):
        self.starts = np.asarray(starts, dtype=np.int64)
        self.gradients = np.asarray(gradients, dtype=np.float32)
        self.fragment_id = int(fragment_id)
        self.served_rows = 0

    def __call__(self, fragment, starts):
        starts = np.asarray(starts, dtype=np.int64)
        rows = np.searchsorted(self.starts, starts)
        if int(fragment.fragment_id) != self.fragment_id or np.any(rows >= len(self.starts)):
            raise KeyError('Anchor not present in this cached trajectory')
        if not np.array_equal(self.starts[rows], starts):
            raise KeyError('Anchor start does not match a cached window')
        self.served_rows += len(rows)
        return self.gradients[rows]


def load_gradient_dataset(cache_dir):
    from assets.gradient.dataset import GradientDataset
    return GradientDataset(cache_dir)


def check_rows_complete(dataset, entries, log=print) -> None:
    """Refuse to replay any trajectory whose windows are not fully cached."""
    positions = getattr(dataset, "_positions", {})
    missing = {}
    for entry in entries:
        fragment = load_entry_fragment(entry)
        starts = fragment.valid_starts(512, 64, 1)
        if entry.get("windows") and len(starts) != entry["windows"]:
            log(f"[check] WARNING {entry['task']} fid{entry['fragment_id']}: manifest "
                f"windows {entry['windows']} != actual {len(starts)}")
        absent = sum(1 for start in starts
                     if (entry["dataset"], int(entry["fragment_id"]), int(start))
                     not in positions)
        if absent:
            missing[f"{entry['task']}__fid{entry['fragment_id']}"] = absent
    if missing:
        raise RuntimeError("gradient cache incomplete (replay after the scan catches "
                           f"up): {missing}")


# ---------------------------------------------------------------------------
# Reconstruction metrics, dual semantics
# ---------------------------------------------------------------------------

def evaluate_reconstruction(truth64: np.ndarray, truth_norm: np.ndarray,
                            result) -> dict:
    """Per-window metrics under block-constant and interpolation semantics."""
    n = int(truth64.shape[0])
    if len(result.valid_starts) != n:
        raise ValueError("result windows do not match truth rows")
    if not np.array_equal(result.valid_starts, np.arange(1, n + 1, dtype=np.int32)):
        raise ValueError("stride-1 start mapping assumption broken")
    block_cos = np.zeros(n)
    interp_cos = np.zeros(n)
    block_rel = np.zeros(n)
    interp_rel = np.zeros(n)
    rel_defined = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    zero_recon_block = zero_recon_interp = 0
    mean_acc = np.zeros(truth64.shape[1], dtype=np.float64)
    size_acc = 0
    single_window = n == 1
    if single_window:
        # total == 1: CRB computes the window itself and emits no blocks.
        gradient = result.anchors.get(0)
        if gradient is None:
            raise ValueError("single legal window without its anchor")
        vector = gradient.astype(np.float64)
        covered[0] = True
        tn = float(truth_norm[0])
        vn = float(np.sqrt(vector @ vector))
        if tn > 0 and vn > 0:
            dot = float(truth64[0] @ vector)
            block_cos[0] = interp_cos[0] = dot / (tn * vn)
            block_rel[0] = interp_rel[0] = float(np.linalg.norm(vector - truth64[0])) / tn
            rel_defined[0] = True
        mean_acc, size_acc = vector, 1
    for block in ([] if single_window else result.blocks):
        left, right = int(block.left_index), int(block.right_index)
        stop = right + 1 if right == n - 1 else right  # final interval owns right
        rows = block.start_positions.astype(np.int64) - 1
        if not np.array_equal(rows, np.arange(left, stop)):
            raise ValueError("block rows deviate from contiguous start range")
        if covered[rows].any():
            raise ValueError("overlapping blocks")
        covered[rows] = True
        a_left = result.anchors[left].astype(np.float64)
        a_right = result.anchors[right].astype(np.float64)
        mean = block.mean_gradient.astype(np.float64)
        window = truth64[left:stop]
        d_left = window @ a_left
        d_right = window @ a_right
        d_mean = window @ mean
        span = right - left
        offset = (rows - left).astype(np.float64)
        alpha = (span - offset) / span
        beta = offset / span
        numerator = alpha * d_left + beta * d_right
        left_sq = float(a_left @ a_left)
        right_sq = float(a_right @ a_right)
        cross = float(a_left @ a_right)
        interp_sq = alpha * alpha * left_sq + beta * beta * right_sq + 2.0 * alpha * beta * cross
        interp_norm = np.sqrt(np.maximum(interp_sq, 0.0))
        mean_norm = float(np.sqrt(mean @ mean))
        tn = truth_norm[rows]
        good_block = (tn > 0) & (mean_norm > 0)
        good_interp = (tn > 0) & (interp_norm > 0)
        zero_recon_block += int(np.sum(~good_block & (tn > 0)))
        zero_recon_interp += int(np.sum(~good_interp & (tn > 0)))
        block_cos[rows] = np.where(good_block, d_mean / np.where(good_block, tn * mean_norm, 1.0), 0.0)
        interp_cos[rows] = np.where(good_interp, numerator / np.where(good_interp, tn * interp_norm, 1.0), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            block_res = np.sqrt(np.maximum(mean_norm * mean_norm - 2.0 * d_mean + tn * tn, 0.0)) / tn
            interp_res = np.sqrt(np.maximum(interp_sq - 2.0 * numerator + tn * tn, 0.0)) / tn
        valid = tn > 0
        block_rel[rows] = np.where(valid, block_res, 0.0)
        interp_rel[rows] = np.where(valid, interp_res, 0.0)
        rel_defined[rows] = valid
        mean_acc = mean_acc + float(len(rows)) * mean
        size_acc += len(rows)
    truth_mean = truth64.mean(axis=0)
    truth_mean_norm = float(np.linalg.norm(truth_mean))
    if size_acc > 0 and truth_mean_norm > 0:
        weighted = mean_acc / size_acc
        weighted_norm = float(np.linalg.norm(weighted))
        traj_cos = (float(weighted @ truth_mean) / (weighted_norm * truth_mean_norm)
                    if weighted_norm > 0 else 0.0)
    else:
        traj_cos = 0.0
    return dict(block_cos=block_cos, interp_cos=interp_cos,
                block_rel=block_rel, interp_rel=interp_rel,
                rel_defined=rel_defined, covered=covered,
                dropped_windows=n - int(covered.sum()),
                zero_truth=int(np.sum(truth_norm <= 0)),
                zero_recon_block=zero_recon_block,
                zero_recon_interp=zero_recon_interp,
                anchors_unique=int(len(result.anchors)),
                cover_used=int(result.cover_computations),
                error_used=int(result.error_computations),
                split_used=int(result.split_computations),
                computed_gradients=int(result.computed_gradients),
                blocks=int(len(result.blocks)),
                retained=int(covered.sum()),
                no_signal=int(result.discarded_no_signal),
                antagonistic=int(result.discarded_antagonistic),
                banned=int(result.banned_budget),
                single_window=bool(single_window),
                trajectory_mean_cosine=float(traj_cos))


def _array_stats(values) -> dict:
    values = np.asarray(values)
    if len(values) == 0:
        return dict(count=0, mean=None, median=None, p10=None, frac_ge_090=None)
    return dict(count=int(len(values)), mean=float(np.mean(values)),
                median=float(np.median(values)),
                p10=float(np.percentile(values, 10)),
                frac_ge_090=float(np.mean(values >= PASS_COSINE)))


# ---------------------------------------------------------------------------
# Grid replay
# ---------------------------------------------------------------------------

def run_simulate(manifest: dict, out_dir, *, entries, grid=None, mode="final",
                 log=print) -> dict:
    from rage.crb import CRBConfig, CRBEngine

    raise RuntimeError("Retired block-constant budget protocol: use anchor_budget_linear with a matching frozen p10 calibration; historical budgets are invalid for window reconstruction")
    grid = GRID if grid is None else grid
    combos = budget_combinations(grid)  # Reject invalid budgets before cache reads or output writes.
    out_dir = Path(out_dir)
    (out_dir / "detail").mkdir(parents=True, exist_ok=True)
    dataset = load_gradient_dataset(manifest["cache"])
    check_rows_complete(dataset, entries, log=log)
    rows, split_trigger_total = [], 0
    anchor_totals: dict[tuple, int] = {}
    task_names = []
    included = [dict(task=e["task"], fragment_id=e["fragment_id"], wave=e["wave"],
                     windows=e["windows"]) for e in entries]
    for task in manifest["tasks"]:
        task_entries = [e for e in entries if e["task"] == task["task"]]
        if not task_entries:
            continue
        task_names.append(task["task"])
        pooled = {i: dict(block=[], interp=[], rel_b=[], rel_i=[], macro=[],
                          anchors=0, split=0, dropped=0, zero_truth=0,
                          single=[]) for i in range(len(combos))}
        for entry in task_entries:
            fragment = load_entry_fragment(entry)
            starts = fragment.valid_starts(512, 64, 1)
            ids = np.column_stack([np.full(len(starts), entry["fragment_id"]), starts])
            truth = np.asarray(dataset.lookup(entry["dataset"], ids), dtype=np.float32)
            truth64 = truth.astype(np.float64)
            truth_norm = np.sqrt(np.einsum("ij,ij->i", truth64, truth64))
            computer = ArrayGradientComputer(starts, truth, entry["fragment_id"])
            detail_rows = []
            for index, combo in enumerate(combos):
                config = CRBConfig(context_length=512, prediction_length=64,
                                   window_stride=1, cover_density=combo["cover"],
                                   error_density=combo["error"],
                                   split_density=combo["split"], **FROZEN)
                result = CRBEngine(config, computer).process_fragment(fragment)
                metrics = evaluate_reconstruction(truth64, truth_norm, result)
                pooled[index]["block"].append(metrics["block_cos"])
                pooled[index]["interp"].append(metrics["interp_cos"])
                pooled[index]["rel_b"].append(metrics["block_rel"][metrics["rel_defined"]])
                pooled[index]["rel_i"].append(metrics["interp_rel"][metrics["rel_defined"]])
                pooled[index]["macro"].append(float(np.mean(metrics["block_cos"])))
                pooled[index]["anchors"] += metrics["anchors_unique"]
                pooled[index]["split"] += metrics["split_used"]
                pooled[index]["dropped"] += metrics["dropped_windows"]
                pooled[index]["zero_truth"] += metrics["zero_truth"]
                pooled[index]["single"].append(metrics["single_window"])
                split_trigger_total += metrics["split_used"]
                detail_rows.append(dict(
                    **combo, nominal_total_density=float(sum(Decimal(str(v)) for v in combo.values())),
                    fragment_id=entry["fragment_id"], wave=entry["wave"],
                    windows=int(truth64.shape[0]),
                    anchors_unique=metrics["anchors_unique"],
                    actual_anchor_ratio=metrics["anchors_unique"] / len(starts),
                    actual_budget_usable=metrics["anchors_unique"] * 10 <= len(starts) * 3,
                    cover_used=metrics["cover_used"], error_used=metrics["error_used"],
                    split_used=metrics["split_used"],
                    computed_gradients=metrics["computed_gradients"],
                    blocks=metrics["blocks"], retained=metrics["retained"],
                    dropped_windows=metrics["dropped_windows"],
                    no_signal=metrics["no_signal"],
                    antagonistic=metrics["antagonistic"], banned=metrics["banned"],
                    zero_truth=metrics["zero_truth"],
                    zero_recon_block=metrics["zero_recon_block"],
                    zero_recon_interp=metrics["zero_recon_interp"],
                    single_window=metrics["single_window"],
                    trajectory_mean_cosine=metrics["trajectory_mean_cosine"],
                    block=_array_stats(metrics["block_cos"]),
                    interp=_array_stats(metrics["interp_cos"]),
                    block_rel=_array_stats(metrics["block_rel"][metrics["rel_defined"]]),
                    interp_rel=_array_stats(metrics["interp_rel"][metrics["rel_defined"]])))
            del truth, truth64
            atomic_write(out_dir / "detail" /
                         f"{task['task']}__fid{entry['fragment_id']}.json", detail_rows)
            log(f"[simulate] {task['task']} fid{entry['fragment_id']} grid done "
                f"({len(combos)} combos)")
        for index, combo in enumerate(combos):
            block_all = np.concatenate(pooled[index]["block"])
            interp_all = np.concatenate(pooled[index]["interp"])
            rel_b = np.concatenate(pooled[index]["rel_b"]) if pooled[index]["rel_b"] else np.array([])
            rel_i = np.concatenate(pooled[index]["rel_i"]) if pooled[index]["rel_i"] else np.array([])
            windows = int(len(block_all))
            anchors = int(pooled[index]["anchors"])
            rows.append(dict(task=task["task"], **combo, windows=windows,
                             nominal_total_density=float(sum(Decimal(str(v)) for v in combo.values())),
                             anchors_total=anchors,
                             actual_budget_usable=anchors * 10 <= windows * 3,
                             anchors_ratio=round(anchors / windows, 6),
                             mean_cos_block=float(np.mean(block_all)),
                             median_cos_block=round(float(np.median(block_all)), 6),
                             p10_cos_block=round(float(np.percentile(block_all, 10)), 6),
                             frac_ge_090_block=round(float(np.mean(block_all >= PASS_COSINE)), 6),
                             mean_rel_l2_block=round(float(np.mean(rel_b)) if len(rel_b) else float("nan"), 6),
                             mean_cos_interp=round(float(np.mean(interp_all)), 6),
                             median_cos_interp=round(float(np.median(interp_all)), 6),
                             p10_cos_interp=round(float(np.percentile(interp_all, 10)), 6),
                             frac_ge_090_interp=round(float(np.mean(interp_all >= PASS_COSINE)), 6),
                             mean_rel_l2_interp=round(float(np.mean(rel_i)) if len(rel_i) else float("nan"), 6),
                             macro_mean_cos_block=round(float(np.mean(pooled[index]["macro"])), 6),
                             dropped_windows=int(pooled[index]["dropped"]),
                             zero_truth=int(pooled[index]["zero_truth"]),
                             split_used_total=int(pooled[index]["split"]),
                             single_window_any=bool(any(pooled[index]["single"])),
                             pass_block=bool(np.mean(block_all) >= PASS_COSINE)))
            anchor_totals[(index, task["task"])] = anchors
    pass_counts: dict[tuple, set] = {}
    for row in rows:
        key = (row["cover"], row["error"], row["split"])
        pass_counts.setdefault(key, set())
        if row["pass_block"]:
            pass_counts[key].add(row["task"])
    all_tasks = {t["task"] for t in manifest["tasks"]}
    complete_coverage = set(task_names) == all_tasks and all(
        len([e for e in entries if e["task"] == t["task"]])
        == len(t["fragments"]) for t in manifest["tasks"])
    all_pass = sorted(key for key, tasks in pass_counts.items()
                      if len(tasks) == len(task_names))
    candidate_costs = []
    for combo in combos:
        group = [row for row in rows if all(row[name] == value for name, value in combo.items())]
        anchors = sum(row["anchors_total"] for row in group)
        windows = sum(row["windows"] for row in group)
        candidate_costs.append(dict(**combo, total_unique_anchors=anchors,
            total_windows=windows, actual_anchor_ratio=anchors / windows if windows else None,
            actual_budget_usable=bool(windows and anchors * 10 <= windows * 3)))
    usable_keys = {tuple(row[name] for name in ("cover", "error", "split"))
                   for row in candidate_costs if row["actual_budget_usable"]}
    all_pass = [key for key in all_pass if key in usable_keys]
    recommendation = best_effort = None
    if mode == "final" and complete_coverage:
        recommendation, best_effort = choose_budget(rows, task_names)
    summary = dict(schema="anchor-budget-simulate-v2", generated_at=now_iso(),
                   budget_limit=dict(max_total_density=MAX_TOTAL_DENSITY,
                       expression="cover_density + error_density + split_density <= 0.30",
                       search="finite grid only; no automatic expansion",
                       actual_ratio_note="actual unique anchors/windows reported separately; integer rounding and required endpoints can differ from nominal density"),
                   evaluated_combos=combos,
                   candidate_actual_costs=candidate_costs,
                   mode=mode, complete_coverage=bool(complete_coverage),
                   included_samples=included, cache=str(manifest["cache"]),
                   manifest=manifest["source"], frozen=FROZEN,
                   pass_rule=dict(metric="pooled_window_weighted_mean_cosine",
                                  semantics="block_constant_production",
                                  threshold=PASS_COSINE,
                                  per_task="each sampled task must pass",
                                  dropped_or_zero_norm_cosine=0.0,
                                  interpolation="diagnostic only",
                                  negative_cosines="kept, not clipped"),
                   cosine_numerics="float64 gemv on float64-cast BF16-decoded "
                                   "truth; consistent with float64-norm-exact-unit-v1",
                   combos_evaluated=len(combos), tasks=task_names,
                   all_manifest_tasks=sorted(all_tasks),
                   split_trigger_total=split_trigger_total,
                   split_identifiable=bool(split_trigger_total > 0),
                   passing_combos=[dict(cover=k[0], error=k[1], split=k[2]) for k in all_pass],
                   recommendation=recommendation,
                   best_effort=best_effort,
                   split_note=("Split events observed; consult candidate-specific split counts" if split_trigger_total else
                               "No split triggered: split_density is not identifiable in this sample; do not claim an optimum"),
                   note=(None if complete_coverage else
                         "PRELIMINARY: not all five tasks/trajectories complete; "
                         "do not read as a five-task final recommendation"))
    atomic_write(out_dir / "summary.json", summary)
    atomic_write(out_dir / "grid_rows.json", rows)
    if rows:
        with open(out_dir / "grid_rows.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    log(f"[simulate] wrote {out_dir / 'summary.json'}: {len(all_pass)} passing "
        f"combos, split_trigger_total={split_trigger_total}, "
        f"complete_coverage={complete_coverage}")
    return summary


# ---------------------------------------------------------------------------
# CPU-only self test against a synthetic cache in the real shard format
# ---------------------------------------------------------------------------

def _encode_bf16(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def run_self_test(directory, log=print) -> dict:
    """Build a synthetic GradientDataset and verify the full replay path.

    Gradients are smooth in the window start so interpolation must dominate
    block-constant reconstruction; zero model/GPU is involved.  Validates the
    shard/index reader, CRB serving, dual-semantics metrics, and outputs.
    """
    from rage.crb import CRBEngine
    from rage.core import TimeSeriesFragment

    directory = Path(directory)
    cache = directory / "grad_cache" / "synthetic--selftest"
    if not (cache / "window_index.npz").exists():
        cache.mkdir(parents=True, exist_ok=True)
        fragment = TimeSeriesFragment((np.sin(np.arange(3000) / 97.0) * 3
                                       + np.cos(np.arange(3000) / 13.0)).astype(np.float32),
                                      entity_id="synthetic", fragment_id=0)
        starts = fragment.valid_starts(512, 64, 1)
        total = len(starts)
        rows_all = np.column_stack([np.zeros(total, np.int64), starts])

        def truth_for(rows):
            s = rows[:, 1].astype(np.float64)
            return np.stack([np.ones(len(s)), s * 0.001,
                             np.sin(s / 500.0) * 5.0], axis=1)

        dataset_name = "reference:selftest"
        shard_size = 1024
        for shard_index, begin in enumerate(range(0, total, shard_size)):
            rows = rows_all[begin:begin + shard_size]
            with open(cache / f"windows-{shard_index:08d}.npz", "wb") as stream:
                np.savez(stream, dataset=dataset_name, window_ids=rows,
                         gradients_bf16=_encode_bf16(truth_for(rows)),
                         losses=np.arange(len(rows), dtype=np.float32) + begin)
        import tempfile
        shard_count = (total + shard_size - 1) // shard_size
        with tempfile.NamedTemporaryFile(dir=cache, delete=False, suffix=".npz") as stream:
            np.savez(stream, datasets=np.asarray([dataset_name], dtype=str),
                     dataset_ids=np.zeros(total, np.int32),
                     window_ids=rows_all,
                     shards=np.asarray([f"windows-{i:08d}.npz" for i in range(shard_count)],
                                       dtype=str),
                     shard_ids=(np.arange(total) // shard_size).astype(np.int32),
                     shard_rows=(np.arange(total) % shard_size).astype(np.int32))
            Path(stream.name).replace(cache / "window_index.npz")
        atomic_write(cache / "cache_manifest.json",
                     dict(format="window-gradients-bf16-v1", data={"purpose": "selftest"},
                          model={"model_id": "synthetic"}, context_length=512,
                          prediction_length=64))
        np.save(directory / "synthetic_fragment.npy", fragment.data)
    fragment_values = np.load(directory / "synthetic_fragment.npy")
    fragment = TimeSeriesFragment(fragment_values, entity_id="synthetic", fragment_id=0)
    starts = fragment.valid_starts(512, 64, 1)
    total = len(starts)
    rows_all = np.column_stack([np.zeros(total, np.int64), starts])

    def truth_for(rows):
        s = rows[:, 1].astype(np.float64)
        return np.stack([np.ones(len(s)), s * 0.001, np.sin(s / 500.0) * 5.0], axis=1)

    dataset = load_gradient_dataset(cache)
    manifest = dict(run_root=str(directory), source="selftest", seed=20260914,
                    total_windows=total, formal_exam_used=False, cache=str(cache),
                    tasks=[dict(task="selftest", dataset="reference:selftest",
                                input_paths=[], fragments=[
                                    dict(task="selftest", dataset="reference:selftest",
                                         fragment_id=0, entity_id="synthetic", row=0,
                                         length=3000, windows=total, wave=0,
                                         input_path=str(directory / "synthetic_fragment.npy"))])])
    # Patch within this module's globals so the override also applies when the
    # file runs as __main__ (a separate module object from the importable one).
    original = globals()["load_entry_fragment"]
    globals()["load_entry_fragment"] = lambda entry: fragment
    try:
        summary = run_simulate(manifest, directory / "out",
                               entries=manifest_entries(manifest), grid=SMALL_GRID,
                               log=log)
        detail = json.loads((directory / "out" / "detail" /
                             "selftest__fid0.json").read_text())
    finally:
        globals()["load_entry_fragment"] = original
    config = _selftest_config()

    def linear_computer(frag, starts):
        starts = np.asarray(starts, dtype=np.float64)
        return np.stack([np.ones(len(starts)), starts * 0.001], axis=1).astype(np.float32)

    linear = CRBEngine(config, linear_computer).process_fragment(fragment)
    linear_truth = np.stack([np.ones(total),
                             rows_all[:, 1].astype(np.float64) * 0.001], axis=1)
    linear_norm = np.sqrt(np.einsum("ij,ij->i", linear_truth, linear_truth))
    metrics = evaluate_reconstruction(linear_truth, linear_norm, linear)
    retained = metrics["covered"]
    checks = dict(
        generated_at=now_iso(), windows=total,
        interp_above_block=bool(all(row["interp"]["mean"] >= row["block"]["mean"] - 1e-9
                                    for row in detail)),
        interp_exact_on_linear=bool(np.all(metrics["interp_cos"][retained] > 0.999999)),
        rows_written=bool((directory / "out" / "grid_rows.csv").exists()),
        summary_written=bool((directory / "out" / "summary.json").exists()),
        block_not_trivially_one=bool(max(row["block"]["mean"] for row in detail) < 1.0),
        anchors_positive=bool(all(row["anchors_unique"] > 0 for row in detail)))
    checks["all_passed"] = bool(all(value for key, value in checks.items()
                                    if isinstance(value, bool)))
    atomic_write(directory / "selftest_report.json", checks)
    for key, value in checks.items():
        log(f"[selftest] {key} = {value}")
    return checks


def _selftest_config():
    from rage.crb import CRBConfig
    return CRBConfig(context_length=512, prediction_length=64, cover_density=0.01,
                     error_density=0.005, split_density=0.005, **FROZEN)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", choices=("final", "preliminary"), default="final")
    parser.add_argument("--waves", default="all", help="'all' or comma list, e.g. '0'")
    parser.add_argument("--grid", choices=("full", "small"), default="full")
    parser.add_argument("--self-test", dest="self_test", default=None,
                        help="run the CPU self test into this directory and exit")
    parser.add_argument("--check-only", action="store_true",
                        help="only verify requested rows are complete in the cache")
    args = parser.parse_args(argv)
    if args.self_test:
        report = run_self_test(args.self_test)
        return 0 if report["all_passed"] else 1
    manifest = load_sample_manifest(args.run_root)
    waves = None if args.waves == "all" else {int(w) for w in args.waves.split(",") if w}
    if args.mode == "final":
        entries = manifest_entries(manifest, waves)
        out = Path(args.out) if args.out else Path(args.run_root) / "simulate_final"
    else:
        done = completed_keys(args.run_root)
        entries = [e for e in manifest_entries(manifest, waves)
                   if (e["task"], e["fragment_id"], e["wave"]) in done]
        if not entries:
            raise SystemExit("no completed samples yet; nothing to replay")
        tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = Path(args.out) if args.out else Path(args.run_root) / f"simulate_prelim_{tag}"
    grid = GRID if args.grid == "full" else SMALL_GRID
    if args.check_only:
        check_rows_complete(load_gradient_dataset(manifest["cache"]), entries)
        print(f"[check] all {len(entries)} requested trajectories complete in cache")
        return 0
    run_simulate(manifest, out, entries=entries, grid=grid, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
