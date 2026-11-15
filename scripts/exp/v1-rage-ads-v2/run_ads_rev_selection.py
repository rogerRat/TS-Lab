#!/usr/bin/env python3
"""Build the reverse-match ADS arm and persist exact match-support counts."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from experiment_config import TASKS
import run_selection as base

ROOT = PROJECT_ROOT
OUTPUT_ROOT = ROOT / "outputs/selection/v1-rage-ads-v2"
REFERENCE_ROOT = base.REFERENCE_ROOT
Q_REVERSE = 16
THRESHOLD = 0.1


def match_statistics(reference, source, histogram_bins=None):
    """Return threshold hits, source maxima, matrix size, and optional cosine histogram."""
    import torch

    ref_rows = [(owner, pos, vec) for owner, state in enumerate(reference)
                for pos, vec in sorted(state["vectors"].items())]
    src_rows = [(owner, pos, vec) for owner, state in enumerate(source)
                for pos, vec in sorted(state["vectors"].items())]
    if not ref_rows or not src_rows:
        raise ValueError("similarity inputs must be non-empty")
    ref_values = np.stack([row[2] for row in ref_rows]).astype(np.float32, copy=False)
    src_values = np.stack([row[2] for row in src_rows]).astype(np.float32, copy=False)
    ref_norms = np.linalg.norm(ref_values, axis=1)
    src_norms = np.linalg.norm(src_values, axis=1)
    np.divide(ref_values, ref_norms[:, None], out=ref_values,
              where=ref_norms[:, None] > 0)
    np.divide(src_values, src_norms[:, None], out=src_values,
              where=src_norms[:, None] > 0)
    ref_hit_gt = np.zeros(len(ref_rows), dtype=bool)
    src_hit_gt = np.zeros(len(src_rows), dtype=bool)
    ref_hit_ge = np.zeros(len(ref_rows), dtype=bool)
    src_hit_ge = np.zeros(len(src_rows), dtype=bool)
    maxima = np.full(len(src_rows), -np.inf, dtype=np.float32)
    hit_pairs_gt = 0
    hit_pairs_ge = 0
    if histogram_bins is not None:
        if int(histogram_bins) <= 0:
            raise ValueError("histogram_bins must be positive")
        histogram_bins = int(histogram_bins)
        histogram_edges = np.linspace(-1.0, 1.0, histogram_bins + 1)
        histogram = np.zeros(histogram_bins, dtype=np.int64)
    else:
        histogram_edges = None
        histogram = None
    if torch.cuda.is_available():
        dimension = int(ref_values.shape[1])
        required = (len(ref_rows) + len(src_rows)) * dimension * 4
        free_bytes, _ = torch.cuda.mem_get_info()
        reserve_bytes = 2 * 1024**3 + 1024 * 8192 * 4
        use_cuda = required + reserve_bytes <= free_bytes
    else:
        use_cuda = False
    if use_cuda:
        device = torch.device("cuda")
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            refs = torch.from_numpy(ref_values).to(device)
            srcs = torch.from_numpy(src_values).to(device)
            ref_flags_gt = torch.zeros(len(ref_rows), dtype=torch.bool, device=device)
            src_flags_gt = torch.zeros(len(src_rows), dtype=torch.bool, device=device)
            ref_flags_ge = torch.zeros(len(ref_rows), dtype=torch.bool, device=device)
            src_flags_ge = torch.zeros(len(src_rows), dtype=torch.bool, device=device)
            max_scores = torch.full((len(src_rows),), -torch.inf, dtype=torch.float32, device=device)
            hit_total_gt = torch.zeros((), dtype=torch.int64, device=device)
            hit_total_ge = torch.zeros((), dtype=torch.int64, device=device)
            for left in range(0, len(ref_rows), 1024):
                rblock = refs[left:min(left + 1024, len(ref_rows))]
                for right in range(0, len(src_rows), 8192):
                    block = torch.matmul(rblock, srcs[right:min(right + 8192, len(src_rows))].T)
                    hits_gt = block > THRESHOLD
                    hits_ge = block >= THRESHOLD
                    width = min(8192, len(src_rows) - right)
                    ref_flags_gt[left:left + len(rblock)] |= hits_gt.any(dim=1)
                    src_flags_gt[right:right + width] |= hits_gt.any(dim=0)
                    ref_flags_ge[left:left + len(rblock)] |= hits_ge.any(dim=1)
                    src_flags_ge[right:right + width] |= hits_ge.any(dim=0)
                    max_scores[right:right + width] = torch.maximum(
                        max_scores[right:right + width], block.max(dim=0).values)
                    hit_total_gt += hits_gt.sum()
                    hit_total_ge += hits_ge.sum()
                    if histogram is not None:
                        scaled = torch.floor((block.clamp(-1.0, 1.0) + 1.0)
                                             * (histogram_bins / 2.0)).to(torch.int64)
                        scaled.clamp_(0, histogram_bins - 1)
                        histogram += torch.bincount(
                            scaled.reshape(-1), minlength=histogram_bins).cpu().numpy()
            ref_hit_gt = ref_flags_gt.cpu().numpy()
            src_hit_gt = src_flags_gt.cpu().numpy()
            ref_hit_ge = ref_flags_ge.cpu().numpy()
            src_hit_ge = src_flags_ge.cpu().numpy()
            maxima = max_scores.cpu().numpy()
            hit_pairs_gt = int(hit_total_gt.item())
            hit_pairs_ge = int(hit_total_ge.item())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    else:
        for left in range(0, len(ref_rows), 128):
            rblock = ref_values[left:left + 128]
            for right in range(0, len(src_rows), 2048):
                sblock = src_values[right:right + 2048]
                block = rblock @ sblock.T
                hits_gt = block > THRESHOLD
                hits_ge = block >= THRESHOLD
                hit_pairs_gt += int(hits_gt.sum())
                hit_pairs_ge += int(hits_ge.sum())
                ref_hit_gt[left:left + len(rblock)] |= hits_gt.any(axis=1)
                src_hit_gt[right:right + len(sblock)] |= hits_gt.any(axis=0)
                ref_hit_ge[left:left + len(rblock)] |= hits_ge.any(axis=1)
                src_hit_ge[right:right + len(sblock)] |= hits_ge.any(axis=0)
                maxima[right:right + len(sblock)] = np.maximum(
                    maxima[right:right + len(sblock)], block.max(axis=0))
                if histogram is not None:
                    histogram += np.histogram(block, bins=histogram_edges)[0]
    ref_hits_gt = {(row[0], row[1]) for row, yes in zip(ref_rows, ref_hit_gt) if yes}
    src_hits_gt = {(row[0], row[1]) for row, yes in zip(src_rows, src_hit_gt) if yes}
    ref_hits_ge = {(row[0], row[1]) for row, yes in zip(ref_rows, ref_hit_ge) if yes}
    src_hits_ge = {(row[0], row[1]) for row, yes in zip(src_rows, src_hit_ge) if yes}
    scores = {}
    for row, value in zip(src_rows, maxima):
        scores.setdefault(row[0], {})[row[1]] = float(value)
    result = (ref_hits_gt, src_hits_gt, scores, len(ref_rows) * len(src_rows),
              hit_pairs_gt, ref_hits_ge, src_hits_ge, hit_pairs_ge)
    if histogram is None:
        return result
    return result + (histogram.tolist(), histogram_edges.tolist())


def reverse_refine(states, matched, spacing):
    plans = []
    by_owner = {}
    for owner, position in matched:
        by_owner.setdefault(owner, set()).add(position)
    for owner, state in enumerate(states):
        anchors = sorted(state["vectors"])
        positive = by_owner.get(owner, set())
        unmatched_indices = [i for i, position in enumerate(anchors) if position not in positive]
        regions = []
        if unmatched_indices:
            first = previous = unmatched_indices[0]
            for index in unmatched_indices[1:]:
                if index != previous + 1:
                    regions.append((anchors[first], anchors[previous]))
                    first = index
                previous = index
            regions.append((anchors[first], anchors[previous]))
        plan = set()
        for left, right in regions:
            plan.update(base.grid_region(left, right, spacing))
        plans.append(plan)
    return plans


def run_task(task, sources, legal, probe, identity, logger):
    task_out = OUTPUT_ROOT / task / "ads-rev"
    if (task_out / ".complete").is_file():
        logger.info("stage=task-skip task=%s arm=ads-rev reason=complete", task)
        return
    ref_dir = REFERENCE_ROOT / task
    ref_manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="ascii"))
    if not (ref_dir / ".complete").is_file() or ref_manifest.get("answer_values_included") is not False:
        raise ValueError("reference input invalid for task=" + task)
    references = base.load_fragments(ref_dir / "references.npz")
    ref_legal = np.asarray([len(item.valid_starts(base.CONTEXT, base.PREDICTION))
                            for item in references], dtype=np.int64)
    source, src_plans = base.task_state(sources, legal, base.Q0)
    reference, ref_plans = base.task_state(references, ref_legal, base.Q0)
    started = time.monotonic()
    base.get_new_gradients(source, src_plans, probe, "source", logger, task, "coarse")
    base.get_new_gradients(reference, ref_plans, probe, "reference", logger, task, "coarse")
    coarse_source_anchors = sum(len(state["vectors"]) for state in source)
    coarse_reference_anchors = sum(len(state["vectors"]) for state in reference)
    (_ref_gt, _src_gt, _scores, coarse_cells, coarse_pairs_gt,
     ref_excluded, src_excluded, coarse_pairs_ge) = match_statistics(reference, source)
    logger.info("stage=ads-rev-match task=%s stage=coarse cells=%d pairs_gt_0_1=%d pairs_ge_0_1=%d source_excluded=%d reference_excluded=%d",
                task, coarse_cells, coarse_pairs_gt, coarse_pairs_ge,
                len(src_excluded), len(ref_excluded))
    reverse_src_plans = reverse_refine(source, src_excluded, Q_REVERSE)
    reverse_ref_plans = reverse_refine(reference, ref_excluded, Q_REVERSE)
    reverse_src = base.get_new_gradients(source, reverse_src_plans, probe, "source", logger, task, "reverse-q16")
    reverse_ref = base.get_new_gradients(reference, reverse_ref_plans, probe, "reference", logger, task, "reverse-q16")
    (final_ref_gt, final_src_gt, scores, final_cells, final_pairs_gt,
     final_ref_ge, final_src_ge, final_pairs_ge) = match_statistics(reference, source)
    logger.info("stage=ads-rev-match task=%s stage=final cells=%d pairs_gt_0_1=%d pairs_ge_0_1=%d source_hit_gt=%d reference_hit_gt=%d",
                task, final_cells, final_pairs_gt, final_pairs_ge,
                len(final_src_gt), len(final_ref_gt))
    source_budget = sum(len(state["vectors"]) for state in source)
    reference_budget = sum(len(state["vectors"]) for state in reference)
    task_out = output_path("selection", OUTPUT_ROOT / task / "ads-rev")
    task_out.mkdir(parents=True, exist_ok=True)
    descriptor = base.descriptor(source, scores)
    base.write_json(task_out / "probability.json", descriptor)
    base.write_json(task_out / "sampling.json", dict(
        schema=1, mode="probability_descriptor", descriptor_path="probability.json",
        replacement=True, candidate_count=int(legal.sum()), seed=0))
    base.write_json(task_out / "selection.json", dict(
        schema="v1-rage-ads-v2-selection-v1", experiment="v1-rage-ads-v2",
        batch="ads-rev", task=task, arm="ads-rev", source_pool="src-256",
        source_trajectories=len(sources), source_windows=int(legal.sum()),
        reference_trajectories=len(references),
        reference_manifest=str(ref_dir / "manifest.json"), model=identity,
        method=dict(q0=base.Q0, q1=Q_REVERSE, total_rounds=2,
                    threshold_excluded_pairs="cos>=0.1",
                    unmatched_anchor_interval="contiguous runs in q0 coarse grid",
                    density_multiplier=2, neighborhood=base.H, aggregation="max"),
        ads_budget=dict(source=source_budget, reference=reference_budget),
        matched_pairs=dict(coarse_gt_0_1=coarse_pairs_gt, coarse_ge_0_1=coarse_pairs_ge,
                           final_gt_0_1=final_pairs_gt, final_ge_0_1=final_pairs_ge),
        matched_anchors=dict(coarse_excluded_source=len(src_excluded),
                             coarse_excluded_reference=len(ref_excluded),
                             final_source_gt_0_1=len(final_src_gt),
                             final_reference_gt_0_1=len(final_ref_gt)),
        matrix_cells=dict(coarse=coarse_cells, final=final_cells),
        added_gradients=dict(coarse_source=coarse_source_anchors,
                             coarse_reference=coarse_reference_anchors,
                             reverse_source=reverse_src, reverse_reference=reverse_ref),
        elapsed_seconds=time.monotonic() - started))
    (task_out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-complete task=%s arm=ads-rev source_anchors=%d reference_anchors=%d final_pairs_gt_0_1=%d source_hit=%d reference_hit=%d elapsed=%s",
                task, source_budget, reference_budget, final_pairs_gt,
                len(final_src_gt), len(final_ref_gt), format_duration(time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    logger = configure_logger("v1_rage_ads_v2_rev_selection", "v1-rage-ads-v2",
                              "ads-rev-selection", "selection-" + args.task)
    started = time.monotonic()
    logger.info("stage=config tasks=%s arm=ads-rev q0=%d q1=%d reverse_threshold=%.1f workers=1",
                ",".join(tasks), base.Q0, Q_REVERSE, THRESHOLD)
    if any(not (REFERENCE_ROOT / task / ".complete").is_file() for task in tasks):
        raise FileNotFoundError("reference task inputs incomplete")
    sources, legal = base.load_sources()
    sys.path.insert(0, str(ROOT / "scripts/exp/v1-crb-redo"))
    from metrics import build_gradient_probe
    model_engine, probe, identity, model_fields = build_gradient_probe()
    probe.vjp_batch_size = base.VJP_BATCH
    if (identity["model_id"] != "thuml/timer-base-84m"
            or probe.config.projection_dimension != base.PROJECTION_DIM
            or probe.config.projection_seed != base.PROJECTION_SEED
            or model_fields["warmup_steps"] != 1000):
        raise ValueError("warmed TimerXL gradient identity mismatch")
    for task in tasks:
        run_task(task, sources, legal, probe, identity, logger)
    arm_root = output_path("selection", OUTPUT_ROOT / "ads-rev")
    arm_root.mkdir(parents=True, exist_ok=True)
    missing = [task for task in tasks
               if not (OUTPUT_ROOT / task / "ads-rev" / ".complete").is_file()]
    if missing:
        raise RuntimeError("ads-rev selection incomplete for tasks: " + ",".join(missing))
    manifest = dict(experiment="v1-rage-ads-v2", arm="ads-rev", tasks=list(tasks),
                    source_pool="src-256", completed=len(tasks),
                    elapsed_seconds=time.monotonic() - started)
    base.write_json(arm_root / "selection-manifest.json", manifest)
    (arm_root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete arm=ads-rev tasks=%d/%d elapsed=%s output=%s",
                len(tasks), len(TASKS), format_duration(time.monotonic() - started),
                OUTPUT_ROOT / "ads-rev")
    del model_engine


if __name__ == "__main__":
    main()
