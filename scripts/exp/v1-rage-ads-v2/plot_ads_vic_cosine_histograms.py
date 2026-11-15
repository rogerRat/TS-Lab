#!/usr/bin/env python3
"""Recompute and plot q0 and q1 cosine frequency histograms for ADS-VIC."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import sys
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from experiment_config import TASKS
import run_selection as base
import run_ads_rev_selection as matrix

Q1 = 8


def build_probe():
    sys.path.insert(0, str(base.ROOT / "scripts/exp/v1-crb-redo"))
    from metrics import build_gradient_probe
    model_engine, probe, identity, model_fields = build_gradient_probe()
    probe.vjp_batch_size = base.VJP_BATCH
    if (identity["model_id"] != "thuml/timer-base-84m"
            or probe.config.projection_dimension != base.PROJECTION_DIM
            or probe.config.projection_seed != base.PROJECTION_SEED
            or model_fields["warmup_steps"] != 1000):
        raise ValueError("warmed TimerXL gradient identity mismatch")
    return model_engine, probe, identity


def process_task(task, bins, sources, legal, probe, logger):
    logger.info("stage=task-start task=%s", task)
    ref_dir = base.REFERENCE_ROOT / task
    ref_manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="ascii"))
    if (not (ref_dir / ".complete").is_file()
            or ref_manifest.get("answer_values_included") is not False):
        raise ValueError("invalid reference input for task=" + task)
    references = base.load_fragments(ref_dir / "references.npz")
    ref_legal = np.asarray([len(item.valid_starts(base.CONTEXT, base.PREDICTION))
                            for item in references], dtype=np.int64)
    source, src_plans = base.task_state(sources, legal, base.Q0)
    reference, ref_plans = base.task_state(references, ref_legal, base.Q0)

    base.get_new_gradients(source, src_plans, probe, "source", logger, task, 0)
    base.get_new_gradients(reference, ref_plans, probe, "reference", logger, task, 0)
    coarse = matrix.match_statistics(reference, source, histogram_bins=bins)
    ref_hits, src_hits = coarse[0], coarse[1]
    logger.info("stage=histogram task=%s stage=q0-coarse cells=%d pairs_gt_0_1=%d bins=%d",
                task, coarse[3], coarse[4], bins)

    src_refine = base.refine_states(source, src_hits, Q1)
    ref_refine = base.refine_states(reference, ref_hits, Q1)
    added_src = base.get_new_gradients(source, src_refine, probe, "source", logger, task, 1)
    added_ref = base.get_new_gradients(reference, ref_refine, probe, "reference", logger, task, 1)
    if added_src + added_ref:
        final = matrix.match_statistics(reference, source, histogram_bins=bins)
    else:
        final = coarse
    logger.info("stage=histogram task=%s stage=q1-final cells=%d pairs_gt_0_1=%d added_source=%d added_reference=%d bins=%d",
                task, final[3], final[4], added_src, added_ref, bins)
    result = {
        "q0_coarse": {"matrix_cells": coarse[3], "pairs_cos_gt_0_1": coarse[4],
                      "histogram": coarse[8], "edges": coarse[9]},
        "q1_final": {"matrix_cells": final[3], "pairs_cos_gt_0_1": final[4],
                     "histogram": final[8], "edges": final[9],
                     "added_source_anchors": added_src,
                     "added_reference_anchors": added_ref},
    }
    del source, reference, src_plans, ref_plans, src_refine, ref_refine, coarse, final
    return result


def process_group(tasks, bins, worker_id):
    logger = configure_logger("v1_rage_ads_v2_ads_vic_hist_worker_" + str(worker_id),
                              "v1-rage-ads-v2", "ads-vic-hist-worker-" + str(worker_id),
                              "tasks")
    logger.info("stage=worker-start worker=%d tasks=%s", worker_id, ",".join(tasks))
    sources, legal = base.load_sources()
    model_engine, probe, identity = build_probe()
    records = {}
    for task in tasks:
        records[task] = process_task(task, bins, sources, legal, probe, logger)
    logger.info("stage=worker-complete worker=%d tasks=%d", worker_id, len(records))
    del model_engine
    return identity, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")

    logger = configure_logger("v1_rage_ads_v2_ads_vic_histograms", "v1-rage-ads-v2",
                              "ads-vic-cosine-histograms", "q0-q1")
    started = time.monotonic()
    logger.info("stage=config tasks=%s arm=ads-vic q0=%d q1=%d bins=%d workers=%d output=outputs/selection/v1-rage-ads-v2/ads-vic",
                ",".join(TASKS), base.Q0, Q1, args.bins, args.workers)

    groups = (TASKS[:4], TASKS[4:])
    if args.workers == 1:
        groups = (TASKS,)
    logger.info("stage=worker-plan groups=%s", [list(group) for group in groups])
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(process_group, tuple(group), args.bins, index)
                   for index, group in enumerate(groups)]
        worker_results = [future.result() for future in futures]

    identities = [result[0] for result in worker_results]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("worker gradient probe identities differ")
    identity = identities[0]
    records = {}
    for _worker_identity, group_records in worker_results:
        records.update(group_records)
    if set(records) != set(TASKS):
        raise RuntimeError("histogram results do not cover all tasks")

    out_dir = output_path("selection", base.OUTPUT_ROOT / "ads-vic")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v1-rage-ads-v2-ads-vic-cosine-histograms-v1",
        "experiment": "v1-rage-ads-v2",
        "arm": "ads-vic",
        "model": identity,
        "histogram_bins": args.bins,
        "histogram_range": [-1.0, 1.0],
        "stages": {
            "q0_coarse": "All q0=32 anchor pairs before refinement.",
            "q1_final": "All pairs after ADS-VIC refinement at q1=8 on contiguous hit-anchor regions; includes q0 anchors.",
        },
        "tasks": records,
    }
    json_path = out_dir / "cosine-histograms.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=4, cols=2, subplot_titles=[task.upper() for task in TASKS],
                        vertical_spacing=0.07)
    colors = {"q0_coarse": "#2878B5", "q1_final": "#D9534F"}
    labels = {"q0_coarse": "q0=32 coarse", "q1_final": "q1=8 ADS-VIC final"}
    for task_index, task in enumerate(TASKS):
        row, col = divmod(task_index, 2)
        row += 1
        col += 1
        for stage in ("q0_coarse", "q1_final"):
            item = records[task][stage]
            edges = np.asarray(item["edges"], dtype=np.float64)
            counts = np.asarray(item["histogram"], dtype=np.int64)
            centers = (edges[:-1] + edges[1:]) / 2.0
            fig.add_trace(go.Scatter(x=centers, y=counts, mode="lines",
                                     line=dict(color=colors[stage], width=1.5, shape="hv"),
                                     name=labels[stage], legendgroup=stage,
                                     showlegend=(task_index == 0),
                                     hovertemplate="cos in [%{x:.2f}], count=%{y:,}<extra></extra>"),
                          row=row, col=col)
        fig.add_vline(x=0.1, line_dash="dash", line_color="#555555", line_width=1,
                      row=row, col=col)
        fig.update_xaxes(title_text="Cosine similarity", range=[-1, 1], row=row, col=col)
        fig.update_yaxes(title_text="Pair frequency (log)", type="log", row=row, col=col)
    fig.update_layout(title="ADS-VIC cosine frequency: q0 coarse vs q1 final",
                      height=1500, width=1500, template="plotly_white",
                      legend=dict(orientation="h", y=1.035, x=0.0))
    html_path = out_dir / "cosine-histograms.html"
    png_path = out_dir / "cosine-histograms.png"
    fig.write_html(str(html_path), include_plotlyjs=True)
    fig.write_image(str(png_path), width=1500, height=1500, scale=1)
    (out_dir / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete tasks=%d/%d json=%s html=%s png=%s elapsed=%s",
                len(records), len(TASKS), json_path, html_path, png_path,
                format_duration(time.monotonic() - started))
    logger.info("===== Summary ===== duration=%s output=%s",
                format_duration(time.monotonic() - started), out_dir)


if __name__ == "__main__":
    main()
