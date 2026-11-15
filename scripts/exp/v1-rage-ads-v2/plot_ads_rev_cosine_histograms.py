#!/usr/bin/env python3
"""Recompute and plot q0 and q1 cosine frequency histograms for ads-rev."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from experiment_config import TASKS
import run_selection as base
import run_ads_rev_selection as reverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.bins <= 0:
        raise ValueError("bins must be positive")

    logger = configure_logger("v1_rage_ads_v2_cosine_histograms", "v1-rage-ads-v2",
                              "ads-rev-cosine-histograms", "q0-q1")
    started = time.monotonic()
    logger.info("stage=config tasks=%s q0=%d q1=%d bins=%d output=outputs/selection/v1-rage-ads-v2/ads-rev",
                ",".join(TASKS), base.Q0, reverse.Q_REVERSE, args.bins)

    sources, legal = base.load_sources()
    import sys
    sys.path.insert(0, str(base.ROOT / "scripts/exp/v1-crb-redo"))
    from metrics import build_gradient_probe
    model_engine, probe, identity, _model_fields = build_gradient_probe()
    probe.vjp_batch_size = base.VJP_BATCH
    records = {}

    for task_index, task in enumerate(TASKS, start=1):
        logger.info("stage=task-start task=%s index=%d/%d", task, task_index, len(TASKS))
        ref_dir = reverse.REFERENCE_ROOT / task
        references = base.load_fragments(ref_dir / "references.npz")
        ref_legal = np.asarray([len(item.valid_starts(base.CONTEXT, base.PREDICTION))
                                for item in references], dtype=np.int64)
        source, src_plans = base.task_state(sources, legal, base.Q0)
        reference, ref_plans = base.task_state(references, ref_legal, base.Q0)

        base.get_new_gradients(source, src_plans, probe, "source", logger, task, "coarse")
        base.get_new_gradients(reference, ref_plans, probe, "reference", logger, task, "coarse")
        coarse = reverse.match_statistics(reference, source, histogram_bins=args.bins)
        coarse_hits_ref, coarse_hits_src = coarse[5], coarse[6]
        logger.info("stage=histogram task=%s stage=q0-coarse cells=%d pairs_gt_0_1=%d bins=%d",
                    task, coarse[3], coarse[4], args.bins)

        src_refine = reverse.reverse_refine(source, coarse_hits_src, reverse.Q_REVERSE)
        ref_refine = reverse.reverse_refine(reference, coarse_hits_ref, reverse.Q_REVERSE)
        added_src = base.get_new_gradients(source, src_refine, probe, "source", logger, task, "reverse-q16")
        added_ref = base.get_new_gradients(reference, ref_refine, probe, "reference", logger, task, "reverse-q16")
        final = reverse.match_statistics(reference, source, histogram_bins=args.bins)
        logger.info("stage=histogram task=%s stage=q1-final cells=%d pairs_gt_0_1=%d added_source=%d added_reference=%d bins=%d",
                    task, final[3], final[4], added_src, added_ref, args.bins)

        records[task] = {
            "q0_coarse": {"matrix_cells": coarse[3], "pairs_cos_gt_0_1": coarse[4],
                          "histogram": coarse[8], "edges": coarse[9]},
            "q1_final": {"matrix_cells": final[3], "pairs_cos_gt_0_1": final[4],
                         "histogram": final[8], "edges": final[9],
                         "added_source_anchors": added_src,
                         "added_reference_anchors": added_ref},
        }
        del source, reference, src_plans, ref_plans, src_refine, ref_refine
        del coarse, final

    out_dir = output_path("selection", reverse.OUTPUT_ROOT / "ads-rev")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v1-rage-ads-v2-cosine-histograms-v1",
        "experiment": "v1-rage-ads-v2",
        "arm": "ads-rev",
        "model": identity,
        "histogram_bins": args.bins,
        "histogram_range": [-1.0, 1.0],
        "stages": {
            "q0_coarse": "All q0=32 anchor pairs before reverse refinement.",
            "q1_final": "All pairs after adding q1=16 anchors in coarse unmatched regions; includes q0 anchors.",
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
    labels = {"q0_coarse": "q0=32 coarse", "q1_final": "q1=16 final"}
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
    fig.update_layout(title="ads-rev cosine frequency: q0 coarse vs q1 final",
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
    del model_engine


if __name__ == "__main__":
    main()
