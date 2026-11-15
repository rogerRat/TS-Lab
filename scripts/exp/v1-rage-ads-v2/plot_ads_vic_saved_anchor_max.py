#!/usr/bin/env python3
"""Plot saved ADS-VIC source-anchor maximum cosine scores by anchor round."""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from experiment_config import TASKS
import run_selection as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()
    selected_tasks = tuple(args.tasks)
    logger = configure_logger("v1_rage_ads_v2_ads_vic_saved_anchor_max",
                              "v1-rage-ads-v2", "ads-vic-saved-anchor-max", "summary")
    started = time.monotonic()
    suffix = "all-tasks" if set(selected_tasks) == set(TASKS) else "current-" + "-".join(selected_tasks)
    out_dir = output_path("selection", base.OUTPUT_ROOT / "ads-vic" / "anchor-max-cosine" / suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for task in selected_tasks:
        selection_dir = base.OUTPUT_ROOT / task / "ads-vic"
        config = json.loads((selection_dir / "selection.json").read_text(encoding="ascii"))
        descriptor = json.loads((selection_dir / "probability.json").read_text(encoding="ascii"))
        q0 = int(config["method"]["q0"])
        if int(config["method"]["q1"]) != 8:
            raise ValueError("unexpected ADS-VIC q1 for task=" + task)
        coarse_scores = []
        added_scores = []
        for trajectory in descriptor["trajectories"]:
            count = int(trajectory["candidate_count"])
            coarse_positions = set(base.uniform_anchor_positions(count, q0))
            for position, score in zip(trajectory["anchor_positions"], trajectory["anchor_scores"]):
                if int(position) in coarse_positions:
                    coarse_scores.append(float(score))
                else:
                    added_scores.append(float(score))
        edges = np.linspace(-1.0, 1.0, 101)
        coarse_hist = np.histogram(coarse_scores, bins=edges)[0]
        added_hist = np.histogram(added_scores, bins=edges)[0]
        records[task] = {
            "statistic": "saved final source-anchor maximum cosine over all final reference anchors",
            "q0_source_anchors": {"count": len(coarse_scores), "histogram": coarse_hist.tolist(),
                                  "edges": edges.tolist()},
            "added_q1_source_anchors": {"count": len(added_scores), "histogram": added_hist.tolist(),
                                        "edges": edges.tolist()},
        }
        logger.info("stage=task-complete task=%s q0_source_anchors=%d added_q1_source_anchors=%d",
                    task, len(coarse_scores), len(added_scores))

    payload = {
        "schema": "v1-rage-ads-v2-ads-vic-saved-anchor-max-v1",
        "experiment": "v1-rage-ads-v2",
        "arm": "ads-vic",
        "statistic": "maximum cosine per saved source anchor, evaluated against the final reference-anchor set",
        "stages": {
            "q0_source_anchors": "Saved final descriptor scores for source positions present on the q0=32 coarse grid.",
            "added_q1_source_anchors": "Saved final descriptor scores for additional source positions absent from q0, selected during q1=8 refinement.",
        },
        "histogram_bins": 100,
        "histogram_range": [-1.0, 1.0],
        "tasks": records,
    }
    json_path = out_dir / "saved-anchor-max-histograms.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    columns = 2 if len(selected_tasks) > 1 else 1
    rows = math.ceil(len(selected_tasks) / columns)
    fig = make_subplots(rows=rows, cols=columns,
                        subplot_titles=[task.upper() for task in selected_tasks],
                        vertical_spacing=0.07)
    labels = {"q0_source_anchors": "q0 source anchors",
              "added_q1_source_anchors": "added q1 source anchors"}
    colors = {"q0_source_anchors": "#2878B5", "added_q1_source_anchors": "#D9534F"}
    for task_index, task in enumerate(selected_tasks):
        row, col = divmod(task_index, columns)
        row += 1
        col += 1
        for stage in labels:
            item = records[task][stage]
            edges = np.asarray(item["edges"], dtype=np.float64)
            counts = np.asarray(item["histogram"], dtype=np.int64)
            centers = (edges[:-1] + edges[1:]) / 2.0
            fig.add_trace(go.Scatter(x=centers, y=counts, mode="lines",
                                     line=dict(color=colors[stage], width=1.5, shape="hv"),
                                     name=labels[stage], legendgroup=stage,
                                     showlegend=(task_index == 0),
                                     hovertemplate="max cos bin=%{x:.2f}, anchors=%{y:,}<extra></extra>"),
                          row=row, col=col)
        fig.add_vline(x=0.1, line_dash="dash", line_color="#555555", line_width=1,
                      row=row, col=col)
        fig.update_xaxes(title_text="Maximum cosine per source anchor", range=[-1, 1], row=row, col=col)
        fig.update_yaxes(title_text="Anchor frequency", type="log", row=row, col=col)
    fig.update_layout(title="Saved ADS-VIC source-anchor maximum cosine scores",
                      height=max(600, 600 * rows), width=900, template="plotly_white",
                      legend=dict(orientation="h", y=1.035, x=0.0))
    html_path = out_dir / "saved-anchor-max-histograms.html"
    png_path = out_dir / "saved-anchor-max-histograms.png"
    fig.write_html(str(html_path), include_plotlyjs=True)
    fig.write_image(str(png_path), width=1500, height=1500, scale=1)
    (out_dir / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete png=%s html=%s json=%s elapsed=%s",
                png_path, html_path, json_path, format_duration(time.monotonic() - started))


if __name__ == "__main__":
    main()
