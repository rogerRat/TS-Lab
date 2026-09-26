"""MASE-only checkpoint and cost curves (D8/D9)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_checkpoint_report(out_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((out_dir / "cells").glob("*.json")):
        cell = json.loads(path.read_text())
        for checkpoint in cell.get("checkpoints", []):
            rows.append({"student": cell.get("student", "chronos-small"),
                         "dataset": cell["dataset"], "arm": cell["arm"], "seed": cell["seed"],
                         "n_windows": cell.get("n_windows"), "epoch": checkpoint["epoch"],
                         "step": checkpoint["step"], "training_s": checkpoint.get("training_elapsed_s", 0),
                         "mase": checkpoint["metrics"]["mase"], "checkpoint": checkpoint["checkpoint"]})
    if not rows:
        return rows
    with (out_dir / "checkpoint_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "checkpoint_curve.json").write_text(json.dumps(rows, indent=2))
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    datasets = sorted({row["dataset"] for row in rows})
    figure = make_subplots(rows=len(datasets), cols=1, subplot_titles=datasets)
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["arm"], row["seed"], row["n_windows"]), []).append(row)
    cost_figure = go.Figure()
    for (dataset, arm, seed, n), values in groups.items():
        values.sort(key=lambda row: row["epoch"])
        name = f"{arm} / seed {seed}" + (f" / N={n}" if n else "")
        figure.add_trace(go.Scatter(x=[r["epoch"] for r in values], y=[r["mase"] for r in values],
            mode="lines+markers", name=name), row=datasets.index(dataset) + 1, col=1)
        cost_figure.add_trace(go.Scatter(x=[r["training_s"] for r in values],
            y=[r["mase"] for r in values], mode="lines+markers", name=f"{dataset} / {name}"))
    figure.update_xaxes(title_text="Training epoch", dtick=1)
    figure.update_yaxes(title_text="MASE")
    figure.update_layout(title="All checkpoints; final epoch is the main result", height=320 * len(datasets))
    figure.write_html(out_dir / "checkpoint_curve.html", include_plotlyjs=True)
    cost_figure.update_layout(xaxis_title="Student training + checkpoint saving time (s)",
                             yaxis_title="MASE", title="Shared selection cost is accounted once in task manifests")
    cost_figure.write_html(out_dir / "cost_curve.html", include_plotlyjs=True)
    data_groups = {}
    for (dataset, arm, seed, n), values in groups.items():
        if n is not None:
            data_groups.setdefault((dataset, arm, seed), []).append(values[-1])
    if data_groups:
        data_figure = go.Figure()
        for (dataset, arm, seed), values in data_groups.items():
            values.sort(key=lambda row: row["n_windows"])
            data_figure.add_trace(go.Scatter(x=[r["n_windows"] for r in values],
                y=[r["mase"] for r in values], mode="lines+markers", name=f"{dataset} / {arm} / {seed}"))
        data_figure.update_layout(xaxis_title="Selected windows N", yaxis_title="Final-epoch MASE")
        data_figure.write_html(out_dir / "data_curve.html", include_plotlyjs=True)
    return rows
