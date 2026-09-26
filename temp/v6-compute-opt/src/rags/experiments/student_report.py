"""Model-faceted MASE table; dataset-level paired comparisons and shared costs."""
from __future__ import annotations

import json
import numpy as np

from .students import atomic_json


def summarize_students(out_dir, specs, seeds, arm_names, students):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    result = {"metric": "mase", "students": {}, "selection_costs_shared_once": {}}
    figure = make_subplots(rows=1, cols=len(students), subplot_titles=list(students))
    for column, student in enumerate(students, 1):
        cells = [json.loads(path.read_text()) for path in (out_dir / "students" / student / "cells").glob("*.json")]
        table = {}
        for arm in arm_names:
            tasks = {}
            for spec in specs:
                records = [r for r in cells if r["dataset"] == spec.dataset and r["arm"] == arm
                           and r.get("mase") is not None and r["seed"] in ([0] if arm == "original" else seeds)]
                if records:
                    tasks[spec.dataset] = {"mean": float(np.mean([r["mase"] for r in records])),
                                           "std": float(np.std([r["mase"] for r in records])),
                                           "seeds": len(records),
                                           "complete": len(records) == (1 if arm == "original" else len(seeds)),
                                           "training_s": sum(r.get("training_elapsed_s", 0) for r in records),
                                           "evaluation_s": sum(r.get("evaluation_elapsed_s", 0) for r in records)}
            table[arm] = {"tasks": tasks}
            if len(tasks) == len(specs) and all(t["complete"] for t in tasks.values()):
                table[arm]["dataset_equal_mean"] = float(np.mean([t["mean"] for t in tasks.values()]))
        for arm in arm_names:
            for comparator in ("random", "original"):
                reductions = {}
                for dataset, entry in table[arm]["tasks"].items():
                    base = table.get(comparator, {}).get("tasks", {}).get(dataset)
                    if entry["complete"] and base and base["complete"] and base["mean"] > 0:
                        reductions[dataset] = 1 - entry["mean"] / base["mean"]
                table[arm][f"vs_{comparator}"] = reductions
                if reductions:
                    vals = np.asarray(list(reductions.values()))
                    resamples = np.random.default_rng(42).choice(vals, (2000, len(vals)), replace=True).mean(axis=1)
                    table[arm][f"vs_{comparator}_dataset_bootstrap95"] = np.quantile(resamples, [.025, .975]).tolist()
            reductions = table[arm]["vs_random"]
            if reductions:
                figure.add_trace(go.Bar(x=list(reductions), y=[100*v for v in reductions.values()], name=arm,
                                       legendgroup=arm, showlegend=column == 1), row=1, col=column)
            ranks = []
            for dataset, entry in table[arm]["tasks"].items():
                entries = [table[a]["tasks"].get(dataset) for a in arm_names]
                if all(e and e["complete"] for e in entries):
                    values = np.asarray([e["mean"] for e in entries])
                    ranks.append(float(1 + (values < entry["mean"]).sum() + ((values == entry["mean"]).sum()-1)/2))
            table[arm]["average_rank_complete_tasks"] = float(np.mean(ranks)) if ranks else None
        result["students"][student] = table
    for path in (out_dir / "tasks").glob("*.json"):
        task = json.loads(path.read_text())
        result["selection_costs_shared_once"][task["dataset"]] = {
            arm: artifact["cost"] for arm, artifact in task["selections"].items()}
    atomic_json(out_dir / "summary.json", result)
    figure.update_yaxes(title_text="MASE reduction vs Random (%)")
    figure.write_html(out_dir / "student_comparison.html", include_plotlyjs=True)
    return result
