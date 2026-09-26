"""Aggregate one orchestrator run directory into a markdown report.

Usage:
    python -m rags.experiments.report --run-dir outputs/<run_id>

Reads the per-step outputs (summary.json / mechanism jsons / curve cells)
and writes ``REPORT.md`` next to the run's RUN.md, printing the same to
stdout.  Pure CPU, no model involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main_table_section(run_dir: Path) -> str | None:
    summary_path = run_dir / "main" / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    rows = []
    order = ["original", "target_only", "random", "rags",
             "less", "tsds", "lp_mirror", "ltsv"]
    for arm in [a for a in order if a in summary] + [
            a for a in summary if a not in order and a != "rags_vs_random"]:
        entry = summary[arm]
        datasets = entry.get("per_dataset", {})
        if not datasets:
            continue
        rows.append([arm,
                     "%.4f" % entry.get("mase_avg", float("nan")),
                     "%.4f" % entry.get("sql_avg", float("nan")),
                     len(datasets)])
    text = ["## 主表 (main)", _table(rows, ["arm", "MASE avg", "SQL avg", "datasets"])]
    if "rags_vs_random" in summary:
        reductions = summary["rags_vs_random"]
        if reductions:
            mean = float(np.mean([v["mase_reduction"] for v in reductions.values()]))
            text.append("RAGS vs Random MASE reduction (mean over datasets): %.2f%%"
                        % (100 * mean))
    return "\n".join(text)


def mechanism1_section(run_dir: Path) -> str | None:
    path = run_dir / "mechanism1" / "mechanism1.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rows = []
    for block in data["densities"]:
        for r in block["reconstruction"]:
            if r["side"] == "reference":
                rows.append(["%.2f" % block["density"], r["method"], r["k"],
                             "%.3f" % r["relative_error"],
                             "%.3f" % r["direction_error"]])
    recall_rows = []
    for block in data["densities"]:
        for r in block["recall"]:
            recall_rows.append([
                "%.2f" % block["density"], r["variant"], r["B"], r["blocks"],
                "%.2f" % r["union_coverage"]["@100"],
                "%.2f" % r["fixed_n_coverage"]["@100"]])
    parts = ["## 机制实验一 (mechanism1)",
             "### 梯度重建（参考侧）",
             _table(rows, ["density", "method", "K", "rel err", "dir err"]),
             "### 块召回 @100",
             _table(recall_rows, ["density", "variant", "B", "blocks",
                                  "union@100", "fixedN@100"])]
    if data.get("replacement"):
        repl_rows = [[r["case"], r["density"],
                      "%.3f" % r["overlap@50"], "%.3f" % r["overlap@100"],
                      "%.3f" % r["overlap@500"]] for r in data["replacement"]]
        parts += ["### 单侧替换（排名前 M 重合率）",
                  _table(repl_rows, ["case", "density", "@50", "@100", "@500"])]
    return "\n".join(parts)


def mechanism2_section(run_dir: Path) -> str | None:
    path = run_dir / "mechanism2" / "mechanism2.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rows = []
    for side in ("reference", "source"):
        summary = data.get("summary", {}).get(side, {})
        rows.append([side, summary.get("intervals", 0),
                     json.dumps(summary.get("class_counts", {})),
                     summary.get("flagged", 0),
                     "%.3f" % summary.get("mean_endpoint_deviation", float("nan"))])
    return "\n".join([
        "## 机制实验二 (mechanism2)",
        _table(rows, ["side", "intervals", "classes", "flagged", "endpoint dev"]),
    ])


def ablation_section(run_dir: Path) -> str | None:
    path = run_dir / "ablation" / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    rows = []
    for arm, entry in summary.items():
        if not isinstance(entry, dict) or "per_dataset" not in entry:
            continue
        dataset = next(iter(entry["per_dataset"].values()))
        rows.append([arm, "%.4f" % dataset["mase_mean"],
                     "%.4f" % dataset["mase_std"], dataset.get("seeds", 0)])
    return "\n".join([
        "## 消融 (ablation)",
        _table(rows, ["variant", "MASE mean", "std", "seeds"]),
    ])


def curve_section(run_dir: Path, mode: str) -> str | None:
    cells_dir = run_dir / mode / "cells"
    if not cells_dir.exists():
        return None
    records = [json.loads(p.read_text())
               for p in sorted(cells_dir.glob("*.json"))]
    if not records:
        return None
    rows = []
    by_point: dict[tuple, list[float]] = {}
    for record in records:
        if mode == "data-curve":
            key = (record["arm"], record.get("n_fraction"))
        else:
            key = (record["arm"], "time")
        by_point.setdefault(key, []).append(record["mase"])
    for (arm, point), values in sorted(by_point.items()):
        rows.append([arm, point, len(values),
                     "%.4f" % float(np.mean(values)),
                     "%.4f" % float(np.std(values))])
    return "\n".join([
        "## 曲线 %s" % mode,
        _table(rows, ["arm", "point", "seeds", "MASE mean", "std"]),
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit("not a run directory: %s" % run_dir)
    sections = [
        main_table_section(run_dir),
        mechanism1_section(run_dir),
        mechanism2_section(run_dir),
        ablation_section(run_dir),
        curve_section(run_dir, "data-curve"),
        curve_section(run_dir, "time-curve"),
    ]
    body = "\n\n".join(s for s in sections if s)
    header = "# Report %s\n" % run_dir.name
    report = header + (body or "(no completed step outputs found)")
    out_path = run_dir / "REPORT.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print("\nwritten -> %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
