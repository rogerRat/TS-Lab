"""Rebuild main-crb-supp result indexes with MASE as the primary metric."""
from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "docs/exp/main-crb-supp/results.md"
LABELS = ROOT / "docs/exp/main-crb-supp/labels.json"
KEYED: dict[tuple[str, str, int], float | None] = {}


def metric_value(benchmark: str, metric_path: Path) -> float:
    if benchmark == "time":
        with np.load(metric_path, allow_pickle=False) as archive:
            if "MASE" not in archive:
                raise KeyError("TIME metrics file lacks MASE: " + str(metric_path))
            return float(np.nanmean(archive["MASE"]))
    with metric_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("task") != "mean":
        raise ValueError("FEV summary must contain one aggregate row")
    return float(rows[0]["MASE"])


def replace_result_table() -> None:
    text = RESULT.read_text(encoding="utf-8")
    output = []
    for line in text.splitlines():
        if line.startswith("| ") and ("| time-" in line or "| fev-" in line):
            columns = [value.strip() for value in line.strip("|").split("|")]
            if len(columns) == 7 and columns[3].isdigit():
                task, benchmark, arm = columns[:3]
                step = int(columns[3])
                path = Path(columns[6]) if columns[6] != "-" else None
                value = None
                if path is not None and path.is_file() and columns[5] != "-":
                    value = metric_value(benchmark, path)
                KEYED[(task, arm, step)] = value
                columns[5] = "-" if value is None else "%.8g" % value
                output.append("| " + " | ".join(columns) + " |")
                continue
        output.append(line)
    text = "\n".join(output) + "\n"
    text = text.replace("| 状态 | MSE | 原始指标文件 |",
                        "| 状态 | MASE | 原始指标文件 |")
    text = text.replace("MSE 按各自测试集定义", "MASE 按各自测试集定义")
    text = text.replace("主汇总为 MSE", "主汇总为 MASE")
    text = text.replace("不同任务的 MSE", "不同任务的 MASE")
    RESULT.write_text(text, encoding="utf-8")


def update_global(path: Path, begin: str, end: str, mode: str) -> None:
    text = path.read_text(encoding="utf-8")
    first, last = text.index(begin), text.index(end)
    prefix, body, suffix = text[:first], text[first:last], text[last:]
    rows = []
    for line in body.splitlines():
        if line.startswith("| main-crb-supp |"):
            columns = [value.strip() for value in line.strip("|").split("|")]
            code = columns[1]
            if mode == "exp":
                arm = columns[2]
                match = re.search(r"step (\d+)", columns[3])
                if match:
                    value = KEYED.get((code, arm, int(match.group(1))))
                    columns[5] = "-" if value is None else "MASE=%.8g" % value
            else:
                method_step = columns[2]
                match = re.search(r"(.+) step (\d+)", method_step)
                if match:
                    arm, step = match.group(1), int(match.group(2))
                    value = KEYED.get((code, arm, step))
                    columns[4] = "-" if value is None else "%.8g" % value
            line = "| " + " | ".join(columns) + " |"
        rows.append(line)
    body = "\n".join(rows)
    if mode == "global":
        body = body.replace("| 状态 | MSE | 结果文件 |",
                            "| 状态 | MASE | 结果文件 |")
    path.write_text(prefix + body + suffix, encoding="utf-8")


def main() -> None:
    replace_result_table()
    update_global(ROOT / "docs/EXP.md", "<!-- main-crb-supp begin -->",
                  "<!-- main-crb-supp end -->", "exp")
    update_global(ROOT / "docs/RESULTS.md", "<!-- main-crb-supp begin -->",
                  "<!-- main-crb-supp end -->", "global")
    labels = LABELS.read_text(encoding="utf-8").replace(" MSE |", " MASE |")
    LABELS.write_text(labels, encoding="utf-8")
    print("stage=complete indexed=%d output=%s" % (len(KEYED), RESULT))


if __name__ == "__main__":
    main()
