"""Report the frozen ten-task V7 matrix without reading gradient payloads.

Native results and the separate shape sidecars are paired by model, task and
seed.  A partial report is explicitly labelled; the final report fails closed
until all four shape arms have every expected cell.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo


MODELS = ("chronos-base", "timerxl-84m", "timesfm-2.5")
EXTENSIONS = (
    ("time_ne_china_wind_h", "ne-china-wind-20260925"),
    ("time_sg_pm25_h", "time_sg_pm25_h-20260925"),
    ("time_coastal_ts_h", "time_coastal_ts_h-20260925"),
)
SHAPE_ARMS = (
    "rags-shape-expSum", "rags-shape-linearSum",
    "rags-shape-zeroClipSum", "rags-shape-maxSum",
)
SEEDS = (0, 1)


def reported_mase(model: str, row: dict) -> float:
    if row.get("arm") == "orig":
        if row.get("trained") is not False:
            raise ValueError("Orig must be the untouched pretrained evaluation")
        return float(row["mase"])
    if model == "chronos-base":
        values = [p["metrics"]["mase"] for p in row["test_curve"]
                  if p.get("epoch") == 3.0]
        if len(values) != 1:
            raise ValueError("Chronos cell lacks exactly one 3.0-epoch test point")
        return float(values[0])
    if row.get("summary_epoch") != 4.0:
        raise ValueError("TimerXL/TimesFM cell must report the 4.0-epoch endpoint")
    return float(row["mase"])


def source_runs(parent: Path, main_tasks: tuple[str, ...]):
    yield from ((task, parent) for task in main_tasks)
    yield from ((task, parent / "extensions" / directory)
                for task, directory in EXTENSIONS)


def collect(parent: Path, main_tasks: tuple[str, ...], arms: tuple[str, ...]):
    results, files = {}, {}
    for task, run in source_runs(parent, main_tasks):
        for model in MODELS:
            for seed in SEEDS:
                base = run / "seeds" / f"seed{seed}" / model
                shape = run / "shape_backfill_20260925" / "seeds" / f"seed{seed}" / model
                for arm in arms:
                    folder = shape if arm in SHAPE_ARMS else base
                    path = folder / "cells" / f"{task}__{arm}.json"
                    if not path.is_file():
                        continue
                    row = json.loads(path.read_text())
                    if (row.get("arm"), row.get("task"), row.get("seed"), row.get("student")) != (arm, task, seed, model):
                        raise ValueError(f"Cell identity mismatch: {path}")
                    key = (task, model, seed, arm)
                    if key in results:
                        raise ValueError(f"Duplicate cell: {key}")
                    results[key] = reported_mase(model, row)
                    files[key] = folder / "selections" / f"{task}__{arm}.npz"
    return results, files


def effects(results, arms, tasks):
    paired = defaultdict(list)
    for task in tasks:
        for model in MODELS:
            for seed in SEEDS:
                baseline = results.get((task, model, seed, "rand"))
                if baseline is None or baseline <= 0:
                    raise ValueError(f"Missing/nonpositive paired rand: {task}/{model}/{seed}")
                for arm in arms:
                    value = results.get((task, model, seed, arm))
                    if value is not None:
                        paired[arm].append((task, model, seed, 100 * (1 - value / baseline)))
    return paired


def rank_shape(paired, target=60):
    if any(len(paired[arm]) != target for arm in SHAPE_ARMS):
        raise ValueError("All four Shape arms must have complete paired cells before top-two freeze")
    return tuple(sorted(SHAPE_ARMS, key=lambda arm: (-mean(x[3] for x in paired[arm]), arm))[:2])


def render(parent: Path, main_tasks: tuple[str, ...], arms: tuple[str, ...], *, allow_partial=False):
    tasks = (*main_tasks, *(task for task, _ in EXTENSIONS))
    if len(set(tasks)) != 10 or "grad-cos" in arms or len(arms) != 23:
        raise ValueError("Final V7 must be ten distinct tasks and 23 arms, excluding grad-cos")
    results, files = collect(parent, main_tasks, arms)
    paired = effects(results, arms, tasks)
    expected = len(tasks) * len(MODELS) * len(SEEDS)
    incomplete = {arm: expected - len(paired[arm]) for arm in arms if len(paired[arm]) != expected}
    if incomplete and not allow_partial:
        raise ValueError("V7 final matrix incomplete: " + str(incomplete))
    top = rank_shape(paired, expected) if not incomplete else None
    snapshot = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    lines = ["# V7 十任务最终矩阵" if not incomplete else "# V7 十任务矩阵（进行中，非最终）", "",
        f"数据快照：{snapshot}；生成入口 `python -m scripts.report_v7_final --parent {parent} --allow-partial`。", "",
        "口径：七项原主任务 + Wind、SG_PM25、Coastal 三项探索扩展；三模型、seed0/1；",
        "原 20 臂去除 `grad-cos`，加入四个 Shape 臂，共 23 臂。`grad-cos` 原始产物保留，",
        "但不是本表方法。Chronos 固定取 3.0 epoch，TimerXL/TimesFM 固定取 4.0 epoch；",
        "不按 test 选最佳检查点。以下均为同一模型、任务、seed 对 rand 的配对改善率",
        "`100×(1−MASE_method/MASE_rand)`，再对单元等权平均；越大越好。", "",
        f"已完成 {len(results)}/{expected * len(arms)} 个预期 cell。" +
        ("Shape 未齐，所有 Shape 排名暂定。" if incomplete else "四个 Shape 臂均已完整验收。"), "",
        "原七任务是主矩阵，新增三任务是依据 REF 几何和领域先验选出的探索任务；",
        "二者合并描述时不得称为预先固定的十任务验证集。仅两个 seed，以下为描述性结果，不附显著性结论。", "",
        "## 全臂汇总（配对改善%，按总均值降序）", "",
        "| 方法 | Chronos | TimerXL | TimesFM | 总均值 | 中位数 | 胜 rand | n/60 | 选集 MiB¹ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(arms, key=lambda arm: (
        -mean(x[3] for x in paired[arm]) if paired[arm] else float("inf"), arm))
    for arm in ordered:
        entries = paired[arm]
        values = [x[3] for x in entries]
        by_model = [mean(x[3] for x in entries if x[1] == model) if any(x[1] == model for x in entries) else None
                    for model in MODELS]
        storage = sum(files[(t, m, s, arm)].stat().st_size for t, m, s, _ in entries
                      if files[(t, m, s, arm)].is_file()) / 2**20
        fmt = lambda value: "—" if value is None else f"{value:+.3f}"
        total = f"{mean(values):+.3f}" if values else "—"
        middle = f"{median(values):+.3f}" if values else "—"
        lines.append(f"| {arm} | {fmt(by_model[0])} | {fmt(by_model[1])} | {fmt(by_model[2])} | "
                     f"{total} | {middle} | {sum(v > 0 for v in values)}/{len(values)} | "
                     f"{len(values)}/{expected} | {storage:.1f} |")
    lines += ["", "¹ 选集 NPZ 实际文件长度求和；不含共享梯度、Shape 特征/分数缓存，",
              "也不是算法存储下界。`orig` 不需要选集。共享资源必须单独核算。", "",
              "## 原主任务与探索扩展分开看（配对改善%）", "",
              "| 方法 | 原七任务（n/42） | 探索三任务（n/18） |",
              "|---|---:|---:|"]
    main_set = set(main_tasks)
    for arm in ordered:
        group = []
        for select, total in ((lambda task: task in main_set, 42),
                              (lambda task: task not in main_set, 18)):
            values = [value for task, _, _, value in paired[arm] if select(task)]
            group.append(f"{mean(values):+.3f} ({len(values)}/{total})" if values else f"— (0/{total})")
        lines.append(f"| {arm} | {group[0]} | {group[1]} |")
    lines += ["",
              "## Shape 与对应 Block / Traj 的严格配对", "",
              "| 汇总规则 | Shape vs Block（百分点；胜/配对） | Shape vs Traj（百分点；胜/配对） |",
              "|---|---:|---:|",
    ]
    for shape in SHAPE_ARMS:
        suffix = shape.removeprefix("rags-shape-")
        anchors = {(t, m, s): v for t, m, s, v in paired[shape]}
        cells = []
        for family in ("block", "traj"):
            baseline = {(t, m, s): v for t, m, s, v in paired[f"rags-{family}-{suffix}"]}
            delta = [v - baseline[key] for key, v in anchors.items() if key in baseline]
            cells.append(f"{mean(delta):+.3f}; {sum(d > 0 for d in delta)}/{len(delta)}" if delta else "—")
        lines.append(f"| {shape} | {cells[0]} | {cells[1]} |")
    lines += ["", "## 逐任务结果（每项三模型×两 seed 等权）", "",
              "| 任务 | Shape-Exp | Shape-Linear | Shape-ZeroClip | Shape-Max | Block-Exp | Block-Linear | Rand | LESS | TSDS | LP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    featured = (*SHAPE_ARMS, "rags-block-expSum", "rags-block-linearSum", "rand", "less-main", "tsds", "lp-mirror")
    for task in tasks:
        cells = []
        for arm in featured:
            values = [v for t, _, _, v in paired[arm] if t == task]
            cells.append(f"{mean(values):+.2f} ({len(values)}/6)" if values else "—")
        lines.append("| " + task + " | " + " | ".join(cells) + " |")
    lines += ["", "## 论文使用边界", "",
              "- Shape 是中点与斜率的块表征，仍按块打分；它不是逐窗口精确梯度。",
              "- 所有梯度臂共享源/参考稀疏梯度资产。选集存储较小不代表梯度资产免费；",
              "  选择时间、共享资产准备 BP/墙钟与物理存储应分栏引用成本报告。",
              "- Grad-dot 使用参考梯度均值；线性点积下等价于逐参考梯度点积后平均。",
              "  `grad-cos` 非双侧逐窗余弦，故依用户决定从最终比较中排除。",
              "- V7-e 的两臂从本十任务结果选出；后续新任务是探索性扩展，",
              "  与用于选方法的十任务必须分开报告，不能合并宣称独立验证。", ""]
    if top:
        lines += ["## V7-e 冻结的 Shape 两臂", "",
                  f"按完整十任务、60 配对单元的平均相对 rand 改善率选取：`{top[0]}`、`{top[1]}`。",
                  "对应 Block/Traj、LP/TSDS/LESS、rand/orig 一同进入 V7-e。", ""]
    runs = dict(source_runs(parent, main_tasks))
    rows = []
    for (task, model, seed, arm), mase in sorted(results.items()):
        baseline = results[(task, model, seed, "rand")]
        selection = files[(task, model, seed, arm)]
        rows.append(dict(task=task, scope="original_seven" if task in main_set else "exploratory_three",
                         model=model, seed=seed, arm=arm, mase=mase, rand_mase=baseline,
                         improvement_vs_rand_pct=100 * (1 - mase / baseline),
                         source_run=str(runs[task]),
                         cell_file=str(selection.parent.parent / "cells" / f"{task}__{arm}.json"),
                         selection_file_bytes=selection.stat().st_size if selection.is_file() else 0))
    return "\n".join(lines), dict(snapshot_time=snapshot, status="complete" if not incomplete else "partial",
        effect_definition="100*(1-method_mase/paired_rand_mase), equal cell weights",
        endpoint_epochs=dict(zip(MODELS, (3.0, 4.0, 4.0))), rows=rows,
        complete=not incomplete, missing=incomplete,
        expected_cells=expected * len(arms), observed_cells=len(results), tasks=list(tasks),
        arms=list(arms), top_two_shape=list(top) if top else None,
        means={arm: mean(x[3] for x in paired[arm]) if paired[arm] else None for arm in arms})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/rags/main_v7_curve4.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/v7_final_10task.md"))
    parser.add_argument("--json-out", type=Path, default=Path("docs/results/v7_final_10task.json"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    arms = tuple(arm for arm in config["arms"] if arm != "grad-cos") + SHAPE_ARMS
    markdown, summary = render(args.parent, tuple(config["datasets"]), arms,
                               allow_partial=args.allow_partial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
