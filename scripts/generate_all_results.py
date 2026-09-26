"""Generate docs/ALL-RESULTS.md (V6, multi-checkpoint) from run cell jsons."""
import json, glob, sys, datetime
import statistics as st
from collections import defaultdict

RUNS = {
    "timerxl-84m": "/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-v6-20260921/rounds",
    # v6 重跑（main-rank-volume-20260921）进行中；文档暂用 v5 全矩阵结果复用，完成后切回并替换本节
    "chronos-base": "/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918/rounds",
    "timesfm-2.5": "/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918/rounds",
}
REUSE_NOTE = {
    "chronos-base": "本节为 v5 复用结果（来源 main-rank-volume-20260918，v5 协议 C512→H64，每 1 epoch 存-评估）；v6 重跑进行中，完成后替换本节并保留 v5 作对照。",
    "timesfm-2.5": "本节为 v5 复用结果（来源 main-rank-volume-20260918，v5 协议 C512→H64，每 1 epoch 存-评估）；v6 重跑进行中，完成后替换本节并保留 v5 作对照。",
}
SEEDS = (0, 1, 2)

def load(model):
    """cells[(ds,arm)][seed] = final mase; ckpts[(ds,arm)][seed] = {epoch: mase}"""
    cells = defaultdict(dict)
    ckpts = defaultdict(lambda: defaultdict(dict))
    tasks, arms = [], []
    for s in SEEDS:
        for f in glob.glob(f"{RUNS[model]}/seed{s}/students/{model}/cells/*.json"):
            name = f.split("/")[-1][:-5]
            ds, arm, seedtag = name.split("__")
            sd = int(seedtag.replace("seed", ""))
            try:
                d = json.load(open(f))
                cells[(ds, arm)][sd] = d["mase"]
                for c in d.get("checkpoints", []):
                    ckpts[(ds, arm)][sd][c["epoch"]] = c["metrics"]["mase"]
            except Exception:
                continue
            if ds not in tasks: tasks.append(ds)
            if arm not in arms: arms.append(arm)
    arms = sorted(arms)
    return cells, ckpts, tasks, arms

def mean(v):
    return st.mean(v.values()) if v else None

MODEL_LABEL = {
    "timerxl-84m": "TimerXL-84m (thuml/timer-base-84m, V6 C480→H96)",
    "chronos-base": "Chronos-base (amazon/chronos-t5-base, v5 C512→H64)",
    "timesfm-2.5": "TimesFM-2.5 (google/timesfm-2.5-200m, v5 C512→H64)",
}

out = []
out.append("# ALL-RESULTS（V6 三模型正式结果）\n")
out.append("生成时间：%s。本文档取代 v5 版（已归档 docs/archive/ALL-RESULTS-v5-20260922.md）。" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
out.append("实验矩阵：9 任务 × 15 方法臂 + orig 对照 × 3 seeds；MASE 越低越好，均为 final-epoch 取值（政策 final_epoch_not_best_test，每 0.5 epoch 检查点指标在各 cell json 内）。训练臂每任务 3 seeds（27 样本），orig 为零样本单次评估。\n")

for model in RUNS:
    cells, ckpts, tasks, arms = load(model)
    n_complete = sum(1 for v in cells.values() if v)
    rags = [a for a in arms if a.startswith("rags")]
    total_cells = 9 * len(arms)  # 该矩阵实际臂数（v6=17，v5 复用=13）
    out.append("## 模型 ckpt：%s" % MODEL_LABEL[model])
    if model in REUSE_NOTE:
        out.append("> %s" % REUSE_NOTE[model])
    import os
    for g, gtitle in (("main", "主实验组"), ("aggregation", "聚合消融组"), ("uni", "uni 消融组")):
        fig = f"/root/TS3/docs/results/figures/{model}__{g}.png"
        if os.path.exists(fig):
            out.append("![%s %s](figures/%s__%s.png)" % (model, gtitle, model, g))
    status = "(complete)" if n_complete >= total_cells else "(进行中：已落盘臂 %d/%d，随运行增长，可重跑 scripts/generate_all_results.py 刷新)" % (len(arms), len(arms))
    out.append("进度：%d/%d cells 已落盘%s。\n" % (n_complete, total_cells, status))

    out.append("### 任务 × 臂 平均 MASE（训练臂 = 3 seeds 均值）\n")
    out.append("| 任务 | " + " | ".join(arms) + " |")
    out.append("|" + "---|" * (len(arms) + 1))
    for ds in tasks:
        row = [ds]
        for a in arms:
            m = mean(cells.get((ds, a), {}))
            row.append("%.4f" % m if m is not None else "—")
        out.append("| " + " | ".join(row) + " |")

    out.append("\n### 臂级汇总（vs 基线）\n")
    out.append("| 臂 | 平均 MASE | 相对 rand 平均改善 | 胜 rand 场景/9 | 相对 orig 平均改善 | 胜 orig 场景/9 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for a in arms:
        ms = {ds: mean(cells.get((ds, a), {})) for ds in tasks}
        am = [m for m in ms.values() if m is not None]
        if not am: continue
        vs = lambda b: ([(mean(cells.get((ds, b), {})) - ms[ds]) / mean(cells.get((ds, b), {}))
                         for ds in tasks if ms[ds] is not None and mean(cells.get((ds, b), {}))],)
        vr, = vs("rand"); vo, = vs("orig")
        out.append("| %s | %.4f | %+.2f%% | %d/9 | %+.2f%% | %d/9 |" % (
            a, st.mean(am),
            100 * st.mean(vr) if vr else 0, sum(1 for x in vr if x > 0),
            100 * st.mean(vo) if vo else 0, sum(1 for x in vo if x > 0)))

    if rags and any("rand" in a or True for a in arms):
        out.append("\n### 逐 seed 配对胜率（/27）\n")
        out.append("| RAGS 臂 | vs rand | vs less-main | vs tsds | vs lp-mirror | vs orig |")
        out.append("|---|---|---|---|---|---|")
        for a in rags:
            row = [a]
            for b in ["rand", "less-main", "tsds", "lp-mirror", "orig"]:
                w = 0; n = 0
                for s in SEEDS:
                    for ds in tasks:
                        x, y = cells.get((ds, a), {}).get(s), cells.get((ds, b), {}).get(s)
                        if x is not None and y is not None:
                            n += 1; w += x < y
                row.append("%d/%d" % (w, n) if n else "—")
            out.append("| " + " | ".join(row) + " |")

        out.append("\n### uni 臂 vs 非 uni 臂\n")
        for bse, unis in (("rags-block-expSum", ["rags-block-expSum-win-uni", "rags-block-expSum-block-uni"]),
                          ("rags-block-linearSum", ["rags-block-linearSum-win-uni", "rags-block-linearSum-block-uni"])):
            for u in unis:
                diffs = [(cells[(ds, u)][s] - cells[(ds, bse)][s]) for s in SEEDS for ds in tasks
                         if cells.get((ds, u), {}).get(s) is not None and cells.get((ds, bse), {}).get(s) is not None]
                if diffs:
                    out.append("- `%s` vs `%s`：%d/%d 次更好，平均 ΔMASE = %+.4f" % (u, bse, sum(1 for x in diffs if x < 0), len(diffs), st.mean(diffs)))

        out.append("\n### block vs traj\n")
        for agg in ("expSum", "linearSum", "maxSum", "zeroClipSum"):
            b_, t_ = "rags-block-" + agg, "rags-traj-" + agg
            pairs = [(ds, s) for s in SEEDS for ds in tasks
                     if cells.get((ds, b_), {}).get(s) is not None and cells.get((ds, t_), {}).get(s) is not None]
            if pairs:
                rel = [(cells[(ds, t_)][s] - cells[(ds, b_)][s]) / cells[(ds, t_)][s] for ds, s in pairs]
                w = sum(1 for ds, s in pairs if cells[(ds, b_)][s] < cells[(ds, t_)][s])
                out.append("- %s: block 胜出 %d/%d，平均相对 traj 改善 %+.2f%%" % (agg, w, len(pairs), 100 * st.mean(rel)))

    train_arms = [a for a in arms if a != "orig"]
    if train_arms and any(ckpts.get((ds, a)) for ds in tasks for a in train_arms):
        out.append("### 逐 ckpt 结果（每 0.5 epoch 存-评估点，3 seeds 均值 MASE）\n")
        out.append("取值政策：final_epoch_not_best_test（正式结果为最后一列）；其余各 ckpt 列供训练动态/早停分析。\n")
        for a in train_arms:
            epoches = sorted({e for ds in tasks for s in SEEDS for e in ckpts.get((ds, a), {}).get(s, {})})
            if not epoches: continue
            out.append("**%s**\n" % a)
            out.append("| 任务 | " + " | ".join("ep%.1f" % e for e in epoches) + " | best | best@ep |")
            out.append("|" + "---|" * (len(epoches) + 3))
            for ds in tasks:
                vals = []
                for e in epoches:
                    ms = [ckpts[(ds, a)][s][e] for s in SEEDS if e in ckpts.get((ds, a), {}).get(s, {})]
                    vals.append(st.mean(ms) if ms else None)
                present = [(e, v) for e, v in zip(epoches, vals) if v is not None]
                if not present: continue
                be, bv = min(present, key=lambda x: x[1])
                row = ["%.4f" % v if v is not None else "—" for v in vals]
                out.append("| %s | %s | %.4f | %.1f |" % (ds, " | ".join(row), bv, be))
            out.append("")
    out.append("")

doc = "\n".join(out) + "\n"
open("/root/TS3/docs/results/ALL-RESULTS.md", "w").write(doc)
print("written", len(doc.splitlines()), "lines")
