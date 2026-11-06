#!/usr/bin/env python
"""Build docs/TIMERXL.md from results_matrix.json + MAIN/UNI comparison context.
Run on exp: /root/miniconda3/envs/ts3/bin/python build_timerxl_md.py
"""
import json, collections

R="/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920"
M=json.load(open(R+"/results_matrix.json"))
tasks=sorted(M)
ARM_ORDER=["orig","original","rand","random","tsds","less","lp",
           "rage-traj-expSum","rage-traj-linearSum","rage-traj-mean","rage-traj-max",
           "rage-block-expSum","rage-block-linearSum","rage-block-mean","rage-block-max",
           "rage-block-expSum-uni","rage-block-linearSum-uni"]
present={a for t in tasks for a in M[t]}
arms=[a for a in ARM_ORDER if a in present]+sorted(present-set(ARM_ORDER))
assert len(arms)==15, f"expected 15 arms, got {len(arms)}: {arms}"

def mean(t,a): return M[t][a]["mean"]
ORIG="orig" if "orig" in present else "original"

lines=[]
lines.append("# TIMERXL.md — v5-timerxl 正式结果（timerxl-84m, 15 臂 × 9 任务 × 3 seeds）\n")
lines.append("- Run: `/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920`（protocol v5-timerxl，387 unique cells，405 cell 文件含一次中断重跑）")
lines.append("- 模型: thuml/timer-base-84m（84,142,080 参数，bf16，96 点 patch）")
lines.append("- 冻结配方: lr 1e-4, epochs 4, schedule_epochs 6, batch 32, wd 0.01, cosine, betas .9/.999, eps 1e-8, max_grad_norm 1, warmup 0, min_lr_ratio 0（safe-history 两阶段标定 + 预注册扩展确认：ext6 0.91860 / ext8 0.92070 均差于 4ep 0.91699）")
lines.append("- MASE 为 3 seeds 均值（seed0/1/2）；uni = 轨迹内均匀化修订版（与 V5-uni 同 revision rage-intra-trajectory-uniform-v1，但为 TimerXL 独立 run 内同配方对比）\n")

# global table
lines.append("## 全局均值（9 任务平均 MASE）\n")
lines.append("| arm | mean MASE | vs original |")
lines.append("|---|---|---|")
g={a: sum(mean(t,a) for t in tasks)/len(tasks) for a in arms}
orig=g[ORIG]
for a in sorted(arms,key=lambda a:g[a]):
    lines.append(f"| {a} | {g[a]:.4f} | {100*(1-g[a]/orig):+.2f}% |")

# block vs uni (same-run direct comparison)
lines.append("\n## Timer block-vs-uni（同 run 直接对比，逐任务 uni−block，负值=uni更差/MASE更高取正? 见注）\n")
lines.append("注：Δ = mean(uni) − mean(block)，Δ>0 表示 uni 更差（MASE 更高）。\n")
for base,uni in [("rage-block-expSum","rage-block-expSum-uni"),("rage-block-linearSum","rage-block-linearSum-uni")]:
    worse=sum(1 for t in tasks if mean(t,uni)>mean(t,base))
    md=sum(mean(t,uni)-mean(t,base) for t in tasks)/len(tasks)
    lines.append(f"- **{base}**: uni 更差 {worse}/{len(tasks)} 任务，平均 ΔMASE {md:+.4f}")

# RAGE vs baselines per task
lines.append("\n## 逐任务结果\n")
for t in tasks:
    lines.append(f"### {t}\n")
    lines.append("| arm | seed0 | seed1 | seed2 | mean |")
    lines.append("|---|---|---|---|---|")
    for a in sorted(arms,key=lambda a:mean(t,a)):
        s=M[t][a]["seeds"]
        lines.append(f"| {a} | {s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} | {M[t][a]['mean']:.4f} |")
    lines.append("")

open("/root/TS3/docs/TIMERXL.md","w").write("\n".join(lines))
print("wrote /root/TS3/docs/TIMERXL.md", len(lines), "lines")
print(json.dumps(g,indent=1))
