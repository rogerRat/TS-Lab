"""Per-task grouped trajectory plots, fixed global color map by overall arm rank.
Color = rank by mean final-epoch MASE across ALL models/tasks/seeds (lower=better).
No error bars. Output: /root/TS-Lab/docs/results/figures/"""
import json, glob
from collections import defaultdict
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "timerxl-84m": "/root/autodl-tmp/rage/outputs/main-rank-volume-timerxl-v6-20260921/rounds",
    "chronos-base": "/root/autodl-tmp/rage/outputs/main-rank-volume-20260918/rounds",
    "timesfm-2.5": "/root/autodl-tmp/rage/outputs/main-rank-volume-20260918/rounds",
}
SEEDS = (0, 1, 2)
OUT = Path("/root/TS-Lab/docs/results/figures")
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "main": ["rage-block-expSum", "rage-block-linearSum", "rand", "less-main", "tsds", "lp-mirror", "orig"],
    "aggregation": [f"rage-{lvl}-{agg}" for lvl in ("block", "traj") for agg in ("expSum", "linearSum", "maxSum", "zeroClipSum")],
    "uni": ["rage-block-expSum", "rage-block-expSum-win-uni", "rage-block-expSum-block-uni",
            "rage-block-linearSum", "rage-block-linearSum-win-uni", "rage-block-linearSum-block-uni"],
}
TITLES = {"main": "Main: Top-2 RAGE vs baselines",
          "aggregation": "Ablation: block vs traj aggregation",
          "uni": "Ablation: uni weight variants"}

ALL_ARMS = sorted({a for g in GROUPS.values() for a in g if a != "orig"})

def load_cells(model):
    final = defaultdict(list)   # (ds,arm) -> [mase per seed]
    ck = defaultdict(lambda: defaultdict(dict))
    orig = {}
    tasks = []
    for s in SEEDS:
        for f in glob.glob(f"{RUNS[model]}/seed{s}/students/{model}/cells/*.json"):
            name = f.split("/")[-1][:-5]
            ds, arm, _ = name.split("__")
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if ds not in tasks: tasks.append(ds)
            if arm == "orig":
                orig[ds] = d["mase"]
                continue
            final[(ds, arm)].append(d["mase"])
            for c in d.get("checkpoints", []):
                ck[(ds, arm)][s][c["epoch"]] = c["metrics"]["mase"]
    return final, ck, orig, tasks

# global ranking across all models
arm_mean = defaultdict(list)
loaded = {}
for m in RUNS:
    final, ck, orig, tasks = load_cells(m)
    loaded[m] = (ck, orig, tasks)
    for (ds, arm), vals in final.items():
        arm_mean[arm].extend(vals)
ranked = sorted(ALL_ARMS, key=lambda a: st.mean(arm_mean[a]) if arm_mean[a] else 9e9)
RANK = {a: i for i, a in enumerate(ranked)}
CMAP = plt.get_cmap("plasma")
def color_map_for(arms):
    """Per-figure fixed normalization: best-present arm -> darkest (0.0),
    worst-present arm -> brightest (1.0). Ordering follows the global rank,
    which stays visible via legend numbers (#n)."""
    present = [a for a in arms if a in RANK]
    ordered = sorted(present, key=lambda a: RANK[a])
    n = max(len(ordered) - 1, 1)
    return {a: CMAP(i / n) for i, a in enumerate(ordered)}
print("Global arm rank (best->worst):")
for a in ranked:
    print("  #%2d %-34s mean=%.4f" % (RANK[a] + 1, a, st.mean(arm_mean[a]) if arm_mean[a] else float("nan")))

for model in RUNS:
    ck, orig, tasks = loaded[model]
    if not ck:
        continue
    for gname, arms in GROUPS.items():
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
        fig.suptitle(f"{model} — {TITLES[gname]} (per-task, mean over {len(SEEDS)} seeds; plasma: darkest = best in figure ... brightest = worst in figure; legend #n = global rank)", fontsize=12)
        cmap_fig = color_map_for(arms)
        for idx, ds in enumerate(tasks):
            ax = axes[idx // 3][idx % 3]
            for arm in arms:
                if arm == "orig":
                    if ds in orig:
                        ax.axhline(orig[ds], color="black", ls="--", lw=1.4, label="orig")
                    continue
                epoches = sorted({e for s in SEEDS for e in ck.get((ds, arm), {}).get(s, {})})
                if not epoches:
                    continue
                means = []
                for e in epoches:
                    ms = [ck[(ds, arm)][s][e] for s in SEEDS if e in ck.get((ds, arm), {}).get(s, {})]
                    means.append(st.mean(ms) if ms else float("nan"))
                ax.plot(epoches, means, marker="o", ms=3, lw=1.8, color=cmap_fig[arm],
                        label="#%d %s" % (RANK[arm] + 1, arm))
            ax.set_title(ds, fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8)
        for r in axes:
            for a in r:
                leg = a.get_legend()
                if leg: leg.remove()
        fig.text(0.5, 0.02, "epoch (checkpoint every 0.5)", ha="center")
        fig.text(0.01, 0.5, "MASE (lower=better)", va="center", rotation="vertical")
        fig.tight_layout(rect=[0.02, 0.04, 1, 0.97])
        fig.savefig(OUT / f"{model}__{gname}.png", dpi=140)
        plt.close(fig)
        print("saved", f"{model}__{gname}.png")
