"""DEBUG figures (internal): per task, performance-vs-epoch (top) + projected-gradient
L2 dynamics per reference trajectory (bottom). One PNG per model: <model>__main_debug.png"""
import json, glob
from collections import defaultdict
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODELS = {
    "timerxl-84m": (
        "/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-v6-20260921/rounds",
        "/root/autodl-tmp/TS3/cache/grad-pool/thuml-timer-base-84m--70077a71acce1b4c00d98332fcaabc694255d8e5"),
    "chronos-base": (
        "/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918/rounds",
        "/root/autodl-tmp/TS3/cache/grad-pool/amazon-chronos-t5-base--ad294eaacead15db499b740ea4122266dd2a81a2"),
    "timesfm-2.5": (
        "/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918/rounds",
        "/root/autodl-tmp/TS3/cache/grad-pool/google-timesfm-2.5-200m-transformers--5a9806b9b291fad9233b5249d88263f1846304d3"),
}
SEEDS = (0, 1, 2)
N_TRAJ = 6
GROUP = ["rags-block-expSum", "rags-block-linearSum", "rand", "less-main", "tsds", "lp-mirror", "orig"]
PL = plt.get_cmap("plasma")
TC = plt.get_cmap("viridis")


def gen_debug(model, run, pool, out_path):
    ck = defaultdict(lambda: defaultdict(dict))
    orig = {}
    tasks = []
    for s in SEEDS:
        for f in glob.glob(f"{run}/seed{s}/students/{model}/cells/*.json"):
            name = f.split("/")[-1][:-5]
            ds, arm, _ = name.split("__")
            d = json.load(open(f))
            if ds not in tasks:
                tasks.append(ds)
            if arm == "orig":
                orig[ds] = d["mase"]
                continue
            for c in d.get("checkpoints", []):
                ck[(ds, arm)][s][c["epoch"]] = c["metrics"]["mase"]

    arm_mean = defaultdict(list)
    for (ds, arm), sd in ck.items():
        for s, series in sd.items():
            if series:
                arm_mean[arm].append(series[max(series)])
    all_arms = sorted([a for a in arm_mean], key=lambda a: st.mean(arm_mean[a]))
    rank = {a: i for i, a in enumerate(all_arms)}

    def perf_color(arm, present):
        ordered = sorted(present, key=lambda a: rank[a])
        n = max(len(ordered) - 1, 1)
        return PL(ordered.index(arm) / n)

    with np.load(pool + "/window_index.npz", allow_pickle=False) as z:
        datasets = z["datasets"]
        did = z["dataset_ids"]; wids = z["window_ids"]
        sids = z["shard_ids"]; srows = z["shard_rows"]
        shard_names = z["shards"]
    name_of = {i: str(datasets[i]) for i in range(len(datasets))}

    def grad_series(task):
        key = "reference:" + task
        mask = np.array([name_of[int(d)] == key for d in did])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return {}
        uniq = np.unique(wids[idx][:, 0])
        picked = uniq[np.linspace(0, len(uniq) - 1, min(N_TRAJ, len(uniq))).astype(int)]
        want = set(int(p) for p in picked)
        out = defaultdict(lambda: ([], []))
        by_shard = defaultdict(list)
        for r in idx:
            fid = int(wids[r][0])
            if fid in want:
                by_shard[int(sids[r])].append((int(srows[r]), fid, int(wids[r][1])))
        for sid, rows in by_shard.items():
            with np.load(pool + "/" + str(shard_names[sid]), allow_pickle=False) as z:
                g = z["gradients_bf16"][[r[0] for r in rows]].astype(np.uint32) << 16
                g = g.view(np.float32)
                norms = np.sqrt((g * g).sum(axis=1))
            for (row, fid, start), n in zip(rows, norms):
                out[fid][0].append(start); out[fid][1].append(float(n))
        return out

    fig = plt.figure(figsize=(17, 15))
    gs = gridspec.GridSpec(6, 3, height_ratios=[4, 1, 4, 1, 4, 1], hspace=0.55, wspace=0.3)
    for t, ds in enumerate(tasks):
        r = (t // 3) * 2; c = t % 3
        axp = fig.add_subplot(gs[r, c])
        present = [a for a in GROUP if a != "orig"]
        for arm in GROUP:
            if arm == "orig":
                if ds in orig:
                    axp.axhline(orig[ds], color="black", ls="--", lw=1.3, label="orig")
                continue
            eps = sorted({e for s in SEEDS for e in ck.get((ds, arm), {}).get(s, {})})
            if not eps:
                continue
            means = [st.mean([ck[(ds, arm)][s][e] for s in SEEDS if e in ck.get((ds, arm), {}).get(s, {})]) for e in eps]
            axp.plot(eps, means, marker="o", ms=2.5, lw=1.6, color=perf_color(arm, present), label=arm)
        axp.set_title(ds, fontsize=10)
        axp.grid(alpha=0.3); axp.tick_params(labelsize=7)
        axp.set_ylabel("MASE", fontsize=7)
        axg = fig.add_subplot(gs[r + 1, c])
        series = grad_series(ds)
        nf = max(len(series), 1)
        for i, (fid, (pos, nm)) in enumerate(sorted(series.items())):
            order = np.argsort(pos)
            axg.plot(np.asarray(pos)[order], np.asarray(nm)[order], lw=0.9,
                     color=TC(i / max(nf - 1, 1)), label=f"traj{fid}")
        axg.set_xlabel("window position", fontsize=6)
        axg.set_ylabel("||g|| (proj 4096)", fontsize=6)
        axg.tick_params(labelsize=6); axg.grid(alpha=0.25)
        if t == 0:
            axp.legend(fontsize=6, loc="upper right")
            axg.legend(fontsize=5, loc="upper right", ncol=2)
    fig.suptitle(f"DEBUG — performance (top) + projected-gradient L2 dynamics per trajectory (bottom), {model}", fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("saved", out_path)


for model, (run, pool) in MODELS.items():
    gen_debug(model, run, pool, Path(f"/root/TS3/docs/results/figures/{model}__main_debug.png"))
