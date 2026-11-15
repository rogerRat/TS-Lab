from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

ROOT = PROJECT_ROOT
FEATURE_ROOT = ROOT / "outputs/selection/v1-crb-redo/features"
OUT = ROOT / "outputs/selection/v1-crb-redo/sgwdev/source_p99_clustering"
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
METHODS = ("crb", "uniform")
GRANULARITIES = ("traj", "block")
NULL_REPLICATES = 500
SEED = 42


def cluster_stats(scores, owners, starts, counts, granularity, rng):
    n = len(scores)
    top_n = max(1, int(math.ceil(0.01 * n)))
    order = np.lexsort((np.arange(n, dtype=np.int64), -scores))
    top = order[:top_n]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[top] = True
    owner_ids, owner_counts = np.unique(owners[top], return_counts=True)
    owner_pairs = sorted(zip(owner_ids.tolist(), owner_counts.tolist()), key=lambda x: (-x[1], x[0]))
    shares = np.asarray([x[1] / float(top_n) for x in owner_pairs], dtype=np.float64)
    hhi = float(np.sum(shares * shares))
    result = dict(
        top_unit_count=top_n,
        p99_threshold=float(scores[order[top_n - 1]]),
        top_score_max=float(scores[top[0]]),
        top_score_mean=float(np.mean(scores[top])),
        top_unit_window_coverage=float(np.sum(counts[top]) / float(np.sum(counts))),
        trajectories_with_top_units=int(len(owner_ids)),
        top10_trajectory_share=float(np.sum(owner_counts[np.argsort(-owner_counts)[:10]]) / float(top_n)),
        trajectory_hhi=hhi,
        effective_trajectory_count=float(1.0 / hhi) if hhi > 0 else 0.0,
        top_trajectory_counts=owner_pairs[:15],
    )
    if granularity == "traj":
        result.update(adjacent_pairs=0, adjacent_pair_fraction=0.0, max_consecutive_blocks=1,
                      null_adjacent_mean=None, null_adjacent_p95=None, temporal_cluster_p=None)
        return result

    owner_groups = {}
    for owner in np.unique(owners):
        ids = np.flatnonzero(owners == owner)
        ids = ids[np.argsort(starts[ids], kind="stable")]
        owner_groups[int(owner)] = ids
    adjacent = 0
    max_run = 0
    selected_by_owner = {}
    for owner, ids in owner_groups.items():
        mask = top_mask[ids]
        selected_by_owner[owner] = int(mask.sum())
        adjacent += int(np.count_nonzero(mask[:-1] & mask[1:]))
        run = 0
        for active in mask:
            run = run + 1 if active else 0
            max_run = max(max_run, run)

    null_adj = np.zeros(NULL_REPLICATES, dtype=np.int64)
    for rep in range(NULL_REPLICATES):
        value = 0
        for owner, ids in owner_groups.items():
            k = selected_by_owner[owner]
            if k < 2:
                continue
            chosen = np.sort(rng.choice(len(ids), size=k, replace=False))
            value += int(np.count_nonzero(np.diff(chosen) == 1))
        null_adj[rep] = value
    result.update(
        adjacent_pairs=int(adjacent),
        adjacent_pair_fraction=float(adjacent / float(top_n)),
        max_consecutive_blocks=int(max_run),
        null_adjacent_mean=float(np.mean(null_adj)),
        null_adjacent_p95=float(np.quantile(null_adj, 0.95)),
        temporal_cluster_p=float((1 + np.count_nonzero(null_adj >= adjacent)) / (NULL_REPLICATES + 1.0)),
    )
    return result


def best_reference_scores(source_dirs, ref_dirs, device, logger, task, method, granularity):
    source_gpu = torch.as_tensor(source_dirs, dtype=torch.float32, device=device)
    best = torch.full((source_dirs.shape[0],), -float("inf"), dtype=torch.float32, device=device)
    rows = ref_dirs.shape[0]
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, rows, 16):
            ref = torch.as_tensor(ref_dirs[start:start + 16], dtype=torch.float32, device=device)
            score = torch.matmul(ref, source_gpu.T)
            best = torch.maximum(best, score.max(dim=0).values)
            if start == 0 or start + 16 >= rows or (start + 16) % 512 == 0:
                done = min(start + 16, rows)
                eta = (time.monotonic() - started) * (rows - done) / max(done, 1)
                logger.info("stage=score task=%s method=%s granularity=%s refs=%d/%d eta=%s",
                            task, method, granularity, done, rows, format_duration(eta))
    del source_gpu
    return best.cpu().numpy()


def main():
    started = time.monotonic()
    logger = configure_logger("v1_crb_redo_src_p99_cluster", "v1-crb-redo", "sgwdev-source-p99", "analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("stage=start device=%s methods=%s granularities=%s tasks=%d null_replicates=%d seed=%d",
                device, METHODS, GRANULARITIES, len(TASKS), NULL_REPLICATES, SEED)
    out_path = output_path("selection", OUT)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/data/src-96.yaml").read_text())
    traj_names = ["%s/%s/%s" % (r["source"], r["subset"], r["series_id"]) for r in cfg["trajectories"]]
    results = []
    rng = np.random.default_rng(SEED)
    for method in METHODS:
        for granularity in GRANULARITIES:
            source_path = FEATURE_ROOT / method / granularity / "source_units.npz"
            with np.load(source_path, allow_pickle=False) as z:
                source_dirs = z["directions"].copy()
                counts = z["counts"].copy()
                starts = z["starts"].copy()
                owners = z["owners"].copy()
            logger.info("stage=source-features method=%s granularity=%s units=%d", method, granularity, len(source_dirs))
            for task in TASKS:
                ref_path = FEATURE_ROOT / method / granularity / task / "reference_units.npz"
                with np.load(ref_path, allow_pickle=False) as z:
                    ref_dirs = z["directions"].copy()
                scores = best_reference_scores(source_dirs, ref_dirs, device, logger, task, method, granularity)
                row = dict(task=task, method=method, granularity=granularity,
                           score_definition="maximum cosine over task reference units for each source unit",
                           percentile_population="source units within method and task; top 1 percent selected by descending score",
                           source_units=int(len(scores)), reference_units=int(len(ref_dirs)),
                           cosine_pairs=int(len(scores) * len(ref_dirs)),
                           p50_score=float(np.quantile(scores, 0.50)),
                           p90_score=float(np.quantile(scores, 0.90)),
                           p95_score=float(np.quantile(scores, 0.95)),
                           p99_score=float(np.quantile(scores, 0.99)))
                row.update(cluster_stats(scores, owners, starts, counts, granularity, rng))
                row["top_trajectory_names"] = [[traj_names[o], c] for o, c in row["top_trajectory_counts"]]
                results.append(row)
                logger.info("stage=cluster task=%s method=%s granularity=%s top_units=%d trajectories=%d top10_share=%.4f adjacent=%d max_run=%d",
                            task, method, granularity, row["top_unit_count"], row["trajectories_with_top_units"],
                            row["top10_trajectory_share"], row["adjacent_pairs"], row["max_consecutive_blocks"])
            del source_dirs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    jsonl = out_path / "source_p99_clustering.jsonl"
    tmp = jsonl.with_suffix(".partial")
    with tmp.open("w", encoding="ascii") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
    tmp.replace(jsonl)
    (out_path / "manifest.json").write_text(json.dumps(dict(
        experiment="v1-crb-redo", batch="sgwdev", source="src-96 cached source_units and reference_units",
        score_definition="per source unit maximum cosine to this task's reference units",
        p99_definition="top 1 percent source units within each task/method/granularity, descending score",
        null="500 within-trajectory randomizations preserving high-unit counts per trajectory; seed=42",
        output=str(jsonl), elapsed_seconds=time.monotonic() - started), indent=2) + "\n", encoding="ascii")
    (out_path / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete rows=%d elapsed=%s output=%s", len(results), format_duration(time.monotonic()-started), out_path)


if __name__ == "__main__":
    main()
