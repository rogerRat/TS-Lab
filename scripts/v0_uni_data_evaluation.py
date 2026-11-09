"""Measure lag differences in full-gradient reference scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, config_file, output_path

DIMENSION = 8_394_496


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty result table")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_refs(lengths, per_track, seed):
    rng = np.random.default_rng(seed)
    refs = []
    for track, count in enumerate(lengths):
        selected = []
        for index in rng.permutation(count):
            if all(abs(int(index) - previous) > 1 for previous in selected):
                selected.append(int(index))
                if len(selected) == per_track:
                    break
        if len(selected) != per_track:
            raise ValueError("cannot draw nonadjacent references")
        refs.extend((track, index) for index in sorted(selected))
    return refs


def lag_rows(scores, valid, max_lag, track):
    rows = []
    n, references, metrics = scores.shape
    for lag in range(1, min(max_lag, n - 1) + 1):
        mask = valid[:-lag] & valid[lag:]
        left = scores[:-lag][mask]
        right = scores[lag:][mask]
        for metric, values_left, values_right in (
            ("cosine", left[:, :, 0], right[:, :, 0]),
            ("dot", left[:, :, 1], right[:, :, 1]),
        ):
            diff = np.abs(values_left - values_right)
            denominator = (np.abs(values_left) + np.abs(values_right)) / 2
            relative = np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator > 0)
            rows.append(dict(track=track, lag=lag, metric=metric,
                             query_pairs=int(mask.sum()), score_pairs=int(diff.size),
                             mean_absolute_difference=float(diff.mean()),
                             mean_relative_difference=float(relative.mean()),
                             median_relative_difference=float(np.median(relative)),
                             p90_relative_difference=float(np.quantile(relative, .9)),
                             zero_denominator_pairs=int(np.sum(denominator == 0)),
                             sign_flip_fraction=float(np.mean(values_left * values_right < 0))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/v0-uni-data-evaluation.json")
    args = parser.parse_args()
    config = json.loads(config_file(args.config).read_text(encoding="ascii"))
    if config["experiment"] != "v0-uni-data-evaluation" or int(config["references_per_track"]) != 10:
        raise ValueError("unexpected experiment configuration")
    out = output_path("train", config["output"])
    if (out / ".complete").exists():
        raise ValueError("completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.uni.data.evaluation", "v0-uni-data-evaluation", "evaluation", "initialization")
    started = time.monotonic()
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device = torch.device("cuda")
    source = PROJECT_ROOT / config["source"]
    gradient_root = PROJECT_ROOT / config["gradient_source"]
    tracks = []
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        n = int(manifest["windows"])
        path = gradient_root / (name + ".bf16")
        if not (directory / ".complete").is_file() or not (gradient_root / (name + ".complete")).is_file():
            raise ValueError("incomplete source: %s" % name)
        if path.stat().st_size != n * DIMENSION * 2:
            raise ValueError("gradient cache size mismatch: %s" % name)
        tracks.append(np.memmap(path, mode="r", dtype=np.uint16, shape=(n, DIMENSION)))
    lengths = [len(item) for item in tracks]
    references = select_refs(lengths, int(config["references_per_track"]), int(config["seed"]))
    if len(references) != 100:
        raise ValueError("reference count mismatch")
    ref_rows = np.stack([np.asarray(tracks[t][i], dtype=np.uint16) for t, i in references])
    ref_gpu = torch.from_numpy(ref_rows).view(torch.bfloat16).to(device=device, dtype=torch.float32)
    ref_norm = torch.linalg.vector_norm(ref_gpu, dim=1)
    del ref_rows
    log.info("Stage start: scores; references=100 tracks=10 total_queries=%d max_lag=%d batch=%d output=%s",
             sum(lengths)-100, int(config["max_lag"]), int(config["query_batch_size"]), out)
    lag_results = []
    ref_set = [set(i for t, i in references if t == track) for track in range(10)]
    for track, mapped in enumerate(tracks):
        scores = np.full((len(mapped), 100, 2), np.nan, dtype=np.float32)
        valid = np.ones(len(mapped), dtype=bool)
        valid[list(ref_set[track])] = False
        indices = np.flatnonzero(valid)
        batch_size = int(config["query_batch_size"])
        for offset in range(0, len(indices), batch_size):
            chosen = indices[offset:offset+batch_size]
            data = np.array(mapped[chosen], dtype=np.uint16, copy=True)
            query = torch.from_numpy(data).view(torch.bfloat16).to(device=device, dtype=torch.float32)
            dot = query @ ref_gpu.T
            norm = torch.linalg.vector_norm(query, dim=1)
            cosine = dot / (norm[:, None] * ref_norm[None, :]).clamp_min(1e-30)
            scores[chosen, :, 0] = cosine.clamp(-1, 1).cpu().numpy()
            scores[chosen, :, 1] = dot.cpu().numpy()
            del data, query, dot, norm, cosine
        if not np.isfinite(scores[valid]).all() or not np.isnan(scores[~valid]).all():
            raise ValueError("score validity mismatch")
        np.save(out / ("trajectory-%02d-scores.npy" % track), scores)
        rows = lag_rows(scores, valid, int(config["max_lag"]), track)
        lag_results.extend(rows)
        elapsed = time.monotonic() - started
        eta = elapsed * (10-track-1)/(track+1)
        log.info("Track complete: %02d queries=%d lag_rows=%d elapsed=%s eta=%s",
                 track, len(indices), len(rows), format_duration(elapsed), format_duration(eta))
    write_csv(out / "trajectory_lag.csv", lag_results)
    summary = []
    for lag in range(1, int(config["max_lag"])+1):
        for metric in ("cosine", "dot"):
            selected = [row for row in lag_results if row["lag"] == lag and row["metric"] == metric]
            summary.append(dict(lag=lag, metric=metric, tracks=len(selected),
                                score_pairs=sum(row["score_pairs"] for row in selected),
                                mean_absolute_difference=float(np.mean([row["mean_absolute_difference"] for row in selected])),
                                mean_relative_difference=float(np.mean([row["mean_relative_difference"] for row in selected])),
                                median_relative_difference_equal_track_mean=float(np.mean([row["median_relative_difference"] for row in selected])),
                                p90_relative_difference_equal_track_mean=float(np.mean([row["p90_relative_difference"] for row in selected])),
                                sign_flip_fraction=float(np.mean([row["sign_flip_fraction"] for row in selected])),
                                zero_denominator_pairs=sum(row["zero_denominator_pairs"] for row in selected)))
    write_csv(out / "lag_summary.csv", summary)
    ref_manifest = [dict(reference_id=j, track=t, start=i) for j, (t, i) in enumerate(references)]
    (out / "references.json").write_text(json.dumps(ref_manifest, indent=2) + "\n", encoding="ascii")
    (out / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="ascii")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, metric in enumerate(("cosine", "dot")):
        selected = [row for row in summary if row["metric"] == metric]
        x = [row["lag"] for row in selected]
        axes[0, col].plot(x, [row["mean_relative_difference"] for row in selected])
        axes[0, col].set_title(metric + " relative score difference")
        axes[0, col].set_ylabel("Mean |a-b| / mean(|a|,|b|)")
        axes[1, col].plot(x, [row["mean_absolute_difference"] for row in selected])
        axes[1, col].set_title(metric + " absolute score difference")
        axes[1, col].set_ylabel("Mean |a-b|")
        axes[1, col].set_xlabel("Lag in legal starts")
    fig.tight_layout()
    fig.savefig(out / "lag_difference.png", dpi=160)
    plt.close(fig)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: evaluation; duration=%s output=%s", format_duration(time.monotonic()-started), out)


if __name__ == "__main__":
    main()
