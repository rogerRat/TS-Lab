"""Measure sparse output-head risk statistics in 64-start blocks."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from configs import logger as logging_config
from configs.logger import configure_logger
from configs.paths import PROJECT_ROOT, output_path

LOGGER = logging.getLogger("v0.gradient_proxy_local_risk_64")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(result) if np.isfinite(result) else None


def block_statistics(gram: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    sub = np.asarray(gram[np.ix_(indices, indices)], dtype=np.float64)
    diagonal = np.diag(sub)
    if not np.isfinite(sub).all() or np.any(diagonal <= 0):
        raise ValueError("nonfinite Gram entries or nonpositive gradient norm")
    variance = float(diagonal.mean() - sub.mean())
    scale = max(1.0, float(diagonal.mean()))
    if variance < -1e-10 * scale:
        raise ValueError("gradient variance is substantially negative")
    variance = max(variance, 0.0)
    norms = np.sqrt(diagonal)
    cosine = np.clip(sub / (norms[:, None] * norms[None, :]), -1.0, 1.0)
    direction_dispersion = float(1.0 - cosine.mean())
    if direction_dispersion < -1e-8 or direction_dispersion > 2.0 + 1e-8:
        raise ValueError("direction dispersion is outside its expected range")
    return variance, min(2.0, max(0.0, direction_dispersion))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-gradient-proxy-local-risk-64.json")
    args = parser.parse_args()
    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="ascii"))
    if cfg.get("substudy") != "Local-Risk 64-point sampling extension":
        raise ValueError("unexpected Local-Risk 64-point protocol")
    block_length = int(cfg["block_length"])
    sample_counts = [int(x) for x in cfg["sample_counts"]]
    repeats = int(cfg["replicates"])
    if block_length != 64 or sample_counts != [64, 48, 32, 16, 8, 4] or repeats != 20:
        raise ValueError("the fixed 64-point sparse-sampling design was changed")
    log_directory = PROJECT_ROOT / "logs/v0-gradient-proxy-test/local-risk-64"
    LOGGER = configure_logger("v0.gradient_proxy_local_risk_64", "v0-gradient-proxy-test",
                              "local-risk-64", "initialization", log_directory=log_directory)
    out = output_path("train", cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".complete").exists():
        raise RuntimeError("completion marker already exists; refusing to overwrite")
    source = PROJECT_ROOT / cfg["source_output"]
    LOGGER.info("[configuration] block_length=%d samples=%s replicates=%d seed=%d output=%s",
                block_length, sample_counts, repeats, cfg["seed"], out)
    started = time.monotonic()
    rng = np.random.default_rng(int(cfg["seed"]))
    sampled_rows: list[dict] = []
    blocks_by_track: dict[int, int] = {}
    baseline_count = 0
    LOGGER.info("[cache] start reading saved full and head Gram matrices")
    for track in range(10):
        track_dir = source / ("trajectory-%02d" % track)
        if not (track_dir / ".complete").exists():
            raise ValueError("incomplete proxy track %d" % track)
        manifest = json.loads((track_dir / "manifest.json").read_text(encoding="ascii"))
        n_windows = int(manifest["windows"])
        with np.load(track_dir / "gram_matrices.npz") as archive:
            full_gram = np.asarray(archive["full"], dtype=np.float64)
            head_gram = np.asarray(archive["head"], dtype=np.float64)
        if full_gram.shape != (n_windows, n_windows) or head_gram.shape != (n_windows, n_windows):
            raise ValueError("unexpected Gram shape for track %d" % track)
        for name, gram in (("full", full_gram), ("head", head_gram)):
            if not np.isfinite(gram).all() or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-7):
                raise ValueError("invalid %s Gram matrix for track %d" % (name, track))
        block_count = n_windows // block_length
        blocks_by_track[track] = block_count
        for block_index in range(block_count):
            block_start = block_index * block_length
            all_indices = np.arange(block_start, block_start + block_length, dtype=np.int64)
            full_variance, full_direction = block_statistics(full_gram, all_indices)
            cached_head_variance, cached_head_direction = block_statistics(head_gram, all_indices)
            for repeat in range(repeats):
                permutation = rng.permutation(block_length)
                for count in sample_counts:
                    if count == block_length:
                        if repeat > 0:
                            continue
                        selected = all_indices
                        replicate = 0
                        head_variance = cached_head_variance
                        head_direction = cached_head_direction
                    else:
                        selected = block_start + np.sort(permutation[:count])
                        replicate = repeat
                        head_variance, head_direction = block_statistics(head_gram, selected)
                    if count == block_length:
                        baseline_count += 1
                    sampled_rows.append({
                        "track": track, "block_start": block_start, "count": count,
                        "replicate": replicate, "full_variance": full_variance,
                        "full_direction_dispersion": full_direction,
                        "sampled_head_variance": head_variance,
                        "sampled_head_direction_dispersion": head_direction,
                        "selected_positions": ",".join(str(int(x - block_start)) for x in selected)
                    })
            if len(sampled_rows) and len(sampled_rows) % 3000 == 0:
                elapsed = time.monotonic() - started
                LOGGER.info("[progress] track=%d block_start=%d records=%d elapsed=%s",
                            track, block_start, len(sampled_rows), logging_config.format_duration(elapsed))
        LOGGER.info("[track] end track=%d windows=%d complete_64_blocks=%d",
                    track, n_windows, block_count)
    total_blocks = sum(blocks_by_track.values())
    if total_blocks < 3 or baseline_count != total_blocks:
        raise RuntimeError("block count or exact 64-point checks are incomplete")
    LOGGER.info("[cache] end complete_64_blocks=%d all_64_point_baselines=%d",
                total_blocks, baseline_count)

    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in sampled_rows:
        groups[(int(row["count"]), int(row["replicate"]), int(row["track"]))].append(row)
    replicate_rows: list[dict] = []
    for (count, replicate, track), rows in sorted(groups.items()):
        for target, proxy in (("full_variance", "sampled_head_variance"),
                              ("full_direction_dispersion", "sampled_head_direction_dispersion")):
            x = np.asarray([r[target] for r in rows], dtype=np.float64)
            y = np.asarray([r[proxy] for r in rows], dtype=np.float64)
            for metric in ("pearson", "spearman"):
                replicate_rows.append({"sample_count": count, "replicate": replicate,
                                       "track": track, "target": target, "proxy_stat": proxy,
                                       "metric": metric, "block_count": len(rows),
                                       "correlation": correlation(x, y, metric)})
    track_values: dict[tuple[int, int, str, str, str], list[float]] = defaultdict(list)
    for row in replicate_rows:
        if row["correlation"] is not None:
            key = (int(row["sample_count"]), int(row["replicate"]), row["target"],
                   row["proxy_stat"], row["metric"])
            track_values[key].append(float(row["correlation"]))
    track_means = {key: float(np.mean(values)) for key, values in track_values.items()}
    summary_rows = []
    for count in sample_counts:
        replicate_ids = [0] if count == block_length else list(range(repeats))
        for target, proxy in (("full_variance", "sampled_head_variance"),
                              ("full_direction_dispersion", "sampled_head_direction_dispersion")):
            for metric in ("pearson", "spearman"):
                values = [track_means[(count, rep, target, proxy, metric)]
                          for rep in replicate_ids if (count, rep, target, proxy, metric) in track_means]
                if not values:
                    raise RuntimeError("no finite correlations for count %d" % count)
                summary_rows.append({
                    "sample_count": count, "target": target, "proxy_stat": proxy, "metric": metric,
                    "tracks": len(blocks_by_track), "total_blocks": total_blocks,
                    "replicates": len(values), "equal_track_mean": float(np.mean(values)),
                    "replicate_median": float(np.median(values)),
                    "replicate_p10": float(np.quantile(values, 0.10)),
                    "replicate_p90": float(np.quantile(values, 0.90)),
                    "replicate_min": float(np.min(values)), "replicate_max": float(np.max(values))
                })
    metadata = {
        "experiment": "v0-gradient-proxy-test", "parent_part": "Local-Risk",
        "substudy": "Local-Risk 64-point sampling extension",
        "block_length": block_length, "sample_counts": sample_counts,
        "replicates_for_counts_below_64": repeats, "count_64_is_exact_full_block": True,
        "seed": int(cfg["seed"]),
        "sampling": "per block and replicate, draw one uniform permutation of 64 positions; counts 4, 8, 16, 32, and 48 use nested prefixes without replacement",
        "targets": ["full vector variance over all 64 full gradients",
                    "full direction dispersion over all 64 full gradients"],
        "proxies": ["output-head variance from sampled head gradients",
                    "output-head direction dispersion from sampled head gradients"],
        "correlation_summary": "per-track Pearson and Spearman across complete blocks; equal-weight mean across tracks; distribution across fixed-seed sampling replicates",
        "complete_blocks_by_track": {str(k): v for k, v in blocks_by_track.items()},
        "total_complete_blocks": total_blocks, "all_64_point_baselines": baseline_count,
        "uses_saved_full_and_head_gram_matrices": True,
        "model_or_gradient_recomputation": False, "elapsed_seconds": time.monotonic() - started
    }
    LOGGER.info("[write] start sampled statistics and correlations")
    write_csv(out / "sampled_block_statistics.csv", sampled_rows)
    write_csv(out / "replicate_correlations.csv", replicate_rows)
    write_csv(out / "summary.csv", summary_rows)
    write_json(out / "input_manifest.json", metadata)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("[complete] sampled_records=%d correlations=%d elapsed=%s output=%s",
                len(sampled_rows), len(replicate_rows),
                logging_config.format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
