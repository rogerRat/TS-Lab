"""Measure local-risk proxy correlations when only k of 32 points are observed."""
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

LOGGER = logging.getLogger("v0.gradient_proxy_local_risk_sparse")


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


def sample_statistics(gram: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    sub = np.asarray(gram[np.ix_(indices, indices)], dtype=np.float64)
    diagonal = np.diag(sub)
    if not np.isfinite(sub).all() or np.any(diagonal <= 0):
        raise ValueError("nonfinite Gram entries or nonpositive head-gradient norm")
    variance = float(diagonal.mean() - sub.mean())
    scale = max(1.0, float(diagonal.mean()))
    if variance < -1e-10 * scale:
        raise ValueError("sampled head variance is substantially negative")
    variance = max(variance, 0.0)
    norms = np.sqrt(diagonal)
    cosine = sub / (norms[:, None] * norms[None, :])
    direction_dispersion = float(1.0 - cosine.mean())
    if direction_dispersion < -1e-10 or direction_dispersion > 2.0 + 1e-10:
        raise ValueError("direction dispersion is outside its expected range")
    return variance, min(2.0, max(0.0, direction_dispersion))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-gradient-proxy-local-risk-sparse.json")
    args = parser.parse_args()
    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="ascii"))
    if cfg.get("substudy") != "Local-Risk sampling extension":
        raise ValueError("unexpected Local-Risk sampling protocol")
    block_length = int(cfg["block_length"])
    sample_counts = [int(x) for x in cfg["sample_counts"]]
    repeats = int(cfg["replicates"])
    if block_length != 32 or sample_counts != [32, 24, 16, 8, 4] or repeats != 20:
        raise ValueError("the fixed sparse-sampling design was changed")
    log_directory = PROJECT_ROOT / "logs/v0-gradient-proxy-test/local-risk-sparse"
    LOGGER = configure_logger("v0.gradient_proxy_local_risk_sparse", "v0-gradient-proxy-test",
                              "local-risk-sparse", "initialization", log_directory=log_directory)
    out = output_path("train", cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".complete").exists():
        raise RuntimeError("completion marker already exists; refusing to overwrite")
    source = PROJECT_ROOT / cfg["source_output"]
    LOGGER.info("[configuration] samples=%s repeats=%d block_length=%d seed=%d output=%s",
                sample_counts, repeats, block_length, cfg["seed"], out)
    start_time = time.monotonic()
    np.random.seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    sampled_rows: list[dict] = []
    blocks_by_track: dict[int, int] = defaultdict(int)
    exact_checks = 0
    LOGGER.info("[cache] start loading saved block targets and head Gram matrices")
    for track in range(10):
        track_dir = source / ("trajectory-%02d" % track)
        block_path = track_dir / "blocks.csv"
        gram_path = track_dir / "gram_matrices.npz"
        if not (track_dir / ".complete").exists() or not gram_path.exists():
            raise ValueError("incomplete proxy track %d" % track)
        with block_path.open(encoding="ascii", newline="") as handle:
            block_rows = list(csv.DictReader(handle))
        with np.load(gram_path) as archive:
            gram = np.asarray(archive["head"], dtype=np.float64)
        if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all():
            raise ValueError("invalid head Gram matrix for track %d" % track)
        if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-8):
            raise ValueError("asymmetric head Gram matrix for track %d" % track)
        n_windows = gram.shape[0]
        full_blocks = [row for row in block_rows if int(row["count"]) == block_length]
        blocks_by_track[track] = len(full_blocks)
        for block in full_blocks:
            block_start = int(block["start"])
            if block_start < 0 or block_start + block_length > n_windows:
                raise ValueError("block exceeds track bounds")
            full_variance = float(block["full_variance"])
            full_direction = float(block["full_direction_dispersion"])
            if not np.isfinite([full_variance, full_direction]).all():
                raise ValueError("nonfinite full-gradient block target")
            local_rows = []
            for repeat in range(repeats):
                permutation = rng.permutation(block_length)
                for count in sample_counts:
                    if count == block_length:
                        if repeat > 0:
                            continue
                        selected = np.arange(block_length, dtype=np.int64)
                        replicate = 0
                    else:
                        selected = np.sort(permutation[:count])
                        replicate = repeat
                    absolute = block_start + selected
                    variance, direction = sample_statistics(gram, absolute)
                    if count == block_length:
                        cached_variance = float(block["head_variance"])
                        cached_direction = float(block["head_direction_dispersion"])
                        if not np.isclose(variance, cached_variance, rtol=1e-7, atol=1e-8):
                            raise ValueError("full-sample head variance does not match cached block statistic")
                        if not np.isclose(direction, cached_direction, rtol=1e-7, atol=1e-8):
                            raise ValueError("full-sample direction dispersion does not match cached statistic")
                        exact_checks += 1
                    row = {"track": track, "block_start": block_start, "count": count,
                           "replicate": replicate, "full_variance": full_variance,
                           "full_direction_dispersion": full_direction,
                           "sampled_head_variance": variance,
                           "sampled_head_direction_dispersion": direction,
                           "selected_positions": ",".join(str(int(x)) for x in selected)}
                    sampled_rows.append(row)
                    local_rows.append(row)
            if len(sampled_rows) % 2000 < repeats:
                elapsed = time.monotonic() - start_time
                total_blocks = sum(blocks_by_track.values()) + sum(
                    1 for r in block_rows if int(r["count"]) == block_length
                )
                LOGGER.info("[progress] track=%d block_start=%d block_records=%d elapsed=%s",
                            track, block_start, len(sampled_rows), logging_config.format_duration(elapsed))
        LOGGER.info("[track] end track=%d full_blocks=%d", track, len(full_blocks))
    if not sampled_rows or exact_checks != sum(blocks_by_track.values()):
        raise RuntimeError("sampled block outputs or exact 32-point checks are incomplete")
    LOGGER.info("[cache] end tracks=10 complete_blocks=%d exact_32_point_checks=%d",
                sum(blocks_by_track.values()), exact_checks)

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
    track_groups: dict[tuple[int, int, str, str, str], list[dict]] = defaultdict(list)
    for row in replicate_rows:
        track_groups[(int(row["sample_count"]), int(row["replicate"]), row["target"],
                      row["proxy_stat"], row["metric"])].append(row)
    replicate_means: dict[tuple[int, int, str, str, str], float] = {}
    for key, rows in track_groups.items():
        values = [float(r["correlation"]) for r in rows if r["correlation"] is not None]
        if values:
            replicate_means[key] = float(np.mean(values))
    summary_rows = []
    for count in sample_counts:
        count_reps = [0] if count == block_length else list(range(repeats))
        for target, proxy in (("full_variance", "sampled_head_variance"),
                              ("full_direction_dispersion", "sampled_head_direction_dispersion")):
            for metric in ("pearson", "spearman"):
                vals = [replicate_means[(count, rep, target, proxy, metric)]
                        for rep in count_reps
                        if (count, rep, target, proxy, metric) in replicate_means]
                if not vals:
                    raise RuntimeError("no finite replicate correlations for count %d" % count)
                summary_rows.append({
                    "sample_count": count, "target": target, "proxy_stat": proxy, "metric": metric,
                    "tracks": len(set(blocks_by_track)), "total_blocks": sum(blocks_by_track.values()),
                    "replicates": len(vals), "equal_track_mean": float(np.mean(vals)),
                    "replicate_median": float(np.median(vals)),
                    "replicate_p10": float(np.quantile(vals, 0.10)),
                    "replicate_p90": float(np.quantile(vals, 0.90)),
                    "replicate_min": float(np.min(vals)), "replicate_max": float(np.max(vals))
                })
    metadata = {"experiment": "v0-gradient-proxy-test", "parent_part": "Local-Risk",
                "substudy": "Local-Risk sampling extension", "sample_counts": sample_counts,
                "block_length": block_length, "replicates_for_counts_below_32": repeats,
                "count_32_is_exact_full_block": True, "seed": int(cfg["seed"]),
                "sampling": "for each block and replicate, draw one uniform permutation of the 32 positions; counts 4, 8, 16, and 24 use nested prefixes; no replacement",
                "targets": ["full vector variance across all 32 full gradients",
                            "full direction dispersion across all 32 full gradients"],
                "proxies": ["output-head variance from sampled head gradients",
                            "output-head direction dispersion from sampled head gradients"],
                "correlation_summary": "per-track Pearson and Spearman across complete blocks; equal-weight mean across ten tracks; distribution across fixed-seed sampling replicates",
                "complete_blocks_by_track": {str(k): v for k, v in blocks_by_track.items()},
                "total_complete_blocks": int(sum(blocks_by_track.values())),
                "exact_count_32_checks": exact_checks,
                "uses_saved_full_and_head_gram_matrices": True,
                "model_or_gradient_recomputation": False,
                "elapsed_seconds": time.monotonic() - start_time}
    write_csv(out / "sampled_block_statistics.csv", sampled_rows)
    write_csv(out / "replicate_correlations.csv", replicate_rows)
    write_csv(out / "summary.csv", summary_rows)
    write_json(out / "input_manifest.json", metadata)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("[complete] sampled_records=%d correlations=%d elapsed=%s output=%s",
                len(sampled_rows), len(replicate_rows),
                logging_config.format_duration(time.monotonic() - start_time), out)


if __name__ == "__main__":
    main()
