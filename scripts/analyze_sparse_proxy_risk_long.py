"""Measure sparse proxy-risk correlations in longer start-index blocks."""
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

LOGGER = logging.getLogger("v0.gradient_proxy_local_risk_long")
BLOCK_LENGTHS = [128, 256, 576]
SAMPLE_FRACTIONS = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]


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


def correlation(x: np.ndarray, y: np.ndarray, metric: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = pearsonr(x, y).statistic if metric == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def statistics(gram: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    sub = np.asarray(gram[np.ix_(indices, indices)], dtype=np.float64)
    diagonal = np.diag(sub)
    if not np.isfinite(sub).all() or np.any(diagonal <= 0):
        raise ValueError("invalid Gram block or nonpositive gradient norm")
    variance = float(diagonal.mean() - sub.mean())
    if variance < -1e-10 * max(1.0, float(diagonal.mean())):
        raise ValueError("subsample variance is substantially negative")
    variance = max(0.0, variance)
    norms = np.sqrt(diagonal)
    cosine = np.clip(sub / (norms[:, None] * norms[None, :]), -1.0, 1.0)
    dispersion = float(1.0 - cosine.mean())
    if dispersion < -1e-8 or dispersion > 2.0 + 1e-8:
        raise ValueError("direction dispersion outside expected bounds")
    return variance, min(2.0, max(0.0, dispersion))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-gradient-proxy-local-risk-long.json")
    args = parser.parse_args()
    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="ascii"))
    if cfg.get("substudy") != "Local-Risk long-block sparse sampling":
        raise ValueError("unexpected long-block protocol")
    if cfg.get("block_lengths") != BLOCK_LENGTHS or cfg.get("sample_fractions") != SAMPLE_FRACTIONS:
        raise ValueError("fixed block lengths or sample fractions changed")
    repeats = int(cfg["replicates"])
    if repeats != 20:
        raise ValueError("fixed sampling replicate count changed")
    log_dir = PROJECT_ROOT / "logs/v0-gradient-proxy-test/local-risk-long"
    LOGGER = configure_logger("v0.gradient_proxy_local_risk_long", "v0-gradient-proxy-test",
                              "local-risk-long", "initialization", log_directory=log_dir)
    out = output_path("train", cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".complete").exists():
        raise RuntimeError("completion marker already exists; refusing to overwrite")
    source = PROJECT_ROOT / cfg["source_output"]
    rng = np.random.default_rng(int(cfg["seed"]))
    started = time.monotonic()
    LOGGER.info("[configuration] block_lengths=%s sample_fractions=%s replicates=%d seed=%d output=%s",
                BLOCK_LENGTHS, SAMPLE_FRACTIONS, repeats, cfg["seed"], out)
    sample_rows: list[dict] = []
    blocks_by_length_track: dict[tuple[int, int], int] = {}
    full_baseline_counts: dict[int, int] = {}
    LOGGER.info("[cache] start reading all-coordinate and output-head Gram matrices")
    for track in range(10):
        track_dir = source / ("trajectory-%02d" % track)
        if not (track_dir / ".complete").exists():
            raise ValueError("incomplete proxy track %d" % track)
        manifest = json.loads((track_dir / "manifest.json").read_text(encoding="ascii"))
        n_windows = int(manifest["windows"])
        with np.load(track_dir / "gram_matrices.npz") as archive:
            full_gram = np.asarray(archive["full"], dtype=np.float64)
            head_gram = np.asarray(archive["head"], dtype=np.float64)
        for name, gram in (("full", full_gram), ("head", head_gram)):
            if gram.shape != (n_windows, n_windows):
                raise ValueError("unexpected %s Gram matrix shape for track %d" % (name, track))
            if not np.isfinite(gram).all() or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-7):
                raise ValueError("invalid %s Gram matrix for track %d" % (name, track))
        for length in BLOCK_LENGTHS:
            block_count = n_windows // length
            blocks_by_length_track[(length, track)] = block_count
            counts = [int(round(length * fraction)) for fraction in SAMPLE_FRACTIONS]
            if counts != sorted(set(counts), reverse=True) or counts[0] != length:
                raise ValueError("invalid nested counts for block length %d" % length)
            for block_index in range(block_count):
                block_start = block_index * length
                all_indices = np.arange(block_start, block_start + length, dtype=np.int64)
                full_variance, full_direction = statistics(full_gram, all_indices)
                for repeat in range(repeats):
                    permutation = rng.permutation(length)
                    for count in counts:
                        if count == length:
                            if repeat > 0:
                                continue
                            selected = all_indices
                            replicate = 0
                        else:
                            selected = block_start + np.sort(permutation[:count])
                            replicate = repeat
                        head_variance, head_direction = statistics(head_gram, selected)
                        if count == length:
                            full_baseline_counts[length] = full_baseline_counts.get(length, 0) + 1
                        sample_rows.append({
                            "block_length": length, "track": track, "block_start": block_start,
                            "count": count, "fraction": count / length, "replicate": replicate,
                            "full_variance": full_variance,
                            "full_direction_dispersion": full_direction,
                            "sampled_head_variance": head_variance,
                            "sampled_head_direction_dispersion": head_direction,
                            "selected_positions": ",".join(str(int(x - block_start)) for x in selected)
                        })
        LOGGER.info("[track] end track=%d windows=%d blocks=%s",
                    track, n_windows,
                    {str(length): blocks_by_length_track[(length, track)] for length in BLOCK_LENGTHS})
    for length in BLOCK_LENGTHS:
        expected = sum(blocks_by_length_track[(length, track)] for track in range(10))
        if expected < 1 or full_baseline_counts.get(length, 0) != expected:
            raise RuntimeError("incomplete all-point baseline at block length %d" % length)
    LOGGER.info("[cache] end block_totals=%s all-point_baselines=%s",
                {str(length): sum(blocks_by_length_track[(length, t)] for t in range(10))
                 for length in BLOCK_LENGTHS}, full_baseline_counts)

    groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in sample_rows:
        groups[(int(row["block_length"]), int(row["count"]),
                int(row["replicate"]), int(row["track"]))].append(row)
    replicate_rows: list[dict] = []
    for (length, count, replicate, track), rows in sorted(groups.items()):
        for target, proxy in (("full_variance", "sampled_head_variance"),
                              ("full_direction_dispersion", "sampled_head_direction_dispersion")):
            x = np.asarray([row[target] for row in rows], dtype=np.float64)
            y = np.asarray([row[proxy] for row in rows], dtype=np.float64)
            for metric in ("pearson", "spearman"):
                replicate_rows.append({
                    "block_length": length, "sample_count": count, "replicate": replicate,
                    "scope": "track", "track": track, "block_count": len(rows),
                    "target": target, "proxy_stat": proxy, "metric": metric,
                    "correlation": correlation(x, y, metric)
                })
    all_groups: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in sample_rows:
        all_groups[(int(row["block_length"]), int(row["count"]), int(row["replicate"]))].append(row)
    for (length, count, replicate), rows in sorted(all_groups.items()):
        for target, proxy in (("full_variance", "sampled_head_variance"),
                              ("full_direction_dispersion", "sampled_head_direction_dispersion")):
            x = np.asarray([row[target] for row in rows], dtype=np.float64)
            y = np.asarray([row[proxy] for row in rows], dtype=np.float64)
            for metric in ("pearson", "spearman"):
                replicate_rows.append({
                    "block_length": length, "sample_count": count, "replicate": replicate,
                    "scope": "pooled_blocks", "track": "all", "block_count": len(rows),
                    "target": target, "proxy_stat": proxy, "metric": metric,
                    "correlation": correlation(x, y, metric)
                })

    summary_rows = []
    for length in BLOCK_LENGTHS:
        counts = [int(round(length * fraction)) for fraction in SAMPLE_FRACTIONS]
        for count in counts:
            replicate_ids = [0] if count == length else list(range(repeats))
            for scope in ("pooled_blocks", "equal_track_mean"):
                for target, proxy in (("full_variance", "sampled_head_variance"),
                                      ("full_direction_dispersion", "sampled_head_direction_dispersion")):
                    for metric in ("pearson", "spearman"):
                        replicate_values = []
                        tracks_per_replicate = []
                        for rep in replicate_ids:
                            if scope == "pooled_blocks":
                                matches = [r for r in replicate_rows
                                           if int(r["block_length"]) == length
                                           and int(r["sample_count"]) == count
                                           and int(r["replicate"]) == rep
                                           and r["scope"] == "pooled_blocks"
                                           and r["target"] == target and r["metric"] == metric]
                                values = [float(r["correlation"]) for r in matches if r["correlation"] is not None]
                            else:
                                matches = [r for r in replicate_rows
                                           if int(r["block_length"]) == length
                                           and int(r["sample_count"]) == count
                                           and int(r["replicate"]) == rep
                                           and r["scope"] == "track" and r["target"] == target
                                           and r["metric"] == metric]
                                values = [float(r["correlation"]) for r in matches if r["correlation"] is not None]
                            if values:
                                replicate_values.append(float(np.mean(values)))
                                tracks_per_replicate.append(len(values))
                        if replicate_values:
                            summary_rows.append({
                                "block_length": length, "sample_count": count, "fraction": count / length,
                                "target": target, "proxy_stat": proxy, "metric": metric, "scope": scope,
                                "complete_blocks": sum(blocks_by_length_track[(length, t)] for t in range(10)),
                                "replicates": len(replicate_values),
                                "tracks_with_correlation_median": float(np.median(tracks_per_replicate)),
                                "correlation_mean": float(np.mean(replicate_values)),
                                "correlation_median": float(np.median(replicate_values)),
                                "sampling_p10": float(np.quantile(replicate_values, 0.10)),
                                "sampling_p90": float(np.quantile(replicate_values, 0.90)),
                                "sampling_min": float(np.min(replicate_values)),
                                "sampling_max": float(np.max(replicate_values))
                            })
                        else:
                            summary_rows.append({
                                "block_length": length, "sample_count": count, "fraction": count / length,
                                "target": target, "proxy_stat": proxy, "metric": metric, "scope": scope,
                                "complete_blocks": sum(blocks_by_length_track[(length, t)] for t in range(10)),
                                "replicates": 0, "tracks_with_correlation_median": 0,
                                "correlation_mean": "", "correlation_median": "",
                                "sampling_p10": "", "sampling_p90": "", "sampling_min": "", "sampling_max": ""
                            })
    block_totals = {str(length): sum(blocks_by_length_track[(length, track)] for track in range(10))
                    for length in BLOCK_LENGTHS}
    metadata = {
        "experiment": "v0-gradient-proxy-test", "parent_part": "Local-Risk",
        "substudy": "Local-Risk long-block sparse sampling",
        "block_lengths": BLOCK_LENGTHS,
        "sample_fractions": SAMPLE_FRACTIONS,
        "sample_counts": {str(length): [int(round(length * f)) for f in SAMPLE_FRACTIONS]
                          for length in BLOCK_LENGTHS},
        "replicates_for_counts_below_full": repeats, "seed": int(cfg["seed"]),
        "sampling": "per block and replicate draw one uniform permutation; proxy count sets use nested prefixes without replacement",
        "targets": ["full vector variance and direction dispersion computed from every full gradient in each block"],
        "proxies": ["output-head variance and direction dispersion from sampled head gradients"],
        "complete_blocks_by_length": block_totals,
        "complete_blocks_by_track": {str(length): {
            str(track): blocks_by_length_track[(length, track)] for track in range(10)}
            for length in BLOCK_LENGTHS},
        "all_point_baselines": full_baseline_counts,
        "within_track_correlations_require_at_least_three_blocks": True,
        "pooled_576_block_result_is_exploratory": True,
        "uses_saved_full_and_head_gram_matrices": True,
        "model_or_gradient_recomputation": False,
        "elapsed_seconds": time.monotonic() - started
    }
    LOGGER.info("[write] start output tables")
    write_csv(out / "sampled_block_statistics.csv", sample_rows)
    write_csv(out / "replicate_correlations.csv", replicate_rows)
    write_csv(out / "summary.csv", summary_rows)
    write_json(out / "input_manifest.json", metadata)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("[complete] sampled_records=%d correlations=%d summary_rows=%d elapsed=%s output=%s",
                len(sample_rows), len(replicate_rows), len(summary_rows),
                logging_config.format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
