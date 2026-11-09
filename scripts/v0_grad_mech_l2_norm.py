"""Summarize relative lag-L2 distances from ten retained full-gradient caches."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path


LOGGER = logging.getLogger("v0.grad_mech_l2_norm")
SELECTED_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 575)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(values: np.ndarray) -> dict:
    return {
        "pair_count": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def direct_check(cache_path: Path, windows: int, coordinates: int,
                 gram: np.ndarray, norms: np.ndarray) -> list[dict]:
    raw = np.memmap(cache_path, mode="r", dtype=np.uint16, shape=(windows, coordinates))
    pairs = sorted(set(((0, 1), (windows // 2, windows // 2 + 1),
                        (0, min(575, windows - 1)))))
    checks = []
    for i, j in pairs:
        bits = np.array(raw[[i, j]], copy=True).astype(np.uint32)
        vectors = np.left_shift(bits, 16).view(np.float32).astype(np.float64)
        direct = float(np.sqrt(np.sum(np.square(vectors[0] - vectors[1]), dtype=np.float64)))
        reconstructed = float(np.sqrt(max(0.0, norms[i] ** 2 + norms[j] ** 2 -
                                           2.0 * gram[i, j])))
        absolute_error = abs(direct - reconstructed)
        if absolute_error > 0.002 * max(1.0, direct):
            raise ValueError("Gram L2 differs from direct BF16 L2: pair=(%d,%d) error=%g" %
                             (i, j, absolute_error))
        checks.append({"start_i": i, "start_j": j, "direct_l2": direct,
                       "gram_l2": reconstructed, "absolute_error": absolute_error})
        del vectors
    del raw
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-grad-mech-l2-norm.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="ascii"))
    if config["experiment"] != "v0-grad-mech-l2-norm":
        raise ValueError("unexpected experiment")
    out = output_path("train", config["output"])
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "config.json"
    if existing.exists() and json.loads(existing.read_text(encoding="ascii")) != config:
        raise ValueError("output config changed")
    write_json(existing, config)
    LOGGER.info("Setup start: config=%s output=%s trajectories=%d max_lag=%d",
                args.config, out, len(config["source_ordinals"]), config["max_lag"])
    started = time.monotonic()

    source_manifest = json.loads((PROJECT_ROOT / config["source_manifest"]).read_text())
    source_rows = source_manifest["trajectories"]
    cache_root = PROJECT_ROOT / config["gradient_cache"]
    gram_root = PROJECT_ROOT / config["gram_source"]
    if not (PROJECT_ROOT / "outputs/train/v0-grad-mech-test/.complete").is_file():
        raise ValueError("source gradient probe is incomplete")
    if not (gram_root / ".complete").is_file():
        raise ValueError("source Gram matrices are incomplete")

    trajectory_rows = []
    pooled = [[] for _ in range(int(config["max_lag"]))]
    lag1_rows = []
    audit = []
    windows_total = 0
    clipped_total = 0
    for track, ordinal in enumerate(config["source_ordinals"]):
        LOGGER.info("Trajectory start: track=%d ordinal=%d", track, ordinal)
        source = source_rows[ordinal]
        proxy_dir = gram_root / ("trajectory-%02d" % track)
        proxy_manifest = json.loads((proxy_dir / "manifest.json").read_text(encoding="ascii"))
        windows = int(source["length"]) - int(config["window_span"]) + 1
        if (proxy_manifest["ordinal"] != ordinal or
                proxy_manifest["series_id"] != source["series_id"] or
                proxy_manifest["source_row"] != source["source_row"] or
                proxy_manifest["windows"] != windows or
                proxy_manifest["full_gradient_dtype"] != "BF16 from FP32 BP" or
                not (proxy_dir / ".complete").is_file()):
            raise ValueError("source manifest or Gram source mismatch for track %d" % track)
        cache_path = cache_root / ("trajectory-%02d.bf16" % track)
        if (not cache_path.with_suffix(".complete").is_file() or
                cache_path.stat().st_size != windows * int(config["gradient_coordinates"]) * 2):
            raise ValueError("gradient cache incomplete or incorrectly sized for track %d" % track)
        with np.load(proxy_dir / "gram_matrices.npz", allow_pickle=False) as matrices:
            gram = matrices["full"]
        if gram.shape != (windows, windows) or not np.isfinite(gram).all():
            raise ValueError("invalid Gram matrix for track %d" % track)
        if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-8):
            raise ValueError("asymmetric Gram matrix for track %d" % track)
        diagonal = np.diag(gram)
        if np.any(diagonal <= 0):
            raise ValueError("zero or negative gradient norm for track %d" % track)
        norms = np.sqrt(diagonal)
        checks = direct_check(cache_path, windows, int(config["gradient_coordinates"]),
                              gram, norms)
        audit.append({"track": track, "ordinal": ordinal, "windows": windows,
                      "direct_pair_checks": checks})
        windows_total += windows
        for lag in range(1, int(config["max_lag"]) + 1):
            if lag >= windows:
                continue
            count = windows - lag
            squared = diagonal[:-lag] + diagonal[lag:] - 2.0 * np.diagonal(gram, offset=lag)
            mean_norm = (norms[:-lag] + norms[lag:]) / 2.0
            negative = squared < 0
            if np.any(squared[negative] < -1e-5 * np.square(mean_norm[negative])):
                raise ValueError("substantial negative squared L2 for track %d lag %d" %
                                 (track, lag))
            clipped_total += int(np.count_nonzero(negative))
            values = np.sqrt(np.maximum(squared, 0.0)) / mean_norm
            if values.size != count or not np.isfinite(values).all() or np.any(values > 2.0001):
                raise ValueError("invalid relative L2 for track %d lag %d" % (track, lag))
            pooled[lag - 1].append(values)
            trajectory_rows.append({"track": track, "ordinal": ordinal,
                                    "series_id": source["series_id"], "lag": lag,
                                    "overlap_ratio": (int(config["window_span"]) - lag) /
                                                     int(config["window_span"]),
                                    **summarize(values)})
            if lag == 1:
                lag1_rows.extend({"track": track, "ordinal": ordinal, "start_i": i,
                                  "start_j": i + 1, "relative_l2": float(value)}
                                 for i, value in enumerate(values))
        LOGGER.info("Trajectory end: track=%d windows=%d checks=%d elapsed=%s",
                    track, windows, len(checks), format_duration(time.monotonic() - started))
    if windows_total != config["expected_windows"]:
        raise ValueError("unexpected total window count")
    if len(trajectory_rows) != len(config["source_ordinals"]) * int(config["max_lag"]):
        raise ValueError("missing trajectory-lag rows")

    pooled_rows = []
    for lag, parts in enumerate(pooled, start=1):
        values = np.concatenate(parts)
        track_means = [float(np.mean(part)) for part in parts]
        pooled_rows.append({"lag": lag,
                            "overlap_ratio": (int(config["window_span"]) - lag) /
                                             int(config["window_span"]),
                            "trajectory_count": len(parts),
                            **summarize(values),
                            "equal_trajectory_mean": float(np.mean(track_means))})
    write_csv(out / "trajectory_lag.csv", trajectory_rows)
    write_csv(out / "lag_summary.csv", pooled_rows)
    write_csv(out / "lag1_pairs.csv", lag1_rows)
    write_json(out / "audit.json", {"gradient_source": config["gradient_cache"],
                                   "gram_source": config["gram_source"],
                                   "gram_method": "FP32 tiled GEMM with FP64 tile accumulation",
                                   "cache_dtype": "BF16 copied from FP32 gradients",
                                   "clipped_small_negative_squared_distances": clipped_total,
                                   "checks": audit})
    write_json(out / "summary.json", {
        "experiment": config["experiment"],
        "trajectory_count": len(config["source_ordinals"]),
        "window_count": windows_total,
        "lag1_pair_count": pooled_rows[0]["pair_count"],
        "selected_lags": [pooled_rows[lag - 1] for lag in SELECTED_LAGS],
        "elapsed_seconds": time.monotonic() - started,
    })
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("Complete: trajectories=%d windows=%d lag1_pairs=%d clips=%d elapsed=%s output=%s",
                len(config["source_ordinals"]), windows_total, pooled_rows[0]["pair_count"],
                clipped_total, format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    LOGGER = configure_logger("grad_mech_l2_norm", "v0-grad-mech-l2-norm",
                              "experiment", "analysis")
    for handler in LOGGER.handlers:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s-[grad_mech_l2_norm]-[analysis]-[%(message)s]"))
    try:
        main()
    except Exception:
        LOGGER.exception("Experiment failed")
        raise
