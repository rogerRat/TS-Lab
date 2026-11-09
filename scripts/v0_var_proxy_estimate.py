"""Relate sparse head-gradient variance to four-anchor reconstruction error."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, config_file, output_path
from scripts.v0_vage_static import spaced


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty result table")
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def reconstruct_metrics(gram: np.ndarray, anchors: np.ndarray) -> dict[str, float]:
    positions = np.arange(len(gram))
    right_i = np.searchsorted(anchors, positions, side="right")
    left = anchors[np.clip(right_i - 1, 0, len(anchors) - 1)]
    right = anchors[np.clip(right_i, 0, len(anchors) - 1)]
    alpha = np.divide(positions - left, right - left,
                      out=np.zeros(len(gram), dtype=np.float64), where=right != left)
    a, b = 1.0 - alpha, alpha
    truth_sq = np.diag(gram)
    estimate_sq = a * a * gram[left, left] + b * b * gram[right, right] + 2 * a * b * gram[left, right]
    dot = a * gram[positions, left] + b * gram[positions, right]
    if np.any(truth_sq <= 0) or np.any(estimate_sq <= 0):
        raise ValueError("nonpositive gradient norm")
    l2 = np.sqrt(np.maximum(0.0, truth_sq + estimate_sq - 2 * dot))
    cosine = np.clip(dot / np.sqrt(truth_sq * estimate_sq), -1.0, 1.0)
    norm_diff = np.abs(np.sqrt(estimate_sq) - np.sqrt(truth_sq))
    l2[anchors] = 0.0
    cosine[anchors] = 1.0
    norm_diff[anchors] = 0.0
    return {"mean_l2_error": float(l2.mean()),
            "mean_cosine": float(cosine.mean()),
            "mean_direction_error": float((1.0 - cosine).mean()),
            "mean_abs_norm_difference": float(norm_diff.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/v0-var-proxy-estimate.json")
    args = parser.parse_args()
    cfg = json.loads(config_file(args.config).read_text(encoding="ascii"))
    if (cfg.get("experiment"), cfg.get("block_length"), cfg.get("proxy_count"),
            cfg.get("full_anchor_count")) != ("v0-var-proxy-estimate", 256, 8, 4):
        raise ValueError("unexpected experiment configuration")
    out = output_path("train", cfg["output"])
    if (out / ".complete").exists():
        raise ValueError("completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.var.proxy.estimate", "v0-var-proxy-estimate", "replay", "initialization",
                           log_directory=PROJECT_ROOT / "logs/v0-var-proxy-estimate")
    started = time.monotonic()
    log.info("[configuration] block_length=256 proxy_count=8 full_anchor_count=4 source=%s output=%s",
             cfg["source"], out)
    source = PROJECT_ROOT / cfg["source"]
    proxy_positions = np.asarray(spaced(0, 256, 8), dtype=np.int64)
    anchors = np.asarray(spaced(0, 256, 4), dtype=np.int64)
    rows = []
    remainder = {}
    log.info("[cache] start reading ten complete trajectory Gram caches")
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        if not (directory / ".complete").is_file():
            raise ValueError("incomplete trajectory: %s" % name)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        count = int(manifest["windows"])
        with np.load(directory / "gram_matrices.npz") as archive:
            full = np.asarray(archive["full"], dtype=np.float64)
            head = np.asarray(archive["head"], dtype=np.float64)
        for label, gram in (("full", full), ("head", head)):
            if gram.shape != (count, count) or not np.isfinite(gram).all():
                raise ValueError("invalid %s Gram cache: %s" % (label, name))
            if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-7):
                raise ValueError("asymmetric %s Gram cache: %s" % (label, name))
        complete = count // 256
        remainder[name] = count % 256
        for block in range(complete):
            start = block * 256
            selected = start + proxy_positions
            proxy = head[np.ix_(selected, selected)]
            proxy_variance = float(np.diag(proxy).mean() - proxy.mean())
            if proxy_variance < -1e-9 * max(1.0, float(np.diag(proxy).mean())):
                raise ValueError("negative proxy variance: %s block=%d" % (name, block))
            block_full = full[start:start + 256, start:start + 256]
            metrics = reconstruct_metrics(block_full, anchors)
            rows.append({"track": name, "block": block, "block_start": start,
                         "block_end_inclusive": start + 255,
                         "proxy_variance": max(0.0, proxy_variance), **metrics})
        log.info("[track] complete track=%s windows=%d blocks=%d terminal_remainder=%d elapsed=%s",
                 name, count, complete, remainder[name], format_duration(time.monotonic() - started))
    targets = ("mean_l2_error", "mean_cosine", "mean_direction_error", "mean_abs_norm_difference")
    correlations = []
    for scope in ["pooled_blocks"] + ["trajectory-%02d" % i for i in range(10)]:
        selected_rows = rows if scope == "pooled_blocks" else [row for row in rows if row["track"] == scope]
        x = np.asarray([row["proxy_variance"] for row in selected_rows])
        for target in targets:
            y = np.asarray([row[target] for row in selected_rows])
            for method in ("pearson", "spearman"):
                correlations.append({"scope": scope, "blocks": len(selected_rows), "target": target,
                                     "method": method, "correlation": correlation(x, y, method)})
    for target in targets:
        for method in ("pearson", "spearman"):
            values = [row["correlation"] for row in correlations if row["target"] == target
                      and row["method"] == method and row["scope"].startswith("trajectory-")
                      and row["correlation"] is not None]
            correlations.append({"scope": "equal_track_mean", "blocks": len(rows), "target": target,
                                 "method": method, "correlation": float(np.mean(values)) if values else None})
    write_csv(out / "block_metrics.csv", rows)
    write_csv(out / "correlations.csv", correlations)
    (out / "input_manifest.json").write_text(json.dumps({"source": cfg["source"],
        "blocks": len(rows), "terminal_remainders": remainder,
        "proxy_positions": proxy_positions.tolist(), "full_anchor_positions": anchors.tolist(),
        "proxy_variance": "population vector variance of eight output-head gradients",
        "full_gradients": "retrospective exact four-anchor replay from saved full Gram matrices",
        "reconstruction": "linear interpolation with exact queried anchors",
        "aggregation": "mean over all 256 positions including four anchors"}, indent=2) + "\n", encoding="ascii")
    (out / "config_snapshot.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("[complete] blocks=%d elapsed=%s output=%s", len(rows),
             format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
