"""Evaluate four-anchor vector mean as a constant block representative."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from scripts.v0_var_proxy_estimate import correlation, write_csv

ANCHORS = np.asarray([0, 85, 170, 255], dtype=np.int64)
TARGETS = ("mean_l2_error", "mean_cosine", "mean_direction_error",
           "mean_abs_norm_difference")


def constant_metrics(gram: np.ndarray) -> dict[str, float]:
    if gram.shape != (256, 256) or not np.isfinite(gram).all():
        raise ValueError("invalid 256-point full-gradient Gram block")
    truth_sq = np.diag(gram)
    estimate_sq = float(gram[np.ix_(ANCHORS, ANCHORS)].mean())
    dot = gram[:, ANCHORS].mean(axis=1)
    if estimate_sq <= 0 or np.any(truth_sq <= 0):
        raise ValueError("nonpositive gradient norm")
    error = np.sqrt(np.maximum(0.0, truth_sq + estimate_sq - 2.0 * dot))
    cosine = np.clip(dot / np.sqrt(truth_sq * estimate_sq), -1.0, 1.0)
    norm_difference = np.abs(np.sqrt(truth_sq) - np.sqrt(estimate_sq))
    return {"mean_l2_error": float(error.mean()),
            "mean_cosine": float(cosine.mean()),
            "mean_direction_error": float((1.0 - cosine).mean()),
            "mean_abs_norm_difference": float(norm_difference.mean())}


def main() -> None:
    parent = PROJECT_ROOT / "outputs/train/v0-var-proxy-estimate"
    true_source = parent / "true_variance"
    cache_source = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
    out = output_path("train", "outputs/train/v0-var-proxy-estimate/four_anchor_mean")
    if not (parent / ".complete").is_file() or not (true_source / ".complete").is_file():
        raise ValueError("incomplete parent experiments")
    if (out / ".complete").exists():
        raise ValueError("completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.four.anchor.mean", "v0-var-proxy-estimate",
                           "four-anchor-mean", "initialization",
                           log_directory=PROJECT_ROOT / "logs/v0-var-proxy-estimate")
    started = time.monotonic()
    log.info("[configuration] blocks=46 anchors=%s output=%s representative=arithmetic_vector_mean",
             ANCHORS.tolist(), out)
    with (true_source / "block_metrics.csv").open(encoding="ascii", newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 46:
        raise ValueError("unexpected block count")
    rows = []
    for track in range(10):
        name = "trajectory-%02d" % track
        cache = cache_source / name
        if not (cache / ".complete").is_file():
            raise ValueError("incomplete gradient cache: %s" % name)
        manifest = json.loads((cache / "manifest.json").read_text(encoding="ascii"))
        count = int(manifest["windows"])
        with np.load(cache / "gram_matrices.npz") as archive:
            gram = np.asarray(archive["full"], dtype=np.float64)
        if gram.shape != (count, count) or not np.isfinite(gram).all():
            raise ValueError("invalid full Gram cache: %s" % name)
        track_rows = [r for r in original if r["track"] == name]
        if len(track_rows) != count // 256:
            raise ValueError("block count mismatch: %s" % name)
        for prior in track_rows:
            block = int(prior["block"])
            start = block * 256
            if int(prior["block_start"]) != start:
                raise ValueError("block alignment mismatch: %s" % name)
            metrics = constant_metrics(gram[start:start + 256, start:start + 256])
            rows.append({"track": name, "block": block, "block_start": start,
                         "true_full_variance": float(prior["true_full_variance"]),
                         "proxy_variance": float(prior["proxy_variance"]),
                         **metrics,
                         "linear_mean_l2_error": float(prior["mean_l2_error"]),
                         "linear_mean_cosine": float(prior["mean_cosine"]),
                         "linear_mean_abs_norm_difference": float(prior["mean_abs_norm_difference"])})
        log.info("[track] complete track=%s blocks=%d elapsed=%s", name, len(track_rows),
                 format_duration(time.monotonic() - started))
    corr = []
    for predictor in ("proxy_variance", "true_full_variance"):
        for scope in ["pooled_blocks"] + ["trajectory-%02d" % i for i in range(10)]:
            chosen = rows if scope == "pooled_blocks" else [r for r in rows if r["track"] == scope]
            x = np.asarray([r[predictor] for r in chosen])
            for target in TARGETS:
                y = np.asarray([r[target] for r in chosen])
                for method in ("pearson", "spearman"):
                    corr.append({"predictor": predictor, "scope": scope, "blocks": len(chosen),
                                 "target": target, "method": method,
                                 "correlation": correlation(x, y, method)})
        for target in TARGETS:
            for method in ("pearson", "spearman"):
                values = [r["correlation"] for r in corr if r["predictor"] == predictor
                          and r["target"] == target and r["method"] == method
                          and r["scope"].startswith("trajectory-") and r["correlation"] is not None]
                corr.append({"predictor": predictor, "scope": "equal_track_mean",
                             "blocks": len(rows), "target": target, "method": method,
                             "correlation": float(np.mean(values)) if values else None})
    write_csv(out / "block_metrics.csv", rows)
    write_csv(out / "correlations.csv", corr)
    summary = {key: float(np.mean([r[key] for r in rows])) for key in
               (*TARGETS, "linear_mean_l2_error", "linear_mean_cosine",
                "linear_mean_abs_norm_difference")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    (out / "input_manifest.json").write_text(json.dumps({
        "parent": "outputs/train/v0-var-proxy-estimate/true_variance/block_metrics.csv",
        "source": "outputs/train/v0-gradient-proxy-test",
        "block_length": 256, "blocks": len(rows), "anchor_positions": ANCHORS.tolist(),
        "representative": "arithmetic vector mean of four true full-gradient anchors",
        "evaluation": "same representative at all 256 positions including the four anchors",
        "predictors": ["eight-sample output-head variance", "all-256 true full-gradient variance"]},
        indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("[complete] blocks=%d elapsed=%s output=%s", len(rows),
             format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
