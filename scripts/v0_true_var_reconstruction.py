"""Correlate complete block gradient variance with reconstruction metrics."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from scripts.v0_var_proxy_estimate import correlation, write_csv


def main() -> None:
    source = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
    parent = PROJECT_ROOT / "outputs/train/v0-var-proxy-estimate"
    out = output_path("train", "outputs/train/v0-var-proxy-estimate/true_variance")
    if not (parent / ".complete").is_file() or (out / ".complete").exists():
        raise ValueError("missing parent completion or supplement already complete")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.true.var.reconstruction", "v0-var-proxy-estimate",
                           "true-variance", "initialization",
                           log_directory=PROJECT_ROOT / "logs/v0-var-proxy-estimate")
    started = time.monotonic()
    log.info("[configuration] source=%s parent=%s output=%s block_length=256", source, parent, out)
    with (parent / "block_metrics.csv").open(encoding="ascii", newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 46:
        raise ValueError("unexpected original block count")
    rows = []
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        if not (directory / ".complete").is_file():
            raise ValueError("incomplete trajectory: %s" % name)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        count = int(manifest["windows"])
        with np.load(directory / "gram_matrices.npz") as archive:
            gram = np.asarray(archive["full"], dtype=np.float64)
        if gram.shape != (count, count) or not np.isfinite(gram).all():
            raise ValueError("invalid full Gram cache: %s" % name)
        for original_row in (r for r in original if r["track"] == name):
            block = int(original_row["block"])
            start = block * 256
            if int(original_row["block_start"]) != start or int(original_row["block_end_inclusive"]) != start + 255:
                raise ValueError("block alignment mismatch: %s block=%d" % (name, block))
            sub = gram[start:start + 256, start:start + 256]
            variance = float(np.diag(sub).mean() - sub.mean())
            if variance < -1e-9 * max(1.0, float(np.diag(sub).mean())):
                raise ValueError("negative full variance: %s block=%d" % (name, block))
            rows.append({"track": name, "block": block, "block_start": start,
                         "true_full_variance": max(0.0, variance),
                         "proxy_variance": float(original_row["proxy_variance"]),
                         **{key: float(original_row[key]) for key in
                            ("mean_l2_error", "mean_cosine", "mean_direction_error",
                             "mean_abs_norm_difference")}})
        log.info("[track] complete track=%s blocks=%d elapsed=%s", name,
                 sum(r["track"] == name for r in rows), format_duration(time.monotonic() - started))
    targets = ("mean_l2_error", "mean_cosine", "mean_direction_error",
               "mean_abs_norm_difference", "proxy_variance")
    corr = []
    for scope in ["pooled_blocks"] + ["trajectory-%02d" % i for i in range(10)]:
        selected = rows if scope == "pooled_blocks" else [r for r in rows if r["track"] == scope]
        x = np.asarray([r["true_full_variance"] for r in selected])
        for target in targets:
            y = np.asarray([r[target] for r in selected])
            for method in ("pearson", "spearman"):
                corr.append({"scope": scope, "blocks": len(selected), "target": target,
                             "method": method, "correlation": correlation(x, y, method)})
    for target in targets:
        for method in ("pearson", "spearman"):
            values = [r["correlation"] for r in corr if r["target"] == target
                      and r["method"] == method and r["scope"].startswith("trajectory-")
                      and r["correlation"] is not None]
            corr.append({"scope": "equal_track_mean", "blocks": len(rows), "target": target,
                         "method": method, "correlation": float(np.mean(values)) if values else None})
    write_csv(out / "block_metrics.csv", rows)
    write_csv(out / "correlations.csv", corr)
    (out / "input_manifest.json").write_text(json.dumps({
        "source": "outputs/train/v0-gradient-proxy-test",
        "parent_metrics": "outputs/train/v0-var-proxy-estimate/block_metrics.csv",
        "blocks": len(rows), "true_variance": "population vector variance over all 256 full gradients",
        "gradient_cache": "saved complete BF16 full-gradient Gram matrices",
        "reconstruction_metrics": "unchanged from parent experiment"}, indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("[complete] blocks=%d elapsed=%s output=%s", len(rows),
             format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
