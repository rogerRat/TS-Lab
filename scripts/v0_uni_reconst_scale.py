"""Replay uniform full-gradient anchor scales with linear interpolation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, config_file, output_path
from scripts.v0_vage_static import reconstruction, spaced


def evaluate(gram, anchors):
    error, cosine, _ = reconstruction(gram, anchors)
    # Queried anchors return their observed vectors exactly.
    error[anchors] = 0.0
    cosine[anchors] = 1.0
    return float(np.mean(error, dtype=np.float64)), float(np.mean(cosine, dtype=np.float64))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/v0-uni-reconst-scale.json")
    args = parser.parse_args()
    config = json.loads(config_file(args.config).read_text(encoding="ascii"))
    denominators = [int(value) for value in config["denominators"]]
    if config["experiment"] != "v0-uni-reconst-scale" or denominators != [1,2,4,8,16,32,64,128]:
        raise ValueError("invalid experiment configuration")
    out = output_path("train", config["output"])
    if (out / ".complete").exists():
        raise ValueError("completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.uni.reconst.scale", "v0-uni-reconst-scale", "replay", "replay")
    started = time.monotonic()
    log.info("Stage start: uniform scale replay; denominators=%s output=%s", denominators, out)
    rows = []
    source = PROJECT_ROOT / config["source"]
    gradient_source = PROJECT_ROOT / config["gradient_source"]
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        n = int(manifest["windows"])
        gradient_file = gradient_source / (name + ".bf16")
        if not (directory / ".complete").is_file() or not (gradient_source / (name + ".complete")).is_file():
            raise ValueError("incomplete source: %s" % name)
        if gradient_file.stat().st_size != n * 8_394_496 * 2:
            raise ValueError("full-gradient cache size mismatch: %s" % name)
        with np.load(directory / "gram_matrices.npz") as payload:
            gram = np.asarray(payload["full"], dtype=np.float64)
        if gram.shape != (n, n) or not np.isfinite(gram).all():
            raise ValueError("invalid full Gram matrix: %s" % name)
        for scale in denominators:
            budget = math.ceil(n / scale)
            anchors = spaced(0, n, budget)
            if len(anchors) != budget or len(set(anchors)) != budget or anchors[0] != 0 or anchors[-1] != n - 1:
                raise ValueError("uniform anchor mismatch: %s q=%d" % (name, scale))
            mean_l2, mean_cosine = evaluate(gram, anchors)
            rows.append(dict(track=name, windows=n, denominator=scale, anchors=budget,
                             mean_l2=mean_l2, mean_cosine=mean_cosine))
        log.info("Track complete: %s windows=%d scale_groups=%d", name, n, len(denominators))
    write_csv(out / "trajectory_metrics.csv", rows)
    summary = []
    for scale in denominators:
        selected = [row for row in rows if row["denominator"] == scale]
        total_windows = sum(row["windows"] for row in selected)
        summary.append(dict(denominator=scale, trajectories=len(selected), windows=total_windows,
                            anchors=sum(row["anchors"] for row in selected),
                            equal_track_mean_l2=float(np.mean([row["mean_l2"] for row in selected])),
                            equal_track_mean_cosine=float(np.mean([row["mean_cosine"] for row in selected])),
                            window_weighted_mean_l2=float(sum(row["mean_l2"]*row["windows"] for row in selected)/total_windows),
                            window_weighted_mean_cosine=float(sum(row["mean_cosine"]*row["windows"] for row in selected)/total_windows)))
    write_csv(out / "summary.csv", summary)
    (out / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: uniform scale replay; duration=%s output=%s", format_duration(time.monotonic()-started), out)


if __name__ == "__main__":
    main()
