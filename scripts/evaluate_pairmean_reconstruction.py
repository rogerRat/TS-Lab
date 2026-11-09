"""Evaluate fixed-anchor pair-mean reconstruction for VAGE and Uniform scales."""
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

DIMENSION = 8_394_496


def pairmean_metrics(gram, anchors):
    n = len(gram)
    anchors = np.asarray(anchors, dtype=np.int64)
    starts = np.arange(n)
    right_index = np.searchsorted(anchors, starts, side="right")
    left = anchors[np.clip(right_index - 1, 0, len(anchors) - 1)]
    right = anchors[np.clip(right_index, 0, len(anchors) - 1)]
    midpoint = left != right
    weight_right = np.where(midpoint, 0.5, 0.0)
    weight_left = 1.0 - weight_right
    truth_sq = np.diag(gram)
    estimate_sq = (weight_left**2 * gram[left, left]
                   + weight_right**2 * gram[right, right]
                   + 2 * weight_left * weight_right * gram[left, right])
    dot = weight_left * gram[starts, left] + weight_right * gram[starts, right]
    error_sq = np.maximum(0.0, truth_sq + estimate_sq - 2 * dot)
    cosine = np.divide(dot, np.sqrt(np.maximum(0.0, truth_sq * estimate_sq)),
                       out=np.zeros(n, dtype=np.float64), where=truth_sq * estimate_sq > 0)
    error = np.sqrt(error_sq)
    error[anchors] = 0.0
    cosine[anchors] = 1.0
    return float(np.mean(error, dtype=np.float64)), float(np.mean(np.clip(cosine, -1, 1), dtype=np.float64))


def linear_metrics(gram, anchors):
    error, cosine, _ = reconstruction(gram, anchors)
    error[anchors] = 0.0
    cosine[anchors] = 1.0
    return float(np.mean(error, dtype=np.float64)), float(np.mean(cosine, dtype=np.float64))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_grams(config):
    root = PROJECT_ROOT / config["gram_source"]
    gradient_root = PROJECT_ROOT / config["gradient_source"]
    result = {}
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        n = int(manifest["windows"])
        gradient = gradient_root / (name + ".bf16")
        if not (directory / ".complete").is_file() or not (gradient_root / (name + ".complete")).is_file():
            raise ValueError("incomplete source: %s" % name)
        if gradient.stat().st_size != n * DIMENSION * 2:
            raise ValueError("gradient cache size mismatch: %s" % name)
        with np.load(directory / "gram_matrices.npz") as payload:
            gram = np.array(payload["full"], dtype=np.float64)
        if gram.shape != (n, n) or not np.isfinite(gram).all():
            raise ValueError("invalid full Gram matrix: %s" % name)
        result[name] = gram
    return result


def evaluate(name, group, gram, anchors):
    n = len(gram)
    anchors = sorted(int(value) for value in anchors)
    if not anchors or len(set(anchors)) != len(anchors) or anchors[0] < 0 or anchors[-1] >= n:
        raise ValueError("invalid anchor set: %s" % name)
    pair_l2, pair_cos = pairmean_metrics(gram, anchors)
    linear_l2, linear_cos = linear_metrics(gram, anchors)
    return dict(group=group, method=name, windows=n, anchors=len(anchors),
                pairmean_l2=pair_l2, pairmean_cosine=pair_cos,
                linear_l2=linear_l2, linear_cosine=linear_cos,
                l2_change=pair_l2 - linear_l2, cosine_change=pair_cos - linear_cos)


def summarize(rows):
    groups = sorted(set(row["group"] for row in rows))
    result = []
    for group in groups:
        selected = [row for row in rows if row["group"] == group]
        if len(selected) != 10:
            raise ValueError("method group does not contain ten trajectories: %s" % group)
        result.append(dict(group=group, trajectories=10,
                           total_anchors=sum(row["anchors"] for row in selected),
                           pairmean_l2=float(np.mean([row["pairmean_l2"] for row in selected])),
                           linear_l2=float(np.mean([row["linear_l2"] for row in selected])),
                           l2_change=float(np.mean([row["l2_change"] for row in selected])),
                           pairmean_cosine=float(np.mean([row["pairmean_cosine"] for row in selected])),
                           linear_cosine=float(np.mean([row["linear_cosine"] for row in selected])),
                           cosine_change=float(np.mean([row["cosine_change"] for row in selected])),
                           pairmean_l2_wins=sum(row["l2_change"] < 0 for row in selected),
                           pairmean_cosine_wins=sum(row["cosine_change"] > 0 for row in selected)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/v0-pairmean-supplements.json")
    args = parser.parse_args()
    config = json.loads(config_file(args.config).read_text(encoding="ascii"))
    if config["experiment"] != "v0-pairmean-supplements":
        raise ValueError("unexpected experiment config")
    vage_out = output_path("train", config["vage_output"])
    scale_out = output_path("train", config["scale_output"])
    if (vage_out / ".complete").exists() or (scale_out / ".complete").exists():
        raise ValueError("completed supplement exists")
    vage_out.mkdir(parents=True, exist_ok=True)
    scale_out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.pairmean.supplements", "v0-pairmean-supplements", "replay", "replay")
    start = time.monotonic()
    log.info("Stage start: pair-mean reconstruction; outputs=%s,%s", vage_out, scale_out)
    grams = load_grams(config)
    vage_rows = []
    for source_name, source_path in config["vage_anchor_sources"].items():
        source = PROJECT_ROOT / source_path
        for record in json.loads(source.read_text(encoding="ascii")):
            track = record["track"]
            row = evaluate(record["method"], source_name + "/" + record["method"],
                           grams[track], record["anchors"])
            row = dict(track=track, **row)
            vage_rows.append(row)
    scale_rows = []
    for denominator in config["scale_denominators"]:
        if denominator < 1:
            raise ValueError("invalid denominator")
        for track, gram in grams.items():
            n = len(gram)
            anchors = spaced(0, n, math.ceil(n / denominator))
            row = evaluate("uniform", "s=%d" % denominator, gram, anchors)
            scale_rows.append(dict(track=track, **row))
    write_csv(vage_out / "trajectory_metrics.csv", vage_rows)
    write_csv(vage_out / "summary.csv", summarize(vage_rows))
    write_csv(scale_out / "trajectory_metrics.csv", scale_rows)
    write_csv(scale_out / "summary.csv", summarize(scale_rows))
    (vage_out / ".complete").write_text("complete\n", encoding="ascii")
    (scale_out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("Stage complete: pair-mean reconstruction; duration=%s output=%s,%s",
             format_duration(time.monotonic()-start), vage_out, scale_out)


if __name__ == "__main__":
    main()
