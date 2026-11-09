"""Replay static proxy-guided anchors against uniform reconstruction."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, config_file, output_path


def allocate(total, lengths, minimum):
    lengths = np.asarray(lengths, dtype=np.int64)
    base = np.minimum(lengths, minimum)
    if total < int(base.sum()) or total > int(lengths.sum()):
        raise ValueError("budget is outside block capacity")
    counts = base.copy()
    remaining = total - int(counts.sum())
    while remaining:
        eligible = counts < lengths
        weights = np.where(eligible, lengths, 0)
        raw = remaining * weights / weights.sum()
        add = np.minimum(np.floor(raw).astype(np.int64), lengths - counts)
        if not add.any():
            order = sorted(np.flatnonzero(eligible), key=lambda i: (-raw[i], i))
            add[order[0]] = 1
        counts += add
        remaining -= int(add.sum())
    return counts.tolist()


def spaced(lo, length, count):
    if count == 1:
        return [lo + (length - 1) // 2]
    result = np.rint(np.linspace(lo, lo + length - 1, count)).astype(int).tolist()
    if len(set(result)) != count:
        raise ValueError("duplicate initial anchors")
    return result


def split_gap(anchors, block_start, block_length):
    if len(anchors) == 1:
        left = block_start
        right = block_start + block_length - 1
        return left if anchors[0] - left >= right - anchors[0] else right
    gaps = [(right - left, left, right) for left, right in zip(anchors[:-1], anchors[1:]) if right - left > 1]
    if not gaps:
        raise ValueError("block is at capacity")
    width, left, right = min(gaps, key=lambda item: (-item[0], item[1]))
    return (left + right) // 2


def vector_variance(gram, indices):
    sub = gram[np.ix_(indices, indices)]
    if len(indices) < 2:
        return float("nan")
    value = (float(np.trace(sub)) - float(sub.sum()) / len(indices)) / (len(indices) - 1)
    return max(0.0, value)


def select_static(lengths, budget, estimates, minimum):
    blocks = []
    lo = 0
    for length in lengths:
        anchors = sorted(spaced(lo, length, min(minimum, length)))
        blocks.append(anchors)
        lo += length
    if sum(map(len, blocks)) > budget:
        raise ValueError("gradient budget cannot cover the initial anchors")
    decisions = []
    while sum(map(len, blocks)) < budget:
        scores = [length * estimates[i] / (len(anchors) * (len(anchors) + 1))
                  if len(anchors) < length else -1.0
                  for i, (length, anchors) in enumerate(zip(lengths, blocks))]
        chosen = int(np.argmax(scores))
        new_anchor = split_gap(blocks[chosen], sum(lengths[:chosen]), lengths[chosen])
        blocks[chosen].append(new_anchor)
        blocks[chosen].sort()
        decisions.append((chosen, new_anchor, scores[chosen]))
    return sorted(a for block in blocks for a in block), blocks, decisions


def reconstruction(gram, anchors):
    n = len(gram)
    anchors = np.asarray(anchors, dtype=np.int64)
    starts = np.arange(n)
    right_index = np.searchsorted(anchors, starts, side="right")
    left = anchors[np.clip(right_index - 1, 0, len(anchors) - 1)]
    right = anchors[np.clip(right_index, 0, len(anchors) - 1)]
    alpha = np.divide(starts - left, right - left, out=np.zeros(n, dtype=np.float64),
                      where=(right != left))
    a, b = 1 - alpha, alpha
    truth_sq = np.diag(gram)
    estimate_sq = a*a*gram[left, left] + b*b*gram[right, right] + 2*a*b*gram[left, right]
    dot = a*gram[starts, left] + b*gram[starts, right]
    error_sq = np.maximum(0.0, truth_sq + estimate_sq - 2*dot)
    cosine = np.divide(dot, np.sqrt(np.maximum(0, truth_sq * estimate_sq)),
                       out=np.zeros(n, dtype=np.float64), where=(truth_sq * estimate_sq > 0))
    cosine = np.clip(cosine, -1, 1)
    return np.sqrt(error_sq), cosine, truth_sq


def metrics(gram, anchors):
    error, cosine, truth_sq = reconstruction(gram, anchors)
    mask = np.ones(len(gram), dtype=bool)
    mask[anchors] = False
    return dict(mean_l2=float(error.mean()), median_l2=float(np.median(error)),
                p95_l2=float(np.quantile(error, .95)),
                mean_cosine=float(cosine.mean()), p05_cosine=float(np.quantile(cosine, .05)),
                relative_rmse=float(np.sqrt(np.sum(error**2) / np.sum(truth_sq))),
                nonanchor_mean_l2=float(error[mask].mean()) if mask.any() else 0.0,
                nonanchor_mean_cosine=float(cosine[mask].mean()) if mask.any() else 1.0)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rage/v0-vage.json")
    args = parser.parse_args()
    config = json.loads(config_file(args.config).read_text(encoding="ascii"))
    if config["experiment"] != "v0-vage" or config["budget_denominator"] <= 0:
        raise ValueError("invalid VAGE config")
    out = output_path("train", config["output"])
    if (out / ".complete").exists():
        raise ValueError("completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("v0.vage.static", "v0-vage", "replay", "replay")
    started = time.monotonic()
    logger.info("Stage start: static replay; config=%s output=%s", args.config, out)
    rows, blocks_out, selections = [], [], []
    source = PROJECT_ROOT / config["proxy_source"]
    for track in range(10):
        name = "trajectory-%02d" % track
        track_dir = source / name
        manifest = json.loads((track_dir / "manifest.json").read_text(encoding="ascii"))
        if not (track_dir / ".complete").exists():
            raise ValueError("proxy cache incomplete: %s" % name)
        n = int(manifest["windows"])
        gradient_dir = PROJECT_ROOT / config["gradient_source"]
        gradient_file = gradient_dir / (name + ".bf16")
        if not (gradient_dir / (name + ".complete")).exists():
            raise ValueError("full-gradient cache incomplete: %s" % name)
        if gradient_file.stat().st_size != n * 8_394_496 * 2:
            raise ValueError("full-gradient cache size mismatch: %s" % name)
        with np.load(track_dir / "gram_matrices.npz") as payload:
            full = np.asarray(payload["full"], dtype=np.float64)
            head = np.asarray(payload["head"], dtype=np.float64)
        if full.shape != (n, n) or head.shape != (n, n):
            raise ValueError("Gram shape mismatch: %s" % name)
        if not np.isfinite(full).all() or not np.isfinite(head).all():
            raise ValueError("nonfinite Gram: %s" % name)
        lengths = [min(config["block_length"], n - lo)
                   for lo in range(0, n, config["block_length"])]
        bg = math.ceil(n / config["budget_denominator"])
        bp = math.ceil(n / config["proxy_denominator"])
        proxy_counts = allocate(bp, lengths, 2)
        proxy_indices = []
        lo = 0
        for length, count in zip(lengths, proxy_counts):
            proxy_indices.append(spaced(lo, length, count))
            lo += length
        observed = np.asarray([vector_variance(head, ids) for ids in proxy_indices])
        prior = float(np.average(observed, weights=lengths))
        threshold = int(config["proxy_shrink_threshold"])
        estimates = []
        for count, value in zip(proxy_counts, observed):
            weight = min(1.0, (count - 1) / max(1, threshold - 1))
            estimates.append(float(weight * value + (1 - weight) * prior))
        vage, anchor_blocks, decisions = select_static(lengths, bg, estimates, config["n_min"])
        uniform = spaced(0, n, bg)
        if len(vage) != bg or len(uniform) != bg or len(set(vage)) != bg:
            raise ValueError("anchor budget mismatch")
        for method, anchors in (("vage_static", vage), ("uniform", uniform)):
            rows.append(dict(track=name, windows=n, gradient_budget=bg, proxy_budget=bp,
                             method=method, **metrics(full, anchors)))
            selections.append(dict(track=name, method=method, anchors=anchors))
        lo = 0
        for i, (length, count, raw, estimate, anchors) in enumerate(zip(
                lengths, proxy_counts, observed, estimates, anchor_blocks)):
            blocks_out.append(dict(track=name, block=i, start=lo, length=length,
                                   proxy_count=count, proxy_variance=float(raw),
                                   shrunk_variance=estimate, gradient_count=len(anchors)))
            lo += length
        logger.info("Track complete: %s windows=%d blocks=%d Bg=%d Bp=%d VAGE_L2=%.6f Uniform_L2=%.6f",
                    name, n, len(lengths), bg, bp, rows[-2]["mean_l2"], rows[-1]["mean_l2"])
    write_csv(out / "trajectory_metrics.csv", rows)
    write_csv(out / "block_allocation.csv", blocks_out)
    (out / "anchors.json").write_text(json.dumps(selections, indent=2) + "\n", encoding="ascii")
    keys = ("mean_l2", "mean_cosine", "relative_rmse", "p95_l2", "p05_cosine",
            "nonanchor_mean_l2", "nonanchor_mean_cosine")
    summary = {"trajectory_count": 10, "windows": sum(r["windows"] for r in rows if r["method"] == "vage_static"),
               "gradient_calls": sum(r["gradient_budget"] for r in rows if r["method"] == "vage_static"),
               "proxy_calls": sum(r["proxy_budget"] for r in rows if r["method"] == "vage_static")}
    for key in keys:
        paired = [next(r[key] for r in rows if r["track"] == "trajectory-%02d" % t and r["method"] == "vage_static") -
                  next(r[key] for r in rows if r["track"] == "trajectory-%02d" % t and r["method"] == "uniform")
                  for t in range(10)]
        summary[key] = dict(vage_equal_track_mean=float(np.mean([r[key] for r in rows if r["method"] == "vage_static"])),
                            uniform_equal_track_mean=float(np.mean([r[key] for r in rows if r["method"] == "uniform"])),
                            paired_differences=paired, paired_mean=float(np.mean(paired)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("Stage complete: static replay; duration=%s output=%s", format_duration(time.monotonic()-started), out)


if __name__ == "__main__":
    main()
