"""Mechanism experiment 2: does cancellation detection save useful directions?

Uses the true-gradient scans cached by mechanism experiment 1.  Anchors and
minimum intervals come from one shared CRB run (first two stages); interval
classes are determined by the *true* gradients inside each interval:

  weak      within-interval gradients are uniformly weak;
  stable    strong and a single direction represents them well;
  cancel    strong but the average cancels (mean norm far below the mean of
            the member norms).

Three treatments are compared on the same intervals:

  average    direct endpoint-average representative, recall as-is;
  drop       detection flags the interval and discards it;
  split      detection flags the interval, one midpoint split + recheck.

Measurements (§4.2): detection precision/recall against true cancellation,
representative-vs-member direction deviation before/after the split,
high-score windows lost by dropping vs recovered by splitting, drop ratios,
actual split counts, the endpoint-mean vs true-arithmetic-mean direction
gap, and the cancellation-degree ratio.  Also exports one real cancelled
interval for the paper case figure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .mechanism1 import (
    DEFAULT_SHARES,
    CountingCache,
    SideScan,
    crb_result_for,
    load_scan,
)


def interval_classes(true_grads: np.ndarray, left: int, right: int, *,
                     weak_epsilon: float, cancel_ratio: float) -> tuple[str, dict]:
    """Classify one interval from the true gradients it covers."""
    members = true_grads[left:right]
    member_norms = np.linalg.norm(members, axis=1)
    avg_norm = float(member_norms.mean())
    mean_norm = float(np.linalg.norm(members.mean(axis=0)))
    ratio = mean_norm / max(avg_norm, 1e-12)
    if avg_norm < weak_epsilon:
        label = "weak"
    elif ratio < cancel_ratio:
        label = "cancel"
    else:
        label = "stable"
    return label, {
        "avg_norm": avg_norm, "mean_norm": mean_norm, "ratio": ratio,
    }


def detected_flag(anchor_left: np.ndarray, anchor_right: np.ndarray, *,
                  mean_epsilon: float, endpoint_epsilon: float) -> bool:
    """The §3.3 antagonistic test: near-zero mean, strong endpoints."""
    mean_norm = float(np.linalg.norm((anchor_left + anchor_right) * 0.5))
    if mean_norm >= mean_epsilon:
        return False
    left_weak = float(np.linalg.norm(anchor_left)) < endpoint_epsilon
    right_weak = float(np.linalg.norm(anchor_right)) < endpoint_epsilon
    return not (left_weak and right_weak)


def direction_deviation(representative: np.ndarray, members: np.ndarray) -> float:
    """Mean 1 - cos(representative, member) over the interval's true windows."""
    rep_norm = np.linalg.norm(representative)
    member_norms = np.linalg.norm(members, axis=1)
    valid = member_norms > 1e-9
    if rep_norm <= 1e-9 or not valid.any():
        return float("nan")
    cosines = (members[valid] @ representative) / (member_norms[valid] * rep_norm)
    return float((1.0 - cosines).mean())


def run_mechanism2(*, reference_scan: SideScan, source_scan: SideScan,
                   density: float = 0.05, cover_share: float = DEFAULT_SHARES[0],
                   error_share: float = DEFAULT_SHARES[1],
                   split_share: float = DEFAULT_SHARES[2],
                   mean_epsilon: float = 1e-8, endpoint_epsilon: float = 1e-8,
                   weak_epsilon: float = 1e-8, cancel_ratio: float = 0.5,
                   curvature_cap: float = 1.0, context_length: int = 512,
                   prediction_length: int = 64, stride: int = 1,
                   seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    report: dict = {"protocol": {
        "density": density, "mean_epsilon": mean_epsilon,
        "endpoint_epsilon": endpoint_epsilon, "weak_epsilon": weak_epsilon,
        "cancel_ratio": cancel_ratio,
    }, "intervals": [], "case_export": None}

    # true full ranking for high-score window sets
    src_true = np.concatenate(source_scan.gradients, axis=0)
    src_unit = src_true / np.maximum(np.linalg.norm(src_true, axis=1, keepdims=True), 1e-8)
    ref_unit = np.concatenate(reference_scan.gradients, axis=0)
    ref_unit = ref_unit / np.maximum(np.linalg.norm(ref_unit, axis=1, keepdims=True), 1e-8)
    mean_scores = (ref_unit @ src_unit.T).mean(axis=0)
    high_score_windows = {
        m: set(np.argsort(-mean_scores)[:m].tolist()) for m in (50, 100, 500)
    }

    for side_name, scan in (("reference", reference_scan), ("source", source_scan)):
        cache = CountingCache(scan)
        results, _reconstructed = crb_result_for(
            scan, cache, total_density=density, cover_share=cover_share,
            error_share=error_share, split_share=split_share,
            context_length=context_length, prediction_length=prediction_length,
            stride=stride, mean_epsilon=mean_epsilon,
            endpoint_epsilon=endpoint_epsilon, curvature_cap=curvature_cap)
        split_budget_total = sum(
            max(0, int(round(density * split_share * len(r.valid_starts))))
            for r in results)

        side_records = []
        split_used = 0
        for index, result in enumerate(results):
            true_grads = scan.gradients[index]
            total = len(true_grads)
            anchors = np.asarray(sorted(result.anchors), dtype=np.int64)
            for i in range(len(anchors) - 1):
                left, right = int(anchors[i]), int(anchors[i + 1])
                if right - left < 2:
                    continue
                label, stats = interval_classes(
                    true_grads, left, right,
                    weak_epsilon=weak_epsilon, cancel_ratio=cancel_ratio)
                anchor_left = result.anchors[left]
                anchor_right = result.anchors[right]
                flagged = detected_flag(
                    anchor_left, anchor_right,
                    mean_epsilon=mean_epsilon, endpoint_epsilon=endpoint_epsilon)
                members = true_grads[left:right]
                endpoint_mean = (anchor_left + anchor_right) * 0.5
                true_mean = members.mean(axis=0)
                record = {
                    "fragment": index, "left": left, "right": right,
                    "class": label, "flagged": flagged,
                    **{k: round(v, 6) for k, v in stats.items()},
                    "deviation_endpoint_mean": round(direction_deviation(
                        endpoint_mean, members), 6),
                    "deviation_true_mean": round(direction_deviation(
                        true_mean, members), 6),
                }
                # treatment comparison for flagged intervals
                if flagged and right - left >= 4 and split_used < split_budget_total:
                    mid = (left + right) // 2
                    child_mid_grad = true_grads[mid]
                    split_used += 1
                    children = {"left": (left, mid), "right": (mid, right)}
                    for name, (cl, cr) in children.items():
                        child_members = true_grads[cl:cr]
                        child_rep = (result.anchors[cl] + child_mid_grad) * 0.5 \
                            if cr == mid else (child_mid_grad + result.anchors[cr]) * 0.5
                        record["deviation_split_%s" % name] = round(
                            direction_deviation(child_rep, child_members), 6)
                        record["mean_norm_split_%s" % name] = round(float(
                            np.linalg.norm(child_rep)), 6)
                    record["split"] = True
                else:
                    record["split"] = False
                side_records.append(record)
                if (report["case_export"] is None and label == "cancel"
                        and flagged and right - left >= 8):
                    report["case_export"] = {
                        "side": side_name, "fragment": index,
                        "left": left, "right": right, "mid": (left + right) // 2,
                        "data": scan.fragments[index].data.tolist()
                        if scan.fragments[index].length <= 8192 else None,
                        "member_directions": (members / np.maximum(
                            np.linalg.norm(members, axis=1, keepdims=True), 1e-9)
                        ).astype(np.float16).tolist(),
                        "anchors": [left, int((left + right) // 2), right],
                    }
        report[side_name] = side_records

    # aggregate detection quality and treatment effects
    summary: dict = {}
    for side_name in ("reference", "source"):
        records = report[side_name]
        true_cancel = [r for r in records if r["class"] == "cancel"]
        flagged = [r for r in records if r["flagged"]]
        both = [r for r in records if r["class"] == "cancel" and r["flagged"]]
        deviations = [r["deviation_endpoint_mean"] for r in records
                      if not np.isnan(r["deviation_endpoint_mean"])]
        split_recovered = {m: 0 for m in high_score_windows}
        dropped_lost = {m: 0 for m in high_score_windows}
        # window-level effect: windows in flagged intervals
        for record in records:
            if not record["flagged"]:
                continue
            fragment_index = record["fragment"]
            offset = sum(len(g) for g in
                         (reference_scan if side_name == "reference" else source_scan).gradients[:fragment_index])
            for m, high in high_score_windows.items():
                hits = sum(1 for row in range(record["left"], record["right"])
                           if offset + row in high)
                if record["split"]:
                    split_recovered[m] += hits
                else:
                    dropped_lost[m] += hits
        summary[side_name] = {
            "intervals": len(records),
            "class_counts": {
                label: sum(1 for r in records if r["class"] == label)
                for label in ("weak", "stable", "cancel")},
            "flagged": len(flagged),
            "detection_precision": len(both) / max(len(flagged), 1),
            "detection_recall": len(both) / max(len(true_cancel), 1),
            "mean_endpoint_deviation": float(np.mean(deviations)) if deviations else None,
            "split_high_score_windows": {m: int(v) for m, v in split_recovered.items()},
            "drop_high_score_windows": {m: int(v) for m, v in dropped_lost.items()},
        }
    report["summary"] = summary
    return report


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", default="outputs/mechanism1")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--mean-epsilon", type=float, default=1e-8,
                        help="epsilon_mu (C3 calibrated value, frozen)")
    parser.add_argument("--endpoint-epsilon", type=float, default=1e-8,
                        help="epsilon_g (same calibrated value)")
    parser.add_argument("--weak-epsilon", type=float, default=1e-8,
                        help="ground-truth weak-interval threshold (= epsilon_g)")
    parser.add_argument("--cancel-ratio", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/mechanism2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    scan_dir = Path(args.scan_dir)
    reference_scan = load_scan(scan_dir / "scan_reference.npz",
                               context_length=args.context_length,
                               prediction_length=args.prediction_length,
                               stride=args.stride)
    source_scan = load_scan(scan_dir / "scan_source.npz",
                            context_length=args.context_length,
                            prediction_length=args.prediction_length,
                            stride=args.stride)
    started = time.time()
    report = run_mechanism2(
        reference_scan=reference_scan, source_scan=source_scan,
        density=args.density, cancel_ratio=args.cancel_ratio,
        mean_epsilon=args.mean_epsilon, endpoint_epsilon=args.endpoint_epsilon,
        weak_epsilon=args.weak_epsilon,
        context_length=args.context_length,
        prediction_length=args.prediction_length, stride=args.stride)
    report["elapsed_minutes"] = round((time.time() - started) / 60, 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "mechanism2.json"
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report.get("summary", {}), indent=2), flush=True)
    print("MECHANISM2 DONE -> %s" % destination, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
