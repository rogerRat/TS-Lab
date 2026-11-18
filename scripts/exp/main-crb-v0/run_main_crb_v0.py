"""Run the main-crb cover-then-cancellation selection histograms.

Read-only reuse of the main-ads-v0 TimerXL LoRA gradient cache: ordinal 1 is the
short reference trajectory (matrix columns), ordinal 0 is the long source
trajectory (matrix rows). For every ``q`` this script builds

* ``crb-q``: coverage anchors at ``cover_spacing_ratio * q``, then a capped
  cancellation split over the strongest ``split_fraction`` of the intervals.
* ``uni-budget-matched``: uniform anchors with the exact same per-side counts.

and stores BF16 cosine matrices as uint8 bucket IDs plus 100/256-bin counts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.crb_select import CRBSplitConfig, select_crb_positions, selection_audit
from rage.ads import uniform_anchor_positions

GRAD_ROOT = ROOT / "outputs/grad_cache/main-ads-v0/traj-replay"
OUT_ROOT = ROOT / "outputs/selection/main-crb-v0/crb-histgram-bf16-u8"
REF_ORDINAL = 1
SRC_ORDINAL = 0
CHUNK_ROWS = 8192
Q_VALUES = (8, 16, 32, 64)
COVER_SPACING_RATIO = 1.5
SPLIT_FRACTION = 0.5


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="ascii")
    os.replace(temporary, path)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values),
                     where=norms[:, None] > 0)


def compute_matrix(src_cache: np.ndarray, ref_cache: np.ndarray,
                   src_positions: np.ndarray, ref_positions: np.ndarray,
                   matrix_path: Path, bins: int, logger, label: str):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 matrix multiplication is required")
    device = torch.device("cuda")
    matrix_path = output_path("selection", matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.uint8,
        shape=(len(src_positions), len(ref_positions)))
    ref_host = normalize_rows(ref_cache[ref_positions])
    ref_vectors = torch.as_tensor(ref_host, dtype=torch.bfloat16, device=device)
    edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    bucket_edges = np.linspace(-1.0, 1.0, 257, dtype=np.float64)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2.0
    bucket_counts = np.zeros(256, dtype=np.int64)
    score_count = 0
    score_sum = 0.0
    score_sumsq = 0.0
    positive_fraction_005 = 0.0
    negative_count = 0
    score_min, score_max = np.inf, -np.inf
    started = time.monotonic()
    for first in range(0, len(src_positions), CHUNK_ROWS):
        last = min(first + CHUNK_ROWS, len(src_positions))
        src_host = normalize_rows(src_cache[src_positions[first:last]])
        src_vectors = torch.as_tensor(src_host, dtype=torch.bfloat16, device=device)
        scores = torch.matmul(src_vectors, ref_vectors.T).float().clamp_(-1.0, 1.0)
        values = scores.cpu().numpy()
        bucket_ids = np.floor((values + 1.0) * 128.0).astype(np.int16)
        np.clip(bucket_ids, 0, 255, out=bucket_ids)
        bucket_ids = bucket_ids.astype(np.uint8)
        matrix[first:last] = bucket_ids
        bucket_counts += np.bincount(bucket_ids.reshape(-1), minlength=256)
        estimated = bucket_centers[bucket_ids]
        counts += np.histogram(estimated, bins=edges)[0]
        score_count += int(values.size)
        score_sum += float(np.sum(values, dtype=np.float64))
        score_sumsq += float(np.sum(values.astype(np.float64) ** 2))
        positive_fraction_005 += int(np.count_nonzero(values > 0.05))
        negative_count += int(np.count_nonzero(values < 0.0))
        score_min = min(score_min, float(values.min()))
        score_max = max(score_max, float(values.max()))
        elapsed = time.monotonic() - started
        done = last
        eta = elapsed * (len(src_positions) - done) / max(done, 1)
        logger.info("stage=matrix-progress label=%s rows=%d/%d cells=%d elapsed=%s eta=%s",
                    label, done, len(src_positions), done * len(ref_positions),
                    format_duration(elapsed), format_duration(eta))
        del src_vectors, scores, values, bucket_ids, estimated
    matrix.flush()
    del matrix
    total = int(bucket_counts.sum())
    if total != len(src_positions) * len(ref_positions) or int(counts.sum()) != total:
        raise ValueError("bucket histogram count differs from matrix size")
    relative = counts.astype(np.float64) / total
    bucket_relative = bucket_counts.astype(np.float64) / total
    result = {
        "matrix_file": "outputs/" + str(matrix_path.relative_to((ROOT / "outputs").resolve())),
        "matrix_dtype": "uint8_bucket_id",
        "shape": [int(len(src_positions)), int(len(ref_positions))],
        "source_anchor_count": int(len(src_positions)),
        "reference_anchor_count": int(len(ref_positions)),
        "pair_count": total,
        "histogram_edges": edges.tolist(),
        "histogram_counts": counts.tolist(),
        "histogram_relative_frequency": relative.tolist(),
        "histogram_frequency_sum": float(relative.sum()),
        "bucket_edges": bucket_edges.tolist(),
        "bucket_centers": bucket_centers.tolist(),
        "histogram_counts_256": bucket_counts.tolist(),
        "histogram_relative_frequency_256": bucket_relative.tolist(),
        "exact_score_statistics": {
            "count": score_count,
            "mean": score_sum / score_count,
            "std": float(max(score_sumsq / score_count
                             - (score_sum / score_count) ** 2, 0.0) ** 0.5),
            "min": score_min,
            "max": score_max,
            "fraction_negative": negative_count / score_count,
            "fraction_nonnegative": 1.0 - negative_count / score_count,
            "fraction_gt_0_05": positive_fraction_005 / score_count,
            "source_dtype": "bf16 matmul scores clamped to [-1, 1]",
        },
    }
    logger.info("stage=matrix-complete label=%s shape=%s pairs=%d output=%s",
                label, result["shape"], total, matrix_path)
    return result


def write_overlay_svg(path: Path, title: str, traces: list[tuple]) -> None:
    width, height = 1100, 680
    left, right, top, bottom = 90, 35, 70, 85
    maximum = max(max(values) for _color, _label, _edges, values in traces)
    maximum = max(maximum, 1e-9) * 1.05
    plot_width = width - left - right
    plot_height = height - top - bottom
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="680" viewBox="0 0 1100 680">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="550" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">%s</text>' % title]
    for tick in range(6):
        y = top + plot_height * tick / 5
        value = maximum * (1 - tick / 5)
        parts.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#dddddd"/>' % (left, y, width-right, y))
        parts.append('<text x="%d" y="%.2f" text-anchor="end" font-family="sans-serif" font-size="12">%.4g</text>' % (left-10, y+4, value))
    for tick in range(5):
        x = left + plot_width * tick / 4
        value = -1 + tick / 2
        parts.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#eeeeee"/>' % (x, top, x, height-bottom))
        parts.append('<text x="%.2f" y="%d" text-anchor="middle" font-family="sans-serif" font-size="12">%.1f</text>' % (x, height-bottom+22, value))
    parts.extend(['<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333333"/>' % (left, top, left, height-bottom),
                  '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333333"/>' % (left, height-bottom, width-right, height-bottom),
                  '<text x="550" y="665" text-anchor="middle" font-family="sans-serif" font-size="15">Cosine similarity</text>',
                  '<text transform="translate(22 350) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="15">Relative frequency</text>'])
    for index, (color, label, edges, values) in enumerate(traces):
        coords = []
        for bucket, value in enumerate(values):
            x0 = left + plot_width * (float(edges[bucket]) + 1) / 2
            x1 = left + plot_width * (float(edges[bucket + 1]) + 1) / 2
            y = top + plot_height * (1 - float(value) / maximum)
            if bucket == 0:
                coords.append("%.2f,%.2f" % (x0, y))
            coords.extend(("%.2f,%.2f" % (x1, y), "%.2f,%.2f" % (x1, y)))
        parts.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>' % (color, " ".join(coords)))
        lx, ly = left + (index % 4) * 230, 54 + (index // 4) * 18
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2"/>' % (lx, ly, lx+22, ly, color))
        parts.append('<text x="%d" y="%d" font-family="sans-serif" font-size="12">%s</text>' % (lx+28, ly+4, label))
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="ascii")


def write_comparison_svg(path: Path, panels: list[tuple]) -> None:
    columns = 2
    rows = (len(panels) + 1) // 2
    width, cell_h = 1400, 380
    height = rows * cell_h + 40
    cell_w = width / columns
    margin_x, margin_y = 65, 60
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' % (width, height, width, height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="%d" y="30" text-anchor="middle" font-family="sans-serif" font-size="21">Budget-matched uni vs CRB cosine relative frequency</text>' % (width // 2)]
    for panel_index, (q, counts, traces) in enumerate(panels):
        row, col = divmod(panel_index, columns)
        x0, y0 = col * cell_w + margin_x, row * cell_h + margin_y
        plot_w, plot_h = cell_w - margin_x - 35, cell_h - margin_y - 45
        maximum = max(max(values) for _color, _label, _edges, values in traces)
        maximum = max(maximum, 1e-9) * 1.05
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="16">q=%d; anchors uni=%d, CRB=%d</text>' % (x0, y0-15, q, counts["uni_budget_matched_total_anchor_count"], counts["crb_total_anchor_count"]))
        for tick in range(4):
            y = y0 + plot_h * tick / 3
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#dddddd"/>' % (x0, y, x0+plot_w, y))
        for tick in range(5):
            x = x0 + plot_w * tick / 4
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eeeeee"/>' % (x, y0, x, y0+plot_h))
        for color, label, edges, values in traces:
            coords = []
            for bucket, value in enumerate(values):
                xleft = x0 + plot_w * (float(edges[bucket]) + 1) / 2
                xright = x0 + plot_w * (float(edges[bucket+1]) + 1) / 2
                y = y0 + plot_h * (1 - float(value) / maximum)
                if bucket == 0:
                    coords.append("%.2f,%.2f" % (xleft, y))
                coords.extend(("%.2f,%.2f" % (xright, y), "%.2f,%.2f" % (xright, y)))
            parts.append('<polyline fill="none" stroke="%s" stroke-width="1.6" points="%s"/>' % (color, " ".join(coords)))
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="12">-1.0</text>' % (x0, y0+plot_h+18))
        parts.append('<text x="%.1f" y="%.1f" text-anchor="end" font-family="sans-serif" font-size="12">1.0</text>' % (x0+plot_w, y0+plot_h+18))
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="12" fill="#2878B5">uni</text>' % (x0+8, y0+18))
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="12" fill="#D9534F">CRB</text>' % (x0+55, y0+18))
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--q-values", type=int, nargs="+", default=list(Q_VALUES))
    parser.add_argument("--cover-spacing-ratio", type=float, default=COVER_SPACING_RATIO)
    parser.add_argument("--split-fraction", type=float, default=SPLIT_FRACTION)
    args = parser.parse_args()
    if args.bins != 100:
        raise ValueError("this experiment requires exactly 100 histogram bins")

    logger = configure_logger("main_crb_v0_histogram", "main-crb-v0",
                              "crb-histgram-bf16-u8", "startup", enable_heartbeat=True)
    started = time.monotonic()
    out_dir = output_path("selection", OUT_ROOT)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("analysis output already exists: " + str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ordinal in (REF_ORDINAL, SRC_ORDINAL):
        if not (GRAD_ROOT / ("trajectory-%03d/.complete" % ordinal)).is_file():
            raise FileNotFoundError("main-ads-v0 gradient cache is incomplete for ordinal=%d" % ordinal)
    ref_cache = np.load(GRAD_ROOT / ("trajectory-%03d/gradient_sketches.npy" % REF_ORDINAL),
                        mmap_mode="r")
    src_cache = np.load(GRAD_ROOT / ("trajectory-%03d/gradient_sketches.npy" % SRC_ORDINAL),
                        mmap_mode="r")
    if ref_cache.ndim != 2 or src_cache.ndim != 2 or ref_cache.shape[1] != src_cache.shape[1]:
        raise ValueError("reference and source gradient dimensions differ")
    logger.info("stage=config batch=crb-histgram-bf16-u8 q_values=%s ref_ordinal=%d ref_shape=%s src_ordinal=%d src_shape=%s cover_spacing_ratio=%.3f split_fraction=%.3f compute_dtype=bfloat16 matrix_dtype=uint8 bins=%d output=%s",
                args.q_values, REF_ORDINAL, ref_cache.shape, SRC_ORDINAL, src_cache.shape,
                args.cover_spacing_ratio, args.split_fraction, args.bins, out_dir)

    records = {"crb": {}, "uni_budget_matched": {}}
    selections = {}
    for q_index, q in enumerate(args.q_values, start=1):
        config = CRBSplitConfig(q=float(q),
                                cover_spacing_ratio=args.cover_spacing_ratio,
                                split_fraction=args.split_fraction)
        logger.info("stage=q-start q=%d/%d cover_spacing=%.3f split_spacing=%.3f",
                    q_index, len(args.q_values), config.cover_spacing, config.split_spacing)
        src_selection = select_crb_positions(src_cache, config)
        ref_selection = select_crb_positions(ref_cache, config)
        for side, selection in (("source", src_selection), ("reference", ref_selection)):
            audit = selection_audit(selection, config)
            audit["side"] = side
            audit["trajectory_ordinal"] = SRC_ORDINAL if side == "source" else REF_ORDINAL
            selections.setdefault(str(q), {})[side] = audit
            logger.info("stage=q-side-selection q=%d side=%s valid=%d cover=%d intervals=%d split_budget=%d split_used=%d anchors=%d equivalent_spacing=%.3f mean_kappa_split=%s mean_kappa_kept=%s",
                        q, side, audit["valid_count"], audit["cover_anchor_count"],
                        audit["interval_count"], audit["split_budget"], audit["split_used"],
                        audit["total_anchor_count"], audit["equivalent_spacing"],
                        "%.5f" % audit["mean_kappa_split"] if audit["mean_kappa_split"] is not None else "none",
                        "%.5f" % audit["mean_kappa_kept"] if audit["mean_kappa_kept"] is not None else "none")

        crb_dir = out_dir / "crb-q"
        crb_dir.mkdir(parents=True, exist_ok=True)
        np.save(crb_dir / ("source_anchor_positions_q%d.npy" % q), src_selection.anchor_positions)
        np.save(crb_dir / ("source_cover_positions_q%d.npy" % q), src_selection.cover_positions)
        np.save(crb_dir / ("source_split_positions_q%d.npy" % q), src_selection.split_positions)
        np.save(crb_dir / ("reference_anchor_positions_q%d.npy" % q), ref_selection.anchor_positions)
        np.save(crb_dir / ("reference_cover_positions_q%d.npy" % q), ref_selection.cover_positions)
        np.save(crb_dir / ("reference_split_positions_q%d.npy" % q), ref_selection.split_positions)
        crb_matrix = compute_matrix(
            src_cache, ref_cache, src_selection.anchor_positions,
            ref_selection.anchor_positions,
            out_dir / ("crb-q/cosine_matrix_q%d.npy" % q), args.bins, logger,
            "crb-q%d" % q)
        crb_matrix.update({
            "q": q,
            "cover_spacing": float(config.cover_spacing),
            "split_spacing_nominal": float(config.split_spacing),
            "cover_anchor_count_source": src_selection.cover_anchor_count,
            "cover_anchor_count_reference": ref_selection.cover_anchor_count,
            "split_anchor_count_source": int(len(src_selection.split_positions)),
            "split_anchor_count_reference": int(len(ref_selection.split_positions)),
            "source_anchor_positions_file": "crb-q/source_anchor_positions_q%d.npy" % q,
            "reference_anchor_positions_file": "crb-q/reference_anchor_positions_q%d.npy" % q,
            "source_cover_positions_file": "crb-q/source_cover_positions_q%d.npy" % q,
            "source_split_positions_file": "crb-q/source_split_positions_q%d.npy" % q,
            "reference_cover_positions_file": "crb-q/reference_cover_positions_q%d.npy" % q,
            "reference_split_positions_file": "crb-q/reference_split_positions_q%d.npy" % q,
            "total_anchor_count": int(src_selection.total_anchor_count + ref_selection.total_anchor_count),
        })
        records["crb"][str(q)] = crb_matrix

        matched_src = np.rint(np.linspace(0, len(src_cache) - 1,
                                          src_selection.total_anchor_count)).astype(np.int64)
        matched_ref = np.rint(np.linspace(0, len(ref_cache) - 1,
                                          ref_selection.total_anchor_count)).astype(np.int64)
        if len(np.unique(matched_src)) != len(matched_src) or len(np.unique(matched_ref)) != len(matched_ref):
            raise ValueError("budget-matched uni positions are not unique")
        match_dir = out_dir / "uni-budget-matched"
        match_dir.mkdir(parents=True, exist_ok=True)
        np.save(match_dir / ("source_anchor_positions_q%d.npy" % q), matched_src)
        np.save(match_dir / ("reference_anchor_positions_q%d.npy" % q), matched_ref)
        uni_matrix = compute_matrix(
            src_cache, ref_cache, matched_src, matched_ref,
            match_dir / ("cosine_matrix_q%d.npy" % q), args.bins, logger,
            "uni-budget-matched-q%d" % q)
        uni_matrix.update({
            "q": q,
            "source_anchor_positions_file": "uni-budget-matched/source_anchor_positions_q%d.npy" % q,
            "reference_anchor_positions_file": "uni-budget-matched/reference_anchor_positions_q%d.npy" % q,
            "total_anchor_count": int(len(matched_src) + len(matched_ref)),
            "matching_rule": "exact source and reference counts from CRB anchors; uniform over endpoints",
        })
        records["uni_budget_matched"][str(q)] = uni_matrix
        logger.info("stage=q-complete q=%d crb_pairs=%d uni_matched_pairs=%d crb_total_anchors=%d uni_total_anchors=%d",
                    q, crb_matrix["pair_count"], uni_matrix["pair_count"],
                    crb_matrix["total_anchor_count"], uni_matrix["total_anchor_count"])

    comparisons = {str(q): {
        "q": q,
        "crb_total_anchor_count": records["crb"][str(q)]["total_anchor_count"],
        "uni_budget_matched_total_anchor_count": records["uni_budget_matched"][str(q)]["total_anchor_count"],
        "crb_pair_count": records["crb"][str(q)]["pair_count"],
        "uni_budget_matched_pair_count": records["uni_budget_matched"][str(q)]["pair_count"],
        "crb_source_equivalent_spacing": selections[str(q)]["source"]["equivalent_spacing"],
        "crb_reference_equivalent_spacing": selections[str(q)]["reference"]["equivalent_spacing"],
    } for q in args.q_values}
    payload = {
        "schema": "main-crb-v0-crb-histgram-bf16-u8-v1",
        "experiment": "main-crb-v0",
        "batch": "crb-histgram-bf16-u8",
        "inherits": {"experiment": "main-ads-v0", "batch": "traj-replay",
                     "gradient_cache_readonly": "outputs/grad_cache/main-ads-v0/traj-replay",
                     "train_outputs_readonly": "outputs/train/main-ads-v0/checkpoint"},
        "reference": {"trajectory_ordinal": REF_ORDINAL,
                      "cache": "outputs/grad_cache/main-ads-v0/traj-replay/trajectory-%03d/gradient_sketches.npy" % REF_ORDINAL,
                      "row_semantics": "shorter trajectory; reference anchors are matrix columns"},
        "source": {"trajectory_ordinal": SRC_ORDINAL,
                   "cache": "outputs/grad_cache/main-ads-v0/traj-replay/trajectory-%03d/gradient_sketches.npy" % SRC_ORDINAL,
                   "row_semantics": "longer trajectory; source anchors are matrix rows"},
        "coverage_anchor_rule": "rage.ads.uniform_anchor_positions with cover_spacing = cover_spacing_ratio * q; both endpoints included",
        "cancellation_rule": "kappa = ||g_left + g_right|| / (||g_left|| + ||g_right||); zero denominator kappa=1; lower kappa ranks higher; ties prefer wider then leftmost interval",
        "split_rule": "one midpoint anchor per selected interval; budget = floor(split_fraction * interval_count); no unsplittable interval is selected",
        "budget_semantics": {"q": "final equivalent anchor spacing", "cover_spacing_ratio": args.cover_spacing_ratio,
                             "split_fraction": args.split_fraction,
                             "expected_total_spacing": "q"},
        "q_values": list(args.q_values),
        "histogram": {"bins": args.bins, "range": [-1.0, 1.0],
                      "y_axis": "relative frequency = bin count / total matrix pairs",
                      "stored_matrix": "uint8 bucket IDs over 256 equal-width cosine intervals",
                      "aggregate_100_bin_values": "bucket centers mapped into 100 bins"},
        "results": records,
        "selections": selections,
        "anchor_count_comparison": comparisons,
        "comparison_rule": "Compare each CRB histogram with a uni histogram using the same exact source and reference anchor counts.",
    }
    json_path = out_dir / "cosine_histograms.json"
    write_json(json_path, payload)

    colors = ["#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39"]
    for method, key in (("crb", "crb"), ("uni-budget-matched", "uni_budget_matched")):
        traces = []
        for color, q in zip(colors, args.q_values):
            hist = records[key][str(q)]
            traces.append((color, "q=%d" % q, hist["histogram_edges"],
                           hist["histogram_relative_frequency"]))
        write_overlay_svg(out_dir / method / "cosine_histogram.svg",
                          method.upper() + " cosine relative-frequency histogram", traces)

    panels = []
    for q in args.q_values:
        uni = records["uni_budget_matched"][str(q)]
        crb = records["crb"][str(q)]
        panels.append((q, comparisons[str(q)], [
            ("#2878B5", "uni budget-matched", uni["histogram_edges"], uni["histogram_relative_frequency"]),
            ("#D9534F", "CRB", crb["histogram_edges"], crb["histogram_relative_frequency"]),
        ]))
    write_comparison_svg(out_dir / "uni-vs-crb" / "cosine_histograms_by_q.svg", panels)
    (out_dir / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete elapsed=%s output=%s json=%s",
                format_duration(time.monotonic() - started), out_dir, json_path)


if __name__ == "__main__":
    main()
