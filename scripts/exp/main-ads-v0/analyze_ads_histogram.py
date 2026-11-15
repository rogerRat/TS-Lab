"""Legacy FP32 analysis using cos>0.1; ADS results are invalid for the ratio rule.

The uni_q statistics remain valid. Use analyze_ads_histogram_bf16.py for the
current full-matrix top-quartile ADS method.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.ads import uniform_anchor_positions

GRAD_ROOT = ROOT / "outputs/grad_cache/main-ads-v0/traj-replay"
OUT_ROOT = ROOT / "outputs/selection/main-ads-v0/ads-histgram"
Q_VALUES = (1, 2, 4, 8, 16, 31, 64)
REF_ORDINAL = 1
SRC_ORDINAL = 0
THRESHOLD = 0.1
CHUNK_ROWS = 512


def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, cuts)
    return [(int(group[0]), int(group[-1])) for group in groups]


def refined_positions(positions: np.ndarray, hit_mask: np.ndarray, spacing: int) -> np.ndarray:
    selected = set(int(value) for value in positions)
    index_hits = np.flatnonzero(hit_mask)
    for first, last in contiguous_runs(index_hits):
        left, right = int(positions[first]), int(positions[last])
        selected.update(left + value for value in uniform_anchor_positions(right - left + 1, spacing))
    return np.asarray(sorted(selected), dtype=np.int64)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values),
                     where=norms[:, None] > 0)


def compute_matrix(src_cache: np.ndarray, ref_cache: np.ndarray,
                   src_positions: np.ndarray, ref_positions: np.ndarray,
                   matrix_path: Path, bins: int, logger, label: str):
    matrix_path = output_path("selection", matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32,
        shape=(len(src_positions), len(ref_positions)))
    ref_vectors = normalize_rows(ref_cache[ref_positions])
    ref_hit = np.zeros(len(ref_positions), dtype=bool)
    src_hit = np.zeros(len(src_positions), dtype=bool)
    edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    matched_pairs = 0
    started = time.monotonic()
    for first in range(0, len(src_positions), CHUNK_ROWS):
        last = min(first + CHUNK_ROWS, len(src_positions))
        src_vectors = normalize_rows(src_cache[src_positions[first:last]])
        block = src_vectors @ ref_vectors.T
        np.clip(block, -1.0, 1.0, out=block)
        matrix[first:last] = block
        src_hit[first:last] = np.any(block > THRESHOLD, axis=1)
        ref_hit |= np.any(block > THRESHOLD, axis=0)
        counts += np.histogram(block, bins=edges)[0]
        matched_pairs += int(np.count_nonzero(block > THRESHOLD))
        elapsed = time.monotonic() - started
        done = last
        eta = elapsed * (len(src_positions) - done) / max(done, 1)
        logger.info("stage=matrix-progress label=%s rows=%d/%d cells=%d elapsed=%s eta=%s",
                    label, done, len(src_positions), done * len(ref_positions),
                    format_duration(elapsed), format_duration(eta))
    matrix.flush()
    del matrix
    total = int(counts.sum())
    if total != len(src_positions) * len(ref_positions):
        raise ValueError("histogram frequency total differs from matrix size")
    relative = counts.astype(np.float64) / total
    result = {
        "matrix_file": "outputs/" + str(matrix_path.relative_to((ROOT / "outputs").resolve())),
        "shape": [int(len(src_positions)), int(len(ref_positions))],
        "source_anchor_count": int(len(src_positions)),
        "reference_anchor_count": int(len(ref_positions)),
        "pair_count": total,
        "cosine_threshold": THRESHOLD,
        "matched_source_anchor_count": int(src_hit.sum()),
        "matched_reference_anchor_count": int(ref_hit.sum()),
        "matched_pair_count": matched_pairs,
        "histogram_edges": edges.tolist(),
        "histogram_counts": counts.tolist(),
        "histogram_relative_frequency": relative.tolist(),
        "histogram_frequency_sum": float(relative.sum()),
    }
    logger.info("stage=matrix-complete label=%s shape=%s pairs=%d matched_pairs=%d output=%s",
                label, result["shape"], total, result["matched_pair_count"], matrix_path)
    return result, src_hit, ref_hit



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
    width, height = 1400, 1500
    columns, rows = 2, 4
    cell_w, cell_h = width / columns, height / rows
    margin_x, margin_y = 65, 60
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1500" viewBox="0 0 1400 1500">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="700" y="30" text-anchor="middle" font-family="sans-serif" font-size="21">Budget-matched uni vs ADS-VIC cosine relative frequency</text>']
    for panel_index, (q, budget, traces) in enumerate(panels):
        row, col = divmod(panel_index, columns)
        x0, y0 = col * cell_w + margin_x, row * cell_h + margin_y
        plot_w, plot_h = cell_w - margin_x - 35, cell_h - margin_y - 45
        maximum = max(max(values) for _color, _label, _edges, values in traces)
        maximum = max(maximum, 1e-9) * 1.05
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="16">q=%d; anchors uni=%d, ADS-VIC=%d</text>' % (x0, y0-15, q, budget["uni_budget_matched_total_anchor_count"], budget["ads_vic_total_anchor_count"]))
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
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="12" fill="#D9534F">ADS-VIC</text>' % (x0+55, y0+18))
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="ascii")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.bins != 100:
        raise ValueError("this experiment requires exactly 100 histogram bins")

    logger = configure_logger("main_ads_v0_histogram", "main-ads-v0",
                              "ads-histgram", "startup", enable_heartbeat=True)
    started = time.monotonic()
    out_dir = output_path("selection", OUT_ROOT)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("analysis output already exists: " + str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = GRAD_ROOT / ("trajectory-%03d/gradient_sketches.npy" % REF_ORDINAL)
    src_path = GRAD_ROOT / ("trajectory-%03d/gradient_sketches.npy" % SRC_ORDINAL)
    for ordinal in (REF_ORDINAL, SRC_ORDINAL):
        if not (GRAD_ROOT / ("trajectory-%03d/.complete" % ordinal)).is_file():
            raise FileNotFoundError("gradient cache is incomplete for ordinal=%d" % ordinal)
    ref_cache = np.load(ref_path, mmap_mode="r")
    src_cache = np.load(src_path, mmap_mode="r")
    if ref_cache.ndim != 2 or src_cache.ndim != 2 or ref_cache.shape[1] != src_cache.shape[1]:
        raise ValueError("reference and source gradient dimensions differ")
    logger.info("stage=config batch=ads-histgram q_values=%s ref_ordinal=%d ref_shape=%s src_ordinal=%d src_shape=%s bins=%d threshold=%.3f output=%s",
                Q_VALUES, REF_ORDINAL, ref_cache.shape, SRC_ORDINAL, src_cache.shape,
                args.bins, THRESHOLD, out_dir)
    records = {"uni_q": {}, "ads_vic": {}, "uni_budget_matched": {}}
    for q_index, q in enumerate(Q_VALUES, start=1):
        logger.info("stage=q-start q=%d/%d spacing=%d", q_index, len(Q_VALUES), q)
        src_coarse = np.asarray(uniform_anchor_positions(len(src_cache), q), dtype=np.int64)
        ref_coarse = np.asarray(uniform_anchor_positions(len(ref_cache), q), dtype=np.int64)
        uni_q, src_hits, ref_hits = compute_matrix(
            src_cache, ref_cache, src_coarse, ref_coarse,
            out_dir / ("uni-q/cosine_matrix_q%d.npy" % q), args.bins, logger,
            "baseline-q%d" % q)
        uni_q["q"] = q
        uni_q_dir = out_dir / "uni-q"
        uni_q_dir.mkdir(parents=True, exist_ok=True)
        np.save(uni_q_dir / ("source_anchor_positions_q%d.npy" % q), src_coarse)
        np.save(uni_q_dir / ("reference_anchor_positions_q%d.npy" % q), ref_coarse)
        uni_q["source_anchor_positions_file"] = "uni-q/source_anchor_positions_q%d.npy" % q
        uni_q["reference_anchor_positions_file"] = "uni-q/reference_anchor_positions_q%d.npy" % q
        records["uni_q"][str(q)] = uni_q

        coarse = uni_q
        refine_q = max(1, q // 4)
        src_final = refined_positions(src_coarse, src_hits, refine_q)
        ref_final = refined_positions(ref_coarse, ref_hits, refine_q)
        final, _final_src_hits, _final_ref_hits = compute_matrix(
            src_cache, ref_cache, src_final, ref_final,
            out_dir / ("ads-vic/final_cosine_matrix_q%d_refine%d.npy" % (q, refine_q)),
            args.bins, logger, "ads-vic-final-q%d-refine%d" % (q, refine_q))
        vic_dir = out_dir / "ads-vic"
        vic_dir.mkdir(parents=True, exist_ok=True)
        np.save(vic_dir / ("source_anchor_positions_q%d.npy" % q), src_final)
        np.save(vic_dir / ("reference_anchor_positions_q%d.npy" % q), ref_final)
        records["ads_vic"][str(q)] = {
            "q": q,
            "refinement_spacing": refine_q,
            "coarse": coarse,
            "final": final,
            "source_anchor_positions_file": "ads-vic/source_anchor_positions_q%d.npy" % q,
            "reference_anchor_positions_file": "ads-vic/reference_anchor_positions_q%d.npy" % q,
            "total_anchor_count": int(len(src_final) + len(ref_final)),
        }
        matched_src = np.rint(np.linspace(0, len(src_cache) - 1, len(src_final))).astype(np.int64)
        matched_ref = np.rint(np.linspace(0, len(ref_cache) - 1, len(ref_final))).astype(np.int64)
        if len(np.unique(matched_src)) != len(matched_src) or len(np.unique(matched_ref)) != len(matched_ref):
            raise ValueError("budget-matched uni positions are not unique")
        match_dir = out_dir / "uni-budget-matched"
        match_dir.mkdir(parents=True, exist_ok=True)
        np.save(match_dir / ("source_anchor_positions_q%d.npy" % q), matched_src)
        np.save(match_dir / ("reference_anchor_positions_q%d.npy" % q), matched_ref)
        uni_match, _, _ = compute_matrix(
            src_cache, ref_cache, matched_src, matched_ref,
            match_dir / ("cosine_matrix_q%d.npy" % q), args.bins, logger,
            "uni-budget-matched-q%d" % q)
        uni_match.update({
            "q": q,
            "source_anchor_positions_file": "uni-budget-matched/source_anchor_positions_q%d.npy" % q,
            "reference_anchor_positions_file": "uni-budget-matched/reference_anchor_positions_q%d.npy" % q,
            "total_anchor_count": int(len(matched_src) + len(matched_ref)),
            "matching_rule": "exact source and reference counts from ADS-VIC final anchors; uniform over endpoints",
        })
        records["uni_budget_matched"][str(q)] = uni_match
        logger.info("stage=q-complete q=%d refine_q=%d uni_q_pairs=%d ads_vic_coarse_pairs=%d ads_vic_final_pairs=%d matched_budget=%d",
                    q, refine_q, uni_q["pair_count"], coarse["pair_count"],
                    final["pair_count"], uni_match["total_anchor_count"])

    comparisons = {str(q): {
        "q": q,
        "uni_q_total_anchor_count": int(records["uni_q"][str(q)]["source_anchor_count"] + records["uni_q"][str(q)]["reference_anchor_count"]),
        "uni_budget_matched_total_anchor_count": records["uni_budget_matched"][str(q)]["total_anchor_count"],
        "ads_vic_total_anchor_count": records["ads_vic"][str(q)]["total_anchor_count"],
    } for q in Q_VALUES}
    payload = {
        "schema": "main-ads-v0-ads-histgram-v1",
        "experiment": "main-ads-v0",
        "batch": "ads-histgram",
        "reference": {"trajectory_ordinal": REF_ORDINAL, "cache": "outputs/grad_cache/main-ads-v0/traj-replay/trajectory-%03d/gradient_sketches.npy" % REF_ORDINAL,
                      "row_semantics": "shorter trajectory; reference anchors are matrix columns"},
        "source": {"trajectory_ordinal": SRC_ORDINAL, "cache": "outputs/grad_cache/main-ads-v0/traj-replay/trajectory-%03d/gradient_sketches.npy" % SRC_ORDINAL,
                   "row_semantics": "longer trajectory; source anchors are matrix rows"},
        "uniform_anchor_rule": "rage.ads.uniform_anchor_positions; zero-based, evenly distributed, both endpoints included",
        "q_values": list(Q_VALUES),
        "ads_vic": {"threshold": THRESHOLD,
                    "refinement_rule": "refine all contiguous runs of coarse anchors with any cosine > threshold; refinement spacing=max(1, floor(q/4)); preserve coarse anchors"},
        "histogram": {"bins": args.bins, "range": [-1.0, 1.0],
                      "y_axis": "relative frequency = bin count / total matrix pairs"},
        "results": records,
        "anchor_count_comparison": comparisons,
        "comparison_rule": "Compare each ADS-VIC final histogram with a uni histogram using the same exact source and reference anchor counts.",
    }
    json_path = out_dir / "cosine_histograms.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")

    colors = ["#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39"]
    overlay_traces = {}
    for method in ("uni_q", "ads_vic"):
        traces = []
        for color, q in zip(colors, Q_VALUES):
            item = records[method][str(q)]
            hist = item if method == "uni_q" else item["final"]
            traces.append((color, "q=%d" % q, hist["histogram_edges"],
                           hist["histogram_relative_frequency"]))
        overlay_traces[method] = traces
        write_overlay_svg(out_dir / method.replace("_", "-") / "cosine_histogram.svg",
                          method.replace("_", "-").upper() + " cosine relative-frequency histogram",
                          traces)

    panels = []
    for q in Q_VALUES:
        uni = records["uni_budget_matched"][str(q)]
        vic = records["ads_vic"][str(q)]["final"]
        panels.append((q, comparisons[str(q)], [
            ("#2878B5", "uni budget-matched", uni["histogram_edges"], uni["histogram_relative_frequency"]),
            ("#D9534F", "ADS-VIC", vic["histogram_edges"], vic["histogram_relative_frequency"]),
        ]))
    write_comparison_svg(out_dir / "uni-vs-ads-vic" / "cosine_histograms_by_q.svg", panels)
    (out_dir / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete elapsed=%s output=%s json=%s",
                format_duration(time.monotonic() - started), out_dir, json_path)


if __name__ == "__main__":
    main()
