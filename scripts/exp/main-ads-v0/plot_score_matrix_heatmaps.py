"""Plot time-ordered score matrices with per-anchor score-sum profiles."""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from configs.logger import configure_logger, format_duration
from configs.paths import output_path

INPUT_ROOT = ROOT / "outputs/selection/main-ads-v0/ads-histgram-bf16-u8"
OUTPUT_ROOT = INPUT_ROOT / "score-matrix-heatmaps"
CHUNK_ROWS = 8192


def load_positions(root: Path, method: str, q: int):
    if method == "uni-q":
        folder = root / "uni-q"
    elif method == "ads-vic":
        folder = root / "ads-vic"
    elif method == "uni-budget-matched":
        folder = root / "uni-budget-matched"
    else:
        raise ValueError("unknown matrix method: " + method)
    src = np.load(folder / ("source_anchor_positions_q%d.npy" % q))
    ref = np.load(folder / ("reference_anchor_positions_q%d.npy" % q))
    if np.any(np.diff(src) <= 0) or np.any(np.diff(ref) <= 0):
        raise ValueError("anchor positions must be strictly increasing in time")
    return src.astype(np.int64), ref.astype(np.int64)


def matrix_record(payload, method: str, q: int):
    key = {"uni-q": "uni_q", "ads-vic": "ads_vic",
           "uni-budget-matched": "uni_budget_matched"}[method]
    item = payload["results"][key][str(q)]
    return item["final"] if method == "ads-vic" else item


def anchor_profile(matrix: np.ndarray, bucket_centers: np.ndarray):
    n_src, n_ref = matrix.shape
    src_sum = np.zeros(n_src, dtype=np.float64)
    ref_sum = np.zeros(n_ref, dtype=np.float64)
    for first in range(0, n_src, CHUNK_ROWS):
        last = min(first + CHUNK_ROWS, n_src)
        block = bucket_centers[np.asarray(matrix[first:last])]
        src_sum[first:last] = block.sum(axis=1, dtype=np.float64)
        ref_sum += block.sum(axis=0, dtype=np.float64)
    src_mass = np.abs(src_sum).sum()
    ref_mass = np.abs(ref_sum).sum()
    src_share = np.abs(src_sum) / src_mass * 100.0 if src_mass else np.zeros(n_src)
    ref_share = np.abs(ref_sum) / ref_mass * 100.0 if ref_mass else np.zeros(n_ref)
    return src_sum, ref_sum, src_share, ref_share


def draw_matrix(plt, root: Path, output_root: Path, payload: dict,
                method: str, q: int, logger):
    record = matrix_record(payload, method, q)
    matrix_path = ROOT / record["matrix_file"]
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.dtype != np.uint8 or list(matrix.shape) != record["shape"]:
        raise ValueError("matrix shape or dtype differs from its manifest")
    src_positions, ref_positions = load_positions(root, method, q)
    if matrix.shape != (len(src_positions), len(ref_positions)):
        raise ValueError("matrix dimensions do not match chronological anchor positions")

    bucket_centers = np.asarray(record["bucket_centers"], dtype=np.float32)
    src_sum, ref_sum, src_share, ref_share = anchor_profile(matrix, bucket_centers)
    selected = payload["results"]["ads_vic"][str(q)]["top_quartile_pair_selection"]
    cutoff_bucket = int(selected["cutoff_bucket_id"])

    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 8.0),
                               height_ratios=(1.35, 8.0), wspace=0.04, hspace=0.04)
    ax_corner = figure.add_subplot(grid[0, 0])
    ax_top = figure.add_subplot(grid[0, 1])
    ax_left = figure.add_subplot(grid[1, 0])
    ax_heat = figure.add_subplot(grid[1, 1], sharex=ax_top, sharey=ax_left)
    ax_corner.axis("off")

    colors = plt.get_cmap("RdBu_r")
    image = ax_heat.imshow(
        matrix, origin="lower", interpolation="nearest", aspect="auto",
        cmap=colors, vmin=0, vmax=255,
        extent=(-0.5, len(ref_positions)-0.5, -0.5, len(src_positions)-0.5),
        rasterized=True,
    )
    overlay_colors = np.zeros((256, 4), dtype=np.float32)
    overlay_colors[cutoff_bucket:, :3] = (1.0, 1.0, 1.0)
    overlay_colors[cutoff_bucket:, 3] = 0.28
    from matplotlib.colors import ListedColormap
    overlay = ListedColormap(overlay_colors)
    ax_heat.imshow(
        matrix, origin="lower", interpolation="nearest", aspect="auto",
        cmap=overlay, vmin=0, vmax=255,
        extent=(-0.5, len(ref_positions)-0.5, -0.5, len(src_positions)-0.5),
        rasterized=True,
    )

    ref_axis = np.arange(len(ref_positions))
    src_axis = np.arange(len(src_positions))
    ref_colors = np.where(ref_sum >= 0, "#b2182b", "#2166ac")
    src_colors = np.where(src_sum >= 0, "#b2182b", "#2166ac")
    ax_top.bar(ref_axis, ref_share, width=1.0, color=ref_colors, linewidth=0)
    ax_top.axhline(0, color="#333333", linewidth=0.7)
    ax_top.set_ylabel("Ref sum magnitude\n(% of ref total)", fontsize=9)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(axis="y", alpha=0.2)

    ax_left.fill_betweenx(src_axis, 0, src_share, color="#8c8c8c", alpha=0.8,
                          linewidth=0)
    ax_left.axvline(0, color="#333333", linewidth=0.7)
    ax_left.set_xlabel("Src sum magnitude\n(% of src total)", fontsize=9)
    ax_left.invert_xaxis()
    ax_left.tick_params(axis="y", labelleft=False)
    ax_left.grid(axis="x", alpha=0.2)

    def time_ticks(positions):
        indexes = np.unique(np.linspace(0, len(positions)-1,
                                        min(5, len(positions)), dtype=int))
        return indexes, [str(int(positions[index])) for index in indexes]

    xticks, xlabels = time_ticks(ref_positions)
    yticks, ylabels = time_ticks(src_positions)
    ax_heat.set_xticks(xticks, xlabels)
    ax_heat.set_yticks(yticks, ylabels)
    ax_heat.set_xlabel("Reference anchor window start (time ascending)")
    ax_heat.set_ylabel("Source anchor window start (time ascending)")
    ax_heat.tick_params(axis="x", labelrotation=0)
    ax_heat.tick_params(axis="y", labelrotation=0)

    method_label = {"uni-q": "Uniform coarse", "ads-vic": "ADS-VIC final",
                    "uni-budget-matched": "Budget-matched uniform"}[method]
    title = "%s | q=%d | matrix=%d x %d | ADS coarse top-quartile cutoff=%.5f" % (
        method_label, q, len(src_positions), len(ref_positions),
        float(selected["cutoff_bf16_cosine"]))
    figure.suptitle(title, fontsize=14)
    colorbar = figure.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.02)
    colorbar.set_ticks([0, 64, 128, 192, 255])
    colorbar.set_ticklabels(["-0.996", "-0.496", "0.004", "0.504", "0.996"])
    colorbar.set_label("Cosine similarity (256-bucket center)")
    figure.text(0.5, 0.015,
                "White overlay: score bucket at or above the coarse top-quartile cutoff bucket. "
                "Marginal bars show absolute signed row/column sum magnitude, normalized within that side; "
                "red/blue indicates positive/negative sum.",
                ha="center", fontsize=9)

    out_dir = output_path("selection", output_root / method)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / ("score_matrix_q%d" % q)
    figure.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(figure)

    np.savez_compressed(
        out_dir / ("marginal_sums_q%d.npz" % q),
        source_anchor_positions=src_positions,
        reference_anchor_positions=ref_positions,
        source_signed_row_sum=src_sum,
        reference_signed_column_sum=ref_sum,
        source_relative_magnitude_percent=src_share,
        reference_relative_magnitude_percent=ref_share,
    )
    logger.info("stage=matrix-plotted method=%s q=%d shape=%s cutoff_bucket=%d output=%s",
                method, q, matrix.shape, cutoff_bucket, stem.with_suffix(".png"))
    return {
        "method": method, "q": q, "shape": list(matrix.shape),
        "matrix_file": record["matrix_file"],
        "heatmap_file": "outputs/" + str(stem.relative_to((ROOT / "outputs").resolve())) + ".png",
        "marginal_file": "outputs/" + str((out_dir / ("marginal_sums_q%d.npz" % q)).relative_to((ROOT / "outputs").resolve())),
        "score_sum_normalization": "absolute per-anchor signed row/column sums normalized to 100 percent within each side",
        "displayed_high_score_cutoff_bucket": cutoff_bucket,
        "displayed_cutoff_cosine": float(selected["cutoff_bf16_cosine"]),
    }


def main():
    logger = configure_logger("main_ads_v0_score_matrix_heatmaps",
                              "main-ads-v0", "score-matrix-heatmaps", "startup")
    started = time.monotonic()
    root = INPUT_ROOT.resolve()
    if not (root / ".complete").is_file():
        raise FileNotFoundError("score matrix batch is incomplete")
    payload = json.loads((root / "cosine_histograms.json").read_text(encoding="ascii"))
    output = output_path("selection", OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    methods = ("uni_q", "ads_vic", "uni_budget_matched")
    method_names = {"uni_q": "uni-q", "ads_vic": "ads-vic",
                    "uni_budget_matched": "uni-budget-matched"}
    for q in payload["q_values"]:
        for method in methods:
            entries.append(draw_matrix(plt, root, output, payload,
                                       method_names[method], int(q), logger))
    summary = {
        "schema": "main-ads-v0-score-matrix-heatmaps-v1",
        "experiment": "main-ads-v0",
        "batch": "ads-histgram",
        "axis_order": "reference columns left-to-right and source rows bottom-to-top, both sorted by trajectory window start",
        "heatmap_values": "dequantized uint8 cosine bucket IDs via bucket centers",
        "marginals": "signed score sum across each row or column; displayed magnitude is normalized by sum of absolute per-anchor sums on that side; sign is shown by color",
        "high_score_overlay": "bucket IDs at or above the coarse top-quartile cutoff bucket; bucket resolution can include boundary-bucket ties",
        "entries": entries,
    }
    (output / "heatmap_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii")
    (output / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete entries=%d output=%s elapsed=%s",
                len(entries), output, format_duration(time.monotonic()-started))


if __name__ == "__main__":
    main()
