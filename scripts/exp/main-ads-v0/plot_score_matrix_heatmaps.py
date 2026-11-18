"""Plot chronological cosine matrices with top-fraction activation counts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from configs.logger import configure_logger, format_duration
from configs.paths import output_path

INPUT_ROOT = ROOT / "outputs/selection/main-ads-v0/ads-histgram-bf16-u8"
OUTPUT_ROOT = (INPUT_ROOT / "score-matrix-heatmaps").resolve()
CHUNK_ROWS = 8192
LOW_BUCKET = 115
HIGH_BUCKET = 140
CENTER_BUCKET = (LOW_BUCKET + HIGH_BUCKET) / 2.0


def load_positions(root: Path, method: str, q: int):
    folder = root / method
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


def top_fraction_activation_counts(matrix: np.ndarray, denominator: int,
                                  positive_only: bool, logger, label: str):
    """Count selected high-score pairs at each endpoint using uint8 scores."""
    if denominator < 1:
        raise ValueError("activation denominator must be positive")
    n_src, n_ref = matrix.shape
    pair_count = int(n_src) * int(n_ref)
    bucket_counts = np.zeros(256, dtype=np.int64)
    total_blocks = (n_src + CHUNK_ROWS - 1) // CHUNK_ROWS
    log_every = max(1, total_blocks // 8)
    started = time.monotonic()

    logger.info("stage=activation-count-pass1-start label=%s pairs=%d denominator=%d positive_only=%s",
                label, pair_count, denominator, positive_only)
    for block_index, first in enumerate(range(0, n_src, CHUNK_ROWS), start=1):
        last = min(first + CHUNK_ROWS, n_src)
        block = np.asarray(matrix[first:last])
        bucket_counts += np.bincount(block.reshape(-1), minlength=256)
        if block_index % log_every == 0 or block_index == total_blocks:
            elapsed = time.monotonic() - started
            eta = elapsed * (total_blocks - block_index) / max(block_index, 1)
            logger.info("stage=activation-count-pass1-progress label=%s rows=%d/%d elapsed=%s eta=%s",
                        label, last, n_src, format_duration(elapsed), format_duration(eta))

    candidate_pair_count = int(bucket_counts[128:].sum()) if positive_only else pair_count
    if candidate_pair_count == 0:
        raise RuntimeError("matrix has no nonnegative score buckets")
    target = (candidate_pair_count + denominator - 1) // denominator
    logger.info("stage=activation-count-candidates label=%s candidates=%d target=%d",
                label, candidate_pair_count, target)

    remaining_above = 0
    cutoff_bucket = None
    cutoff_take = 0
    for bucket_id in range(255, -1, -1):
        count = int(bucket_counts[bucket_id])
        if remaining_above + count >= target:
            cutoff_bucket = bucket_id
            cutoff_take = target - remaining_above
            break
        remaining_above += count
    if cutoff_bucket is None:
        raise RuntimeError("failed to find uint8 cutoff bucket")

    src_count = np.zeros(n_src, dtype=np.int64)
    ref_count = np.zeros(n_ref, dtype=np.int64)
    cutoff_remaining = cutoff_take
    started = time.monotonic()
    logger.info("stage=activation-count-pass2-start label=%s cutoff_bucket=%d cutoff_take=%d",
                label, cutoff_bucket, cutoff_take)
    for block_index, first in enumerate(range(0, n_src, CHUNK_ROWS), start=1):
        last = min(first + CHUNK_ROWS, n_src)
        block = np.asarray(matrix[first:last])
        higher = block > cutoff_bucket
        src_count[first:last] = higher.sum(axis=1, dtype=np.int64)
        ref_count += higher.sum(axis=0, dtype=np.int64)

        tied = block == cutoff_bucket
        tied_count = int(tied.sum())
        take = min(cutoff_remaining, tied_count)
        if take == tied_count:
            src_count[first:last] += tied.sum(axis=1, dtype=np.int64)
            ref_count += tied.sum(axis=0, dtype=np.int64)
        elif take > 0:
            flat_indices = np.flatnonzero(tied.reshape(-1))[:take]
            src_rows = flat_indices // n_ref + first
            ref_cols = flat_indices % n_ref
            src_count[first:last] += np.bincount(src_rows - first,
                                                minlength=last-first)
            ref_count += np.bincount(ref_cols, minlength=n_ref)
        cutoff_remaining -= take

        if block_index % log_every == 0 or block_index == total_blocks:
            elapsed = time.monotonic() - started
            eta = elapsed * (total_blocks - block_index) / max(block_index, 1)
            logger.info("stage=activation-count-pass2-progress label=%s rows=%d/%d elapsed=%s eta=%s",
                        label, last, n_src, format_duration(elapsed), format_duration(eta))

    if cutoff_remaining != 0 or int(src_count.sum()) != target or int(ref_count.sum()) != target:
        raise RuntimeError("activation counts do not match selected pair budget")
    logger.info("stage=activation-count-complete label=%s selected=%d cutoff_bucket=%d src_active=%d ref_active=%d",
                label, target, cutoff_bucket, int(np.count_nonzero(src_count)),
                int(np.count_nonzero(ref_count)))
    return src_count, ref_count, {
        "pair_count": pair_count,
        "candidate_pair_count": candidate_pair_count,
        "positive_only": positive_only,
        "selected_pair_count": target,
        "selected_fraction_of_candidates": target / candidate_pair_count,
        "selected_fraction_of_matrix": target / pair_count,
        "cutoff_bucket_id": int(cutoff_bucket),
        "cutoff_bucket_total_count": int(bucket_counts[cutoff_bucket]),
        "cutoff_bucket_selected_count": int(cutoff_take),
        "tie_break_rule": "descending uint8 bucket ID, then row-major order within cutoff bucket",
    }


def draw_matrix(plt, root: Path, output_root: Path, final_output_root: Path,
                payload: dict, method: str, q: int, denominator: int,
                positive_only: bool, logger):
    record = matrix_record(payload, method, q)
    matrix_path = ROOT / record["matrix_file"]
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.dtype != np.uint8 or list(matrix.shape) != record["shape"]:
        raise ValueError("matrix shape or dtype differs from its manifest")
    src_positions, ref_positions = load_positions(root, method, q)
    if matrix.shape != (len(src_positions), len(ref_positions)):
        raise ValueError("matrix dimensions do not match chronological anchor positions")

    label = "%s-q%d" % (method, q)
    src_count, ref_count, activation = top_fraction_activation_counts(
        matrix, denominator, positive_only, logger, label)

    figure = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 8.0),
                               height_ratios=(1.35, 8.0), wspace=0.04, hspace=0.04)
    ax_corner = figure.add_subplot(grid[0, 0])
    ax_top = figure.add_subplot(grid[0, 1])
    ax_left = figure.add_subplot(grid[1, 0])
    ax_heat = figure.add_subplot(grid[1, 1], sharex=ax_top, sharey=ax_left)
    ax_corner.axis("off")

    from matplotlib.colors import Normalize, TwoSlopeNorm
    extent = (-0.5, len(ref_positions)-0.5, -0.5, len(src_positions)-0.5)
    if positive_only:
        display_matrix = np.ma.masked_less(matrix, 128)
        colors = plt.get_cmap("Reds").copy()
        colors.set_bad(color="#ffffff")
        norm = Normalize(vmin=128, vmax=148, clip=True)
        image = ax_heat.imshow(
            display_matrix, origin="lower", interpolation="nearest", aspect="auto",
            cmap=colors, norm=norm, extent=extent, rasterized=True,
        )
    else:
        norm = TwoSlopeNorm(vmin=LOW_BUCKET, vcenter=CENTER_BUCKET, vmax=HIGH_BUCKET)
        image = ax_heat.imshow(
            matrix, origin="lower", interpolation="nearest", aspect="auto",
            cmap=plt.get_cmap("RdBu_r"), norm=norm, extent=extent, rasterized=True,
        )

    ref_axis = np.arange(len(ref_positions))
    src_axis = np.arange(len(src_positions))
    bar_color = "#5b4b9a"
    ax_top.bar(ref_axis, ref_count, width=1.0, color=bar_color, linewidth=0)
    profile_label = ("positive top 1/%d" if positive_only else "top 1/%d") % denominator
    ax_top.set_ylabel("Ref activation count (%s)" % profile_label, fontsize=9)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(axis="y", alpha=0.2)
    ax_top.set_ylim(0, max(1, int(ref_count.max())) * 1.05)

    ax_left.fill_betweenx(src_axis, 0, src_count, color=bar_color, alpha=0.9,
                          linewidth=0)
    ax_left.set_xlabel("Src activation count (%s)" % profile_label, fontsize=9)
    ax_left.invert_xaxis()
    ax_left.tick_params(axis="y", labelleft=False)
    ax_left.grid(axis="x", alpha=0.2)
    ax_left.set_xlim(max(1, int(src_count.max())) * 1.05, 0)

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
    title = "%s | q=%d | matrix=%d x %d | %s pairs=%d" % (
        method_label, q, len(src_positions), len(ref_positions),
        profile_label, activation["selected_pair_count"])
    figure.suptitle(title, fontsize=14)
    if positive_only:
        tick_ids = np.asarray([128, 132, 136, 140, 144, 148], dtype=np.float64)
        tick_labels = ["%.3f" % ((float(bucket) + 0.5) / 128.0 - 1.0)
                       for bucket in tick_ids]
        colorbar = figure.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.02,
                                   extend="max", ticks=tick_ids)
        colorbar.set_ticklabels(tick_labels)
        colorbar.set_label("Nonnegative cosine bucket center; clipped above 0.160")
        description = "Negative score buckets are blank. Marginal profiles count incident pairs among positive "
    else:
        tick_ids = np.asarray([LOW_BUCKET, 118, 121, 124, CENTER_BUCKET,
                               131, 134, 137, HIGH_BUCKET], dtype=np.float64)
        tick_labels = ["%.3f" % ((float(bucket) + 0.5) / 128.0 - 1.0)
                       for bucket in tick_ids]
        colorbar = figure.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.02,
                                   extend="both", ticks=tick_ids)
        colorbar.set_ticklabels(tick_labels)
        colorbar.set_label("Cosine similarity (bucket centers; clipped outside +/-0.10)")
        description = "Matrix colors show cosine score. Marginal profiles count incident pairs among "
    figure.text(0.5, 0.015,
                description + "top 1/%d pairs; cutoff-bucket ties use row-major order." % denominator,
                ha="center", fontsize=9)

    out_dir = output_path("selection", output_root / method)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / ("score_matrix_q%d" % q)
    figure.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(figure)

    marginal_path = out_dir / ("activation_counts_q%d.npz" % q)
    np.savez_compressed(
        marginal_path,
        source_anchor_positions=src_positions,
        reference_anchor_positions=ref_positions,
        source_activation_count=src_count,
        reference_activation_count=ref_count,
        **{"activation_" + key: value for key, value in activation.items()},
    )
    output_relative = final_output_root.relative_to((ROOT / "outputs").resolve())
    logger.info("stage=matrix-plotted method=%s q=%d shape=%s selected=%d cutoff_bucket=%d output=%s",
                method, q, matrix.shape, activation["selected_pair_count"],
                activation["cutoff_bucket_id"], final_output_root / method / stem.name)
    return {
        "method": method,
        "q": q,
        "shape": list(matrix.shape),
        "matrix_file": record["matrix_file"],
        "heatmap_file": "outputs/" + str(output_relative / method / (stem.name + ".png")),
        "activation_file": "outputs/" + str(output_relative / method / marginal_path.name),
        "activation_selection": activation,
        "source_active_anchor_count": int(np.count_nonzero(src_count)),
        "reference_active_anchor_count": int(np.count_nonzero(ref_count)),
        "source_activation_count_min_max": [int(src_count.min()), int(src_count.max())],
        "reference_activation_count_min_max": [int(ref_count.min()), int(ref_count.max())],
    }


def replace_previous_output(staging: Path, destination: Path, logger):
    backup = destination.parent / ".score-matrix-heatmaps-previous"
    if backup.exists():
        raise FileExistsError("previous-output backup already exists: %s" % backup)
    moved_previous = False
    if destination.exists():
        os.replace(destination, backup)
        moved_previous = True
    try:
        os.replace(staging, destination)
    except BaseException:
        if moved_previous and backup.exists():
            os.replace(backup, destination)
        raise
    if moved_previous:
        shutil.rmtree(backup)
    logger.info("stage=previous-output-replaced destination=%s", destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-denominator", type=int, default=4)
    parser.add_argument("--methods", nargs="+",
                        choices=("uni-q", "ads-vic", "uni-budget-matched"),
                        default=("uni-q", "ads-vic", "uni-budget-matched"))
    parser.add_argument("--positive-only", action="store_true")
    args = parser.parse_args()
    if args.activation_denominator < 1:
        raise ValueError("activation denominator must be positive")
    if args.activation_denominator == 4 and not args.positive_only:
        final_output_root = OUTPUT_ROOT
    elif args.positive_only:
        final_output_root = (INPUT_ROOT / ("score-matrix-heatmaps-positive-top-1-%d" %
                                           args.activation_denominator)).resolve()
    else:
        final_output_root = (INPUT_ROOT / ("score-matrix-heatmaps-top-1-%d" %
                                           args.activation_denominator)).resolve()
    staging_root = final_output_root.parent / ("." + final_output_root.name + "-redraw")

    logger = configure_logger("main_ads_v0_score_matrix_heatmaps",
                              "main-ads-v0", "score-matrix-heatmaps-redraw", "startup")
    started = time.monotonic()
    logger.info("stage=start q_values=%s methods=%s activation_fraction=1/%d positive_only=%s color_bucket_range=%d..%d",
                [1, 2, 4, 8, 16, 31, 64], args.methods,
                args.activation_denominator, args.positive_only, LOW_BUCKET, HIGH_BUCKET)
    root = INPUT_ROOT.resolve()
    if not (root / ".complete").is_file():
        raise FileNotFoundError("score matrix batch is incomplete")
    payload = json.loads((root / "cosine_histograms.json").read_text(encoding="ascii"))
    if staging_root.exists():
        shutil.rmtree(staging_root)
    output = output_path("selection", staging_root)
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for q in payload["q_values"]:
        for method in args.methods:
            entries.append(draw_matrix(plt, root, output, final_output_root, payload,
                                       method, int(q), args.activation_denominator, args.positive_only, logger))
    summary = {
        "schema": "main-ads-v0-score-matrix-heatmaps-v3",
        "experiment": "main-ads-v0",
        "batch": "ads-histgram",
        "axis_order": "reference columns left-to-right and source rows bottom-to-top, both sorted by trajectory window start",
        "heatmap_values": ("negative buckets below 128 masked; nonnegative uint8 buckets shown with a Reds scale from 128 to 148" if args.positive_only else "uint8 cosine bucket IDs; diverging color scale clipped to bucket IDs 115..140, centered at cosine zero"),
        "activation_profiles": ("filter to nonnegative score buckets, select ceil(positive_pair_count/denominator) highest buckets, trim cutoff-bucket ties row-major, and count incident pairs per endpoint" if args.positive_only else "rank uint8 bucket IDs descending, select ceil(matrix_pair_count/denominator) pairs, trim cutoff-bucket ties in row-major order, and count selected incident pairs per endpoint"),
        "positive_only": args.positive_only,
        "activation_denominator": args.activation_denominator,
        "entries": entries,
    }
    (output / "heatmap_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii")
    (output / ".complete").write_text("complete\n", encoding="ascii")
    replace_previous_output(output, final_output_root, logger)
    logger.info("stage=complete entries=%d output=%s elapsed=%s",
                len(entries), final_output_root, format_duration(time.monotonic()-started))


if __name__ == "__main__":
    main()
