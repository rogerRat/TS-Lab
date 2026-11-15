"""Render ADS histogram summaries with matplotlib from the rage environment."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from configs.logger import configure_logger
from configs.paths import output_path

OUT_DIR = ROOT / "outputs/selection/main-ads-v0/ads-histgram-bf16-u8"


def draw_overlay(plt, records, method, output):
    q_values = sorted(int(value) for value in records[method])
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.cm.viridis([i / max(1, len(q_values) - 1) for i in range(len(q_values))])
    for color, q in zip(colors, q_values):
        record = records[method][str(q)]
        hist = record if method != "ads_vic" else record["final"]
        edges = hist["histogram_edges"]
        frequency = hist["histogram_relative_frequency"]
        label = "q=%d" % q
        if method == "ads_vic":
            label += ", refine=%d" % record["refinement_spacing"]
        axis.stairs(frequency, edges, color=color, label=label, linewidth=1.5)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Relative frequency (bin count / total pairs)")
    axis.set_title(method.replace("_", "-").upper() + " cosine relative-frequency histogram")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def draw_budget_comparison(plt, records, comparisons, output):
    q_values = sorted(int(value) for value in records["ads_vic"])
    rows = (len(q_values) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(13, 4 * rows), squeeze=False,
                                constrained_layout=True)
    for index, q in enumerate(q_values):
        axis = axes.flat[index]
        uni = records["uni_budget_matched"][str(q)]
        vic = records["ads_vic"][str(q)]["final"]
        axis.stairs(uni["histogram_relative_frequency"], uni["histogram_edges"],
                    label="uni budget-matched", color="#2878B5", linewidth=1.5)
        axis.stairs(vic["histogram_relative_frequency"], vic["histogram_edges"],
                    label="ADS-VIC", color="#D9534F", linewidth=1.5)
        count = comparisons[str(q)]
        axis.set_title("q=%d; anchors uni=%d, ADS-VIC=%d" % (
            q, count["uni_budget_matched_total_anchor_count"],
            count["ads_vic_total_anchor_count"]))
        axis.set_xlim(-1, 1)
        axis.set_xlabel("Cosine similarity")
        axis.set_ylabel("Relative frequency")
        axis.grid(True, alpha=0.25)
        axis.legend()
    axes.flat[-1].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main():
    logger = configure_logger("main_ads_v0_histogram_renderer", "main-ads-v0",
                              "ads-histgram-bf16-u8-render", "startup")
    directory = output_path("selection", OUT_DIR)
    if not (directory / ".complete").is_file():
        raise FileNotFoundError("ADS histogram computation is not complete")
    payload = json.loads((directory / "cosine_histograms.json").read_text(encoding="ascii"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = payload["results"]
    draw_overlay(plt, records, "uni_q", directory / "uni-q/cosine_histogram")
    draw_overlay(plt, records, "uni_budget_matched",
                 directory / "uni-budget-matched/cosine_histogram")
    draw_overlay(plt, records, "ads_vic", directory / "ads-vic/cosine_histogram")
    draw_budget_comparison(plt, records, payload["anchor_count_comparison"],
                           directory / "uni-vs-ads-vic/cosine_histograms_by_q")
    logger.info("stage=complete outputs=%s", directory)


if __name__ == "__main__":
    main()
