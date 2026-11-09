"""Summarize completed proxy geometry measurements with equal trajectory weight."""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from configs.logger import configure_logger
from configs.paths import PROJECT_ROOT, output_path
from scripts.gradient_proxy_experiment import save_csv, save_json


def main():
    log = configure_logger("proxy_summary", "v0-gradient-proxy-test", "summary", "summary")
    for handler in log.handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s-[proxy_summary]-[summary]-[%(message)s]"))
    out = output_path("train", "outputs/train/v0-gradient-proxy-test")
    if not (out / ".complete").exists():
        raise RuntimeError("all ten trajectories must complete before final summary")
    log.info("Starting summary from %s", out)
    torch.set_num_threads(2)
    config = json.loads((out / "config.json").read_text())
    audit = []
    total_windows = 0
    for track in range(10):
        dest = out / ("trajectory-%02d" % track)
        if not (dest / ".complete").exists():
            raise RuntimeError("missing trajectory completion marker")
        manifest = json.loads((dest / "manifest.json").read_text())
        n = manifest["windows"]
        total_windows += n
        raw = np.memmap(PROJECT_ROOT / config["cache_root"] / ("trajectory-%02d.bf16" % track),
                        mode="r", dtype=np.uint16, shape=(n, 8394496))
        reference = torch.from_numpy(np.array(raw[[0, n-1]], copy=True)).view(torch.bfloat16).double()
        exact = (reference @ reference.T).numpy()
        gram = np.load(dest / "gram_matrices.npz")["full"]
        observed = gram[np.ix_([0, n-1], [0, n-1])]
        scale = np.sqrt(np.diag(exact)[:, None]*np.diag(exact)[None, :])
        error = float(np.max(np.abs(exact-observed)/scale))
        if error > 1e-5:
            raise RuntimeError("Gram matrix disagrees with direct FP64 dot product")
        audit.append({"track": track, "max_norm_scaled_gram_error": error})
    if total_windows != 12836:
        raise RuntimeError("unexpected total window count")
    save_json(out / "analysis_validation.json", {"window_count": total_windows, "checks": audit})
    with (out / "correlations.csv").open() as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scope"], row["lag"], row["target"], row["proxy"])].append(row)
    summary = []
    for (scope, lag, target, proxy), values in grouped.items():
        item = dict(scope=scope, lag=int(lag), target=target, proxy=proxy, trajectories=len(values))
        for statistic in ("pearson", "spearman"):
            numbers = [float(r[statistic]) for r in values if r[statistic]]
            item[statistic + "_mean"] = float(np.mean(numbers)) if numbers else None
            item[statistic + "_median"] = float(np.median(numbers)) if numbers else None
            item[statistic + "_min"] = min(numbers) if numbers else None
            item[statistic + "_max"] = max(numbers) if numbers else None
        summary.append(item)
    save_csv(out / "trajectory_equal_weight_summary.csv", summary)
    calibration = []
    for track in range(10):
        pair = np.load(out / ("trajectory-%02d" % track) / "pair_metrics.npz")
        for lag in np.unique(pair["lag"]):
            mask = pair["lag"] == lag
            row = {"track": track, "lag": int(lag), "pair_count": int(mask.sum())}
            for name in ("full", "head", "embedding"):
                row[name + "_cosine_mean"] = float(np.mean(pair[name + "_cosine"][mask]))
                row[name + "_cosine_std"] = float(np.std(pair[name + "_cosine"][mask]))
            for name in ("head", "embedding"):
                row[name + "_cosine_mae"] = float(np.mean(np.abs(pair[name + "_cosine"][mask]-pair["full_cosine"][mask])))
            calibration.append(row)
    save_csv(out / "cosine_calibration.csv", calibration)
    lines = ["# Full-gradient proxy assessment", "", "Last modified: " + datetime.now().strftime("%Y-%m-%d"), "",
             "Status: all ten retained trajectories completed. Scope: 12,836 windows of the selected M5 tail sample at the fixed warmed Chronos-T5 checkpoint.", "",
             "Protocol: ../gradient-proxy-protocol.md. Direction is primary. Full reference gradients are BF16 copies of FP32 BP; every coordinate is included. Proxy head gradients are FP32 output-path derivatives with detached backbone states. Embeddings are valid-token means of final decoder states under teacher forcing. Loss is the original CE.", "",
             "All reported summary correlations weight trajectories equally. Ranges are observed trajectory ranges, not confidence intervals. Overlapping pairs are dependent; no pair-independent p-values are used.", "",
             "## Window norm associations", "",
             "| Proxy | Mean Pearson | Mean Spearman | Spearman range |", "|---|---:|---:|---:|"]
    def fmt(x):
        return "NA" if x is None else "%.4f" % x
    for r in summary:
        if r["scope"] == "window":
            lines.append("| %s | %s | %s | %s to %s |" % (r["proxy"], fmt(r["pearson_mean"]), fmt(r["spearman_mean"]), fmt(r["spearman_min"]), fmt(r["spearman_max"])))
    lines += ["", "## Direction associations at fixed lag", "",
              "Head and embedding entries correlate pairwise cosine with full-gradient cosine. Loss entries correlate absolute loss difference with full-gradient angular distance (1-cosine). Larger positive values indicate better ordering in all columns. These measure association, not equality of cosine values.", "",
              "| Lag | Proxy | Mean Pearson | Mean Spearman | Spearman range |", "|---:|---|---:|---:|---:|"]
    for r in summary:
        if r["scope"] == "pair" and r["target"] in ("full_cosine", "full_angular_distance"):
            lines.append("| %d | %s | %s | %s | %s to %s |" % (r["lag"], r["proxy"], fmt(r["pearson_mean"]), fmt(r["spearman_mean"]), fmt(r["spearman_min"]), fmt(r["spearman_max"])))
    lines += ["", "## Within-block variation", "",
              "Nonoverlapping blocks of 32 starts; partial terminal blocks are excluded from these correlations. Vector variance includes magnitude and direction. Direction dispersion uses normalized gradients and therefore isolates directional spread.", "",
              "| Target | Proxy | Mean Pearson | Mean Spearman | Spearman range |", "|---|---|---:|---:|---:|"]
    for r in summary:
        if r["scope"] == "block32":
            lines.append("| %s | %s | %s | %s | %s to %s |" % (r["target"], r["proxy"], fmt(r["pearson_mean"]), fmt(r["spearman_mean"]), fmt(r["spearman_min"]), fmt(r["spearman_max"])))
    lines += ["", "## Checks and artifacts", "",
              "Three full FP32 BP cache-alignment checks per trajectory and one analytic-head/autograd comparison per trajectory passed. See per-trajectory validation_and_timing.json for numerical errors. Cache checks use relative L2 <=0.01 and cosine >=0.9999; head derivative checks use relative L2 <=0.0001 for FP32 softmax cancellation/accumulation differences.", "",
              "Artifacts under outputs/train/v0-gradient-proxy-test: correlations.csv (per trajectory), trajectory_equal_weight_summary.csv, cosine_calibration.csv (absolute cosine means/spread and proxy MAE), blocks.csv, and per-trajectory head_gradients.npy, window_features.npz, gram_matrices.npz, pair_metrics.npz, manifests and validation records.", "",
              "## Interpretation limits", "",
              "Full and proxy dimensions differ: direct cross-space cosine was not computed. A strong correlation does not imply calibrated absolute cosine. The head proxy shares output-path information with the full gradient. Teacher-forced features require future labels. This study does not fit a mapping or evaluate CoVar allocation, held-out training outcomes, or matched proxy speed benchmarks. Any proxy selection from these trajectories needs held-out confirmation.", ""]
    report = PROJECT_ROOT / "docs/results/v0-gradient-proxy-test.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="ascii")
    (out / ".analysis_complete").write_text("complete\n", encoding="ascii")
    log.info("Completed summary: output=%s report=%s", out, report)


if __name__ == "__main__":
    main()
