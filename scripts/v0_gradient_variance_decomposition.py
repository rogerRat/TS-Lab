"""Decompose block gradient variance and explain reconstruction errors."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from scripts.v0_var_proxy_estimate import write_csv

LENGTH = 256
ANCHORS = np.asarray([0, 85, 170, 255], dtype=np.int64)
PAIRS = (
    ("true_vector_variance", "mean_gradient_norm"),
    ("true_vector_variance", "amplitude_variance"),
    ("true_vector_variance", "direction_dispersion"),
    ("true_vector_variance", "relative_vector_variance"),
    ("true_vector_variance", "amplitude_fraction"),
    ("true_vector_variance", "magnitude_error"),
    ("true_vector_variance", "direction_error"),
    ("true_vector_variance", "mean_cosine"),
    ("amplitude_variance", "magnitude_error"),
    ("direction_dispersion", "direction_error"),
    ("direction_dispersion", "mean_cosine"),
    ("relative_vector_variance", "direction_error"),
    ("relative_vector_variance", "mean_cosine"),
    ("proxy_variance", "mean_cosine"),
    ("proxy_variance", "direction_error"),
)


def corr(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def controls(rows: list[dict], ranked: bool) -> np.ndarray:
    norm = np.asarray([r["mean_gradient_norm"] for r in rows])
    if ranked:
        norm = rankdata(norm)
    tracks = np.asarray([int(r["track"].split("-")[-1]) for r in rows])
    return np.column_stack([np.ones(len(rows)), norm,
                            *[(tracks == track).astype(float) for track in range(1, 10)]])


def residual(value: np.ndarray, design: np.ndarray) -> np.ndarray:
    return value - design @ np.linalg.lstsq(design, value, rcond=None)[0]


def partial_corr(rows: list[dict], x_name: str, y_name: str, ranked: bool) -> float | None:
    if x_name == "mean_gradient_norm" or y_name == "mean_gradient_norm":
        return None
    x = np.asarray([r[x_name] for r in rows], dtype=float)
    y = np.asarray([r[y_name] for r in rows], dtype=float)
    if ranked:
        x, y = rankdata(x), rankdata(y)
    design = controls(rows, ranked)
    return corr(residual(x, design), residual(y, design), "pearson")


def reconstruction(gram: np.ndarray) -> dict[str, float]:
    positions = np.arange(LENGTH)
    right_i = np.searchsorted(ANCHORS, positions, side="right")
    left = ANCHORS[np.clip(right_i - 1, 0, len(ANCHORS) - 1)]
    right = ANCHORS[np.clip(right_i, 0, len(ANCHORS) - 1)]
    alpha = np.divide(positions - left, right - left,
                      out=np.zeros(LENGTH, dtype=float), where=right != left)
    a, b = 1.0 - alpha, alpha
    truth_sq = np.diag(gram)
    estimate_sq = a * a * gram[left, left] + b * b * gram[right, right] + 2.0 * a * b * gram[left, right]
    dot = a * gram[positions, left] + b * gram[positions, right]
    if np.any(truth_sq <= 0) or np.any(estimate_sq <= 0):
        raise ValueError("nonpositive reconstruction norm")
    l2 = np.sqrt(np.maximum(0.0, truth_sq + estimate_sq - 2.0 * dot))
    cosine = np.clip(dot / np.sqrt(truth_sq * estimate_sq), -1.0, 1.0)
    magnitude = np.abs(np.sqrt(truth_sq) - np.sqrt(estimate_sq))
    l2[ANCHORS] = 0.0
    cosine[ANCHORS] = 1.0
    magnitude[ANCHORS] = 0.0
    mask = np.ones(LENGTH, dtype=bool)
    mask[ANCHORS] = False
    result = {}
    for label, selected in (("all", np.ones(LENGTH, dtype=bool)), ("nonanchor", mask)):
        result[label + "_mean_l2"] = float(l2[selected].mean())
        result[label + "_mean_cosine"] = float(cosine[selected].mean())
        result[label + "_magnitude_error"] = float(magnitude[selected].mean())
        result[label + "_direction_error"] = float((1.0 - cosine[selected]).mean())
    return result


def analyze_block(gram: np.ndarray) -> dict[str, float]:
    if gram.shape != (LENGTH, LENGTH) or not np.isfinite(gram).all():
        raise ValueError("invalid full-gradient block Gram")
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0):
        raise ValueError("nonpositive full-gradient norm")
    norm = np.sqrt(diagonal)
    mean_norm = float(norm.mean())
    mean_sq = float(diagonal.mean())
    mean_vector_sq = float(gram.mean())
    variance = max(0.0, mean_sq - mean_vector_sq)
    amplitude_variance = max(0.0, mean_sq - mean_norm * mean_norm)
    cosine_gram = np.clip(gram / (norm[:, None] * norm[None, :]), -1.0, 1.0)
    dispersion = float(np.clip(1.0 - cosine_gram.mean(), 0.0, 2.0))
    directional_component = max(0.0, variance - amplitude_variance)
    result = {
        "mean_gradient_norm": mean_norm,
        "mean_squared_gradient_norm": mean_sq,
        "amplitude_variance": amplitude_variance,
        "direction_dispersion": dispersion,
        "true_vector_variance": variance,
        "relative_vector_variance": variance / mean_sq,
        "directional_variance_component": directional_component,
        "amplitude_fraction": amplitude_variance / variance if variance > 0 else 0.0,
    }
    result.update(reconstruction(gram))
    return result


def build_correlations(rows: list[dict]) -> list[dict]:
    output = []
    for subset in ("all", "nonanchor"):
        view = [{**r, "mean_l2": r[subset + "_mean_l2"],
                 "mean_cosine": r[subset + "_mean_cosine"],
                 "magnitude_error": r[subset + "_magnitude_error"],
                 "direction_error": r[subset + "_direction_error"]} for r in rows]
        for predictor, target in PAIRS:
            x = np.asarray([r[predictor] for r in view])
            y = np.asarray([r[target] for r in view])
            by_track = []
            for track in range(10):
                selected = [r for r in view if r["track"] == "trajectory-%02d" % track]
                value = corr(np.asarray([r[predictor] for r in selected]),
                             np.asarray([r[target] for r in selected]), "spearman")
                if value is not None:
                    by_track.append(value)
            output.append({"subset": subset, "predictor": predictor, "target": target,
                           "blocks": len(view), "pooled_pearson": corr(x, y, "pearson"),
                           "pooled_spearman": corr(x, y, "spearman"),
                           "equal_track_spearman": float(np.mean(by_track)) if by_track else None,
                           "tracks_within": len(by_track),
                           "partial_pearson": partial_corr(view, predictor, target, False),
                           "partial_spearman": partial_corr(view, predictor, target, True)})
    return output


def draw_scatter(rows: list[dict], correlations: list[dict], subset: str, out: Path) -> None:
    view = [{**r, "mean_cosine": r[subset + "_mean_cosine"],
             "direction_error": r[subset + "_direction_error"]} for r in rows]
    cases = (("true_vector_variance", "mean_cosine", "Full variance vs cosine"),
             ("proxy_variance", "mean_cosine", "Proxy variance vs cosine"),
             ("direction_dispersion", "direction_error", "Direction spread vs angle error"))
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (x_name, y_name, title) in enumerate(cases):
        record = next(r for r in correlations if r["subset"] == subset
                      and r["predictor"] == x_name and r["target"] == y_name)
        for row_index in range(2):
            ax = axes[row_index, column]
            if row_index == 0:
                x = np.asarray([r[x_name] for r in view])
                y = np.asarray([r[y_name] for r in view])
                ax.set_title("%s\nraw Spearman %.3f" % (title, record["pooled_spearman"]))
                ax.set_xlabel(x_name.replace("_", " "))
                ax.set_ylabel(y_name.replace("_", " "))
            else:
                design = controls(view, True)
                x = residual(rankdata([r[x_name] for r in view]), design)
                y = residual(rankdata([r[y_name] for r in view]), design)
                ax.set_title("%s\nadjusted rank r %.3f" % (title, record["partial_spearman"]))
                ax.set_xlabel("rank residual: " + x_name.replace("_", " "))
                ax.set_ylabel("rank residual: " + y_name.replace("_", " "))
            for track in range(10):
                mask = np.asarray([r["track"] == "trajectory-%02d" % track for r in view])
                ax.scatter(x[mask], y[mask], color=colors(track), s=30, alpha=0.85,
                           label="Track %02d" % track if row_index == 0 and column == 0 else None)
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                span = np.asarray([x.min(), x.max()])
                ax.plot(span, slope * span + intercept, color="0.2", linewidth=1)
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=10, fontsize=8)
    fig.suptitle("256-start blocks: raw and controlled associations (%s)" % subset)
    fig.savefig(out / ("scatter_%s.png" % subset), dpi=180)
    fig.savefig(out / ("scatter_%s.svg" % subset))
    plt.close(fig)


def main() -> None:
    source = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
    prior = PROJECT_ROOT / "outputs/train/v0-var-proxy-estimate/true_variance"
    out = output_path("train", "outputs/train/v0-gradient-variance-decomposition")
    if not (prior / ".complete").is_file() or (out / ".complete").exists():
        raise ValueError("missing prior completion or output already complete")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.gradient.variance.decomposition", "v0-gradient-variance-decomposition",
                           "analysis", "initialization",
                           log_directory=PROJECT_ROOT / "logs/v0-gradient-variance-decomposition")
    started = time.monotonic()
    log.info("[configuration] tracks=10 blocks=46 length=256 anchors=%s output=%s",
             ANCHORS.tolist(), out)
    with (prior / "block_metrics.csv").open(encoding="ascii", newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 46:
        raise ValueError("unexpected prior block count")
    rows = []
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        if not (directory / ".complete").is_file():
            raise ValueError("incomplete trajectory: %s" % name)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        n = int(manifest["windows"])
        with np.load(directory / "gram_matrices.npz") as archive:
            gram = np.asarray(archive["full"], dtype=np.float64)
        if gram.shape != (n, n) or not np.isfinite(gram).all():
            raise ValueError("invalid full Gram cache: %s" % name)
        for old in (r for r in original if r["track"] == name):
            block = int(old["block"])
            start = block * LENGTH
            if int(old["block_start"]) != start or start + LENGTH > n:
                raise ValueError("block misalignment: %s block=%d" % (name, block))
            record = analyze_block(gram[start:start + LENGTH, start:start + LENGTH])
            if abs(record["true_vector_variance"] - float(old["true_full_variance"])) > 1e-9:
                raise ValueError("variance mismatch: %s block=%d" % (name, block))
            for new, old_key in (("all_mean_l2", "mean_l2_error"),
                                 ("all_mean_cosine", "mean_cosine"),
                                 ("all_magnitude_error", "mean_abs_norm_difference")):
                if abs(record[new] - float(old[old_key])) > 1e-8:
                    raise ValueError("reconstruction mismatch: %s block=%d metric=%s" % (name, block, new))
            rows.append({"track": name, "block": block, "block_start": start,
                         "proxy_variance": float(old["proxy_variance"]), **record})
        log.info("[track] complete track=%s blocks=%d elapsed=%s", name,
                 sum(r["track"] == name for r in rows), format_duration(time.monotonic() - started))
    correlations = build_correlations(rows)
    write_csv(out / "block_metrics.csv", rows)
    write_csv(out / "correlations.csv", correlations)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    log.info("[plot] start raw and controlled scatter figures")
    for subset in ("all", "nonanchor"):
        draw_scatter(rows, correlations, subset, figures)
    (out / "input_manifest.json").write_text(json.dumps({
        "source": "outputs/train/v0-gradient-proxy-test",
        "prior": "outputs/train/v0-var-proxy-estimate/true_variance/block_metrics.csv",
        "blocks": len(rows), "block_length": LENGTH, "anchors": ANCHORS.tolist(),
        "direction_dispersion": "1 - squared norm of mean unit full-gradient vector",
        "variance": "population vector variance over 256 full gradients",
        "amplitude_variance": "population variance of 256 full-gradient norms",
        "relative_variance": "vector variance divided by mean squared full-gradient norm",
        "partial_pearson": "residual correlation after linear control for mean norm and track fixed effects",
        "partial_spearman": "residual correlation of global ranks after rank mean-norm and track fixed-effect control",
        "nonanchor_positions": 252, "significance_tests": False}, indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("[complete] rows=%d correlations=%d elapsed=%s output=%s",
             len(rows), len(correlations), format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
