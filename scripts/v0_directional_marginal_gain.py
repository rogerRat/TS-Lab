"""Test whether block direction dispersion predicts one-anchor gains."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from scripts.v0_vage_static import spaced, split_gap, reconstruction
from scripts.analyze_sparse_proxy_risk_long import statistics
from scripts.v0_var_proxy_estimate import write_csv

LENGTH = 256
BUDGETS = (2, 4, 8, 16)
PROXY_POSITIONS = np.asarray([0, 36, 73, 109, 146, 182, 219, 255], dtype=np.int64)
PREDICTORS = ("true_vector_variance", "relative_vector_variance",
              "true_direction_dispersion", "sparse_proxy_direction_dispersion")
TARGETS = ("initial_direction_error", "total_gain", "gain_excluding_new_anchor",
           "gain_remaining_unqueried")


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def direction_error(gram: np.ndarray, anchors: list[int]) -> np.ndarray:
    _, cosine, _ = reconstruction(gram, anchors)
    error = 1.0 - np.clip(cosine, -1.0, 1.0)
    error[np.asarray(anchors, dtype=np.int64)] = 0.0
    return error


def make_record(gram: np.ndarray, track: str, block: int, prior: dict,
                proxy_dispersion: float, budget: int) -> dict:
    anchors = spaced(0, LENGTH, budget)
    new_anchor = split_gap(anchors, 0, LENGTH)
    after = sorted(anchors + [new_anchor])
    old_error = direction_error(gram, anchors)
    new_error = direction_error(gram, after)
    improvement = old_error - new_error
    exclude_new = np.ones(LENGTH, dtype=bool)
    exclude_new[new_anchor] = False
    remaining = np.ones(LENGTH, dtype=bool)
    remaining[after] = False
    if not remaining.any() or abs(new_error[new_anchor]) > 1e-12:
        raise ValueError("invalid post-insertion anchor state")
    total_gain = float(improvement.mean())
    nonnew_gain = float(improvement[exclude_new].mean())
    unqueried_gain = float(improvement[remaining].mean())
    direct_gain = float(old_error[new_anchor] / LENGTH)
    if not np.isclose(total_gain, direct_gain + nonnew_gain * (LENGTH - 1) / LENGTH,
                      rtol=1e-10, atol=1e-11):
        raise ValueError("gain decomposition mismatch")
    return {"track": track, "block": block, "block_start": block * LENGTH,
            "budget": budget, "initial_anchors": ",".join(map(str, anchors)),
            "new_anchor": new_anchor, "new_anchors": ",".join(map(str, after)),
            "true_vector_variance": float(prior["true_vector_variance"]),
            "relative_vector_variance": float(prior["relative_vector_variance"]),
            "true_direction_dispersion": float(prior["direction_dispersion"]),
            "sparse_proxy_direction_dispersion": proxy_dispersion,
            "initial_direction_error": float(old_error.mean()),
            "post_direction_error": float(new_error.mean()),
            "total_gain": total_gain,
            "gain_at_new_anchor": direct_gain,
            "gain_excluding_new_anchor": nonnew_gain,
            "gain_remaining_unqueried": unqueried_gain,
            "remaining_unqueried_count": int(remaining.sum()),
            "worsened_positions_excluding_new": int(np.sum(improvement[exclude_new] < -1e-10))}


def make_correlations(rows: list[dict]) -> list[dict]:
    result = []
    for budget in BUDGETS:
        selected = [row for row in rows if row["budget"] == budget]
        if len(selected) != 46:
            raise ValueError("unexpected block count at budget %d" % budget)
        for predictor in PREDICTORS:
            x = np.asarray([row[predictor] for row in selected])
            for target in TARGETS:
                y = np.asarray([row[target] for row in selected])
                track_values = []
                for track in range(10):
                    subset = [row for row in selected if row["track"] == "trajectory-%02d" % track]
                    value = correlation(np.asarray([row[predictor] for row in subset]),
                                        np.asarray([row[target] for row in subset]), "spearman")
                    if value is not None:
                        track_values.append(value)
                result.append({"budget": budget, "predictor": predictor, "target": target,
                               "blocks": len(selected),
                               "pooled_pearson": correlation(x, y, "pearson"),
                               "pooled_spearman": correlation(x, y, "spearman"),
                               "equal_track_spearman": float(np.mean(track_values)) if track_values else None,
                               "tracks_within": len(track_values)})
    return result


def make_summary(rows: list[dict]) -> list[dict]:
    result = []
    for budget in BUDGETS:
        selected = [row for row in rows if row["budget"] == budget]
        for metric in ("initial_direction_error", "post_direction_error", "total_gain",
                       "gain_at_new_anchor", "gain_excluding_new_anchor",
                       "gain_remaining_unqueried"):
            values = np.asarray([row[metric] for row in selected])
            result.append({"budget": budget, "metric": metric, "blocks": len(selected),
                           "mean": float(values.mean()), "median": float(np.median(values)),
                           "p10": float(np.quantile(values, 0.1)),
                           "p90": float(np.quantile(values, 0.9)),
                           "positive_blocks": int(np.sum(values > 1e-10)),
                           "negative_blocks": int(np.sum(values < -1e-10))})
    return result


def make_leave_one_track_out(rows: list[dict]) -> list[dict]:
    result = []
    for budget in BUDGETS:
        for predictor in PREDICTORS:
            for omitted in range(10):
                selected = [row for row in rows if row["budget"] == budget
                            and row["track"] != "trajectory-%02d" % omitted]
                x = np.asarray([row[predictor] for row in selected])
                y = np.asarray([row["gain_excluding_new_anchor"] for row in selected])
                result.append({"budget": budget, "predictor": predictor,
                               "target": "gain_excluding_new_anchor",
                               "omitted_track": "trajectory-%02d" % omitted,
                               "blocks": len(selected),
                               "spearman": correlation(x, y, "spearman")})
    return result


def main() -> None:
    source = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
    prior = PROJECT_ROOT / "outputs/train/v0-gradient-variance-decomposition"
    out = output_path("train", "outputs/train/v0-directional-marginal-gain")
    if not (prior / ".complete").is_file() or (out / ".complete").exists():
        raise ValueError("missing prior completion or completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.directional.marginal.gain", "v0-directional-marginal-gain",
                           "replay", "initialization",
                           log_directory=PROJECT_ROOT / "logs/v0-directional-marginal-gain")
    started = time.monotonic()
    log.info("[configuration] block_length=%d budgets=%s proxy_count=%d source=%s output=%s",
             LENGTH, BUDGETS, len(PROXY_POSITIONS), source, out)
    with (prior / "block_metrics.csv").open(encoding="ascii", newline="") as handle:
        previous = list(csv.DictReader(handle))
    if len(previous) != 46:
        raise ValueError("unexpected source block count")
    rows = []
    proxy_rows = []
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        if not (directory / ".complete").is_file():
            raise ValueError("incomplete cache: %s" % name)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        count = int(manifest["windows"])
        with np.load(directory / "gram_matrices.npz") as archive:
            full = np.asarray(archive["full"], dtype=np.float64)
            head = np.asarray(archive["head"], dtype=np.float64)
        if full.shape != (count, count) or head.shape != (count, count):
            raise ValueError("Gram shape mismatch: %s" % name)
        track_prior = [row for row in previous if row["track"] == name]
        if len(track_prior) != count // LENGTH:
            raise ValueError("block count mismatch: %s" % name)
        for prior_row in track_prior:
            block = int(prior_row["block"])
            start = block * LENGTH
            if int(prior_row["block_start"]) != start:
                raise ValueError("block alignment mismatch: %s block=%d" % (name, block))
            block_full = full[start:start + LENGTH, start:start + LENGTH]
            _, full_dispersion = statistics(full, np.arange(start, start + LENGTH))
            if abs(full_dispersion - float(prior_row["direction_dispersion"])) > 1e-9:
                raise ValueError("true direction dispersion mismatch")
            _, proxy_dispersion = statistics(head, start + PROXY_POSITIONS)
            proxy_rows.append({"track": name, "block": block, "block_start": start,
                               "true_direction_dispersion": full_dispersion,
                               "sparse_proxy_direction_dispersion": proxy_dispersion})
            for budget in BUDGETS:
                rows.append(make_record(block_full, name, block, prior_row,
                                        proxy_dispersion, budget))
        log.info("[track] complete track=%s blocks=%d budget_rows=%d elapsed=%s", name,
                 len(track_prior), len(track_prior) * len(BUDGETS),
                 format_duration(time.monotonic() - started))
    associations = make_correlations(rows)
    proxy_x = np.asarray([r["true_direction_dispersion"] for r in proxy_rows])
    proxy_y = np.asarray([r["sparse_proxy_direction_dispersion"] for r in proxy_rows])
    proxy_association = {"blocks": len(proxy_rows),
                         "pearson": correlation(proxy_x, proxy_y, "pearson"),
                         "spearman": correlation(proxy_x, proxy_y, "spearman")}
    write_csv(out / "block_budget_metrics.csv", rows)
    write_csv(out / "correlations.csv", associations)
    write_csv(out / "budget_summary.csv", make_summary(rows))
    write_csv(out / "proxy_direction_dispersion.csv", proxy_rows)
    write_csv(out / "leave_one_track_out.csv", make_leave_one_track_out(rows))
    (out / "proxy_association.json").write_text(json.dumps(proxy_association, indent=2) + "\n", encoding="ascii")
    (out / "input_manifest.json").write_text(json.dumps({
        "source": "outputs/train/v0-gradient-proxy-test",
        "prior": "outputs/train/v0-gradient-variance-decomposition/block_metrics.csv",
        "block_length": LENGTH, "complete_blocks": len(proxy_rows),
        "budgets": BUDGETS, "proxy_positions": PROXY_POSITIONS.tolist(),
        "initial_anchors": {str(n): spaced(0, LENGTH, n) for n in BUDGETS},
        "new_anchor_rule": "midpoint floor of largest consecutive anchor gap; leftmost tied gap",
        "error": "mean one-minus-cosine over all 256 positions",
        "gain": "initial mean error minus post-insertion mean error",
        "gain_excluding_new_anchor": "mean per-position improvement over 255 positions other than new anchor",
        "gain_remaining_unqueried": "mean improvement excluding all initial and new anchors",
        "sparse_proxy_direction_dispersion": "eight evenly spaced output-head gradients, population unit-vector dispersion",
        "gradient_recomputation": False}, indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("[complete] block_budget_rows=%d associations=%d elapsed=%s output=%s",
             len(rows), len(associations), format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
