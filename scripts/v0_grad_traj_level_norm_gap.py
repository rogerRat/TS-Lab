"""Compare mean-gradient and mean-window-gradient norms across ten trajectories."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
    out = output_path("train", "outputs/train/v0-grad-traj-level-norm-gap")
    if (out / ".complete").exists():
        raise ValueError("completed output exists")
    out.mkdir(parents=True, exist_ok=True)
    log = configure_logger("v0.grad.traj.level.norm.gap", "v0-grad-traj-level-norm-gap",
                           "analysis", "initialization",
                           log_directory=PROJECT_ROOT / "logs/v0-grad-traj-level-norm-gap")
    started = time.monotonic()
    log.info("[configuration] tracks=10 gradient_kinds=full,output_head source=%s output=%s", source, out)
    rows = []
    for track in range(10):
        name = "trajectory-%02d" % track
        directory = source / name
        if not (directory / ".complete").is_file():
            raise ValueError("incomplete gradient cache: %s" % name)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        count = int(manifest["windows"])
        with np.load(directory / "gram_matrices.npz") as archive:
            gram_by_kind = {kind: np.asarray(archive[key], dtype=np.float64)
                            for kind, key in (("full", "full"), ("output_head", "head"))}
        for kind, gram in gram_by_kind.items():
            if gram.shape != (count, count) or not np.isfinite(gram).all():
                raise ValueError("invalid %s Gram cache: %s" % (kind, name))
            if not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-7):
                raise ValueError("asymmetric %s Gram cache: %s" % (kind, name))
            diagonal = np.diag(gram)
            if np.any(diagonal <= 0):
                raise ValueError("nonpositive %s squared norms: %s" % (kind, name))
            mean_squared_norm = float(diagonal.mean())
            mean_vector_squared_norm = float(gram.mean())
            if mean_vector_squared_norm < -1e-10 * mean_squared_norm:
                raise ValueError("negative mean-vector squared norm: %s" % name)
            mean_vector_norm = float(np.sqrt(max(0.0, mean_vector_squared_norm)))
            mean_window_norm = float(np.sqrt(diagonal).mean())
            variance = max(0.0, mean_squared_norm - mean_vector_squared_norm)
            rows.append({"track": name, "windows": count, "kind": kind,
                         "norm_of_mean_gradient": mean_vector_norm,
                         "mean_of_gradient_norms": mean_window_norm,
                         "rms_gradient_norm": float(np.sqrt(mean_squared_norm)),
                         "population_vector_variance": variance,
                         "mean_to_mean_norm_ratio": mean_vector_norm / mean_window_norm})
        log.info("[track] complete track=%s windows=%d elapsed=%s", name, count,
                 format_duration(time.monotonic() - started))
    summary = []
    for kind in ("full", "output_head"):
        selected = [r for r in rows if r["kind"] == kind]
        for metric in ("norm_of_mean_gradient", "mean_of_gradient_norms",
                       "rms_gradient_norm", "mean_to_mean_norm_ratio"):
            values = np.asarray([r[metric] for r in selected], dtype=np.float64)
            low = int(np.argmin(values))
            high = int(np.argmax(values))
            summary.append({"kind": kind, "metric": metric,
                            "min": float(values[low]), "min_track": selected[low]["track"],
                            "max": float(values[high]), "max_track": selected[high]["track"],
                            "max_to_min": float(values[high] / values[low]) if values[low] > 0 else None,
                            "median": float(np.median(values)),
                            "mean": float(values.mean()),
                            "population_cv": float(values.std() / values.mean())})
    paired = []
    full = [r for r in rows if r["kind"] == "full"]
    head = [r for r in rows if r["kind"] == "output_head"]
    for metric in ("norm_of_mean_gradient", "mean_of_gradient_norms",
                   "mean_to_mean_norm_ratio"):
        x = np.asarray([r[metric] for r in full])
        y = np.asarray([r[metric] for r in head])
        paired.append({"metric": metric, "tracks": 10,
                       "pearson": float(pearsonr(x, y).statistic),
                       "spearman": float(spearmanr(x, y).statistic)})
    write_csv(out / "trajectory_metrics.csv", rows)
    write_csv(out / "spread_summary.csv", summary)
    write_csv(out / "full_head_association.csv", paired)
    (out / "input_manifest.json").write_text(json.dumps({
        "source": "outputs/train/v0-gradient-proxy-test",
        "tracks": 10, "total_windows": sum(r["windows"] for r in full),
        "gradient_kinds": ["full", "output_head"],
        "primary_metric": "norm of arithmetic mean gradient vector across all legal windows per trajectory",
        "secondary_metric": "arithmetic mean of per-window gradient norms",
        "computation": "all-coordinate saved full and head Gram matrices; no model recomputation",
        "inference": "descriptive across ten fixed selected trajectories; no independent-window p-values"},
        indent=2) + "\n", encoding="ascii")
    (out / ".complete").write_text("complete\n", encoding="ascii")
    log.info("[complete] trajectory_rows=%d elapsed=%s output=%s", len(rows),
             format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
