"""Replay cached projected window gradients to compare anchor layouts.

Reads cached gradient stores without computing any gradient. For each configured
trajectory store and each equivalent spacing ``q`` this tool builds two anchor
layouts with an identical, exact anchor budget:

* ``bisect``: repeated widest-gap midpoint bisection, i.e. the CRB coverage
  stage with the error and split stages disabled.
* ``uniform``: budget-matched nearest-integer ``linspace`` anchors.

Every window is then rebuilt by linear interpolation between its bracketing
anchors and compared with the cached gradient, so the tool only spends replay
work, never model backward passes.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
import json
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_bytes, format_duration
from configs.paths import output_path
from rage.replay_cache import (
    ProjectedGradientStore,
    anchor_budget,
    bisection_plan,
    compare_layouts,
    reconstruction_metrics,
    uniform_plan,
)
from scripts.tools._common import apply_overrides, load_config, required, resolve

METHODS = ("bisect", "uniform")


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(path)


def svg_budget_curves(path: Path, title: str, panels: list[tuple], q_values: list[float]) -> None:
    """Write a dependency-free SVG comparing both layouts across budgets."""
    width, cell_height = 1400, 360
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    height = rows * cell_height + 50
    cell_width = width / columns
    margin_x, margin_y = 72, 68
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
             % (width, height, width, height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="%d" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">%s</text>'
             % (width // 2, title)]
    label_step = max(1, len(q_values) // 8)
    for panel_index, (panel_title, traces) in enumerate(panels):
        row, column = divmod(panel_index, columns)
        x0 = column * cell_width + margin_x
        y0 = row * cell_height + margin_y
        plot_width = cell_width - margin_x - 40
        plot_height = cell_height - margin_y - 55
        maximum = max(max(series) for _color, _label, series in traces)
        maximum = max(maximum, 1e-9) * 1.08
        parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="15">%s</text>'
                     % (x0, y0 - 18, panel_title))
        for tick in range(4):
            y = y0 + plot_height * tick / 3
            value = maximum * (1 - tick / 3)
            parts.append('<line x1="%.1f" y1="%.2f" x2="%.1f" y2="%.2f" stroke="#dddddd"/>'
                         % (x0, y, x0 + plot_width, y))
            parts.append('<text x="%.1f" y="%.2f" text-anchor="end" font-family="sans-serif" font-size="11">%.4g</text>'
                         % (x0 - 8, y + 4, value))
        for tick, label in enumerate(q_values):
            x = x0 + plot_width * tick / float(len(q_values) - 1)
            parts.append('<line x1="%.2f" y1="%.1f" x2="%.2f" y2="%.1f" stroke="#efefef"/>'
                         % (x, y0, x, y0 + plot_height))
            if tick % label_step == 0 or tick == len(q_values) - 1:
                parts.append('<text x="%.2f" y="%.1f" text-anchor="middle" font-family="sans-serif" font-size="11">%g</text>'
                             % (x, y0 + plot_height + 20, label))
        for trace_index, (color, label, values) in enumerate(traces):
            coordinates = []
            for tick, value in enumerate(values):
                x = x0 + plot_width * tick / float(len(values) - 1)
                y = y0 + plot_height * (1 - float(value) / maximum)
                coordinates.append("%.2f,%.2f" % (x, y))
            parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
                         % (color, " ".join(coordinates)))
            for tick, value in enumerate(values):
                x = x0 + plot_width * tick / float(len(values) - 1)
                y = y0 + plot_height * (1 - float(value) / maximum)
                parts.append('<circle cx="%.2f" cy="%.2f" r="2.6" fill="%s"/>' % (x, y, color))
            legend_x = x0 + 8
            legend_y = y0 + 16 * (trace_index + 1)
            parts.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>'
                         % (legend_x, legend_y, legend_x + 20, legend_y, color))
            parts.append('<text x="%.1f" y="%.1f" font-family="sans-serif" font-size="12">%s</text>'
                         % (legend_x + 26, legend_y + 4, label))
        parts.append('<text x="%.1f" y="%.1f" text-anchor="middle" font-family="sans-serif" font-size="12">q (equivalent spacing)</text>'
                     % (x0 + plot_width / 2, y0 + plot_height + 40))
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="ascii")


def replay(config: dict) -> Path:
    replay_config = dict(required(config, "replay"))
    raw_q = replay_config.get("q_values")
    if not isinstance(raw_q, list) or not raw_q:
        raise ValueError("replay.q_values must be a non-empty list")
    q_values = [float(value) for value in raw_q]
    if any(not np.isfinite(value) or value < 1 for value in q_values):
        raise ValueError("replay.q_values must be finite and at least 1")
    chunk_rows = int(replay_config.get("chunk_rows", 4096))
    cosine_ge = float(replay_config.get("cosine_ge", 0.9))
    context_length = int(required(config, "context_length"))
    prediction_length = int(required(config, "prediction_length"))
    window_stride = int(config.get("window_stride", 1))
    metric_records = bool(replay_config.get("save_window_metrics", True))
    run_id = str(required(config, "run_id"))
    destination = output_path("selection", required(config, "output_dir"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("replay output is not empty: %s" % destination)
    destination.mkdir(parents=True, exist_ok=True)

    raw_stores = required(replay_config, "stores")
    if not isinstance(raw_stores, list) or not raw_stores:
        raise ValueError("replay.stores must be a non-empty list")
    logger = configure_logger("tsl.tools.replay", "tools/replay", run_id, "REPLAY")
    started = time.monotonic()
    logger.info("stage=config run_id=%s stores=%d q_values=%s chunk_rows=%d cosine_ge=%.3f "
                "context_length=%d prediction_length=%d window_stride=%d output=%s",
                run_id, len(raw_stores), q_values, chunk_rows, cosine_ge,
                context_length, prediction_length, window_stride, destination)

    records = {}
    for store_index, entry in enumerate(raw_stores):
        if not isinstance(entry, dict):
            raise ValueError("each replay store must be a mapping")
        label = str(required(entry, "label"))
        cache_dir = resolve(required(entry, "cache_dir"))
        store = ProjectedGradientStore(cache_dir)
        logger.info("stage=store ordinal=%d label=%s rows=%d dimensions=%d dtype=%s "
                    "subset=%s series_id=%s bytes=%s",
                    store_index, label, store.rows, store.dimensions, store.gradients.dtype,
                    store.manifest.get("subset"), store.manifest.get("series_id"),
                    format_bytes((store.root / "gradient_sketches.npy").stat().st_size))
        store_record = {
            "label": label,
            "ordinal": store_index,
            "cache_dir": str(required(entry, "cache_dir")),
            "windows": store.rows,
            "dimensions": store.dimensions,
            "storage_dtype": str(store.gradients.dtype),
            "subset": store.manifest.get("subset"),
            "series_id": store.manifest.get("series_id"),
            "manifest": store.manifest,
            "q": {},
        }
        for q in q_values:
            budget = anchor_budget(store.rows, q)
            layouts = {}
            layouts["bisect"], bisect_cost = bisection_plan(
                store, q, context_length=context_length,
                prediction_length=prediction_length, window_stride=window_stride)
            layouts["uniform"] = uniform_plan(store, q)
            results = {}
            for method in METHODS:
                anchors = layouts[method]
                result = reconstruction_metrics(store, anchors, chunk_rows=chunk_rows,
                                                cosine_ge=cosine_ge)
                results[method] = result
                anchor_dir = destination / "anchors" / method
                anchor_dir.mkdir(parents=True, exist_ok=True)
                np.save(anchor_dir / ("%s_anchor_positions_q%g.npy" % (label, q)), anchors)
                if metric_records:
                    metrics_dir = destination / "window_metrics" / method
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    np.save(metrics_dir / ("%s_q%g.npy" % (label, q)),
                            np.stack([result["relative_l2"], result["cosine"]],
                                     axis=1).astype(np.float32))
                logger.info("stage=layout-complete label=%s q=%g method=%s anchors=%d "
                            "mean_relative_l2=%.6f mean_cosine=%.6f "
                            "mean_relative_l2_non_anchor=%.6f mean_cosine_non_anchor=%.6f "
                            "fraction_cos_ge=%.4f",
                            label, q, method, result["summary"]["anchor_count"],
                            result["summary"]["mean_relative_l2"],
                            result["summary"]["mean_cosine"],
                            result["summary"]["mean_relative_l2_non_anchor"],
                            result["summary"]["mean_cosine_non_anchor"],
                            result["summary"]["fraction_cosine_ge_threshold"])
            comparison = compare_layouts(results["bisect"], results["uniform"])
            comparison["served_anchor_rows"] = int(bisect_cost)
            store_record["q"][("%g" % q)] = {
                "anchor_budget": int(budget),
                "bisect": results["bisect"]["summary"],
                "uniform": results["uniform"]["summary"],
                "comparison": comparison,
            }
            logger.info("stage=q-complete label=%s q=%g anchors=%d cosine_delta=%.6f "
                        "relative_l2_delta=%.6f cosine_wins_bisect=%d uniform=%d ties=%d",
                        label, q, budget,
                        comparison["cosine_delta_bisect_minus_uniform"],
                        comparison["relative_l2_delta_bisect_minus_uniform"],
                        comparison["window_cosine_wins_bisect"],
                        comparison["window_cosine_wins_uniform"],
                        comparison["window_cosine_ties"])
        records[label] = store_record

    payload = {
        "schema": "tsl.tools.replay.v1",
        "run_id": run_id,
        "tool": "scripts/tools/replay.py",
        "model_free_replay": True,
        "replay_semantics": ("Anchors are served from the cached projected gradients; no window gradient is "
                             "computed, so no model forward/backward pass occurs during this stage"),
        "gradient_cache_readonly": True,
        "anchor_budget_rule": "K = max(2, ceil(N / q)); both endpoints always included",
        "bisect_rule": ("CRB coverage stage only: repeated widest-gap midpoint bisection, error and split "
                        "density set to 0, detection disabled; ties prefer the wider gap then the leftmost gap"),
        "uniform_rule": "Budget-matched nearest-integer linspace(0, N-1, K) anchors",
        "reconstruction_rule": ("rage.window_reconstruction.interpolate_window_gradients: piecewise linear on "
                                "raw anchor vectors, anchors returned exactly, never extrapolated"),
        "metric_rule": ("Per window relative L2 = ||estimate - truth|| / ||truth|| and cosine(estimate, truth); "
                        "averages include anchor windows unless marked non_anchor"),
        "translation_invariant_note": ("The cached vectors are Count-Sketch projected gradients, so absolute "
                                       "cosines are measured in the projected space, not the parameter space"),
        "chunk_rows": chunk_rows,
        "cosine_ge": cosine_ge,
        "q_values": q_values,
        "stores": records,
        "outputs": {
            "anchor_positions": "anchors/<method>/<label>_anchor_positions_q<q>.npy",
            "window_metrics": "window_metrics/<method>/<label>_q<q>.npy (columns: relative_l2, cosine)",
        },
    }
    json_path = destination / "replay_metrics.json"
    write_json(json_path, payload)
    if bool(replay_config.get("plot", True)):
        panels = []
        for label, store_record in records.items():
            cosine_traces = []
            relative_traces = []
            for color, method in (("#2878B5", "uniform"), ("#D9534F", "bisect")):
                cosine_traces.append((color, method, [store_record["q"]["%g" % q][method]["mean_cosine"]
                                                      for q in q_values]))
                relative_traces.append((color, method, [store_record["q"]["%g" % q][method]["mean_relative_l2"]
                                                        for q in q_values]))
            panels.append(("%s (%d windows) mean cosine" % (label, store_record["windows"]), cosine_traces))
            panels.append(("%s (%d windows) mean relative L2" % (label, store_record["windows"]), relative_traces))
        svg_budget_curves(destination / "bisect-vs-uniform.svg",
                          "Longest-gap bisection vs budget-matched uniform anchors", panels, q_values)
    (destination / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete elapsed=%s output=%s json=%s",
                format_duration(time.monotonic() - started), destination, json_path)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    replay(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
