"""QuitoBench under the explicit RAGE raw-unit, univariate evaluation protocol."""
import csv
import json
import logging
from pathlib import Path

import numpy as np

from configs.paths import project_path
from evaluate.schemas import resolve_public_class
from evaluate.quito_data import (REPO_ID, REVISION, UPSTREAM_COMMIT, CUTOFF,
                                TARGETS, SEASONALITY, data_file, iter_series,
                                window_starts, seasonal_scale, validate_snapshot)

LOGGER = logging.getLogger("evaluate")


def validate_config(value):
    if set(value) != {"dataset", "tasks", "stride"}:
        raise ValueError("Quito bench_config requires dataset, tasks, stride")
    dataset = value["dataset"]
    if set(dataset) != {"repo_id", "revision", "root"} or dataset["repo_id"] != REPO_ID or dataset["revision"] != REVISION:
        raise ValueError("Quito dataset must match pinned repo_id and revision")
    root = project_path(dataset["root"])
    validate_snapshot(root)
    stride = value["stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("Quito stride must be a positive integer")
    tasks = value["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Quito tasks must be a non-empty list")
    identities = set()
    for task in tasks:
        if set(task) != {"subset", "prediction_length"}:
            raise ValueError("Quito task requires subset and prediction_length")
        horizon = task["prediction_length"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("Quito prediction_length must be a positive integer")
        path = data_file(root, task["subset"])
        if not path.is_file():
            raise FileNotFoundError(path)
        identity = (task["subset"], horizon)
        if identity in identities:
            raise ValueError("duplicate Quito task")
        identities.add(identity)
    return root, tasks, stride


def evaluate_task(adapter, root, task, inference, stride):
    """Stream batches, count exclusions, and macro-average windows in raw units."""
    subset, horizon = task["subset"], task["prediction_length"]
    context = inference["context_length"]
    levels = np.asarray(inference["quantile_levels"])
    median = list(levels).index(0.5)
    sums = {key: 0.0 for key in ("MAE", "MSE", "MASE", "SQL")}
    absolute_target = absolute_error = quantile_loss = 0.0
    count = scaled_count = invalid = candidates = unavailable = 0
    batch, truths, scales = [], [], []

    def flush():
        nonlocal count, scaled_count, absolute_target, absolute_error, quantile_loss
        predictions = np.asarray(adapter.predict_quantiles(batch, horizon))
        if predictions.shape != (len(batch), len(levels), horizon) or not np.isfinite(predictions).all():
            raise ValueError("Quito adapter returned invalid predictions")
        actual = np.asarray(truths)
        error = actual - predictions[:, median, :]
        delta = actual[:, None, :] - predictions
        pinball = 2 * np.maximum(levels[None, :, None] * delta,
                                (levels[None, :, None] - 1) * delta)
        mae = np.abs(error).mean(axis=1)
        sums["MAE"] += float(mae.sum())
        sums["MSE"] += float((error ** 2).mean(axis=1).sum())
        scale = np.asarray(scales)
        valid = np.isfinite(scale) & (scale > 0)
        sums["MASE"] += float((mae[valid] / scale[valid]).sum())
        sums["SQL"] += float((pinball.mean(axis=(1, 2))[valid] / scale[valid]).sum())
        scaled_count += int(valid.sum())
        absolute_target += float(np.abs(actual).sum())
        absolute_error += float(np.abs(error).sum())
        quantile_loss += float(pinball.mean(axis=1).sum())
        count += len(batch)
        batch.clear(); truths.clear(); scales.clear()

    for entity, values, boundary in iter_series(root, subset):
        starts = window_starts(len(values), boundary, context, horizon,
                               inference.get("evaluation_mode", "single_window"), stride)
        if not len(starts):
            unavailable += 1
        for start in starts:
            candidates += 1
            window = values[start-context:start+horizon]
            if not np.isfinite(window).all():
                invalid += 1
                continue
            batch.append({"item_id": entity, "target": window[:context]})
            truths.append(window[context:])
            scales.append(seasonal_scale(values[:start], SEASONALITY[subset]))
            if len(batch) == inference["batch_size"]:
                flush()
    if batch:
        flush()
    if count == 0:
        raise ValueError("Quito task produced no finite evaluation windows")
    return {"task": "%s/h%d" % (subset, horizon), "windows": count,
            "candidate_windows": candidates, "invalid_windows": invalid,
            "unavailable_channels": unavailable, "scaled_windows": scaled_count,
            "MAE": sums["MAE"] / count, "MSE": sums["MSE"] / count,
            "MASE": sums["MASE"] / scaled_count if scaled_count else None,
            "SQL": sums["SQL"] / scaled_count if scaled_count else None,
            "WAPE": absolute_error / absolute_target if absolute_target else None,
            "WQL": quantile_loss / absolute_target if absolute_target else None}


def run_quito(config, checkpoint: Path, output_dir: Path, resume: bool):
    from evaluate.run import _write_json
    root, tasks, stride = validate_config(config.bench_config)
    adapter_type = resolve_public_class(config.adapter_class)
    parsed = adapter_type.parse_config(dict(config.adapter_config), config.source_path.parent)
    metadata = {"benchmark": "quito", "protocol": "rage-quito-raw-univariate-v1",
                "config": config.as_mapping(), "checkpoint": str(checkpoint.resolve()),
                "cutoff": str(CUTOFF), "targets": list(TARGETS),
                "upstream_commit": UPSTREAM_COMMIT, "adapter_config": parsed,
                "aggregation": "window-macro MAE/MSE/MASE/SQL; pooled WAPE/WQL; task-macro summary",
                "missing_policy": "skip nonfinite windows and report counts",
                "official_leaderboard_comparable": False}
    metadata_path = output_dir / "quito_metadata.json"
    if metadata_path.exists():
        if not resume or json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("Quito resume metadata does not match this run")
    else:
        _write_json(metadata_path, metadata)
    adapter = adapter_type.from_config(parsed, checkpoint=checkpoint)
    rows = []
    for task in tasks:
        path = output_dir / ("%s_h%d.json" % (task["subset"], task["prediction_length"]))
        if resume and path.is_file():
            row = json.loads(path.read_text())
            expected_task = "%s/h%d" % (task["subset"], task["prediction_length"])
            required = {"MAE", "MSE", "MASE", "SQL", "WAPE", "WQL",
                        "task", "windows", "candidate_windows", "invalid_windows",
                        "unavailable_channels", "scaled_windows"}
            if not isinstance(row, dict) or set(row) != required or row["task"] != expected_task or row["windows"] <= 0:
                raise ValueError("Quito resume task is incomplete: %s" % path)
        else:
            if path.exists():
                raise FileExistsError(path)
            LOGGER.info("Quito task started: %s", task)
            row = evaluate_task(adapter, root, task, parsed["inference"], stride)
            _write_json(path, row)
        rows.append(row)
    result = output_dir / "quito_results.csv"
    with result.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"task": "mean"}
    for metric in ("MAE", "MSE", "MASE", "SQL", "WAPE", "WQL"):
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = float(np.mean(values)) if values else None
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader(); writer.writerow(summary)
    return result
