"""Summarize rolling outcomes and descriptive alignment associations."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

ROOT = PROJECT_ROOT
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ARMS = ("fullRand", "rage", "ancUni", "Traj")
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling"
TRAIN_ROOT = ROOT / "outputs/train/v1-rage-ads/rolling"
EVAL_ROOT = ROOT / "outputs/eval/v1-rage-ads/rolling"
AUDIT_ROOT = SELECTION_ROOT / "audit"


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                 for row in rows), encoding="ascii")
    os.replace(temporary, path)


def load_metrics(task, arm):
    directory = EVAL_ROOT / task / arm
    if not (directory / ".complete").is_file():
        raise FileNotFoundError("rolling evaluation is incomplete: %s/%s" % (task, arm))
    if task == "fev-09":
        path = directory / "fev_results.csv"
        with path.open(newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1:
            raise ValueError("FEV 9 must produce exactly one task summary row")
        metrics = {}
        for key in ("MASE", "MAE", "MSE", "WAPE", "WQL", "SQL"):
            if key in rows[0] and rows[0][key] != "":
                metrics[key] = float(rows[0][key])
        return metrics, str(path)
    files = list(directory.rglob("metrics.npz"))
    if len(files) != 1:
        raise ValueError("TIME evaluation must produce one metrics.npz file")
    metrics = {}
    with np.load(files[0], allow_pickle=False) as values:
        for key in values.files:
            row = np.asarray(values[key], dtype=np.float64)
            finite = row[np.isfinite(row)]
            if len(finite):
                metrics[key] = float(finite.mean())
    if "MASE" not in metrics:
        raise ValueError("TIME rolling metrics do not include MASE")
    return metrics, str(files[0])


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="ascii").splitlines()
            if line.strip()]


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and values[order[right]] == values[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + right - 1) / 2.0 + 1.0
        left = right
    return ranks


def correlation(left, right):
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(x) < 2 or len(y) != len(x) or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])
    return pearson, spearman


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    started = time.monotonic()
    logger = configure_logger("v1_rage_ads_result_summary", "v1-rage-ads",
                              "rolling-eight-task", "summary")
    for root in (TRAIN_ROOT, EVAL_ROOT):
        if not (root / ".complete").is_file():
            raise FileNotFoundError("training/evaluation batch is incomplete: " + str(root))
    rows = []
    metric_lookup = {}
    for task in TASKS:
        for arm in ARMS:
            if not (TRAIN_ROOT / task / arm / ".complete").is_file():
                raise FileNotFoundError("training cell is incomplete: %s/%s" % (task, arm))
            metrics, artifact = load_metrics(task, arm)
            metric_lookup[(task, arm)] = metrics
            rows.append(dict(task=task, arm=arm, status="complete",
                             primary_metric="MASE", primary_value=metrics["MASE"],
                             metrics=metrics, evaluation_artifact=artifact,
                             training_manifest=str(TRAIN_ROOT / task / arm / "training_manifest.json")))
    output = output_path("selection", AUDIT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "downstream-metrics.jsonl", rows)

    source_rows = {row["task"]: row for row in read_rows(output / "source-alignment.jsonl")}
    arm_rows = {(row["task"], row["arm"]): row for row in read_rows(output / "arm-alignment.jsonl")}
    continuity_rows = {row["task"]: row for row in read_rows(output / "reference-continuity.jsonl")}
    if (set(source_rows) != set(TASKS)
            or set(continuity_rows) != set(TASKS)
            or set(arm_rows) != {(task, arm) for task in TASKS for arm in ARMS}):
        raise ValueError("alignment audit rows do not cover all eight tasks and four arms")
    correlations = []
    for arm in ARMS[1:]:
        improvement = []
        features = {
            "source_p1_filtered_pairwise_mean": [],
            "arm_best_reference_mean": [],
            "arm_p1_filtered_pairwise_mean": [],
            "arm_p1_per_window_reference_mean": [],
            "continuity_mean_history_test_cosine": [],
            "continuity_best_reference_mean": [],
        }
        for task in TASKS:
            baseline = float(metric_lookup[(task, "fullRand")]["MASE"])
            selected = float(metric_lookup[(task, arm)]["MASE"])
            if baseline == 0:
                raise ValueError("fullRand MASE must be nonzero for relative improvement")
            improvement.append((baseline - selected) / baseline)
            source = source_rows[task]
            selected_alignment = arm_rows[(task, arm)]
            continuity = continuity_rows[task]
            features["source_p1_filtered_pairwise_mean"].append(
                float(source["p1_filtered_pairwise_mean"]))
            features["arm_best_reference_mean"].append(float(selected_alignment["mean"]))
            features["arm_p1_filtered_pairwise_mean"].append(
                float(selected_alignment["p1_filtered_pairwise_mean"]))
            features["arm_p1_per_window_reference_mean"].append(
                float(selected_alignment["p1_filtered_per_window_mean"]["mean"]))
            features["continuity_mean_history_test_cosine"].append(
                float(continuity["mean_history_test_cosine"]))
            features["continuity_best_reference_mean"].append(float(continuity["mean"]))
        for feature, values in features.items():
            pearson, spearman = correlation(values, improvement)
            correlations.append(dict(
                arm=arm, outcome="relative_mase_improvement_over_fullRand",
                alignment_feature=feature, task_count=len(TASKS),
                pearson_r=pearson, spearman_rho=spearman,
                interpretation="descriptive_only_single_seed_eight_task_association",
            ))
    write_json(output / "alignment-outcome-correlations.json", dict(
        experiment="v1-rage-ads", batch="rolling-eight-task",
        outcome="(fullRand_MASE - arm_MASE) / fullRand_MASE",
        observations=len(TASKS), arms=list(ARMS[1:]),
        correlations=correlations,
        caveat="Eight tasks and one seed per arm; correlations are descriptive and do not establish causation.",
        seed=0,
    ))
    write_json(output / "summary-manifest.json", dict(
        experiment="v1-rage-ads", batch="rolling-eight-task",
        tasks=list(TASKS), arms=list(ARMS), primary_metric="MASE",
        metric_rows=len(rows), elapsed_seconds=time.monotonic() - started,
        artifacts=[str(output / "downstream-metrics.jsonl"),
                   str(output / "alignment-outcome-correlations.json")],
    ))
    (output / ".summary-complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete rows=%d associations=%d elapsed=%s output=%s",
                len(rows), len(correlations), format_duration(time.monotonic() - started), output)


if __name__ == "__main__":
    main()

