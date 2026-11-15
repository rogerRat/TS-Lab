"""Count exact zero-probability source windows for each rolling training arm."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.ads import SourceProbabilitySampler

ROOT = PROJECT_ROOT
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ARMS = ("fullRand", "rage", "ancUni", "Traj")
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling"
SOURCE_CONFIG = ROOT / "configs/data/src-96.yaml"
SOURCE_MANIFEST = ROOT / "data/src/shard_dataset/src-96/manifest.json"
OUTPUT_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling/audit/probability-support"
WINDOW_LENGTH = 576
EXPECTED_POPULATION = 2666312


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


def load_legal_counts() -> np.ndarray:
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="ascii"))
    lengths = np.asarray([int(row["length"]) for row in config["trajectories"]],
                         dtype=np.int64)
    legal = lengths - WINDOW_LENGTH + 1
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="ascii"))
    if (len(legal) != 96 or np.any(legal <= 0) or int(legal.sum()) != EXPECTED_POPULATION
            or manifest.get("name") != "src-96"
            or int(manifest.get("trajectory_count", -1)) != 96):
        raise ValueError("src-96 source population differs from the registered pool")
    return legal


def count_zero_probabilities(task: str, arm: str, legal: np.ndarray) -> dict:
    directory = SELECTION_ROOT / task / arm
    if not (directory / ".complete").is_file():
        raise FileNotFoundError("selection arm is incomplete: %s/%s" % (task, arm))
    sampling = json.loads((directory / "sampling.json").read_text(encoding="ascii"))
    population = int(legal.sum())
    if int(sampling.get("candidate_count", -1)) != population:
        raise ValueError("sampler population differs from src-96")
    mode = sampling.get("mode")

    if mode == "uniform_all":
        zero_count = 0
        support_mode = "uniform_all"
    elif mode == "probability_descriptor":
        descriptor_path = directory / sampling["descriptor_path"]
        descriptor = json.loads(descriptor_path.read_text(encoding="ascii"))
        sampler = SourceProbabilitySampler(descriptor)
        if (int(sampler.offsets[-1]) != population
                or not np.array_equal(sampler.fragment_ids, np.arange(len(legal)))
                or not np.array_equal(sampler.counts, legal)):
            raise ValueError("probability descriptor does not match src-96 trajectory counts")
        zero_count = (0 if sampler.mode == "uniform"
                      else population - int(len(sampler.support_rows)))
        support_mode = str(sampler.mode)
    elif mode == "trajectory_uniform":
        ids = np.asarray(sampling["fragment_ids"], dtype=np.int64)
        probabilities = np.asarray(sampling["trajectory_probabilities"], dtype=np.float64)
        if (ids.ndim != 1 or probabilities.shape != ids.shape or not len(ids)
                or np.any(ids < 0) or np.any(ids >= len(legal))
                or len(np.unique(ids)) != len(ids)
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("trajectory probabilities are invalid")
        positive_ids = set(int(ids[index]) for index in np.flatnonzero(probabilities > 0))
        zero_count = sum(int(legal[index]) for index in range(len(legal))
                         if index not in positive_ids)
        support_mode = "trajectory_uniform"
    else:
        raise ValueError("unsupported sampling mode: " + str(mode))

    if zero_count < 0 or zero_count > population:
        raise ValueError("zero-probability support count is outside the source pool")
    positive_count = population - zero_count
    return dict(
        task=task,
        arm=arm,
        candidate_count=population,
        zero_probability_count=zero_count,
        zero_probability_fraction=zero_count / population,
        positive_probability_count=positive_count,
        positive_probability_fraction=positive_count / population,
        sampler_mode=mode,
        support_mode=support_mode,
        calculation="exact_source_window_support_count",
    )


def main() -> None:
    started = time.monotonic()
    logger = configure_logger("v1_rage_ads_probability_support", "v1-rage-ads",
                              "ads-probability-support", "zero-probability-audit")
    output = output_path("selection", OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    legal = load_legal_counts()
    logger.info("stage=config tasks=%d arms=%d candidate_windows=%d statistic=exact_zero_probability_fraction",
                len(TASKS), len(ARMS), int(legal.sum()))

    rows = []
    for index, task in enumerate(TASKS):
        for arm in ARMS:
            row = count_zero_probabilities(task, arm, legal)
            rows.append(row)
            logger.info("stage=progress task=%s arm=%s zero=%d/%d fraction=%.8f cells=%d/32 eta=%s",
                        task, arm, row["zero_probability_count"], row["candidate_count"],
                        row["zero_probability_fraction"], len(rows),
                        format_duration((time.monotonic() - started)
                                        * (32 - len(rows)) / len(rows)))

    rows_path = output / "probability-support.jsonl"
    temporary = rows_path.with_name(rows_path.name + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                         for row in rows), encoding="ascii")
    os.replace(temporary, rows_path)
    by_arm = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        zero_total = sum(row["zero_probability_count"] for row in selected)
        candidate_total = sum(row["candidate_count"] for row in selected)
        fractions = [row["zero_probability_fraction"] for row in selected]
        by_arm[arm] = dict(
            task_count=len(selected),
            zero_probability_count=zero_total,
            candidate_count=candidate_total,
            zero_probability_fraction=zero_total / candidate_total,
            task_fraction_min=min(fractions),
            task_fraction_max=max(fractions),
        )
    summary = dict(
        experiment="v1-rage-ads",
        batch="rolling-eight-task",
        tasks=list(TASKS),
        arms=list(ARMS),
        definition="fraction of all legal src-96 window starts assigned exactly zero probability",
        source_windows_per_task=int(legal.sum()),
        rows=len(rows),
        aggregation="exact counts; task-weighted equally because each task uses the same source pool",
        by_arm=by_arm,
        rows_path=str(rows_path),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "summary.json", summary)
    (output / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete rows=%d rage_mean_zero=%.6f ancUni_mean_zero=%.6f output=%s elapsed=%s",
                len(rows), by_arm["rage"]["zero_probability_fraction"],
                by_arm["ancUni"]["zero_probability_fraction"], output,
                format_duration(time.monotonic() - started))


if __name__ == "__main__":
    main()
