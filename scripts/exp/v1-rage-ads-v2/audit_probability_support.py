#!/usr/bin/env python3
"""Audit source-window probability support for v1-rage-ads-v2."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.ads import SourceProbabilitySampler
from experiment_config import ARMS, TASKS

ROOT = PROJECT_ROOT
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads-v2"
OUTPUT_ROOT = SELECTION_ROOT / "audit/probability-support"
SOURCE_CONFIG = ROOT / "configs/data/src-256.yaml"
SOURCE_MANIFEST = ROOT / "data/src/shard_dataset/src-256/manifest.json"
WINDOW_LENGTH = 576


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


def source_legal_counts() -> np.ndarray:
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="ascii"))
    lengths = np.asarray([int(row["length"]) for row in config["trajectories"]], dtype=np.int64)
    legal = lengths - WINDOW_LENGTH + 1
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="ascii"))
    if (len(legal) != 256 or np.any(legal <= 0)
            or int(manifest.get("trajectory_count", -1)) != 256
            or int(manifest.get("total_points", -1)) != int(lengths.sum())):
        raise ValueError("src-256 population differs from its manifest")
    return legal


def audit_cell(task: str, arm: str, legal: np.ndarray) -> dict:
    directory = SELECTION_ROOT / task / arm
    if not (directory / ".complete").is_file():
        raise FileNotFoundError("selection arm is incomplete: %s/%s" % (task, arm))
    manifest = json.loads((directory / "sampling.json").read_text(encoding="ascii"))
    mode = manifest.get("mode")
    population = int(legal.sum())
    if arm == "reft":
        if mode != "reference_uniform" or int(manifest.get("candidate_count", -1)) <= 0:
            raise ValueError("reference-only sampler metadata is invalid")
        return dict(task=task, arm=arm, applicable=False, source_population=population,
                    candidate_count=int(manifest["candidate_count"]), zero_probability_count=None,
                    zero_probability_fraction=None, mode=mode)
    if int(manifest.get("candidate_count", -1)) != population:
        raise ValueError("sampler population differs from src-256")
    if mode == "uniform_all":
        zero_count = 0
        support_mode = mode
    elif mode == "probability_descriptor":
        descriptor_path = directory / manifest["descriptor_path"]
        sampler = SourceProbabilitySampler(json.loads(descriptor_path.read_text(encoding="ascii")))
        if (int(sampler.offsets[-1]) != population
                or not np.array_equal(sampler.fragment_ids, np.arange(len(legal)))
                or not np.array_equal(sampler.counts, legal)):
            raise ValueError("probability descriptor differs from src-256")
        zero_count = population - int(len(sampler.support_rows))
        support_mode = sampler.mode
    elif mode == "trajectory_uniform":
        ids = np.asarray(manifest["fragment_ids"], dtype=np.int64)
        probabilities = np.asarray(manifest["trajectory_probabilities"], dtype=np.float64)
        if (ids.ndim != 1 or probabilities.shape != ids.shape or not len(ids)
                or np.any(ids < 0) or np.any(ids >= len(legal))
                or len(np.unique(ids)) != len(ids) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("trajectory-uniform probabilities are invalid")
        positive = set(map(int, ids[probabilities > 0]))
        zero_count = sum(int(legal[index]) for index in range(len(legal)) if index not in positive)
        support_mode = mode
    else:
        raise ValueError("unsupported source sampler mode: " + str(mode))
    if zero_count < 0 or zero_count > population:
        raise ValueError("zero-probability count is outside source population")
    return dict(task=task, arm=arm, applicable=True, source_population=population,
                candidate_count=population, zero_probability_count=zero_count,
                zero_probability_fraction=zero_count / population,
                positive_probability_count=population - zero_count, support_mode=support_mode,
                mode=mode)


def main() -> None:
    started = time.monotonic()
    logger = configure_logger("v1_rage_ads_v2_probability_audit", "v1-rage-ads-v2",
                              "probability-support-audit", "audit")
    output = output_path("selection", OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    legal = source_legal_counts()
    logger.info("stage=config tasks=%d arms=%d source_windows=%d", len(TASKS), len(ARMS),
                int(legal.sum()))
    rows = []
    for task in TASKS:
        for arm in ARMS:
            row = audit_cell(task, arm, legal)
            rows.append(row)
            logger.info("stage=progress task=%s arm=%s applicable=%s zero=%s/%s",
                        task, arm, row["applicable"], row["zero_probability_count"],
                        row["source_population"])
    rows_path = output / "probability-support.jsonl"
    temporary = rows_path.with_name(rows_path.name + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                 for row in rows), encoding="ascii")
    os.replace(temporary, rows_path)
    by_arm = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm and row["applicable"]]
        by_arm[arm] = (dict(applicable=False) if not selected else dict(
            applicable=True, task_count=len(selected),
            zero_probability_count=sum(row["zero_probability_count"] for row in selected),
            source_population=sum(row["source_population"] for row in selected),
            zero_probability_fraction=(sum(row["zero_probability_count"] for row in selected)
                                       / sum(row["source_population"] for row in selected))))
    write_json(output / "summary.json", dict(
        experiment="v1-rage-ads-v2", phase="source-probability-support-audit",
        tasks=list(TASKS), arms=list(ARMS), rows=len(rows), by_arm=by_arm,
        definition="exact count of legal src-256 windows assigned zero probability",
        rows_path=str(rows_path), elapsed_seconds=time.monotonic() - started))
    (output / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete rows=%d elapsed=%s output=%s", len(rows),
                format_duration(time.monotonic() - started), output)


if __name__ == "__main__":
    main()
