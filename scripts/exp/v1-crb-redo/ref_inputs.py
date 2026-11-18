"""Prepare safe reference histories for the twelve TimerXL evaluation tasks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.core import TimeSeriesFragment
from rage.experiments.development_inputs import save_fragments

TASKS = [(f"fev-{i:02d}", "fev", i) for i in (9, 11, 18, 19)] + [
    (f"time-{i:02d}", "time", name) for i, name in enumerate((
        "CPHL/15T", "CPHL/30T", "CPHL/H", "ECDC_COVID/D", "ECDC_COVID/W",
        "epf_electricity_price/H", "Commodity_Import/M", "Commodity_Production/M"), 1)]
ROOT = PROJECT_ROOT
DATA_ROOT = ROOT / "data/ref/shard_dataset/v1-crb-redo-12tasks"


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def fev_rows(index, config_path=None):
    import fev
    from datasets import Dataset
    config = yaml.safe_load((Path(config_path) if config_path is not None else ROOT / "configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml").read_text())
    declaration = config["tasks"][index]
    source = Path(declaration["dataset_path"])
    if not source.is_file() or source.suffix != ".parquet":
        raise FileNotFoundError(source)
    task = fev.Task(**declaration)
    task._full_dataset = Dataset.from_parquet(str(source))
    first = task.get_window(0, num_proc=1)
    past, _ = first.get_input_data()
    targets = tuple(task.target_columns)
    def rows():
        for ordinal, row in enumerate(past.with_format("numpy")):
            entity = str(row[task.id_column])
            for target in targets:
                yield f"{entity}:{target}", np.asarray(row[target], dtype=np.float32), dict(
                    entity=entity, channel=target, ordinal=ordinal, boundary_source="fev_first_window_input")
    return rows(), dict(benchmark="fev", task_index=index, task_name=task.task_name,
        dataset_config=declaration["dataset_config"], source=str(source),
        source_revision="d3e7e720f4158eab03ebcb021f1cce779cd92acc",
        horizon=int(declaration["horizon"]), evaluation_windows=int(declaration["num_windows"]),
        target_columns=list(targets), boundary="first_get_window_0_get_input_data")


def time_rows(name, config_path=None):
    from evaluate.time import _validate_bench_config
    from timebench.evaluation.data import Dataset
    config = yaml.safe_load((Path(config_path) if config_path is not None else ROOT / "configs/eval/time/v1-rage-trev-timerxl.yaml").read_text())
    checked = _validate_bench_config(config["bench_config"])
    matches = [task for task in checked["tasks"] if task["dataset"] == name and task["term"] == "short"]
    if len(matches) != 1:
        raise ValueError("TIME task is not uniquely selected")
    settings = checked["datasets"][name]
    horizon = int(settings["short"]["prediction_length"])
    mode = config["adapter"]["config"]["inference"].get("evaluation_mode", "single_window")
    from evaluate.time import evaluation_test_length
    test_length = evaluation_test_length(int(settings["test_length"]), horizon, mode)
    kwargs = dict(name=name, term="short", prediction_length=horizon, test_length=test_length,
                  val_length=settings.get("val_length", 0), storage_path=checked["dataset_root"])
    native = Dataset(to_univariate=False, **kwargs)
    dataset = native if native.target_dim == 1 else Dataset(to_univariate=True, **kwargs)
    def rows():
        seen = set()
        for ordinal, row in enumerate(dataset.test_data.input):
            values = np.asarray(row["target"], dtype=np.float32)
            if values.ndim != 1:
                raise ValueError("TIME input must be native univariate after evaluation conversion")
            entity = str(row.get("item_id", ordinal))
            if entity in seen:
                continue
            seen.add(entity)
            yield entity, values, dict(entity=entity, ordinal=ordinal,
                boundary_source="time_first_rolling_input_before_test_tail")
    return rows(), dict(benchmark="time", dataset=name, term="short", horizon=horizon,
        source=str(checked["dataset_root"] / name),
        source_revision=checked["dataset_values"]["revision"],
        boundary="evaluation_test_data_input_with_matched_evaluation_test_length",
        evaluation_mode=mode, test_length=test_length,
        to_univariate=bool(native.target_dim != 1))


def prepare_one(task_id, kind, key, logger, config_path=None, data_root=None):
    out = output_path("reference_shard", (Path(data_root) if data_root is not None else DATA_ROOT) / task_id)
    marker, payload = out / "manifest.json", out / "references.npz"
    if marker.is_file() and payload.is_file() and (out / ".complete").is_file():
        logger.info("stage=skip task=%s reason=complete", task_id)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Incomplete reference task directory needs audit: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    logger.info("stage=task-start task=%s kind=%s", task_id, kind)
    rows, source = fev_rows(key, config_path) if kind == "fev" else time_rows(key, config_path)
    fragments, audit, seen = [], [], set()
    for entity, values, fields in rows:
        if entity in seen:
            raise ValueError("Duplicate reference entity/channel: " + entity)
        seen.add(entity)
        if values.ndim != 1:
            raise ValueError("Reference target is not univariate")
        finite = np.isfinite(values)
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
        kept = 0
        for begin, end in zip(edges[::2], edges[1::2]):
            length = int(end - begin)
            if length < 576:
                continue
            fragment = TimeSeriesFragment(values[begin:end].copy(), entity_id=entity,
                                          fragment_id=len(fragments))
            fragments.append(fragment)
            audit.append(dict(**fields, fragment_id=fragment.fragment_id, begin=int(begin),
                              end=int(end), points=length, legal_windows=length - 575))
            kept += 1
        if not kept:
            audit.append(dict(**fields, fragment_id=None, begin=None, end=None,
                              points=int(len(values)), legal_windows=0, skip_reason="no_finite_segment_576"))
    if not fragments:
        raise ValueError("No safe reference windows for task " + task_id)
    partial = out / "references.partial.npz"
    save_fragments(partial, fragments)
    os.replace(partial, payload)
    manifest = dict(task_id=task_id, source=source, extraction="safe_history_v1",
        context_length=480, prediction_length=96, stride=1, fragments=len(fragments),
        legal_windows=sum(item["legal_windows"] for item in audit),
        entity_count=len(seen), segments=audit, answer_values_included=False)
    write_json(marker, manifest)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-finished task=%s entities=%d fragments=%d legal_windows=%d output=%s",
                task_id, len(seen), len(fragments), manifest["legal_windows"], out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[row[0] for row in TASKS] + ["all"], default="all")
    args = parser.parse_args()
    logger = configure_logger("v1_crb_redo_ref_inputs", "v1-crb-redo", "12tasks-ref-grad-prepare", "inputs")
    started = time.monotonic()
    selected = TASKS if args.task == "all" else [row for row in TASKS if row[0] == args.task]
    logger.info("stage=config tasks=%d context=480 prediction=96 boundary=first_exam_input output=%s",
                len(selected), DATA_ROOT)
    for row in selected:
        prepare_one(*row, logger)
    if args.task == "all":
        if any(not (DATA_ROOT / row[0] / ".complete").is_file() for row in TASKS):
            raise RuntimeError("Incomplete safe-reference task set")
        (DATA_ROOT / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete tasks=%d elapsed=%s output=%s",
                len(selected), format_duration(time.monotonic() - started), DATA_ROOT)


if __name__ == "__main__":
    main()
