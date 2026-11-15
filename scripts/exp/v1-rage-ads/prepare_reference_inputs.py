"""Extract leakage-safe rolling reference histories for v1-rage-ads."""
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

ROOT = PROJECT_ROOT
OUT_ROOT = ROOT / "data/ref/shard_dataset/v1-rage-ads-rolling"
FEV_TASKS = ROOT / "configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml"
TIME_TASKS = {
    "time-01": ("CPHL/15T", "configs/eval/time/v1-crb-redo-time-01.yaml"),
    "time-02": ("CPHL/30T", "configs/eval/time/v1-crb-redo-time-02.yaml"),
    "time-03": ("CPHL/H", "configs/eval/time/v1-crb-redo-time-03.yaml"),
    "time-04": ("ECDC_COVID/D", "configs/eval/time/v1-crb-redo-time-04.yaml"),
    "time-06": ("epf_electricity_price/H", "configs/eval/time/v1-crb-redo-time-06.yaml"),
    "time-07": ("Commodity_Import/M", "configs/eval/time/v1-crb-redo-time-07.yaml"),
    "time-08": ("Commodity_Production/M", "configs/eval/time/v1-crb-redo-time-08.yaml"),
}
TASKS = ("fev-09",) + tuple(TIME_TASKS)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def fev_rows():
    import fev
    from datasets import Dataset

    declaration = yaml.safe_load(FEV_TASKS.read_text(encoding="utf-8"))["tasks"][9]
    source = Path(declaration["dataset_path"])
    if not source.is_file() or source.suffix != ".parquet":
        raise FileNotFoundError("FEV local parquet is missing: " + str(source))
    task = fev.Task(**declaration)
    task._full_dataset = Dataset.from_parquet(str(source))
    first_window = task.get_window(0, num_proc=1)
    past, _ = first_window.get_input_data()
    targets = tuple(task.target_columns)

    def rows():
        for ordinal, row in enumerate(past.with_format("numpy")):
            entity = str(row[task.id_column])
            for target in targets:
                yield "%s:%s" % (entity, target), np.asarray(row[target], dtype=np.float32), dict(
                    entity=entity, channel=target, ordinal=ordinal,
                    boundary_source="fev_first_rolling_window_input")

    return rows(), dict(
        benchmark="fev", task_index=9, task_name=task.task_name,
        dataset_config=declaration["dataset_config"], source=str(source),
        source_revision="d3e7e720f4158eab03ebcb021f1cce779cd92acc",
        horizon=int(task.horizon), evaluation_windows=int(task.num_windows),
        target_columns=list(targets),
        boundary="first_declared_rolling_cutoff_input_before_answer",
    )


def time_rows(task_id: str):
    from evaluate.time import _validate_bench_config
    from timebench.evaluation.data import Dataset

    dataset_name, config_name = TIME_TASKS[task_id]
    config = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8"))
    checked = _validate_bench_config(config["bench_config"])
    matches = [item for item in checked["tasks"]
               if item["dataset"] == dataset_name and item["term"] == "short"]
    if len(matches) != 1:
        raise ValueError("TIME task is not uniquely selected")
    settings = checked["datasets"][dataset_name]
    horizon = int(settings["short"]["prediction_length"])
    test_length = int(settings["test_length"])
    kwargs = dict(name=dataset_name, term="short", prediction_length=horizon,
                  test_length=test_length, val_length=settings.get("val_length", 0),
                  storage_path=checked["dataset_root"])
    native = Dataset(to_univariate=False, **kwargs)
    dataset = native if native.target_dim == 1 else Dataset(to_univariate=True, **kwargs)

    def rows():
        for ordinal, item in enumerate(dataset.gluonts_dataset):
            values = np.asarray(item["target"], dtype=np.float32)
            if values.ndim != 1:
                raise ValueError("TIME reference target must be univariate after evaluation conversion")
            safe_end = len(values) - test_length
            if safe_end < 0:
                raise ValueError("TIME test_length exceeds a reference series length")
            entity = str(item.get("item_id", ordinal))
            yield entity, values[:safe_end].copy(), dict(
                entity=entity, ordinal=ordinal,
                boundary_source="pinned_TIME_offset_split_before_full_test_length",
                series_length=int(len(values)), safe_end=int(safe_end),
            )

    return rows(), dict(
        benchmark="time", dataset=dataset_name, term="short", horizon=horizon,
        test_length=test_length, rolling_windows=test_length // horizon,
        source=str(checked["dataset_root"] / dataset_name),
        source_revision=checked["dataset_values"]["revision"],
        to_univariate=bool(native.target_dim != 1),
        boundary="gluonts_dataset_target_before_test_length_offset",
    )


def prepare_one(task_id: str, logger) -> None:
    destination = output_path("reference_shard", OUT_ROOT / task_id)
    manifest_path = destination / "manifest.json"
    payload_path = destination / "references.npz"
    if manifest_path.is_file() and payload_path.is_file() and (destination / ".complete").is_file():
        logger.info("stage=skip task=%s reason=complete", task_id)
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Incomplete reference input directory needs audit: " + str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("stage=task-start task=%s output=%s", task_id, destination)
    rows, source = fev_rows() if task_id == "fev-09" else time_rows(task_id)
    fragments, audit = [], []
    seen = set()
    started = time.monotonic()
    entity_count = 0
    for entity, values, fields in rows:
        if entity in seen:
            raise ValueError("Duplicate reference entity/channel: " + entity)
        seen.add(entity)
        entity_count += 1
        if values.ndim != 1:
            raise ValueError("Reference target is not one-dimensional")
        finite = np.isfinite(values)
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False].astype(np.int8)))
        kept = 0
        for begin, end in zip(edges[::2], edges[1::2]):
            length = int(end - begin)
            if length < 576:
                continue
            fragment = TimeSeriesFragment(
                values[begin:end].copy(), entity_id=entity, fragment_id=len(fragments))
            fragments.append(fragment)
            audit.append(dict(**fields, fragment_id=fragment.fragment_id,
                              begin=int(begin), end=int(end), points=length,
                              legal_windows=length - 575))
            kept += 1
        if not kept:
            audit.append(dict(**fields, fragment_id=None, begin=None, end=None,
                              points=int(len(values)), legal_windows=0,
                              skip_reason="no_finite_segment_576"))
        if entity_count == 1 or entity_count % 100 == 0:
            logger.info("stage=progress task=%s entities=%d fragments=%d elapsed=%s",
                        task_id, entity_count, len(fragments),
                        format_duration(time.monotonic() - started))
    if not fragments:
        raise ValueError("No safe rolling reference windows for task " + task_id)
    partial = destination / "references.partial.npz"
    save_fragments(partial, fragments)
    os.replace(partial, payload_path)
    manifest = dict(
        schema="v1-rage-ads-reference-v1", task_id=task_id, source=source,
        extraction="first_rolling_cutoff_prefix_v1", context_length=480,
        prediction_length=96, stride=1, fragments=len(fragments),
        legal_windows=sum(item["legal_windows"] for item in audit),
        entity_count=entity_count, segments=audit, answer_values_included=False,
    )
    write_json(manifest_path, manifest)
    (destination / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-finished task=%s entities=%d fragments=%d legal_windows=%d output=%s",
                task_id, entity_count, len(fragments), manifest["legal_windows"], destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    args = parser.parse_args()
    selected = TASKS if args.task == "all" else (args.task,)
    logger = configure_logger("v1_rage_ads_reference_inputs", "v1-rage-ads",
                              "rolling-reference-inputs", "prepare")
    started = time.monotonic()
    logger.info("stage=config tasks=%s mode=rolling context=480 prediction=96 output=%s",
                ",".join(selected), OUT_ROOT)
    for task_id in selected:
        prepare_one(task_id, logger)
    if args.task == "all":
        if any(not (OUT_ROOT / task_id / ".complete").is_file() for task_id in TASKS):
            raise RuntimeError("Rolling reference input set is incomplete")
        (OUT_ROOT / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete tasks=%d elapsed=%s output=%s",
                len(selected), format_duration(time.monotonic() - started), OUT_ROOT)


if __name__ == "__main__":
    main()
