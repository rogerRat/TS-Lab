"""Prepare task-specific reference gradients, source selections, and arm configs."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/exp/main-crb-v0"))

import run_directional_time1 as base
from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.core import TimeSeriesFragment
from assets.gradient import GradientCache

BATCH = "q16-src256-eight-task"
REFERENCE_ROOT = ROOT / "data/ref/shard_dataset/v1-crb-redo-12tasks"
CACHE_ROOT = output_path("grad_cache", "outputs/grad_cache/main-crb-v0/" + BATCH)
COMPONENT_ROOT = CACHE_ROOT / "components"
SELECTION_ROOT = output_path("selection", "outputs/selection/main-crb-v0/" + BATCH)
TRAIN_ROOT = output_path("train", "outputs/train/main-crb-v0/" + BATCH)
EVAL_ROOT = output_path("eval", "outputs/eval/main-crb-v0/" + BATCH)
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-v0/eight-task"
PREVIOUS_BATCH = "outputs/selection/main-crb-v0/q16-src256-time1-q8"
TASKS = (
    {"code": "FEV09", "ref": "fev-09", "dataset": "fev-09-safe-reference-v1",
     "label": "FEV task index 9 hospital_admissions_1D",
     "eval": "configs/eval/fev/v1-rage-ads-fev-09.yaml"},
    {"code": "TIME1", "ref": "time-01", "dataset": "time-01-safe-reference-v1",
     "label": "TIME CPHL/15T", "eval": "configs/eval/time/v1-rage-ads-time-01.yaml"},
    {"code": "TIME2", "ref": "time-02", "dataset": "time-02-safe-reference-v1",
     "label": "TIME CPHL/30T", "eval": "configs/eval/time/v1-rage-ads-time-02.yaml"},
    {"code": "TIME3", "ref": "time-03", "dataset": "time-03-safe-reference-v1",
     "label": "TIME CPHL/H", "eval": "configs/eval/time/v1-rage-ads-time-03.yaml"},
    {"code": "TIME4", "ref": "time-04", "dataset": "time-04-safe-reference-v1",
     "label": "TIME ECDC_COVID/D", "eval": "configs/eval/time/v1-rage-ads-time-04.yaml"},
    {"code": "TIME6", "ref": "time-06", "dataset": "time-06-safe-reference-v1",
     "label": "TIME epf_electricity_price/H", "eval": "configs/eval/time/v1-rage-ads-time-06.yaml"},
    {"code": "TIME7", "ref": "time-07", "dataset": "time-07-safe-reference-v1",
     "label": "TIME Commodity_Import/M", "eval": "configs/eval/time/v1-rage-ads-time-07.yaml"},
    {"code": "TIME8", "ref": "time-08", "dataset": "time-08-safe-reference-v1",
     "label": "TIME Commodity_Production/M", "eval": "configs/eval/time/v1-rage-ads-time-08.yaml"},
)
ARMS = ("ori", "crb", "uni", "reft")
TRAIN_ARMS = ("crb", "uni", "reft")
RECIPE = {
    "learning_rate": 0.0001,
    "epochs": 1.0,
    "batch_size": 32,
    "micro_batch_size": 8,
    "weight_decay": 0.0,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.03,
    "schedule": "cosine",
    "min_lr_ratio": 0.0,
    "schedule_epochs": 4000,
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n", encoding="ascii")
    partial.replace(path)


def load_task_reference(task: dict):
    task_root = REFERENCE_ROOT / task["ref"]
    manifest = json.loads((task_root / "manifest.json").read_text(encoding="ascii"))
    if manifest.get("answer_values_included") is not False:
        raise ValueError("safe reference archive includes answer values: " + task["code"])
    if manifest.get("context_length") != base.CONTEXT or manifest.get("prediction_length") != base.HORIZON:
        raise ValueError("reference window contract differs for " + task["code"])
    with np.load(task_root / "references.npz", allow_pickle=False) as archive:
        values, offsets, ids, entities = (archive["values"], archive["offsets"],
                                          archive["ids"], archive["entities"])
    if len(offsets) != len(ids) + 1 or len(ids) != manifest.get("fragments"):
        raise ValueError("reference archive fragment index differs for " + task["code"])
    fragments = [
        TimeSeriesFragment(np.asarray(values[int(offsets[i]):int(offsets[i + 1])]).copy(),
                           entity_id=str(entities[i]), fragment_id=int(ids[i]))
        for i in range(len(ids))
    ]
    counts = np.asarray([len(fragment.valid_starts(base.CONTEXT, base.HORIZON))
                         for fragment in fragments], dtype=np.int64)
    if int(counts.sum()) != int(manifest.get("legal_windows", -1)):
        raise ValueError("reference legal-window count differs for " + task["code"])
    positions = [base.exact_uniform_positions(int(count), base.REFERENCE_Q)
                 for count in counts]
    return manifest, fragments, counts, positions


def install_shared_source_assets(logger) -> dict:
    shared = SELECTION_ROOT / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    old = ROOT / PREVIOUS_BATCH
    mapping = {
        "crb_block_index.npz": old / "crb/crb_block_index.npz",
        "crb_block_directions_fp16.npy": old / "crb/crb_block_directions_fp16.npy",
        "uni_block_index.npz": old / "uni/uni_block_index.npz",
        "uni_block_directions_fp16.npy": old / "uni/uni_block_directions_fp16.npy",
        "crb_source_anchors.npz": old / "crb/source_anchors.npz",
        "uni_source_anchors.npz": old / "uni/source_anchors.npz",
    }
    for name, source in mapping.items():
        if not source.is_file():
            raise FileNotFoundError("shared q16 source asset is missing: " + str(source))
        destination = shared / name
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("refusing to replace existing source asset: " + str(destination))
        destination.symlink_to(source.resolve())
    logger.info("stage=source-assets-ready q=%d anchors_per_method=399044 reused=%s",
                base.SOURCE_Q, shared)
    return {name: str((shared / name).resolve()) for name in mapping}


def materialize_windows(fragments, counts, path: Path, logger, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(counts.sum())
    windows = np.empty((total, base.WINDOW), dtype=np.float32)
    cursor = 0
    offsets = np.arange(base.WINDOW, dtype=np.int64)[None, :]
    started = time.monotonic()
    for fragment, count in zip(fragments, counts):
        starts = fragment.valid_starts(base.CONTEXT, base.HORIZON)
        if len(starts) != int(count):
            raise ValueError("window count changed while materializing " + label)
        for first in range(0, len(starts), 8192):
            part = starts[first:first + 8192]
            stop = cursor + len(part)
            row_ids = (part - 1)[:, None] + offsets
            windows[cursor:stop] = fragment.data[row_ids].astype(np.float32, copy=False)
            cursor = stop
    if cursor != total:
        raise RuntimeError("window materialization count differs for " + label)
    np.savez_compressed(path, windows=windows)
    del windows
    logger.info("stage=reference-windows-complete task=%s windows=%d elapsed=%s path=%s",
                label, total, format_duration(time.monotonic() - started), path)


def prepare_ori_checkpoint_view(task: dict, logger) -> Path:
    ori_root = TRAIN_ROOT / task["code"] / "ori"
    checkpoint_view = ori_root / "checkpoint"
    checkpoint_view.mkdir(parents=True, exist_ok=True)
    for item in base.BASE_MODEL.iterdir():
        target = checkpoint_view / item.name
        if target.exists() or target.is_symlink():
            if target.resolve() != item.resolve():
                raise FileExistsError("ORI checkpoint path conflicts with base model: " + str(target))
            continue
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    manifest_path = checkpoint_view / "rage_checkpoint_manifest.json"
    if not manifest_path.is_file():
        write_json(manifest_path, {
            "schema_version": 1,
            "experiment_version": "main-crb-v0/" + BATCH + "/" + task["code"] + "/ori",
            "model_key": "timerxl-84m",
            "model": "thuml/timer-base-84m",
            "training_step": 0,
            "precision": "bfloat16",
            "training": "not_applicable",
        })
    logger.info("stage=ori-checkpoint-view-ready task=%s checkpoint=%s",
                task["code"], checkpoint_view)
    return checkpoint_view


def write_configs(task: dict, selection_task_root: Path, logger) -> None:
    code = task["code"]
    config_dir = CONFIG_ROOT / code
    config_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = selection_task_root
    eval_dir = EVAL_ROOT / code
    train_dir = TRAIN_ROOT / code
    reference_windows = selection_dir / "reference/training_windows.npz"
    windows_by_arm = {
        "crb": selection_dir / "crb/training_windows.npz",
        "uni": selection_dir / "uni/training_windows.npz",
        "reft": reference_windows,
    }
    for arm in TRAIN_ARMS:
        train_config = {
            "schema_version": 1,
            "run_id": "main-crb-v0-eight-task-" + code.lower() + "-" + arm,
            "model": "timerxl-84m",
            "model_path": str(base.BASE_MODEL),
            "device": "cuda",
            "data": {
                "id": "src-256-" + code.lower() + "-" + arm
                      if arm != "reft" else code.lower() + "-safe-reference-all-windows",
                "windows": str(windows_by_arm[arm]),
                "windows_key": "windows",
                "context_length": base.CONTEXT,
                "prediction_length": base.HORIZON,
            },
            "training": {
                "steps": 4000,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
                "seed": 0,
                "log_every": 100,
                "schedule_total_steps": 4000,
                "recipe": RECIPE,
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "output_dir": str(train_dir / arm),
        }
        train_path = config_dir / (arm + "-train.yaml")
        train_path.write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="ascii")

    ori_checkpoint = prepare_ori_checkpoint_view(task, logger)
    for arm in ARMS:
        checkpoint = ori_checkpoint if arm == "ori" else train_dir / arm / "checkpoint"
        eval_config = {
            "schema_version": 1,
            "checkpoint": str(checkpoint),
            "evaluation_config": task["eval"],
            "output_dir": str(eval_dir / arm),
        }
        eval_path = config_dir / (arm + "-eval.yaml")
        eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="ascii")
        stages = ([{"name": "eval", "config": str(eval_path.relative_to(ROOT))}]
                  if arm == "ori" else [
                      {"name": "train", "config": str((config_dir / (arm + "-train.yaml")).relative_to(ROOT))},
                      {"name": "eval", "config": str(eval_path.relative_to(ROOT))},
                  ])
        pipeline = {
            "schema_version": 1,
            "run_id": "main-crb-v0-eight-task-" + code.lower() + "-" + arm,
            "stages": stages,
        }
        (config_dir / (arm + "-pipeline.yaml")).write_text(
            yaml.safe_dump(pipeline, sort_keys=False), encoding="ascii")
    logger.info("stage=task-configs-ready task=%s config_root=%s", code, config_dir)

def prepare_resources(logger) -> None:
    marker = CACHE_ROOT / ".complete"
    if marker.is_file():
        logger.info("stage=resources-skip reason=complete output=%s", CACHE_ROOT)
        return
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    _, source_fragments, _source_values, _source_offsets, source_counts = base.load_source_fragments()
    source_assets = install_shared_source_assets(logger)
    task_resources = []
    reference_fragments = []
    reference_counts = []
    reference_requests = []
    for task in TASKS:
        manifest, fragments, counts, positions = load_task_reference(task)
        task_resources.append({"task": task, "manifest": manifest,
                               "fragments": fragments, "counts": counts,
                               "positions": positions})
        reference_fragments.extend(fragments)
        reference_counts.extend(counts.tolist())
        reference_requests.append((task, fragments, positions))
        anchors = sum(len(row) for row in positions)
        logger.info("stage=task-reference-ready task=%s fragments=%d legal_windows=%d q8_anchors=%d answer_values_included=%s",
                    task["code"], len(fragments), int(counts.sum()), anchors,
                    manifest["answer_values_included"])

    engine = probe = cache = None
    try:
        engine, probe, identity = base.build_probe(logger)
        os.environ["RAGE_GRADIENT_CACHE_ROOT"] = str(CACHE_ROOT / "raw_bf16")
        cache = GradientCache(
            CACHE_ROOT, probe,
            model_identity={**identity, "probe_checkpoint": str(base.PROBE_ADAPTER),
                            "probe_training_manifest": str(
                                ROOT / "outputs/train/main-ads-v0/checkpoint/training_manifest.json")},
            data_fields={
                "experiment": "main-crb-v0/" + BATCH,
                "source_pool": "src-256",
                "reference_tasks": [task["code"] for task in TASKS],
                "reference_manifests": [str(REFERENCE_ROOT / task["ref"] / "manifest.json")
                                         for task in TASKS],
                "probe_checkpoint": str(base.PROBE_ADAPTER),
                "projection_dimension": base.PROJECTION_DIM,
                "projection_seed": base.PROJECTION_SEED,
            },
            context_length=base.CONTEXT, prediction_length=base.HORIZON)
        for task, fragments, positions in reference_requests:
            logger.info("stage=reference-gradient-start task=%s q=%d anchors=%d",
                        task["code"], base.REFERENCE_Q,
                        sum(len(row) for row in positions))
            base.prepare_cache_positions(cache, task["dataset"], fragments,
                                         positions, logger)
        logger.info("stage=reference-gradient-cache-complete rows=%d computed=%d",
                    len(cache.rows), cache.computed_windows)
        union_keys = base.collect_union_keys(cache)
        base.COMPONENT_ROOT = COMPONENT_ROOT
        row_lookup, directions_path, magnitude_path, ids_path = base.write_direction_components(
            cache, union_keys, source_counts, source_fragments,
            np.asarray(reference_counts, dtype=np.int64), reference_fragments, logger)
        cache.close()
        cache = None
        del engine, probe
        engine = probe = None
        gc.collect()

        anchor_total = 0
        for item in task_resources:
            task = item["task"]
            fragments = item["fragments"]
            counts = item["counts"]
            positions = item["positions"]
            task_root = SELECTION_ROOT / task["code"]
            ref_root = task_root / "reference"
            ref_root.mkdir(parents=True, exist_ok=True)
            ref_rows = []
            for fragment, selected in zip(fragments, positions):
                starts = fragment.valid_starts(base.CONTEXT, base.HORIZON)
                ref_rows.append(np.asarray([
                    row_lookup[(task["dataset"], fragment.fragment_id,
                                int(starts[position]))]
                    for position in selected
                ], dtype=np.int64))
            np.savez_compressed(
                ref_root / "reference_anchors.npz",
                fragment_counts=counts,
                positions=np.concatenate(positions),
                gradient_rows=np.concatenate(ref_rows),
            )
            pairs = [[(int(left), int(right))
                      for left, right in zip(selected[:-1], selected[1:])]
                     for selected in positions]
            block_index, block_directions, block_count = base.write_block_assets(
                "reference", fragments, positions, row_lookup, directions_path,
                pairs, task["dataset"], ref_root, logger)
            materialize_windows(
                fragments, counts, ref_root / "training_windows.npz",
                logger, task["code"])
            reference_manifest = {
                "experiment": "main-crb-v0",
                "batch": BATCH,
                "task": task["code"],
                "reference_task": task["label"],
                "reference_dataset": task["dataset"],
                "reference_segments": len(fragments),
                "legal_window_count": int(counts.sum()),
                "anchor_interval": base.REFERENCE_Q,
                "anchor_count": int(sum(len(row) for row in positions)),
                "block_count": block_count,
                "training_windows": str(ref_root / "training_windows.npz"),
                "answer_values_included": False,
                "manifest": str(REFERENCE_ROOT / task["ref"] / "manifest.json"),
                "block_index": str(block_index),
                "block_directions": str(block_directions),
            }
            write_json(ref_root / "selection.json", reference_manifest)
            (ref_root / ".complete").write_text("complete\n", encoding="ascii")
            item["resource_manifest"] = reference_manifest
            anchor_total += reference_manifest["anchor_count"]
            logger.info("stage=task-reference-assets-complete task=%s anchors=%d blocks=%d legal_windows=%d",
                        task["code"], reference_manifest["anchor_count"], block_count,
                        reference_manifest["legal_window_count"])

        manifest = {
            "experiment": "main-crb-v0",
            "batch": BATCH,
            "source_pool": "src-256",
            "source_window_count": int(source_counts.sum()),
            "source_q": base.SOURCE_Q,
            "source_crb_anchor_count": 399044,
            "source_uni_anchor_count": 399044,
            "source_anchor_policy": "shared task-independent q16 source anchor assets; each task recomputes direction scores and selected top-10-percent source windows",
            "reference_q": base.REFERENCE_Q,
            "task_count": len(TASKS),
            "reference_anchor_count": anchor_total,
            "reference_tasks": [item["resource_manifest"] for item in task_resources],
            "unique_reference_gradient_count": len(union_keys),
            "projection_dimension": base.PROJECTION_DIM,
            "projection_seed": base.PROJECTION_SEED,
            "direction_dtype": "float16 unit vectors",
            "magnitude_dtype": "float32 L2 norm; audit only",
            "shared_source_assets": source_assets,
            "direction_sidecar": str(directions_path),
            "magnitude_sidecar": str(magnitude_path),
            "gradient_ids": str(ids_path),
            "training_steps": 4000,
            "effective_batch_size": 32,
            "micro_batch_size": 8,
            "seed": 0,
            "selection_fraction": base.TRAIN_COUNT_FRACTION,
        }
        write_json(CACHE_ROOT / "resource_manifest.json", manifest)
        write_configs_for_all(task_resources, logger)
        (marker.parent / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=resources-complete tasks=%d reference_anchors=%d unique_gradients=%d output=%s",
                    len(TASKS), anchor_total, len(union_keys), CACHE_ROOT)
    finally:
        if cache is not None:
            cache.close()
        if probe is not None or engine is not None:
            del probe, engine
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def write_configs_for_all(task_resources, logger) -> None:
    for item in task_resources:
        write_configs(item["task"], SELECTION_ROOT / item["task"]["code"], logger)


def prepare_selection(task: dict, logger) -> None:
    task_root = SELECTION_ROOT / task["code"]
    marker = task_root / ".complete"
    if marker.is_file():
        logger.info("stage=selection-skip task=%s reason=complete", task["code"])
        return
    global_manifest_path = CACHE_ROOT / "resource_manifest.json"
    if not global_manifest_path.is_file():
        raise FileNotFoundError("eight-task reference resources are not prepared")
    with np.load(REFERENCE_ROOT / task["ref"] / "references.npz", allow_pickle=False) as archive:
        values, offsets, ids, entities = (archive["values"], archive["offsets"],
                                          archive["ids"], archive["entities"])
    fragments = [
        TimeSeriesFragment(np.asarray(values[int(offsets[i]):int(offsets[i + 1])]).copy(),
                           entity_id=str(entities[i]), fragment_id=int(ids[i]))
        for i in range(len(ids))
    ]
    counts = np.asarray([len(fragment.valid_starts(base.CONTEXT, base.HORIZON))
                         for fragment in fragments], dtype=np.int64)
    positions = [base.exact_uniform_positions(int(count), base.REFERENCE_Q)
                 for count in counts]
    ref_root = task_root / "reference"
    reference_directions_path = ref_root / "reference_block_directions_fp16.npy"
    if not reference_directions_path.is_file():
        raise FileNotFoundError("task q8 reference directions are missing: " + task["code"])
    _rows, source_fragments, source_values, source_offsets, source_counts = base.load_source_fragments()
    source_total = int(source_counts.sum())
    task_manifest = json.loads(global_manifest_path.read_text(encoding="ascii"))
    if task_manifest.get("source_window_count") != source_total:
        raise ValueError("src-256 source window count differs from the registered pool")
    base.SELECTION_ROOT = task_root
    base.SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    shared = SELECTION_ROOT / "shared"
    started = time.monotonic()
    for method in ("crb", "uni"):
        method_root = task_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        source_anchor = shared / (method + "_source_anchors.npz")
        anchor_link = method_root / "source_anchors.npz"
        if not anchor_link.exists() and not anchor_link.is_symlink():
            anchor_link.symlink_to(source_anchor.resolve())
        block_index = shared / (method + "_block_index.npz")
        block_directions = shared / (method + "_block_directions_fp16.npy")
        logger.info("stage=task-source-selection-start task=%s method=%s source_windows=%d budget=%d reference_blocks=%d",
                    task["code"], method, source_total,
                    int(math.floor(base.TRAIN_COUNT_FRACTION * source_total)),
                    int(np.load(reference_directions_path, mmap_mode="r").shape[0]))
        selection_manifest = base.score_and_select(
            method, source_fragments, source_counts, source_values, source_offsets,
            block_index, block_directions, reference_directions_path, logger)
        selection_manifest["batch"] = BATCH
        selection_manifest["task"] = task["code"]
        selection_manifest["reference_task"] = task["label"]
        selection_manifest["score"] = "max cosine to task-specific reference block directions"
        write_json(method_root / "selection.json", selection_manifest)
    (marker.parent / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-selection-complete task=%s elapsed=%s output=%s",
                task["code"], format_duration(time.monotonic() - started), task_root)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("resources", "select"), required=True)
    parser.add_argument("--task", choices=tuple(task["code"] for task in TASKS))
    args = parser.parse_args(argv)
    logger = configure_logger(
        "main_crb_v0_eight_task_prepare", "main-crb-v0", BATCH,
        "prepare-" + args.stage + ("-" + args.task.lower() if args.task else ""),
        enable_heartbeat=True)
    logger.info("stage=config stage=%s task=%s tasks=%s source_q=%d reference_q=%d",
                args.stage, args.task or "all", ",".join(task["code"] for task in TASKS),
                base.SOURCE_Q, base.REFERENCE_Q)
    if args.stage == "resources":
        prepare_resources(logger)
    else:
        if not args.task:
            raise ValueError("--task is required when --stage=select")
        task = next(row for row in TASKS if row["code"] == args.task)
        prepare_selection(task, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
