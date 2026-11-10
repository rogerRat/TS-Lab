#!/usr/bin/env python
"""Train and evaluate the two Chronos-T5-Base Trev batches sequentially."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.experiments.data import load_bundle_index, _read_series
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora
from rage.experiments.regions import RegionWindows
from rage.experiments.train_eval import TrainingRecipe, fine_tune

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CHILD = None
LOGGER = None


def _signal_handler(signum, _frame):
    global ACTIVE_CHILD
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        os.killpg(ACTIVE_CHILD.pid, signal.SIGINT)
        ACTIVE_CHILD.wait()
    raise KeyboardInterrupt("received signal %s" % signum)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(path)


def _source_rows(config: dict) -> tuple[list[dict], dict[str, object]]:
    descriptors = []
    counts = {}
    bundles = {}
    for item in config["candidate_sources"]:
        key = "%s/%s" % (item["source"], item["subset"])
        bundle = load_bundle_index(ROOT / item["load_dir"])
        bundles[key] = bundle
        source_count = 0
        for row in range(bundle.series_count()):
            if bundle.meta["dataset"][row] != item["source"] or bundle.meta["subset"][row] != item["subset"]:
                continue
            values = _read_series(bundle, row)
            if len(values) < config["window_length"] or not np.isfinite(values).all() or float(np.std(values)) == 0.0:
                continue
            descriptors.append({"source_key": key, "load_dir": item["load_dir"], "row": row,
                                "series_id": str(bundle.meta["series_id"][row]), "length": int(len(values))})
            source_count += 1
        counts[key] = {"eligible_series": source_count,
                       "eligible_points": sum(d["length"] for d in descriptors if d["source_key"] == key)}
        LOGGER.info("stage=pool-scan source=%s eligible_series=%d eligible_points=%d",
                    key, source_count, counts[key]["eligible_points"])
    if not descriptors:
        raise ValueError("no eligible source series")
    return descriptors, bundles, counts


def _pool_contract(config: dict) -> dict:
    return {"experiment_version": config["experiment_version"],
            "candidate_sources": config["candidate_sources"],
            "random_fraction": config["random_fraction"],
            "sampling_seed": config["sampling_seed"],
            "window_length": config["window_length"],
            "eligibility": "length>=window_length; finite values; non-constant series",
            "sampling_unit": "series"}


def prepare_pool(config: dict, phase_dir: Path) -> tuple[RegionWindows, dict]:
    input_dir = phase_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = input_dir / "input_manifest.json"
    contract = _pool_contract(config)
    names = [input_dir / "values.npy", input_dir / "offsets.npy", input_dir / "regions.npy"]
    if manifest_path.exists() and all(path.exists() for path in names):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise ValueError("input manifest differs from the frozen phase config")
        values = np.load(names[0], mmap_mode="r")
        offsets = np.load(names[1], mmap_mode="r")
        regions = np.load(names[2], mmap_mode="r")
        windows = RegionWindows(values, offsets, regions, window_length=config["window_length"], budget_windows=4000 * config["batch_size"])
        windows.training_contract = {"experiment_version": config["experiment_version"],
                                     "input_manifest_path": str(manifest_path),
                                     "selected_series": manifest["selected_series"],
                                     "selected_points": manifest["selected_points"]}
        return windows, manifest

    for path in input_dir.glob("*.partial"):
        path.unlink()
    descriptors, bundles, counts = _source_rows(config)
    rng = np.random.default_rng(config["sampling_seed"])
    fraction = float(config["random_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("random_fraction must be in (0, 1]")
    selected_count = len(descriptors) if fraction == 1 else max(1, int(np.floor(fraction * len(descriptors))))
    selected_indices = np.sort(rng.choice(len(descriptors), size=selected_count, replace=False))
    selected = [descriptors[int(index)] for index in selected_indices]
    total_points = sum(item["length"] for item in selected)
    values_tmp = input_dir / "values.npy.partial"
    values_map = np.lib.format.open_memmap(values_tmp, mode="w+", dtype=np.float32, shape=(total_points,))
    offsets = np.empty(len(selected) + 1, dtype=np.int64)
    offsets[0] = 0
    source_rows = []
    position = 0
    for index, item in enumerate(selected):
        bundle = bundles[item["source_key"]]
        values = _read_series(bundle, item["row"])
        end = position + item["length"]
        values_map[position:end] = values
        offsets[index + 1] = end
        source_rows.append({"source": item["source_key"], "row": item["row"],
                            "series_id": item["series_id"], "length": item["length"]})
        position = end
    values_map.flush()
    del values_map
    offsets_tmp = input_dir / "offsets.npy.partial"
    regions_tmp = input_dir / "regions.npy.partial"
    with offsets_tmp.open("wb") as stream:
        np.save(stream, offsets)
    regions = np.column_stack((np.arange(len(selected), dtype=np.int64),
                               np.zeros(len(selected), dtype=np.int64),
                               np.asarray([item["length"] for item in selected], dtype=np.int64)))
    with regions_tmp.open("wb") as stream:
        np.save(stream, regions)
    values_tmp.replace(names[0])
    offsets_tmp.replace(names[1])
    regions_tmp.replace(names[2])
    actual_by_source = {}
    for item in selected:
        row = actual_by_source.setdefault(item["source_key"], {"series": 0, "points": 0})
        row["series"] += 1
        row["points"] += item["length"]
    manifest = {"contract": contract, "candidate_series": len(descriptors),
                "selected_series": len(selected), "candidate_points": sum(d["length"] for d in descriptors),
                "selected_points": total_points, "source_counts": counts,
                "selected_source_counts": actual_by_source, "selected_series_manifest": source_rows,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _write_json(manifest_path, manifest)
    LOGGER.info("stage=pool-ready phase=%s series=%d points=%d windows=%d",
                config["experiment_version"], len(selected), total_points,
                int(np.sum(np.diff(offsets) - config["window_length"] + 1)))
    windows = RegionWindows(np.load(names[0], mmap_mode="r"), offsets, regions,
                            window_length=config["window_length"],
                            budget_windows=4000 * config["batch_size"])
    windows.training_contract = {"experiment_version": config["experiment_version"],
                                 "input_manifest_path": str(manifest_path),
                                 "selected_series": manifest["selected_series"],
                                 "selected_points": manifest["selected_points"]}
    return windows, manifest


def save_adapter(engine, destination: Path, step: int, config: dict) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.mkdir(parents=True)
    engine.model.save_pretrained(temporary, safe_serialization=True)
    _write_json(temporary / "rage_adapter_manifest.json", {
        "schema_version": 1, "experiment_version": config["experiment_version"],
        "model": "amazon/chronos-t5-base", "training_step": step,
        "precision": "bfloat16", "lora": config["lora"]})
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def train_phase(config: dict, config_path: Path) -> Path:
    phase_dir = output_path("train", Path("outputs/train") / config["experiment_version"])
    phase_dir.mkdir(parents=True, exist_ok=True)
    complete = phase_dir / ".complete"
    if complete.exists():
        LOGGER.info("stage=phase-skip already complete: %s", config["experiment_version"])
        return phase_dir
    windows, input_manifest = prepare_pool(config, phase_dir)
    LOGGER.info("stage=model-load-start model=Chronos-T5-Base dtype=bfloat16")
    engine, _probe, identity = build_model("chronos-base", model_path=config["model_path"],
        device="cuda", context_length=config["context_length"], prediction_length=config["prediction_length"])
    adapter = apply_student_lora(engine, "chronos-base", config["lora"])
    if any(parameter.dtype != torch.bfloat16 for parameter in engine.model.parameters()):
        raise TypeError("all base and LoRA parameters must remain bfloat16")
    trainable = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.bfloat16 for parameter in trainable):
        raise TypeError("all trainable parameters must be bfloat16")
    LOGGER.info("stage=model-ready model=%s parameters=%d trainable=%d dtype=bfloat16",
                identity["model_id"], identity["parameter_count"], adapter["trainable_parameters"])
    recipe = TrainingRecipe(learning_rate=config["learning_rate"], epochs=1.0,
        batch_size=config["batch_size"], micro_batch_size=config["batch_size"],
        weight_decay=config["weight_decay"], betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"], max_grad_norm=config["optimizer"]["max_grad_norm"],
        warmup_ratio=config["optimizer"]["warmup_ratio"], schedule=config["optimizer"]["schedule"],
        schedule_epochs=1.0)
    checkpoints = [int(step) for step in config["step_checkpoints"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0 or checkpoints[-1] != 4000:
        raise ValueError("step checkpoints must be the fixed ordered 0..4000 schedule")
    adapter_root = phase_dir / "checkpoints"
    adapter_root.mkdir(parents=True, exist_ok=True)
    for step in checkpoints[1:]:
        target = adapter_root / ("adapter-step-%04d" % step)
        if not target.exists():
            pass
    start = time.monotonic()
    last_log = start
    def on_step(step: int) -> None:
        nonlocal last_log
        if step in checkpoints[1:]:
            save_adapter(engine, adapter_root / ("adapter-step-%04d" % step), step, config)
            LOGGER.info("stage=checkpoint step=%d adapter=%s", step, adapter_root / ("adapter-step-%04d" % step))
        now = time.monotonic()
        if step in checkpoints[1:] or now - last_log >= 30:
            rate = step / max(now - start, 1e-6)
            eta = (4000 - step) / rate if rate else None
            LOGGER.info("stage=train-progress step=%d/4000 eta=%s", step, format_duration(eta))
            last_log = now
    LOGGER.info("stage=train-start phase=%s steps=4000 batch=%d lr=%g precision=bfloat16",
                config["experiment_version"], config["batch_size"], config["learning_rate"])
    result = fine_tune(engine, windows, steps=4000, batch_size=config["batch_size"],
        learning_rate=config["learning_rate"], seed=config["sampling_seed"], on_step=on_step,
        resume_path=adapter_root / "resume.pt", checkpoint_steps=checkpoints[1:], recipe=recipe,
        schedule_total_steps=4000)
    if result.get("steps") != 4000:
        _write_json(phase_dir / "training_partial.json", result)
        raise RuntimeError("training stopped before 4,000 optimizer updates")
    training_manifest = {"experiment_version": config["experiment_version"],
        "config": str(config_path), "model": identity, "lora": adapter,
        "precision": "bfloat16", "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"], "context_length": config["context_length"],
        "prediction_length": config["prediction_length"], "window_length": config["window_length"],
        "step_checkpoints": checkpoints, "training_result": result,
        "input_manifest": str(phase_dir / "inputs/input_manifest.json")}
    _write_json(phase_dir / "training_manifest.json", training_manifest)
    LOGGER.info("stage=train-finished phase=%s elapsed=%s", config["experiment_version"], format_duration(result["cumulative_training_seconds"]))
    del engine, _probe
    torch.cuda.empty_cache()
    evaluate_phase(config, adapter_root, phase_dir)
    expected = set(checkpoints)
    completed = set()
    for step in expected:
        eval_dir = output_path("eval", Path("outputs/eval") / config["experiment_version"] / ("step-%04d" % step))
        fev = eval_dir / "fev/fev_results.csv"
        time_meta = eval_dir / "time/time_metadata.json"
        if fev.exists() and time_meta.exists():
            metadata = json.loads(time_meta.read_text())
            if metadata.get("completed_at") and len(fev.read_text().splitlines()) == 5:
                completed.add(step)
    if completed != expected:
        raise RuntimeError("evaluation incomplete; completed steps=%s" % sorted(completed))
    complete.write_text(json.dumps({"completed_steps": sorted(completed), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n")
    LOGGER.info("stage=phase-complete phase=%s output=%s", config["experiment_version"], phase_dir)
    return phase_dir


def _export_checkpoint(config: dict, adapter_dir: Path, step: int, destination: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, GenerationConfig
    base = AutoModelForSeq2SeqLM.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, local_files_only=True)
    wrapped = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = wrapped.merge_and_unload()
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.mkdir(parents=True)
    merged.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    GenerationConfig.from_pretrained(config["model_path"], local_files_only=True).save_pretrained(temporary)
    _write_json(temporary / "rage_checkpoint_manifest.json", {
        "schema_version": 1, "experiment_version": config["experiment_version"],
        "model": "amazon/chronos-t5-base", "training_step": step,
        "precision": "bfloat16", "reference_task_record": {"benchmark": "fev", "task_indices": [9, 11, 18, 19]}})
    temporary.replace(destination)
    del base, wrapped, merged




def _base_eval_checkpoint(config: dict, adapter_root: Path) -> Path:
    destination = adapter_root / "base-step-0000"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "generation_config.json", "model.safetensors"):
        target = destination / filename
        source = Path(config["model_path"]) / filename
        if not target.exists():
            target.symlink_to(source)
    manifest = destination / "rage_checkpoint_manifest.json"
    if not manifest.exists():
        _write_json(manifest, {"schema_version": 1,
            "experiment_version": config["experiment_version"],
            "model": "amazon/chronos-t5-base", "training_step": 0,
            "precision": "bfloat16",
            "reference_task_record": {"benchmark": "fev", "task_indices": [9, 11, 18, 19]}})
    return destination

def evaluate_phase(config: dict, adapter_root: Path, phase_dir: Path) -> None:
    global ACTIVE_CHILD
    eval_root = output_path("eval", Path("outputs/eval") / config["experiment_version"])
    for step in config["step_checkpoints"]:
        step = int(step)
        checkpoint = _base_eval_checkpoint(config, adapter_root) if step == 0 else adapter_root / ("adapter-step-%04d" % step)
        export_path = None
        if step:
            export_path = phase_dir / ("export-step-%04d" % step)
            LOGGER.info("stage=checkpoint-export-start step=%d", step)
            _export_checkpoint(config, checkpoint, step, export_path)
            checkpoint = export_path
        step_dir = eval_root / ("step-%04d" % step)
        for benchmark, eval_config in (("fev", config["evaluation_configs"][0]),
                                       ("time", config["evaluation_configs"][1])):
            output_dir = step_dir / benchmark
            expected = (output_dir / "fev_results.csv" if benchmark == "fev"
                        else output_dir / "time_metadata.json")
            if expected.exists() and (benchmark == "fev" or json.loads(expected.read_text()).get("completed_at")):
                LOGGER.info("stage=evaluation-skip step=%d benchmark=%s already complete", step, benchmark)
                continue
            command = [sys.executable, "-u", "-m", "evaluate.run", "--checkpoint", str(checkpoint),
                       "--config", eval_config, "--output-dir", str(output_dir)]
            if output_dir.exists():
                command.append("--resume")
            env = dict(os.environ)
            env["PYTHONPATH"] = "src:third_parties/TIME/src:." + os.pathsep + env.get("PYTHONPATH", "")
            LOGGER.info("stage=evaluation-start step=%d benchmark=%s tasks=%d", step, benchmark, 4 if benchmark == "fev" else 8)
            ACTIVE_CHILD = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
            return_code = ACTIVE_CHILD.wait()
            ACTIVE_CHILD = None
            if return_code:
                raise RuntimeError("evaluation failed at step %d benchmark %s with exit %d" % (step, benchmark, return_code))
            LOGGER.info("stage=evaluation-finished step=%d benchmark=%s", step, benchmark)
        if export_path is not None:
            shutil.rmtree(export_path)
            torch.cuda.empty_cache()


def main() -> int:
    global LOGGER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-config", type=Path, default=Path("configs/rage/v1-rage-chronosT5-trev-base-rand.json"))
    parser.add_argument("--full-config", type=Path, default=Path("configs/rage/v1-rage-chronosT5-trev-base-full.json"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    random_config_path = args.random_config.resolve()
    full_config_path = args.full_config.resolve()
    random_config = json.loads(random_config_path.read_text())
    full_config = json.loads(full_config_path.read_text())
    LOGGER = configure_logger("v1_rage_chronost5_trev", "v1-rage-chronosT5-trev-base",
                              "experiment", "initialization")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal_handler)
    LOGGER.info("stage=configuration random=%s full=%s", random_config_path, full_config_path)
    if args.prepare_only:
        for config in (random_config, full_config):
            phase_dir = output_path("train", Path("outputs/train") / config["experiment_version"])
            prepare_pool(config, phase_dir)
        return 0
    train_phase(random_config, random_config_path)
    train_phase(full_config, full_config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
