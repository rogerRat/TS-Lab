#!/usr/bin/env python
"""Train and evaluate TimerXL or TimesFM Trev batches in fixed-step order."""
from __future__ import annotations
import argparse
import json
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
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora
from rage.experiments.regions import RegionWindows
from rage.experiments.train_eval import TrainingRecipe, fine_tune

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CHILD = None
LOGGER = None

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temp.replace(path)

def on_signal(signum, _frame):
    global ACTIVE_CHILD
    if ACTIVE_CHILD is not None and ACTIVE_CHILD.poll() is None:
        os.killpg(ACTIVE_CHILD.pid, signal.SIGINT)
        ACTIVE_CHILD.wait()
    raise KeyboardInterrupt("received signal %s" % signum)

def load_pool(config):
    root = ROOT / config["pool_phase"] / "inputs"
    manifest_path = root / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fraction = config["random_fraction"]
    expected = 0.1 if config["phase"] == "rand" else 1.0
    if fraction != expected or manifest["contract"]["random_fraction"] != fraction:
        raise ValueError("shared candidate-pool fraction does not match this phase")
    if int(manifest["contract"]["sampling_seed"]) != int(config.get("sampling_seed", 0)):
        raise ValueError("shared candidate-pool sampling seed does not match this phase")
    windows = RegionWindows(np.load(root / "values.npy", mmap_mode="r"),
        np.load(root / "offsets.npy", mmap_mode="r"), np.load(root / "regions.npy", mmap_mode="r"),
        window_length=576, budget_windows=4000 * 32)
    windows.training_contract = {"experiment_version": config["experiment_version"],
        "shared_candidate_pool": str(manifest_path), "selected_series": manifest["selected_series"],
        "selected_points": manifest["selected_points"]}
    LOGGER.info("stage=pool-ready phase=%s series=%d points=%d legal_windows=%d",
        config["phase"], manifest["selected_series"], manifest["selected_points"], len(windows))
    return windows, manifest

def save_adapter(engine, destination, step, config):
    temp = destination.with_name(destination.name + ".partial")
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    engine.model.save_pretrained(temp, safe_serialization=True)
    write_json(temp / "rage_adapter_manifest.json", {"schema_version": 1,
        "experiment_version": config["experiment_version"], "model_key": config["model_key"],
        "model": config["model_id"], "training_step": step, "precision": "bfloat16",
        "lora": config["lora"]})
    if destination.exists(): shutil.rmtree(destination)
    temp.replace(destination)

def copy_model_code(model_path, destination):
    for source in Path(model_path).glob("*.py"):
        shutil.copy2(source, destination / source.name)

def export_checkpoint(config, adapter_dir, step, destination):
    destination.mkdir(parents=True, exist_ok=True)
    from peft import PeftModel
    if config["model_key"] == "timerxl-84m":
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(config["model_path"], trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16)
        copy_model_code(config["model_path"], destination)
    else:
        from transformers import TimesFm2_5ModelForPrediction
        base = TimesFm2_5ModelForPrediction.from_pretrained(config["model_path"],
            local_files_only=True, torch_dtype=torch.bfloat16)
    destination.mkdir(parents=True, exist_ok=True)
    wrapped = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = wrapped.merge_and_unload()
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    write_json(destination / "rage_checkpoint_manifest.json", {"schema_version": 1,
        "experiment_version": config["experiment_version"], "model_key": config["model_key"],
        "model": config["model_id"], "training_step": step, "precision": "bfloat16",
        "reference_task_record": {"benchmark": "fev", "task_indices": [9, 11, 18, 19]}})
    del base, wrapped, merged

def base_checkpoint(config, phase_dir):
    destination = phase_dir / "checkpoints/base-step-0000"
    destination.mkdir(parents=True, exist_ok=True)
    for source in Path(config["model_path"]).iterdir():
        if source.is_file() and (source.name.endswith(".json") or source.suffix in {".safetensors", ".py"}):
            target = destination / source.name
            if not target.exists(): target.symlink_to(source)
    write_json(destination / "rage_checkpoint_manifest.json", {"schema_version": 1,
        "experiment_version": config["experiment_version"], "model_key": config["model_key"],
        "model": config["model_id"], "training_step": 0, "precision": "bfloat16",
        "reference_task_record": {"benchmark": "fev", "task_indices": [9, 11, 18, 19]}})
    return destination

def evaluate_phase(config, phase_dir):
    global ACTIVE_CHILD
    eval_root = output_path("eval", Path("outputs/eval") / config["experiment_version"])
    for step in config["step_checkpoints"]:
        step = int(step)
        step_dir = eval_root / ("step-%04d" % step)
        benchmark_done = []
        for benchmark in ("fev", "time"):
            marker = step_dir / benchmark / ("fev_results.csv" if benchmark == "fev" else "time_metadata.json")
            benchmark_done.append(marker.exists() and (benchmark == "fev" or json.loads(marker.read_text()).get("completed_at")))
        if all(benchmark_done):
            LOGGER.info("stage=evaluation-step-skip step=%d both benchmarks complete", step)
            if step < 4000:
                shutil.rmtree(phase_dir / "checkpoints" / ("adapter-step-%04d" % step), ignore_errors=True)
            continue
        export_path = None
        if step == 0:
            checkpoint = base_checkpoint(config, phase_dir)
        else:
            adapter_dir = phase_dir / "checkpoints" / ("adapter-step-%04d" % step)
            export_path = phase_dir / ("export-step-%04d" % step)
            if not export_path.exists():
                LOGGER.info("stage=checkpoint-export-start step=%d", step)
                export_checkpoint(config, adapter_dir, step, export_path)
            checkpoint = export_path
        for benchmark, cfg in (("fev", config["evaluation_configs"][0]),
                               ("time", config["evaluation_configs"][1])):
            out = step_dir / benchmark
            expected = out / ("fev_results.csv" if benchmark == "fev" else "time_metadata.json")
            done = expected.exists() and (benchmark == "fev" or json.loads(expected.read_text()).get("completed_at"))
            if done:
                LOGGER.info("stage=evaluation-skip step=%d benchmark=%s", step, benchmark)
                continue
            command = [sys.executable, "-u", "-m", "evaluate.run", "--checkpoint", str(checkpoint),
                "--config", cfg, "--output-dir", str(out)]
            if out.exists(): command.append("--resume")
            env = dict(os.environ)
            env["PYTHONPATH"] = "src:third_parties/fev/src:third_parties/TIME/src:." + os.pathsep + env.get("PYTHONPATH", "")
            LOGGER.info("stage=evaluation-start step=%d benchmark=%s", step, benchmark)
            ACTIVE_CHILD = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
            result = ACTIVE_CHILD.wait(); ACTIVE_CHILD = None
            if result: raise RuntimeError("evaluation failed: step=%d benchmark=%s exit=%d" % (step, benchmark, result))
            LOGGER.info("stage=evaluation-finished step=%d benchmark=%s", step, benchmark)
        if step < 4000:
            shutil.rmtree(phase_dir / "checkpoints" / ("adapter-step-%04d" % step), ignore_errors=True)
            LOGGER.info("stage=intermediate-checkpoint-deleted step=%d", step)
        if export_path is not None:
            shutil.rmtree(export_path)
            torch.cuda.empty_cache()

def train_phase(config, config_path):
    phase_dir = output_path("train", Path("outputs/train") / config["experiment_version"])
    phase_dir.mkdir(parents=True, exist_ok=True)
    complete = phase_dir / ".complete"
    if complete.exists(): return phase_dir
    windows, pool_manifest = load_pool(config)
    LOGGER.info("stage=model-load-start model=%s dtype=bfloat16", config["model_id"])
    engine, _probe, identity = build_model(config["model_key"], model_path=config["model_path"],
        device="cuda", context_length=512, prediction_length=64)
    lora = apply_student_lora(engine, config["model_key"], config["lora"])
    if any(p.dtype != torch.bfloat16 for p in engine.model.parameters()):
        raise TypeError("all model and LoRA parameters must remain BF16")
    trainable = [p for p in engine.model.parameters() if p.requires_grad]
    if not trainable or any(p.dtype != torch.bfloat16 for p in trainable):
        raise TypeError("all trainable parameters must remain BF16")
    recipe = TrainingRecipe(learning_rate=config["learning_rate"], epochs=1.0, batch_size=32,
        micro_batch_size=8, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8,
        max_grad_norm=1.0, warmup_ratio=0.03, schedule="cosine", schedule_epochs=1.0)
    checkpoint_steps = [int(x) for x in config["step_checkpoints"]]
    ckpt_root = phase_dir / "checkpoints"; ckpt_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); last_log = started
    def on_step(step):
        nonlocal last_log
        if step in checkpoint_steps[1:]:
            save_adapter(engine, ckpt_root / ("adapter-step-%04d" % step), step, config)
            LOGGER.info("stage=checkpoint step=%d", step)
        now = time.monotonic()
        if step in checkpoint_steps[1:] or now - last_log >= 30:
            eta = (4000-step) / (step/max(now-started, 1e-6))
            LOGGER.info("stage=train-progress step=%d/4000 eta=%s", step, format_duration(eta)); last_log=now
    LOGGER.info("stage=train-start phase=%s steps=4000 effective_batch=32 micro_batch=8 precision=bfloat16",
        config["experiment_version"])
    result = fine_tune(engine, windows, steps=4000, batch_size=32,
        learning_rate=config["learning_rate"], seed=int(config.get("sampling_seed", 0)), on_step=on_step,
        resume_path=ckpt_root / "resume.pt", checkpoint_steps=checkpoint_steps[1:], recipe=recipe,
        schedule_total_steps=4000)
    if result.get("steps") != 4000: raise RuntimeError("training stopped before 4,000 optimizer updates")
    write_json(phase_dir / "training_manifest.json", {"experiment_version": config["experiment_version"],
        "config": str(config_path), "model": identity, "lora": lora, "precision": "bfloat16",
        "recipe": recipe.as_dict(), "step_checkpoints": checkpoint_steps,
        "training_result": result, "input_manifest": str(config["pool_phase"] + "/inputs/input_manifest.json"),
        "input_pool": pool_manifest})
    LOGGER.info("stage=train-finished phase=%s elapsed=%s", config["experiment_version"],
        format_duration(result["cumulative_training_seconds"]))
    del engine, _probe; torch.cuda.empty_cache()
    evaluate_phase(config, phase_dir)
    complete.write_text(json.dumps({"completed_steps": checkpoint_steps,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2) + "\n", encoding="ascii")
    (ckpt_root / "resume.pt").unlink(missing_ok=True)
    shutil.rmtree(ckpt_root / "base-step-0000", ignore_errors=True)
    LOGGER.info("stage=phase-complete phase=%s output=%s", config["experiment_version"], phase_dir)
    return phase_dir

def main():
    global LOGGER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rand-config", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    args = parser.parse_args()
    LOGGER = configure_logger("v1_rage_foundation_trev", "v1-rage-foundation-trev", "experiment", "initialization")
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, on_signal)
    for path in (args.rand_config, args.full_config):
        config = json.loads(path.read_text()); LOGGER.info("stage=configuration phase=%s config=%s", config["phase"], path)
        train_phase(config, path)

if __name__ == "__main__": main()
