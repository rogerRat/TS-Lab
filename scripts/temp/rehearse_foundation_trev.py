#!/usr/bin/env python
"""Exercise one BF16 LoRA update and representative FEV/TIME evaluations."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora
from rage.experiments.train_eval import TrainingRecipe, fine_tune

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def make_eval_configs(model_key, model_name):
    import yaml
    fev = yaml.safe_load((ROOT / "configs/eval/fev/v1-rage-trev-run.yaml").read_text())
    fev["model"]["id"] = "rehearsal_" + model_name
    fev["adapter"]["class"] = "adapters.foundation.eval.fev:FoundationFEVEvaluationAdapter"
    fev["adapter"]["config"] = {"model": {"model_key": model_key, "device": "cpu", "torch_dtype": "bfloat16"},
        "inference": {"batch_size": 2, "evaluation_mode": "single_window", "seed": 20260722}}
    fev["bench_config"]["expected_tasks"] = 1
    fev["bench_config"]["task_indices"] = [9]
    fev_path = ROOT / ("configs/eval/fev/rehearsal-" + model_name + ".yaml")
    fev_path.write_text(yaml.safe_dump(fev, sort_keys=False), encoding="ascii")

    time = yaml.safe_load((ROOT / "configs/eval/time/v1-rage-trev.yaml").read_text())
    time["model"]["id"] = "rehearsal_" + model_name
    time["adapter"]["class"] = "adapters.foundation.eval.time:FoundationTIMEEvaluationAdapter"
    time["adapter"]["config"] = {"model": {"model_key": model_key, "device": "cpu", "torch_dtype": "bfloat16"},
        "inference": {"batch_size": 2, "context_length": 512,
            "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "seed": 20260722, "evaluation_mode": "single_window"}}
    time["bench_config"]["dataset"]["root"] = "data/bench/time/v1-rage-trev-rehearsal/dataset"
    time["bench_config"]["tasks"]["items"] = [{"dataset": "CPHL/15T", "terms": ["short"]}]
    time_path = ROOT / ("configs/eval/time/rehearsal-" + model_name + ".yaml")
    time_path.write_text(yaml.safe_dump(time, sort_keys=False), encoding="ascii")
    return fev_path, time_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    model_key = args.model_key
    model_name = "timerxl" if model_key == "timerxl-84m" else "timesfm25"
    output = ROOT / "outputs/train/rehearsal/foundation" / model_name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    engine, _probe, identity = build_model(model_key, model_path=args.model_path,
        device="cpu", context_length=512, prediction_length=64)
    lora = apply_student_lora(engine, model_key, {"r": 8, "alpha": 16, "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"]})
    if any(parameter.dtype != torch.bfloat16 for parameter in engine.model.parameters()):
        raise TypeError("model and LoRA parameters must be BF16")
    windows = np.random.default_rng(7).normal(size=(1, 576)).astype(np.float32)
    recipe = TrainingRecipe(learning_rate=1e-4, epochs=1.0, batch_size=1, micro_batch_size=1,
        weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, max_grad_norm=1.0,
        warmup_ratio=0.0, schedule="cosine", schedule_epochs=1.0)
    training = fine_tune(engine, windows, steps=1, batch_size=1, learning_rate=1e-4,
        seed=7, weight_decay=0.0, recipe=recipe, schedule_total_steps=1, checkpoint_steps=[1])
    if training.get("steps") != 1 or not np.isfinite(training.get("final_loss", np.nan)):
        raise ValueError("one-step BF16 LoRA rehearsal failed")
    adapter_path = output / "adapter-step-0001"
    engine.model.save_pretrained(adapter_path, safe_serialization=True)
    del engine
    torch.cuda.empty_cache()

    if model_key == "timerxl-84m":
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16)
    else:
        from transformers import TimesFm2_5ModelForPrediction
        base = TimesFm2_5ModelForPrediction.from_pretrained(args.model_path, local_files_only=True,
            torch_dtype=torch.bfloat16)
    from peft import PeftModel
    merged = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).merge_and_unload()
    checkpoint = output / "checkpoint"
    checkpoint.mkdir()
    if model_key == "timerxl-84m":
        for source in Path(args.model_path).glob("*.py"):
            shutil.copy2(source, checkpoint / source.name)
    merged.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="5GB")
    write_json(checkpoint / "rage_checkpoint_manifest.json", {"schema_version": 1,
        "experiment_version": "rehearsal-" + model_name, "model_key": model_key,
        "precision": "bfloat16", "training_step": 1,
        "reference_task_record": {"benchmark": "fev", "task_indices": [9]}})
    fev_config, time_config = make_eval_configs(model_key, model_name)
    rehearsal_data = ROOT / "data/bench/time/v1-rage-trev-rehearsal"
    rehearsal_data.mkdir(parents=True, exist_ok=True)
    link = rehearsal_data / "dataset/CPHL/15T"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(ROOT / "data/bench/time/83e3d0b3be28d11c7182bffcc1892d19b36c4da1/dataset/CPHL/15T")
    write_json(rehearsal_data / "data_manifest.json", {"repo_id": "Real-TSF/TIME",
        "revision": "83e3d0b3be28d11c7182bffcc1892d19b36c4da1",
        "license": "CC BY-NC 4.0", "verified_tasks": [{"dataset": "CPHL/15T", "term": "short"}],
        "selection_protocol": "temporary one-task foundation-model rehearsal"})
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = "src:third_parties/fev/src:third_parties/TIME/src:."
    for benchmark, config in (("fev", fev_config), ("time", time_config)):
        command = [sys.executable, "-u", "-m", "evaluate.run", "--checkpoint", str(checkpoint),
            "--config", str(config.relative_to(ROOT)), "--output-dir",
            str((ROOT / "outputs/eval/rehearsal/foundation" / model_name / benchmark).relative_to(ROOT))]
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode:
            raise RuntimeError("%s rehearsal failed for %s" % (benchmark, model_name))
    write_json(output / "rehearsal_summary.json", {"model_key": model_key,
        "model_id": identity["model_id"], "lora": lora, "training": training,
        "precision": "bfloat16", "fev": "completed", "time": "completed"})
    print("REHEARSAL_COMPLETE", model_name, "step=1", flush=True)


if __name__ == "__main__":
    main()
