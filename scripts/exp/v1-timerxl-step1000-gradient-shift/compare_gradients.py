"""Compare paired TimerXL LoRA loss gradients at initialization and step 1000."""
from __future__ import annotations

import csv
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from gradient.timerxl import TimerXLGradientEngine
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora

MODEL_PATH = Path("/root/autodl-tmp/pretrained_models/timer-base-84m-84m")
ADAPTER_PATH = ROOT / "outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000"
DATA_DIR = ROOT / "data/src/shard_dataset/src-96"
OUT = ROOT / "outputs/train/v1-timerxl-step1000-gradient-shift"
CONTEXT = 480
HORIZON = 96
WINDOW = CONTEXT + HORIZON
PER_TRAJECTORY = 64
SAMPLE_SEED = 42
PROJECTION_DIM = 4096
PROJECTION_SEED = 42
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 42
BATCH_SIZE = 4
LORA = {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    temporary.replace(path)


def validate_checkpoint() -> dict:
    manifest_path = ADAPTER_PATH.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest["model"]
    lora = manifest["lora"]
    recipe = manifest["recipe"]
    if manifest["result"]["steps"] != 1000 or manifest["seed"] != 0:
        raise ValueError("step-1000 checkpoint training identity differs")
    if model["revision"] != "70077a71acce1b4c00d98332fcaabc694255d8e5":
        raise ValueError("checkpoint base revision differs")
    if manifest["context_length"] != CONTEXT or manifest["prediction_length"] != HORIZON:
        raise ValueError("checkpoint loss window protocol differs")
    if lora["r"] != 8 or lora["alpha"] != 16 or lora["dropout"] != 0.0:
        raise ValueError("checkpoint LoRA settings differ")
    if lora["target_modules"] != LORA["target_modules"]:
        raise ValueError("checkpoint LoRA targets differ")
    if recipe["batch_size"] != 32 or recipe["micro_batch_size"] != 8:
        raise ValueError("checkpoint training batch differs")
    return manifest


def create_sample() -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest["trajectory_count"] != 96:
        raise ValueError("src-96 trajectory count differs from the approved design")
    values = np.load(DATA_DIR / "values.npy", mmap_mode="r")
    offsets = np.load(DATA_DIR / "offsets.npy", mmap_mode="r")
    if len(offsets) != manifest["trajectory_count"] + 1:
        raise ValueError("src-96 offset count differs from the manifest")
    rng = np.random.default_rng(SAMPLE_SEED)
    ids = []
    lengths = []
    for ordinal in range(manifest["trajectory_count"]):
        begin, end = int(offsets[ordinal]), int(offsets[ordinal + 1])
        length = end - begin
        legal = length - WINDOW + 1
        if legal < PER_TRAJECTORY:
            raise ValueError("trajectory %d has only %d legal windows" % (ordinal, legal))
        starts = np.sort(rng.choice(legal, size=PER_TRAJECTORY, replace=False))
        ids.extend((ordinal, int(start)) for start in starts)
        lengths.append(length)
    ids_array = np.asarray(ids, dtype=np.int64)
    windows = np.empty((len(ids_array), WINDOW), dtype=np.float32)
    for row, (ordinal, start) in enumerate(ids_array):
        absolute = int(offsets[ordinal]) + int(start)
        windows[row] = values[absolute:absolute + WINDOW]
    if not np.isfinite(windows).all():
        raise ValueError("sampled windows contain non-finite values")
    return ids_array, windows, {
        "dataset": "src-96",
        "trajectory_count": len(lengths),
        "trajectory_lengths": lengths,
        "window_length": WINDOW,
        "context_length": CONTEXT,
        "prediction_length": HORIZON,
        "windows_per_trajectory": PER_TRAJECTORY,
        "sample_count": len(ids_array),
        "sample_seed": SAMPLE_SEED,
        "sampling": "uniform_without_replacement_per_trajectory",
    }


def build_probe(adapter_path: Path | None):
    torch.manual_seed(0)
    engine, _, identity = build_model(
        "timerxl-84m", model_path=str(MODEL_PATH), device="cuda",
        context_length=CONTEXT, prediction_length=HORIZON,
    )
    lora_info = apply_student_lora(engine, "timerxl-84m", LORA)
    if lora_info["trainable_parameters"] != 262144:
        raise ValueError("LoRA trainable parameter count differs")
    if adapter_path is not None:
        adapter = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
        incompatible = set_peft_model_state_dict(engine.model, adapter)
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if unexpected:
            raise ValueError("Unexpected adapter keys: " + repr(unexpected))
    engine.model.eval()
    probe = TimerXLGradientEngine(
        engine, model_key=identity["model_id"] + "@" + identity["revision"],
        projection_dimension=PROJECTION_DIM, projection_seed=PROJECTION_SEED,
    )
    if not probe.parameters:
        raise ValueError("Unexpected number of trainable LoRA tensors")
    return engine, probe, identity


def compute(probe, windows: np.ndarray, label: str, logger) -> tuple[np.ndarray, np.ndarray]:
    all_losses = np.empty(len(windows), dtype=np.float32)
    all_sketches = np.empty((len(windows), PROJECTION_DIM), dtype=np.float32)
    start_time = time.monotonic()
    completed = 0
    for begin in range(0, len(windows), BATCH_SIZE):
        stop = min(begin + BATCH_SIZE, len(windows))
        losses, sketches = probe.batch_gradient_sketch(windows[begin:stop], batch_size=BATCH_SIZE)
        expected = (stop - begin, PROJECTION_DIM)
        if sketches.shape != expected or losses.shape != (stop - begin,):
            raise ValueError("gradient output shape mismatch")
        if not np.isfinite(losses).all() or not np.isfinite(sketches).all():
            raise ValueError("gradient output contains non-finite values")
        all_losses[begin:stop] = losses
        all_sketches[begin:stop] = sketches
        completed = stop
        if completed == len(windows) or completed % 256 == 0:
            elapsed = time.monotonic() - start_time
            eta = elapsed * (len(windows) - completed) / max(completed, 1)
            logger.info("stage=gradient-progress model=%s windows=%d/%d elapsed=%s eta=%s",
                        label, completed, len(windows), format_duration(elapsed),
                        format_duration(eta))
    return all_losses, all_sketches


def summarize(ids: np.ndarray, losses0: np.ndarray, g0: np.ndarray,
              losses1: np.ndarray, g1: np.ndarray) -> tuple[dict, list[dict]]:
    norm0 = np.linalg.norm(g0.astype(np.float64), axis=1)
    norm1 = np.linalg.norm(g1.astype(np.float64), axis=1)
    dot = np.sum(g0.astype(np.float64) * g1.astype(np.float64), axis=1)
    cosine = np.divide(dot, norm0 * norm1, out=np.zeros_like(dot),
                       where=(norm0 > 0) & (norm1 > 0))
    both_zero = (norm0 == 0) & (norm1 == 0)
    cosine[both_zero] = 1.0
    rel_l2 = np.linalg.norm((g1 - g0).astype(np.float64), axis=1) / np.maximum(norm0, 1e-12)
    norm_ratio = norm1 / np.maximum(norm0, 1e-12)
    angular = 1.0 - np.clip(cosine, -1.0, 1.0)
    trajectory_rows = []
    for ordinal in range(96):
        mask = ids[:, 0] == ordinal
        trajectory_rows.append({
            "trajectory": ordinal,
            "windows": int(mask.sum()),
            "mean_angular_difference": float(angular[mask].mean()),
            "mean_relative_l2": float(rel_l2[mask].mean()),
            "mean_norm_ratio": float(norm_ratio[mask].mean()),
            "mean_cosine": float(cosine[mask].mean()),
            "mean_loss_original": float(losses0[mask].mean()),
            "mean_loss_step1000": float(losses1[mask].mean()),
        })
    matrix = np.asarray([[row[k] for row in trajectory_rows] for k in (
        "mean_angular_difference", "mean_relative_l2", "mean_norm_ratio")], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(trajectory_rows), size=(BOOTSTRAPS, len(trajectory_rows)))
    boot = matrix[:, picks].mean(axis=2)
    names = ("mean_angular_difference", "mean_relative_l2", "mean_norm_ratio")
    estimates = matrix.mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975], axis=1)
    summary = {
        "sample_count": int(len(ids)),
        "trajectory_count": 96,
        "windows_per_trajectory": PER_TRAJECTORY,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "trajectory",
        "projection_dimension": PROJECTION_DIM,
        "projection_seed": PROJECTION_SEED,
        "zero_norm_original": int((norm0 == 0).sum()),
        "zero_norm_step1000": int((norm1 == 0).sum()),
        "mean_cosine": float(cosine.mean()),
        "mean_loss_original": float(losses0.mean()),
        "mean_loss_step1000": float(losses1.mean()),
        "metrics": {
            name: {"estimate": float(estimate), "ci95": [float(ci[0, i]), float(ci[1, i])]}
            for i, (name, estimate) in enumerate(zip(names, estimates))
        },
        "significant_angular_difference": bool(ci[0, 0] > 0),
        "interpretation_scope": "fixed src-96 pool and fixed seed-0 step-1000 checkpoint",
    }
    return summary, trajectory_rows


def main() -> None:
    logger = configure_logger(
        "timerxl_step1000_gradient_shift", "v1-timerxl-step1000-gradient-shift",
        "paired-src96-step1000", "startup",
    )
    out = output_path("train", OUT)
    if (out / ".complete").exists():
        raise FileExistsError("completed output already exists: " + str(out))
    if not ADAPTER_PATH.is_dir():
        raise FileNotFoundError("step-1000 adapter is missing: " + str(ADAPTER_PATH))
    checkpoint_manifest = validate_checkpoint()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("partial output exists; preserve it and use a new batch path: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    logger.info("stage=checkpoint-validated steps=%d seed=%d revision=%s",
                checkpoint_manifest["result"]["steps"], checkpoint_manifest["seed"],
                checkpoint_manifest["model"]["revision"])
    logger.info("stage=config model=timerxl-84m steps=1000 samples=6144 batch=%d projection=%d projection_seed=%d sample_seed=%d bootstrap=%d output=%s",
                BATCH_SIZE, PROJECTION_DIM, PROJECTION_SEED, SAMPLE_SEED, BOOTSTRAPS, out)
    started = time.monotonic()

    ids, windows, sample_manifest = create_sample()
    np.save(out / "window_ids.npy", ids)
    write_json(out / "sample_manifest.json", sample_manifest)
    logger.info("stage=sample-complete windows=%d trajectories=%d", len(ids), sample_manifest["trajectory_count"])

    logger.info("stage=model-load-start model=original")
    engine0, probe0, identity0 = build_probe(None)
    logger.info("stage=model-load-complete model=original")
    names0 = tuple((name, tuple(parameter.shape)) for name, parameter in probe0.named_parameters)
    losses0, grad0 = compute(probe0, windows, "original", logger)
    control_count = min(128, len(windows))
    _, repeat0 = compute(probe0, windows[:control_count], "original-repeat-control", logger)
    repeat_error = float(np.max(np.abs(repeat0 - grad0[:control_count])))
    np.save(out / "gradient_original.npy", grad0)
    del probe0, engine0
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("stage=model-load-start model=step1000")
    engine1, probe1, identity1 = build_probe(ADAPTER_PATH)
    logger.info("stage=model-load-complete model=step1000")
    names1 = tuple((name, tuple(parameter.shape)) for name, parameter in probe1.named_parameters)
    if names0 != names1:
        raise ValueError("LoRA parameter names or shapes differ between model states")
    losses1, grad1 = compute(probe1, windows, "step1000", logger)
    np.save(out / "gradient_step1000.npy", grad1)
    del probe1, engine1
    gc.collect()
    torch.cuda.empty_cache()

    summary, trajectory_rows = summarize(ids, losses0, grad0, losses1, grad1)
    summary["original_repeat_control_max_abs_sketch_error"] = repeat_error
    summary["model_identity"] = identity0
    summary["step1000_model_identity"] = identity1
    summary["checkpoint"] = str(ADAPTER_PATH)
    summary["lora_parameter_tensors"] = len(names0)
    summary["lora_parameter_names"] = [name for name, _ in names0]
    write_json(out / "summary.json", summary)
    with (out / "trajectory_metrics.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with (out / "window_metrics.csv").open("w", newline="", encoding="ascii") as handle:
        fields = ["trajectory", "window_start", "loss_original", "loss_step1000",
                  "cosine", "angular_difference", "relative_l2", "norm_ratio"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        norm0 = np.linalg.norm(grad0.astype(np.float64), axis=1)
        norm1 = np.linalg.norm(grad1.astype(np.float64), axis=1)
        cosine = np.divide(np.sum(grad0.astype(np.float64) * grad1.astype(np.float64), axis=1),
                           norm0 * norm1, out=np.zeros(len(ids)),
                           where=(norm0 > 0) & (norm1 > 0))
        both_zero = (norm0 == 0) & (norm1 == 0)
        cosine[both_zero] = 1.0
        relative = np.linalg.norm((grad1-grad0).astype(np.float64), axis=1) / np.maximum(norm0, 1e-12)
        for i, (ordinal, start) in enumerate(ids):
            writer.writerow({
                "trajectory": int(ordinal), "window_start": int(start),
                "loss_original": float(losses0[i]), "loss_step1000": float(losses1[i]),
                "cosine": float(cosine[i]), "angular_difference": float(1-np.clip(cosine[i], -1, 1)),
                "relative_l2": float(relative[i]), "norm_ratio": float(norm1[i]/max(norm0[i], 1e-12)),
            })
    if repeat_error > 1e-3:
        raise ValueError("Repeated original gradient sketch error exceeds tolerance: %.6g" % repeat_error)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=finished significant=%s angular=%s relative_l2=%s repeat_error=%.6g elapsed=%s output=%s",
                summary["significant_angular_difference"],
                summary["metrics"]["mean_angular_difference"],
                summary["metrics"]["mean_relative_l2"], repeat_error,
                format_duration(time.monotonic()-started), out)


if __name__ == "__main__":
    main()
