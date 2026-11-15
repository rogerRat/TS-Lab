"""Train the main ADS-v0 TimerXL probe and cache full-trajectory gradients."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from gradient.timerxl import TimerXLGradientEngine
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora

DATA_CONFIG = ROOT / "configs/data/src-256.yaml"
DATA_DIR = ROOT / "data/src/shard_dataset/src-256"
MODEL_PATH = Path("/root/autodl-tmp/pretrained_models/timer-base-84m-84m")
TRAIN_OUT = ROOT / "outputs/train/main-ads-v0/checkpoint"
GRAD_OUT = ROOT / "outputs/grad_cache/main-ads-v0/traj-replay"
CONTEXT, PREDICTION = 480, 96
WINDOW = CONTEXT + PREDICTION
STEPS, SEED = 2000, 0
EFFECTIVE_BATCH, MICRO_BATCH = 32, 8
LR, WEIGHT_DECAY, MAX_GRAD_NORM = 1e-4, 0.0, 1.0
BETAS, EPS, WARMUP_RATIO = (0.9, 0.999), 1e-8, 0.03
LORA = {"r": 8, "alpha": 16, "dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"]}
PROJECTION_DIM, PROJECTION_SEED = 4096, 0
TRAJECTORIES = (0, 1)
GRADIENT_BATCH = 1


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="ascii")
    os.replace(temporary, path)


def load_source():
    config = yaml.safe_load(DATA_CONFIG.read_text(encoding="ascii"))
    rows = config["trajectories"]
    if len(rows) != 256:
        raise ValueError("src-256 must contain exactly 256 trajectories")
    values = np.load(DATA_DIR / "values.npy", mmap_mode="r")
    offsets = np.asarray(np.load(DATA_DIR / "offsets.npy", mmap_mode="r"),
                         dtype=np.int64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    if (len(offsets) != 257 or not np.array_equal(np.diff(offsets), lengths)
            or int(offsets[-1]) != len(values)):
        raise ValueError("src-256 values and offsets differ from the registered config")
    legal = lengths - WINDOW + 1
    if np.any(legal <= 0):
        raise ValueError("src-256 has a trajectory shorter than the 576-point window")
    return rows, values, offsets, lengths, legal


def sample_batch(rng, values, offsets, legal):
    owners = rng.integers(0, len(legal), size=EFFECTIVE_BATCH)
    starts = np.asarray([rng.integers(0, legal[int(owner)]) for owner in owners],
                        dtype=np.int64)
    windows = np.stack([values[int(offsets[owner] + start):
                               int(offsets[owner] + start + WINDOW)]
                        for owner, start in zip(owners, starts)]).astype(np.float32)
    if windows.shape != (EFFECTIVE_BATCH, WINDOW) or not np.isfinite(windows).all():
        raise ValueError("sampled training windows are invalid")
    return windows


def save_loss_svg(points, path):
    width, height = 1000, 600
    left, right, top, bottom = 90, 30, 35, 75
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    low, high = min(ys), max(ys)
    if high <= low:
        high = low + 1e-6
    def x_at(step):
        return left + (step - min(xs)) * (width - left - right) / max(1, max(xs) - min(xs))
    def y_at(loss):
        return top + (high - loss) * (height - top - bottom) / (high - low)
    polyline = " ".join("%.2f,%.2f" % (x_at(step), y_at(loss)) for step, loss in points)
    grid = []
    for index in range(6):
        y = top + index * (height - top - bottom) / 5
        label = high - index * (high - low) / 5
        grid.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#d9dee7"/><text x="%d" y="%.2f" text-anchor="end" font-size="13">%.5f</text>' %
                    (left, y, width - right, y, left - 12, y + 4, label))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600" viewBox="0 0 1000 600">',
           '<rect width="1000" height="600" fill="white"/>',
           '<text x="500" y="24" text-anchor="middle" font-size="18">TimerXL LoRA training loss (main-ads-v0)</text>']
    svg.extend(grid)
    svg.extend(['<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#303846"/>' %
                (left, top, left, height - bottom),
                '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#303846"/>' %
                (left, height - bottom, width - right, height - bottom),
                '<polyline fill="none" stroke="#2457a7" stroke-width="1.5" points="%s"/>' % polyline,
                '<text x="500" y="580" text-anchor="middle" font-size="15">Training step</text>',
                '<text transform="translate(22 300) rotate(-90)" text-anchor="middle" font-size="15">Training loss</text>',
                '</svg>'])
    path.write_text("\n".join(svg) + "\n", encoding="ascii")


def train(logger, rows, values, offsets, lengths, legal):
    out = output_path("train", TRAIN_OUT)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("training output already exists: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logger.info("stage=model-load-start model=timerxl-84m device=cuda")
    engine, _probe, identity = build_model(
        "timerxl-84m", model_path=str(MODEL_PATH), device="cuda",
        context_length=CONTEXT, prediction_length=PREDICTION)
    lora = apply_student_lora(engine, "timerxl-84m", LORA)
    trainable = [parameter for parameter in engine.model.parameters()
                 if parameter.requires_grad]
    if lora["trainable_parameters"] != 262144:
        raise ValueError("TimerXL LoRA trainable parameter count differs")
    logger.info("stage=model-load-complete revision=%s trainable_parameters=%d",
                identity["revision"], lora["trainable_parameters"])
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY,
                                  betas=BETAS, eps=EPS, foreach=False)
    rng = np.random.default_rng(SEED)
    curve_path = out / "loss_every_10_steps.csv"
    points = []
    started = time.monotonic()
    engine.model.train()
    logger.info("stage=train-start steps=%d batch=%d micro_batch=%d seed=%d dataset=src-256",
                STEPS, EFFECTIVE_BATCH, MICRO_BATCH, SEED)
    with curve_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])
        for step in range(STEPS):
            completed = step + 1
            warmup_steps = int(WARMUP_RATIO * STEPS)
            if completed <= warmup_steps:
                multiplier = completed / max(1, warmup_steps)
            else:
                progress = (completed - warmup_steps) / max(1, STEPS - warmup_steps)
                multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = LR * multiplier
            windows = sample_batch(rng, values, offsets, legal)
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for begin in range(0, EFFECTIVE_BATCH, MICRO_BATCH):
                loss = engine.training_loss(windows[begin:begin + MICRO_BATCH])
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("non-finite TimerXL loss at step %d" % (step + 1))
                (loss * MICRO_BATCH / EFFECTIVE_BATCH).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
            optimizer.step()
            value = float(torch.stack(losses).mean().float().cpu())
            if completed % 10 == 0:
                writer.writerow([completed, "%.9g" % value])
                handle.flush()
                points.append((completed, value))
                elapsed = time.monotonic() - started
                eta = elapsed * (STEPS - completed) / completed
                logger.info("stage=train-progress step=%d/%d loss=%.8f elapsed=%s eta=%s",
                            completed, STEPS, value, format_duration(elapsed),
                            format_duration(eta))
    save_loss_svg(points, out / "loss_curve.svg")
    engine.model.eval()
    engine.model.save_pretrained(out / "adapter", safe_serialization=True)
    manifest = {
        "experiment": "main-ads-v0", "checkpoint_type": "lora_adapter",
        "model": {"student": "timerxl-84m", "model_id": identity["model_id"],
                  "revision": identity["revision"], "parameter_count": identity["parameter_count"]},
        "source_dataset": "src-256",
        "source_config": "configs/data/src-256.yaml", "source_trajectory_count": 256,
        "window": {"context": CONTEXT, "prediction": PREDICTION, "length": WINDOW,
                   "sampling": "uniform_trajectory_then_uniform_legal_window"},
        "steps": STEPS, "seed": SEED, "effective_batch_size": EFFECTIVE_BATCH,
        "micro_batch_size": MICRO_BATCH, "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY, "betas": list(BETAS), "eps": EPS,
        "gradient_clip": MAX_GRAD_NORM, "schedule": "cosine",
        "warmup_ratio": WARMUP_RATIO, "lora": lora,
        "loss_logging_interval": 10, "loss_curve": str((out / "loss_curve.svg").relative_to(ROOT)),
        "training_seconds": time.monotonic() - started,
        "first_recorded_loss": points[0][1] if points else None,
        "last_recorded_loss": points[-1][1] if points else None,
    }
    write_json(out / "training_manifest.json", manifest)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=train-complete steps=%d duration=%s output=%s",
                STEPS, format_duration(time.monotonic() - started), out)
    del engine, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return out / "adapter"


def compute_trajectory_gradients(logger, adapter_path, rows, values, offsets, lengths, legal):
    out = output_path("grad_cache", GRAD_OUT)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("gradient output already exists: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    logger.info("stage=gradient-model-load-start projection_dimension=%d projection_seed=%d",
                PROJECTION_DIM, PROJECTION_SEED)
    engine, _probe, identity = build_model(
        "timerxl-84m", model_path=str(MODEL_PATH), device="cuda",
        context_length=CONTEXT, prediction_length=PREDICTION)
    from peft import PeftModel
    engine.model = PeftModel.from_pretrained(engine.model, adapter_path, is_trainable=True)
    engine.adapter._model = engine.model
    engine.model.eval()
    probe = TimerXLGradientEngine(
        engine, model_key=identity["model_id"] + "@" + identity["revision"],
        projection_dimension=PROJECTION_DIM, projection_seed=PROJECTION_SEED)
    logger.info("stage=gradient-model-load-complete checkpoint=%s", adapter_path)
    start_all = time.monotonic()
    records = []
    for ordinal in TRAJECTORIES:
        begin = int(offsets[ordinal])
        count = int(legal[ordinal])
        row = rows[ordinal]
        target = out / ("trajectory-%03d" % ordinal)
        target.mkdir(parents=True, exist_ok=False)
        sketches = np.lib.format.open_memmap(
            target / "gradient_sketches.npy", mode="w+", dtype=np.float16,
            shape=(count, PROJECTION_DIM))
        started = time.monotonic()
        logger.info("stage=gradient-start trajectory=%d subset=%s series_id=%s windows=%d",
                    ordinal, row.get("subset", ""), row.get("series_id", ""), count)
        for first in range(0, count, GRADIENT_BATCH):
            last = min(first + GRADIENT_BATCH, count)
            batch = np.stack([values[begin + index:begin + index + WINDOW]
                              for index in range(first, last)]).astype(np.float32)
            _losses, sketch = probe.batch_gradient_sketch(batch, batch_size=GRADIENT_BATCH)
            if sketch.shape != (last - first, PROJECTION_DIM) or not np.isfinite(sketch).all():
                raise ValueError("gradient projection has invalid values or shape")
            sketches[first:last] = sketch.astype(np.float16)
            if last % 1000 == 0 or last == count:
                sketches.flush()
                elapsed = time.monotonic() - started
                eta = elapsed * (count - last) / max(last, 1)
                logger.info("stage=gradient-progress trajectory=%d windows=%d/%d elapsed=%s eta=%s",
                            ordinal, last, count, format_duration(elapsed), format_duration(eta))
        del sketches
        record = {
            "trajectory_ordinal": ordinal, "source": row.get("source"),
            "subset": row.get("subset"), "series_id": row.get("series_id"),
            "length": int(lengths[ordinal]), "window_length": WINDOW,
            "context_length": CONTEXT, "prediction_length": PREDICTION,
            "window_start_count": count, "window_start_rule": "all legal starts, stride 1",
            "projection": "Count-Sketch random projection",
            "projection_dimension": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
            "storage_dtype": "float16", "checkpoint": str(adapter_path.relative_to(ROOT)),
            "gradient_file": "outputs/grad_cache/main-ads-v0/traj-replay/trajectory-%03d/gradient_sketches.npy" % ordinal,
            "training_manifest": str((adapter_path.parent / "training_manifest.json").relative_to(ROOT)),
            "model_identity": identity,
        }
        write_json(target / "manifest.json", record)
        (target / ".complete").write_text("complete\n", encoding="ascii")
        records.append(record)
        logger.info("stage=gradient-complete trajectory=%d duration=%s output=%s",
                    ordinal, format_duration(time.monotonic() - started), target)
    write_json(out / "manifest.json", {
        "experiment": "main-ads-v0", "batch": "traj-replay",
        "checkpoint": str(adapter_path.relative_to(ROOT)),
        "projection_dimension": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
        "trajectories": records, "replay_content": "pending",
    })
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=gradient-cache-complete trajectories=%d duration=%s output=%s",
                len(records), format_duration(time.monotonic() - start_all), out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "gradients", "all"), default="all")
    args = parser.parse_args()
    logger = configure_logger("main_ads_v0", "main-ads-v0", "traj-replay", "startup")
    rows, values, offsets, lengths, legal = load_source()
    logger.info("stage=config stage=%s model=timerxl-84m dataset=src-256 steps=%d seed=%d lora_r=%d lora_alpha=%d projection=%d projection_seed=%d trajectories=%s replay_content=pending",
                args.stage, STEPS, SEED, LORA["r"], LORA["alpha"],
                PROJECTION_DIM, PROJECTION_SEED, list(TRAJECTORIES))
    adapter_path = TRAIN_OUT / "adapter"
    if args.stage in ("train", "all"):
        adapter_path = train(logger, rows, values, offsets, lengths, legal)
    if args.stage in ("gradients", "all"):
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError("trained TimerXL LoRA adapter is missing: " + str(adapter_path))
        compute_trajectory_gradients(logger, adapter_path, rows, values, offsets,
                                     lengths, legal)


if __name__ == "__main__":
    main()
