"""Compute the main-ads-v0 full-trajectory gradient cache in parallel."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
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

DATA_CONFIG = ROOT / "configs/data/src-256.yaml"
DATA_DIR = ROOT / "data/src/shard_dataset/src-256"
MODEL_PATH = "/root/autodl-tmp/pretrained_models/timer-base-84m-84m"
CHECKPOINT = ROOT / "outputs/train/main-ads-v0/checkpoint/adapter"
GRAD_ROOT = ROOT / "outputs/grad_cache/main-ads-v0/traj-replay"
CONTEXT, PREDICTION, WINDOW = 480, 96, 576
PROJECTION_DIM, PROJECTION_SEED = 4096, 0


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="ascii")
    os.replace(temporary, path)


def load_source():
    config = yaml.safe_load(DATA_CONFIG.read_text(encoding="ascii"))
    rows = config["trajectories"]
    values = np.load(DATA_DIR / "values.npy", mmap_mode="r")
    offsets = np.asarray(np.load(DATA_DIR / "offsets.npy", mmap_mode="r"), dtype=np.int64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    if len(rows) != 256 or len(offsets) != 257 or not np.array_equal(np.diff(offsets), lengths):
        raise ValueError("src-256 layout differs from its registered configuration")
    if int(offsets[-1]) != len(values):
        raise ValueError("src-256 values and offsets do not align")
    legal = lengths - WINDOW + 1
    if np.any(legal <= 0):
        raise ValueError("src-256 contains a trajectory shorter than 576 points")
    return rows, values, offsets, lengths, legal


def _build_probe():
    from peft import PeftModel
    engine, _probe, identity = build_model(
        "timerxl-84m", model_path=MODEL_PATH, device="cuda",
        context_length=CONTEXT, prediction_length=PREDICTION)
    engine.model = PeftModel.from_pretrained(engine.model, CHECKPOINT, is_trainable=True)
    engine.adapter._model = engine.model
    engine.model.eval()
    probe = TimerXLGradientEngine(
        engine, model_key=identity["model_id"] + "@" + identity["revision"],
        projection_dimension=PROJECTION_DIM, projection_seed=PROJECTION_SEED)
    return engine, probe, identity


def _compute_range(ordinal, first, last, worker_id, batch_size):
    torch.set_num_threads(1)
    logger = configure_logger(
        "main_ads_v0_gradient_worker_%02d" % worker_id,
        "main-ads-v0", "gradient-worker-%02d" % worker_id, "gradient-worker",
        enable_heartbeat=True)
    rows, values, offsets, lengths, legal = load_source()
    engine, probe, _identity = _build_probe()
    output = output_path("grad_cache", GRAD_ROOT / ("trajectory-%03d" % ordinal))
    output.mkdir(parents=True, exist_ok=True)
    sketches_path = output / "gradient_sketches.npy"
    sketches = np.load(sketches_path, mmap_mode="r+")
    if sketches.shape != (int(legal[ordinal]), PROJECTION_DIM) or sketches.dtype != np.float16:
        raise ValueError("gradient cache shape or dtype differs from the design")
    value_begin = int(offsets[ordinal])
    started = time.monotonic()
    logger.info("stage=worker-start worker=%d trajectory=%d range=%d:%d batch=%d",
                worker_id, ordinal, first, last, batch_size)
    cursor = first
    while cursor < last:
        stop = min(cursor + batch_size, last)
        windows = np.stack([
            values[value_begin + index:value_begin + index + WINDOW]
            for index in range(cursor, stop)
        ]).astype(np.float32)
        _losses, sketch = probe.batch_gradient_sketch(windows, batch_size=batch_size)
        if sketch.shape != (stop - cursor, PROJECTION_DIM) or not np.isfinite(sketch).all():
            raise ValueError("gradient projection has invalid values or shape")
        sketches[cursor:stop] = sketch.astype(np.float16)
        cursor = stop
        if cursor == last or (cursor - first) % 5000 < batch_size:
            sketches.flush()
            elapsed = time.monotonic() - started
            done = cursor - first
            eta = elapsed * (last - cursor) / max(done, 1)
            logger.info("stage=worker-progress worker=%d trajectory=%d windows=%d/%d elapsed=%s eta=%s",
                        worker_id, ordinal, cursor - first, last - first,
                        format_duration(elapsed), format_duration(eta))
    sketches.flush()
    del sketches
    del probe, engine
    gc_collect = __import__("gc").collect
    gc_collect()
    torch.cuda.empty_cache()
    duration = time.monotonic() - started
    logger.info("stage=worker-complete worker=%d trajectory=%d range=%d:%d duration=%s",
                worker_id, ordinal, first, last, format_duration(duration))
    return {"worker": worker_id, "trajectory": ordinal,
            "start": first, "stop": last, "duration_seconds": duration}


def make_manifest(ordinal, row, length, count):
    target = GRAD_ROOT / ("trajectory-%03d" % ordinal)
    record = {
        "trajectory_ordinal": ordinal, "source": row.get("source"),
        "subset": row.get("subset"), "series_id": row.get("series_id"),
        "length": int(length), "window_length": WINDOW,
        "context_length": CONTEXT, "prediction_length": PREDICTION,
        "window_start_count": int(count), "window_start_rule": "all legal starts, stride 1",
        "gradient_variable": "per-window loss gradient for all trainable LoRA parameters",
        "projection": "Count-Sketch random projection",
        "projection_dimension": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
        "storage_dtype": "float16", "checkpoint": "outputs/train/main-ads-v0/checkpoint/adapter/",
        "gradient_file": "outputs/grad_cache/main-ads-v0/traj-replay/trajectory-%03d/gradient_sketches.npy" % ordinal,
        "training_manifest": "outputs/train/main-ads-v0/checkpoint/training_manifest.json",
    }
    write_json(target / "manifest.json", record)
    (target / ".complete").write_text("complete\n", encoding="ascii")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume-trajectory-0-from", type=int, default=16000)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch size must be positive")
    logger = configure_logger("main_ads_v0_gradient_parallel", "main-ads-v0",
                              "traj-replay-parallel", "startup")
    rows, _values, _offsets, lengths, legal = load_source()
    if not (CHECKPOINT / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("trained TimerXL adapter is missing")
    root = output_path("grad_cache", GRAD_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    logger.info("stage=config workers=%d batch_size=%d projection=%d projection_seed=%d resume_ordinal0_from=%d trajectories=0,1",
                args.workers, args.batch_size, PROJECTION_DIM, PROJECTION_SEED,
                args.resume_trajectory_0_from)
    started = time.monotonic()
    records = []
    for ordinal in (0, 1):
        count = int(legal[ordinal])
        target = output_path("grad_cache", GRAD_ROOT / ("trajectory-%03d" % ordinal))
        target.mkdir(parents=True, exist_ok=True)
        cache_path = target / "gradient_sketches.npy"
        if cache_path.exists():
            existing = np.load(cache_path, mmap_mode="r+")
            if existing.shape != (count, PROJECTION_DIM) or existing.dtype != np.float16:
                raise ValueError("existing partial gradient cache has unexpected layout")
            del existing
        else:
            np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                      shape=(count, PROJECTION_DIM)).flush()
        begin = args.resume_trajectory_0_from if ordinal == 0 else 0
        if begin < 0 or begin > count:
            raise ValueError("resume start is outside trajectory window range")
        remaining = count - begin
        workers = (min(args.workers, remaining) if remaining >= 1000
                   else (1 if remaining > 0 else 0))
        ranges = []
        if workers:
            total = count - begin
            quotient, remainder = divmod(total, workers)
            cursor = begin
            for worker_id in range(workers):
                size = quotient + (worker_id < remainder)
                ranges.append((ordinal, cursor, cursor + size, worker_id, args.batch_size))
                cursor += size
        logger.info("stage=trajectory-start ordinal=%d subset=%s series_id=%s windows=%d remaining=%d workers=%d",
                    ordinal, rows[ordinal].get("subset", ""),
                    rows[ordinal].get("series_id", ""), count, count - begin, workers)
        if workers > 1:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = [executor.submit(_compute_range, *item) for item in ranges]
                for future in as_completed(futures):
                    logger.info("stage=worker-return %s", future.result())
        elif workers == 1:
            _compute_range(*ranges[0])
        records.append(make_manifest(ordinal, rows[ordinal], lengths[ordinal], count))
        logger.info("stage=trajectory-complete ordinal=%d windows=%d output=%s",
                    ordinal, count, target)
    write_json(root / "manifest.json", {
        "experiment": "main-ads-v0", "batch": "traj-replay",
        "checkpoint": "outputs/train/main-ads-v0/checkpoint/adapter/",
        "projection_dimension": PROJECTION_DIM, "projection_seed": PROJECTION_SEED,
        "worker_count": args.workers, "batch_size": args.batch_size,
        "trajectories": records, "replay_content": "pending",
    })
    (root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=gradient-cache-complete trajectories=2 duration=%s output=%s",
                format_duration(time.monotonic() - started), root)


if __name__ == "__main__":
    main()
