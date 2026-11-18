"""Prepare direction-only CRB/uni assets and run the four TIME1 comparison arms."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngineMulti
from rage.execution import SelectionExecution, selection_execution
from assets.gradient import FragmentGradients, GradientCache, WindowRequest

SOURCE_CONFIG = ROOT / "configs/data/src-256.yaml"
SOURCE_DATA = ROOT / "data/src/shard_dataset/src-256"
REFERENCE_DATA = ROOT / "data/ref/shard_dataset/v1-crb-redo-12tasks/time-01"
REFERENCE_ARCHIVE = REFERENCE_DATA / "references.npz"
BASE_MODEL = Path("/root/autodl-tmp/pretrained_models/timer-base-84m-84m")
PROBE_ADAPTER = ROOT / "outputs/train/main-ads-v0/checkpoint/adapter"
CONTEXT, HORIZON, WINDOW = 480, 96, 576
PROJECTION_DIM, PROJECTION_SEED = 4096, 0
SOURCE_Q, REFERENCE_Q = 16, 8
SOURCE_DATASET = "src-256-v1"
REFERENCE_DATASET = "time-01-safe-reference-v1"
TRAIN_COUNT_FRACTION = 0.10
SCORE_CHUNK = 4096
CACHE_ROOT = output_path(
    "grad_cache", "outputs/grad_cache/main-crb-v0/q16-src256-time1-q8")
COMPONENT_ROOT = CACHE_ROOT / "components"
SELECTION_ROOT = output_path(
    "selection", "outputs/selection/main-crb-v0/q16-src256-time1-q8")
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-v0/time1"
ARMS = ("ori", "crb", "uni", "reft")


@dataclass(frozen=True)
class ExactQCRBConfig(CRBConfig):
    q: int = SOURCE_Q

    def budgets(self, positions: int) -> tuple[int, int, int]:
        if positions <= 0:
            return 0, 0, 0
        if positions == 1:
            return 1, 0, 0
        total = min(positions, max(2, int(math.ceil(positions / self.q))))
        extra = total - 2
        cover = 2 + (extra + 1) // 2
        split = extra // 2
        return cover, 0, split


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(path)


def load_source_fragments():
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="ascii"))
    rows = config["trajectories"]
    values = np.load(SOURCE_DATA / "values.npy", mmap_mode="r")
    offsets = np.asarray(np.load(SOURCE_DATA / "offsets.npy", mmap_mode="r"),
                         dtype=np.int64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    if len(rows) != 256 or len(offsets) != 257:
        raise ValueError("src-256 must contain 256 registered trajectories")
    if not np.array_equal(np.diff(offsets), lengths) or int(offsets[-1]) != len(values):
        raise ValueError("src-256 values and offsets do not match the registered config")
    legal = lengths - WINDOW + 1
    if np.any(legal <= 0):
        raise ValueError("src-256 contains an empty legal window population")
    fragments = [
        TimeSeriesFragment(np.asarray(values[int(offsets[i]):int(offsets[i + 1])]).copy(),
                           entity_id=str(rows[i].get("series_id", i)),
                           fragment_id=i)
        for i in range(len(rows))
    ]
    counts = np.asarray([len(fragment.valid_starts(CONTEXT, HORIZON))
                         for fragment in fragments], dtype=np.int64)
    if not np.array_equal(counts, legal):
        raise ValueError("src-256 legal window counts do not match")
    return rows, fragments, values, offsets, counts


def load_reference_fragments():
    manifest = json.loads((REFERENCE_DATA / "manifest.json").read_text(encoding="ascii"))
    if (manifest.get("task_id") != "time-01" or manifest.get("fragments") != 10
            or manifest.get("legal_windows") != 3250
            or manifest.get("answer_values_included") is not False):
        raise ValueError("safe TIME1 reference manifest differs from the registered contract")
    with np.load(REFERENCE_ARCHIVE, allow_pickle=False) as archive:
        values, offsets, ids, entities = (
            archive["values"], archive["offsets"], archive["ids"], archive["entities"])
    if len(ids) != 10 or len(offsets) != 11:
        raise ValueError("TIME1 reference archive must contain ten safe segments")
    fragments = [
        TimeSeriesFragment(values[int(offsets[i]):int(offsets[i + 1])].copy(),
                           entity_id=str(entities[i]), fragment_id=int(ids[i]))
        for i in range(len(ids))
    ]
    counts = np.asarray([len(fragment.valid_starts(CONTEXT, HORIZON))
                         for fragment in fragments], dtype=np.int64)
    if int(counts.sum()) != 3250:
        raise ValueError("TIME1 safe reference legal window count differs from 3250")
    return fragments, counts


def exact_uniform_positions(count: int, q: int) -> np.ndarray:
    target = max(2, int(math.ceil(count / q))) if count > 1 else 1
    target = min(count, target)
    positions = np.rint(np.linspace(0, count - 1, target)).astype(np.int64)
    if len(np.unique(positions)) != target:
        raise RuntimeError("uniform anchor positions contain duplicates")
    return positions


def make_request(dataset: str, fragment: TimeSeriesFragment, positions: np.ndarray):
    starts = fragment.valid_starts(CONTEXT, HORIZON)[np.asarray(positions, dtype=np.int64)]
    return WindowRequest(
        dataset,
        np.column_stack((np.full(len(starts), fragment.fragment_id, dtype=np.int64),
                         starts.astype(np.int64))),
        lambda rows, fragment=fragment, starts=starts: fragment.data[
            (starts[np.asarray(rows, dtype=np.int64)] - 1)[:, None]
            + np.arange(WINDOW, dtype=np.int64)[None, :]
        ].astype(np.float32, copy=False),
    )


class UnitDirectionComputer:
    def __init__(self, cache, dataset, logger, expected_rows):
        self.base = FragmentGradients(cache, dataset)
        self.logger = logger
        self.expected_rows = int(expected_rows)
        self.done_rows = 0
        self.started = time.monotonic()
        self.last_log_count = 0

    def __call__(self, fragment, starts):
        return self.compute_many([(fragment, starts)])

    def compute_many(self, entries):
        gradients = self.base.compute_many(entries)
        norms = np.linalg.norm(gradients, axis=1)
        directions = np.divide(
            gradients, norms[:, None], out=np.zeros_like(gradients),
            where=norms[:, None] > 0)
        self.done_rows += len(gradients)
        if (self.done_rows - self.last_log_count >= 8192
                or self.done_rows >= self.expected_rows):
            elapsed = time.monotonic() - self.started
            eta = elapsed * max(self.expected_rows - self.done_rows, 0) / max(self.done_rows, 1)
            self.logger.info("stage=crb-gradient-progress rows=%d/%d elapsed=%s eta=%s",
                             self.done_rows, self.expected_rows,
                             format_duration(elapsed), format_duration(eta))
            self.last_log_count = self.done_rows
        return directions.astype(np.float32, copy=False)


def build_probe(logger):
    from peft import PeftModel
    from gradient.timerxl import TimerXLGradientEngine
    from rage.experiments.full_pool_models import build_model

    if not BASE_MODEL.is_dir():
        raise FileNotFoundError("TimerXL base checkpoint is missing: " + str(BASE_MODEL))
    if not (PROBE_ADAPTER / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("main-ads-v0 LoRA probe checkpoint is missing")
    engine, _probe, identity = build_model(
        "timerxl-84m", model_path=str(BASE_MODEL), device="cuda",
        context_length=CONTEXT, prediction_length=HORIZON)
    engine.model = PeftModel.from_pretrained(
        engine.model, str(PROBE_ADAPTER), is_trainable=True)
    engine.adapter._model = engine.model
    engine.model.eval()
    probe = TimerXLGradientEngine(
        engine, model_key=identity["model_id"] + "@" + identity["revision"],
        projection_dimension=PROJECTION_DIM, projection_seed=PROJECTION_SEED)
    probe.execution = SelectionExecution(
        gradient_batch=128, gradient_workers=1, memory_fraction=0.55, cpu_threads=4,
        gradient_vjp_batch=128)
    probe.max_gradient_batch = 128
    logger.info("stage=model-ready model=timerxl-84m context=%d horizon=%d projection=%d seed=%d gradient_batch=%d vjp_batch=%d probe_checkpoint=%s",
                CONTEXT, HORIZON, PROJECTION_DIM, PROJECTION_SEED,
                probe.execution.gradient_batch, probe.execution.gradient_vjp_batch, PROBE_ADAPTER)
    return engine, probe, identity


def prepare_cache_positions(cache, dataset, fragments, positions_by_fragment, logger):
    total = sum(len(value) for value in positions_by_fragment)
    done = 0
    started = time.monotonic()
    for fragment, positions in zip(fragments, positions_by_fragment):
        positions = np.asarray(positions, dtype=np.int64)
        for first in range(0, len(positions), 1024):
            request = make_request(dataset, fragment, positions[first:first + 1024])
            cache.get_many([request])
            done += len(request.ids)
            if done == total or done % 8192 < len(request.ids):
                elapsed = time.monotonic() - started
                eta = elapsed * max(total - done, 0) / max(done, 1)
                logger.info("stage=additional-gradient-progress dataset=%s rows=%d/%d elapsed=%s eta=%s",
                            dataset, done, total, format_duration(elapsed),
                            format_duration(eta))


def collect_union_keys(cache):
    keys = sorted(cache.rows)
    if not keys:
        raise ValueError("gradient cache contains no prepared gradients")
    return keys


def write_direction_components(cache, union_keys, source_counts, source_fragments,
                               reference_counts, reference_fragments, logger):
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    row_lookup = {key: row for row, key in enumerate(union_keys)}
    total = len(union_keys)
    direction_path = COMPONENT_ROOT / "directions_fp16.npy"
    magnitude_path = COMPONENT_ROOT / "magnitudes_fp32.npy"
    ids_path = COMPONENT_ROOT / "gradient_rows.npz"
    directions = np.lib.format.open_memmap(
        direction_path, mode="w+", dtype=np.float16,
        shape=(total, PROJECTION_DIM))
    magnitudes = np.lib.format.open_memmap(
        magnitude_path, mode="w+", dtype=np.float32, shape=(total,))
    datasets = np.asarray([key[0] for key in union_keys], dtype=str)
    window_ids = np.asarray([[key[1], key[2]] for key in union_keys], dtype=np.int64)
    np.savez_compressed(ids_path, datasets=datasets, window_ids=window_ids)
    processed = 0
    for shard_index, shard_path in enumerate(cache.shards):
        with np.load(shard_path, allow_pickle=False) as payload:
            dataset = str(payload["dataset"])
            shard_ids = np.asarray(payload["window_ids"], dtype=np.int64)
        raw = cache._decoded(shard_path).astype(np.float32, copy=False)
        if raw.ndim != 2 or raw.shape != (len(shard_ids), PROJECTION_DIM):
            raise ValueError("gradient shard shape differs from the registered projection")
        norms = np.linalg.norm(raw, axis=1)
        unit = np.divide(raw, norms[:, None], out=np.zeros_like(raw),
                         where=norms[:, None] > 0)
        fragment_ids = shard_ids[:, 0]
        rows = np.fromiter(
            (row_lookup[(dataset, int(fragment_id), int(start))]
             for fragment_id, start in shard_ids),
            dtype=np.int64, count=len(shard_ids))
        directions[rows] = unit.astype(np.float16)
        magnitudes[rows] = norms.astype(np.float32)
        processed += len(shard_ids)
        if processed == total or processed % 16384 < len(shard_ids):
            logger.info("stage=direction-decomposition shards=%d/%d rows=%d/%d",
                        shard_index + 1, len(cache.shards), processed, total)
    directions.flush()
    magnitudes.flush()
    del directions, magnitudes
    logger.info("stage=direction-components-complete rows=%d directions=%s magnitudes=%s",
                total, direction_path, magnitude_path)
    return row_lookup, direction_path, magnitude_path, ids_path


def _normalize_mean(left, right):
    mean = (np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32)) * 0.5
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else np.zeros_like(mean)


def write_block_assets(method, fragments, positions_by_fragment, row_lookup,
                       directions_path, block_pairs_by_fragment, dataset,
                       output_root, logger):
    output_root.mkdir(parents=True, exist_ok=True)
    blocks = []
    for fragment, positions, pairs in zip(
            fragments, positions_by_fragment, block_pairs_by_fragment):
        position_to_row = {}
        starts = fragment.valid_starts(CONTEXT, HORIZON)
        for position in np.asarray(positions, dtype=np.int64):
            key = (dataset, fragment.fragment_id, int(starts[int(position)]))
            position_to_row[int(position)] = row_lookup[key]
        for left, right in pairs:
            if int(left) not in position_to_row or int(right) not in position_to_row:
                raise ValueError("block endpoint is absent from the anchor resource")
            blocks.append((int(fragment.fragment_id), int(left), int(right),
                           (float(left) + float(right)) * 0.5,
                           position_to_row[int(left)], position_to_row[int(right)]))
    block_ids_path = output_root / (method + "_block_index.npz")
    block_directions_path = output_root / (method + "_block_directions_fp16.npy")
    block_directions = np.lib.format.open_memmap(
        block_directions_path, mode="w+", dtype=np.float16,
        shape=(len(blocks), PROJECTION_DIM))
    left_rows = np.asarray([row[4] for row in blocks], dtype=np.int64)
    right_rows = np.asarray([row[5] for row in blocks], dtype=np.int64)
    directions = np.load(directions_path, mmap_mode="r")
    for first in range(0, len(blocks), 256):
        stop = min(first + 256, len(blocks))
        left = directions[left_rows[first:stop]].astype(np.float32)
        right = directions[right_rows[first:stop]].astype(np.float32)
        means = (left + right) * 0.5
        norms = np.linalg.norm(means, axis=1)
        means = np.divide(means, norms[:, None], out=np.zeros_like(means),
                          where=norms[:, None] > 0)
        block_directions[first:stop] = means.astype(np.float16)
        if stop == len(blocks) or stop % 16384 < 256:
            logger.info("stage=block-directions method=%s blocks=%d/%d",
                        method, stop, len(blocks))
    block_directions.flush()
    del block_directions
    np.savez_compressed(
        block_ids_path,
        fragment_ids=np.asarray([row[0] for row in blocks], dtype=np.int32),
        left_positions=np.asarray([row[1] for row in blocks], dtype=np.int32),
        right_positions=np.asarray([row[2] for row in blocks], dtype=np.int32),
        centers=np.asarray([row[3] for row in blocks], dtype=np.float32),
        left_gradient_rows=left_rows,
        right_gradient_rows=right_rows,
    )
    return block_ids_path, block_directions_path, len(blocks)


def interpolate_directions(centers, directions, positions):
    positions = np.asarray(positions, dtype=np.float32)
    right = np.searchsorted(centers, positions, side="right")
    left = np.clip(right - 1, 0, len(centers) - 1)
    right = np.clip(right, 0, len(centers) - 1)
    denominator = centers[right] - centers[left]
    alpha = np.divide(
        positions - centers[left], denominator,
        out=np.zeros_like(positions), where=denominator > 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    values = (directions[left] * (1.0 - alpha[:, None])
              + directions[right] * alpha[:, None])
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values),
                     where=norms[:, None] > 0)


def score_and_select(method, source_fragments, source_counts, values, value_offsets,
                     block_index_path, block_directions_path, reference_directions_path,
                     logger):
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("direction cosine scoring requires CUDA BF16 support")
    with np.load(block_index_path, allow_pickle=False) as archive:
        block_fragment_ids = archive["fragment_ids"]
        block_centers = archive["centers"]
    block_directions = np.load(block_directions_path, mmap_mode="r")
    reference_directions = np.load(reference_directions_path, mmap_mode="r").astype(np.float32)
    reference_vectors = torch.as_tensor(
        reference_directions, dtype=torch.bfloat16, device="cuda")
    total = int(source_counts.sum())
    score_values = np.empty(total, dtype=np.float32)
    global_offsets = np.cumsum(np.r_[0, source_counts])
    started = time.monotonic()
    for fragment_index, (fragment, count) in enumerate(zip(source_fragments, source_counts)):
        rows = np.flatnonzero(block_fragment_ids == fragment.fragment_id)
        centers = block_centers[rows]
        reps = block_directions[rows].astype(np.float32)
        if not len(reps):
            raise ValueError("a source trajectory has no CRB/uni blocks")
        first_position = int(global_offsets[fragment_index])
        for first in range(0, int(count), SCORE_CHUNK):
            stop = min(first + SCORE_CHUNK, int(count))
            positions = np.arange(first, stop, dtype=np.float32)
            interpolated = interpolate_directions(centers, reps, positions)
            source_vectors = torch.as_tensor(
                interpolated, dtype=torch.bfloat16, device="cuda")
            scores = torch.matmul(source_vectors, reference_vectors.T).float()
            score_values[first_position + first:first_position + stop] = (
                scores.max(dim=1).values.cpu().numpy())
            if stop == int(count):
                completed = int(global_offsets[fragment_index + 1])
                elapsed = time.monotonic() - started
                eta = elapsed * (total - completed) / max(completed, 1)
                logger.info("stage=cosine-score-progress method=%s windows=%d/%d elapsed=%s eta=%s",
                            method, completed, total, format_duration(elapsed),
                            format_duration(eta))
            del interpolated, source_vectors, scores
        del reps
    budget = int(math.floor(TRAIN_COUNT_FRACTION * total))
    if budget <= 0 or budget > total:
        raise ValueError("source training-window budget is invalid")
    threshold = float(np.partition(score_values, total - budget)[total - budget])
    above = np.flatnonzero(score_values > threshold)
    tied = np.flatnonzero(score_values == threshold)
    remaining = budget - len(above)
    chosen = np.concatenate((above, tied[:remaining]))
    if len(chosen) != budget:
        raise RuntimeError("deterministic top-score selection returned the wrong count")
    chosen.sort()
    ids = np.empty((budget, 2), dtype=np.int64)
    for row, global_index in enumerate(chosen):
        owner = int(np.searchsorted(global_offsets[1:], global_index, side="right"))
        local = int(global_index - global_offsets[owner])
        ids[row] = (source_fragments[owner].fragment_id, local + 1)
    out = SELECTION_ROOT / method
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "selected_window_ids.npy", ids)
    windows_path = out / "training_windows.npz"
    windows = np.empty((budget, WINDOW), dtype=np.float32)
    for row, (fragment_id, physical_start) in enumerate(ids):
        fragment = source_fragments[int(fragment_id)]
        windows[row] = fragment.extract_window(
            int(physical_start), CONTEXT, HORIZON)
        if row and row % 65536 == 0:
            logger.info("stage=window-materialization method=%s rows=%d/%d",
                        method, row, budget)
    np.savez_compressed(windows_path, windows=windows)
    del windows
    manifest = {
        "experiment": "main-crb-v0",
        "batch": "q16-src256-time1-q8",
        "method": method,
        "source_pool": "src-256",
        "source_window_count": total,
        "selection_budget_fraction": TRAIN_COUNT_FRACTION,
        "selected_window_count": budget,
        "score": "max cosine to TIME1 reference block directions",
        "direction_interpolation": "linear between adjacent block-center direction representatives; normalize each interpolated vector",
        "block_representative": "unit-normalized mean of the left and right anchor unit directions",
        "score_precision": "CUDA BF16 matrix multiplication",
        "score_cutoff": threshold,
        "tie_break": "ascending source global window index",
        "selected_ids": str(windows_path.parent / "selected_window_ids.npy"),
        "training_windows": str(windows_path),
        "block_index": str(block_index_path),
        "block_directions": str(block_directions_path),
    }
    write_json(out / "selection.json", manifest)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=selection-complete method=%s selected=%d/%d output=%s",
                method, budget, total, out)
    return manifest


def _make_training_configs(logger):
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    recipe = {
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
    training_windows = {
        "crb": SELECTION_ROOT / "crb/training_windows.npz",
        "reft": SELECTION_ROOT / "reference/training_windows.npz",
        "uni": SELECTION_ROOT / "uni/training_windows.npz",
    }
    for arm in ("crb", "reft", "uni"):
        train_config = {
            "schema_version": 1,
            "run_id": "main-crb-v0-time1-" + arm,
            "model": "timerxl-84m",
            "model_path": str(BASE_MODEL),
            "device": "cuda",
            "data": {
                "id": "src-256-direction-selected" if arm != "reft"
                      else "time-01-safe-reference-all-windows",
                "windows": str(training_windows[arm]),
                "windows_key": "windows",
                "context_length": CONTEXT,
                "prediction_length": HORIZON,
            },
            "training": {
                "steps": 4000,
                "batch_size": 32,
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
                "seed": 0,
                "log_every": 100,
                "schedule_total_steps": 4000,
                "recipe": recipe,
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "output_dir": "outputs/train/main-crb-v0/q16-src256-time1-q8/" + arm,
        }
        train_path = CONFIG_ROOT / (arm + "-train.yaml")
        train_path.write_text(yaml.safe_dump(train_config, sort_keys=False),
                              encoding="ascii")
        eval_config = {
            "schema_version": 1,
            "checkpoint": "outputs/train/main-crb-v0/q16-src256-time1-q8/"
                          + arm + "/checkpoint",
            "evaluation_config": "configs/eval/time/v1-rage-ads-time-01.yaml",
            "output_dir": "outputs/eval/main-crb-v0/q16-src256-time1-q8/" + arm,
            "resume": True,
        }
        eval_path = CONFIG_ROOT / (arm + "-eval.yaml")
        eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False),
                             encoding="ascii")
        pipeline = {
            "schema_version": 1,
            "run_id": "main-crb-v0-time1-" + arm,
            "stages": [
                {"name": "train", "config": str(train_path.relative_to(ROOT))},
                {"name": "eval", "config": str(eval_path.relative_to(ROOT))},
            ],
        }
        (CONFIG_ROOT / (arm + "-pipeline.yaml")).write_text(
            yaml.safe_dump(pipeline, sort_keys=False), encoding="ascii")
    ori_eval = {
        "schema_version": 1,
        "checkpoint": "outputs/train/main-crb-v0/q16-src256-time1-q8/ori/checkpoint",
        "evaluation_config": "configs/eval/time/v1-rage-ads-time-01.yaml",
        "output_dir": "outputs/eval/main-crb-v0/q16-src256-time1-q8/ori",
        "resume": True,
    }
    ori_eval_path = CONFIG_ROOT / "ori-eval.yaml"
    ori_eval_path.write_text(yaml.safe_dump(ori_eval, sort_keys=False), encoding="ascii")
    ori_pipeline = {
        "schema_version": 1,
        "run_id": "main-crb-v0-time1-ori",
        "stages": [{"name": "eval", "config": str(ori_eval_path.relative_to(ROOT))}],
    }
    (CONFIG_ROOT / "ori-pipeline.yaml").write_text(
        yaml.safe_dump(ori_pipeline, sort_keys=False), encoding="ascii")
    ori_root = output_path(
        "train", "outputs/train/main-crb-v0/q16-src256-time1-q8/ori")
    checkpoint_view = ori_root / "checkpoint"
    checkpoint_view.mkdir(parents=True, exist_ok=True)
    for item in BASE_MODEL.iterdir():
        target = checkpoint_view / item.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    write_json(checkpoint_view / "rage_checkpoint_manifest.json", {
        "schema_version": 1,
        "experiment_version": "main-crb-v0/q16-src256-time1-q8/ori",
        "model_key": "timerxl-84m",
        "model": "thuml/timer-base-84m",
        "training_step": 0,
        "precision": "bfloat16",
        "training": "not_applicable",
    })
    (ori_root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=tool-configs-ready root=%s", CONFIG_ROOT)


def prepare(logger):
    marker = CACHE_ROOT / ".complete"
    if marker.is_file():
        logger.info("stage=prepare-skip reason=complete output=%s", CACHE_ROOT)
        return json.loads((CACHE_ROOT / "resource_manifest.json").read_text(encoding="ascii"))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    _, source_fragments, source_values, source_offsets, source_counts = load_source_fragments()
    reference_fragments, reference_counts = load_reference_fragments()
    source_crb_positions = []
    source_crb_pairs = []
    source_q16_totals = []
    logger.info("stage=config source=src-256 tracks=%d windows=%d q=%d reference=TIME1 segments=%d windows=%d q=%d",
                len(source_fragments), int(source_counts.sum()), SOURCE_Q,
                len(reference_fragments), int(reference_counts.sum()), REFERENCE_Q)
    engine = probe = cache = None
    try:
        engine, probe, identity = build_probe(logger)
        os.environ["RAGE_GRADIENT_CACHE_ROOT"] = str(CACHE_ROOT / "raw_bf16")
        cache = GradientCache(
            CACHE_ROOT, probe,
            model_identity={**identity, "probe_checkpoint": str(PROBE_ADAPTER),
                            "probe_training_manifest": str(
                                ROOT / "outputs/train/main-ads-v0/checkpoint/training_manifest.json")},
            data_fields={
                "experiment": "main-crb-v0/q16-src256-time1-q8",
                "source_pool": "src-256",
                "reference_task": "TIME1/CPHL/15T",
                "reference_archive": str(REFERENCE_ARCHIVE),
                "probe_checkpoint": str(PROBE_ADAPTER),
                "projection_dimension": PROJECTION_DIM,
                "projection_seed": PROJECTION_SEED,
            },
            context_length=CONTEXT, prediction_length=HORIZON)
        expected_crb = sum(max(2, int(math.ceil(int(count) / SOURCE_Q)))
                           for count in source_counts)
        directional = UnitDirectionComputer(
            cache, SOURCE_DATASET, logger, expected_crb)
        config = ExactQCRBConfig(
            q=SOURCE_Q, context_length=CONTEXT, prediction_length=HORIZON,
            cover_density=0.0, error_density=0.0, split_density=0.0)
        logger.info("stage=crb-source-start q=%d ratio=1:0:1 cover=longest-bisection cancel=recursive-min-kappa",
                    SOURCE_Q)
        started = time.monotonic()
        with selection_execution(probe):
            crb_results = CRBEngineMulti(config, directional).process_fragments(source_fragments)
        for fragment, count, result in zip(source_fragments, source_counts, crb_results):
            positions = np.asarray(sorted(result.anchors), dtype=np.int64)
            pairs = [(int(block.left_index), int(block.right_index))
                     for block in result.blocks]
            expected = min(int(count), max(2, int(math.ceil(int(count) / SOURCE_Q))))
            if len(positions) != expected:
                raise RuntimeError("exact q16 CRB anchor count differs for fragment %d"
                                   % fragment.fragment_id)
            source_crb_positions.append(positions)
            source_crb_pairs.append(pairs)
            source_q16_totals.append({
                "fragment_id": int(fragment.fragment_id),
                "valid_windows": int(count),
                "anchor_count": int(len(positions)),
                "cover_count": int(result.cover_computations),
                "error_count": int(result.error_computations),
                "split_count": int(result.split_computations),
                "block_count": int(len(pairs)),
            })
            result.anchors.clear()
            result.blocks.clear()
        del crb_results, directional
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("stage=crb-source-complete anchors=%d elapsed=%s",
                    sum(len(value) for value in source_crb_positions),
                    format_duration(time.monotonic() - started))

        source_uni_positions = [
            exact_uniform_positions(int(count), SOURCE_Q) for count in source_counts
        ]
        reference_positions = [
            exact_uniform_positions(int(count), REFERENCE_Q) for count in reference_counts
        ]
        logger.info("stage=additional-gradient-requests source_uni=%d reference=%d",
                    sum(len(row) for row in source_uni_positions),
                    sum(len(row) for row in reference_positions))
        with selection_execution(probe):
            prepare_cache_positions(cache, SOURCE_DATASET, source_fragments,
                                    source_uni_positions, logger)
            prepare_cache_positions(cache, REFERENCE_DATASET, reference_fragments,
                                    reference_positions, logger)
        logger.info("stage=gradient-cache-complete unique_windows=%d computed_windows=%d",
                    len(cache.rows), cache.computed_windows)
        union_keys = collect_union_keys(cache)
        row_lookup, directions_path, magnitudes_path, ids_path = write_direction_components(
            cache, union_keys, source_counts, source_fragments,
            reference_counts, reference_fragments, logger)
        cache.close()
        cache = None
        del probe, engine
        probe = engine = None
        gc.collect()
        torch.cuda.empty_cache()

        source_crb_rows = [
            np.asarray([row_lookup[(SOURCE_DATASET, fragment.fragment_id,
                                    int(fragment.valid_starts(CONTEXT, HORIZON)[position]))]
                        for position in positions], dtype=np.int64)
            for fragment, positions in zip(source_fragments, source_crb_positions)
        ]
        source_uni_rows = [
            np.asarray([row_lookup[(SOURCE_DATASET, fragment.fragment_id,
                                    int(fragment.valid_starts(CONTEXT, HORIZON)[position]))]
                        for position in positions], dtype=np.int64)
            for fragment, positions in zip(source_fragments, source_uni_positions)
        ]
        reference_rows = [
            np.asarray([row_lookup[(REFERENCE_DATASET, fragment.fragment_id,
                                    int(fragment.valid_starts(CONTEXT, HORIZON)[position]))]
                        for position in positions], dtype=np.int64)
            for fragment, positions in zip(reference_fragments, reference_positions)
        ]
        (SELECTION_ROOT / "crb").mkdir(parents=True, exist_ok=True)
        (SELECTION_ROOT / "uni").mkdir(parents=True, exist_ok=True)
        (SELECTION_ROOT / "reference").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            SELECTION_ROOT / "crb/source_anchors.npz",
            fragment_counts=source_counts,
            positions=np.concatenate(source_crb_positions),
            gradient_rows=np.concatenate(source_crb_rows),
        )
        np.savez_compressed(
            SELECTION_ROOT / "uni/source_anchors.npz",
            fragment_counts=source_counts,
            positions=np.concatenate(source_uni_positions),
            gradient_rows=np.concatenate(source_uni_rows),
        )
        np.savez_compressed(
            SELECTION_ROOT / "reference/reference_anchors.npz",
            fragment_counts=reference_counts,
            positions=np.concatenate(reference_positions),
            gradient_rows=np.concatenate(reference_rows),
        )

        crb_block_index, crb_block_directions, crb_block_count = write_block_assets(
            "crb", source_fragments, source_crb_positions, row_lookup, directions_path,
            source_crb_pairs, SOURCE_DATASET, SELECTION_ROOT / "crb", logger)
        uni_pairs = [
            [(int(left), int(right)) for left, right in zip(positions[:-1], positions[1:])]
            for positions in source_uni_positions
        ]
        uni_block_index, uni_block_directions, uni_block_count = write_block_assets(
            "uni", source_fragments, source_uni_positions, row_lookup, directions_path,
            uni_pairs, SOURCE_DATASET, SELECTION_ROOT / "uni", logger)
        reference_pairs = [
            [(int(left), int(right)) for left, right in zip(positions[:-1], positions[1:])]
            for positions in reference_positions
        ]
        ref_block_index, ref_block_directions, ref_block_count = write_block_assets(
            "reference", reference_fragments, reference_positions, row_lookup, directions_path,
            reference_pairs, REFERENCE_DATASET, SELECTION_ROOT / "reference", logger)

        reference_direction_path = SELECTION_ROOT / "reference/reference_block_directions_fp16.npy"
        # Use the q8 reference block means directly for the shared cosine bank.
        if Path(ref_block_directions).resolve() != reference_direction_path.resolve():
            source = np.load(ref_block_directions, mmap_mode="r")
            target = np.lib.format.open_memmap(
                reference_direction_path, mode="w+", dtype=np.float16, shape=source.shape)
            target[:] = source[:]
            target.flush()
            del target, source
        ref_directions = np.load(reference_direction_path, mmap_mode="r")
        if len(ref_directions) != ref_block_count:
            raise RuntimeError("TIME1 q8 reference block count mismatch")

        crb_manifest = score_and_select(
            "crb", source_fragments, source_counts, source_values, source_offsets,
            crb_block_index, crb_block_directions, reference_direction_path, logger)
        uni_manifest = score_and_select(
            "uni", source_fragments, source_counts, source_values, source_offsets,
            uni_block_index, uni_block_directions, reference_direction_path, logger)

        # Materialize all legal TIME1 reference windows for direct reference training.
        reference_output = SELECTION_ROOT / "reference"
        reference_output.mkdir(parents=True, exist_ok=True)
        reference_windows_path = reference_output / "training_windows.npz"
        reference_total = int(reference_counts.sum())
        reference_windows = np.empty((reference_total, WINDOW), dtype=np.float32)
        cursor = 0
        for fragment in reference_fragments:
            starts = fragment.valid_starts(CONTEXT, HORIZON)
            for start in starts:
                reference_windows[cursor] = fragment.extract_window(
                    int(start), CONTEXT, HORIZON)
                cursor += 1
        if cursor != reference_total:
            raise RuntimeError("TIME1 reference training window count differs")
        np.savez_compressed(reference_windows_path, windows=reference_windows)
        del reference_windows
        reference_manifest = {
            "experiment": "main-crb-v0",
            "batch": "q16-src256-time1-q8",
            "method": "reft",
            "reference_dataset": "TIME1/CPHL/15T safe history",
            "segment_count": len(reference_fragments),
            "legal_window_count": reference_total,
            "anchor_interval": REFERENCE_Q,
            "training_windows": str(reference_windows_path),
            "answer_values_included": False,
        }
        write_json(reference_output / "selection.json", reference_manifest)
        (reference_output / ".complete").write_text("complete\n", encoding="ascii")

        _make_training_configs(logger)
        resource_manifest = {
            "experiment": "main-crb-v0",
            "batch": "q16-src256-time1-q8",
            "source_pool": "src-256",
            "source_trajectory_count": len(source_fragments),
            "source_valid_window_count": int(source_counts.sum()),
            "source_q": SOURCE_Q,
            "source_crb_anchor_count": int(sum(len(row) for row in source_crb_positions)),
            "source_uni_anchor_count": int(sum(len(row) for row in source_uni_positions)),
            "source_crb_cover_count": int(sum(row["cover_count"] for row in source_q16_totals)),
            "source_crb_error_count": int(sum(row["error_count"] for row in source_q16_totals)),
            "source_crb_split_count": int(sum(row["split_count"] for row in source_q16_totals)),
            "reference_task": "TIME1/CPHL/15T",
            "reference_segment_count": len(reference_fragments),
            "reference_valid_window_count": int(reference_counts.sum()),
            "reference_q": REFERENCE_Q,
            "reference_anchor_count": int(sum(len(row) for row in reference_positions)),
            "unique_gradient_count": len(union_keys),
            "projection_dimension": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "raw_cache": str(CACHE_ROOT / "raw_bf16"),
            "gradient_ids": str(ids_path),
            "direction_sidecar": str(directions_path),
            "direction_dtype": "float16 unit vectors",
            "magnitude_sidecar": str(magnitudes_path),
            "magnitude_dtype": "float32 L2 norm",
            "magnitude_use": "audit only; excluded from CRB kappa, interpolation, cosine scores, and selection",
            "source_crb_block_count": crb_block_count,
            "source_uni_block_count": uni_block_count,
            "reference_block_count": ref_block_count,
            "source_crb_selection": crb_manifest,
            "source_uni_selection": uni_manifest,
            "reference_training_windows": str(reference_windows_path),
            "per_fragment_source_budget": source_q16_totals,
            "training_steps": 4000,
            "effective_batch_size": 32,
            "micro_batch_size": 8,
            "seed": 0,
            "evaluation_config": "configs/eval/time/v1-rage-ads-time-01.yaml",
        }
        write_json(CACHE_ROOT / "resource_manifest.json", resource_manifest)
        (SELECTION_ROOT / ".complete").write_text("complete\n", encoding="ascii")
        (marker.parent / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=prepare-complete source_crb=%d source_uni=%d reference=%d unique_gradients=%d output=%s",
                    resource_manifest["source_crb_anchor_count"],
                    resource_manifest["source_uni_anchor_count"],
                    resource_manifest["reference_anchor_count"],
                    resource_manifest["unique_gradient_count"], CACHE_ROOT)
        return resource_manifest
    finally:
        if cache is not None:
            cache.close()
        if probe is not None or engine is not None:
            del probe, engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


_ACTIVE = None


def _forward_interrupt(signum, _frame):
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE.poll() is None:
        _ACTIVE.send_signal(signal.SIGINT)


def run_arms(logger):
    global _ACTIVE
    train_root = ROOT / "outputs/train/main-crb-v0/q16-src256-time1-q8"
    eval_root = ROOT / "outputs/eval/main-crb-v0/q16-src256-time1-q8"
    for arm in ARMS:
        train_complete = (train_root / arm / ".complete").is_file()
        eval_complete = (eval_root / arm / ".complete").is_file()
        if eval_complete:
            logger.info("stage=arm-skip arm=%s reason=evaluation-complete", arm)
            continue
        config = CONFIG_ROOT / (arm + "-pipeline.yaml")
        command = [sys.executable, "-u", "-m", "scripts.tools.run",
                   "--config", str(config.relative_to(ROOT))]
        if train_complete and arm != "ori":
            command.extend(["--from-stage", "eval"])
        logger.info("stage=arm-start arm=%s command=%s", arm, command)
        _ACTIVE = subprocess.Popen(command, cwd=ROOT)
        code = _ACTIVE.wait()
        _ACTIVE = None
        if code:
            raise RuntimeError("tool pipeline failed for %s with exit code %d" % (arm, code))
        if not (eval_root / arm / ".complete").is_file():
            raise RuntimeError("TIME1 evaluation completion marker is missing for " + arm)
        logger.info("stage=arm-complete arm=%s output=%s", arm, eval_root / arm)
    output_path("train", train_root).mkdir(parents=True, exist_ok=True)
    output_path("eval", eval_root).mkdir(parents=True, exist_ok=True)
    (train_root / ".complete").write_text("complete\n", encoding="ascii")
    (eval_root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=arms-complete train_output=%s eval_output=%s", train_root, eval_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare"), default="all")
    args = parser.parse_args(argv)
    logger = configure_logger(
        "main_crb_v0_directional_time1", "main-crb-v0",
        "q16-src256-time1-q8", "startup", enable_heartbeat=True)
    started = time.monotonic()
    logger.info("stage=config stage=%s source=src-256 q=16 reference=TIME1 q=8 arms=%s",
                args.stage, ",".join(ARMS))
    manifest = prepare(logger)
    if args.stage == "all":
        run_arms(logger)
    logger.info("stage=complete elapsed=%s output=%s",
                format_duration(time.monotonic() - started), CACHE_ROOT)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _forward_interrupt)
    raise SystemExit(main())

