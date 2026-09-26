#!/usr/bin/env python3
"""Measure whether cu-r1 source supports have a more reference-aligned gradient.

For each usable FEV task this orchestrator performs the same candidate-uniform
ratio (cu-r1) solve used by TS3, samples a configurable number of source and
candidate analysis segments without replacement, and reads every legal segment
from the task's full reference catalog. It then computes projected ChronosTiny
per-example gradients for all three sets under each supplied checkpoint.

The reported direction is the mean projected per-example gradient. The key
quantity is candidate_cosine_to_reference - random_cosine_to_reference;
positive values support the claim that cu-r1 improves gradient alignment.

The main process prepares windows and schedules sample shards. Each worker loads
all requested warm-up models, computes one shard at a time, and writes raw
projected gradients and losses to an output-backed cache. Aggregation reads the
cache incrementally, so the complete gradient tensor is never retained in the
main process. The source Feature Bundle is read once and reused for every task;
reference feature computation is released after each task to bound memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paths import project_path
from gradient.chronos import ChronosTinyConfig, ChronosTinyGradientEngine
from ts3.core.bundle.bench import reference_bench_adapter
from ts3.core.bundle.reader import ReferenceBundleReader, SourceBundleReader
from ts3.core.policy.ratio_candidate_uniform import RatioCandidateUniformSegmentPlanner
from ts3.core.runtime.segment_model import CompactRuntimeDatasetComputationPart, RuntimeDatasetComputationPart
from ts3.core.runtime.windowing import legal_start_count
from ts3.core.runtime.window_accessor import WindowAccessor
from ts3.core.sampling.parts import PseudoWindowsStartsRepresentation
from ts3.core.transfer.config import OTConfig
from ts3.core.transfer.feature_space import ReferenceAnalysisSegmentFeatureSpace, SourceAnalysisSegmentFeatureSpace
from ts3.spec.catalogs.models import ReferenceCatalog, SourceCatalog
from ts3.spec.catalogs.reader import CatalogReader
from ts3.spec.models.data_model import RuntimeSourceDataset, Trajectory, WindowLocator
from ts3.spec.models.training_protocol import WindowEligibilitySpecification

LOG = logging.getLogger("exp_filter_grad")
DEFAULT_CU_CONFIG = "configs/scripts/run.46m.cu-ratio.62k.l1024s1024.yaml"
DEFAULT_TASK_FILE = "configs/eval/fev/profile-10b-local.yaml"
DEFAULT_SOURCE_CATALOG = "configs/catalog/src/profile-10b-segl1024s1024.yaml"
DEFAULT_REFERENCE_CATALOG_ROOT = "configs/catalog/ref/profile-10b-segl1024s1024/fev"
DEFAULT_CHECKPOINT_ROOT = "outputs/models/warmup/chronos-tiny-from-scratch"
DEFAULT_OUTPUT = "outputs/exp/filter-grad-profile-10b-segl1024s1024-20260829"
DEFAULT_CHECKPOINTS = [f"{DEFAULT_CHECKPOINT_ROOT}/seed-{seed}/model" for seed in (0, 3, 8, 42)]
DEFAULT_EXPERIMENT_CONFIG = "configs/scripts/exp_filter_grad.yaml"
DEFAULT_TASK_SELECTION_CONFIG = "configs/experiments/chronos46m-main-v2-s0/cu-r1.eval.yaml"


@dataclass
class TaskWindows:
    index: int
    dataset_config: str
    source: np.ndarray
    candidate: np.ndarray
    reference: np.ndarray
    candidate_support_count: int
    source_segment_count: int
    reference_segment_count: int
    filter_diagnostics: dict[str, Any]


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="optional experiment-parameter YAML; omitted means the built-in current defaults")
    parser.add_argument("--cu-config", default=DEFAULT_CU_CONFIG, help="config carrying the current cu-r1 OT settings")
    parser.add_argument("--task-file", default=DEFAULT_TASK_FILE)
    parser.add_argument("--source-catalog")
    parser.add_argument("--reference-catalog-root")
    parser.add_argument("--task-selection-config", default=DEFAULT_TASK_SELECTION_CONFIG, help="main-experiment eval YAML declaring the legal FEV task indices")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--sample-segments", type=int, default=1000, help="source and candidate segment sample size")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--gradient-batch-size", type=int, default=32)
    parser.add_argument("--projection-dimension", type=int, default=4096)
    parser.add_argument("--projection-repeats", type=int, default=1)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--token-chunk-size", type=int, default=8)
    parser.add_argument("--mmap-cache-capacity", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task-indices", type=int, nargs="*", help="restrict the task list; default is all usable tasks")
    parser.add_argument("--max-tasks", type=int, help="development/test limit after eligibility filtering")
    parser.add_argument("--save-sketches", action="store_true", help="compatibility flag; raw gradients are always cached under outputs")
    parser.add_argument("--allow-checkpoint-subset", action="store_true", help="development-only: permit fewer than four checkpoints")
    parser.add_argument("--parallel-workers", type=int, default=4, help="sample-shard workers run concurrently under the main process")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepared-file", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--warmup-root", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-seeds", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="validate paths, task eligibility, and checkpoint list only")
    return parser


def _load_experiment_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    config_path = project_path(args.config)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("experiment config must be a mapping: %s" % config_path)
    value = value.get("experiment", value)
    if not isinstance(value, dict):
        raise SystemExit("experiment config experiment section must be a mapping")
    supported = {
        "cu_config", "task_file", "source_catalog", "reference_catalog_root", "task_selection_config", "output_dir", "checkpoints",
        "sample_segments", "sample_seed", "gradient_batch_size", "projection_dimension", "projection_repeats",
        "projection_seed", "token_chunk_size", "mmap_cache_capacity", "device", "task_indices", "max_tasks",
        "save_sketches", "parallel_workers",
    }
    unknown = sorted(set(value) - supported)
    if unknown:
        raise SystemExit("unsupported experiment config keys: %s" % ", ".join(unknown))
    for key, item in value.items():
        setattr(args, key, item)
    return args


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit("%s must be a positive integer" % name)


def _load_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, list[dict[str, Any]], dict[str, Any]]:
    cu_path = project_path(args.cu_config)
    task_path = project_path(args.task_file)
    cu_config = yaml.safe_load(cu_path.read_text(encoding="utf-8"))
    task_config = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(cu_config, dict) or not isinstance(task_config, dict):
        raise SystemExit("YAML configs must be mappings")
    assets = cu_config.get("assets", {})
    source_value = args.source_catalog or assets.get("source_catalog") or DEFAULT_SOURCE_CATALOG
    reference_value = args.reference_catalog_root or assets.get("reference_catalog_root") or DEFAULT_REFERENCE_CATALOG_ROOT
    source_path = project_path(source_value)
    reference_root = project_path(reference_value)
    if not args.source_catalog and not source_path.is_file():
        source_path = project_path(DEFAULT_SOURCE_CATALOG)
    if not args.reference_catalog_root and not reference_root.is_dir():
        reference_root = project_path(DEFAULT_REFERENCE_CATALOG_ROOT)
    tasks = task_config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("task file has no tasks")
    if not source_path.is_file():
        raise SystemExit("missing source catalog: %s" % source_path)
    if not reference_root.is_dir():
        raise SystemExit("missing reference catalog root: %s" % reference_root)
    return source_path, reference_root, task_path, tasks, cu_config


def _selected_task_indices(args: argparse.Namespace) -> list[int] | None:
    if args.task_indices is not None:
        return list(args.task_indices)
    if not args.task_selection_config:
        return None
    selection_path = project_path(args.task_selection_config)
    value = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("bench_config"), dict):
        raise SystemExit("task selection config must contain bench_config: %s" % selection_path)
    bench_config = value["bench_config"]
    indices = bench_config.get("task_indices")
    expected = bench_config.get("expected_tasks")
    if not isinstance(indices, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in indices):
        raise SystemExit("task selection config has invalid bench_config.task_indices: %s" % selection_path)
    if expected is not None and int(expected) != len(indices):
        raise SystemExit("task selection config expected_tasks does not match task_indices: %s" % selection_path)
    return list(indices)


def _cu_settings(cu_config: dict[str, Any]) -> tuple[OTConfig, int, int]:
    sampling = cu_config.get("sampling", {})
    if not isinstance(sampling, dict):
        raise SystemExit("cu config sampling must be a mapping")
    ratio_spec = None
    for spec in sampling.values():
        if not isinstance(spec, dict) or spec.get("policy") != "candidate-uniform-ratio":
            continue
        if float(spec.get("support_ratio", -1.0)) == 1.0:
            ratio_spec = spec
            break
    if ratio_spec is None:
        raise SystemExit("cu config does not contain a candidate-uniform-ratio support_ratio=1 method")
    ot_mapping = ratio_spec.get("ot") or sampling.get("ot")
    if not isinstance(ot_mapping, dict):
        raise SystemExit("cu-r1 config is missing OT settings")
    return OTConfig(**dict(ot_mapping)), int(ratio_spec.get("support_sample_seed", 0)), int(cu_config.get("data", {}).get("context_length", 512))


def _reference_path(root: Path, dataset_config: str) -> Path:
    from ts3.spec.catalogs.discovery import discover_long_reference_catalogs, resolve_long_reference_catalog_dir
    directory = resolve_long_reference_catalog_dir(root, str(dataset_config))
    paths = discover_long_reference_catalogs(directory)
    if len(paths) != 1:
        raise ValueError("expected one catalog header in %s" % directory)
    return paths[0]

def _runtime_source_from_computation(computation: RuntimeDatasetComputationPart | CompactRuntimeDatasetComputationPart) -> RuntimeSourceDataset:
    if isinstance(computation, CompactRuntimeDatasetComputationPart):
        trajectories = tuple(Trajectory(locator.catalog_trajectory_id, locator, int(length)) for locator, length in zip(computation.trajectory_locators, computation.trajectory_lengths))
    else:
        trajectories = tuple(Trajectory(item.locator.catalog_trajectory_id, item.locator, item.length) for item in computation.trajectories)
    quality = {item.catalog_trajectory_id: 1.0 for item in trajectories}
    mass = {item.catalog_trajectory_id: 1.0 / len(trajectories) for item in trajectories}
    return RuntimeSourceDataset(trajectories, {item.catalog_trajectory_id: item for item in trajectories}, quality, mass, {"dataset_kind": "window-read-only"})


def _segment_rows(computation: RuntimeDatasetComputationPart | CompactRuntimeDatasetComputationPart, positions: np.ndarray) -> list[tuple[Any, int, int, int]]:
    values = np.asarray(positions, dtype=np.int64)
    if isinstance(computation, CompactRuntimeDatasetComputationPart):
        offsets = computation.trajectory_offsets
        trajectory_indices = np.searchsorted(offsets, values, side="right") - 1
        return [
            (computation.trajectory_locators[trajectory_index], int(computation.trajectory_lengths[trajectory_index]), int(computation.segment_starts[position]), int(computation.segment_lengths[position]))
            for position, trajectory_index in zip(values.tolist(), trajectory_indices.tolist())
        ]
    flattened = [(segment.trajectory_locator, trajectory.length, segment.start, segment.length) for trajectory in computation.trajectories for segment in trajectory.segments]
    return [flattened[int(position)] for position in values]


def _windows_for_segments(accessor: WindowAccessor, computation: RuntimeDatasetComputationPart | CompactRuntimeDatasetComputationPart, positions: np.ndarray, spec: WindowEligibilitySpecification, rng: np.random.Generator) -> np.ndarray:
    locators: list[WindowLocator] = []
    output_slots: list[np.ndarray | None] = []
    long_slots: list[int] = []
    for trajectory_locator, trajectory_length, start, length in _segment_rows(computation, positions):
        if trajectory_length < spec.window_length:
            # FEV's benchmark adapter deliberately keeps short task histories.
            # WindowLocator cannot represent them, so preserve the complete
            # history and repeat its final value to the fixed Chronos geometry.
            readers = getattr(accessor, "_readers", {})
            reader = readers.get(trajectory_locator.bundle_directory)
            if reader is None:
                raise ValueError("no shard reader for short trajectory: %s" % trajectory_locator.catalog_trajectory_id)
            raw = np.asarray(reader.read(trajectory_locator.catalog_trajectory_id), dtype=np.float32)
            if len(raw) != trajectory_length or not len(raw) or not np.isfinite(raw).all():
                raise ValueError("short reference trajectory is invalid: %s" % trajectory_locator.catalog_trajectory_id)
            output_slots.append(np.pad(raw, (0, spec.window_length - len(raw)), mode="edge"))
            continue
        representation = PseudoWindowsStartsRepresentation.build(trajectory_length, start, length, spec.window_length, spec.window_stride)
        if representation._ranges.start_count == 0:
            raise ValueError("selected segment has no legal window: %s %s" % (trajectory_locator.catalog_trajectory_id, start))
        locators.append(WindowLocator(trajectory_locator, representation.sample_start(rng), spec.context_length, spec.prediction_length, spec.window_stride))
        output_slots.append(None)
        long_slots.append(len(output_slots) - 1)
    for start in range(0, len(locators), 512):
        chunk = np.asarray(accessor.read_batch(tuple(locators[start:start + 512])), dtype=np.float32)
        for offset, value in enumerate(chunk):
            output_slots[long_slots[start + offset]] = value
    values = np.stack([value for value in output_slots if value is not None], axis=0) if output_slots else np.empty((0, spec.window_length), dtype=np.float32)
    if values.shape != (len(locators), spec.window_length) or not np.isfinite(values).all():
        raise ValueError("read segment windows are invalid")
    return values


def _eligible_indices(tasks: list[dict[str, Any]], catalogs: dict[int, ReferenceCatalog], spec: WindowEligibilitySpecification, requested: list[int] | None) -> tuple[list[int], list[dict[str, Any]]]:
    indices = list(range(len(tasks))) if requested is None else list(requested)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(tasks) for index in indices):
        raise SystemExit("task indices are invalid")
    # The main FEV experiment's generated eval config is authoritative for
    # task legality. Its 18-task selection intentionally includes short FEV
    # histories accepted by FEVReferenceBenchAdapter.
    if requested is not None:
        return indices, []
    usable: list[int] = []
    skipped: list[dict[str, Any]] = []
    for index in indices:
        catalog = catalogs[index]
        legal_records = sum(not record.dropped and legal_start_count(record.length, spec) > 0 for record in catalog.records_data)
        if legal_records:
            usable.append(index)
        else:
            skipped.append({"task_index": index, "dataset_config": tasks[index].get("dataset_config"), "reason": "reference has no legal window for the configured ChronosTiny window"})
    return usable, skipped


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left)); right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _build_task_windows(task_index: int, task: dict[str, Any], catalog: ReferenceCatalog, source_computation: RuntimeDatasetComputationPart | CompactRuntimeDatasetComputationPart, source_runtime: RuntimeSourceDataset, source_space: SourceAnalysisSegmentFeatureSpace, spec: WindowEligibilitySpecification, ot_config: OTConfig, support_sample_seed: int, sample_size: int, sample_seed: int, mmap_cache_capacity: int) -> TaskWindows:
    dataset_config = str(task["dataset_config"])
    filter_computation = ReferenceBundleReader(catalog, spec, reference_bench_adapter("fev")).build_computation()
    full_computation = ReferenceBundleReader(catalog, spec, reference_bench_adapter("fev")).build_computation()
    filter_space = ReferenceAnalysisSegmentFeatureSpace(filter_computation)
    full_space = ReferenceAnalysisSegmentFeatureSpace(full_computation)
    try:
        target_mass = RatioCandidateUniformSegmentPlanner(support_ratio=1.0, support_sample_seed=support_sample_seed).plan(source=source_space, references=(filter_space,), solver_config=ot_config, lambda_value=1.0)
        if target_mass.segment_masses is None:
            raise ValueError("cu-r1 planner did not return positional segment masses")
        candidate_support = np.flatnonzero(target_mass.segment_masses > 0.0).astype(np.int64)
        if not len(candidate_support):
            raise ValueError("cu-r1 returned an empty candidate support for task %d" % task_index)
        source_count = len(source_space.segment_ids)
        rng = np.random.default_rng(sample_seed)
        source_positions = np.sort(rng.choice(source_count, size=min(sample_size, source_count), replace=False)).astype(np.int64)
        candidate_positions = np.sort(rng.choice(candidate_support, size=min(sample_size, len(candidate_support)), replace=False)).astype(np.int64)
        reference_positions = np.arange(len(full_space.segment_ids), dtype=np.int64)
        source_accessor = WindowAccessor(source_runtime, mmap_cache_capacity=mmap_cache_capacity)
        try:
            source_windows = _windows_for_segments(source_accessor, source_computation, source_positions, spec, rng)
            candidate_windows = _windows_for_segments(source_accessor, source_computation, candidate_positions, spec, rng)
        finally:
            source_accessor.close()
        full_runtime = _runtime_source_from_computation(full_computation)
        reference_accessor = WindowAccessor(full_runtime, mmap_cache_capacity=mmap_cache_capacity)
        try:
            reference_windows = _windows_for_segments(reference_accessor, full_computation, reference_positions, spec, rng)
        finally:
            reference_accessor.close()
        return TaskWindows(task_index, dataset_config, source_windows, candidate_windows, reference_windows, int(len(candidate_support)), source_count, len(reference_positions), dict(target_mass.diagnostics))
    finally:
        filter_space.release(); full_space.release()


def _worker(args: argparse.Namespace) -> int:
    required = (args.prepared_file, args.worker_output_dir, args.worker_index, args.worker_count, args.warmup_root, args.warmup_seeds)
    if any(value is None for value in required):
        raise SystemExit("worker requires prepared file, shard range, output directory, and warm-up model arguments")
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise SystemExit("worker index is outside worker count")
    prepared_path = project_path(args.prepared_file)
    metadata = json.loads(prepared_path.with_suffix(".json").read_text(encoding="utf-8"))
    window_matrix = np.load(prepared_path, mmap_mode="r", allow_pickle=False)
    if window_matrix.ndim != 2:
        raise SystemExit("prepared window matrix is invalid: %s" % prepared_path)
    shard_start, shard_stop = metadata["shards"][args.worker_index]["span"]
    shard_start, shard_stop = int(shard_start), int(shard_stop)
    if not 0 <= shard_start <= shard_stop <= len(window_matrix):
        raise SystemExit("prepared shard span is invalid")
    if shard_stop <= shard_start:
        raise SystemExit("prepared shard is empty")
    worker_dir = project_path(args.worker_output_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    model_seeds = tuple(str(value) for value in args.warmup_seeds)
    engine = ChronosTinyGradientEngine(ChronosTinyConfig(
        model_path=str(project_path(metadata["checkpoints"][0])),
        warmup_model_root=str(project_path(args.warmup_root)),
        warmup_model_count=len(model_seeds),
        warmup_seeds=model_seeds,
        require_warmup_models=True,
        context_length=int(metadata["window_spec"]["context_length"]),
        prediction_length=int(metadata["window_spec"]["prediction_length"]),
        layer="lm_head.weight", projection_dimension=args.projection_dimension,
        projection_repeats=args.projection_repeats, projection_seed=args.projection_seed,
        token_chunk_size=args.token_chunk_size, gradient_batch_size=args.gradient_batch_size,
        seed=args.projection_seed,
    ), device=args.device)
    count = shard_stop - shard_start
    chunk_size = max(args.gradient_batch_size, args.gradient_batch_size * 8)
    loss_cache: np.memmap | None = None
    sketch_cache: np.memmap | None = None
    try:
        for local_start in range(0, count, chunk_size):
            local_stop = min(local_start + chunk_size, count)
            losses, sketches = engine.batch_gradient_sketch(
                np.asarray(window_matrix[shard_start + local_start:shard_start + local_stop], dtype=np.float32).copy(),
                args.gradient_batch_size,
            )
            if losses.ndim == 1:
                losses = losses[None, :]
                sketches = sketches[None, :, :]
            if losses.shape[0] != len(model_seeds) or sketches.shape != (len(model_seeds), local_stop - local_start, args.projection_dimension):
                raise SystemExit("multi-seed gradient output shape is invalid: losses=%s sketches=%s" % (losses.shape, sketches.shape))
            if loss_cache is None:
                loss_cache = np.lib.format.open_memmap(worker_dir / "losses.npy", mode="w+", dtype=np.float64, shape=(len(model_seeds), count))
                sketch_cache = np.lib.format.open_memmap(worker_dir / "gradient_sketches.npy", mode="w+", dtype=np.float32, shape=(len(model_seeds), count, args.projection_dimension))
            loss_cache[:, local_start:local_stop] = losses
            sketch_cache[:, local_start:local_stop, :] = sketches
            LOG.info("shard=%d/%d cached windows=%d/%d", args.worker_index + 1, args.worker_count, local_stop, count)
    finally:
        if loss_cache is not None:
            loss_cache.flush()
        if sketch_cache is not None:
            sketch_cache.flush()
        del loss_cache, sketch_cache
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (worker_dir / "worker.done.json").write_text(json.dumps({
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "global_span": [shard_start, shard_stop], "model_seeds": list(model_seeds),
        "losses": "losses.npy", "gradient_sketches": "gradient_sketches.npy",
    }, indent=2) + "\n", encoding="utf-8")
    LOG.info("worker complete shard=%d span=[%d,%d)", args.worker_index, shard_start, shard_stop)
    return 0


def _checkpoint_protocol(checkpoint_paths: list[Path]) -> tuple[Path, list[str]]:
    roots = {path.parent.parent.resolve() for path in checkpoint_paths}
    if len(roots) != 1:
        raise SystemExit("all checkpoints must belong to one warm-up model root")
    seeds: list[str] = []
    for path in checkpoint_paths:
        label = path.parent.name[5:] if path.parent.name.startswith("seed-") else path.parent.name
        if not label or label in seeds:
            raise SystemExit("checkpoint directories must expose unique seed labels")
        seeds.append(label)
    return next(iter(roots)), seeds


def _start_worker(args: argparse.Namespace, prepared_path: Path, output_dir: Path, warmup_root: Path, model_seeds: list[str], worker_index: int, worker_count: int) -> tuple[subprocess.Popen, Any, Path]:
    worker_dir = output_dir / "gradient-shards" / ("shard-%02d" % worker_index)
    worker_log = output_dir / ("gradient-shard-%02d.log" % worker_index)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--prepared-file", str(prepared_path), "--worker-output-dir", str(worker_dir),
        "--worker-index", str(worker_index), "--worker-count", str(worker_count),
        "--warmup-root", str(warmup_root), "--warmup-seeds", *model_seeds,
        "--gradient-batch-size", str(args.gradient_batch_size),
        "--projection-dimension", str(args.projection_dimension),
        "--projection-repeats", str(args.projection_repeats),
        "--projection-seed", str(args.projection_seed), "--token-chunk-size", str(args.token_chunk_size),
        "--device", str(args.device),
    ]
    stream = worker_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=str(ROOT), env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
    )
    return process, stream, worker_log


def _aggregate_cached_gradients(output_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    model_seeds = [str(value) for value in metadata["model_seeds"]]
    checkpoints = [str(value) for value in metadata["checkpoints"]]
    dimension = int(metadata["projection"]["dimension"])
    sums = {(int(task["task_index"]), name): np.zeros((len(model_seeds), dimension), dtype=np.float64)
            for task in metadata["tasks"] for name in ("source", "candidate", "reference")}
    loss_sums = {(int(task["task_index"]), name): np.zeros(len(model_seeds), dtype=np.float64)
                 for task in metadata["tasks"] for name in ("source", "candidate", "reference")}
    counts = {(int(task["task_index"]), name): int(metadata["spans"][str(task["task_index"])][name][1] - metadata["spans"][str(task["task_index"])][name][0])
              for task in metadata["tasks"] for name in ("source", "candidate", "reference")}
    for shard in metadata["shards"]:
        shard_index = int(shard["worker_index"])
        shard_start, shard_stop = (int(value) for value in shard["span"])
        shard_dir = output_dir / "gradient-shards" / ("shard-%02d" % shard_index)
        sketches = np.load(shard_dir / "gradient_sketches.npy", mmap_mode="r", allow_pickle=False)
        losses = np.load(shard_dir / "losses.npy", mmap_mode="r", allow_pickle=False)
        if sketches.shape != (len(model_seeds), shard_stop - shard_start, dimension) or losses.shape != (len(model_seeds), shard_stop - shard_start):
            raise SystemExit("cached gradient shape is invalid in %s" % shard_dir)
        for task in metadata["tasks"]:
            task_index = int(task["task_index"])
            for name in ("source", "candidate", "reference"):
                global_start, global_stop = (int(value) for value in metadata["spans"][str(task_index)][name])
                start, stop = max(global_start, shard_start), min(global_stop, shard_stop)
                if start >= stop:
                    continue
                local_start, local_stop = start - shard_start, stop - shard_start
                sums[(task_index, name)] += np.asarray(sketches[:, local_start:local_stop, :], dtype=np.float64).sum(axis=1)
                loss_sums[(task_index, name)] += np.asarray(losses[:, local_start:local_stop], dtype=np.float64).sum(axis=1)
        del sketches, losses
    rows: list[dict[str, Any]] = []
    mean_arrays: dict[str, np.ndarray] = {}
    for task in metadata["tasks"]:
        task_index = int(task["task_index"])
        means = {name: sums[(task_index, name)] / counts[(task_index, name)] for name in ("source", "candidate", "reference")}
        for model_index, seed in enumerate(model_seeds):
            row: dict[str, Any] = {
                "task_index": task_index, "dataset_config": task["dataset_config"],
                "model_seed": seed, "checkpoint": checkpoints[model_index],
                "candidate_support_count": int(task["candidate_support_count"]),
                "source_segment_count": int(task["source_segment_count"]),
                "reference_segment_count": int(task["reference_segment_count"]),
            }
            for name in ("source", "candidate", "reference"):
                mean = np.asarray(means[name][model_index], dtype=np.float32)
                mean_arrays["seed-%s_task-%d_%s" % (seed, task_index, name)] = mean
                row[name + "_count"] = counts[(task_index, name)]
                row[name + "_loss_mean"] = float(loss_sums[(task_index, name)][model_index] / counts[(task_index, name)])
                row[name + "_gradient_norm"] = float(np.linalg.norm(mean))
            row["random_cosine_to_reference"] = _cosine(means["source"][model_index], means["reference"][model_index])
            row["candidate_cosine_to_reference"] = _cosine(means["candidate"][model_index], means["reference"][model_index])
            row["alignment_delta_candidate_minus_random"] = row["candidate_cosine_to_reference"] - row["random_cosine_to_reference"]
            rows.append(row)
    np.savez_compressed(output_dir / "mean_gradient_sketches.npz", **mean_arrays)
    return rows


def main() -> int:
    args = _load_experiment_config(_parser().parse_args())
    if args.worker:
        return _worker(args)
    for name, value in (("sample-segments", args.sample_segments), ("gradient-batch-size", args.gradient_batch_size), ("projection-dimension", args.projection_dimension), ("projection-repeats", args.projection_repeats), ("token-chunk-size", args.token_chunk_size), ("mmap-cache-capacity", args.mmap_cache_capacity), ("parallel-workers", args.parallel_workers)):
        _positive_int(name, value)
    if len(args.checkpoints) != 4 and not args.dry_run and not args.allow_checkpoint_subset:
        raise SystemExit("the actual experiment requires exactly four Tiny checkpoints")
    source_path, reference_root, task_path, tasks, cu_config = _load_paths(args)
    ot_config, support_sample_seed, context_length = _cu_settings(cu_config)
    data_config = cu_config.get("data", {})
    prediction_length = int(data_config.get("prediction_length", 64)); window_stride = int(data_config.get("window_stride", 1)); missing_policy = str(data_config.get("missing_value_policy", "reject"))
    spec = WindowEligibilitySpecification(context_length, prediction_length, window_stride, missing_policy)
    checkpoint_paths = [project_path(value) for value in args.checkpoints]
    warmup_root, model_seeds = _checkpoint_protocol(checkpoint_paths)
    missing_checkpoints = [str(path) for path in checkpoint_paths if not path.is_dir()]
    if missing_checkpoints:
        raise SystemExit("missing Tiny checkpoint directories: %s" % ", ".join(missing_checkpoints))
    reader = CatalogReader(); source_catalog = reader.read(source_path, verify=True)
    if not isinstance(source_catalog, SourceCatalog):
        raise SystemExit("source catalog is not a SourceCatalog")
    reference_catalogs: dict[int, ReferenceCatalog] = {}
    for index, task in enumerate(tasks):
        catalog = reader.read(_reference_path(reference_root, str(task.get("dataset_config"))), verify=True)
        if not isinstance(catalog, ReferenceCatalog):
            raise SystemExit("reference catalog is not a ReferenceCatalog")
        reference_catalogs[index] = catalog
    requested_task_indices = _selected_task_indices(args)
    usable, skipped = _eligible_indices(tasks, reference_catalogs, spec, requested_task_indices)
    if args.max_tasks is not None:
        _positive_int("max-tasks", args.max_tasks); usable = usable[:args.max_tasks]
    LOG.info("task plan: requested=%d usable=%d skipped=%d sample_segments=%d checkpoints=%d", len(tasks) if args.task_indices is None else len(args.task_indices), len(usable), len(skipped), args.sample_segments, len(checkpoint_paths))
    LOG.info("skipped tasks: %s", skipped)
    if args.dry_run:
        LOG.info("dry-run validated source, reference catalogs, cu-r1 settings, and checkpoints")
        return 0
    if not usable:
        raise SystemExit("no usable tasks remain")
    output_dir = project_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory is not empty: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "ts3.filter-gradient-experiment.v2", "started_at": datetime.now(timezone.utc).isoformat(), "git_commit_id": _git_commit(), "source_catalog": str(source_path), "reference_catalog_root": str(reference_root), "task_file": str(task_path), "task_selection_config": str(project_path(args.task_selection_config)) if args.task_selection_config else None, "task_indices": usable, "skipped_tasks": skipped, "window_spec": spec.as_mapping(), "cu_r1": {"ot": ot_config.__dict__, "support_sample_seed": support_sample_seed}, "sample_segments": args.sample_segments, "sample_seed": args.sample_seed, "checkpoints": [str(path) for path in checkpoint_paths], "model_seeds": model_seeds, "warmup_model_root": str(warmup_root), "projection": {"dimension": args.projection_dimension, "repeats": args.projection_repeats, "seed": args.projection_seed}, "gradient_cache": "gradient-shards/shard-*/gradient_sketches.npy and losses.npy"}
    (output_dir / "manifest.json").write_text(json.dumps(_safe_json(manifest), indent=2) + "\n", encoding="utf-8")

    LOG.info("building source Feature Bundle computation once: %s", source_path)
    source_computation = SourceBundleReader(source_catalog, spec).build_computation()
    source_runtime = _runtime_source_from_computation(source_computation); source_space = SourceAnalysisSegmentFeatureSpace(source_computation)
    task_windows: list[TaskWindows] = []
    try:
        for position, task_index in enumerate(usable, 1):
            LOG.info("preparing task %d/%d index=%d dataset=%s", position, len(usable), task_index, tasks[task_index]["dataset_config"])
            task_windows.append(_build_task_windows(task_index, tasks[task_index], reference_catalogs[task_index], source_computation, source_runtime, source_space, spec, ot_config, support_sample_seed, args.sample_segments, args.sample_seed + task_index * 1000003, args.mmap_cache_capacity))
    finally:
        source_space.release(); del source_space, source_computation, source_runtime; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    LOG.info("prepared windows: tasks=%d source=%d candidate=%d reference=%d", len(task_windows), sum(len(x.source) for x in task_windows), sum(len(x.candidate) for x in task_windows), sum(len(x.reference) for x in task_windows))
    (output_dir / "task_sampling.json").write_text(json.dumps(_safe_json([
        {"task_index": item.index, "dataset_config": item.dataset_config, "source_sample_count": len(item.source), "candidate_sample_count": len(item.candidate), "reference_segment_count": item.reference_segment_count, "source_segment_count": item.source_segment_count, "candidate_support_count": item.candidate_support_count, "cu_r1_diagnostics": item.filter_diagnostics}
        for item in task_windows
    ]), indent=2) + "\n", encoding="utf-8")

    all_windows: list[np.ndarray] = []; spans: dict[int, dict[str, tuple[int, int]]] = {}; cursor = 0
    for item in task_windows:
        task_spans = {}
        for name, values in (("source", item.source), ("candidate", item.candidate), ("reference", item.reference)):
            all_windows.append(values); task_spans[name] = (cursor, cursor + len(values)); cursor += len(values)
        spans[item.index] = task_spans
    window_matrix = np.concatenate(all_windows, axis=0)
    prepared_path = output_dir / "prepared_windows.npy"
    np.save(prepared_path, window_matrix)
    prepared_metadata = {
        "window_spec": spec.as_mapping(),
        "tasks": [{"task_index": item.index, "dataset_config": item.dataset_config, "candidate_support_count": item.candidate_support_count, "source_segment_count": item.source_segment_count, "reference_segment_count": item.reference_segment_count} for item in task_windows],
        "spans": {str(task_index): {name: list(span) for name, span in task_spans.items()} for task_index, task_spans in spans.items()},
        "projection": {"dimension": args.projection_dimension}, "checkpoints": [str(path) for path in checkpoint_paths],
        "model_seeds": model_seeds,
    }
    worker_count = min(args.parallel_workers, cursor)
    prepared_metadata["shards"] = [{"worker_index": index, "span": [cursor * index // worker_count, cursor * (index + 1) // worker_count]} for index in range(worker_count)]
    prepared_path.with_suffix(".json").write_text(json.dumps(_safe_json(prepared_metadata), indent=2) + "\n", encoding="utf-8")
    if args.parallel_workers <= 0:
        raise SystemExit("parallel-workers must be positive")
    worker_processes: dict[subprocess.Popen, tuple[int, Any, Path]] = {}
    try:
        next_worker = 0
        while next_worker < worker_count or worker_processes:
            while next_worker < worker_count and len(worker_processes) < args.parallel_workers:
                process, log_stream, worker_log = _start_worker(args, prepared_path, output_dir, warmup_root, model_seeds, next_worker, worker_count)
                worker_processes[process] = (next_worker, log_stream, worker_log)
                LOG.info("started gradient shard worker %d/%d pid=%d log=%s", next_worker + 1, worker_count, process.pid, worker_log)
                next_worker += 1
            for process, (worker_index, worker_log_stream, worker_log) in list(worker_processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                worker_log_stream.close(); worker_processes.pop(process)
                if return_code != 0:
                    raise SystemExit("gradient shard worker failed rc=%d index=%d; inspect %s" % (return_code, worker_index, worker_log))
                LOG.info("gradient shard worker complete index=%d log=%s", worker_index, worker_log)
            if worker_processes:
                time.sleep(1)
    except BaseException:
        for process, (_, worker_log_stream, _worker_log) in worker_processes.items():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            worker_log_stream.close()
        raise

    metrics = _aggregate_cached_gradients(output_dir, prepared_metadata)
    import csv

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    deltas = np.asarray([row["alignment_delta_candidate_minus_random"] for row in metrics], dtype=np.float64); random_cosines = np.asarray([row["random_cosine_to_reference"] for row in metrics], dtype=np.float64); candidate_cosines = np.asarray([row["candidate_cosine_to_reference"] for row in metrics], dtype=np.float64)
    summary = {"schema_version": "ts3.filter-gradient-summary.v2", "task_count": len(task_windows), "checkpoint_count": len(checkpoint_paths), "model_seeds": model_seeds, "metric_row_count": len(metrics), "random_cosine_to_reference_mean": float(np.nanmean(random_cosines)), "candidate_cosine_to_reference_mean": float(np.nanmean(candidate_cosines)), "alignment_delta_candidate_minus_random_mean": float(np.nanmean(deltas)), "alignment_delta_candidate_minus_random_median": float(np.nanmedian(deltas)), "positive_delta_fraction": float(np.mean(deltas > 0.0)), "interpretation": "cu-r1 is more reference-gradient aligned when alignment_delta_candidate_minus_random_mean is positive", "skipped_tasks": skipped, "gradient_cache": "gradient-shards/shard-*/gradient_sketches.npy and losses.npy"}
    (output_dir / "summary.json").write_text(json.dumps(_safe_json(summary), indent=2) + "\n", encoding="utf-8")
    LOG.info("RESULT random=%.6f candidate=%.6f delta=%.6f positive_fraction=%.3f", summary["random_cosine_to_reference_mean"], summary["candidate_cosine_to_reference_mean"], summary["alignment_delta_candidate_minus_random_mean"], summary["positive_delta_fraction"])
    LOG.info("experiment complete: %s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
