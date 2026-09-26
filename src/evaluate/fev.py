"""Evaluate local Chronos checkpoints with the pinned installed FEV package."""

from __future__ import annotations
import multiprocessing as mp
import os
import random

import csv
import inspect
import importlib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml
import numpy as np
import torch
from tqdm import tqdm

from runtime.memory import MemoryBudget
from evaluate.schemas import EvaluationRunConfig, resolve_public_class


LOGGER = logging.getLogger("evaluate")
UTC8 = timezone(timedelta(hours=8))
from configs.paths import PROJECT_ROOT
DEFAULT_TASK_FILE = PROJECT_ROOT / "configs" / "eval" / "fev" / "default.yaml"
DEFAULT_LOCK_FILE = PROJECT_ROOT / "requ" / "third_parties.lock"
RQ3_METRICS = ("SQL", "MASE", "WAPE", "WQL", "MAE", "MSE")
RQ3_TASK_COUNT = 20


def _adapter_source(spec: str) -> str:
    module, attribute = spec.split(":", 1)
    value = importlib.import_module(module)
    for part in attribute.split("."):
        value = getattr(value, part)
    source = Path(inspect.getsourcefile(value) or "").resolve()
    return str(source)


def _metric_name(value: Any) -> str:
    """Normalize one FEV metric declaration.

    @input: value is a metric string or mapping with a name field.
    @output: Non-empty metric name.
    @scenario: Compare explicit task protocols without mutating them.
    @param value: Raw primary or extra metric declaration.
    """
    name = value.get("name") if isinstance(value, Mapping) else value
    if not isinstance(name, str) or not name:
        raise ValueError("FEV metric declaration needs a non-empty name")
    return name


def load_task_declarations(path: Path) -> list[dict[str, Any]]:
    """Load and validate the project-owned FEV task protocol.

    @input: path points directly to a YAML mapping with a tasks list.
    @output: Twenty task mappings, each declaring the six RQ3 metrics.
    @scenario: Make the complete evaluation protocol visible in one project file.
    @param path: Project benchmark task YAML.
    """
    if not path.is_file():
        raise FileNotFoundError("FEV task file not found: %s" % path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"tasks"}:
        raise ValueError("FEV task declaration must contain only a direct tasks list")
    tasks = value["tasks"]
    if not isinstance(tasks, list) or not all(isinstance(item, dict) for item in tasks):
        raise ValueError("FEV tasks must be a list of mappings")
    if len(tasks) != RQ3_TASK_COUNT:
        raise ValueError("FEV RQ3 declaration must contain exactly %d tasks" % RQ3_TASK_COUNT)
    expected = set(RQ3_METRICS)
    for index, task in enumerate(tasks):
        dataset_path = task.get("dataset_path")
        if isinstance(dataset_path, str) and not Path(dataset_path).is_file():
            dataset_config = task.get("dataset_config")
            if dataset_path == "autogluon/fev_datasets" and isinstance(dataset_config, str):
                matches = sorted((PROJECT_ROOT / "data/bench/fev/datasets/autogluon___fev_datasets" /
                                  dataset_config / "0.0.0").glob("*/fev_datasets-train.arrow"))
                if len(matches) == 1:
                    dataset_path = task["dataset_path"] = str(matches[0])
        if (not isinstance(dataset_path, str) or dataset_path.startswith(("hf://", "http://", "https://"))
                or Path(dataset_path).suffix not in {".parquet", ".arrow"}
                or not Path(dataset_path).is_file()):
            raise ValueError("FEV task %d must use an existing local parquet/arrow dataset_path" % index)
        declared = {_metric_name(task.get("eval_metric"))}
        extra = task.get("extra_metrics", [])
        if not isinstance(extra, list):
            raise ValueError("FEV task %d extra_metrics must be a list" % index)
        declared.update(_metric_name(metric) for metric in extra)
        if declared != expected:
            raise ValueError(
                "FEV task %d metrics must equal %s, got %s"
                % (index, sorted(expected), sorted(declared))
            )
    return tasks


def _locked_commit(lock_file: Path, repository_name: str) -> str:
    """Read one benchmark dependency commit from the project lock file.

    @input: lock_file uses name URL commit rows; repository_name is exact.
    @output: Locked commit string.
    @scenario: Record FEV source provenance without reading its checkout at runtime.
    @param lock_file: Project third-party lock path.
    @param repository_name: Dependency row name.
    """
    for line in lock_file.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 3:
            raise ValueError("Malformed third-party lock entry: %r" % line)
        if fields[0] == repository_name:
            return fields[2]
    raise ValueError("Repository is missing from lock file: %s" % repository_name)


def _write_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Write task, mean, and provenance results without overwrite.

    @input: output_dir is a new result directory; rows and metadata are complete.
    @output: Path to fev_results.csv.
    @scenario: Preserve one immutable benchmark result set per run.
    @param output_dir: Evaluation artifact directory.
    @param rows: Task-level FEV summaries.
    @param metadata: JSON-compatible evaluation provenance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [
        output_dir / "fev_results.csv",
        output_dir / "summary.csv",
        output_dir / "fev_metadata.json",
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite evaluation output: %s" % existing[0])
    fieldnames = sorted({key for row in rows for key in row})
    with destinations[0].open("w", newline="", encoding="ascii", errors="backslashreplace") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, Any] = {"task": "mean"}
    for field in fieldnames:
        if field in {"task", "task_id"}:
            continue
        values = [
            float(row[field]) for row in rows
            if isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
        ]
        if values:
            summary[field] = sum(values) / len(values)
    with destinations[1].open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(summary))
        writer.writeheader()
        writer.writerow(summary)
    destinations[2].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return destinations[0]


def _run_fev_task_worker(
    task_index: int,
    checkpoint: Path,
    task_file: Path,
    result_dir: Path,
    adapter_class: str,
    adapter_config: Mapping[str, Any],
    base_seed: int,
    model_id: str,
) -> None:
    """Evaluate one FEV task in a fresh child process.

    @input: Task index, checkpoint, task declaration, output directory, and
            explicit Chronos inference settings.
    @output: One immutable task JSON result in result_dir.
    @scenario: Reclaim task datasets and model-side temporary objects at exit.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / ("task_%03d.json" % int(task_index))
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="ascii"))
            if isinstance(payload, Mapping) and isinstance(payload.get("row"), Mapping):
                return
        except (OSError, ValueError):
            pass
    import importlib
    import fev
    module_name, attribute = adapter_class.split(":", 1)
    adapter_type = importlib.import_module(module_name)
    for part in attribute.split("."):
        adapter_type = getattr(adapter_type, part)
    derived_seed = int(base_seed) + int(task_index)
    random.seed(derived_seed)
    np.random.seed(derived_seed)
    torch.manual_seed(derived_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived_seed)
    benchmark = fev.Benchmark.from_yaml(task_file)
    task = benchmark.tasks[int(task_index)]
    # The project declaration is portable and names the upstream dataset;
    # resolve it to the immutable local materialization validated above.
    task.dataset_path = load_task_declarations(task_file)[int(task_index)]["dataset_path"]
    model = adapter_type.from_config(adapter_config, checkpoint=checkpoint, seed=derived_seed)
    model.model_name = model_id
    predictions = model.fit_predict(task)
    summary = task.evaluation_summary(
        predictions,
        model_name=getattr(model, "model_name", type(model).__name__),
        training_time_s=model.training_time,
        inference_time_s=model.inference_time,
        trained_on_this_dataset=task.dataset_config in model.trained_on_datasets,
        extra_info={
            "adapter_class": adapter_class,
            "task_seed": derived_seed,
        },
    )
    payload = {
        "task_index": int(task_index),
        "pid": os.getpid(),
        "task": str(task.task_name),
        "row": dict(summary),
    }
    partial_path = result_path.with_suffix(".json.partial")
    partial_path.write_text(
        json.dumps(payload, sort_keys=True, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n",
        encoding="ascii",
    )
    partial_path.replace(result_path)

def evaluate_with_fev(
    checkpoint: Path,
    task_file: Path,
    output_dir: Path,
    adapter_class: str,
    adapter_config: Mapping[str, Any],
    base_seed: int,
    num_tasks: Optional[int] = None,
    task_indices: Optional[Sequence[int]] = None,
    lock_file: Path = DEFAULT_LOCK_FILE,
    planning_fraction: float = 0.70,
    stop_fraction: float = 0.80,
    model_id: str = "chronos",
) -> Path:
    """Evaluate tasks in independent child processes with resumable results.

    @input: checkpoint, task file, output path, adapter class/config, seed, task cap.
    @output: Path to the final ordered task-level result CSV.
    @scenario: Reclaim each task dataset at child exit and resume unfinished tasks.
    """
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not checkpoint.is_dir():
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint)
    declarations = load_task_declarations(task_file)
    if task_indices is None:
        task_count = len(declarations) if num_tasks is None else int(num_tasks)
        if not 1 <= task_count <= len(declarations):
            raise ValueError("num_tasks must be between 1 and %d" % len(declarations))
        selected_task_indices = tuple(range(task_count))
    else:
        selected_task_indices = tuple(task_indices)
        if not selected_task_indices:
            raise ValueError("task_indices must not be empty")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in selected_task_indices):
            raise ValueError("task_indices must contain integers")
        if len(set(selected_task_indices)) != len(selected_task_indices):
            raise ValueError("task_indices must not contain duplicates")
        if any(index < 0 or index >= len(declarations) for index in selected_task_indices):
            raise ValueError("task_indices must be between 0 and %d" % (len(declarations) - 1))
        if num_tasks is not None and int(num_tasks) != len(selected_task_indices):
            raise ValueError("num_tasks must equal the number of task_indices")
        task_count = len(selected_task_indices)
    budget = MemoryBudget.resolve(planning_fraction, stop_fraction)
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    process_ids = []
    context = mp.get_context("spawn")
    LOGGER.info("FEV evaluation started: tasks=%d task_file=%s", task_count, task_file)
    evaluation_progress = tqdm(
        selected_task_indices,
        desc="FEV evaluation",
        unit="task",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    try:
        for completed, task_index in enumerate(evaluation_progress, start=1):
            result_path = task_dir / ("task_%03d.json" % task_index)
            payload = None
            if result_path.is_file():
                try:
                    candidate = json.loads(result_path.read_text(encoding="ascii"))
                    if int(candidate.get("task_index", -1)) == task_index and isinstance(candidate.get("row"), Mapping):
                        payload = candidate
                except (OSError, ValueError, TypeError):
                    payload = None
            if payload is None:
                budget.check(
                    "fev_task",
                    batch_rows=int(adapter_config["inference"]["batch_size"]),
                    worker_count=1,
                    logger=LOGGER,
                )
                LOGGER.info("[ARGS] Evaluating selected task index=%d (%d/%d) in child process", task_index, completed, task_count)
                process = context.Process(
                    target=_run_fev_task_worker,
                    args=(task_index, checkpoint, task_file, task_dir, adapter_class, adapter_config, base_seed, model_id),
                )
                process.start()
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError("FEV task child failed: index=%d exit=%s" % (task_index, process.exitcode))
                if not result_path.is_file():
                    raise RuntimeError("FEV task child did not publish result: %s" % result_path)
                payload = json.loads(result_path.read_text(encoding="ascii"))
            if int(payload.get("task_index", -1)) != task_index or not isinstance(payload.get("row"), Mapping):
                raise ValueError("invalid FEV task result: %s" % result_path)
            rows.append(dict(payload["row"]))
            process_ids.append(int(payload.get("pid", -1)))
            evaluation_progress.set_postfix(task=task_index, refresh=False)
    finally:
        evaluation_progress.close()
    metadata = {
        "adapter_class": adapter_class,
        "model_id": model_id,
        "adapter_source": _adapter_source(adapter_class),
        "adapter_config": dict(adapter_config),
        "checkpoint": str(checkpoint.resolve()),
        "created_at": datetime.now(UTC8).isoformat(),
        "fev_commit": _locked_commit(lock_file, "fev"),
        "fev_package_version": importlib_metadata.version("fev"),
        "metrics": list(RQ3_METRICS),
        "sampling": {key: adapter_config["inference"][key] for key in ("num_samples", "temperature", "top_k", "top_p")},
        "base_seed": base_seed,
        "task_seed_formula": "base_seed + task_index",
        "task_indices": list(selected_task_indices),
        "task_seeds": [base_seed + index for index in selected_task_indices],
        "num_tasks": task_count,
        "task_file": str(task_file.resolve()),
        "task_process_ids": process_ids,
        "memory_contract": {
            "limit_bytes": budget.limit_bytes,
            "planning_budget_bytes": budget.planning_budget_bytes,
            "stop_threshold_bytes": budget.stop_threshold_bytes,
        },
    }
    result = _write_results(output_dir, rows, metadata)
    LOGGER.info("FEV evaluation completed: tasks=%d result=%s", task_count, result)
    return result


def _parse_fev_bench_config(value: Mapping[str, Any], base: Path) -> tuple[Path, int, Optional[tuple[int, ...]]]:
    """Validate the FEV-only benchmark mapping.

    @input: value is bench_config; base is the run YAML parent directory.
    @output: Resolved task YAML path, expected count, and optional selected task indices.
    @scenario: Keep FEV task protocol outside the common run schema.
    @logic: Supports either legacy first-N selection or explicit task_indices.
    @param value: FEV bench_config mapping.
    @param base: Run configuration parent directory.
    """
    allowed = {"task_file", "expected_tasks", "task_indices"}
    if set(value) not in ({"task_file", "expected_tasks"}, allowed):
        raise ValueError("FEV bench_config requires task_file, expected_tasks, and optional task_indices")
    task_file = value["task_file"]
    if not isinstance(task_file, str) or not task_file.isascii() or not task_file:
        raise ValueError("FEV task_file must be non-empty ASCII text")
    path = Path(task_file)
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    expected = value["expected_tasks"]
    if isinstance(expected, bool) or int(expected) != expected or int(expected) <= 0:
        raise ValueError("FEV expected_tasks must be a positive integer")
    task_indices = value.get("task_indices")
    if task_indices is None:
        return path, int(expected), None
    if not isinstance(task_indices, list) or not task_indices:
        raise ValueError("FEV task_indices must be a non-empty list")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in task_indices):
        raise ValueError("FEV task_indices must contain non-negative integers")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("FEV task_indices must not contain duplicates")
    if int(expected) != len(task_indices):
        raise ValueError("FEV expected_tasks must equal the number of task_indices")
    return path, int(expected), tuple(task_indices)


def _warn_reference_task_mismatch(checkpoint: Path, selected_task_indices: Sequence[int]) -> None:
    """Emit an unmistakable non-blocking warning for reference/evaluation mismatches."""
    manifest_path = checkpoint / "ts3_checkpoint_manifest.json"
    try:
        checkpoint_manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        LOGGER.warning("====== REFERENCE TASK CHECK UNAVAILABLE ======\ncheckpoint manifest could not be read: %s\n==============================================", manifest_path)
        return
    run_manifest_path = checkpoint / "run_manifest.json"
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        run_manifest = {}
    if isinstance(run_manifest, Mapping) and run_manifest.get("policy") == "random":
        LOGGER.info("Reference task check skipped: checkpoint policy=random does not use a reference catalog.")
        return
    reference_task_record = checkpoint_manifest.get("reference_task_record")
    if reference_task_record is None:
        LOGGER.warning("====== REFERENCE TASK CHECK UNAVAILABLE ======\ncheckpoint has no reference_task_record declaration; cannot verify FEV task alignment.\ncheckpoint=%s\n==============================================", checkpoint)
        return
    if not isinstance(reference_task_record, Mapping) or reference_task_record.get("benchmark") != "fev" or not isinstance(reference_task_record.get("task_indices"), list):
        LOGGER.warning("====== REFERENCE TASK CHECK UNAVAILABLE ======\ncheckpoint reference_task_record is invalid for FEV: %r\ncheckpoint=%s\n==============================================", reference_task_record, checkpoint)
        return
    reference_indices = tuple(reference_task_record["task_indices"])
    selected_indices = tuple(selected_task_indices)
    if reference_indices == selected_indices:
        LOGGER.info("Reference task check passed: checkpoint FEV tasks=%s; evaluation tasks=%s", list(reference_indices), list(selected_indices))
        return
    LOGGER.warning("====== REFERENCE TASK MISMATCH ======\ncheckpoint reference FEV tasks: %s\nevaluation selected FEV tasks: %s\nThe selection reference task record does not exactly match this evaluation.\n=======================================", list(reference_indices), list(selected_indices))


def run_fev(
    config: EvaluationRunConfig,
    checkpoint: Path,
    output_dir: Path,
    resume: bool,
) -> Path:
    """Run the FEV backend under the common evaluation contract.

    @input: Common FEV config, checkpoint directory, output path, and resume flag.
    @output: Path to the benchmark task-level result file.
    @scenario: Preserve the existing FEV task worker and metric implementation.
    @logic: Validates FEV-only fields and adapter config, then delegates to the
            existing child-process runner. A completed explicit resume returns
            the existing immutable result.
    @param config: Common config with benchmark set to fev.
    @param checkpoint: Completed local model checkpoint.
    @param output_dir: Common evaluation artifact directory.
    @param resume: Whether existing matching artifacts may be reused.
    """
    if not isinstance(config, EvaluationRunConfig) or config.benchmark != "fev":
        raise TypeError("config must be an EvaluationRunConfig for fev")
    adapter_type = resolve_public_class(config.adapter_class)
    if not callable(getattr(adapter_type, "parse_config", None)) or not callable(getattr(adapter_type, "from_config", None)):
        raise TypeError("FEV adapter must provide parse_config and from_config")
    adapter_config = adapter_type.parse_config(dict(config.adapter_config), config.source_path.parent)
    if not isinstance(adapter_config, Mapping):
        raise TypeError("FEV adapter parse_config must return a mapping")
    task_file, expected_tasks, task_indices = _parse_fev_bench_config(config.bench_config, config.source_path.parent)
    declarations = load_task_declarations(task_file)
    if expected_tasks > len(declarations):
        raise ValueError("expected_tasks exceeds declared FEV tasks")
    selected_task_indices = tuple(range(expected_tasks)) if task_indices is None else task_indices
    if any(index >= len(declarations) for index in selected_task_indices):
        raise ValueError("task_indices exceeds declared FEV tasks")
    _warn_reference_task_mismatch(checkpoint, selected_task_indices)
    completed = output_dir / "fev_results.csv"
    if resume and completed.is_file() and (output_dir / "summary.csv").is_file() and (output_dir / "fev_metadata.json").is_file():
        return completed
    return evaluate_with_fev(
        checkpoint,
        task_file,
        output_dir,
        config.adapter_class,
        json.loads(json.dumps(dict(adapter_config))),
        int(adapter_config["inference"]["seed"]),
        num_tasks=expected_tasks,
        task_indices=task_indices,
        model_id=config.model_id,
    )
