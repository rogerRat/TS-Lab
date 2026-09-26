"""TIME backend for the unified project evaluation command."""

from __future__ import annotations

import copy
import logging
import importlib.metadata
import inspect
import json
import platform
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from evaluate.schemas import EvaluationRunConfig, resolve_public_class
from configs.logger import format_duration


from configs.paths import PROJECT_ROOT
UTC8 = timezone(timedelta(hours=8))
LOGGER = logging.getLogger("evaluate")
TIME_LICENSE = "CC BY-NC 4.0"


def evaluation_test_length(declared_test_length: int, prediction_length: int,
                           evaluation_mode: str) -> int:
    """Return one true-history horizon by default or the declared rolling tail."""
    if evaluation_mode == "single_window":
        return int(prediction_length)
    if evaluation_mode == "rolling":
        return int(declared_test_length)
    raise ValueError("evaluation_mode must be single_window or rolling")


def _project_relative(value: Any, name: str) -> Path:
    """Resolve an absolute or project-relative source/data path.

    @input: value is a non-empty ASCII path; name labels validation errors.
    @output: Resolved path, including external data disks and symlinks.
    @scenario: Keep large benchmark assets on the configured data disk.
    @logic: Relative paths resolve against PROJECT_ROOT.
    @param value: Declared relative path.
    @param name: Configuration field name for diagnostics.
    """
    if not isinstance(value, str) or not value.isascii() or not value:
        raise ValueError("%s must be non-empty ASCII text" % name)
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    """Read a JSON mapping.

    @input: path is an existing JSON file; label identifies its role.
    @output: Parsed mapping.
    @scenario: Validate immutable TIME data and resume provenance.
    @logic: Converts decode failures to a clear validation error.
    @param path: JSON file path.
    @param label: Human-readable file role.
    """
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise ValueError("cannot read %s: %s" % (label, path)) from error
    if not isinstance(value, Mapping):
        raise ValueError("%s must contain a JSON mapping: %s" % (label, path))
    return value


def _git_state(checkout: Path, expected_commit: str) -> None:
    """Require a clean TIME source checkout at its declared commit.

    @input: checkout is a Git repository; expected_commit is a full SHA-1 string.
    @output: None; raises on mismatch.
    @scenario: Bind benchmark behavior to the locked upstream implementation.
    @logic: Checks repository metadata, HEAD, and all tracked/untracked changes.
    @param checkout: TIME source checkout.
    @param expected_commit: Locked TIME commit.
    """
    if not (checkout / ".git").exists():
        raise FileNotFoundError("TIME checkout is not a git repository: %s" % checkout)
    if not isinstance(expected_commit, str) or len(expected_commit) != 40 or any(char not in "0123456789abcdef" for char in expected_commit):
        raise ValueError("TIME source.commit must be a lowercase 40-character SHA-1")
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"], text=True)
    if head != expected_commit:
        raise ValueError("TIME checkout HEAD does not match source.commit")
    if dirty:
        raise ValueError("TIME checkout must be clean")


def _load_source(value: Mapping[str, Any]) -> tuple[Path, Path, Mapping[str, Any]]:
    """Validate locked TIME source and load its task declaration.

    @input: value is ``bench_config.source``.
    @output: Checkout path, datasets YAML path, and decoded YAML mapping.
    @scenario: Establish the exact upstream task vocabulary before task selection.
    @logic: Strictly validates keys, Git state, and YAML structure.
    @param value: TIME source configuration.
    """
    required = {"checkout", "commit", "datasets_yaml"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("TIME source requires checkout, commit, datasets_yaml")
    checkout = _project_relative(value["checkout"], "TIME source.checkout")
    datasets_yaml = _project_relative(value["datasets_yaml"], "TIME source.datasets_yaml")
    if checkout not in datasets_yaml.parents:
        raise ValueError("TIME datasets_yaml must be inside TIME source.checkout")
    _git_state(checkout, value["commit"])
    if not datasets_yaml.is_file():
        raise FileNotFoundError("TIME datasets YAML is missing: %s" % datasets_yaml)
    loaded = yaml.safe_load(datasets_yaml.read_text(encoding="ascii"))
    if not isinstance(loaded, Mapping) or set(loaded) != {"datasets"} or not isinstance(loaded["datasets"], Mapping):
        raise ValueError("TIME datasets YAML must contain only a datasets mapping")
    return checkout, datasets_yaml, loaded


def _parse_tasks(value: Mapping[str, Any], datasets: Mapping[str, Any]) -> list[dict[str, str]]:
    """Expand the strict TIME task selection.

    @input: value is ``bench_config.tasks``; datasets is locked TIME YAML content.
    @output: Ordered unique task dictionaries with dataset and term fields.
    @scenario: Support explicitly reviewed smoke tasks and source-defined official all.
    @logic: Validates every declared term against the pinned YAML; official_all
            iterates YAML insertion order and never guesses directories.
    @param value: Task selection mapping.
    @param datasets: Pinned TIME dataset configuration mapping.
    """
    if not isinstance(value, Mapping) or "mode" not in value:
        raise ValueError("TIME tasks requires a mode")
    mode = value["mode"]
    if mode == "explicit":
        if set(value) != {"mode", "items"} or not isinstance(value["items"], list) or not value["items"]:
            raise ValueError("TIME explicit tasks requires a non-empty items list")
        expanded: list[dict[str, str]] = []
        for item in value["items"]:
            if not isinstance(item, Mapping) or set(item) != {"dataset", "terms"}:
                raise ValueError("TIME task item requires only dataset and terms")
            dataset = item["dataset"]
            terms = item["terms"]
            if not isinstance(dataset, str) or not dataset.isascii() or dataset not in datasets:
                raise ValueError("TIME task dataset is absent from datasets.yaml: %r" % dataset)
            if not isinstance(terms, list) or not terms:
                raise ValueError("TIME task terms must be a non-empty list")
            for term in terms:
                if not isinstance(term, str) or term not in datasets[dataset] or not isinstance(datasets[dataset][term], Mapping) or "prediction_length" not in datasets[dataset][term]:
                    raise ValueError("TIME task term is absent from datasets.yaml: %r/%r" % (dataset, term))
                expanded.append({"dataset": dataset, "term": term})
    elif mode == "official_all":
        if set(value) != {"mode"}:
            raise ValueError("TIME official_all forbids items")
        expanded = [
            {"dataset": dataset, "term": term}
            for dataset, settings in datasets.items()
            for term, declaration in settings.items()
            if isinstance(declaration, Mapping) and "prediction_length" in declaration
        ]
    else:
        raise ValueError("TIME tasks.mode must be explicit or official_all")
    identities = [(item["dataset"], item["term"]) for item in expanded]
    if len(identities) != len(set(identities)):
        raise ValueError("TIME tasks must not contain duplicates")
    return expanded


def _validate_bench_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all TIME-private configuration and provenance inputs.

    @input: value is the common run config's bench_config mapping.
    @output: Normalized source, data, YAML, and ordered task values.
    @scenario: Keep all TIME-specific validation in the TIME backend.
    @logic: Validates exact nested keys, source state, data manifest, and task list.
    @param value: Raw TIME bench_config mapping.
    """
    if set(value) != {"source", "dataset", "tasks"}:
        raise ValueError("TIME bench_config requires only source, dataset, tasks")
    checkout, datasets_yaml, loaded = _load_source(value["source"])
    dataset = value["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {"repo_id", "revision", "root"}:
        raise ValueError("TIME dataset requires repo_id, revision, root")
    if dataset["repo_id"] != "Real-TSF/TIME" or dataset["revision"] != "83e3d0b3be28d11c7182bffcc1892d19b36c4da1":
        raise ValueError("TIME dataset repo_id and revision must match the pinned protocol")
    root = _project_relative(dataset["root"], "TIME dataset.root")
    manifest_path = root.parent / "data_manifest.json"
    manifest = _read_json(manifest_path, "TIME data manifest")
    if manifest.get("repo_id") != dataset["repo_id"] or manifest.get("revision") != dataset["revision"] or manifest.get("license") != TIME_LICENSE:
        raise ValueError("TIME data manifest does not match repo, revision, and license")
    tasks = _parse_tasks(value["tasks"], loaded["datasets"])
    if manifest.get("verified_tasks") != tasks:
        raise ValueError("TIME data manifest was not validated for this exact ordered task protocol")
    expected_paths = [root / item["dataset"] for item in tasks]
    missing = [path for path in expected_paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError("TIME dataset directories are missing, first: %s" % missing[0])
    return {
        "checkout": checkout,
        "datasets_yaml": datasets_yaml,
        "datasets": loaded["datasets"],
        "dataset_root": root,
        "dataset_manifest": manifest_path,
        "dataset_values": copy.deepcopy(dict(dataset)),
        "source_values": copy.deepcopy(dict(value["source"])),
        "tasks": tasks,
    }


def _adapter_source(spec: str) -> str:
    """Return source file and digest for an adapter class.

    @input: spec is a public adapter class path.
    @output: Absolute source path and SHA-256.
    @scenario: Preserve adapter implementation provenance in TIME output.
    @logic: Resolves the class through the common public class resolver.
    """
    source = Path(inspect.getsourcefile(resolve_public_class(spec)) or "").resolve()
    return str(source)


def _package_versions() -> dict[str, str]:
    """Collect installed runtime package versions.

    @input: No arguments.
    @output: Mapping of required package names to versions or ``unavailable``.
    @scenario: Record the isolated TIME environment used for a result.
    @logic: Reads distribution metadata without importing heavyweight packages.
    """
    result = {"python": platform.python_version()}
    for package in ("timebench", "chronos-forecasting", "numpy", "datasets", "gluonts", "torch", "transformers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _metadata(config: EvaluationRunConfig, checkpoint: Path, checked: Mapping[str, Any], adapter_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build TIME provenance before task execution.

    @input: Common config, checkpoint, validated TIME settings, adapter config.
    @output: Complete start metadata mapping.
    @scenario: Make raw TIME results independently reproducible.
    @logic: Hashes every local source/data/checkpoint artifact and ordered task list.
    """
    manifest = checkpoint / "ts3_checkpoint_manifest.json"
    adapter_path = _adapter_source(config.adapter_class)
    return {
        "benchmark": "time",
        "model_id": config.model_id,
        "checkpoint": str(checkpoint),
        "adapter_class": config.adapter_class,
        "adapter_source": adapter_path,
        "adapter_config": copy.deepcopy(dict(adapter_config)),
        "time_source": checked["source_values"],
        "time_datasets_yaml": str(checked["datasets_yaml"]),
        "data": checked["dataset_values"],
        "data_manifest": str(checked["dataset_manifest"]),
        "data_manifest": str(checked["dataset_manifest"]),
        "data_license": TIME_LICENSE,
        "tasks": copy.deepcopy(checked["tasks"]),
        "package_versions": _package_versions(),
        "started_at": datetime.now(UTC8).isoformat(),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish metadata JSON.

    @input: path is a destination; value is JSON-compatible.
    @output: Complete ASCII JSON at path.
    @scenario: Preserve metadata integrity across an interrupted TIME run.
    @logic: Uses a sibling partial file and atomic replace.
    """
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    partial.replace(path)


def _task_directory(output_dir: Path, task: Mapping[str, str]) -> Path:
    """Return the official TIME output directory for one task.

    @input: output_dir is run root; task has dataset and term.
    @output: ``<root>/<dataset>/<frequency>/<term>`` path.
    @scenario: Keep output tree exactly compatible with TIME saver.
    @logic: Dataset names already include the frequency component.
    """
    return output_dir / task["dataset"] / task["term"]


def _complete_task(path: Path) -> bool:
    """Check that one official TIME task output is readable and complete.

    @input: path is a task directory.
    @output: True only when predictions, metrics, and config parse successfully.
    @scenario: Permit explicit resume only for intact immutable task artifacts.
    @logic: Opens both NPZ files without pickle and parses config JSON.
    """
    required = (path / "predictions.npz", path / "metrics.npz", path / "config.json")
    if any(not item.is_file() for item in required):
        return False
    try:
        with np.load(required[0], allow_pickle=False) as predictions:
            if "predictions_quantiles" not in predictions or "quantile_levels" not in predictions:
                return False
        with np.load(required[1], allow_pickle=False) as metrics:
            if not metrics.files:
                return False
        value = _read_json(required[2], "TIME task config")
        return bool(value)
    except (OSError, ValueError):
        return False


def _resume_metadata_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Compare stable TIME metadata fields for resume.

    @input: actual is existing metadata; expected is current metadata.
    @output: True when all immutable run identities match.
    @scenario: Prevent mixed-source or mixed-model TIME result trees.
    @logic: Ignores only timestamps and completion state.
    """
    keys = (
        "benchmark", "model_id", "checkpoint", "adapter_class", "adapter_source", "adapter_config", "time_source",
        "time_datasets_yaml", "data", "data_manifest", "data_license", "tasks",
    )
    return all(actual.get(key) == expected.get(key) for key in keys)


def run_time(config: EvaluationRunConfig, checkpoint: Path, output_dir: Path, resume: bool) -> Path:
    """Execute the TIME backend and publish official raw task artifacts.

    @input: Common TIME config, complete checkpoint, prepared output root, resume flag.
    @output: Path to time_metadata.json after all tasks succeed.
    @scenario: Run the pinned TIME protocol serially with one Chronos-T5 model.
    @logic: Validates source/data/task provenance, creates a single adapter, loads
            each TIME Dataset with explicit storage_path, predicts, and delegates
            raw artifact writing and metrics to TIME's official saver.
    @param config: Common config with benchmark ``time``.
    @param checkpoint: Complete TS3 Chronos checkpoint directory.
    @param output_dir: Existing run root prepared by evaluate.run.
    @param resume: Explicit permission to reuse only complete task outputs.
    """
    if not isinstance(config, EvaluationRunConfig) or config.benchmark != "time":
        raise TypeError("config must be an EvaluationRunConfig for time")
    checked = _validate_bench_config(config.bench_config)
    adapter_type = resolve_public_class(config.adapter_class)
    if not callable(getattr(adapter_type, "parse_config", None)) or not callable(getattr(adapter_type, "from_config", None)):
        raise TypeError("TIME adapter must provide parse_config and from_config")
    adapter_config = adapter_type.parse_config(dict(config.adapter_config), config.source_path.parent)
    if not isinstance(adapter_config, Mapping):
        raise TypeError("TIME adapter parse_config must return a mapping")
    expected_metadata = _metadata(config, checkpoint, checked, adapter_config)
    metadata_path = output_dir / "time_metadata.json"
    if metadata_path.exists():
        if not resume:
            raise FileExistsError("TIME metadata already exists: %s" % metadata_path)
        if not _resume_metadata_matches(_read_json(metadata_path, "TIME metadata"), expected_metadata):
            raise ValueError("TIME resume metadata does not match this run")
    else:
        _write_json(metadata_path, expected_metadata)

    from gluonts.time_feature import get_seasonality
    from timebench.evaluation.data import Dataset
    from timebench.evaluation.saver import save_window_predictions

    adapter = adapter_type.from_config(adapter_config, checkpoint=checkpoint)
    tasks = checked["tasks"]
    LOGGER.info("TIME evaluation started: tasks=%d dataset_root=%s", len(tasks), checked["dataset_root"])
    progress_started_at = time.monotonic()
    for index, task in enumerate(tasks):
        destination = _task_directory(output_dir, task)
        if destination.exists():
            if not resume:
                raise FileExistsError("TIME task output already exists: %s" % destination)
            if not _complete_task(destination):
                raise ValueError("TIME resume refuses missing or damaged task output: %s" % destination)
            LOGGER.info("TIME task %d/%d already complete; reusing %s", index + 1, len(tasks), destination)
            continue
        LOGGER.info("TIME task %d/%d started: dataset=%s term=%s", index + 1, len(tasks), task["dataset"], task["term"])
        settings = checked["datasets"][task["dataset"]]
        term_settings = settings[task["term"]]
        prediction_length = int(term_settings["prediction_length"])
        evaluation_mode = adapter_config["inference"].get("evaluation_mode", "single_window")
        test_length = evaluation_test_length(settings["test_length"], prediction_length, evaluation_mode)
        native = Dataset(
            name=task["dataset"], term=task["term"], to_univariate=False,
            prediction_length=prediction_length, test_length=test_length,
            val_length=settings.get("val_length", 0), storage_path=checked["dataset_root"],
        )
        dataset = native if native.target_dim == 1 else Dataset(
            name=task["dataset"], term=task["term"], to_univariate=True,
            prediction_length=prediction_length, test_length=test_length,
            val_length=settings.get("val_length", 0), storage_path=checked["dataset_root"],
        )
        forecasts = adapter.predict_quantiles(dataset.test_data.input, dataset.prediction_length)
        save_window_predictions(
            dataset=dataset,
            fc_quantiles=forecasts,
            ds_config="%s/%s" % (task["dataset"], task["term"]),
            output_base_dir=str(output_dir),
            seasonality=get_seasonality(dataset.freq),
            model_hyperparams={"model_id": config.model_id, "adapter_config": copy.deepcopy(dict(adapter_config))},
            quantile_levels=list(adapter_config["inference"]["quantile_levels"]),
        )
        percent = 100.0 * (index + 1) / len(tasks)
        filled = min(20, int(percent // 5))
        elapsed = time.monotonic() - progress_started_at
        completed = index + 1
        rate = completed / elapsed if elapsed > 0 else 0.0
        eta = (len(tasks) - completed) / rate if rate else None
        LOGGER.info("TIME evaluation is %.1f%% complete: tasks %d/%d, eta %s, progress [%s]", percent, completed, len(tasks), format_duration(eta), "=" * filled + "." * (20 - filled))
    completed_metadata = dict(expected_metadata)
    completed_metadata["completed_at"] = datetime.now(UTC8).isoformat()
    _write_json(metadata_path, completed_metadata)
    LOGGER.info("TIME evaluation completed: tasks=%d result=%s", len(tasks), metadata_path)
    # TODO: When external TIME leaderboard comparison is required, fetch the
    # pinned Seasonal Naive results and invoke TIME's official aggregation script.
    return metadata_path


__all__ = ["run_time"]
