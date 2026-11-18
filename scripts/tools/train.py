"""Train one registered forecasting model on prepared windows."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
import copy
import json
from pathlib import Path
import time


from configs.logger import configure_logger, format_duration
from configs.paths import output_path

from scripts.tools._common import apply_overrides, dispatch_model_environment, load_config, required, resolve


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    temporary.replace(path)


def _save_model(engine, destination: Path) -> None:
    import torch
    model = engine.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        engine.model = model
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(destination, safe_serialization=True)
    else:
        destination.mkdir(parents=True, exist_ok=False)
        torch.save(model.state_dict(), destination / "model_state.pt")


def _save_model_snapshot(engine, destination: Path) -> None:
    import torch
    snapshot = copy.deepcopy(engine.model)
    try:
        if hasattr(snapshot, "merge_and_unload"):
            snapshot = snapshot.merge_and_unload()
        snapshot.eval()
        snapshot.save_pretrained(destination, safe_serialization=True)
    finally:
        del snapshot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(config: dict, *, stage: str) -> Path:
    import numpy as np
    import torch
    from rage.experiments.full_pool_models import build_model
    from rage.experiments.lora import apply_student_lora
    from rage.experiments.students import STUDENTS, HISTORICAL_STUDENTS, NON_DEPLOYABLE_STUDENTS
    from rage.experiments.train_eval import TrainingRecipe, fine_tune

    model_key = str(required(config, "model"))
    if model_key not in STUDENTS + HISTORICAL_STUDENTS or model_key in NON_DEPLOYABLE_STUDENTS:
        raise ValueError("unsupported training model: " + model_key)
    data = config.get("data", {})
    training = config.get("training", {})
    context = int(required(data, "context_length"))
    horizon = int(required(data, "prediction_length"))
    fixed_subset_payload = data.get("fixed_subset_payload")
    resampling_payload = data.get("resampling_payload")
    if fixed_subset_payload:
        from rage.experiments.resampling import FixedSubsetWindows
        payload_path = resolve(fixed_subset_payload)
        pool_root = resolve(required(data, "resampling_pool"))
        with np.load(payload_path, allow_pickle=False) as archive:
            selected_indices = np.asarray(archive["selected_indices"], dtype=np.int64)
            metadata = json.loads(str(archive["metadata"]))
        pool_manifest = json.loads((pool_root / "manifest.json").read_text(encoding="ascii"))
        if int(pool_manifest["window_length"]) != context + horizon:
            raise ValueError("fixed subset pool window width differs from training config")
        values = np.load(pool_root / "values.npy", mmap_mode="r", allow_pickle=False)
        locations = np.load(pool_root / "window_locations.npy", mmap_mode="r", allow_pickle=False)
        windows = FixedSubsetWindows(values, locations, selected_indices,
            window_length=context + horizon, metadata=metadata)
        windows_path = payload_path
    elif resampling_payload:
        from rage.experiments.resampling import ResampledWindows
        payload_path = resolve(resampling_payload)
        pool_root = resolve(required(data, "resampling_pool"))
        with np.load(payload_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["resampling_metadata"]))
            layout = str(archive["unit_layout"])
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
            if layout == "window":
                unit_offsets = np.arange(len(probabilities) + 1, dtype=np.int64)
            elif layout in ("trajectory", "interval"):
                unit_offsets = np.asarray(archive["unit_offsets"], dtype=np.int64)
            else:
                raise ValueError("unknown resampling unit layout: " + layout)
        values = np.load(pool_root / "values.npy", mmap_mode="r", allow_pickle=False)
        locations = np.load(pool_root / "window_locations.npy", mmap_mode="r", allow_pickle=False)
        window_ids = np.load(pool_root / "window_ids.npy", mmap_mode="r", allow_pickle=False)
        pool_manifest = json.loads((pool_root / "manifest.json").read_text(encoding="ascii"))
        if int(pool_manifest["window_length"]) != context + horizon:
            raise ValueError("resampling pool window width differs from training config")
        windows = ResampledWindows(values, locations, window_ids, unit_offsets,
            np.arange(len(locations), dtype=np.int64), probabilities,
            window_length=context + horizon, pool_n=int(pool_manifest["total_equivalents"]),
            metadata=metadata)
        windows_path = payload_path
        if not len(windows):
            raise ValueError("resampling pool must be nonempty")
    else:
        windows_path = resolve(required(data, "windows"))
        with np.load(windows_path, allow_pickle=False) as archive:
            key = data.get("windows_key", "windows")
            if key not in archive:
                raise ValueError("window archive lacks array: " + key)
            windows = np.asarray(archive[key], dtype=np.float32)
        if windows.ndim != 2 or windows.shape[1] != context + horizon or not len(windows):
            raise ValueError("windows must be non-empty [N, context_length + prediction_length]")
        if not np.isfinite(windows).all():
            raise ValueError("windows contain non-finite values")

    run_id = str(required(config, "run_id"))
    destination = output_path("train", required(config, "output_dir"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("training output is not empty: %s" % destination)
    destination.mkdir(parents=True, exist_ok=True)

    logger = configure_logger("tsl.tools.train", "tools/train", run_id, "TRAIN")
    started = time.monotonic()
    steps = int(required(training, "steps"))
    batch_size = int(required(training, "batch_size"))
    learning_rate = float(required(training, "learning_rate"))
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    warmup_limit = int(config.get("warmup_max_steps", 1000))
    if stage == "warmup" and steps > warmup_limit:
        raise ValueError("warmup steps exceed warmup_max_steps")
    seed = int(training.get("seed", 0))
    model_path = config.get("model_path")
    device = str(config.get("device", "cuda"))
    logger.info("stage=start model=%s rows=%d context=%d horizon=%d steps=%d batch=%d lr=%g seed=%d",
                model_key, len(windows), context, horizon, steps, batch_size, learning_rate, seed)
    engine, _probe, identity = build_model(model_key, model_path=model_path, device=device,
                                           context_length=context, prediction_length=horizon)
    lora = apply_student_lora(engine, model_key, training.get("lora"))
    interval = max(1, int(training.get("log_every", 100)))
    evaluation_steps = tuple(int(value) for value in training.get("evaluation_steps", ()))
    if evaluation_steps and (tuple(sorted(set(evaluation_steps))) != evaluation_steps
                             or evaluation_steps[-1] != steps or evaluation_steps[0] <= 0):
        raise ValueError("evaluation_steps must be sorted, unique, positive, and end at steps")

    def on_step(step: int) -> None:
        if step in evaluation_steps and step != steps:
            snapshot_root = destination / "checkpoints" / ("step-%d" % step)
            checkpoint = snapshot_root / "checkpoint"
            if snapshot_root.exists():
                raise FileExistsError("snapshot output already exists: %s" % snapshot_root)
            logger.info("stage=snapshot-start step=%d output=%s", step, checkpoint)
            checkpoint.parent.mkdir(parents=True, exist_ok=False)
            _save_model_snapshot(engine, checkpoint)
            logger.info("stage=snapshot-complete step=%d output=%s", step, checkpoint)
        if step == 1 or step % interval == 0 or step == steps:
            logger.info("stage=train-progress step=%d/%d", step, steps)

    recipe_config = training.get("recipe")
    recipe = TrainingRecipe(**recipe_config) if recipe_config else None
    if recipe is not None and (recipe.batch_size != batch_size
                               or recipe.learning_rate != learning_rate):
        raise ValueError("training recipe batch_size and learning_rate must match training config")
    result = fine_tune(engine, windows, steps=steps, batch_size=batch_size,
                       learning_rate=learning_rate, seed=seed,
                       weight_decay=float(training.get("weight_decay", 0.01)),
                       recipe=recipe,
                       schedule_total_steps=(int(training.get("schedule_total_steps", steps))
                                             if recipe is not None else None),
                       on_step=on_step)
    if result.get("steps") != steps:
        raise RuntimeError("training stopped before the requested step count")
    model_dir = destination / "checkpoint"
    _save_model(engine, model_dir)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "run_id": run_id,
        "model": identity,
        "dataset": str(data.get("id", windows_path)),
        "window_count": int(len(windows)),
        "context_length": context,
        "prediction_length": horizon,
        "training": {**training, "result": result, "lora": lora},
        "checkpoint": str(model_dir),
    }
    _write_json(model_dir / "rage_checkpoint_manifest.json", manifest)
    _write_json(destination / "training_manifest.json", manifest)
    for checkpoint_step in evaluation_steps:
        if checkpoint_step == steps:
            continue
        snapshot_root = destination / "checkpoints" / ("step-%d" % checkpoint_step)
        snapshot_dir = snapshot_root / "checkpoint"
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(snapshot_dir)
        snapshot_manifest = dict(manifest)
        snapshot_manifest["checkpoint"] = str(snapshot_dir)
        snapshot_manifest["training"] = {
            **training, "checkpoint_step": checkpoint_step,
            "result": {"steps": checkpoint_step, "sample_exposures": checkpoint_step * batch_size},
            "lora": lora}
        _write_json(snapshot_dir / "rage_checkpoint_manifest.json", snapshot_manifest)
        (snapshot_root / ".complete").write_text("complete\n", encoding="ascii")
    (destination / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete elapsed=%s output=%s",
                format_duration(time.monotonic() - started), destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    model = str(required(config, "model"))
    dispatched = dispatch_model_environment(
        model, [str(Path(__file__).resolve()), "--config", str(resolve(args.config)),
                *sum((["--set", value] for value in args.set), [])])
    if dispatched is not None:
        return dispatched
    run(config, stage="train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
