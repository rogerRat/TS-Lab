"""Train one registered forecasting model on prepared windows."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
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


def run(config: dict, *, stage: str) -> Path:
    import numpy as np
    import torch
    from rage.experiments.full_pool_models import build_model
    from rage.experiments.lora import apply_student_lora
    from rage.experiments.students import STUDENTS, HISTORICAL_STUDENTS, NON_DEPLOYABLE_STUDENTS
    from rage.experiments.train_eval import fine_tune

    model_key = str(required(config, "model"))
    if model_key not in STUDENTS + HISTORICAL_STUDENTS or model_key in NON_DEPLOYABLE_STUDENTS:
        raise ValueError("unsupported training model: " + model_key)
    data = config.get("data", {})
    training = config.get("training", {})
    windows_path = resolve(required(data, "windows"))
    with np.load(windows_path, allow_pickle=False) as archive:
        key = data.get("windows_key", "windows")
        if key not in archive:
            raise ValueError("window archive lacks array: " + key)
        windows = np.asarray(archive[key], dtype=np.float32)
    context = int(required(data, "context_length"))
    horizon = int(required(data, "prediction_length"))
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

    def on_step(step: int) -> None:
        if step == 1 or step % interval == 0 or step == steps:
            logger.info("stage=train-progress step=%d/%d", step, steps)

    result = fine_tune(engine, windows, steps=steps, batch_size=batch_size,
                       learning_rate=learning_rate, seed=seed,
                       weight_decay=float(training.get("weight_decay", 0.01)),
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
