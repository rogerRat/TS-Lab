"""Build a projected gradient cache from a prepared window archive."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
import time
from pathlib import Path

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from assets.gradient.runner import main as gradient_main
from scripts.tools._common import apply_overrides, dispatch_model_environment, load_config, required, resolve


def main(argv=None) -> int:
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

    data = config.get("data", {})
    gradient = config.get("gradient", {})
    dataset_id = str(required(data, "id"))
    windows = resolve(required(data, "windows"))
    destination = output_path("grad_cache", required(config, "output_dir"))
    logger = configure_logger("tsl.tools.grad_cache", "tools/grad_cache",
                              str(config.get("run_id", dataset_id)), "GRAD_CACHE")
    started = time.monotonic()
    logger.info("stage=start model=%s dataset=%s windows=%s output=%s",
                model, dataset_id, windows, destination)
    runner_args = [
        "--model", model, "--windows", str(windows),
        "--run-dir", str(destination), "--dataset", dataset_id,
        "--device", str(config.get("device", "cuda")),
        "--gradient-kind", str(gradient.get("kind", "full_model")),
        "--gradient-batch-size", str(gradient.get("batch_size", 0)),
        "--gradient-workers", str(gradient.get("workers", 2)),
    ]
    if config.get("model_path"):
        runner_args.extend(["--model-path", str(resolve(config["model_path"]))])
    status = gradient_main(runner_args)
    if status == 0:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=complete elapsed=%s output=%s",
                    format_duration(time.monotonic() - started), destination)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
