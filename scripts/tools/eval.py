"""Run one FEV, TIME or QuitoBench evaluation from YAML configuration."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
from pathlib import Path

import yaml

from scripts.tools._common import apply_overrides, dispatch_model_environment, load_config, required, resolve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--evaluation-mode", choices=("single_window", "rolling"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    checkpoint = args.checkpoint or required(config, "checkpoint")
    output_dir = args.output_dir or required(config, "output_dir")
    eval_config = resolve(required(config, "evaluation_config"))
    raw_eval = yaml.safe_load(eval_config.read_text(encoding="ascii"))
    model = raw_eval.get("adapter", {}).get("config", {}).get("model", {}).get("model_key")
    if model:
        forwarded = [str(Path(__file__).resolve()), "--config", str(resolve(args.config))]
        if args.checkpoint:
            forwarded.extend(["--checkpoint", args.checkpoint])
        if args.output_dir:
            forwarded.extend(["--output-dir", args.output_dir])
        if args.evaluation_mode:
            forwarded.extend(["--evaluation-mode", args.evaluation_mode])
        forwarded.extend(sum((["--set", value] for value in args.set), []))
        dispatched = dispatch_model_environment(str(model), forwarded)
        if dispatched is not None:
            return dispatched
    from evaluate.run import main as evaluate_main

    forwarded = ["--checkpoint", str(resolve(checkpoint)), "--config", str(eval_config),
                 "--output-dir", str(resolve(output_dir))]
    if args.evaluation_mode:
        forwarded.extend(["--evaluation-mode", args.evaluation_mode])
    if config.get("resume", False):
        forwarded.append("--resume")
    status = evaluate_main(forwarded)
    if status == 0:
        from configs.paths import output_path
        destination = output_path("eval", output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / ".complete").write_text("complete\n", encoding="ascii")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
