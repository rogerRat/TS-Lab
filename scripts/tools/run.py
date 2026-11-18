"""Run an ordered set of configured TS-Lab tool stages."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time

from configs.logger import configure_logger, format_duration
from scripts.tools._common import apply_overrides, load_config, resolve, required

TOOLS = {
    "warmup": "warmup.py",
    "grad_cache": "grad_cache.py",
    "select": "select.py",
    "train": "train.py",
    "eval": "eval.py",
    "replay": "replay.py",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--from-stage", choices=tuple(TOOLS))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    logger = configure_logger("tsl.tools.run", "tools/run",
                              str(config.get("run_id", "pipeline")), "PIPELINE")
    started = time.monotonic()
    stages = required(config, "stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty ordered list")
    root = Path(__file__).resolve().parents[2]
    start_index = 0
    if args.from_stage:
        names = [row.get("name") if isinstance(row, dict) else None for row in stages]
        if args.from_stage not in names:
            raise ValueError("--from-stage is absent from the configured pipeline")
        start_index = names.index(args.from_stage)

    active = None

    def stop_child(_signum, _frame):
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT)

    previous = signal.signal(signal.SIGTERM, stop_child)
    try:
        for row in stages[start_index:]:
            if not isinstance(row, dict) or row.get("name") not in TOOLS:
                raise ValueError("each stage must name one supported tool")
            name = row["name"]
            stage_config = resolve(required(row, "config"))
            module = "scripts.tools." + Path(TOOLS[name]).stem
            command = [sys.executable, "-m", module, "--config", str(stage_config)]
            if row.get("method"):
                command.extend(["--method", str(row["method"])])
            logger.info("stage=start name=%s command=%s", name, command)
            active = subprocess.Popen(command, cwd=root)
            code = active.wait()
            active = None
            logger.info("stage=end name=%s code=%d", name, code)
            if code:
                raise SystemExit(code)
        logger.info("stage=complete elapsed=%s", format_duration(time.monotonic() - started))
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous)
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGINT)
            active.wait()


if __name__ == "__main__":
    raise SystemExit(main())
