"""Run a bounded command queue with process-group cleanup."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from scripts.tools._common import apply_overrides, load_config, required, resolve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    tasks = required(config, "tasks")
    workers = int(config.get("workers", 1))
    if not isinstance(tasks, list) or not tasks or workers < 1:
        raise ValueError("tasks must be non-empty and workers must be positive")
    task_names = [row.get("name") if isinstance(row, dict) else None for row in tasks]
    if len(set(task_names)) != len(task_names):
        raise ValueError("task names must be unique")
    for row in tasks:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not isinstance(row.get("command"), list)
                or not row["command"] or not all(isinstance(part, str) for part in row["command"])):
            raise ValueError("each task requires a name and a non-empty command string list")

    run_id = str(required(config, "run_id"))
    logger = configure_logger("tsl.tools.parallel", "tools/parallel", run_id, "QUEUE")
    root = Path(__file__).resolve().parents[2]
    out = output_path("train", required(config, "output_dir"))
    out.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    active: dict[str, subprocess.Popen] = {}
    failures: list[dict] = []
    stopping = False

    def terminate(_signum, _frame):
        nonlocal stopping
        stopping = True
        for process in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

    old_int = signal.signal(signal.SIGINT, terminate)
    old_term = signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    logger.info("stage=start tasks=%d workers=%d output=%s", len(pending), workers, out)
    try:
        while (pending or active) and not stopping:
            while pending and len(active) < workers:
                row = pending.pop(0)
                name = row["name"]
                command = [part.replace("{project_root}", str(root)) for part in row["command"]]
                logger.info("stage=launch task=%s command=%s", name, command)
                active[name] = subprocess.Popen(command, cwd=root, start_new_session=True)
            for name, process in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[name]
                logger.info("stage=worker-exit task=%s code=%d", name, code)
                if code:
                    failures.append({"name": name, "exit_code": code})
                    stopping = True
                    for sibling in active.values():
                        if sibling.poll() is None:
                            try:
                                os.killpg(sibling.pid, signal.SIGINT)
                            except ProcessLookupError:
                                pass
            if active or pending:
                time.sleep(1)
        for name, process in list(active.items()):
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                code = process.wait()
            if code:
                failures.append({"name": name, "exit_code": code})
        active.clear()
        if failures or stopping:
            raise RuntimeError("queue stopped; failures=%s interrupted=%s" % (failures, stopping))
        (out / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=complete elapsed=%s output=%s",
                    format_duration(time.monotonic() - started), out)
        return 0
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        for process in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for process in active.values():
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
