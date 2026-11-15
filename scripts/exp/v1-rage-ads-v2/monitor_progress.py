"""Record v2 experiment status every ten minutes until train/eval completes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path("/root/TS-Lab")
from experiment_config import TASKS
EXP = "v1-rage-ads-v2"
LOG_DIR = ROOT / "logs" / EXP
DOC_LOG_DIR = ROOT / "docs/logs"
SELECTION = ROOT / "outputs/selection" / EXP
TRAIN = ROOT / "outputs/train" / EXP
EVAL = ROOT / "outputs/eval" / EXP
INTERVAL = 600
last_documented_signature = None


def run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as exc:
        return "unavailable: " + str(exc)


def last_line(path: Path, pattern: str | None = None) -> str:
    if not path.is_file():
        return "no task log"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if pattern:
        matching = [line for line in lines if re.search(pattern, line)]
        if matching:
            return matching[-1][-500:]
    return lines[-1][-500:] if lines else "empty task log"


def poll() -> dict:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    gpu = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"])
    memory_values = {}
    for field in ("memory.current", "memory.max", "memory.peak", "memory.events"):
        path = Path("/sys/fs/cgroup") / field
        try:
            memory_values[field] = path.read_text(encoding="ascii").strip()
        except OSError:
            memory_values[field] = "unavailable"
    processes = run(["bash", "-lc", "ps -eo pid,etime,stat,%cpu,%mem,args | grep -E 'run_selection.py|run_parallel_train_eval.py|train_eval.py' | grep -v grep"])
    task_status = {}
    for task in TASKS:
        marker = (SELECTION / task / ".complete").is_file()
        task_status[task] = {
            "complete": marker,
            "progress": last_line(LOG_DIR / ("selection-" + task + ".log"), r"stage=(gradient-progress|task-complete)|Traceback|Error|Killed"),
        }
    complete_train = (TRAIN / ".complete").is_file()
    complete_eval = (EVAL / ".complete").is_file()
    status = {
        "checked_at": stamp,
        "selection_queue": last_line(LOG_DIR / "selection-queue.log"),
        "pipeline": last_line(LOG_DIR / "pipeline.log"),
        "selection_completed_tasks": sum(item["complete"] for item in task_status.values()),
        "selection_total_tasks": len(TASKS),
        "task_status": task_status,
        "training_complete": complete_train,
        "evaluation_complete": complete_eval,
        "gpu": gpu,
        "container_memory": memory_values,
        "processes": processes,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "progress-poll.jsonl").open("a", encoding="ascii") as stream:
        stream.write(json.dumps(status, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    signature = (tuple(task for task, item in task_status.items() if item["complete"]),
                 complete_train, complete_eval)
    global last_documented_signature
    if signature != last_documented_signature:
        DOC_LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe_stamp = time.strftime("%Y-%m-%d-%H%M%S")
        doc = DOC_LOG_DIR / (safe_stamp + "-v1-rage-ads-v2-status.md")
        lines = [
            "# v1-rage-ads-v2 \u9636\u6bb5\u72b6\u6001\u53d8\u5316",
            "",
            "- \u68c0\u67e5\u65f6\u95f4\uff1a" + stamp,
            "- \u9009\u62e9\u5b8c\u6210\u4efb\u52a1\uff1a" + str(status["selection_completed_tasks"]) + "/" + str(len(TASKS)),
            "- \u8bad\u7ec3\u5b8c\u6210\uff1a" + str(complete_train),
            "- \u8bc4\u4f30\u5b8c\u6210\uff1a" + str(complete_eval),
            "- GPU\uff1a" + gpu,
            "- \u961f\u5217\u72b6\u6001\uff1a" + status["selection_queue"],
            "- \u9636\u6bb5\u72b6\u6001\uff1a" + status["pipeline"],
            "",
            "## \u4efb\u52a1\u72b6\u6001",
            "",
        ]
        for task, item in task_status.items():
            lines.append("- %s: complete=%s; %s" % (task, item["complete"], item["progress"]))
        lines.extend(["", "## \u8fd0\u884c\u8fdb\u7a0b", "", processes or "no matching process", ""])
        doc.write_text("\n".join(lines), encoding="utf-8")
        last_documented_signature = signature
    return status


def main() -> None:
    while True:
        status = poll()
        if status["training_complete"] and status["evaluation_complete"]:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
