"""Run fresh selection, training, and evaluation for the registered eight tasks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration
from configs.paths import output_path

BATCH = "q16-src256-eight-task"
TASKS = ("FEV09", "TIME1", "TIME2", "TIME3", "TIME4", "TIME6", "TIME7", "TIME8")
ARMS = ("ori", "crb", "uni", "reft", "crb_reft", "uni_reft")
TRAIN_ARMS = ("crb", "uni", "reft", "crb_reft", "uni_reft")
PREP_SCRIPT = ROOT / "scripts/exp/main-crb-v0/prepare_eight_task.py"
QUEUE_SCRIPT = ROOT / "scripts/exp/main-crb-v0/run_eight_task_queue.py"
SELECTION_ROOT = output_path("selection", "outputs/selection/main-crb-v0/" + BATCH)
TRAIN_ROOT = output_path("train", "outputs/train/main-crb-v0/" + BATCH)
EVAL_ROOT = output_path("eval", "outputs/eval/main-crb-v0/" + BATCH)
ACTIVE: list[subprocess.Popen] = []
INTERRUPTED = False
ENV = dict(os.environ)
ENV["TSL_TOOL_ENV_DISPATCHED"] = "1"
TIME_SOURCE = "/root/autodl-tmp/rage/third_parties/TIME/src"
ENV["PYTHONPATH"] = TIME_SOURCE + os.pathsep + str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")
ENV["OMP_NUM_THREADS"] = "4"
ENV["MKL_NUM_THREADS"] = "4"
ENV["OPENBLAS_NUM_THREADS"] = "4"


def stop_children(_signum, _frame) -> None:
    global INTERRUPTED
    INTERRUPTED = True
    for process in list(ACTIVE):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


def spawn(command: list[str], label: str, logger):
    logger.info("stage=child-start label=%s command=%s", label, command)
    process = subprocess.Popen(command, cwd=ROOT, env=ENV, start_new_session=True)
    ACTIVE.append(process)
    return process


def wait_one(process: subprocess.Popen, label: str, logger) -> int:
    code = process.wait()
    if process in ACTIVE:
        ACTIVE.remove(process)
    logger.info("stage=child-end label=%s code=%d", label, code)
    return code


def run_one(command: list[str], label: str, logger) -> None:
    process = spawn(command, label, logger)
    code = wait_one(process, label, logger)
    if INTERRUPTED:
        raise SystemExit(130)
    if code:
        raise RuntimeError("child failed: %s code=%d" % (label, code))


def stop_and_reap(processes, logger) -> None:
    for label, process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for label, process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait()
        if process in ACTIVE:
            ACTIVE.remove(process)
        logger.info("stage=child-reaped label=%s code=%s", label, process.returncode)


def run_parallel(commands, parallelism: int, stage: str, logger) -> None:
    pending = list(commands)
    active: list[tuple[str, subprocess.Popen]] = []
    failed = False
    while pending or active:
        if INTERRUPTED:
            stop_and_reap(active, logger)
            raise SystemExit(130)
        while pending and len(active) < parallelism:
            label, command = pending.pop(0)
            active.append((label, spawn(command, label, logger)))
        for label, process in list(active):
            code = process.poll()
            if code is None:
                continue
            active.remove((label, process))
            if process in ACTIVE:
                ACTIVE.remove(process)
            logger.info("stage=child-end stage=%s label=%s code=%d", stage, label, code)
            if code:
                failed = True
                stop_and_reap(active, logger)
                raise RuntimeError("parallel child failed: %s code=%d" % (label, code))
        if active:
            time.sleep(0.25)
    if failed:
        raise RuntimeError("parallel stage failed: " + stage)


def run_task(task: str, logger) -> None:
    train_task_root = TRAIN_ROOT / task
    eval_task_root = EVAL_ROOT / task
    for arm in ARMS:
        eval_complete = eval_task_root / arm / ".complete"
        if eval_complete.is_file():
            logger.info("stage=arm-skip task=%s arm=%s reason=evaluation-complete", task, arm)
            continue
        pipeline = ROOT / "configs/experiments/main-crb-v0/eight-task" / task / (arm + "-pipeline.yaml")
        command = [sys.executable, "-u", "-m", "scripts.tools.run",
                   "--config", str(pipeline.relative_to(ROOT))]
        train_complete = (train_task_root / arm / ".complete").is_file()
        if arm in TRAIN_ARMS and train_complete:
            command.extend(["--from-stage", "eval"])
        logger.info("stage=arm-start task=%s arm=%s resume_eval_only=%s",
                    task, arm, train_complete and arm in TRAIN_ARMS)
        process = spawn(command, task + "-" + arm, logger)
        code = wait_one(process, task + "-" + arm, logger)
        if INTERRUPTED:
            raise SystemExit(130)
        if code:
            raise RuntimeError("tool pipeline failed for %s/%s code=%d" % (task, arm, code))
        if not eval_complete.is_file():
            raise RuntimeError("evaluation completion marker is missing for %s/%s" % (task, arm))
        if arm in TRAIN_ARMS and not (train_task_root / arm / ".complete").is_file():
            raise RuntimeError("training completion marker is missing for %s/%s" % (task, arm))
        logger.info("stage=arm-complete task=%s arm=%s", task, arm)
    (eval_task_root / ".complete").write_text("complete\n", encoding="ascii")
    (train_task_root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-complete task=%s", task)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "task", "union"), default="all")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args(argv)
    if args.parallelism < 1 or args.parallelism > len(TASKS):
        raise ValueError("parallelism must be in [1, 8]")
    logger = configure_logger(
        "main_crb_v0_eight_task_queue", "main-crb-v0", BATCH,
        "queue" if args.stage in ("all", "union") else "task-" + str(args.task).lower(),
        enable_heartbeat=True)
    logger.info("stage=config batch=%s stage_name=%s tasks=%s parallelism=%d",
                BATCH, args.stage, ",".join(TASKS), args.parallelism)
    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    if args.stage == "task":
        if not args.task:
            raise ValueError("--task is required for --stage=task")
        run_task(args.task, logger)
        return 0

    if args.stage == "union":
        for marker in (TRAIN_ROOT / ".complete", EVAL_ROOT / ".complete"):
            marker.unlink(missing_ok=True)
        for task in TASKS:
            (TRAIN_ROOT / task / ".complete").unlink(missing_ok=True)
            (EVAL_ROOT / task / ".complete").unlink(missing_ok=True)
        task_commands = [
            (task, [sys.executable, "-u", str(QUEUE_SCRIPT), "--stage", "task", "--task", task])
            for task in TASKS
        ]
        logger.info("stage=union-training-evaluation-start tasks=%d parallelism=%d", len(TASKS), args.parallelism)
        run_parallel(task_commands, args.parallelism, "union-training-evaluation", logger)
        for task in TASKS:
            if not (EVAL_ROOT / task / ".complete").is_file():
                raise RuntimeError("task completion marker is missing: " + task)
        (TRAIN_ROOT / ".complete").write_text("complete\n", encoding="ascii")
        (EVAL_ROOT / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=union-batch-complete tasks=%d train_output=%s eval_output=%s", len(TASKS), TRAIN_ROOT, EVAL_ROOT)
        return 0

    prep_command = [sys.executable, "-u", str(PREP_SCRIPT), "--stage", "resources"]
    run_one(prep_command, "reference-resources", logger)
    selection_commands = [
        (task, [sys.executable, "-u", str(PREP_SCRIPT), "--stage", "select", "--task", task])
        for task in TASKS
    ]
    logger.info("stage=selection-start tasks=%d parallelism=%d", len(TASKS), args.parallelism)
    run_parallel(selection_commands, args.parallelism, "selection", logger)
    (SELECTION_ROOT / ".complete").write_text("complete\n", encoding="ascii")

    task_commands = [
        (task, [sys.executable, "-u", str(QUEUE_SCRIPT), "--stage", "task", "--task", task])
        for task in TASKS
    ]
    logger.info("stage=training-evaluation-start tasks=%d parallelism=%d", len(TASKS), args.parallelism)
    run_parallel(task_commands, args.parallelism, "training-evaluation", logger)
    for task in TASKS:
        if not (EVAL_ROOT / task / ".complete").is_file():
            raise RuntimeError("task completion marker is missing: " + task)
    (TRAIN_ROOT / ".complete").write_text("complete\n", encoding="ascii")
    (EVAL_ROOT / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=batch-complete tasks=%d train_output=%s eval_output=%s",
                len(TASKS), TRAIN_ROOT, EVAL_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
