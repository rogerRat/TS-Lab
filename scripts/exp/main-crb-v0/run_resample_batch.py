"""Run the eight-task full-pool resampling batch in resumable stages."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from configs.logger import configure_logger
BATCH = "q16-src256-eight-task-resample"
TASKS = ("FEV09", "TIME1", "TIME2", "TIME3", "TIME4", "TIME6", "TIME7", "TIME8")
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-v0/resample"
TRAIN_ROOT = Path("/root/autodl-tmp/rage/outputs/train/main-crb-v0") / BATCH
EVAL_ROOT = Path("/root/autodl-tmp/rage/outputs/eval/main-crb-v0") / BATCH
ACTIVE = []
INTERRUPTED = False
ENV = dict(os.environ)
ENV["TSL_TOOL_ENV_DISPATCHED"] = "1"
ENV["PYTHONPATH"] = "/root/autodl-tmp/rage/third_parties/TIME/src" + os.pathsep + str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    ENV[key] = "4"

def yaml_read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def yaml_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=False), encoding="ascii")

def prepare_bootstrap():
    source_root = ROOT / "configs/experiments/main-crb-v0/eight-task"
    for task in TASKS:
        out_dir = CONFIG_ROOT / task
        # Reuse the registered safe-reference windows and frozen evaluation configs.
        ori_eval = yaml_read(source_root / task / "ori-eval.yaml")
        ori_eval["output_dir"] = str(EVAL_ROOT / task / "ori")
        yaml_write(out_dir / "ori-eval.yaml", ori_eval)
        yaml_write(out_dir / "ori-pipeline.yaml", {"schema_version": 1,
            "run_id": "main-crb-v0-resample-" + task.lower() + "-ori",
            "stages": [{"name": "eval", "config": str((out_dir / "ori-eval.yaml").relative_to(ROOT))}]})
        train = yaml_read(source_root / task / "reft-train.yaml")
        train["run_id"] = "main-crb-v0-resample-" + task.lower() + "-reft"
        train["output_dir"] = str(TRAIN_ROOT / task / "reft")
        yaml_write(out_dir / "reft-train.yaml", train)
        ev = yaml_read(source_root / task / "reft-eval.yaml")
        ev["checkpoint"] = str(TRAIN_ROOT / task / "reft" / "checkpoint")
        ev["output_dir"] = str(EVAL_ROOT / task / "reft")
        yaml_write(out_dir / "reft-eval.yaml", ev)
        yaml_write(out_dir / "reft-pipeline.yaml", {"schema_version": 1,
            "run_id": train["run_id"], "stages": [
                {"name": "train", "config": str((out_dir / "reft-train.yaml").relative_to(ROOT))},
                {"name": "eval", "config": str((out_dir / "reft-eval.yaml").relative_to(ROOT))}]})

def on_signal(_sig, _frame):
    global INTERRUPTED
    INTERRUPTED = True
    for process in list(ACTIVE):
        if process.poll() is None:
            try: os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError: pass

def run_task(task, logger):
    for arm in ("ori", "reft"):
        done = EVAL_ROOT / task / arm / ".complete"
        if done.is_file():
            logger.info("stage=arm-skip task=%s arm=%s reason=complete", task, arm)
            continue
        pipeline = CONFIG_ROOT / task / (arm + "-pipeline.yaml")
        command = [sys.executable, "-u", "-m", "scripts.tools.run", "--config", str(pipeline.relative_to(ROOT))]
        if arm == "reft" and (TRAIN_ROOT / task / arm / ".complete").is_file():
            command += ["--from-stage", "eval"]
        logger.info("stage=arm-start task=%s arm=%s", task, arm)
        proc = subprocess.Popen(command, cwd=ROOT, env=ENV, start_new_session=True)
        ACTIVE.append(proc)
        code = proc.wait()
        ACTIVE.remove(proc)
        if INTERRUPTED: raise SystemExit(130)
        if code: raise RuntimeError("pipeline failed task=%s arm=%s code=%d" % (task, arm, code))
        if not done.is_file(): raise RuntimeError("missing completion marker: " + str(done))
        logger.info("stage=arm-complete task=%s arm=%s", task, arm)
    (EVAL_ROOT / task / ".complete").write_text("complete\n", encoding="ascii")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("bootstrap",), default="bootstrap")
    p.add_argument("--parallelism", type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.parallelism <= len(TASKS): raise ValueError("parallelism must be in [1, 8]")
    logger = configure_logger("main_crb_resample_batch", "main-crb-v0", BATCH, a.stage, enable_heartbeat=True)
    logger.info("stage=config batch=%s stage_name=%s tasks=%s parallelism=%d arms=ori,reft", BATCH, a.stage, ",".join(TASKS), a.parallelism)
    signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)
    prepare_bootstrap()
    pending = list(TASKS); active = []; failed = False
    while pending or active:
        if INTERRUPTED:
            for task, proc in active:
                if proc.poll() is None:
                    try: os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError: pass
            for _task, proc in active: proc.wait()
            raise SystemExit(130)
        while pending and len(active) < a.parallelism:
            task = pending.pop(0)
            command = [sys.executable, "-u", str(Path(__file__)), "--stage", "task", "--task", task]
            # Task mode is handled in the child, keeping each arm serialized per task.
            proc = subprocess.Popen(command, cwd=ROOT, env=ENV, start_new_session=True)
            active.append((task, proc)); ACTIVE.append(proc)
            logger.info("stage=task-start task=%s", task)
        for task, proc in list(active):
            code = proc.poll()
            if code is None: continue
            active.remove((task, proc)); ACTIVE.remove(proc)
            logger.info("stage=task-end task=%s code=%d", task, code)
            if code:
                failed = True
                for _name, other in active:
                    if other.poll() is None:
                        try: os.killpg(other.pid, signal.SIGINT)
                        except ProcessLookupError: pass
                for _name, other in active: other.wait()
                raise RuntimeError("task failed: %s code=%d" % (task, code))
        if active: time.sleep(1)
    if not failed:
        (EVAL_ROOT / ".bootstrap-complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=complete batch=%s eval_output=%s train_output=%s", BATCH, EVAL_ROOT, TRAIN_ROOT)

if __name__ == "__main__":
    # Child task mode shares configuration and logger but processes one task only.
    if "--stage" in sys.argv and "task" in sys.argv:
        parser = argparse.ArgumentParser(); parser.add_argument("--stage"); parser.add_argument("--task", choices=TASKS)
        args = parser.parse_args(); prepare_bootstrap()
        logger = configure_logger("main_crb_resample_batch", "main-crb-v0", BATCH, "task-" + args.task.lower(), enable_heartbeat=True)
        signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)
        run_task(args.task, logger)
    else:
        main()
