"""Train and evaluate native-task-conditioned probability arms on eight targets."""
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from configs.logger import configure_logger
BATCH = "q16-src256-eight-task-resample"
TASKS = ("FEV09", "TIME1", "TIME2", "TIME3", "TIME4", "TIME6", "TIME7", "TIME8")
METHODS = ("tsds", "lp_mirror")
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-v0/resample"
SELECTION_ROOT = Path("/root/autodl-tmp/rage/outputs/selection/main-crb-v0") / BATCH
TRAIN_ROOT = Path("/root/autodl-tmp/rage/outputs/train/main-crb-v0") / BATCH
EVAL_ROOT = Path("/root/autodl-tmp/rage/outputs/eval/main-crb-v0") / BATCH
ENV = dict(os.environ); ENV["TSL_TOOL_ENV_DISPATCHED"] = "1"
ENV["PYTHONPATH"] = "/root/autodl-tmp/rage/third_parties/TIME/src" + os.pathsep + str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): ENV[key] = "4"
ACTIVE = []
INTERRUPTED = False

def read_yaml(path): return yaml.safe_load(path.read_text(encoding="utf-8"))
def write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=False), encoding="ascii")
def prepare():
    source_root = ROOT / "configs/experiments/main-crb-v0/eight-task"
    for task in TASKS:
        for method in METHODS:
            config_dir = CONFIG_ROOT / task
            train = read_yaml(source_root / task / "reft-train.yaml")
            train["run_id"] = "main-crb-v0-resample-" + task.lower() + "-" + method
            data = dict(train["data"]); data.pop("windows", None); data.pop("windows_key", None)
            data["id"] = "src-256-" + method + "-" + task.lower()
            data["resampling_payload"] = str(SELECTION_ROOT / method / task / "resampling_payload.npz")
            data["resampling_pool"] = str(SELECTION_ROOT / "shared/pool")
            train["data"] = data
            train["output_dir"] = str(TRAIN_ROOT / task / method)
            write_yaml(config_dir / (method + "-train.yaml"), train)
            ev = read_yaml(source_root / task / "ori-eval.yaml")
            ev["checkpoint"] = str(TRAIN_ROOT / task / method / "checkpoint")
            ev["output_dir"] = str(EVAL_ROOT / task / method)
            write_yaml(config_dir / (method + "-eval.yaml"), ev)
            write_yaml(config_dir / (method + "-pipeline.yaml"), {
                "schema_version": 1, "run_id": train["run_id"], "stages": [
                    {"name": "train", "config": str((config_dir / (method + "-train.yaml")).relative_to(ROOT))},
                    {"name": "eval", "config": str((config_dir / (method + "-eval.yaml")).relative_to(ROOT))}]})

def signal_children(_sig, _frame):
    global INTERRUPTED
    INTERRUPTED = True
    for proc in list(ACTIVE):
        if proc.poll() is None:
            try: os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError: pass

def task_main(task):
    logger = configure_logger("main_crb_resample_native_task", "main-crb-v0", BATCH, "task-" + task.lower(), enable_heartbeat=True)
    signal.signal(signal.SIGINT, signal_children); signal.signal(signal.SIGTERM, signal_children)
    for method in METHODS:
        eval_done = EVAL_ROOT / task / method / ".complete"
        if eval_done.is_file(): continue
        pipeline = CONFIG_ROOT / task / (method + "-pipeline.yaml")
        command = [sys.executable, "-u", "-m", "scripts.tools.run", "--config", str(pipeline.relative_to(ROOT))]
        if (TRAIN_ROOT / task / method / ".complete").is_file(): command += ["--from-stage", "eval"]
        logger.info("stage=arm-start task=%s method=%s", task, method)
        proc = subprocess.Popen(command, cwd=ROOT, env=ENV, start_new_session=True); ACTIVE.append(proc)
        code = proc.wait(); ACTIVE.remove(proc)
        if INTERRUPTED: raise SystemExit(130)
        if code or not eval_done.is_file(): raise RuntimeError("pipeline failed for %s/%s code=%d" % (task, method, code))
        logger.info("stage=arm-complete task=%s method=%s", task, method)
    (EVAL_ROOT / task / "native-methods.complete").write_text("complete\n", encoding="ascii")

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.parallelism <= 8: raise ValueError("parallelism must be in [1, 8]")
    logger = configure_logger("main_crb_resample_native_task_queue", "main-crb-v0", BATCH, "native-task-conditioned", enable_heartbeat=True)
    logger.info("stage=config methods=%s tasks=%s parallelism=%d", ",".join(METHODS), ",".join(TASKS), args.parallelism)
    prepare(); signal.signal(signal.SIGINT, signal_children); signal.signal(signal.SIGTERM, signal_children)
    pending = list(TASKS); active = []
    while pending or active:
        if INTERRUPTED:
            signal_children(signal.SIGINT, None)
            for _task, proc in active: proc.wait()
            raise SystemExit(130)
        while pending and len(active) < args.parallelism:
            task = pending.pop(0)
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--task", task],
                cwd=ROOT, env=ENV, start_new_session=True)
            active.append((task, proc)); ACTIVE.append(proc)
            logger.info("stage=task-start task=%s", task)
        for task, proc in list(active):
            code = proc.poll()
            if code is None: continue
            active.remove((task, proc)); ACTIVE.remove(proc)
            logger.info("stage=task-end task=%s code=%d", task, code)
            if code:
                signal_children(signal.SIGINT, None)
                for _name, other in active: other.wait()
                raise RuntimeError("task failed: %s" % task)
        if active: time.sleep(1)
    (EVAL_ROOT / "native-task-conditioned.complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete eval_output=%s train_output=%s", EVAL_ROOT, TRAIN_ROOT)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--task":
        prepare(); task_main(sys.argv[2])
    else: main()
