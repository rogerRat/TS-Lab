"""Train shared full-pool probability arms and evaluate them on all tasks."""
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
METHODS = ("rand", "grand-static")
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-v0/resample/shared"
TRAIN_ROOT = Path("/root/autodl-tmp/rage/outputs/train/main-crb-v0") / BATCH
EVAL_ROOT = Path("/root/autodl-tmp/rage/outputs/eval/main-crb-v0") / BATCH
SELECTION_ROOT = Path("/root/autodl-tmp/rage/outputs/selection/main-crb-v0") / BATCH
ENV = dict(os.environ)
ENV["TSL_TOOL_ENV_DISPATCHED"] = "1"
ENV["PYTHONPATH"] = "/root/autodl-tmp/rage/third_parties/TIME/src" + os.pathsep + str(ROOT / "src") + os.pathsep + ENV.get("PYTHONPATH", "")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    ENV[key] = "4"

def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=False), encoding="ascii")

def prepare():
    template = read_yaml(ROOT / "configs/experiments/main-crb-v0/eight-task/TIME1/crb-train.yaml")
    for method in METHODS:
        train = dict(template)
        train["run_id"] = "main-crb-v0-resample-shared-" + method
        train_data = dict(train["data"])
        train_data.pop("windows", None)
        train_data.pop("windows_key", None)
        train_data["id"] = "src-256-full-pool-" + method
        train_data["resampling_payload"] = str(SELECTION_ROOT / method / "shared/resampling_payload.npz")
        train_data["resampling_pool"] = str(SELECTION_ROOT / "shared/pool")
        train["data"] = train_data
        train["output_dir"] = str(TRAIN_ROOT / "shared" / method)
        write_yaml(CONFIG_ROOT / (method + "-train.yaml"), train)
        for task in TASKS:
            eval_config = read_yaml(ROOT / "configs/experiments/main-crb-v0/eight-task" / task / "ori-eval.yaml")
            eval_config["checkpoint"] = str(TRAIN_ROOT / "shared" / method / "checkpoint")
            eval_config["output_dir"] = str(EVAL_ROOT / task / method)
            write_yaml(CONFIG_ROOT / task / (method + "-eval.yaml"), eval_config)
            write_yaml(CONFIG_ROOT / task / (method + "-pipeline.yaml"), {
                "schema_version": 1,
                "run_id": "main-crb-v0-resample-" + task.lower() + "-" + method,
                "stages": [{"name": "eval", "config": str((CONFIG_ROOT / task / (method + "-eval.yaml")).relative_to(ROOT))}]})

def run_process(command, label, logger):
    logger.info("stage=child-start label=%s command=%s", label, command)
    code = subprocess.call(command, cwd=ROOT, env=ENV)
    logger.info("stage=child-end label=%s code=%d", label, code)
    if code:
        raise RuntimeError("child failed: %s code=%d" % (label, code))

def run_train(method, logger):
    root = TRAIN_ROOT / "shared" / method
    if (root / ".complete").is_file():
        logger.info("stage=train-skip method=%s reason=complete", method)
        return
    run_process([sys.executable, "-u", "-m", "scripts.tools.train", "--config",
        str((CONFIG_ROOT / (method + "-train.yaml")).relative_to(ROOT))], "train-" + method, logger)
    if not (root / ".complete").is_file():
        raise RuntimeError("training completion marker missing for " + method)

def run_evaluations(parallelism, logger):
    pending = [(task, method) for method in METHODS for task in TASKS]
    active = []
    while pending or active:
        while pending and len(active) < parallelism:
            task, method = pending.pop(0)
            done = EVAL_ROOT / task / method / ".complete"
            if done.is_file():
                logger.info("stage=eval-skip task=%s method=%s", task, method)
                continue
            config = CONFIG_ROOT / task / (method + "-pipeline.yaml")
            command = [sys.executable, "-u", "-m", "scripts.tools.run", "--config", str(config.relative_to(ROOT))]
            proc = subprocess.Popen(command, cwd=ROOT, env=ENV)
            active.append((task, method, proc))
            logger.info("stage=eval-start task=%s method=%s", task, method)
        for task, method, proc in list(active):
            code = proc.poll()
            if code is None: continue
            active.remove((task, method, proc))
            logger.info("stage=eval-end task=%s method=%s code=%d", task, method, code)
            if code or not (EVAL_ROOT / task / method / ".complete").is_file():
                for _task, _method, other in active:
                    if other.poll() is None: other.send_signal(signal.SIGINT)
                for _task, _method, other in active: other.wait()
                raise RuntimeError("evaluation failed for %s/%s" % (task, method))
        if active: time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args()
    logger = configure_logger("main_crb_resample_shared", "main-crb-v0", BATCH, "shared-rand-grand", enable_heartbeat=True)
    logger.info("stage=config methods=%s tasks=%s parallelism=%d", ",".join(METHODS), ",".join(TASKS), args.parallelism)
    prepare()
    pending = list(METHODS)
    active = []
    while pending or active:
        while pending and len(active) < len(METHODS):
            method = pending.pop(0)
            if (TRAIN_ROOT / "shared" / method / ".complete").is_file(): continue
            command = [sys.executable, "-u", str(Path(__file__)), "--train-one", method]
            proc = subprocess.Popen(command, cwd=ROOT, env=ENV)
            active.append((method, proc))
            logger.info("stage=train-start method=%s", method)
        for method, proc in list(active):
            code = proc.poll()
            if code is None: continue
            active.remove((method, proc))
            logger.info("stage=train-end method=%s code=%d", method, code)
            if code:
                for _name, other in active:
                    if other.poll() is None: other.send_signal(signal.SIGINT)
                for _name, other in active: other.wait()
                raise RuntimeError("shared training failed: " + method)
        if active: time.sleep(1)
    run_evaluations(args.parallelism, logger)
    for method in METHODS: (EVAL_ROOT / method).mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "shared-rand-grand.complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete train_output=%s eval_output=%s", TRAIN_ROOT / "shared", EVAL_ROOT)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--train-one":
        method = sys.argv[2]
        logger = configure_logger("main_crb_resample_shared", "main-crb-v0", BATCH, "train-" + method, enable_heartbeat=True)
        prepare(); run_train(method, logger)
    else:
        main()
