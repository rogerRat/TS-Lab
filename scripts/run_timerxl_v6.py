"""Launch the document-frozen TimerXL V6 15-arm supplementary matrix."""
import json
import os
from pathlib import Path

from rage.experiments.seed_rounds import run_rounds
from rage.experiments.full_pool_run import run_suite
from rage.experiments.students import atomic_json


CONFIG = Path(os.environ.get("RAGE_TIMER_V6_CONFIG", "/root/TS-Lab/configs/rage/main_v6_timerxl.json"))
RUN = Path(os.environ.get("RAGE_TIMER_V6_RUN", "/root/autodl-tmp/rage/outputs/main-rank-volume-timerxl-v6-20260921"))


def main():
    config = json.loads(CONFIG.read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    control = RUN / "execution_control.json"
    if not control.exists():
        atomic_json(control, dict(mode="formal", training_enabled=True,
                                  selection_enabled=True, evaluation_enabled=True,
                                  reason="approved TimerXL V6 480->96 matrix"))
    atomic_json(RUN / "supervisor.json", dict(status="running", pid=os.getpid(),
                                               experiment="v6-timerxl", config=str(CONFIG)))

    # run_rounds expects a runner(config, folder, execution) callback. Keep the
    # callback explicit so every seed round uses the same frozen config and
    # execution-only GPU settings.
    def execute(round_config, folder, execution):
        return run_suite(round_config, folder, execution=execution)

    try:
        code = run_rounds(config, RUN, runner=execute)
        atomic_json(RUN / "supervisor.json", dict(status="complete", pid=os.getpid(),
                                                   exit_code=code, experiment="v6-timerxl"))
        return code
    except BaseException as error:
        atomic_json(RUN / "supervisor.json", dict(status="failed", pid=os.getpid(),
                                                   error=repr(error), experiment="v6-timerxl"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
