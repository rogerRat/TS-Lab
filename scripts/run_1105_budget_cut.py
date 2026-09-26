"""1105BudgetCutExp single-seed launcher: one model x one task x two arms, seed0 only."""
import argparse, json, os
from pathlib import Path

import sys
sys.path.insert(0, "/root/TS3/src")

from rags.experiments.full_pool_run import run_suite
from rags.experiments.students import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    run = Path(args.run)
    folder = run / "rounds" / "seed0"
    folder.mkdir(parents=True, exist_ok=True)
    control_payload = dict(mode="formal", training_enabled=True,
                           selection_enabled=True, evaluation_enabled=True,
                           reason="1105BudgetCutExp REF CRB 1/1/0.5 vs 2/1/1")
    control = run / "execution_control.json"
    if not control.exists():
        atomic_json(control, dict(control_payload))
    atomic_json(folder / "execution_control.json", dict(control_payload, round_seed=0))
    atomic_json(run / "supervisor.json", dict(status="running", pid=os.getpid(),
                experiment="1105-budget-cut", config=str(args.config)))
    code = run_suite(config, folder, execution=dict(student_parallelism=1))
    atomic_json(run / "supervisor.json", dict(status="complete" if code in (None, 0) else "failed",
                pid=os.getpid(), exit_code=code, experiment="1105-budget-cut"))
    return code or 0


if __name__ == "__main__":
    raise SystemExit(main())
