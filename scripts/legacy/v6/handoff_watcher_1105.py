"""Handoff watcher: after CT run seed0 completes, block further seeds and launch 1105BudgetCutExp."""
import json, os, subprocess, time
from pathlib import Path

CT = Path("/root/autodl-tmp/rage/outputs/main-rank-volume-20260921")
EXP = Path("/root/autodl-tmp/rage/outputs/exp-1105-budget-cut-20260922")
LOG = EXP / "handoff.log"

def log(msg):
    LOG.open("a").write(time.strftime("%F %T ") + msg + "\n")

def ct_alive():
    out = subprocess.run(["pgrep", "-f", "main-rank-volume-20260921/rounds"],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)

def seed0_complete():
    p = CT / "rounds" / "seed0" / "results.json"
    try:
        return json.loads(p.read_text()).get("complete") is True
    except Exception:
        return False

def block_seeds():
    ctl = CT / "execution_control.json"
    c = json.loads(ctl.read_text())
    if c.get("training_enabled") is not False:
        c["training_enabled"] = False
        c["mode"] = "seed0_done_multiseed_deferred"
        c["reason"] = "multi-seed deferred to TODO per 2026-09-22 instruction; 1105BudgetCutExp is next"
        ctl.write_text(json.dumps(c, indent=2) + "\n")
        log("execution_control: training_enabled=false (seed1/2 blocked)")

def launch():
    if (EXP / "LAUNCHED").exists():
        return
    (EXP / "LAUNCHED").write_text(time.strftime("%F %T"))
    log("launching 1105BudgetCutExp")
    subprocess.run(["tmux", "new-session", "-d", "-s", "exp-1105",
        "export PATH=/root/miniconda3/bin:$PATH PYTHONPATH=/root/TS-Lab/src RAGE_GRADIENT_CACHE_ROOT=/root/autodl-tmp/rage/cache/grad-pool; "
        "for m in timerxl_84m chronos_base timesfm_2_5; do "
        "echo === 1105 $m ===; "
        "cd /root/TS-Lab && /root/autodl-tmp/rage/envs/rage/bin/python -m scripts.run_1105_budget_cut "
        "--config /root/TS-Lab/configs/rage/exp_1105_budget_cut/1105_$m.json "
        "--run /root/autodl-tmp/rage/outputs/exp-1105-budget-cut-20260922/$m || break; done; exec bash"])
    log("tmux exp-1105 created")

log("watcher started")
alerted = False
while True:
    try:
        s0 = seed0_complete()
        alive = ct_alive()
        if s0:
            block_seeds()
            if not alive:
                launch()
                log("handoff complete; watcher exiting")
                break
        elif not alive:
            if not alerted:
                log("ALERT: CT supervisor gone but seed0 not complete; 1105 NOT launched; manual resume required")
                (EXP / "HANDOFF_BLOCKED.txt").write_text(
                    "CT seed0 incomplete and supervisor exited. Resolve CT seed0 first.\n")
                alerted = True
    except Exception as e:
        log("watcher error: %r" % e)
    time.sleep(5)
