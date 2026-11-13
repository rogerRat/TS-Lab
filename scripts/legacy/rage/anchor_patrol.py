#!/usr/bin/env python
"""Observational patrol for the anchor-budget run (anchor-budget-20260914).

Purely observational for the scan owned by main Codex (tmux
``anchor-budget-scan``): every check samples GPU utilisation/memory, disk,
``scan_status.json`` progress, scan pid liveness, incremental scan.log error
scan, and cache shard growth.  It never restarts or kills anything and never
touches the GPU itself.

Orchestration duties (each fires at most once, guarded by marker files):
- when every task's rank-0 trajectory is complete: launch the preliminary
  cache-only CRB replay (CPU, tmux ``anchor-sim-prelim``);
- when every manifest sample is complete: launch the final replay
  (CPU, tmux ``anchor-sim-final``);
- then watch for the final summary.json and switch to phase ``done``.

Outputs under the run root: patrol.jsonl (one snapshot per check),
patrol_alerts.jsonl, patrol_status.json (latest snapshot), markers under
patrol_state/.  Stop by creating PATROL_STOP in the run root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

DEFAULT_RUN_ROOT = "/root/autodl-tmp/rage/outputs/anchor-budget-20260914"
DEFAULT_PYTHON = "/root/miniconda3/envs/time/bin/python"
DEFAULT_REPO = "/root/TS-Lab"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    temporary.replace(path)


def sh(argv):
    """Run a command from an argv list; no shell, so a timeout kill is final.

    (A shell pipeline can leave grandchildren holding the output pipes open,
    which blocked a previous patrol build in select() forever.)
    """
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        try:
            out, _ = proc.communicate(timeout=30)
            return (out or "").strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return "ERR timeout"
    except Exception as error:  # noqa: BLE001 - patrol must never die
        return f"ERR {error}"


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def tmux_has(session):
    return f"{session}: " in sh(["tmux", "ls"])


def launch_simulate(mode, run_root, python, repo):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if mode == "final":
        out = run_root / "simulate_final"
        session = "anchor-sim-final"
    else:
        out = run_root / f"simulate_prelim_{stamp}"
        session = "anchor-sim-prelim"
    log = run_root / f"simulate_{mode}_{stamp}.log"
    command = (f"cd {repo} && PYTHONPATH={repo}/src {python} -u "
               f"-m rage.experiments.anchor_budget_simulate --run-root {run_root} "
               f"--mode {mode} --out {out} >> {log} 2>&1")
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-c",
                        command], capture_output=True, text=True, timeout=30)
        return dict(session=session, out=str(out), log=str(log), launched_at=utc())
    except Exception as error:  # noqa: BLE001
        return dict(error=repr(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--interval", type=float, default=240.0)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--max-stall-minutes", type=float, default=15.0)
    parser.add_argument("--min-disk-gib", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    (run_root / "patrol_state").mkdir(parents=True, exist_ok=True)
    for name in ("patrol.jsonl", "patrol_alerts.jsonl"):
        (run_root / name).touch(exist_ok=True)

    manifest = read_json(run_root / "sample_manifest.json")
    all_keys = set()
    rank0_keys = set()
    if manifest:
        for sample in manifest.get("samples", []):
            key = (sample.get("task"), int(sample.get("fragment_id", -1)),
                   int(sample.get("rank", 0)))
            all_keys.add(key)
            if int(sample.get("rank", 0)) == 0:
                rank0_keys.add(key)

    state = dict(last_cached=None, last_progress_ts=None,
                 log_offset=(run_root / "scan.log").stat().st_size
                 if (run_root / "scan.log").exists() else 0,
                 prelim_launched=False, final_launched=False,
                 final_ready_alerted=bool((run_root / "patrol_state" /
                                           "final_ready.json").exists()),
                 final_summary_seen=bool((run_root / "simulate_final" /
                                          "summary.json").exists()),
                 checks=0)
    prelim_marker = run_root / "patrol_state" / "prelim_launched.json"
    final_marker = run_root / "patrol_state" / "final_launched.json"
    if prelim_marker.exists():
        state["prelim_launched"] = True
    if final_marker.exists():
        state["final_launched"] = True

    while not (run_root / "PATROL_STOP").exists():
        snapshot = dict(ts=utc(), checks=state["checks"],
                        prelim_launched=state["prelim_launched"],
                        final_launched=state["final_launched"])
        alerts = []

        gpu = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                  "--format=csv,noheader,nounits"])
        try:
            util, memory = (float(piece) for piece in gpu.split(","))
            snapshot["gpu_util_pct"], snapshot["gpu_mem_mib"] = util, memory
        except ValueError:
            snapshot["gpu_raw"] = gpu
        df = sh(["df", "-B1G", "--output=avail", "/root/autodl-tmp"])
        try:
            snapshot["disk_free_gib"] = round(float(df.splitlines()[-1]), 1)
        except (ValueError, IndexError):
            snapshot["disk_raw"] = df

        status = read_json(run_root / "scan_status.json")
        if status:
            snapshot.update(
                event=status.get("event"),
                cached_windows=status.get("cached_windows"),
                completed_samples=len(status.get("completed_samples") or []),
                current_sample=(status.get("current_sample") or {}).get("task"),
                wps=round(float(status.get("windows_per_second") or 0), 2),
                eta_minutes=round(float(status.get("eta_seconds") or 0) / 60, 1),
                scan_pid=status.get("pid"),
                scan_pid_alive=pid_alive(status.get("pid")))
            if status.get("event") == "SCANNING":
                if state["last_cached"] == status.get("cached_windows"):
                    stall = ((time.time() - state["last_progress_ts"]) / 60
                             if state["last_progress_ts"] else 0)
                    snapshot["stall_minutes"] = round(stall, 1)
                    if stall > args.max_stall_minutes:
                        alerts.append(f"scan_stalled_{int(stall)}min")
                        if snapshot.get("gpu_util_pct") == 0:
                            alerts.append("gpu_idle_while_stalled")
                else:
                    state["last_cached"] = status.get("cached_windows")
                    state["last_progress_ts"] = time.time()
                if not snapshot["scan_pid_alive"]:
                    alerts.append("scan_pid_dead_while_SCANNING")
            elif status.get("event") in ("FAILED", "ERROR"):
                alerts.append(f"scan_event_{status.get('event')}")
        else:
            alerts.append("scan_status_unreadable")

        if (run_root / "scan_error.json").exists():
            alerts.append("scan_error_json_present")

        log_path = run_root / "scan.log"
        if log_path.exists():
            try:
                data = log_path.read_bytes()
            except OSError as error:
                alerts.append(f"log_read_error {error!r}")
                data = b""
            fresh = data[state["log_offset"]:]
            state["log_offset"] = len(data)
            bad = [line for line in fresh.decode("utf-8", "replace").splitlines()
                   if ("Traceback" in line or "OutOfMemory" in line
                       or "CUDA out of memory" in line or line.startswith("ERROR"))]
            if bad:
                alerts.append(f"log_errors+{len(bad)}")
                snapshot["log_error_lines"] = bad[:3]

        if isinstance(snapshot.get("disk_free_gib"), float) and \
                snapshot["disk_free_gib"] < args.min_disk_gib:
            alerts.append("disk_low")

        done = completed_keys(run_root)
        snapshot["completed_total"] = len(done)
        snapshot["completed_of"] = len(all_keys)
        if all_keys and done:
            if not state["prelim_launched"] and rank0_keys and rank0_keys.issubset(done):
                record = launch_simulate("preliminary", run_root, args.python, args.repo)
                atomic_json(prelim_marker, record)
                state["prelim_launched"] = True
                snapshot["prelim_launched"] = True
                alerts.append("prelim_simulate_launched")
            # Final replay is manually gated: it must use main Codex's updated
            # simulate (30% budget cap / boundary points / best_effort).  Patrol
            # only flags readiness; the operator verifies the update and
            # launches tmux anchor-sim-final by hand.
            if (all_keys.issubset(done) and not state["final_ready_alerted"]
                    and not (run_root / "patrol_state" / "final_ready.json").exists()):
                state["final_ready_alerted"] = True
                atomic_json(run_root / "patrol_state" / "final_ready.json",
                            {"ts": snapshot["ts"]})
                alerts.append("final_ready_manual_launch_pending_simulate_update")

        summary = run_root / "simulate_final" / "summary.json"
        if summary.exists():
            snapshot["final_summary_present"] = True
            if not state["final_summary_seen"]:
                state["final_summary_seen"] = True
                alerts.append("final_simulate_summary_written")
            snapshot["phase"] = "done"
        elif state["final_ready_alerted"] or (run_root / "patrol_state" /
                                              "final_ready.json").exists():
            state["final_ready_alerted"] = True
            snapshot["phase"] = "awaiting_final_simulate"

        snapshot["alerts"] = alerts
        with open(run_root / "patrol.jsonl", "a") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        for alert in alerts:
            with open(run_root / "patrol_alerts.jsonl", "a") as stream:
                stream.write(json.dumps({"ts": snapshot["ts"], "alert": alert}) + "\n")
        atomic_json(run_root / "patrol_status.json", snapshot)
        state["checks"] += 1
        print(f"[patrol] check {state['checks']} at {snapshot['ts']}: "
              f"cached={snapshot.get('cached_windows')} "
              f"alerts={alerts}", flush=True)

        if args.once:
            break
        if state["final_summary_seen"] and state["checks"] % 3 == 0:
            with open(run_root / "patrol.jsonl", "a") as stream:
                stream.write(json.dumps({"ts": utc(), "phase": "patrol_complete"}) + "\n")
            break
        time.sleep(args.interval)


def completed_keys(run_root):
    status = read_json(Path(run_root) / "scan_status.json")
    if not status:
        return set()
    return {(s.get("task"), int(s.get("fragment_id", -1)), int(s.get("rank", 0)))
            for s in (status.get("completed_samples") or [])}


if __name__ == "__main__":
    main()
