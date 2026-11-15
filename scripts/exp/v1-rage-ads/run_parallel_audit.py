#!/usr/bin/env python3
"""Run the rolling audit as independently resumable task processes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

ROOT = PROJECT_ROOT
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling"
AUDIT_ROOT = SELECTION_ROOT / "audit"
SOURCE_UNIT_CACHE_ROOT = ROOT / "outputs/selection/v1-rage-ads/parallelism/source-q32-unit-cache"



def process_tree_pids(root_pids):
    parents = {}
    try:
        entries = Path("/proc").iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                raw = (entry / "stat").read_text(encoding="ascii")
                fields = raw[raw.rfind(")") + 2:].split()
                parents[pid] = int(fields[1])
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        return list(root_pids)
    children = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    seen = set(int(pid) for pid in root_pids)
    pending = list(seen)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return sorted(seen)

def cpu_snapshot(pids):
    pids = process_tree_pids(pids)
    process_seconds = 0.0
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    for pid in pids:
        try:
            raw = Path("/proc/%d/stat" % pid).read_text(encoding="ascii")
            fields = raw[raw.rfind(")") + 2:].split()
            process_seconds += (int(fields[11]) + int(fields[12])) / ticks_per_second
        except (OSError, IndexError, ValueError):
            pass
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
        counters = [int(value) for value in fields]
        host_counters = (sum(counters), counters[3] + counters[4])
    except (OSError, IndexError, ValueError):
        host_counters = None
    return process_seconds, host_counters


def resource_sample(pids):
    pids = process_tree_pids(pids)
    sample = {}
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True, timeout=3).strip().splitlines()[0]
        util, used, total = [float(value.strip()) for value in raw.split(",")]
        sample.update(gpu_utilization_percent=util, gpu_memory_used_mib=used,
                      gpu_memory_total_mib=total)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    try:
        values = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(value.split()[0])
        sample.update(host_memory_used_mib=(values["MemTotal"] - values["MemAvailable"]) / 1024.0,
                      host_memory_total_mib=values["MemTotal"] / 1024.0)
    except (OSError, KeyError, ValueError):
        pass
    rss_kib = 0
    for pid in pids:
        try:
            with open("/proc/%d/status" % pid, encoding="ascii") as stream:
                for line in stream:
                    if line.startswith("VmRSS:"):
                        rss_kib += int(line.split()[1])
                        break
        except (OSError, ValueError):
            pass
    sample["audit_process_rss_mib"] = rss_kib / 1024.0
    return sample


def stop_all(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _handle_termination(signum, _frame):
    raise KeyboardInterrupt("received signal %d" % signum)


def main():
    signal.signal(signal.SIGTERM, _handle_termination)
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-window-batch", type=int, default=32)
    parser.add_argument("--vjp-batch-size", type=int, default=16)
    args = parser.parse_args()
    if (args.workers <= 0 or args.sample_count <= 0
            or args.gradient_window_batch <= 0 or args.vjp_batch_size <= 0):
        raise ValueError("worker, sample, and gradient batch sizes must be positive")
    if not (SELECTION_ROOT / ".complete").is_file():
        raise FileNotFoundError("selection output is incomplete")
    output = output_path("selection", AUDIT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("v1_rage_ads_parallel_audit", "v1-rage-ads",
                              "parallel-audit", "audit-runner")
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false",
               PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:"
                          "/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:" +
                          os.environ.get("PYTHONPATH", ""),
               V1_RAGE_ADS_SOURCE_UNIT_CACHE=str(SOURCE_UNIT_CACHE_ROOT))
    sys.path.insert(0, str(ROOT / "scripts/exp/v1-rage-ads"))
    from audit_metrics import prepare_source_unit_cache
    _cache_root, cache_prepare_seconds, cache_created = prepare_source_unit_cache(logger)
    logger.info("stage=source-unit-cache-ready created=%s elapsed=%s path=%s",
                cache_created, format_duration(cache_prepare_seconds), SOURCE_UNIT_CACHE_ROOT)
    pending = [task for task in TASKS if not (
        (output / "tasks" / task / ".complete").is_file()
        and (output / "tasks" / task / "manifest.json").is_file()
        and json.loads((output / "tasks" / task / "manifest.json").read_text(encoding="ascii")).get("sample_count") == args.sample_count
        and json.loads((output / "tasks" / task / "manifest.json").read_text(encoding="ascii")).get("seed") == args.seed
        and json.loads((output / "tasks" / task / "manifest.json").read_text(encoding="ascii")).get("gradient_window_batch_size") == args.gradient_window_batch
        and json.loads((output / "tasks" / task / "manifest.json").read_text(encoding="ascii")).get("gradient_vjp_batch_size") == args.vjp_batch_size)]
    logger.info("stage=config tasks=%s workers=%d sample_count=%d seed=%d window_batch=%d vjp_batch=%d output=%s",
                ",".join(pending), args.workers, args.sample_count, args.seed,
                args.gradient_window_batch, args.vjp_batch_size, output)
    running = {}
    completed = []
    errors = []
    samples = []
    durations = []
    started = time.monotonic()
    last_heartbeat = started
    previous_cpu_seconds = None
    previous_host_counters = None
    previous_sample_time = None
    interrupted = False
    try:
        while pending or running:
            while pending and len(running) < args.workers and not errors:
                task = pending.pop(0)
                command = [sys.executable, "-u", str(ROOT / "scripts/exp/v1-rage-ads/audit_metrics.py"),
                           "--task", task, "--sample-count", str(args.sample_count),
                           "--seed", str(args.seed), "--gradient-window-batch",
                           str(args.gradient_window_batch), "--vjp-batch-size",
                           str(args.vjp_batch_size), "--output-tag", "production"]
                process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
                running[process] = dict(task=task, started=time.monotonic())
                logger.info("stage=task-start task=%s active=%d/%d", task, len(running), args.workers)
            for process, info in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                elapsed = time.monotonic() - info["started"]
                durations.append(elapsed)
                del running[process]
                if code == 0 and (output / "tasks" / info["task"] / ".complete").is_file():
                    completed.append(info["task"])
                    logger.info("stage=task-complete task=%s elapsed=%s completed=%d/%d",
                                info["task"], format_duration(elapsed), len(completed), len(TASKS))
                else:
                    errors.append(dict(task=info["task"], return_code=code))
                    logger.error("stage=task-error task=%s return_code=%d", info["task"], code)
            pids = [process.pid for process in running]
            measured = resource_sample(pids)
            now = time.monotonic()
            current_cpu_seconds, current_host_counters = cpu_snapshot(pids)
            cpu_fields = {}
            if previous_sample_time is not None and now > previous_sample_time:
                interval = now - previous_sample_time
                if previous_cpu_seconds is not None:
                    core_equivalents = max(0.0, current_cpu_seconds - previous_cpu_seconds) / interval
                    cpu_fields["process_cpu_core_equivalents"] = core_equivalents
                    cpu_fields["process_cpu_percent_of_host_capacity"] = (
                        100.0 * core_equivalents / max(os.cpu_count() or 1, 1))
                if previous_host_counters and current_host_counters:
                    total_delta = current_host_counters[0] - previous_host_counters[0]
                    idle_delta = current_host_counters[1] - previous_host_counters[1]
                    if total_delta > 0:
                        cpu_fields["host_cpu_utilization_percent"] = max(
                            0.0, 100.0 * (1.0 - idle_delta / total_delta))
            previous_cpu_seconds, previous_host_counters = current_cpu_seconds, current_host_counters
            previous_sample_time = now
            if measured:
                samples.append(dict(**measured, **cpu_fields, sampled_at=time.time()))
            if now - last_heartbeat >= 30:
                done = len(completed)
                eta = ((now - started) * max(len(TASKS) - done, 0) / max(done, 1))
                logger.info("stage=heartbeat completed=%d/%d active=%d pending=%d eta=%s",
                            done, len(TASKS), len(running), len(pending), format_duration(eta))
                last_heartbeat = now
            if errors and not running:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        interrupted = True
        stop_all(list(running))
        raise
    finally:
        if interrupted:
            logger.warning("stage=interrupted active_processes_stopped=true")
    if errors or pending:
        raise RuntimeError("audit task workers failed; completed tasks remain resumable")
    finalizer = [sys.executable, "-u", str(ROOT / "scripts/exp/v1-rage-ads/audit_metrics.py"),
                 "--task", "all", "--sample-count", str(args.sample_count),
                 "--seed", str(args.seed), "--gradient-window-batch",
                 str(args.gradient_window_batch), "--vjp-batch-size",
                 str(args.vjp_batch_size), "--output-tag", "production", "--finalize-only"]
    subprocess.run(finalizer, cwd=ROOT, env=env, check=True)
    if not (output / ".complete").is_file():
        raise RuntimeError("audit finalizer did not create the complete marker")
    manifest = dict(experiment="v1-rage-ads", phase="audit", tasks=list(TASKS),
                    sample_count=args.sample_count, seed=args.seed, workers=args.workers,
                    gradient_window_batch_size=args.gradient_window_batch,
                    gradient_vjp_batch_size=args.vjp_batch_size,
                    source_unit_cache_preparation_seconds=cache_prepare_seconds,
                    completed_task_durations_seconds=durations,
                    gpu_utilization_average_percent=(sum(x["gpu_utilization_percent"] for x in samples)
                                                     / len(samples) if samples else None),
                    gpu_memory_average_mib=(sum(x["gpu_memory_used_mib"] for x in samples)
                                            / len(samples) if samples else None),
                    gpu_memory_peak_mib=(max(x["gpu_memory_used_mib"] for x in samples)
                                         if samples else None),
                    host_memory_average_mib=(sum(x["host_memory_used_mib"] for x in samples)
                                             / len(samples) if samples else None),
                    host_memory_peak_mib=(max(x["host_memory_used_mib"] for x in samples)
                                          if samples else None),
                    process_rss_average_mib=(sum(x["audit_process_rss_mib"] for x in samples)
                                             / len(samples) if samples else None),
                    process_rss_peak_mib=(max(x["audit_process_rss_mib"] for x in samples)
                                          if samples else None),
                    process_cpu_core_equivalents_average=(sum(x["process_cpu_core_equivalents"] for x in samples
                                                               if "process_cpu_core_equivalents" in x)
                                                          / sum("process_cpu_core_equivalents" in x for x in samples)
                                                          if any("process_cpu_core_equivalents" in x for x in samples) else None),
                    process_cpu_percent_of_host_capacity_average=(sum(x["process_cpu_percent_of_host_capacity"] for x in samples
                                                                       if "process_cpu_percent_of_host_capacity" in x)
                                                                  / sum("process_cpu_percent_of_host_capacity" in x for x in samples)
                                                                  if any("process_cpu_percent_of_host_capacity" in x for x in samples) else None),
                    host_cpu_utilization_average_percent=(sum(x["host_cpu_utilization_percent"] for x in samples
                                                              if "host_cpu_utilization_percent" in x)
                                                         / sum("host_cpu_utilization_percent" in x for x in samples)
                                                         if any("host_cpu_utilization_percent" in x for x in samples) else None),
                    monitor_samples=len(samples), elapsed_seconds=time.monotonic() - started,
                    output=str(output))
    target = output / "parallel-run-manifest.json"
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, target)
    logger.info("stage=complete tasks=%d workers=%d elapsed=%s gpu_avg=%s gpu_peak_mib=%s host_avg_mib=%s host_peak_mib=%s output=%s",
                len(completed), args.workers, format_duration(manifest["elapsed_seconds"]),
                manifest["gpu_utilization_average_percent"], manifest["gpu_memory_peak_mib"],
                manifest["host_memory_average_mib"], manifest["host_memory_peak_mib"], output)


if __name__ == "__main__":
    main()
