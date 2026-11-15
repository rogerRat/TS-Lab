#!/usr/bin/env python3
"""Benchmark phase-level concurrency with measured GPU and host resources."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

ROOT = PROJECT_ROOT
SOURCE_UNIT_CACHE_ROOT = ROOT / "outputs/selection/v1-rage-ads/parallelism/source-q32-unit-cache"
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
MODEL_PATH = Path("/root/autodl-tmp/pretrained_models/timer-base-84m-84m")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


def gpu_sample():
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"], text=True, timeout=3).strip().splitlines()[0]
        util, used, total, power = [float(value.strip()) for value in raw.split(",")]
        return dict(gpu_utilization_percent=util, gpu_memory_used_mib=used,
                    gpu_memory_total_mib=total, gpu_power_watts=power)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def cpu_snapshot(pids):
    process_seconds = 0.0
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    for pid in pids:
        try:
            raw = Path("/proc/%d/stat" % pid).read_text(encoding="ascii")
            fields = raw[raw.rfind(")") + 2:].split()
            process_seconds += (int(fields[11]) + int(fields[12])) / ticks_per_second
        except (OSError, IndexError, ValueError):
            pass
    host = None
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
        counters = [int(value) for value in fields]
        host = (sum(counters), counters[3] + counters[4])
    except (OSError, IndexError, ValueError):
        pass
    return process_seconds, host


def memory_sample(pids):
    try:
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    fields[key] = int(value.split()[0])
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
        return dict(host_memory_used_mib=(fields["MemTotal"] - fields["MemAvailable"]) / 1024.0,
                    host_memory_total_mib=fields["MemTotal"] / 1024.0,
                    profile_children_rss_mib=rss_kib / 1024.0)
    except (OSError, KeyError, ValueError):
        return None


def terminate_all(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
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


def profile_setting(args, workers, root, logger):
    tag_suffix = ("-" + args.result_tag) if args.result_tag else ""
    setting_root = root / ("workers-%d" % workers + tag_suffix)
    setting_root.mkdir(parents=True, exist_ok=True)
    gate = setting_root / "release.gate"
    if gate.exists():
        gate.unlink()
    jobs = []
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS=str(args.cpu_threads),
                       MKL_NUM_THREADS=str(args.cpu_threads),
                       OPENBLAS_NUM_THREADS=str(args.cpu_threads),
                       NUMEXPR_NUM_THREADS=str(args.cpu_threads),
                       TOKENIZERS_PARALLELISM="false",
                       PYTHONUNBUFFERED="1",
                       PYTHONPATH="/root/TS-Lab/src:/root/TS-Lab/third_parties/fev/src:"
                                  "/root/TS-Lab/third_parties/TIME/src:/root/TS-Lab:" +
                                  os.environ.get("PYTHONPATH", ""),
                       V1_RAGE_ADS_SOURCE_UNIT_CACHE=str(SOURCE_UNIT_CACHE_ROOT))
    for index in range(workers):
        ready = setting_root / ("job-%02d.ready" % index)
        if ready.exists():
            ready.unlink()
        if args.phase == "train":
            destination = output_path("train", setting_root / ("job-%02d" % index))
            destination.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(ROOT / "scripts/exp/v1-rage-ads/parallel_profile.py"),
                       "--train-child", "--steps", str(args.steps), "--worker-id", str(index),
                       "--output", str(destination / "result.json")]
        elif args.phase == "audit":
            tag_prefix = (args.result_tag + "-") if args.result_tag else ""
            tag = "profile-%sw%d-j%d" % (tag_prefix, workers, index)
            command = [sys.executable, "-u", str(ROOT / "scripts/exp/v1-rage-ads/audit_metrics.py"),
                       "--task", args.audit_task, "--sample-count", str(args.audit_samples),
                       "--seed", "42", "--gradient-window-batch", str(args.gradient_window_batch),
                       "--vjp-batch-size", str(args.vjp_batch_size), "--output-tag", tag]
        else:
            task = args.eval_task
            folder = "fev" if task == "fev-09" else "time"
            config = ROOT / "configs/eval" / folder / ("v1-rage-ads-" + task + ".yaml")
            destination = output_path("eval", setting_root
                                      / ("job-%02d-%s" % (index, task)))
            command = [sys.executable, "-u", "-m", "evaluate.run", "--checkpoint", str(MODEL_PATH),
                       "--config", str(config), "--output-dir", str(destination)]
        wrapper = [sys.executable, "-u", str(ROOT / "scripts/exp/v1-rage-ads/parallel_profile.py"),
                   "--exec-child", "--ready-file", str(ready), "--gate-file", str(gate),
                   "--", *command]
        process = subprocess.Popen(wrapper, cwd=ROOT, env=environment, start_new_session=True)
        jobs.append(dict(process=process, ready=ready, command=command, index=index))
    logger.info("stage=profile-setting-start phase=%s workers=%d cpu_threads=%d", args.phase,
                workers, args.cpu_threads)
    samples = []
    try:
        ready_deadline = time.monotonic() + 180
        while not all(job["ready"].is_file() for job in jobs):
            failed = [job for job in jobs if job["process"].poll() not in (None, 0)]
            if failed or time.monotonic() >= ready_deadline:
                terminate_all([job["process"] for job in jobs])
                raise RuntimeError("profile workers failed to reach the synchronized start gate")
            time.sleep(0.2)
        gate.write_text("release\n", encoding="ascii")
        started = time.monotonic()
        last_heartbeat = started
        pids = [job["process"].pid for job in jobs]
        previous_cpu_seconds, previous_host_counters = cpu_snapshot(pids)
        previous_sample_time = started
        while any(job["process"].poll() is None for job in jobs):
            gpu = gpu_sample()
            memory = memory_sample(pids)
            now = time.monotonic()
            current_cpu_seconds, current_host_counters = cpu_snapshot(pids)
            cpu_fields = {}
            interval = now - previous_sample_time
            if interval > 0:
                core_equivalents = max(0.0, current_cpu_seconds - previous_cpu_seconds) / interval
                cpu_fields["profile_cpu_core_equivalents"] = core_equivalents
                cpu_fields["profile_cpu_percent_of_host_capacity"] = (
                    100.0 * core_equivalents / max(os.cpu_count() or 1, 1))
                if previous_host_counters and current_host_counters:
                    total_delta = current_host_counters[0] - previous_host_counters[0]
                    idle_delta = current_host_counters[1] - previous_host_counters[1]
                    if total_delta > 0:
                        cpu_fields["host_cpu_utilization_percent"] = max(
                            0.0, 100.0 * (1.0 - idle_delta / total_delta))
            previous_cpu_seconds, previous_host_counters = current_cpu_seconds, current_host_counters
            previous_sample_time = now
            if gpu and memory:
                samples.append(dict(**gpu, **memory, **cpu_fields, sampled_at=time.time()))
            if now - last_heartbeat >= 10:
                done = sum(job["process"].poll() is not None for job in jobs)
                logger.info("stage=profile-heartbeat phase=%s workers=%d completed=%d/%d elapsed=%s",
                            args.phase, workers, done, workers, format_duration(now - started))
                last_heartbeat = now
            time.sleep(1.0)
    except KeyboardInterrupt:
        terminate_all([job["process"] for job in jobs])
        raise
    elapsed = time.monotonic() - started
    return_codes = [job["process"].wait() for job in jobs]
    failed = [dict(index=job["index"], return_code=job["process"].returncode,
                   command=job["command"])
              for job in jobs if job["process"].returncode != 0]
    row = dict(phase=args.phase, workers=workers, jobs=workers, wall_seconds=elapsed,
               jobs_per_second=workers / max(elapsed, 1e-9), cpu_threads_per_worker=args.cpu_threads,
               gpu_utilization_average_percent=(float(np.mean([x["gpu_utilization_percent"] for x in samples]))
                                                if samples else None),
               gpu_utilization_peak_percent=(float(np.max([x["gpu_utilization_percent"] for x in samples]))
                                             if samples else None),
               gpu_memory_average_mib=(float(np.mean([x["gpu_memory_used_mib"] for x in samples]))
                                       if samples else None),
               gpu_memory_peak_mib=(float(np.max([x["gpu_memory_used_mib"] for x in samples]))
                                    if samples else None),
               host_memory_average_mib=(float(np.mean([x["host_memory_used_mib"] for x in samples]))
                                        if samples else None),
               host_memory_peak_mib=(float(np.max([x["host_memory_used_mib"] for x in samples]))
                                     if samples else None),
               profile_children_rss_average_mib=(float(np.mean([x["profile_children_rss_mib"] for x in samples]))
                                                 if samples else None),
               profile_children_rss_peak_mib=(float(np.max([x["profile_children_rss_mib"] for x in samples]))
                                              if samples else None),
               profile_cpu_core_equivalents_average=(float(np.mean([x["profile_cpu_core_equivalents"] for x in samples if "profile_cpu_core_equivalents" in x]))
                                                     if any("profile_cpu_core_equivalents" in x for x in samples) else None),
               profile_cpu_percent_of_host_capacity_average=(float(np.mean([x["profile_cpu_percent_of_host_capacity"] for x in samples if "profile_cpu_percent_of_host_capacity" in x]))
                                                            if any("profile_cpu_percent_of_host_capacity" in x for x in samples) else None),
               host_cpu_utilization_average_percent=(float(np.mean([x["host_cpu_utilization_percent"] for x in samples if "host_cpu_utilization_percent" in x]))
                                                     if any("host_cpu_utilization_percent" in x for x in samples) else None),
               monitor_sample_count=len(samples), return_codes=return_codes,
               errors=failed, status="ok" if not failed else "error")
    logger.info("stage=profile-setting-complete phase=%s workers=%d status=%s jobs_per_second=%.4f gpu_avg=%s gpu_peak=%s gpu_mem_avg_mib=%s gpu_mem_peak_mib=%s host_mem_avg_mib=%s host_mem_peak_mib=%s samples=%d elapsed=%s",
                args.phase, workers, row["status"], row["jobs_per_second"],
                row["gpu_utilization_average_percent"], row["gpu_utilization_peak_percent"],
                row["gpu_memory_average_mib"], row["gpu_memory_peak_mib"],
                row["host_memory_average_mib"], row["host_memory_peak_mib"],
                len(samples), format_duration(elapsed))
    return row


def train_child(args):
    import torch
    import yaml
    sys.path.insert(0, str(ROOT / "scripts/exp/v1-rage-ads"))
    from train_eval import (BATCH_SIZE, CONTEXT, MODEL_PATH as TRAIN_MODEL_PATH,
                            SEED, SourceWindowView, load_source, sampling_probabilities)
    from rage.experiments.full_pool_models import build_model
    from rage.experiments.lora import apply_student_lora
    from rage.experiments.train_eval import TrainingRecipe, fine_tune
    logger = configure_logger("v1_rage_ads_train_profile_%s" % args.worker_id,
                              "v1-rage-ads", "profile-train-%s" % args.worker_id, "profile-train")
    torch.set_num_threads(args.cpu_threads)
    values, legal, value_offsets, legal_offsets = load_source()
    task_dir = ROOT / "outputs/selection/v1-rage-ads/rolling/fev-09/rage"
    _probabilities, positions, sampling_manifest, sampler = sampling_probabilities(
        task_dir, "rage", legal, legal_offsets)
    windows = SourceWindowView(values, value_offsets, legal, positions, sampler)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    engine, _probe, identity = build_model(
        "timerxl-84m", model_path=str(TRAIN_MODEL_PATH), device="cuda",
        context_length=CONTEXT, prediction_length=96)
    lora = apply_student_lora(engine, "timerxl-84m",
                              dict(r=8, alpha=16, dropout=0.0,
                                   target_modules=["q_proj", "v_proj"]))
    recipe = TrainingRecipe(learning_rate=1e-4, epochs=1.0, batch_size=BATCH_SIZE,
                            micro_batch_size=8, weight_decay=0.0, betas=(0.9, 0.999),
                            eps=1e-8, max_grad_norm=1.0, warmup_ratio=0.03,
                            schedule="cosine", min_lr_ratio=0.0, schedule_epochs=4000)
    windows.training_contract.update(sampler_mode=sampling_manifest["mode"],
                                    task="fev-09", arm="rage", profile_trial=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    def on_step(step):
        if step == args.steps or step % 10 == 0:
            logger.info("stage=train-profile-progress worker=%s steps=%d/%d",
                        args.worker_id, step, args.steps)
    result = fine_tune(engine, windows, steps=args.steps, batch_size=BATCH_SIZE,
                       learning_rate=recipe.learning_rate, seed=SEED, recipe=recipe,
                       schedule_total_steps=4000, on_step=on_step)
    elapsed = time.monotonic() - started
    torch.cuda.synchronize()
    output = dict(phase="train", worker_id=args.worker_id, task="fev-09", arm="rage",
                  steps=args.steps, elapsed_seconds=elapsed, steps_per_second=args.steps / elapsed,
                  sampler_mode=sampling_manifest["mode"], sampler_contract=windows.training_contract,
                  lora=lora, model=identity, result=result,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / (1024 ** 2),
                  peak_cuda_reserved_mib=torch.cuda.max_memory_reserved() / (1024 ** 2),
                  warmup_checkpoint="student training starts from pretrained model; gradient benchmark separately uses warmed step-1000 probe")
    write_json(Path(args.output), output)
    Path(args.output + ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=train-profile-complete worker=%s steps=%d elapsed=%s output=%s",
                args.worker_id, args.steps, format_duration(elapsed), args.output)


def exec_child(args):
    ready = Path(args.ready_file)
    gate = Path(args.gate_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ready\n", encoding="ascii")
    started = time.monotonic()
    while not gate.is_file():
        if time.monotonic() - started > 600:
            raise TimeoutError("profile release gate timed out")
        time.sleep(0.1)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("profile child command is empty")
    os.execvpe(command[0], command, os.environ.copy())


def _handle_termination(signum, _frame):
    raise KeyboardInterrupt("received signal %d" % signum)


def main():
    signal.signal(signal.SIGTERM, _handle_termination)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "audit", "eval"), default="train")
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--result-tag", default="")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--audit-task", default="time-01")
    parser.add_argument("--audit-samples", type=int, default=64)
    parser.add_argument("--gradient-window-batch", type=int, default=32)
    parser.add_argument("--vjp-batch-size", type=int, default=16)
    parser.add_argument("--eval-task", default="time-01")
    parser.add_argument("--worker-id", default="0")
    parser.add_argument("--output", default="")
    parser.add_argument("--exec-child", action="store_true")
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--gate-file", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.add_argument("--train-child", action="store_true")
    args = parser.parse_args()
    if args.exec_child:
        exec_child(args)
        return
    if args.train_child:
        if not args.output or args.steps <= 0:
            raise ValueError("train child requires positive steps and an output path")
        train_child(args)
        return
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    if args.result_tag and not all(character.isalnum() or character in "_-" for character in args.result_tag):
        raise ValueError("result-tag may contain only letters, digits, underscore, and hyphen")
    worker_counts = [int(value) for value in args.workers.split(",")]
    if not worker_counts or any(value <= 0 for value in worker_counts):
        raise ValueError("workers must be a comma-separated list of positive integers")
    if args.phase == "audit" and args.audit_task not in TASKS:
        raise ValueError("audit-task must be a registered task")
    if args.phase == "eval" and args.eval_task not in TASKS:
        raise ValueError("eval-task must be a registered task")
    phase_roots = {
        "train": output_path("train", ROOT / "outputs/train/v1-rage-ads/parallelism"),
        "audit": output_path("selection", ROOT / "outputs/selection/v1-rage-ads/parallelism"),
        "eval": output_path("eval", ROOT / "outputs/eval/v1-rage-ads/parallelism"),
    }
    root = phase_roots[args.phase]
    root.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("v1_rage_ads_parallel_profile_%s" % args.phase,
                              "v1-rage-ads", "parallel-profile-%s" % args.phase, "profile")
    logger.info("stage=config phase=%s worker_sweep=%s cpu_threads=%d steps=%d audit_samples=%d eval_task=%s",
                args.phase, args.workers, args.cpu_threads, args.steps, args.audit_samples,
                args.eval_task)
    source_cache_preparation_seconds = 0.0
    if args.phase == "audit":
        from audit_metrics import prepare_source_unit_cache
        _cache_root, source_cache_preparation_seconds, cache_created = prepare_source_unit_cache(logger)
        logger.info("stage=source-unit-cache-ready created=%s elapsed=%s path=%s",
                    cache_created, format_duration(source_cache_preparation_seconds), SOURCE_UNIT_CACHE_ROOT)
    started = time.monotonic()
    rows = []
    for workers in worker_counts:
        row = profile_setting(args, workers, root, logger)
        rows.append(row)
        if row["status"] != "ok":
            logger.warning("stage=profile-stop reason=worker_failure workers=%d", workers)
            break
    successful = [row for row in rows if row["status"] == "ok"]
    best = max(successful, key=lambda row: row["jobs_per_second"]) if successful else None
    result = dict(experiment="v1-rage-ads", phase=args.phase,
                  trial_protocol=(("%d actual-recipe training updates per worker"
                                   % args.steps)
                                  if args.phase == "train" else
                                  "sampled task-level audit" if args.phase == "audit" else
                                  "full rolling evaluation on pretrained TimerXL checkpoint"),
                  model="thuml/timer-base-84m", checkpoint=(str(MODEL_PATH) if args.phase == "eval" else None),
                  cpu_threads_per_worker=args.cpu_threads, worker_counts=worker_counts,
                  audit_task=args.audit_task if args.phase == "audit" else None,
                  audit_sample_count=args.audit_samples if args.phase == "audit" else None,
                  gradient_window_batch_size=args.gradient_window_batch if args.phase == "audit" else None,
                  gradient_vjp_batch_size=args.vjp_batch_size if args.phase == "audit" else None,
                  eval_task=args.eval_task if args.phase == "eval" else None,
                  train_steps=args.steps if args.phase == "train" else None,
                  source_unit_cache_preparation_seconds=(source_cache_preparation_seconds
                                                         if args.phase == "audit" else None),
                  trials=rows, selected_workers=(None if best is None else best["workers"]),
                  selected_jobs_per_second=(None if best is None else best["jobs_per_second"]),
                  total_elapsed_seconds=time.monotonic() - started)
    suffix = ("-" + args.result_tag) if args.result_tag else ""
    target = root / (args.phase + "-parallel-profile" + suffix + ".json")
    write_json(target, result)
    (root / (args.phase + "-parallel-profile" + suffix + ".complete")).write_text(
        "complete\n", encoding="ascii")
    logger.info("stage=complete phase=%s selected_workers=%s jobs_per_second=%s elapsed=%s output=%s",
                args.phase, result["selected_workers"], result["selected_jobs_per_second"],
                format_duration(time.monotonic() - started), target)


if __name__ == "__main__":
    main()
