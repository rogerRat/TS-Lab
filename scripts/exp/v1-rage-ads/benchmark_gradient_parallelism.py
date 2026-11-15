#!/usr/bin/env python3
"""Profile warmed TimerXL gradient throughput, batch sizing, and GPU concurrency."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path

ROOT = PROJECT_ROOT
SOURCE_DATA = ROOT / 'data/src/shard_dataset/src-96'
WARM_ROOT = ROOT / 'outputs/train/v1-rage-ads/parallelism'


def source_windows(count: int, seed: int) -> tuple[np.ndarray, float]:
    started = time.monotonic()
    config = yaml.safe_load((ROOT / 'configs/data/src-96.yaml').read_text(encoding='ascii'))
    lengths = np.asarray([int(row['length']) for row in config['trajectories']], dtype=np.int64)
    legal = lengths - 575
    offsets = np.asarray(np.load(SOURCE_DATA / 'offsets.npy', mmap_mode='r'), dtype=np.int64)
    values = np.load(SOURCE_DATA / 'values.npy', mmap_mode='r')
    legal_offsets = np.cumsum(np.r_[0, legal])
    if len(legal) != 96 or int(legal.sum()) != 2666312:
        raise ValueError('src-96 window population mismatch')
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, int(legal_offsets[-1]), size=count)
    owners = np.searchsorted(legal_offsets[1:], positions, side='right')
    local = positions - legal_offsets[owners]
    begins = offsets[owners] + local
    windows = np.asarray(values[begins[:, None] + np.arange(576)[None, :]], dtype=np.float32).copy()
    if not np.isfinite(windows).all():
        raise ValueError('sampled gradient probe windows are nonfinite')
    return windows, time.monotonic() - started


def worker(windows, batch_size, vjp_batch, cpu_threads, gate, event_queue):
    try:
        os.environ['OMP_NUM_THREADS'] = str(cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_threads)
        sys.path.insert(0, str(ROOT / 'scripts/exp/v1-crb-redo'))
        import torch
        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        from metrics import build_gradient_probe
        model_engine, probe, identity, model_fields = build_gradient_probe()
        probe.vjp_batch_size = vjp_batch
        warm_count = min(len(windows), max(1, batch_size))
        probe.batch_gradient_sketch(windows[:warm_count], batch_size=warm_count)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        event_queue.put(('ready', os.getpid(), time.monotonic(),
                         int(torch.cuda.memory_allocated())))
        if not gate.wait(600):
            raise TimeoutError('benchmark release gate timed out')
        started = time.monotonic()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        for begin in range(0, len(windows), batch_size):
            batch = windows[begin:begin + batch_size]
            probe.batch_gradient_sketch(batch, batch_size=batch_size)
        cuda_end.record()
        torch.cuda.synchronize()
        ended = time.monotonic()
        result = dict(pid=os.getpid(), windows=len(windows), batch_size=batch_size,
                      vjp_batch=vjp_batch, compute_seconds=ended - started,
                      cuda_elapsed_seconds=cuda_start.elapsed_time(cuda_end) / 1000.0,
                      cuda_max_allocated_mib=torch.cuda.max_memory_allocated() / (1024 ** 2),
                      cuda_max_reserved_mib=torch.cuda.max_memory_reserved() / (1024 ** 2),
                      process_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      identity_model=identity['model_id'], warmup_steps=model_fields['warmup_steps'])
        event_queue.put(('done', result))
        del probe, model_engine
        torch.cuda.empty_cache()
    except BaseException:
        event_queue.put(('error', os.getpid(), traceback.format_exc()))


def gpu_sample():
    try:
        raw = subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
            '--format=csv,noheader,nounits'], text=True, timeout=3).strip().splitlines()[0]
        util, used, total, power = [float(value.strip()) for value in raw.split(',')]
        return dict(utilization_percent=util, memory_used_mib=used,
                    memory_total_mib=total, power_watts=power)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def cpu_snapshot(pids):
    process_seconds = 0.0
    ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
    for pid in pids:
        try:
            raw = open("/proc/%d/stat" % pid, encoding="ascii").read()
            fields = raw[raw.rfind(")") + 2:].split()
            process_seconds += (int(fields[11]) + int(fields[12])) / ticks_per_second
        except (OSError, IndexError, ValueError):
            pass
    try:
        fields = open("/proc/stat", encoding="ascii").read().splitlines()[0].split()[1:]
        counters = [int(value) for value in fields]
        host_counters = (sum(counters), counters[3] + counters[4])
    except (OSError, IndexError, ValueError):
        host_counters = None
    return process_seconds, host_counters


def memory_sample():
    values = {}
    try:
        with open('/proc/meminfo', encoding='ascii') as stream:
            for line in stream:
                key, value = line.split(':', 1)
                if key in ('MemTotal', 'MemAvailable'):
                    values[key] = int(value.split()[0])
        return dict(host_memory_used_mib=(values['MemTotal'] - values['MemAvailable']) / 1024.0,
                    host_memory_total_mib=values['MemTotal'] / 1024.0)
    except (OSError, KeyError, ValueError):
        return None


def run_setting(workers, batch_size, vjp_batch, windows_per_worker,
                cpu_threads, logger):
    context = mp.get_context('spawn')
    gate = context.Event()
    event_queue = context.Queue()
    rng = np.random.default_rng(1701 + workers * 100 + batch_size * 3 + vjp_batch)
    windows, prepare_seconds = source_windows(workers * windows_per_worker, int(rng.integers(0, 2**31)))
    chunks = [windows[i * windows_per_worker:(i + 1) * windows_per_worker]
              for i in range(workers)]
    processes = [context.Process(target=worker,
                                 args=(chunk, batch_size, vjp_batch, cpu_threads,
                                       gate, event_queue))
                 for chunk in chunks]
    ready = set()
    results = []
    errors = []
    samples = []
    released = False
    sample_started = None
    previous_cpu_seconds = None
    previous_host_counters = None
    previous_sample_time = None
    started = time.monotonic()
    logger.info('stage=setting-start workers=%d batch=%d vjp_batch=%d windows_per_worker=%d cpu_threads=%d',
                workers, batch_size, vjp_batch, windows_per_worker, cpu_threads)
    try:
        for process in processes:
            process.start()
        while len(results) + len(errors) < workers:
            try:
                message = event_queue.get(timeout=0.5)
            except queue.Empty:
                message = None
            if message:
                if message[0] == 'ready':
                    ready.add(message[1])
                    if len(ready) == workers and not released:
                        gate.set()
                        released = True
                        sample_started = time.monotonic()
                        previous_cpu_seconds, previous_host_counters = cpu_snapshot(
                            [process.pid for process in processes])
                        previous_sample_time = sample_started
                        logger.info('stage=workers-ready count=%d released=true', len(ready))
                elif message[0] == 'done':
                    results.append(message[1])
                elif message[0] == 'error':
                    errors.append(dict(pid=message[1], traceback=message[2]))
                    gate.set()
            if released:
                sample = gpu_sample()
                memory = memory_sample()
                now = time.monotonic()
                current_cpu_seconds, current_host_counters = cpu_snapshot(
                    [process.pid for process in processes])
                cpu_fields = {}
                interval = now - previous_sample_time if previous_sample_time is not None else 0.0
                if interval > 0 and previous_cpu_seconds is not None:
                    core_equivalents = max(0.0, current_cpu_seconds - previous_cpu_seconds) / interval
                    cpu_fields['process_cpu_core_equivalents'] = core_equivalents
                    cpu_fields['process_cpu_percent_of_host_capacity'] = (
                        100.0 * core_equivalents / max(os.cpu_count() or 1, 1))
                    if previous_host_counters and current_host_counters:
                        total_delta = current_host_counters[0] - previous_host_counters[0]
                        idle_delta = current_host_counters[1] - previous_host_counters[1]
                        if total_delta > 0:
                            cpu_fields['host_cpu_utilization_percent'] = max(
                                0.0, 100.0 * (1.0 - idle_delta / total_delta))
                previous_cpu_seconds, previous_host_counters = current_cpu_seconds, current_host_counters
                previous_sample_time = now
                if sample and memory:
                    samples.append(dict(**sample, **memory, **cpu_fields, sampled_at=time.time()))
            dead_without_message = [p for p in processes if p.exitcode not in (None, 0)
                                    and p.pid not in ready
                                    and not any(row.get('pid') == p.pid for row in errors)]
            if dead_without_message:
                errors.extend(dict(pid=p.pid, traceback='worker exited with code %s before reporting' % p.exitcode)
                              for p in dead_without_message)
                gate.set()
            if errors:
                break
        for process in processes:
            process.join(timeout=30)
        alive = [process for process in processes if process.is_alive()]
        for process in alive:
            os.kill(process.pid, signal.SIGINT)
        for process in alive:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    except KeyboardInterrupt:
        gate.set()
        for process in processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
        for process in processes:
            process.join(timeout=20)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        raise
    finally:
        gate.set()
        event_queue.close()
    elapsed = time.monotonic() - started
    compute_makespan = max((row['compute_seconds'] for row in results), default=0.0)
    setting_after_release = (None if sample_started is None else
                             time.monotonic() - sample_started)
    result = dict(workers=workers, batch_size=batch_size, vjp_batch=vjp_batch,
                  windows_per_worker=windows_per_worker, windows_per_second=0.0,
                  preparation_seconds=prepare_seconds, setup_and_warmup_wall_seconds=elapsed,
                  compute_makespan_seconds=compute_makespan,
                  setting_after_release_seconds=setting_after_release,
                  worker_compute_seconds=[row['compute_seconds'] for row in results],
                  worker_cuda_seconds=[row['cuda_elapsed_seconds'] for row in results],
                  worker_peak_cuda_allocated_mib=[row['cuda_max_allocated_mib'] for row in results],
                  worker_peak_cuda_reserved_mib=[row['cuda_max_reserved_mib'] for row in results],
                  worker_peak_rss_mib=[row['process_max_rss_mib'] for row in results],
                  process_cpu_core_equivalents_average=(float(np.mean([x['process_cpu_core_equivalents'] for x in samples if 'process_cpu_core_equivalents' in x]))
                                                        if any('process_cpu_core_equivalents' in x for x in samples) else None),
                  process_cpu_percent_of_host_capacity_average=(float(np.mean([x['process_cpu_percent_of_host_capacity'] for x in samples if 'process_cpu_percent_of_host_capacity' in x]))
                                                               if any('process_cpu_percent_of_host_capacity' in x for x in samples) else None),
                  host_cpu_utilization_average_percent=(float(np.mean([x['host_cpu_utilization_percent'] for x in samples if 'host_cpu_utilization_percent' in x]))
                                                        if any('host_cpu_utilization_percent' in x for x in samples) else None),
                  gpu_utilization_average_percent=(float(np.mean([x['utilization_percent'] for x in samples]))
                                                   if samples else None),
                  gpu_utilization_max_percent=(float(np.max([x['utilization_percent'] for x in samples]))
                                               if samples else None),
                  gpu_memory_average_mib=(float(np.mean([x['memory_used_mib'] for x in samples]))
                                          if samples else None),
                  gpu_memory_peak_mib=(float(np.max([x['memory_used_mib'] for x in samples]))
                                       if samples else None),
                  host_memory_average_mib=(float(np.mean([x['host_memory_used_mib'] for x in samples]))
                                           if samples else None),
                  host_memory_peak_mib=(float(np.max([x['host_memory_used_mib'] for x in samples]))
                                        if samples else None),
                  cpu_count=os.cpu_count(), host_load_1=float(os.getloadavg()[0]),
                  monitor_sample_count=len(samples), worker_results=results,
                  errors=errors, status='ok' if len(results) == workers and not errors else 'error')
    if result['status'] == 'ok':
        result['windows_per_second'] = (workers * windows_per_worker
                                        / max(compute_makespan, 1e-9))
    logger.info('stage=setting-complete status=%s workers=%d batch=%d vjp_batch=%d throughput=%.2f windows_per_second gpu_avg=%s gpu_peak=%s gpu_mem_avg_mib=%s gpu_mem_peak_mib=%s host_mem_avg_mib=%s host_mem_peak_mib=%s samples=%d elapsed=%s',
                result['status'], workers, batch_size, vjp_batch, result['windows_per_second'],
                result['gpu_utilization_average_percent'], result['gpu_utilization_max_percent'],
                result['gpu_memory_average_mib'], result['gpu_memory_peak_mib'],
                result['host_memory_average_mib'], result['host_memory_peak_mib'], len(samples),
                format_duration(elapsed))
    if errors:
        logger.error('stage=setting-errors details=%s', json.dumps(errors, sort_keys=True))
    return result


def main():
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--windows-per-worker', type=int, default=128)
    parser.add_argument('--workers', default='1,2,4,8')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-tag', default='')
    args = parser.parse_args()
    if args.windows_per_worker <= 0 or args.cpu_threads <= 0:
        raise ValueError('trial sizes and CPU threads must be positive')
    if args.output_tag and not all(character.isalnum() or character in '_-' for character in args.output_tag):
        raise ValueError('output-tag may contain only letters, digits, underscore, and hyphen')
    worker_counts = [int(value) for value in args.workers.split(',')]
    if any(value <= 0 for value in worker_counts):
        raise ValueError('worker counts must be positive')
    logger = configure_logger('v1_rage_ads_gradient_benchmark', 'v1-rage-ads',
                              'gradient-parallelism-benchmark', 'gradient-benchmark')
    started = time.monotonic()
    windows, prepare_seconds = source_windows(args.windows_per_worker, args.seed)
    batch_specs = [(4, 4), (8, 4), (16, 4), (32, 4), (32, 8), (32, 16), (32, 32)]
    rows = []
    logger.info('stage=config model=TimerXL-84M warmup_steps=1000 windows=%d cpu_threads=%d worker_sweep=%s batch_specs=%s',
                len(windows), args.cpu_threads, args.workers, batch_specs)
    for batch_size, vjp_batch in batch_specs:
        row = run_setting(1, batch_size, vjp_batch, args.windows_per_worker,
                          args.cpu_threads, logger)
        rows.append(row)
    successful = [row for row in rows if row['status'] == 'ok']
    best = max(successful, key=lambda row: row['windows_per_second']) if successful else None
    if best is None:
        raise RuntimeError('all single-worker gradient batch trials failed')
    chosen_batch = (best['batch_size'], best['vjp_batch'])
    for workers in worker_counts:
        if workers == 1:
            continue
        row = run_setting(workers, chosen_batch[0], chosen_batch[1],
                          args.windows_per_worker, args.cpu_threads, logger)
        rows.append(row)
        if row['status'] != 'ok':
            logger.warning('stage=concurrency-limit workers=%d failed; stop increasing concurrency', workers)
            break
    output = output_path('selection', ROOT / 'outputs/selection/v1-rage-ads/parallelism')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(experiment='v1-rage-ads', phase='warmed-gradient-preparation',
                  output_tag=args.output_tag,
                  model='thuml/timer-base-84m', warmup_steps=1000,
                  gradient_protocol='independent-native-loss-bf16-v2',
                  context_length=480, prediction_length=96,
                  projection_dimension=4096, projection_seed=42,
                  cpu_threads=args.cpu_threads, cpu_count=os.cpu_count(),
                  windows_per_worker=args.windows_per_worker,
                  source_window_preparation_seconds=prepare_seconds,
                  best_single_worker=dict(batch_size=chosen_batch[0], vjp_batch=chosen_batch[1],
                                          windows_per_second=best['windows_per_second']),
                  trials=rows, total_elapsed_seconds=time.monotonic() - started)
    suffix = ('-' + args.output_tag) if args.output_tag else ''
    target = output / ('gradient-parallelism-benchmark' + suffix + '.json')
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='ascii')
    os.replace(temporary, target)
    (output / ('gradient-parallelism-benchmark' + suffix + '.complete')).write_text(
        'complete\n', encoding='ascii')
    logger.info('stage=complete best_batch=%d best_vjp=%d throughput=%.2f total_elapsed=%s output=%s',
                chosen_batch[0], chosen_batch[1], best['windows_per_second'],
                format_duration(time.monotonic() - started), target)


if __name__ == '__main__':
    main()
