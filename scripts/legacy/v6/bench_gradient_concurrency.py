"""Measure one model's steady gradient throughput behind a file barrier.

Launch one copy per model (including from different Python environments) with
the same barrier directory.  Model loading and warmup finish before any timed
region begins, so the measurements reflect actual concurrent GPU execution.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rage.execution import SelectionExecution
from rage.experiments.full_pool_models import build_model


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--duration", type=float, default=0,
                        help="If positive, repeat one batch until this many seconds elapse")
    parser.add_argument("--barrier-dir", type=Path, required=True)
    parser.add_argument("--participants", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    _, probe, identity = build_model(args.model, model_path=args.model_path)
    probe.execution = SelectionExecution(
        gradient_batch=args.batch, gradient_workers=args.workers, generation="eager"
    )
    values = np.stack([
        np.sin(np.arange(576) / (12 + i) + i * .2) + (2 if i % 2 else 0)
        for i in range(args.windows)
    ]).astype(np.float32)

    # Materialize persistent worker pools and CUDA kernels before the barrier.
    probe.batch_gradient_sketch(values[:args.batch])
    torch.cuda.synchronize()
    args.barrier_dir.mkdir(parents=True, exist_ok=True)
    (args.barrier_dir / (args.model + ".ready")).touch()
    deadline = time.monotonic() + 300
    while len(list(args.barrier_dir.glob("*.ready"))) < args.participants:
        if time.monotonic() >= deadline:
            raise TimeoutError("concurrent benchmark barrier timed out")
        time.sleep(.02)
    # Give every participant time to observe the completed barrier.
    time.sleep(.25)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    measured_windows = len(values)
    if args.duration > 0:
        measured_windows = 0
        deadline = started + args.duration
        while time.perf_counter() < deadline:
            losses, gradients = probe.batch_gradient_sketch(values[:args.batch])
            measured_windows += args.batch
    else:
        losses, gradients = probe.batch_gradient_sketch(values)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if not np.isfinite(losses).all() or not np.isfinite(gradients).all():
        raise ValueError("non-finite benchmark result")
    result = dict(
        model=identity,
        batch=args.batch,
        workers=args.workers,
        windows=measured_windows,
        seconds=seconds,
        windows_per_second=measured_windows / seconds,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        execution=getattr(probe, "last_gradient_execution", {}),
    )
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    release = getattr(probe, "release_gradient_workers", None)
    if release:
        release()


if __name__ == "__main__":
    main()
