"""Measure per-window gradient batching against independent single-window gradients."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from rags.experiments.full_pool_models import build_model
from rags.execution import SelectionExecution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, probe, identity = build_model(args.model, model_path=args.model_path)
    values = np.stack([np.sin(np.arange(576) / (12 + i) + i * .2) + (2 if i % 2 else 0)
                       for i in range(args.windows)]).astype(np.float32)
    records = []
    reference = None
    for size in args.batches:
        probe.execution = SelectionExecution(gradient_batch=size, gradient_workers=args.workers, generation="eager")
        probe.batch_gradient_sketch(values[:min(size, len(values))])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses, gradients = probe.batch_gradient_sketch(values)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if reference is None:
            if size != 1:
                raise ValueError("first batch must be 1 to establish independent-window reference")
            reference = losses, gradients
        expected_loss, expected = reference
        dots = (gradients.astype(np.float64) * expected).sum(axis=1)
        cosine = dots / (np.linalg.norm(gradients.astype(np.float64), axis=1) *
                         np.linalg.norm(expected.astype(np.float64), axis=1))
        relative = np.linalg.norm(gradients - expected, axis=1) / np.linalg.norm(expected, axis=1)
        row = dict(batch=size, seconds=seconds, windows_per_second=len(values) / seconds,
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                   minimum_cosine=float(cosine.min()), maximum_relative_error=float(relative.max()),
                   maximum_loss_difference=float(np.max(np.abs(losses - expected_loss))),
                   execution=getattr(probe, "last_gradient_execution", {}))
        records.append(row)
        print(json.dumps(row), flush=True)
        if not np.isfinite(gradients).all() or not np.isfinite(losses).all():
            raise ValueError("nonfinite batch result")
        np.testing.assert_allclose(losses, expected_loss, rtol=1e-4, atol=1e-6)
        if cosine.min() < .9999 or relative.max() > .01:
            raise ValueError("batched gradients exceed the BF16 numerical agreement gate")
    if args.output:
        from rags.experiments.students import atomic_json
        atomic_json(args.output, dict(model=identity, windows=len(values), results=records))
    release = getattr(probe, "release_gradient_workers", None)
    if release:
        release()


if __name__ == "__main__":
    main()
