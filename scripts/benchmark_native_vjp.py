"""Synthetic TimerXL V6 throughput probe; never opens formal gradient caches."""
import argparse
from dataclasses import replace
import json
import time

import numpy as np
import torch

from rage.execution import SelectionExecution, selection_execution
from rage.experiments.full_pool_models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--vjp", type=int, nargs="+", default=[4, 8, 16])
    args = parser.parse_args()
    if args.samples < 2 or args.batch < 1 or any(v < 1 for v in args.vjp):
        parser.error("samples >= 2, batch and VJP sizes >= 1 required")
    _, probe, _ = build_model("timerxl-84m", model_path=args.model_path,
                              device="cuda", context_length=480, prediction_length=96)
    probe.execution = SelectionExecution(gradient_batch=args.batch, cpu_threads=2)
    windows = np.random.default_rng(2718).normal(size=(args.samples, 576)).astype(np.float32)
    with selection_execution(probe):
        # Synthetic correctness reference, not an experiment result or cache.
        ref_loss, reference = probe.batch_gradient_sketch(windows, batch_size=1)
        for vjp in args.vjp:
            probe.execution = replace(probe.execution, gradient_vjp_batch=vjp)
            probe.batch_gradient_sketch(windows)  # warm-up model and projection kernels
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            losses, values = probe.batch_gradient_sketch(windows)
            seconds = time.perf_counter() - start  # NumPy return synchronizes
            norms = np.linalg.norm(reference, axis=1) * np.linalg.norm(values, axis=1)
            cos = np.divide((reference * values).sum(1), norms,
                            out=np.ones(len(values)), where=norms != 0)
            print(json.dumps(dict(vjp=vjp, samples=args.samples, seconds=seconds,
                windows_per_second=args.samples / seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                max_loss_abs=float(np.max(np.abs(losses - ref_loss))),
                relative_l2=float(np.linalg.norm(values - reference) / max(np.linalg.norm(reference), 1e-30)),
                min_cosine=float(cos.min()), execution=probe.last_gradient_execution,
                scope="synthetic fixture only; not a formal result")), flush=True)


if __name__ == "__main__":
    main()
