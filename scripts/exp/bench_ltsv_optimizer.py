"""Bounded first-step Adam reuse benchmark on production BF16 parameter shapes.

No data, gradients or results from the formal experiment are modified. Fixed
synthetic gradients isolate optimizer allocation/reset overhead from forecasting.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import torch

from rags.experiments.full_pool_models import build_model
from rags.experiments.students import atomic_json


def reset_first_step(optimizer):
    for state in optimizer.state.values():
        state['step'].zero_()
        state['exp_avg'].zero_()
        state['exp_avg_sq'].zero_()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--student', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output == args.run.resolve() or args.run.resolve() in output.parents:
        raise ValueError('choose a new report outside the formal run')
    config = json.loads((args.run / 'protocol.json').read_text())
    torch.set_num_threads(4)
    engine, _, identity = build_model(args.student, model_path=config['model_paths'][args.student])
    if identity != json.loads((args.run / 'students' / args.student / 'model.json').read_text()):
        raise ValueError('model identity differs from formal run')
    parameters = list(engine.model.parameters())
    base = [p.detach().clone() for p in parameters]
    gradients = [torch.full_like(p, .001) if p.requires_grad else None for p in parameters]
    def new_optimizer():
        return torch.optim.Adam(parameters, lr=1e-5, foreach=False,
            betas=(.9, .999), eps=1e-8, weight_decay=.1)
    persistent = new_optimizer()
    times = {'fresh': [], 'reset': []}
    all_equal = True
    torch.cuda.reset_peak_memory_stats()
    # Alternating order and an excluded warmup pair limit allocator/startup bias.
    for iteration in range(7):
        expected = None
        for mode in (('fresh', 'reset') if iteration % 2 == 0 else ('reset', 'fresh')):
            with torch.no_grad():
                for p, b, g in zip(parameters, base, gradients):
                    p.copy_(b)
                    p.grad = g
            torch.cuda.synchronize()
            started = time.perf_counter()
            if mode == 'fresh':
                optimizer = new_optimizer()
            else:
                optimizer = persistent
                reset_first_step(optimizer)
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            if iteration:
                times[mode].append(seconds)
            current = [p.detach().clone() for p in parameters]
            if expected is None:
                expected = current
            else:
                all_equal &= all(torch.equal(a, b) for a, b in zip(current, expected))
            del current, optimizer
        del expected
    result = dict(student=args.student, recorded_at=datetime.now(timezone.utc).isoformat(),
        passed=all_equal, exact_parameter_update_agreement=all_equal,
        seconds=times, median_seconds={key: statistics.median(value) for key, value in times.items()},
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        caveat='synthetic fixed gradients, optimizer-only, one isolated model under live MPS contention',
        formal_read_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result), flush=True)
    if not all_equal:
        raise SystemExit('first-step Adam equality gate failed')


if __name__ == '__main__':
    main()
