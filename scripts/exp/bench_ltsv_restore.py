"""Bounded real-weight LTSV preflight, without formal cache or result writes.

Compare CPU and GPU restoration of the identical BF16 pretrained parameters.
Use only source windows and safe reference histories; never read exam windows.
This script is diagnostic-only. Production enablement requires matched timing
calibration and separately reports the unoptimized-baseline runtime estimate.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from baselines.common import reference_windows
from baselines.ltsv.runner import _ltsv_reference_mse, _ltsv_training_loss
from rage.execution import SelectionExecution, prepare_contexts, selection_execution
from rage.experiments.development_inputs import load_fragments
from rage.experiments.full_pool_models import build_model
from rage.experiments.students import atomic_json


def _restore_snapshot_on_device(base_flat, device):
    """Diagnostic candidate only; production LTSV keeps its CPU snapshot."""
    device = torch.device(device)
    if device.type != 'cuda':
        return base_flat
    free, _ = torch.cuda.mem_get_info(device)
    if base_flat.numel() * base_flat.element_size() > free * .05:
        return base_flat
    try:
        return base_flat.to(device=device)
    except torch.cuda.OutOfMemoryError:
        return base_flat


def timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--student', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--blocks', type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output == args.run.resolve() or args.run.resolve() in output.parents:
        raise ValueError('choose a new report outside the formal run')
    if not 1 <= args.blocks <= 4:
        raise ValueError('preflight is limited to 1..4 source blocks')
    config = json.loads((args.run / 'protocol.json').read_text())
    inputs = Path(config['inputs'])
    pool = load_fragments(inputs / 'pool.npz')
    dataset = config['datasets'][0]
    refs = load_fragments(inputs / f'{dataset}__reference.npz')
    c, h = config['context_length'], config['prediction_length']
    width, stride = c + h, (c + h) // 2
    total_blocks = sum((f.length - width) // stride + 1 +
                       int((f.length - width) % stride != 0)
                       for f in pool if f.length >= width)
    chosen = [pool[i] for i in np.linspace(0, len(pool) - 1, args.blocks, dtype=int)]
    block_ids, windows = [], []
    for f in chosen:
        start = ((f.length - width) // (2 * stride)) * stride
        block_ids.append(dict(fragment_id=int(f.fragment_id), physical_start_1based=start + 1))
        windows.append(np.asarray(f.data[start:start + width], dtype=np.float32)[None])
    val, val_ids = reference_windows(refs, context_length=c, prediction_length=h,
                                     stride=16, max_windows=64, seed=0)
    torch.set_num_threads(4)
    _, probe, identity = build_model(args.student, model_path=config['model_paths'][args.student])
    if identity != json.loads((args.run / 'students' / args.student / 'model.json').read_text()):
        raise ValueError('model identity differs from formal run')
    execution = dict(config['selection_execution'])
    execution.update(config.get('student_selection_execution', {}).get(args.student, {}))
    execution['gradient_workers'] = 1  # This diagnostic never starts gradient children.
    probe.execution = SelectionExecution(**execution)
    model = probe.model
    if not callable(getattr(probe, 'forecast', None)):
        val = prepare_contexts(probe, val)
    parameters = [p for p in model.parameters() if p.requires_grad]
    base_cpu = torch.cat([p.detach().cpu().reshape(-1) for p in parameters])
    nbytes = base_cpu.numel() * base_cpu.element_size()
    free, _ = torch.cuda.mem_get_info()
    if nbytes > free * .05:
        raise RuntimeError('insufficient spare GPU memory for bounded restore comparison')
    base_gpu = _restore_snapshot_on_device(base_cpu, probe.device)
    if base_gpu.device.type != 'cuda':
        raise RuntimeError('diagnostic restore candidate fell back to CPU')

    def restore(base):
        offset = 0
        with torch.no_grad():
            for p in parameters:
                p.copy_(base[offset:offset + p.numel()].view_as(p))
                offset += p.numel()

    rows, copy_times = [], {}
    torch.cuda.reset_peak_memory_stats()
    with selection_execution(probe):
        model.eval()
        base_loss, base_seconds = timed(lambda: _ltsv_reference_mse(probe, val))
        for label, base in [('cpu', base_cpu), ('cuda', base_gpu)]:
            copy_times[label] = [timed(lambda: restore(base))[1] for _ in range(5)]
        for i, window in enumerate(windows):
            # Alternate order to limit drift from concurrent formal workloads.
            variants = [('cpu', base_cpu), ('cuda', base_gpu)]
            if i % 2:
                variants.reverse()
            for label, base in variants:
                restore(base)
                torch.manual_seed(20260915 + i)
                torch.cuda.manual_seed_all(20260915 + i)
                model.train()
                def update():
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False,
                        betas=(.9, .999), eps=1e-8, weight_decay=.1)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _ltsv_training_loss(probe, window)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Production retains Adam moments during reference scoring.
                    return float(loss.detach().cpu()), optimizer
                try:
                    (loss, optimizer), update_seconds = timed(update)
                    model.eval()
                    context_loss, forecast_seconds = timed(lambda: _ltsv_reference_mse(probe, val))
                finally:
                    _, restore_seconds = timed(lambda: restore(base))
                    model.eval()
                del optimizer
                row = dict(block=i, restore_device=label, loss=loss,
                           context_mse=context_loss, value=base_loss - context_loss,
                           update_seconds=update_seconds, forecast_seconds=forecast_seconds,
                           restore_seconds=restore_seconds)
                rows.append(row)
                print(json.dumps(row), flush=True)
    restored = torch.cat([p.detach().cpu().reshape(-1) for p in parameters])
    exact_restore = torch.equal(restored, base_cpu)
    agreement = all(all(np.isfinite(a[key]) and a[key] == b[key]
                        for key in ('loss', 'context_mse', 'value'))
                    for i in range(args.blocks)
                    for a, b in [[row for row in rows if row['block'] == i]])
    report = dict(student=args.student, model=identity,
        recorded_at=datetime.now(timezone.utc).isoformat(), formal_read_only=True,
        dataset=dataset, block_ids=block_ids, reference_ids=val_ids,
        total_production_blocks_per_task=total_blocks, reference_windows=len(val_ids),
        snapshot_gib=nbytes / 2**30, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        base_context_mse=base_loss, base_forecast_seconds=base_seconds,
        copy_seconds=copy_times, rows=rows, exact_parameter_restore=exact_restore,
        exact_loss_and_mse_agreement=agreement, passed=exact_restore and agreement,
        caveat='Few blocks under live MPS contention; not an end-to-end throughput estimate')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    if not report['passed']:
        raise SystemExit('restore agreement gate failed')


if __name__ == '__main__':
    main()
