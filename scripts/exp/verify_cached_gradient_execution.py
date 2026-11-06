"""Recompute a bounded source-window sample against published BF16 shards.

Reads the formal cache without opening its writer/lock. Reports go to a separate
directory. The caller chooses the last shard published before an execution change.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from assets.gradient.dataset import decode_bf16
from rage.execution import SelectionExecution
from rage.experiments.development_inputs import load_fragments
from rage.experiments.full_pool_models import build_model
from rage.experiments.students import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--last-shard', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--windows', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.windows <= 32 or args.last_shard < args.windows - 1:
        raise ValueError('choose 1..32 windows and enough existing shards')
    output = args.output.resolve()
    if output == args.run.resolve() or args.run.resolve() in output.parents:
        raise ValueError('audit output must be outside the formal run')
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((args.cache / 'cache_manifest.json').read_text())
    if manifest['format'] != 'window-gradients-bf16-v1':
        raise ValueError('unexpected gradient format')
    config = json.loads((args.run / 'protocol.json').read_text())
    pool = {f.fragment_id: f for f in load_fragments(Path(config['inputs']) / 'pool.npz')}
    identities, windows, expected, expected_losses = [], [], [], []
    for number in np.linspace(0, args.last_shard, args.windows, dtype=int):
        shard_path = args.cache / f'windows-{number:08d}.npz'
        with np.load(shard_path, allow_pickle=False) as shard:
            if str(shard['dataset']) != 'source':
                raise ValueError('audit sample must use source data')
            ids = shard['window_ids']
            row = len(ids) // 2
            fid, start = map(int, ids[row])
            windows.append(pool[fid].extract_window(start, manifest['context_length'],
                                                   manifest['prediction_length']))
            expected.append(decode_bf16(shard['gradients_bf16'][row]))
            expected_losses.append(float(shard['losses'][row]))
            identities.append(dict(shard=shard_path.name, row=row, fragment_id=fid,
                                   physical_start_1based=start))
    torch.set_num_threads(2)
    student = manifest['model']['student']
    _, probe, identity = build_model(student, model_path=config['model_paths'][student])
    if identity != manifest['model']:
        raise ValueError('loaded model fields differ from published cache')
    # Use the existing production geometry; keep this audit to one GPU process.
    execution = dict(config.get('selection_execution', {}))
    execution.update(config.get('student_selection_execution', {}).get(student, {}))
    execution.update(json.loads((args.run / 'execution.json').read_text()).get(
        'student_selection_execution', {}).get(student, {}))
    execution['gradient_workers'] = 1
    probe.execution = SelectionExecution(**execution)
    values = np.asarray(windows, dtype=np.float32)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses, actual = probe.batch_gradient_sketch(
        values, batch_size=1 if student in ('chronos-base', 'timemoe-200m') else None)
    seconds = time.perf_counter() - started
    expected, actual = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    norms = np.linalg.norm(expected, axis=1)
    actual_norms = np.linalg.norm(actual, axis=1)
    differences = np.linalg.norm(actual - expected, axis=1)
    relative = differences / np.maximum(norms, 1e-30)
    cosine = np.sum(actual * expected, axis=1) / np.maximum(
        actual_norms * norms, 1e-30)
    both_zero = (norms == 0) & (actual_norms == 0)
    cosine[both_zero] = 1.  # Equal zero vectors agree; their angle is undefined.
    finite = bool(np.isfinite(actual).all() and np.isfinite(losses).all())
    passed = bool(finite and np.allclose(losses, expected_losses, rtol=1e-4, atol=1e-6)
                  and np.all(relative <= .01) and np.all(cosine >= .9999))
    mps = subprocess.check_output(['nvidia-cuda-mps-control'], input='get_server_list\n', text=True)
    clients = {server: subprocess.check_output(['nvidia-cuda-mps-control'],
               input=f'get_client_list {server}\n', text=True).split()
               for server in mps.split()}
    import os
    connected = any(str(os.getpid()) in pids for pids in clients.values())
    passed = passed and connected
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(student=student, recorded_at=datetime.now(timezone.utc).isoformat(),
                  passed=passed, mps_client_verified=connected, identities=identities,
                  execution=asdict(probe.execution), actual_execution=probe.last_gradient_execution,
                  gradient_seconds=seconds,
                  peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                  maximum_loss_difference=float(np.max(np.abs(losses - expected_losses))),
                  minimum_cosine=float(cosine.min()), maximum_relative_error=float(relative.max()),
                  both_zero_windows=int(both_zero.sum()),
                  row_norms=[dict(expected=float(e), actual=float(a))
                             for e, a in zip(norms, actual_norms)],
                  comparison='fresh FP32 projection against stored BF16 projection; quantization included',
                  cache_read_only=True, windows=len(windows))
    atomic_json(output, report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit('real cached-window agreement gate failed')


if __name__ == '__main__':
    main()
