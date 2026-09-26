"""Scan fixed random Ref trajectories once into reusable gradient assets."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from assets.gradient import GradientCache, WindowRequest
from assets.gradient.cache import atomic_json
from .development_inputs import load_fragments


def prepare_plan(config_path, out):
    config = json.loads(Path(config_path).read_text())
    rng = np.random.default_rng(20260914)
    samples, fragments = [], {}
    for task in config['datasets'][:5]:
        root = Path(config['input_roots'][task])
        provenance = json.loads((root / (task + '__manifest.json')).read_text())
        if provenance.get('formal_exam_excluded') is not True:
            raise ValueError('Reference exam exclusion not established: ' + task)
        path = root / (task + '__reference.npz')
        refs = load_fragments(path)
        eligible = [i for i, f in enumerate(refs) if f.length >= 576]
        rows = rng.choice(eligible, min(3, len(eligible)), replace=False)
        if not len(rows):
            raise ValueError('No legal reference trajectory: ' + task)
        for rank, row in enumerate(rows):
            f = refs[int(row)]
            sample = dict(task=task, dataset='reference:' + task,
                          fragment_id=f.fragment_id, entity_id=f.entity_id,
                          rank=rank, length=f.length, windows=len(f.valid_starts(512, 64)),
                          input_path=str(path), row=int(row))
            samples.append(sample)
            fragments[(task, f.fragment_id)] = f
    samples.sort(key=lambda s: (s['rank'], config['datasets'].index(s['task'])))
    manifest = dict(version=1, seed=20260914, context_length=512, prediction_length=64,
                    stride=1, samples=samples, total_windows=sum(s['windows'] for s in samples),
                    formal_exam_used=False, sampling='uniform complete trajectories without replacement')
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'sample_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Existing sample manifest differs; use a new output directory')
    atomic_json(path, manifest)
    for s in samples:
        f = fragments[(s['task'], s['fragment_id'])]
        starts = f.valid_starts(512, 64)
        assert len(starts) == f.length - 575 and starts[0] == 1
        for pos in (int(starts[0]), int(starts[-1])):
            np.testing.assert_array_equal(f.extract_window(pos, 512, 64), f.data[pos-1:pos+575])
    return manifest, fragments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rags/shared_tasks_10.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    manifest, fragments = prepare_plan(args.config, args.out)
    print(json.dumps(dict(event='PLAN_READY', **manifest)), flush=True)
    if args.plan_only:
        return
    from .full_pool_models import build_model
    from rags.execution import SelectionExecution, selection_execution
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    _, probe, identity = build_model('timemoe-200m', model_path=args.model_path)
    probe.execution = SelectionExecution(gradient_batch=8, gradient_workers=args.workers)
    cache = GradientCache(args.out, probe, model_identity=identity, data_fields=manifest,
                          context_length=512, prediction_length=64)
    atomic_json(args.out / 'cache_location.json', dict(path=str(cache.path), model=identity))
    started = time.monotonic()
    complete = []
    def status(event, sample=None):
        seconds = time.monotonic() - started
        rate = cache.computed_windows / seconds if seconds else 0
        value = dict(event=event, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                     pid=os.getpid(), cache_path=str(cache.path),
                     cached_windows=len(cache.rows), computed_this_run=cache.computed_windows,
                     total_windows=manifest['total_windows'], elapsed_seconds=seconds,
                     windows_per_second=rate,
                     eta_seconds=(manifest['total_windows']-len(cache.rows))/rate if rate else None,
                     completed_samples=complete, current_sample=sample)
        atomic_json(args.out / 'scan_status.json', value)
        print(json.dumps(value), flush=True)
    try:
        with selection_execution(probe):
            status('MODEL_READY')
            for sample in manifest['samples']:
                f = fragments[(sample['task'], sample['fragment_id'])]
                starts = f.valid_starts(512, 64)
                # Each prepare publishes at most 256 new rows; interrupted scans
                # recover completed shards using the existing asset protocol.
                for begin in range(0, len(starts), 256):
                    chunk = starts[begin:begin+256]
                    ids = np.column_stack([np.full(len(chunk), f.fragment_id), chunk])
                    def windows(rows, f=f, chunk=chunk):
                        return np.stack([f.extract_window(int(chunk[r]), 512, 64) for r in rows])
                    cache.prepare([WindowRequest(sample['dataset'], ids, windows)])
                    if begin % 1024 == 0:
                        cache._publish_index()
                    status('SCANNING', sample)
                cache._publish_index()
                complete.append(sample)
                status('TRAJECTORY_COMPLETE', sample)
            status('SCAN_COMPLETE')
    except BaseException as exc:
        atomic_json(args.out / 'scan_error.json', dict(error=repr(exc), pid=os.getpid()))
        raise
    finally:
        cache.close()


if __name__ == '__main__':
    main()
