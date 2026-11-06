"""Execution-only acceleration. No changes to loss, projection or window identity."""
from collections import OrderedDict
from dataclasses import asdict, replace
import time

import numpy as np


class PublishedGradientRows:
    """Thread-owned reader over immutable published shards, never a cache writer.

    Snapshot only one completed fragment. GPU/cache writes can continue while
    this reader runs CRB/geometry on the CPU; no mutable LRU or index is shared.
    """
    def __init__(self, cache, dataset, fragment_id, count, budget=512*2**20):
        self.locations = tuple(cache.rows[(dataset, fragment_id, i+1)] for i in range(count))
        self.dataset, self.fragment_id = dataset, fragment_id
        self.budget, self.bytes = budget, 0
        self.shards = OrderedDict()

    def read(self, lo, hi):
        if not 0 <= lo < hi <= len(self.locations):
            raise ValueError('invalid published-gradient slice')
        grouped = {}
        for target, (path, row) in enumerate(self.locations[lo:hi]):
            grouped.setdefault(path, []).append((target,row))
        result = None
        for path, locations in grouped.items():
            if path not in self.shards:
                with np.load(path, allow_pickle=False) as shard:
                    if str(shard['dataset']) != self.dataset:
                        raise ValueError('published shard dataset identity mismatch')
                    g = (shard['gradients_bf16'].astype(np.uint32) << 16).view(np.float32)
                    ids = shard['window_ids']
                self.shards[path] = (g,ids); self.bytes += g.nbytes+ids.nbytes
                while self.bytes > self.budget and len(self.shards)>1:
                    _, (old, old_ids) = self.shards.popitem(last=False)
                    self.bytes -= old.nbytes+old_ids.nbytes
            else:
                self.shards.move_to_end(path)
            g, ids = self.shards[path]
            if g.ndim != 2:
                raise ValueError('census requires independent [N,D] gradient rows')
            targets, rows = np.asarray(locations,dtype=np.int64).T
            if not (np.all(ids[rows,0] == self.fragment_id) and np.array_equal(ids[rows,1],lo+targets+1)):
                raise ValueError('published shard window identity mismatch')
            if result is None:
                result = np.empty((hi-lo,g.shape[1]),dtype=np.float32)
            result[targets] = g[rows]
        return result


def gradient_agreement(loss, gradient, baseline_loss, baseline_gradient, tolerance=.05):
    loss, gradient = np.asarray(loss), np.asarray(gradient)
    a, b = np.asarray(baseline_loss), np.asarray(baseline_gradient)
    if loss.shape != a.shape or gradient.shape != b.shape or gradient.ndim != 2:
        return dict(passed=False,reason='shape_mismatch')
    if not all(np.isfinite(x).all() for x in (loss,gradient,a,b)):
        return dict(passed=False,reason='nonfinite')
    # FP64 avoids tiny BF16-gradient squared norms underflowing in FP32.
    gradient, b = gradient.astype(np.float64), b.astype(np.float64)
    denom = np.maximum(np.linalg.norm(gradient,axis=1),np.linalg.norm(b,axis=1))
    delta = np.linalg.norm(gradient-b,axis=1)
    error = np.divide(delta,denom,out=np.zeros_like(delta),where=denom>0)
    le = np.abs(loss-a)/np.maximum(np.maximum(np.abs(loss),np.abs(a)),1e-30)
    return dict(passed=bool(error.max()<=tolerance and le.max()<=tolerance),
        max_gradient_relative_l2=float(error.max()),max_loss_relative_error=float(le.max()),tolerance=tolerance)


def choose_execution(records):
    eligible = [r for r in records if r.get('agreement',{}).get('passed')
                and r.get('windows_per_second',0)>0 and not r.get('oom_retries',0)]
    if not eligible:
        raise ValueError('no numerically valid execution candidate; inspect execution_profile.json')
    fastest = max(r['windows_per_second'] for r in eligible)
    # Prefer lower memory/parallelism within 5% of the measured fastest.
    return min((r for r in eligible if r['windows_per_second']>=.95*fastest),
               key=lambda r:(r['peak_bytes'],r['workers'],r['batch']))


def tune_execution(probe, windows, batches, workers, *, repeats=3):
    """Bounded safe-history microbenchmark; every candidate checked at its real shape.

    Probe caps (TimesFM16, Timer32) are respected. Chronos BF16 uses its
    existing independent GPU processes, never FP32/vmap substitution.
    """
    import torch
    original = probe.execution
    device = torch.device(probe.device)
    cuda = device.type == 'cuda'
    def release():
        if hasattr(probe,'release_gradient_workers'):
            probe.release_gradient_workers()
        if cuda:
            torch.cuda.empty_cache()
    release()
    probe.execution = replace(original,gradient_batch=1,gradient_workers=1)
    single = [probe.batch_gradient_sketch(w[None]) for w in windows]
    baseline_loss = np.concatenate([x[0] for x in single])
    baseline_gradient = np.concatenate([x[1] for x in single])
    records = []
    cap = getattr(probe,'max_gradient_batch',max(batches))
    try:
        for count in workers:
            for batch in batches:
                if batch>cap:
                    continue
                release()
                probe.execution = replace(original,gradient_batch=batch,gradient_workers=count)
                record = dict(batch=batch,workers=count)
                try:
                    # Warm kernels and worker pools; startup isn't steady throughput.
                    loss, gradient = probe.batch_gradient_sketch(windows)
                    record['agreement'] = gradient_agreement(loss,gradient,baseline_loss,baseline_gradient)
                    if not record['agreement']['passed']:
                        records.append(record); continue
                    if cuda:
                        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
                    durations=[]
                    retried=int(getattr(probe,'last_gradient_execution',{}).get('oom_retries',0))
                    for _ in range(repeats):
                        started=time.perf_counter()
                        loss, gradient = probe.batch_gradient_sketch(windows)
                        if cuda:
                            torch.cuda.synchronize(device)
                        durations.append(time.perf_counter()-started)
                        retried += int(getattr(probe,'last_gradient_execution',{}).get('oom_retries',0))
                    backend=getattr(probe,'last_gradient_execution',{})
                    peaks=backend.get('worker_peak_gpu_gib',{})
                    peak=(torch.cuda.max_memory_reserved(device) if cuda else 0)+sum(peaks.values())*2**30
                    record.update(windows_per_second=float(len(windows)/np.median(durations)),
                        durations=durations,peak_bytes=int(peak),oom_retries=retried,
                        backend=backend,agreement=gradient_agreement(loss,gradient,baseline_loss,baseline_gradient))
                except torch.cuda.OutOfMemoryError:
                    record.update(oom_retries=1,agreement=dict(passed=False,reason='oom'))
                records.append(record)
    finally:
        release(); probe.execution=original
    try:
        chosen=choose_execution(records)
    except ValueError:
        chosen=None
    if chosen:
        probe.execution=replace(original,gradient_batch=chosen['batch'],gradient_workers=chosen['workers'])
    return dict(revision='census-execution-profile-v1',samples=len(windows),repeats=repeats,
        candidates=records,chosen=chosen,effective_execution=asdict(probe.execution),
        objective='maximum valid windows/second, lower memory within 5%; not GPU utilization alone')
