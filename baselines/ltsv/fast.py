"""Explicit LTSV budget variant with full point coverage and resumable shards.

The opted-in production variant reduces overlap (stride 288 -> 576), preserves
64 reference windows and native one-step Adam, and uses deterministic per-block
RNG plus parallel GPU execution. It is not numerically identical to the original
budget. Cost notes preserve the original full-budget counterfactual separately.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from baselines.common import reference_windows
from baselines.ltsv.algorithms import ltsv_window_scores
from rage.execution import prepare_contexts, selection_execution
from rage.experiments.students import atomic_json

REVISION = 'ltsv-budget-execution-v1'


def block_starts(length, width, stride):
    if length < width:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - width + 1, stride, dtype=np.int64)
    if starts[-1] != length - width:
        starts = np.append(starts, length - width)
    return starts


def atomic_npz(path, **payload):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez(stream, **payload)
    temp.replace(path)


@contextmanager
def block_rng(device, seed, ordinal):
    """A scheduling-independent stream; never changes the caller's RNG stream."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed) + int(ordinal))
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(int(seed) + int(ordinal))
        yield


def reference_mse(probe, val, batch):
    from baselines.ltsv.runner import _ltsv_reference_mse
    if not callable(getattr(probe, 'forecast', None)):
        return _ltsv_reference_mse(probe, val)
    # Preserve original 32-row accumulation groups, even with a larger GPU batch.
    c, h = probe.config.context_length, probe.config.prediction_length
    errors = []
    probe.model.eval()
    with torch.no_grad():
        for start in range(0, len(val), batch):
            rows = val[start:start + batch]
            pred = np.asarray(probe.forecast(rows[:, :c]))
            errors.append(np.square(pred - rows[:, c:]))
    errors = np.concatenate(errors)
    return sum(float(errors[i:i + 32].sum()) for i in range(0, len(errors), 32)) / (len(val) * h)


class BlockScorer:
    def __init__(self, probe, val, *, lr=1e-5, inference_batch=64, generation='eager', seed=20260916):
        from baselines.ltsv.runner import _restore_snapshot_on_device
        self.probe, self.model = probe, probe.model
        self.original_execution = probe.execution
        probe.execution = replace(probe.execution, inference_batch=inference_batch, generation=generation)
        self.batch, self.lr, self.seed = inference_batch, lr, seed
        self.val = (val if callable(getattr(probe, 'forecast', None)) else prepare_contexts(probe, val))
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        cpu = torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters])
        self.base = _restore_snapshot_on_device(cpu, probe.device)
        self.views = list(self.base.split([p.numel() for p in self.parameters]))
        self.model.eval()
        try:
            self.base_loss = reference_mse(probe, self.val, self.batch)
        except BaseException:
            self.close()
            raise

    def restore(self):
        with torch.no_grad():
            for p, base in zip(self.parameters, self.views):
                p.copy_(base.view_as(p))

    def score(self, window, ordinal):
        from baselines.ltsv.runner import _ltsv_training_loss
        with block_rng(self.probe.device, self.seed, ordinal):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, foreach=False,
                betas=(.9, .999), eps=1e-8, weight_decay=.1)
            try:
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _ltsv_training_loss(self.probe, window)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.model.eval()
                return self.base_loss - reference_mse(self.probe, self.val, self.batch)
            finally:
                self.restore()
                self.model.eval()
                optimizer.zero_grad(set_to_none=True)

    def close(self):
        self.restore()
        self.probe.execution = self.original_execution
        from rage.execution import clear_execution_cache
        clear_execution_cache(self.probe)


def score_fragment(scorer, task, *, width, stride, checkpoint_every=32):
    """One writer per trajectory; a published prefix is safe to resume."""
    fid, data, ordinal, path = task
    starts = block_starts(len(data), width, stride)
    values, elapsed, previous = [], 0., 0
    path = Path(path) if path else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            previous = len(saved['values'])
            if int(saved['fragment_id']) != fid or not np.array_equal(saved['starts'], starts[:previous]):
                raise ValueError('LTSV shard window identities differ')
            values = saved['values'].tolist()
            elapsed = float(saved['compute_seconds'])
    begun = time.perf_counter()
    checkpoint = begun
    for i in range(previous, len(starts)):
        window = np.asarray(data[starts[i]:starts[i] + width], dtype=np.float32)
        native = np.lib.stride_tricks.sliding_window_view(window,
            scorer.probe.config.context_length + scorer.probe.config.prediction_length).copy()
        canonical = block_starts(len(data), width, width // 2)
        block_id = int(np.searchsorted(canonical, starts[i]))
        values.append(scorer.score(native, ordinal + block_id))
        if path and ((i + 1) % checkpoint_every == 0 or i + 1 == len(starts)):
            now = time.perf_counter()
            elapsed += now - checkpoint
            atomic_npz(path, fragment_id=fid, starts=starts[:len(values)],
                values=np.asarray(values, dtype=np.float64), compute_seconds=elapsed)
            checkpoint = time.perf_counter()
    if torch.device(scorer.probe.device).type == 'cuda':
        torch.cuda.synchronize(scorer.probe.device)
    finished = time.perf_counter()
    return dict(fragment_id=fid, starts=starts, values=np.asarray(values),
        work_started=begun, work_finished=finished,
        resumed_blocks=previous, computed_blocks=len(starts) - previous,
        compute_seconds=elapsed if path else time.perf_counter() - begun,
        peak_gpu_gib=(torch.cuda.max_memory_allocated(scorer.probe.device)/2**30
                      if torch.device(scorer.probe.device).type == 'cuda' else 0.))


_SCORER = None

def _init_worker(student, model_path, identity, val, options, threads):
    global _SCORER
    from rage.experiments.full_pool_models import build_model
    torch.set_num_threads(threads)
    _, probe, loaded = build_model(student, model_path=model_path)
    if loaded != identity:
        raise ValueError('LTSV worker model identity differs')
    probe.execution = replace(probe.execution, cpu_threads=threads)
    _SCORER = BlockScorer(probe, val, **options)


def _work(task, width, stride):
    with selection_execution(_SCORER.probe):
        return score_fragment(_SCORER, task, width=width, stride=stride)


def run_fragments(probe, identity, val, tasks, *, width, stride, workers, options):
    if workers == 1:
        scorer = BlockScorer(probe, val, **options)
        try:
            for task in tasks:
                yield score_fragment(scorer, task, width=width, stride=stride)
        finally:
            scorer.close()
        return
    # Spawn, never fork a process with an initialized CUDA context.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
            initializer=_init_worker, initargs=(identity['student'], identity['loaded_path'],
                identity, val, options, probe.execution.cpu_threads or 4)) as executor:
        pending = set()
        iterator = iter(tasks)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 2 * workers and not exhausted:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(_work, task, width, stride))
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()


def optimized_select(engine, *, reference_fragments, index, context_length,
        prediction_length, cache_dir, pool_manifest, dataset_tag, seed, resampling,
        execution, block_length=None, block_stride=None, icf_lr=1e-5,
        context_windows=64, n_select=None):
    from rage.experiments.arms import ArmOutput, SelectionCost
    from rage.experiments.resampling import window_distribution
    if resampling is None:
        raise ValueError('Optimized LTSV currently requires the full-pool protocol')
    started = time.perf_counter()
    width = block_length or context_length + prediction_length
    stride = execution.get('block_stride', block_stride or width // 2)
    original_context_windows = context_windows
    context_windows = execution.get('context_windows', context_windows)
    if not 1 <= context_windows <= original_context_windows:
        raise ValueError('Budget reference count must be in the original reference set')
    if width < context_length + prediction_length or not 1 <= stride <= width:
        raise ValueError('Invalid LTSV block shape')
    if execution['schema'] != REVISION:
        raise ValueError('Unknown LTSV execution schema')
    identity = resampling['model']
    if identity != execution['model']:
        raise ValueError('LTSV execution/model fields differ')
    workers = execution['workers']
    if workers not in (1, 2, 4):
        raise ValueError('LTSV worker count must be 1, 2 or 4')
    requested_workers = workers
    # Leave room for concurrently running students and allocator/driver overhead.
    if torch.device(engine.device).type == 'cuda' and execution.get('worker_peak_gpu_gib'):
        free, _ = torch.cuda.mem_get_info(engine.device)
        capacity = int(free * .6 / (execution['worker_peak_gpu_gib'] * 1.25 * 2**30))
        workers = max(n for n in (1, 2, 4) if n <= requested_workers and (n <= capacity or n == 1))
    print(f'  [LTSV budget] stride={stride}, Ref={context_windows}, '
          f'workers={workers}/{requested_workers}, generation={execution["generation"]}', flush=True)
    val, ref_ids = reference_windows(reference_fragments, context_length=context_length,
        prediction_length=prediction_length, stride=16, max_windows=original_context_windows, seed=seed)
    subset = np.linspace(0, len(val)-1, min(context_windows, len(val)), dtype=int)
    val, ref_ids = val[subset], [ref_ids[i] for i in subset]
    options = dict(lr=icf_lr, inference_batch=execution['inference_batch'],
        generation=execution['generation'], seed=execution['block_seed'])
    # Worker count changes scheduling only, and therefore is absent from identity.
    manifest = json.loads(json.dumps(dict(schema=REVISION, rng='canonical-original-block-seed-v1', model=identity,
        pool=pool_manifest, reference_ids=ref_ids, task=dataset_tag, reference_seed=seed,
        context=context_length, horizon=prediction_length, width=width, stride=stride,
        context_windows=context_windows, options=options, index_stride=index.stride)))
    directory = Path(cache_dir) / REVISION / dataset_tag
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = directory / 'cache_manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('LTSV cache fields differ; choose a new cache directory')
        else:
            atomic_json(manifest_path, manifest)
        tasks, ordinal, seed_ordinal = [], 0, 0
        original_blocks = sum(len(block_starts(f.length, width, width//2))
                              for f in index.by_fragment.values())
        for fid, fragment in index.by_fragment.items():
            tasks.append((int(fid), fragment.data, seed_ordinal, str(directory / f'fragment-{fid}.npz')))
            ordinal += len(block_starts(fragment.length, width, stride))
            seed_ordinal += len(block_starts(fragment.length, width, width//2))
        scores = np.full(len(index), -np.inf, dtype=np.float32)
        computed = resumed = completed = 0
        peak_gpu_gib = 0.
        for row in run_fragments(engine, identity, val, tasks, width=width, stride=stride,
                workers=workers, options=options):
            fragment = index.by_fragment[row['fragment_id']]
            sums = np.zeros(fragment.length + 1, dtype=np.float64)
            counts = np.zeros(fragment.length + 1, dtype=np.int64)
            for begin, value in zip(row['starts'], row['values']):
                sums[begin:begin + width] += value
                counts[begin:begin + width] += 1
            legal = fragment.valid_starts(context_length, prediction_length, index.stride)
            lo = index.bounds(fragment.fragment_id)[0]
            scores[lo:lo + len(legal)] = ltsv_window_scores(sums, counts, legal - 1,
                context_length + prediction_length)
            peak_gpu_gib = max(peak_gpu_gib, row['peak_gpu_gib'])
            atomic_json(directory / 'progress.json', dict(
                completed_trajectories=completed + 1, total_trajectories=len(tasks),
                computed_blocks=computed + row['computed_blocks'],
                resumed_blocks=resumed + row['resumed_blocks'],
                actual_session_seconds=time.perf_counter()-started,
                original_blocks=original_blocks, budget_blocks=ordinal, workers=workers))
            computed += row['computed_blocks']
            resumed += row['resumed_blocks']
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f'  [LTSV fast] {completed}/{len(tasks)} trajectories; '
                      f'{computed} new + {resumed} cached blocks; {time.perf_counter()-started:.1f}s', flush=True)
        seconds = time.perf_counter() - started
        cold_estimate = execution.get('baseline_seconds_per_block')
        adjustment = None
        if cold_estimate is not None and not resumed:
            adjustment = dict(schema='baseline-runtime-adjustment-v1',
                optimization='ltsv_budget_reduction_and_execution',
                add_back_seconds=original_blocks * cold_estimate - seconds,
                report_basis='estimated_original_full_budget_serial_implementation',
                formula='reported = original_full_budget_blocks * measured_original_mean_seconds_per_block',
                calibration=execution.get('validation_report'),
                uncertainty='Small source/reference pilot under live GPU contention; excludes one-time setup; not upstream runtime')
        notes = dict(scores_cached=resumed == ordinal, runtime_adjustment=adjustment,
            resumed_blocks=resumed, computed_blocks=computed, execution=execution,
            partial_resume=bool(resumed and computed), actual_session_seconds=seconds,
            max_worker_peak_gpu_gib=peak_gpu_gib, actual_workers=workers,
            requested_workers=requested_workers, original_blocks=original_blocks,
            original_context_windows=original_context_windows, budget_blocks=ordinal)
        if resumed:
            # A resumed session's wall time is not a cold execution measurement.
            notes['scores_cached'] = True
        selected = window_distribution(index, scores, pool_manifest=pool_manifest,
            metadata=dict(resampling, ltsv_execution=execution, ltsv_rng='canonical-original-block-seed-v1'),
            temperature=resampling.get('temperature', 1.))
        seconds = time.perf_counter() - started
        notes['actual_session_seconds'] = seconds
        if adjustment is not None:
            adjustment['add_back_seconds'] = original_blocks * cold_estimate - seconds
        atomic_json(directory / 'complete.json', dict(blocks=ordinal, computed_blocks=computed,
            resumed_blocks=resumed, actual_session_seconds=seconds, execution=execution,
            actual_workers=workers, requested_workers=requested_workers,
            original_blocks=original_blocks, original_context_windows=original_context_windows,
            original_seconds_estimate=(original_blocks*cold_estimate if cold_estimate else None),
            estimated_original_to_actual_ratio=(original_blocks*cold_estimate/seconds
                                               if cold_estimate and not resumed else None)))
    return ArmOutput('ltsv', selected, [], SelectionCost(gradient_computations=computed,
        feature_computations=computed * context_windows, wall_seconds=seconds, notes=notes),
        dict(blocks=ordinal, context_windows=context_windows, block_length=width,
            block_stride=stride, reference_loss='forecast_mse', execution=execution,
            native_kernel='block_to_point_mean_to_window_mean', resampling=selected.coverage()))
