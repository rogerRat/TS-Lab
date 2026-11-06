"""Exhaustive safe-reference gradient census; formal execution belongs to the operator."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .reference_geometry import summarize_trajectory, summarize_task, alignment_summary

SCHEMA = 'reference-census-20260921-v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    from assets.gradient.cache import atomic_json
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def safe_references(inputs, task, c, h):
    """Validate exhaustive native finite segments against the pre-exam manifest."""
    from .development_inputs import load_fragments
    inputs = Path(inputs)
    manifest = read_json(inputs / f'{task}__manifest.json')
    required = dict(full_reference_history=True, formal_exam_excluded_from_reference=True,
                    source_training_only=True, context_length=c, prediction_length=h)
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError(f'{task}: safe full-history identity missing/mismatched')
    if manifest.get('dataset') != task:
        raise ValueError('reference dataset identity mismatch')
    fragments = load_fragments(inputs / f'{task}__reference.npz')
    expected = {}
    for series in manifest['series']:
        for segment in series['reference_segments']:
            fid = int(segment['fragment_id'])
            if fid in expected or not 0 <= segment['start'] < segment['stop'] <= series['exam_boundary']:
                raise ValueError('duplicate segment or unsafe reference boundary')
            expected[fid] = (str(series['entity']), segment['stop']-segment['start'])
    if len(fragments) != len(expected) or len({f.fragment_id for f in fragments}) != len(fragments):
        raise ValueError('full reference fragment coverage mismatch')
    for f in fragments:
        if expected.get(f.fragment_id) != (f.entity_id, len(f.data)) or not np.isfinite(f.data).all():
            raise ValueError('reference segment identity/finite values mismatch')
        if len(f.data) < c+h:
            raise ValueError('manifest contains a segment without legal starts')
    return fragments, manifest


def inventory(runs):
    rows = []
    identities = {}
    for run in map(Path, runs):
        config = read_json(run/'protocol.json')
        c, h = config['context_length'], config['prediction_length']
        for task in config['datasets']:
            refs, manifest = safe_references(config['inputs'], task, c, h)
            n = sum(len(f.data)-c-h+1 for f in refs)
            for student in config['students']:
                key = (student, task)
                identity = dict(inputs=config['inputs'], manifest=manifest, c=c, h=h,
                                model_path=config.get('model_paths', {}).get(student))
                if key in identities and identities[key] != identity:
                    raise ValueError(f'conflicting reference/model identity across runs: {key}')
                duplicate = key in identities
                identities[key] = identity
                models = list((run/'students'/student).glob('model.json'))
                models += sorted(run.glob(f'rounds/seed*/students/{student}/model.json'))
                model = read_json(models[0]) if models else {}
                d = model.get('projection_dimension')
                timings = []
                cache_root = Path(config.get('gradient_cache_source') or run/'grad_cache')
                for timing_path in cache_root.glob('*/gradient_timing.json'):
                    for rec in read_json(timing_path).get('records', []):
                        if rec.get('identity', {}).get('model', {}).get('student') == student and rec.get('gradient_windows', 0)>0:
                            timings.append(rec['gradient_seconds']/rec['gradient_windows'])
                seconds = float(np.median(timings)) if timings else None
                rows.append(dict(run=str(run), student=student, dataset=task,
                    reference_fragments=len(refs), legal_windows=n, stride=1,
                    already_listed_reference=duplicate, projection_dimension=d,
                    gradient_bytes_bf16_lower_bound=n*d*2 if d else None,
                    historical_seconds_per_window=seconds,
                    estimated_gradient_seconds=n*seconds if seconds is not None else None,
                    note='BF16 gradient payload only; add IDs, losses, index, source draws, working memory and headroom'))
    return dict(schema=SCHEMA, status='planned_not_computed', runs=list(map(str, runs)), tasks=rows,
                unique_model_task_pairs=len(identities),
                unique_reference_windows=sum(r['legal_windows'] for r in rows if not r['already_listed_reference']))


def fragment_request(fragment, dataset, lo, hi, c, h):
    from assets.gradient.cache import WindowRequest
    starts = np.arange(lo+1, hi+1, dtype=np.int64)
    def windows(rows):
        locations = starts[rows]-1
        return fragment.data[locations[:, None]+np.arange(c+h)]
    return WindowRequest(dataset, np.column_stack((np.full(len(starts), fragment.fragment_id), starts)), windows)


def prepare_with_oom(cache, request):
    import torch
    from rage.execution import execution_policy
    while True:
        try:
            cache.prepare([request])
            return
        except torch.cuda.OutOfMemoryError:
            policy = execution_policy(cache.engine)
            current = policy.batch_size(cache.engine, 'gradient', fallback=32, count=len(request.ids))
            if current <= 1:
                raise
            cache.engine.execution = replace(policy, gradient_batch=max(1, current//2))
            torch.cuda.empty_cache()
            print('CENSUS_OOM_RETRY', current, '->', cache.engine.execution.gradient_batch, flush=True)


def validate_probe(probe, windows):
    """Small native check: independent batched rows agree with single windows."""
    losses, gradients = probe.batch_gradient_sketch(windows)
    singles = [probe.batch_gradient_sketch(windows[i:i+1]) for i in range(len(windows))]
    losses, gradients = np.asarray(losses), np.asarray(gradients)
    single_loss = np.concatenate([np.asarray(x[0]) for x in singles])
    single_gradient = np.concatenate([np.asarray(x[1]) for x in singles])
    if losses.shape != (len(windows),) or gradients.ndim != 2 or gradients.shape != single_gradient.shape:
        raise ValueError('probe does not return independent [N] losses and [N,D] gradients')
    if not all(np.isfinite(a).all() for a in (losses, gradients, single_loss, single_gradient)):
        raise ValueError('nonfinite native probe smoke result')
    denominator = np.maximum(np.linalg.norm(gradients,axis=1),np.linalg.norm(single_gradient,axis=1))
    delta = np.linalg.norm(gradients-single_gradient,axis=1)
    error = np.divide(delta,denominator,out=np.zeros_like(delta),where=denominator>0)
    loss_error = np.abs(losses-single_loss)/np.maximum(np.maximum(np.abs(losses),np.abs(single_loss)),1e-30)
    result = dict(windows=len(windows),loss_relative_error=loss_error.tolist(),
                  gradient_relative_l2_error=error.tolist(), tolerance=.05,
                  passed=bool(np.max(error)<=.05 and np.max(loss_error)<=.05))
    return result


def fragment_geometry(f, reader, c, h, crb, chunk):
    from rage.crb import CRBEngine
    from .reference_geometry import cosine
    n = len(f.data)-c-h+1
    def gradients(fragment, starts):
        # CRB requests need not be consecutive; preserve exact request order.
        return np.concatenate([reader.read(int(s)-1,int(s)) for s in starts])
    result = CRBEngine(crb, gradients).process_fragment(f)
    metrics = summarize_trajectory(reader.read, n, lags=(1,h,c+h),chunk=chunk,anchors=result.anchors)
    values=np.asarray(f.data,dtype=np.float64);scale=float(values.std())
    metrics['signal']=dict(std=scale,zero_fraction=float(np.mean(values==0)),constant=bool(scale==0),
        lag1_correlation=cosine(values[:-1]-values[:-1].mean(),values[1:]-values[1:].mean()),
        normalized_step_rms=float(np.sqrt(np.square(np.diff(values)).mean())/scale) if scale else None,
        standardized_early_late_mean_shift=float(abs(values[-max(1,len(values)//4):].mean()-values[:max(1,len(values)//4)].mean())/scale) if scale else None)
    return dict(metrics,fragment_id=int(f.fragment_id),entity_id=f.entity_id)


def collect_references(cache, refs, task, c, h, crb, *, chunk, progress=None, overlap_geometry=True):
    from .reference_execution import PublishedGradientRows
    namespace = 'reference:'+task
    rows = []
    pending = None
    # At most one CPU geometry job plus one active GPU fragment. CPU only
    # receives immutable published shard locations, never the live cache.
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='reference-geometry') as executor:
        for f in refs:
            n = len(f.data)-c-h+1
            for lo in range(0,n,chunk):
                request=fragment_request(f,namespace,lo,min(n,lo+chunk),c,h)
                prepare_with_oom(cache,request)
                if progress:
                    progress(f.fragment_id,min(n,lo+chunk),n)
            reader=PublishedGradientRows(cache,namespace,f.fragment_id,n)
            if pending is not None:
                rows.append(pending.result())
            if overlap_geometry:
                pending=executor.submit(fragment_geometry,f,reader,c,h,crb,chunk)
            else:
                rows.append(fragment_geometry(f,reader,c,h,crb,chunk));pending=None
        if pending is not None:
            rows.append(pending.result())
    expected = sum(len(f.data)-c-h+1 for f in refs)
    actual = sum(1 for key in cache.rows if key[0] == namespace)
    if actual != expected:
        raise ValueError(f'census coverage mismatch {task}: {actual} != {expected}')
    summary = summarize_task(rows)
    # Sensitivity: actual native channels may have several finite segments.
    entities = sorted({r['entity_id'] for r in rows})
    weights = np.asarray([1/len(entities)/sum(x['windows'] for x in rows if x['entity_id']==r['entity_id'])*r['windows'] for r in rows])
    summary['native_entity_mean_raw'] = (weights @ np.asarray([r['mean_raw'] for r in rows])).tolist()
    summary['native_entity_mean_unit'] = (weights @ np.asarray([r['mean_unit'] for r in rows])).tolist()
    return dict(coverage=dict(expected=expected, actual=actual, complete=True, stride=1),
                weighting='equal_finite_reference_fragment_primary; native_entity_equal_sensitivity',
                summary=summary, trajectories=rows)


def selection_path(run, student, task, arm):
    candidates = [Path(run)/'students'/student/'selections'/f'{task}__{arm}.npz']
    candidates += sorted(Path(run).glob(f'rounds/seed*/students/{student}/selections/{task}__{arm}.npz'))
    found = [p for p in candidates if p.is_file()]
    if not found:
        raise FileNotFoundError(f'No completed selection for {student}/{task}/{arm}')
    return found[0]


def collect_alignment(cache, run, student, task, arms, reference, *, samples, seed, config, crb):
    from .regions import load_selection
    from .full_pool import validate_cached_selection
    from assets.gradient.cache import WindowRequest
    output = {}
    for arm in arms:
        if arm == 'orig':
            continue
        path = selection_path(run, student, task, arm)
        selected = load_selection(path).selected
        if not hasattr(selected, 'sample_indices'):
            raise ValueError('alignment requires current frozen resampling selections')
        temperatures = config['settings'][student]['temperature']
        temperature = temperatures.get(arm, 1.) if isinstance(temperatures, dict) else temperatures
        validate_cached_selection(selected.metadata, arm=arm, identity=cache.manifest['model'],
            manifest=cache.manifest['data']['pool'], crb=crb, temperature=temperature)
        if selected.metadata.get('task', task) != task or selected.window_length != cache.context_length+cache.prediction_length:
            raise ValueError('selection task/window identity mismatch')
        # Same RNG seed per arm supplies common random numbers, never reused
        # reference/model gradients from another checkpoint or model.
        rows = selected.sample_indices(np.random.default_rng(seed), samples)
        ids = selected.window_ids[rows]
        request = WindowRequest('source', ids, lambda indices: selected[rows[indices]])
        prepare_with_oom(cache, request)
        gradients = cache.get(request)
        metrics = alignment_summary(gradients, reference)
        entity_ref = dict(mean_raw=reference['native_entity_mean_raw'], mean_unit=reference['native_entity_mean_unit'])
        metrics['native_entity_weighting'] = alignment_summary(gradients, entity_ref)
        counts = np.diff(selected.unit_offsets)
        p = selected.probabilities
        # Per-window ESS works across block, trajectory and window objects.
        metrics['window_ess'] = float(1/np.sum(p*p/counts))
        metrics['window_ess_fraction'] = metrics['window_ess']/len(selected)
        unique, count = np.unique(ids[:, 0], return_counts=True)
        metrics['sampled_source_fragments'] = len(unique)
        metrics['sampled_max_fragment_share'] = float(count.max()/samples)
        metrics['unique_sampled_windows'] = len(np.unique(ids, axis=0))
        output[arm] = dict(metrics=metrics, selection_path=str(path), seed=seed,
                           sampled_window_ids=ids.tolist())
    baseline = output.get('rand', {}).get('metrics', {})
    for arm in output:
        metrics = output[arm]['metrics']
        metrics['alignment_gain_vs_rand'] = (metrics['mean_pair_cosine']-baseline['mean_pair_cosine']
                                             if 'mean_pair_cosine' in baseline else None)
    return output


def compute(args, *, model_bundle=None):
    import torch
    from rage.execution import SelectionExecution, selection_execution
    from rage.crb import CRBConfig
    from assets.gradient.cache import GradientCache
    from .full_pool_models import build_model
    run = Path(args.run); config = read_json(run/'protocol.json')
    if args.student not in config['students']:
        raise ValueError('student is absent from this run')
    c, h = config['context_length'], config['prediction_length']
    out = Path(args.out)/args.student
    out.mkdir(parents=True, exist_ok=True)
    tasks = config['datasets']
    references = {task: safe_references(config['inputs'], task, c, h) for task in tasks}
    engine, probe, identity = model_bundle or build_model(args.student,
        model_path=config.get('model_paths', {}).get(args.student), context_length=c, prediction_length=h,
        device=config.get('device', 'cuda'))
    execution = dict(config.get('selection_execution', {}))
    execution.update(config.get('student_selection_execution', {}).get(args.student, {}))
    if (run/'execution.json').exists():
        execution.update(read_json(run/'execution.json').get('student_selection_execution', {}).get(args.student, {}))
    if args.gradient_batch is not None:
        execution['gradient_batch'] = args.gradient_batch
    for option in ('gradient_workers','memory_fraction','cpu_threads'):
        if getattr(args,option,None) is not None:
            execution[option]=getattr(args,option)
    probe.execution = SelectionExecution(**execution)
    data_fields = dict(schema=SCHEMA, inputs=str(Path(config['inputs']).resolve()),
        pool=read_json(Path(config['inputs'])/'pool_manifest.json'),
        references={t: m for t, (_, m) in references.items()})
    total = sum(len(f.data)-c-h+1 for fs, _ in references.values() for f in fs)
    dimension = identity['projection_dimension']
    crb = CRBConfig(context_length=c, prediction_length=h, **config['crb'])
    if crb.window_stride != 1:
        raise ValueError('reference census and CRB must use stride=1')
    profile_path=out/'execution_profile.json'
    if getattr(args,'tune_batches',None):
        from .reference_execution import tune_execution
        f=next(iter(references.values()))[0][0]
        count=min(32,len(f.data)-c-h+1)
        positions=np.rint(np.linspace(0,len(f.data)-c-h,count)).astype(int)
        windows=f.data[positions[:,None]+np.arange(c+h)]
        tuning_identity=dict(schema='census-tuning-v1',model=identity,data=data_fields,
            execution=asdict(probe.execution),batches=args.tune_batches,
            workers=getattr(args,'tune_workers',None) or [probe.execution.gradient_workers])
        saved=read_json(profile_path) if profile_path.exists() else None
        if saved and saved.get('identity')==tuning_identity and saved.get('chosen'):
            probe.execution=SelectionExecution(**saved['effective_execution'])
        else:
            with selection_execution(probe):
                result=tune_execution(probe,windows,tuning_identity['batches'],tuning_identity['workers'])
            publish(profile_path,dict(result,identity=tuning_identity))
            if not result.get('chosen'):
                raise ValueError('no valid execution candidate; inspect execution_profile.json')
    cache = GradientCache(out, probe, model_identity=identity, data_fields=data_fields,
                          context_length=c, prediction_length=h)
    started = time.monotonic()
    try:
        remaining = total-sum(k[0].startswith('reference:') for k in cache.rows)
        lower_bound = max(0, remaining)*(2*dimension+32)
        if shutil.disk_usage(out).free < lower_bound*1.25 + 2**30:
            raise RuntimeError(f'insufficient data disk headroom: additional minimum {lower_bound} bytes')
        census_identity = dict(schema=SCHEMA, model=identity, data=data_fields, crb=config['crb'],
                               lags=[1,h,c+h], gradient_space='frozen_native_loss_projected_BF16')
        with selection_execution(probe):
            smoke_path = out/'probe_validation.json'
            smoke = read_json(smoke_path) if smoke_path.exists() else None
            if smoke is None or smoke.get('identity') != census_identity or not smoke.get('passed'):
                f = next(iter(references.values()))[0][0]
                count = min(3,len(f.data)-c-h+1)
                request = fragment_request(f,'smoke',0,count,c,h)
                smoke = validate_probe(probe,request.windows(np.arange(count)))
                publish(smoke_path,dict(smoke,identity=census_identity))
                if not smoke['passed']:
                    raise ValueError('native batched/single-window probe validation failed; inspect probe_validation.json')
            for task, (refs, _) in references.items():
                def progress(fid, done, n):
                    publish(out/'state.json', dict(schema=SCHEMA, status='running', dataset=task,
                        fragment_id=int(fid), fragment_windows_done=done, fragment_windows=n,
                        new_gradients=cache.computed_windows, elapsed_seconds=time.monotonic()-started,
                        cache=str(cache.path)))
                destination = out/f'{task}__geometry.json'
                if destination.exists():
                    record = read_json(destination)
                    if record.get('identity') != census_identity or not record['coverage']['complete']:
                        raise ValueError('geometry identity changed; use a new analysis directory')
                    if sum(k[0]=='reference:'+task for k in cache.rows) != record['coverage']['expected']:
                        raise ValueError('completed geometry has missing gradient resources; restore the cache before resuming')
                else:
                    record = collect_references(cache, refs, task, c, h, crb, chunk=args.chunk, progress=progress)
                    record.update(identity=census_identity, student=args.student, dataset=task, cache=str(cache.path))
                    publish(destination, record)
                if args.stage in ('all', 'alignment'):
                    alignment = collect_alignment(cache, run, args.student, task, config['arms'],
                        record['summary'], samples=args.alignment_samples, seed=args.seed, config=config, crb=crb)
                    # Keep every source-MC repetition; the conventional filename
                    # below points to the most recently requested primary analysis.
                    publish(out/f'{task}__alignment__{run.name}__seed{args.seed}__n{args.alignment_samples}.json', dict(schema=SCHEMA,
                        identity=census_identity, student=args.student, dataset=task, run=str(run),
                        alignment=alignment))
                    publish(out/f'{task}__alignment__{run.name}.json', dict(schema=SCHEMA,
                        identity=census_identity, student=args.student, dataset=task, run=str(run),
                        alignment=alignment))
                print('CENSUS_TASK_COMPLETE', args.student, task, record['coverage'], flush=True)
        publish(out/'state.json', dict(schema=SCHEMA, status='complete', stage=args.stage,
            run=str(run), new_gradients=cache.computed_windows, cache=str(cache.path),
            elapsed_seconds=time.monotonic()-started, benchmark=cache.gradient_benchmark(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0))
    except BaseException as error:
        publish(out/'state.json', dict(schema=SCHEMA, status='failed', stage=args.stage,
            run=str(run), error=repr(error), cache=str(cache.path),
            new_gradients=cache.computed_windows, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        cache.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan'); plan.add_argument('--runs', nargs='+', required=True); plan.add_argument('--out', required=True)
    worker = sub.add_parser('compute')
    worker.add_argument('--run', required=True); worker.add_argument('--student', required=True); worker.add_argument('--out', required=True)
    worker.add_argument('--stage', choices=['references', 'alignment', 'all'], default='all')
    worker.add_argument('--chunk', type=int, default=1024); worker.add_argument('--gradient-batch', type=int)
    worker.add_argument('--gradient-workers',type=int);worker.add_argument('--memory-fraction',type=float)
    worker.add_argument('--cpu-threads',type=int)
    worker.add_argument('--tune-batches',type=int,nargs='+');worker.add_argument('--tune-workers',type=int,nargs='+')
    worker.add_argument('--alignment-samples', type=int, default=2048); worker.add_argument('--seed', type=int, default=20260921)
    args = p.parse_args()
    if args.command == 'plan':
        result = inventory(args.runs); publish(args.out, result)
        print(json.dumps(result, indent=2))
    else:
        if args.chunk < 1 or args.alignment_samples < 2:
            p.error('positive chunk and at least two source alignment samples required')
        if any(x<1 for x in (args.tune_batches or [])+(args.tune_workers or [])):
            p.error('execution tuning candidates must be positive')
        compute(args)


if __name__ == '__main__':
    main()
