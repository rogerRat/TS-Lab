"""Nonblocking V7 cell queue, preserving immutable selection/training identities."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time


@contextmanager
def claim(path, *, shared=False):
    """A busy claim is skipped, never waited on; lock files are never removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def cell_config(config, arm):
    # asset_protocol and training_protocol are immutable scientific identities.
    # orig/rand are required by native report generation, not extra training.
    if 'asset_protocol' not in config or arm not in config['arms'] or arm == 'orig':
        raise ValueError('Queue requires a frozen V7 trained arm')
    subset = deepcopy(config)
    subset['arms'] = list(dict.fromkeys(('orig', 'rand', arm)))
    return subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rags/main_v7_curve4.json')
    parser.add_argument('--run', required=True)
    parser.add_argument('--student', required=True)
    parser.add_argument('--initialize-only', action='store_true',help='Prepare shared original evaluations, no training')
    args = parser.parse_args()
    from .v7_run import read, freeze, folder, gate, seed_pool, artifact_identity
    from .v7_protocol import validate, WINDOWS, asset_protocol, MODELS
    from .v7_amendment import freeze_training, compatible_identity
    from .v7_train_leases import model_lease
    from .v7_train import train_task
    from .full_pool_models import build_model
    from .development_inputs import load_fragments
    from .arms import WindowIndex
    from assets.gradient.cache import atomic_json
    import os
    config = read(args.config)
    validate(config)
    if args.student not in MODELS:
        parser.error('Unknown student')
    run = Path(args.run).resolve()
    limits = read(run/'execution_limits.json')['allowed_training_seeds']
    if any(s not in (0, 1) for s in limits):
        raise ValueError('This queue is limited to user-approved seeds 0/1')
    freeze(run/'protocol.json', asset_protocol(config))
    freeze_training(run, config)
    gate(config, run, 'assets')
    gate(config, run, 'select', student=args.student)
    c, h = WINDOWS[args.student]
    # Shared model lease excludes asset writers. Legacy task workers take EX;
    # queue workers take SH then an individual EX cell claim, all nonblocking.
    with model_lease(run, args.student, 'train', parallel_train=True):
        engine = identity = base = feasible = None
        executed = 0
        def load_engine():
            nonlocal engine, identity, base, feasible
            if engine is None:
                engine, probe, identity = build_model(args.student,
                    model_path=config['model_paths'].get(args.student),context_length=c,prediction_length=h)
                freeze(run/'models'/f'{args.student}.json',identity)
                base = {k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
                feasible = load_fragments(run/'inputs/feasible.npz')
        for seed in sorted(limits):
            root = folder(run, args.student, seed)
            if not args.initialize_only and seed and any(not (folder(run,args.student,prior)/'cells'/f'{t}__{a}.json').is_file()
                            for prior in sorted(limits) if prior < seed
                            for t in config['datasets'] for a in config['arms']):
                print('QUEUE_PRIOR_SEED_PENDING', args.student, seed, flush=True)
                break
            index = manifest = None
            for task in config['datasets']:
                if args.initialize_only:
                    task_lock = run/'training_locks'/args.student/f'seed{seed}'/f'{task}.lock'
                    with claim(task_lock) as free:
                        if not free:
                            raise RuntimeError('Original initialization collided with a training task')
                        load_engine()
                        if index is None:
                            pool,manifest = seed_pool(run,seed,feasible)
                            index = WindowIndex(pool,context_length=c,prediction_length=h)
                        engine.model.load_state_dict(base)
                        train_task(config,run,args.student,seed,task,engine,index,manifest,identity,initialize_only=True)
                    continue
                # Existing seven-task extension only: pretrained evaluation must
                # already be durable; do not race to create shared originals.
                if not (root/'cells'/f'{task}__orig.json').is_file():
                    raise RuntimeError('Prepare original evaluation before queue admission: '+task)
                for arm in config['arms']:
                    if arm == 'orig':
                        continue
                    cell = root/'cells'/f'{task}__{arm}.json'
                    if cell.exists():
                        continue
                    if arm != 'rand' and not (root/'cells'/f'{task}__rand.json').is_file():
                        continue
                    task_lock = run/'training_locks'/args.student/f'seed{seed}'/f'{task}.lock'
                    with claim(task_lock, shared=True) as task_free:
                        if not task_free:
                            continue
                        with claim(run/'cell_queue_locks'/args.student/f'seed{seed}'/f'{task}__{arm}.lock') as free:
                            if not free or cell.exists():
                                continue
                            started = time.time()
                            state = run/'logs'/f'queue_{os.getpid()}.json'
                            atomic_json(state,dict(status='claimed', student=args.student,
                                seed=seed,task=task,arm=arm,pid=os.getpid(),started=started))
                            load_engine()
                            if index is None:
                                pool, manifest = seed_pool(run, seed, feasible)
                                index = WindowIndex(pool,context_length=c,prediction_length=h)
                            expected = dict(artifact_identity(config,args.student,seed,task,manifest,identity),
                                training_protocol=config['training_protocol'])
                            for baseline in ('orig', 'rand'):
                                path = root/'cells'/f'{task}__{baseline}.json'
                                if path.exists() and not compatible_identity(read(path).get('identity'),expected,config):
                                    raise ValueError('Queue baseline identity differs: '+str(path))
                            # Native train_task snapshots its base; restore the same
                            # pretrained weights before every independently claimed cell.
                            engine.model.load_state_dict(base)
                            train_task(cell_config(config,arm),run,args.student,seed,task,
                                       engine,index,manifest,identity)
                            if not cell.exists() or not compatible_identity(read(cell).get('identity'),expected,config):
                                raise RuntimeError('Claim did not produce a valid native cell')
                            executed += 1
                            atomic_json(state,dict(status='complete',student=args.student,
                                seed=seed,task=task,arm=arm,pid=os.getpid(),elapsed_seconds=time.time()-started))
                            print('QUEUE_CELL_COMPLETE',args.student,seed,task,arm,flush=True)
        print('QUEUE_PASS_COMPLETE',args.student,'executed',executed,flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
