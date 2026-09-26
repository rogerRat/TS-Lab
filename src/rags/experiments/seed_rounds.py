"""Run exactly seeds 0,1,2 with a whole-matrix barrier and parallel models per round."""
import argparse
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path

from .students import atomic_json
from .protocol import validate_config


def round_config(template, seed):
    config=deepcopy(template)
    config.update(seeds=[seed],batch=f"{template['batch']}-seed{seed}")
    return config


def round_completion(config, folder):
    missing=[]
    version=config.get('experiment_version','v5')
    for student in config['students']:
        for dataset in config['datasets']:
            for arm in config['arms']:
                seed=0 if arm=='orig' else config['seeds'][0]
                p=Path(folder)/'students'/student/'cells'/f'{dataset}__{arm}__seed{seed}.json'
                try:
                    record=json.loads(p.read_text())
                    if any(record.get(k)!=v for k,v in dict(student=student,dataset=dataset,
                            arm=arm,seed=seed,experiment_version=version).items()):
                        raise ValueError('Cell identity differs')
                    if not math.isfinite(float(record['mase'])):
                        raise ValueError('Nonfinite MASE')
                except (OSError,ValueError,KeyError,TypeError):
                    missing.append(str(p))
    return dict(complete=not missing,missing=missing)


def prepare_round(template, parent, seed, control):
    config=round_config(template,seed);folder=parent/'rounds'/f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    frozen=folder/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=config:
        raise ValueError('Frozen seed round changed; use a new V5 suite directory')
    atomic_json(frozen,config)
    atomic_json(folder/'execution_control.json',dict(control,round_seed=seed))
    cache=folder/'grad_cache'
    source=Path(template.get('gradient_cache_source') or parent/'grad_cache').resolve()
    if not template.get('gradient_cache_source'):source.mkdir(parents=True,exist_ok=True)
    if not source.is_dir():
        raise FileNotFoundError(f'Gradient cache source missing: {source}')
    if cache.exists() or cache.is_symlink():
        if cache.resolve()!=source:
            raise ValueError('Seed-round gradient cache points to another source')
    else:
        cache.symlink_to(source,target_is_directory=True)
    # Static distributions and seed-independent orig measurements are published
    # by the completed seed-0 round. Later rounds only vary training/draw RNG.
    if seed>0:
        first=parent/'rounds'/'seed0'
        for student in config['students']:
            out=folder/'students'/student;out.mkdir(parents=True,exist_ok=True)
            selections=out/'selections';origin=first/'students'/student/'selections'
            if not selections.exists():
                selections.symlink_to(origin,target_is_directory=True)
            cells=out/'cells';cells.mkdir(exist_ok=True)
            for dataset in config['datasets']:
                name=f'{dataset}__orig__seed0.json';destination=cells/name
                if not destination.exists():
                    destination.symlink_to(first/'students'/student/'cells'/name)
    return config,folder


def run_rounds(template, parent, *, max_seeds=None, runner=None):
    from .full_pool_run import run_suite
    from .full_pool_worker import _assert_training_enabled
    validate_config(template)
    if template['seeds']!=[0,1,2]:
        raise ValueError('V5 suite template must use exactly seeds 0,1,2')
    max_seeds = 3 if max_seeds is None else max_seeds
    if type(max_seeds) is not int or not 1<=max_seeds<=3:
        raise ValueError('max_seeds must be within the fixed three-seed protocol')
    parent=Path(parent).resolve();parent.mkdir(parents=True,exist_ok=True)
    runner=runner or run_suite
    with (parent/'suite.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=parent/'protocol.json'
        if frozen.exists() and json.loads(frozen.read_text())!=template:
            raise ValueError('V5 suite configuration changed; use a new directory')
        atomic_json(frozen,template)
        seed=0
        while seed<max_seeds:
            _assert_training_enabled(parent,experiment_version=template.get('experiment_version','v5'))
            control=json.loads((parent/'execution_control.json').read_text())
            config,folder=prepare_round(template,parent,seed,control)
            status=round_completion(config,folder)
            if not status['complete']:
                atomic_json(parent/'state.json',dict(seed=seed,status='running',completed_seeds=list(range(seed))))
                try:
                    code=runner(config,folder,execution=dict(student_parallelism=2))
                    status=round_completion(config,folder)
                    if code not in (None,0) or not status['complete']:
                        raise RuntimeError(f'Seed {seed} incomplete; next seed blocked: {status["missing"]}')
                except Exception as error:
                    atomic_json(parent/'state.json',dict(seed=seed,status='failed',error=repr(error),
                        completed_seeds=list(range(seed))))
                    raise
            atomic_json(parent/'state.json',dict(seed=seed,status='round_complete',
                completed_seeds=list(range(seed+1)),next_seed=seed+1))
            print(f'V5 SEED_ROUND_COMPLETE seed={seed}; next_seed={seed+1}',flush=True)
            seed+=1
        atomic_json(parent/'state.json',dict(status='complete',completed_seeds=list(range(seed)),
            next_seed=None if seed==3 else seed))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--max-seeds',type=int,choices=[3],default=3,help='Fixed seeds 0,1,2')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());validate_config(config)
    if args.dry_run:
        print(json.dumps(dict(students=config['students'],seed_schedule=config['seed_schedule'],
            methods=config['arms'],cells_per_first_round=234,training_cells_per_seed=216,
            max_seeds=args.max_seeds,jobs_launched=0),indent=2));return 0
    return run_rounds(config,args.run,max_seeds=args.max_seeds)


if __name__=='__main__':
    raise SystemExit(main())
