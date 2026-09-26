"""Event-gated, two-seed shape-arm supplement after all three V7 extensions."""
import argparse
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time

from assets.gradient.cache import atomic_json
from rags.experiments.v7_amendment import report_view
from rags.experiments.v7_protocol import MODELS, validate
from rags.experiments.v7_run import artifact_identity, read
from rags.experiments.v7_shape import ARMS
from rags.experiments.v7_shape_backfill import EXTENSIONS, checked_model_lease, publish_receipt, run_model
from scripts.supervise_v7_night import pidfd

CONFIGS = ('configs/rags/main_v7_curve4.json',
           'configs/rags/v7_extension_ne_china_wind.json',
           'configs/rags/v7_extension_time_sg_pm25_h.json',
           'configs/rags/v7_extension_time_coastal_ts_h.json')


def sources(parent):
    parent = Path(parent).resolve()
    runs = [parent, *(parent/'extensions'/name for name in EXTENSIONS)]
    configs = [read(path) for path in CONFIGS]
    for run, config in zip(runs, configs):
        validate(config)
        if Path(config['run_root']).resolve() != run:
            raise ValueError('V7 shape source run/config identity differs')
    return runs, configs


def complete(parent, runs, configs):
    first = runs[0]
    for model in MODELS:
        for seed in (0,1):
            for task in configs[0]['datasets']:
                for arm in configs[0]['arms']:
                    if not (first/f'seeds/seed{seed}'/model/'cells'/f'{task}__{arm}.json').is_file():
                        return False
    for run in runs[1:]:
        marker = run/'task_queue_complete.json'
        if not marker.is_file():
            return False
        row = read(marker)
        if row.get('complete') is not True or row.get('cells') != 120 or row.get('seeds') != [0,1]:
            return False
    return True


def wait_queue(queue_pid, parent, runs, configs):
    if complete(parent,runs,configs):
        return
    if queue_pid is None:
        raise RuntimeError('Extensions incomplete; queue PID is required for event wait')
    args = Path(f'/proc/{queue_pid}/cmdline').read_bytes().decode().split('\0')
    if ('scripts.run_v7_extension_queue' not in args or '--parent' not in args or
            Path(args[args.index('--parent')+1]).resolve() != Path(parent).resolve()):
        raise ValueError('Extension queue PID identity differs')
    fd = pidfd(queue_pid)
    try:
        with selectors.DefaultSelector() as watcher:
            watcher.register(fd, selectors.EVENT_READ)
            watcher.select()
    finally:
        os.close(fd)
    if not complete(parent,runs,configs):
        raise RuntimeError('Extension queue exited without all native cells; no shape start')


def arm_complete(runs, configs, arm):
    """Global cross-model/seed/task barrier; files alone are not sufficient."""
    if arm not in ARMS:
        raise ValueError('Unknown shape arm')
    main = runs[0]
    for parent, config in zip(runs, configs):
        sidecar = parent/'shape_backfill_20260925'
        side_config_path = sidecar/'shape_config.json'
        if not side_config_path.is_file():
            return False
        side_config = read(side_config_path)
        for model in MODELS:
            model_path = main/'models'/f'{model}.json'
            if not model_path.is_file():
                return False
            identity = read(model_path)
            for seed in (0, 1):
                manifest_path = parent/'inputs'/f'seed{seed}_manifest.json'
                if not manifest_path.is_file():
                    return False
                manifest = read(manifest_path)
                for task in config['datasets']:
                    path = sidecar/'seeds'/f'seed{seed}'/model/'cells'/f'{task}__{arm}.json'
                    if not path.is_file():
                        return False
                    expected = dict(artifact_identity(side_config,model,seed,task,manifest,identity),
                                    training_protocol=side_config['training_protocol'])
                    result = report_view(read(path),expected,side_config,model)
                    if (result.get('arm') != arm or result.get('seed') != seed or
                            result.get('task') != task):
                        raise ValueError('Completed shape arm cell identity differs')
    return True


def accept_arm_phase(parent, runs, configs, arm):
    """Persist a verified global barrier, including when it predates restart."""
    if not arm_complete(runs, configs, arm):
        raise RuntimeError('Shape arm phase ended without every accepted cell: '+arm)
    tasks = sum(len(config['datasets']) for config in configs)
    expected = dict(arm=arm,complete=True,models=list(MODELS),tasks=tasks,
                    seeds=[0,1],cells=tasks*len(MODELS)*2)
    marker = Path(parent)/'shape_backfill_20260925'/'arm_phases'/f'{arm}.json'
    if marker.is_file():
        old = read(marker)
        if any(old.get(key) != value for key,value in expected.items()):
            raise ValueError('Existing shape arm phase receipt differs: '+str(marker))
    else:
        publish_receipt(marker,dict(expected,time=time.time()))
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True)
    parser.add_argument('--queue-pid',type=int)
    parser.add_argument('--student',choices=MODELS)
    parser.add_argument('--arm',choices=ARMS)
    args = parser.parse_args()
    parent = Path(args.parent).resolve()
    runs,configs = sources(parent)
    if args.student:
        if args.arm is None:
            raise ValueError('Shape worker requires explicit arm phase')
        if not complete(parent,runs,configs):
            raise RuntimeError('Shape worker cannot start before ten-task source barrier')
        with checked_model_lease(parent,args.student):
            run_model(parent,runs,configs,args.student,args.arm)
        return 0
    if args.arm is not None:
        raise ValueError('Only a model worker may specify --arm')
    status = parent/'shape_backfill_20260925'/'queue_status.json'
    status.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(status,dict(state='waiting_for_extensions',queue_pid=args.queue_pid,time=time.time()))
    wait_queue(args.queue_pid,parent,runs,configs)
    free = shutil.disk_usage(parent).free
    if free < 75*2**30:
        raise RuntimeError(f'Shape backfill needs 75GiB conservative headroom; free={free}')
    atomic_json(status,dict(state='starting_arm_phases',free_bytes=free,time=time.time()))
    mapping={'chronos-base':'0','timerxl-84m':'1','timesfm-2.5':'1'}
    for arm in ARMS:
        if arm_complete(runs,configs,arm):
            accept_arm_phase(parent,runs,configs,arm)
            continue
        active={}
        with selectors.DefaultSelector() as watcher:
            for model in MODELS:
                log=status.parent/'logs'/f'{model}.log'
                log.parent.mkdir(exist_ok=True)
                command=[configs[0]['student_environments'][model],'-m',
                         'scripts.run_v7_shape_backfill','--parent',str(parent),
                         '--student',model,'--arm',arm]
                stream=log.open('a')
                process=subprocess.Popen(command,cwd='/root/TS3',
                    env=dict(os.environ,PYTHONPATH='src:.',CUDA_VISIBLE_DEVICES=mapping[model]),
                    stdout=stream,stderr=subprocess.STDOUT)
                stream.close()
                fd=pidfd(process.pid)
                watcher.register(fd,selectors.EVENT_READ)
                active[fd]=(process,model)
            atomic_json(status,dict(state='running',arm=arm,
                workers={m:p.pid for p,m in active.values()},gpu=mapping,time=time.time()))
            errors=[]
            while active:
                for key,_ in watcher.select():
                    process,model=active.pop(key.fd)
                    watcher.unregister(key.fd);os.close(key.fd)
                    rc=process.wait()
                    if rc:
                        errors.append(dict(model=model,exit_code=rc))
                    atomic_json(status,dict(state='running' if active else 'finishing',arm=arm,
                        active={m:p.pid for p,m in active.values()},errors=errors,time=time.time()))
        if errors:
            atomic_json(status,dict(state='failed',arm=arm,errors=errors,time=time.time()))
            raise RuntimeError('Shape worker failed: '+str(errors))
        accept_arm_phase(parent,runs,configs,arm)
        atomic_json(status,dict(state='arm_complete',arm=arm,cells=60,time=time.time()))
    atomic_json(status,dict(state='complete',models=list(MODELS),tasks=10,
                            seeds=[0,1],new_cells=240,time=time.time()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
