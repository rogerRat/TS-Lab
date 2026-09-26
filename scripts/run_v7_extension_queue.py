"""Event-driven approved task queue: sparse assets -> select -> free cell workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from assets.gradient.cache import atomic_json
from scripts.supervise_v7_night import pidfd
from rags.experiments.v7_run import gate, read, freeze, folder
from rags.experiments.v7_extension import CANDIDATES, make_config, prepare
from rags.experiments.v7_queue import claim


def complete(config,run,model):
    return all((folder(run,model,s)/'cells'/f'{t}__{a}.json').is_file()
               for s in config['seeds'] for t in config['datasets'] for a in config['arms'])


def available(config,run,model):
    for seed in config['seeds']:
        if any(not (folder(run,model,p)/'cells'/f'{t}__{a}.json').is_file()
               for p in config['seeds'] if p < seed for t in config['datasets'] for a in config['arms']):
            return False
        for task in config['datasets']:
            root = folder(run,model,seed)
            for arm in config['arms']:
                if arm=='orig' or (root/'cells'/f'{task}__{arm}.json').exists():
                    continue
                if arm!='rand' and not (root/'cells'/f'{task}__rand.json').exists():
                    continue
                with claim(run/'cell_queue_locks'/model/f'seed{seed}'/f'{task}__{arm}.lock') as free:
                    if free:
                        return True
    return False


def finalize_task(config,path):
    run=Path(config['run_root'])
    if not all(complete(config,run,m) for m in config['students']):
        raise RuntimeError('Cannot finalize incomplete task cells')
    gate(config,run,'assets')
    gate(config,run,'select')
    # Validate full curves/identity, not just counts, before entering next task.
    from rags.experiments.v7_amendment import report_view
    from rags.experiments.v7_run import artifact_identity
    for m in config['students']:
        identity=read(run/'models'/f'{m}.json')
        for s in config['seeds']:
            manifest=read(run/'inputs'/f'seed{s}_manifest.json')
            for t in config['datasets']:
                expected=dict(artifact_identity(config,m,s,t,manifest,identity),training_protocol=config['training_protocol'])
                for arm in config['arms']:
                    report_view(read(folder(run,m,s)/'cells'/f'{t}__{arm}.json'),expected,config,m)
    subprocess.run([sys.executable,'-m','scripts.report_v7_costs','--config',str(path),'--run',str(run)],check=True)
    expected = len(config['arms']) * len(config['students']) * len(config['seeds'])
    atomic_json(run/'task_queue_complete.json',dict(complete=True,cells=expected,
                                                    seeds=[0,1],time=time.time()))


def wait_external_claim(config,run):
    """A claimed last cell may be owned by a separately admitted GPU worker."""
    receipt=Path(run)/'logs/gpu0_extra_chronos_pid.json'
    if not receipt.is_file():
        return False
    row=read(receipt)
    pid=int(row['pid'])
    if row.get('model')!='chronos-base' or row.get('gpu')!=0:
        raise ValueError('External worker receipt differs')
    try:
        fd=pidfd(pid)
    except ProcessLookupError:
        return False
    try:
        words=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        expected=[('--student','chronos-base'),('--run',str(run))]
        if 'rags.experiments.v7_queue' not in words or any(
                flag not in words or words[words.index(flag)+1]!=value for flag,value in expected):
            raise RuntimeError('External worker PID identity differs')
        with selectors.DefaultSelector() as watcher:
            watcher.register(fd,selectors.EVENT_READ)
            watcher.select()
    finally:
        os.close(fd)
    return True


def run_task(config,path,adopt=False):
    run=Path(config['run_root']); logs=run/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH='src:.',TS3_GRADIENT_CACHE_ROOT=str(run/'gradient_cache'))
    models=config['students']; gpu_for={'chronos-base':'0','timerxl-84m':'1','timesfm-2.5':'1'}
    active={}; sequence=0
    selector=selectors.DefaultSelector()
    def register(process,model,phase,slot):
        fd=pidfd(process.pid)
        selector.register(fd,selectors.EVENT_READ)
        active[fd]=(process,model,phase,slot)
    def launch(model,phase,slot):
        nonlocal sequence
        sequence+=1
        if phase in ('assets','select'):
            command=['-m','scripts.run_v7','--phase',phase]
        else:
            command=['-m','rags.experiments.v7_queue']
            if phase=='initialize':command+=['--initialize-only']
        command += ['--config',str(path),'--run',str(run),'--student',model]
        log=(logs/f'auto_{sequence}_{phase}_{model}_gpu{slot[0]}.log').open('a')
        process=subprocess.Popen([config['student_environments'][model],*command],cwd='/root/TS3',
            env=dict(env,CUDA_VISIBLE_DEVICES=slot[0]),stdout=log,stderr=subprocess.STDOUT)
        register(process,model,phase,slot)
        atomic_json(logs/f'auto_{sequence}_pid.json',dict(pid=process.pid,phase=phase,model=model,gpu=slot[0]))
    slots=[('0',0),('1',0),('1',1)]
    # First task's already-running asset processes are adopted without relaunch.
    if adopt:
        for model in models:
            row=read(logs/f'assets__{model}__pid.json');pid=row['pid']
            try:
                fd=pidfd(pid)
            except ProcessLookupError:
                continue
            words=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            assert words[words.index('--student')+1]==model and words[words.index('--phase')+1]=='assets'
            assert words[words.index('--run')+1]==str(run)
            selector.register(fd,selectors.EVENT_READ)
            active[fd]=(None,model,'assets',None)
    else:
        for model,slot in zip(models,[slots[0],slots[1],slots[2]]):launch(model,'assets',slot)
    while active:
        for key,_ in selector.select():
            process,model,phase,slot=active.pop(key.fd)
            selector.unregister(key.fd);os.close(key.fd)
            if process is not None and process.wait()!=0:raise RuntimeError(f'{model} assets failed')
    gate(config,run,'assets')
    status={m:'select' for m in models}
    for model,slot in zip(models,[slots[0],slots[1],slots[2]]):launch(model,'select',slot)
    while active:
        for key,_ in selector.select():
            process,model,phase,slot=active.pop(key.fd)
            selector.unregister(key.fd);os.close(key.fd)
            if process.wait()!=0:raise RuntimeError(f'{model}/{phase} failed; see auto logs')
            if phase=='select':
                gate(config,run,'select',student=model)
                status[model]='initialize';launch(model,'initialize',slot)
            elif phase=='initialize':
                status[model]='ready'
        busy={row[3] for row in active.values()}
        for slot in slots:
            if slot in busy:continue
            preferred=['chronos-base','timesfm-2.5','timerxl-84m'] if slot[0]=='0' else ['timerxl-84m','timesfm-2.5','chronos-base']
            for model in preferred:
                if status[model]=='ready' and not complete(config,run,model) and available(config,run,model):
                    launch(model,'train',slot)
                    break
    selector.close()
    if not all(complete(config,run,m) for m in models):
        if not wait_external_claim(config,run) or not all(complete(config,run,m) for m in models):
            raise RuntimeError('Queue drained before all task cells completed')
    finalize_task(config,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True)
    p.add_argument('--adopt-first-assets',action='store_true')
    args=p.parse_args()
    parent=Path(args.parent).resolve()
    lock=(parent/'extension_queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=read('configs/rags/main_v7_curve4.json')
    for i,task in enumerate(CANDIDATES):
        name='ne-china-wind-20260925' if i==0 else task['id']+'-20260925'
        run=parent/'extensions'/name
        config=make_config(base,parent,run,task)
        path=Path('configs/rags')/('v7_extension_ne_china_wind.json' if i==0 else 'v7_extension_'+task['id']+'.json')
        freeze(path,config)
        if (run/'task_queue_complete.json').exists():
            if not all(complete(config,run,m) for m in config['students']):
                raise RuntimeError('Task completion marker exists but native cells are missing')
            continue
        if all(complete(config,run,m) for m in config['students']):
            # Recovery after an external worker finished the last claimed cell.
            # No asset/selection/model work is repeated for this task.
            finalize_task(config,path)
            continue
        atomic_json(parent/'extension_queue_status.json',dict(task=task['id'],state='preparing_inputs',position=i+1,total=len(CANDIDATES)))
        prepare(config)
        if not (run/'task_queue_complete.json').exists():
            atomic_json(parent/'extension_queue_status.json',dict(task=task['id'],state='running',position=i+1,total=len(CANDIDATES)))
            run_task(config,path,adopt=i==0 and args.adopt_first_assets)
    atomic_json(parent/'extension_queue_status.json',dict(state='approved_queue_complete',tasks=len(CANDIDATES),new_cells=360))


if __name__=='__main__':
    main()
