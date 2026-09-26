"""Event-driven Chronos dual-GPU seed continuation; no polling or blind retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from rags.experiments.v7_costs import atomic_text


def pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    import platform
    if platform.machine() != 'x86_64':
        raise RuntimeError('pidfd syscall fallback requires x86_64')
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.syscall(434,pid,0)
    if fd<0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    return fd


def wait_all(pids, verify):
    # Acquire every descriptor before waiting: one worker may exit first.
    with selectors.DefaultSelector() as selector:
        handles=[]
        try:
            for pid in pids:
                fd=pidfd(pid);handles.append(fd)
                verify(pid)
                selector.register(fd,selectors.EVENT_READ)
            while selector.get_map():
                for key,_ in selector.select():
                    selector.unregister(key.fd)
        finally:
            for fd in handles:os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--seed0-pids',type=int,nargs=2,required=True)
    args=parser.parse_args()
    root=Path(args.run).resolve();config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text())
    status_path=root/'logs/codex_chronos_night.json'
    def status(**fields):
        atomic_text(status_path,json.dumps(dict(time=time.time(),**fields),indent=2))
    def verify(pid):
        words=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        for flag,value in [('--student','chronos-base'),('--phase','train'),
                           ('--seed','0'),('--run',str(root))]:
            if flag not in words or words[words.index(flag)+1]!=value:
                raise RuntimeError(f'PID {pid}: identity mismatch {flag}')
        if 'scripts.run_v7' not in words:raise RuntimeError('Not a V7 worker')
    def complete(seed):
        # Each exact configured cell is required; a raw glob count is insufficient.
        for task in config['datasets']:
            for arm in config['arms']:
                path=root/f'seeds/seed{seed}/chronos-base/cells/{task}__{arm}.json'
                row=json.loads(path.read_text())
                if (row.get('task'),row.get('arm'),row.get('seed'),row.get('student')) != (
                        task,arm,seed,'chronos-base'):
                    raise RuntimeError('Cell identity mismatch: '+str(path))
        # Native worker additionally checks full scientific identities before advancing.
    env=dict(os.environ,PYTHONPATH='src:.',
        TS3_GRADIENT_CACHE_ROOT='/root/autodl-tmp/TS3/cache/grad-pool')
    def command(seed):
        return [sys.executable,'-m','scripts.run_v7','--config',str(config_path),
            '--run',str(root),'--phase','train','--student','chronos-base',
            '--seed',str(seed),'--parallel-train']
    helper_tasks=[t for t in ('time_port_activity_d','time_solar_h','time_nem_price_5t')
                  if t in config['datasets']]
    def helper(seed):
        for task in helper_tasks:
            with (root/f'logs/chronos_night_seed{seed}_helper.log').open('a') as log:
                child=subprocess.Popen(command(seed)+['--train-task',task],
                    env=dict(env,CUDA_VISIBLE_DEVICES='1'),stdout=log,stderr=subprocess.STDOUT)
                atomic_text(root/f'logs/chronos_night_seed{seed}_helper.json',
                    json.dumps(dict(pid=child.pid,seed=seed,task=task)))
                rc=child.wait()
                if rc:raise RuntimeError(f'Helper seed{seed}/{task} failed rc={rc}')
    try:
        status(state='waiting_seed0_both',pids=args.seed0_pids)
        wait_all(args.seed0_pids,verify)
        complete(0)
        for seed in (1,2):
            with (root/f'logs/chronos_night_seed{seed}_main.log').open('a') as log:
                main_worker=subprocess.Popen(command(seed),env=dict(env,CUDA_VISIBLE_DEVICES='0'),
                    stdout=log,stderr=subprocess.STDOUT)
                status(state='training',seed=seed,main_pid=main_worker.pid)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future=pool.submit(helper,seed)
                    rc=main_worker.wait()
                    future.result()
                if rc:raise RuntimeError(f'Main seed{seed} failed rc={rc}')
            complete(seed)
            status(state='seed_complete',seed=seed)
        status(state='all_complete')
    except BaseException as error:
        status(state='failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
