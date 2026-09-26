"""Adopt nonblocking queues at a durable cell boundary, using inotify/pidfd."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import time
from scripts.supervise_v7_night import pidfd
from assets.gradient.cache import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--student', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--extra-chronos-gpu1', action='store_true')
    a = p.parse_args()
    run = Path(a.run).resolve()
    words = Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().split('\0')
    assert 'scripts.run_v7' in words
    for flag,value in [('--student',a.student),('--phase','train'),('--seed','1'),('--run',str(run))]:
        assert words[words.index(flag)+1] == value
    env = dict(item.split('=',1) for item in Path(f'/proc/{a.pid}/environ').read_bytes().decode().split('\0') if '=' in item)
    root = run/'seeds/seed1'/a.student
    cells = root/'cells'
    libc = ctypes.CDLL(None,use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    assert fd >= 0 and libc.inotify_add_watch(fd,os.fsencode(cells),0x80) >= 0
    process_fd = pidfd(a.pid)
    baseline = {x.name for x in cells.glob('*.json')}
    receipt = run/'logs'/f'queue_migration_{a.pid}.json'
    atomic_json(receipt,dict(state='armed',pid=a.pid,gpu=a.gpu,student=a.student))
    finished_cell = None
    with selectors.DefaultSelector() as selector:
        selector.register(fd,selectors.EVENT_READ)
        selector.register(process_fd,selectors.EVENT_READ)
        while finished_cell is None:
            events = selector.select()
            if any(key.fd == process_fd for key,_ in events):
                break
            data = os.read(fd,65536)
            offset = 0
            while offset < len(data):
                _,_,_,length = struct.unpack_from('iIII',data,offset)
                name = data[offset+16:offset+16+length].split(b'\0')[0].decode()
                offset += 16+length
                if name in baseline or not name.endswith('.json'):
                    continue
                row = json.loads((cells/name).read_text())
                assert row['student'] == a.student and row['seed'] == 1
                assert Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().split('\0') == words
                os.kill(a.pid,signal.SIGTERM)
                finished_cell = cells/name
                break
        selector.unregister(fd)
        # Process exit releases all legacy locks before admitting new workers.
        selector.select()
    os.close(fd)
    os.close(process_fd)
    if finished_cell:
        from rags.experiments.checkpoint_retention import cleanup_completed_weights
        row = json.loads(finished_cell.read_text())
        cleanup_completed_weights(finished_cell,root/'checkpoints'/row['task']/row['arm'])
    config = json.loads(Path(a.config).read_text())
    env.update(PYTHONPATH='src:.',TS3_GRADIENT_CACHE_ROOT='/root/autodl-tmp/TS3/cache/grad-pool')
    def launch(student,gpu,label):
        python = config['student_environments'][student]
        log = (run/'logs'/f'queue_{label}_{student}.log').open('a')
        return subprocess.Popen([python,'-m','rags.experiments.v7_queue','--config',a.config,
            '--run',str(run),'--student',student],cwd='/root/TS3',
            env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    worker = launch(a.student,a.gpu,'gpu'+a.gpu)
    extra = launch('chronos-base','1','gpu1_fill') if a.extra_chronos_gpu1 else None
    atomic_json(receipt,dict(state='queue_started',old_pid=a.pid,pid=worker.pid,
        extra_pid=extra.pid if extra else None,gpu=a.gpu,time=time.time()))
    rc = worker.wait()
    if rc:
        raise RuntimeError(f'Queue worker failed rc={rc}; no automatic retry')
    other = 'timesfm-2.5' if a.student == 'chronos-base' else 'chronos-base'
    following = launch(other,a.gpu,'gpu'+a.gpu+'_fill')
    atomic_json(receipt,dict(state='next_model_queue',student=other,pid=following.pid,gpu=a.gpu))
    rc = following.wait()
    extra_rc = extra.wait() if extra else 0
    atomic_json(receipt,dict(state='finished',rc=rc,extra_rc=extra_rc,gpu=a.gpu))
    if rc or extra_rc:
        raise RuntimeError('Queue failed; inspect logs before recovery')


if __name__ == '__main__':
    main()
