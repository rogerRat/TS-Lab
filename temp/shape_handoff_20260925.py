"""One-shot event handoff from old imported shape modules to patched queue."""
import json
import os
from pathlib import Path
import selectors
import sys
import time

from assets.gradient.cache import atomic_json
from scripts.supervise_v7_night import pidfd

RUN = Path('/root/autodl-tmp/TS3/outputs/main-v7-20260923')
WATCH = {
    586713: ('scripts.run_v7_shape_backfill', None),
    586779: ('scripts.run_v7_shape_backfill', 'chronos-base'),
    586780: ('scripts.run_v7_shape_backfill', 'timesfm-2.5'),
    599146: ('scripts.run_v7_shape_backfill', 'timerxl-84m'),
}


def main():
    finished = []
    with selectors.DefaultSelector() as selector:
        for pid, (module, model) in WATCH.items():
            try:
                fd = pidfd(pid)
            except ProcessLookupError:
                finished.append(pid)
                continue
            try:
                words = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            except FileNotFoundError:
                os.close(fd)
                finished.append(pid)
                continue
            if module not in words or '--parent' not in words or Path(words[words.index('--parent')+1]).resolve() != RUN:
                raise RuntimeError(f'Unexpected shape PID identity: {pid}')
            if model is None:
                if '--student' in words:
                    raise RuntimeError('Original supervisor PID is a worker')
            elif '--student' not in words or words[words.index('--student')+1] != model or '--arm' not in words or words[words.index('--arm')+1] != 'rags-shape-expSum':
                raise RuntimeError(f'Unexpected shape worker model/arm: {pid}')
            selector.register(fd, selectors.EVENT_READ, pid)
        while selector.get_map():
            for key, _ in selector.select():
                selector.unregister(key.fd)
                os.close(key.fd)
                finished.append(key.data)
                print('SHAPE_OLD_PID_EXITED', key.data, flush=True)
    if set(finished) != set(WATCH):
        raise RuntimeError('Handoff did not observe every original PID')
    atomic_json(RUN/'shape_backfill_20260925'/'handoff_287c4d6.json',
                dict(old_pids=finished, new_commit='287c4d6', event_wait=True,
                     state='resuming_original_identity', time=time.time()))
    print('SHAPE_HANDOFF_RESTARTING_QUEUE', flush=True)
    os.chdir('/root/TS3')
    env = dict(os.environ, PYTHONPATH='src:.')
    os.execvpe('/root/miniconda3/envs/ts3/bin/python',
               ['/root/miniconda3/envs/ts3/bin/python', '-m', 'scripts.run_v7_shape_backfill',
                '--parent', str(RUN)], env)


if __name__ == '__main__':
    sys.exit(main())
