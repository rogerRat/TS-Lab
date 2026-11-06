"""One-shot pidfd gate: power off only after the final Shape arm is accepted."""

import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from assets.gradient.cache import atomic_json
from rage.experiments.v7_run import read
from scripts.run_v7_shape_backfill import arm_complete, sources
from scripts.supervise_v7_night import pidfd


ROOT = Path('/root/autodl-tmp/TS3/outputs/main-v7-20260923')
SHAPE = ROOT / 'shape_backfill_20260925'
SUPERVISOR_PID = 601125
PENDING_V7_E_PID = 784280
RECEIPT = SHAPE / 'shutdown_after_max.json'
FINAL_ARM = 'rage-shape-maxSum'


def verify_supervisor():
    words = Path(f'/proc/{SUPERVISOR_PID}/cmdline').read_bytes().decode().split('\0')
    if ('scripts.run_v7_shape_backfill' not in words or '--parent' not in words or
            Path(words[words.index('--parent')+1]).resolve() != ROOT or
            '--student' in words):
        raise RuntimeError('Shape supervisor PID identity differs; refuse shutdown')


def accepted():
    marker = SHAPE / 'arm_phases' / f'{FINAL_ARM}.json'
    if not marker.is_file():
        raise RuntimeError('Final Shape arm phase marker is absent')
    row = read(marker)
    if (row.get('arm') != FINAL_ARM or row.get('complete') is not True or
            row.get('cells') != 60 or row.get('tasks') != 10 or
            row.get('seeds') != [0, 1] or
            set(row.get('models', [])) != {'chronos-base', 'timesfm-2.5', 'timerxl-84m'}):
        raise RuntimeError('Final Shape arm marker did not accept all 60 cells')
    status = read(SHAPE / 'queue_status.json')
    if status.get('state') != 'complete' or status.get('new_cells') != 240:
        raise RuntimeError('Shape queue did not finish all four arm phases')
    runs, configs = sources(ROOT)
    if not arm_complete(runs, configs, FINAL_ARM):
        raise RuntimeError('Final Shape cell identities/curves failed read-only acceptance')
    if Path(f'/proc/{PENDING_V7_E_PID}').exists():
        raise RuntimeError('V7-e pending queue is still alive; refuse shutdown race')
    trash = Path('/root/.local/share/Trash')
    if trash.is_symlink() or (trash.exists() and any(trash.iterdir())):
        raise RuntimeError('Platform shutdown wrapper would delete a nonempty Trash; refuse')
    return row


def main():
    handle = pidfd(SUPERVISOR_PID)
    verify_supervisor()
    atomic_json(RECEIPT, dict(state='armed', supervisor_pid=SUPERVISOR_PID,
        v7_e_queue_pid=PENDING_V7_E_PID, final_arm=FINAL_ARM, time=time.time(),
        shutdown_command='/bin/bash /usr/bin/shutdown'))
    print('SHAPE_SHUTDOWN_ARMED', SUPERVISOR_PID, flush=True)
    try:
        with selectors.DefaultSelector() as watcher:
            watcher.register(handle, selectors.EVENT_READ)
            watcher.select()  # Event-driven wait; no timer/poll loop.
    finally:
        os.close(handle)
    try:
        marker = accepted()
    except BaseException as error:
        atomic_json(RECEIPT, dict(state='aborted', reason=str(error),
            supervisor_pid=SUPERVISOR_PID, time=time.time()))
        print('SHAPE_SHUTDOWN_ABORTED', repr(error), flush=True)
        raise
    atomic_json(RECEIPT, dict(state='invoking_shutdown', final_arm=FINAL_ARM,
        accepted_cells=marker['cells'], supervisor_pid=SUPERVISOR_PID,
        v7_e_queue_cancelled=True, time=time.time()))
    print('SHAPE_SHUTDOWN_INVOKING', flush=True)
    subprocess.run(['/bin/bash', '/usr/bin/shutdown'], check=True)
    print('SHAPE_SHUTDOWN_COMMAND_RETURNED', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
