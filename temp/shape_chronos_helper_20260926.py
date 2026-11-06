"""Scoped GPU1 Chronos fill: seed1 extension tasks of current linear arm only.

The old full Chronos worker is still on the earlier seven tasks. A separate
inotify/pidfd guard terminates this helper if it has not finished by the old
worker's seed1 solar completion, one full main task before the extension set.
"""
from pathlib import Path
import sys

from rage.experiments.v7_run import read
from rage.experiments.v7_shape_backfill import run_model
from scripts.run_v7_shape_backfill import complete, sources

ROOT = Path('/root/autodl-tmp/TS3/outputs/main-v7-20260923')
TASKS = {'time_ne_china_wind_h', 'time_sg_pm25_h', 'time_coastal_ts_h'}
ARM = 'rage-shape-linearSum'


def main():
    runs, configs = sources(ROOT)
    if not complete(ROOT, runs, configs):
        raise RuntimeError('Original ten-task source barrier differs')
    if set().union(*(set(c['datasets']) for c in configs[1:])) != TASKS:
        raise RuntimeError('Scoped extension task identity differs')
    current = read(ROOT/'shape_backfill_20260925/queue_status.json')
    if current.get('state') != 'running' or current.get('arm') != ARM:
        raise RuntimeError('Only the active linear arm may be supplemented')
    for parent, config in zip(runs[1:], configs[1:]):
        task = config['datasets'][0]
        for seed in (0,1):
            receipt = (parent/'shape_backfill_20260925'/'seeds'/f'seed{seed}'/
                       'chronos-base'/'shape_select'/f'{task}.json')
            row = read(receipt)
            if row.get('complete') is not True or row.get('selection_new_bp') != 0:
                raise RuntimeError('Scoped helper requires verified zero-BP selection')
    run_model(ROOT, runs, configs, 'chronos-base', ARM,
              seed_filter=1, task_filter=TASKS)
    print('SCOPED_CHRONOS_LINEAR_EXTENSION_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
