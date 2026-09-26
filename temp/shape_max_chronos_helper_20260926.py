"""GPU1 fill: only Chronos seed1 extension cells of the active Shape max arm."""
from pathlib import Path
import sys

from rags.experiments.v7_run import read
from rags.experiments.v7_shape_backfill import run_model
from scripts.run_v7_shape_backfill import complete, sources


ROOT = Path('/root/autodl-tmp/TS3/outputs/main-v7-20260923')
TASKS = {'time_ne_china_wind_h', 'time_sg_pm25_h', 'time_coastal_ts_h'}
ARM = 'rags-shape-maxSum'


def main():
    runs, configs = sources(ROOT)
    if not complete(ROOT, runs, configs):
        raise RuntimeError('Original V7 source/extension barrier differs')
    if set().union(*(set(config['datasets']) for config in configs[1:])) != TASKS:
        raise RuntimeError('Scoped extension task identity differs')
    status = read(ROOT/'shape_backfill_20260925/queue_status.json')
    if status.get('state') != 'running' or status.get('arm') != ARM:
        raise RuntimeError('Only the active maxSum arm may be supplemented')
    for parent, config in zip(runs[1:], configs[1:]):
        task = config['datasets'][0]
        selection = parent/'shape_backfill_20260925/seeds/seed1/chronos-base/shape_select'/f'{task}.json'
        row = read(selection)
        if row.get('complete') is not True or row.get('selection_new_bp') != 0:
            raise RuntimeError('Helper requires verified zero-BP selection')
        cell = parent/'shape_backfill_20260925/seeds/seed1/chronos-base/cells'/f'{task}__{ARM}.json'
        if cell.is_file():
            raise RuntimeError('Scoped extension cell already exists; helper not needed')
    run_model(ROOT, runs, configs, 'chronos-base', ARM,
              seed_filter=1, task_filter=TASKS)
    print('SCOPED_CHRONOS_MAX_EXTENSION_COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
