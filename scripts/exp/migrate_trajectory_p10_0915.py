"""Migrate the stopped pre-selection shared-ten run, retaining window assets."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from assets.gradient.cache import atomic_json
from rage.experiments.formal_main_entry import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = proc.read_bytes().split(b'\0')
        except OSError:
            continue
        if (b'rage.experiments.full_pool_worker' in argv or b'rage.experiments.formal_main_entry' in argv) and str(run).encode() in argv:
            raise RuntimeError('Stop active run before migration: ' + str(proc))
    old = json.loads((run/'protocol.json').read_text())
    new = json.loads(args.config.read_text())
    expected = json.loads(json.dumps(old))
    expected['crb']['epsilon_policy'] = 'trajectory_p10'
    for setting in expected['settings'].values():
        setting.pop('epsilon_calibration', None)
    if new != expected:
        raise ValueError('Migration may change only epsilon policy and external paths')
    derived = list((run/'students').glob('*/selections/*.npz'))
    derived += [p for p in (run/'students').glob('*/cells/*.json') if '__original__' not in p.name]
    derived += list((run/'students').glob('*/checkpoints/**/*.pt'))
    if derived:
        raise RuntimeError('Derived artifacts require explicit invalidation: ' + str(derived))
    ready = json.loads((run/'readiness.json').read_text())
    ready.update(config=new, reason='User method update: phase 1/2 per-trajectory frozen linear P10; regression verified')
    preflight(new, ready)
    caches = []
    for marker in sorted((run/'grad_cache').glob('*/cache_manifest.json')):
        with np.load(marker.parent/'window_index.npz', allow_pickle=False) as payload:
            count = len(payload['window_ids'])
        caches.append(dict(path=str(marker.parent), cached_windows=count,
                           model=json.loads(marker.read_text())['model']))
    report = dict(revision='trajectory-p10-20260915-v1', caches=caches,
                  original_cells=len(list((run/'students').glob('*/cells/*__original__*.json'))),
                  derived_artifacts=len(derived), old_protocol=old, new_protocol=new,
                  recovery='Replay CRB; reuse same-identity window gradients; no old derived artifacts')
    backup = run/'migrations'/'trajectory-p10-20260915-v1'
    if args.apply:
        backup.mkdir(parents=True, exist_ok=False)
        for name in ('protocol.json', 'readiness.json', 'results.json'):
            if (run/name).exists():
                shutil.copy2(run/name, backup/name)
        atomic_json(backup/'migration.json', report)
        atomic_json(run/'protocol.json', new)
        atomic_json(run/'readiness.json', ready)
    print(json.dumps(dict(applied=args.apply, backup=str(backup),
        original_cells=report['original_cells'], caches=[dict(path=c['path'], cached_windows=c['cached_windows']) for c in caches]), indent=2))


if __name__ == '__main__':
    main()
