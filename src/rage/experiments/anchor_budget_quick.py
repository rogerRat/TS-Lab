"""Small CPU-only budget sweep on the first completed trajectory of each task."""
import argparse
from pathlib import Path

from .anchor_budget_simulate import (load_sample_manifest, manifest_entries,
                                    completed_keys, run_simulate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--covers', type=float, nargs='+', default=(.04, .08, .12, .16, .20, .24, .28))
    parser.add_argument('--error', type=float, default=.02)
    parser.add_argument('--split', type=float, default=0.)
    args = parser.parse_args()
    manifest = load_sample_manifest(args.run_root)
    entries = manifest_entries(manifest, {0})
    done = completed_keys(args.run_root)
    if len(entries) != 5 or any((e['task'], e['fragment_id'], e['wave']) not in done for e in entries):
        raise ValueError('All five rank-0 trajectories must be complete')
    run_simulate(manifest, args.out, entries=entries,
                 grid={'cover': tuple(args.covers),
                       'error': (args.error,), 'split': (args.split,)}, mode='preliminary')


if __name__ == '__main__':
    main()
