"""Render twelve V7 figures, then replace same-name images after success."""
import argparse
from pathlib import Path
import os
import tempfile
from rage.experiments.v7_figures import read, render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/rage/main_v7_curve4.json')
    parser.add_argument('--run')
    parser.add_argument('--out',default='docs/results/figures')
    parser.add_argument('--reuse-snapshot',action='store_true',
                        help='Restyle the existing completed-cell snapshot without changing means/colors')
    args=parser.parse_args()
    config=read(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.v7-render-',dir=out))
    existing=read(out/'v7_plot_snapshot.json') if args.reuse_snapshot else None
    snapshot=render(config,args.run or config['run_root'],staging,snapshot=existing)
    images=list(staging.glob('*.png'))
    if len(images)!=12:raise RuntimeError('Expected all twelve figures before publishing')
    for path in sorted(staging.iterdir()):
        target=out/path.name
        os.replace(path,target)
    staging.rmdir()
    print(f'Refreshed {len(images)} V7 figures from {len(snapshot["rows"])} completed-cell curves')


if __name__=='__main__':main()
