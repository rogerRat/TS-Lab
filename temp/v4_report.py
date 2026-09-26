"""Collect seed-round cells and publish V4 through the identical V3 template."""
import argparse
import fcntl
import json
from pathlib import Path

from .v3_report import render as render_template


def collect(parent):
    parent=Path(parent).resolve();config=json.loads((parent/'protocol.json').read_text())
    if config['experiment_version']!='v4':raise ValueError('Expected V4 suite')
    for seed in config['seeds']:
        round_dir=parent/'rounds'/f'seed{seed}'
        for student in config['students']:
            out=parent/'students'/student;cells=out/'cells';cells.mkdir(parents=True,exist_ok=True)
            source=round_dir/'students'/student
            for p in (source/'cells').glob('*.json'):
                record=json.loads(p.read_text())
                if record.get('experiment_version')!='v4':continue
                if record.get('arm')=='orig' and seed!=0:continue
                link=cells/p.name
                if not link.exists():link.symlink_to(p)
            state=source/'state.json'
            if state.exists():
                link=out/'state.json'
                if link.is_symlink():link.unlink()
                if not link.exists():link.symlink_to(state)
    return config


def render(parent,destination='docs/MAIN-v4.md'):
    parent=Path(parent).resolve()
    with (parent/'report.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        collect(parent)
        return render_template(parent,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',default='docs/MAIN-v4.md')
    args=parser.parse_args();result=render(args.run,args.out)
    print(f"MAIN-v4 updated: {result['completed_cells']}/666 unique cells")


if __name__=='__main__':main()
