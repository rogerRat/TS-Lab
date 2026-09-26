"""Run the frozen V5 suite; execution controls must be explicitly enabled."""
import argparse
import json
from pathlib import Path
from .protocol import validate_config
from .seed_rounds import run_rounds


def preflight(config,readiness):
    if not readiness.get('ready') or readiness.get('config')!=config:
        raise ValueError('Queued formal config differs from the reviewed ready configuration')
    validate_config(config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/rags/main.json'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());validate_config(config)
    if args.dry_run:
        print(json.dumps(dict(version='v5',models=config['students'],tasks=config['datasets'],
            methods=config['arms'],seeds=config['seeds'],jobs_launched=0),indent=2));return 0
    return run_rounds(config,args.run)


if __name__=='__main__':raise SystemExit(main())
