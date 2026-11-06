"""Freeze a fresh V5 run directory without launching model jobs."""
import argparse
import json
from pathlib import Path
from .protocol import validate_config
from .students import atomic_json


def prepare_suite(config,run_dir):
    validate_config(config)
    run=Path(run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    path=run/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Existing protocol differs; use a fresh run directory')
    atomic_json(path,config)
    control=run/'execution_control.json'
    if not control.exists():
        atomic_json(control,dict(mode='prepared',selection_enabled=False,
            training_enabled=False,evaluation_enabled=False))
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/rage/main.json'))
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    print(prepare_suite(json.loads(a.config.read_text()),a.run))


if __name__=='__main__':main()
