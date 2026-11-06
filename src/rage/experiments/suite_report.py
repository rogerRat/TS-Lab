"""Collect and summarize the frozen V5 seed-round suite."""
import argparse
import json
from pathlib import Path
from .protocol import validate_config
from .seed_rounds import round_completion,round_config
from .full_pool_run import summarize
from .students import atomic_json


def collect(parent):
    parent=Path(parent).resolve();config=json.loads((parent/'protocol.json').read_text());validate_config(config)
    rounds=[]
    for seed in config['seeds']:
        folder=parent/'rounds'/f'seed{seed}'
        status=round_completion(round_config(config,seed),folder)
        if not folder.exists():rounds.append(dict(seed=seed,**status));continue
        summary=summarize(folder,round_config(config,seed))
        rounds.append(dict(seed=seed,**summary))
    cells=[cell for r in rounds for cell in r.get('cells',[]) if cell['arm']!='orig' or r['seed']==0]
    report=dict(experiment_version='v5',complete=all(r['complete'] for r in rounds),
        completed_cells=len(cells),rounds=rounds,cells=cells)
    atomic_json(parent/'results.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);a=p.parse_args()
    r=collect(a.run);print(f"V5 cells={r['completed_cells']}; complete={r['complete']}")


if __name__=='__main__':main()
