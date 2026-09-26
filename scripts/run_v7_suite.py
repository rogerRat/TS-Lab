"""Event-driven phase/seed barriers, not a polling monitor. Kimi owns admission."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
from assets.gradient.cache import atomic_json
from rags.experiments.v7_protocol import MODELS, CURVE_PROTOCOL, REDUCED_PROTOCOL, validate, asset_protocol
from rags.experiments.v7_amendment import freeze_training
from rags.experiments.v7_run import freeze, gate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/rags/main_v7_curve4.json')
    parser.add_argument('--run')
    parser.add_argument('--parallel-models',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--phase',choices=['assets','select','train','all'],default='all')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());validate(config)
    run=Path(args.run or config['run_root']);run.mkdir(parents=True,exist_ok=True)
    freeze(run/'protocol.json',asset_protocol(config))
    if 'training_protocol' in config:
        freeze_training(run,config)
    logs=run/'logs';logs.mkdir(exist_ok=True)
    def execute(student,phase,seed=None):
        command=[config['student_environments'][student],'-m','rags.experiments.v7_run',
            '--config',str(Path(args.config).resolve()),'--run',str(run),'--phase',phase,'--student',student]
        if seed is not None:command+=['--seed',str(seed)]
        with (logs/f'{phase}__{student}__seed{seed}.log').open('a') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'V7 worker failed: {student} {phase} seed={seed}; see {log.name}')
    def parallel(phase,seed=None):
        # Blocking completions, not periodic process/GPU polling.
        with ThreadPoolExecutor(max_workers=args.parallel_models) as executor:
            futures=[executor.submit(execute,s,phase,seed) for s in MODELS]
            for future in futures:future.result()
    def model_pipeline(student):
        if args.phase == 'all':
            execute(student, 'select')
        gate(config,run,'select',student=student)
        for seed in config['seeds']:
            execute(student,'train',seed)
    state=run/'supervisor.json'
    atomic_json(state,dict(status='running',pid=os.getpid(),phase=args.phase,
                           parallel_models=args.parallel_models))
    try:
        if args.phase=='all':
            from rags.experiments.v7_inputs import prepare
            prepare(config,run)
        if args.phase in ('all','assets'):
            parallel('assets');gate(config,run,'assets')
        if args.phase == 'select':
            gate(config,run,'assets');parallel('select');gate(config,run,'select')
        if args.phase in ('all','train'):
            if config.get('training_protocol', {}).get('version') not in (CURVE_PROTOCOL, REDUCED_PROTOCOL):
                raise ValueError('Training requires the approved four-epoch curve protocol')
            gate(config,run,'assets')
            with ThreadPoolExecutor(max_workers=args.parallel_models) as executor:
                futures=[executor.submit(model_pipeline,s) for s in MODELS]
                for future in futures:future.result()
        atomic_json(state,dict(status='complete',phase=args.phase,pid=os.getpid()))
    except BaseException as error:
        atomic_json(state,dict(status='failed',phase=args.phase,error=repr(error),pid=os.getpid()))
        raise
    return 0

if __name__=='__main__':raise SystemExit(main())
