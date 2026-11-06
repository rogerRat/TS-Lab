"""Concurrent model streams; a single writer per model, no global A/B barrier."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil


def execute_streams(streams, runner):
    def stream(jobs):
        for job in jobs:
            runner(job)  # Failure stops this model, not unrelated model streams.
    failures = []
    with ThreadPoolExecutor(max_workers=len(streams)) as pool:
        futures = {pool.submit(stream, jobs): name for name, jobs in streams.items()}
        for future, name in futures.items():
            try:
                future.result()
            except Exception as error:
                failures.append((name, str(error)))
    if failures:
        raise RuntimeError(f'model streams failed; completed caches retained: {failures}')


def commands(config):
    streams = {}
    for model in config['models']:
        base = [model['python'], '-u', '-m', 'rage.experiments.reference_census',
                'compute', '--student', model['student'], '--out', config['out'],
                '--cpu-threads', '2', '--memory-fraction', str(model.get('memory_fraction', .25)),
                '--tune-batches', *map(str, model['batches']),
                '--tune-workers', *map(str, model['workers'])]
        # Always invoke stages: validated cache/identity, not marker existence,
        # determines what is reusable after interruption or a changed setting.
        stages = [('references', model['runs'][0])]
        stages += [('alignment', run) for run in model['runs']]
        streams[model['student']] = [
            dict(name=f"{model['student']}-{stage}-{Path(run).name}",
                 command=base+['--stage', stage, '--run', run])
            for stage, run in stages]
    return streams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config['out']); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH='src', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (out/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Budget active streams together, not independent checks of the same
        # free bytes. Resume credits already allocated artifacts.
        allocated = sum(p.stat().st_blocks*512 for p in out.rglob('*') if p.is_file())
        if shutil.disk_usage(out).free + allocated < config['total_budget_gib']*2**30:
            raise RuntimeError('insufficient shared data-disk budget; request capacity, never delete results')
        def run(job):
            log = out/'logs'/f"{job['name']}.log"
            log.parent.mkdir(exist_ok=True)
            print('START', job['name'], job['command'], flush=True)
            with log.open('a') as handle:
                subprocess.run(job['command'], cwd=config['repo'], env=env,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            print('DONE', job['name'], flush=True)
        execute_streams(commands(config), run)
        runs = list(dict.fromkeys(run for m in config['models'] for run in m['runs']))
        run(dict(name='associations', command=[config['report_python'], '-u', '-m',
            'rage.experiments.reference_associations', '--runs', *runs,
            '--census', str(out), '--out', str(out/config.get('report_subdir', 'report'))]))


if __name__ == '__main__':
    main()
