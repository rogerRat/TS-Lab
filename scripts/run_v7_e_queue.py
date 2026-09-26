"""V7-e: freeze the two strongest complete Shape arms, then extend TIME tasks.

Each task gets the same two source-pool seeds and 11 reported arms: the two
Shape arms, their Block/Traj counterparts, LP/TSDS/LESS, rand and orig.
The ten-task result matrix is never edited.  This process waits on pidfds,
not a progress-polling loop, and stops at task boundaries on STOP.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from assets.gradient.cache import atomic_json
from rags.experiments.v7_extension import V7_E_CANDIDATES, make_config, prepare
from rags.experiments.v7_protocol import MODELS, validate
from rags.experiments.v7_run import artifact_identity, freeze, read, folder, gate
from rags.experiments.v7_amendment import report_view
from rags.experiments.v7_shape_backfill import checked_model_lease, run_model, sidecar_config
from scripts.report_v7_final import SHAPE_ARMS, render
from scripts.run_v7_extension_queue import complete, finalize_task, run_task
from scripts.supervise_v7_night import pidfd


VERSION = 'v7-e-two-shape-two-seed-exploration-20260926'
GPU = {'chronos-base': '0', 'timerxl-84m': '1', 'timesfm-2.5': '1'}
MAIN_CONFIG = Path('configs/rags/main_v7_curve4.json')


def frozen_shape_choice(parent, shape_pid=None):
    """No post-hoc task-specific choice: rank all four on the same 60 cells."""
    base = read(MAIN_CONFIG)
    arms = tuple(arm for arm in base['arms'] if arm != 'grad-cos') + SHAPE_ARMS
    def current():
        try:
            _, summary = render(parent, tuple(base['datasets']), arms)
        except ValueError as error:
            if 'matrix incomplete' not in str(error):
                raise
            return None
        return summary
    summary = current()
    if summary is None:
        if shape_pid is None:
            raise RuntimeError('The four Shape arms are incomplete; supply the verified Shape queue PID')
        words = Path(f'/proc/{shape_pid}/cmdline').read_bytes().decode().split('\0')
        if ('scripts.run_v7_shape_backfill' not in words or '--parent' not in words or
                Path(words[words.index('--parent') + 1]).resolve() != parent or '--student' in words):
            raise ValueError('Shape queue PID identity differs')
        fd = pidfd(shape_pid)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_READ)
                selector.select()
        finally:
            os.close(fd)
        summary = current()
        if summary is None:
            raise RuntimeError('Shape queue exited without the complete ten-task result barrier')
    chosen = summary['top_two_shape']
    if chosen is None or len(chosen) != 2:
        raise RuntimeError('Complete Shape ranking did not produce exactly two arms')
    receipt = dict(version=VERSION, selection_basis='mean_paired_rand_improvement_over_all_60_original_ten_task_cells',
        training_tasks=list(summary['tasks']), source_cells=summary['expected_cells'],
        selected_shape_arms=chosen, shape_means={a: summary['means'][a] for a in SHAPE_ARMS},
        seeds=[0, 1], models=list(MODELS), future_task_test_metrics_used=False)
    freeze(parent/'v7_e_20260926'/'frozen_shape_choice.json', receipt)
    return tuple(chosen)


def audit_shape(parent, run, config, shape_arms):
    sidecar = run/'shape_backfill_20260925'
    side_config = sidecar_config(config, sidecar)
    if tuple(side_config['arms'][2:]) != tuple(shape_arms):
        raise ValueError('V7-e Shape sidecar methods differ from frozen choice')
    freeze(sidecar/'shape_config.json', side_config)
    task = config['datasets'][0]
    for model in MODELS:
        identity = read(parent/'models'/f'{model}.json')
        for seed in (0, 1):
            manifest = read(run/'inputs'/f'seed{seed}_manifest.json')
            expected = dict(artifact_identity(side_config, model, seed, task, manifest, identity),
                            training_protocol=side_config['training_protocol'])
            root = folder(sidecar, model, seed)
            selection = read(root/'shape_select'/f'{task}.json')
            if (not selection.get('complete') or selection.get('selection_new_bp') != 0 or
                    [row['arm'] for row in selection['methods']] != list(shape_arms)):
                raise ValueError('V7-e Shape selection is incomplete or performed new BP')
            for arm in shape_arms:
                row = report_view(read(root/'cells'/f'{task}__{arm}.json'), expected, side_config, model)
                if (row.get('arm'), row.get('seed'), row.get('task')) != (arm, seed, task):
                    raise ValueError('V7-e Shape result identity differs')
            finished = read(root/'shape_complete'/f'{task}.json')
            if (finished.get('identity') != expected or finished.get('method_count') != 2 or
                    finished.get('zero_new_bp') is not True):
                raise ValueError('V7-e Shape completion receipt differs')
    return sidecar


def run_shape_phase(parent, run, config_path, shape_arms, arm):
    config = read(config_path)
    active = {}
    with selectors.DefaultSelector() as selector:
        for model in MODELS:
            log = run/'v7_e_logs'/f'{arm}__{model}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [config['student_environments'][model], '-m', 'scripts.run_v7_e_queue',
                       '--parent', str(parent), '--worker-config', str(config_path),
                       '--student', model, '--arm', arm]
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd='/root/TS3',
                    env=dict(os.environ, PYTHONPATH='src:.', CUDA_VISIBLE_DEVICES=GPU[model]),
                    stdout=stream, stderr=subprocess.STDOUT)
            fd = pidfd(process.pid)
            selector.register(fd, selectors.EVENT_READ)
            active[fd] = (process, model)
        errors = []
        while active:
            for key, _ in selector.select():
                process, model = active.pop(key.fd)
                selector.unregister(key.fd)
                os.close(key.fd)
                if process.wait():
                    errors.append(model)
        if errors:
            raise RuntimeError(f'V7-e Shape workers failed: {errors}/{arm}; inspect v7_e_logs')


def run_one(parent, task, shape_arms):
    base = read(MAIN_CONFIG)
    run = parent/'extensions'/'v7-e'/f"{task['id']}-20260926"
    config = make_config(base, parent, run, task, shape_arms=shape_arms)
    validate(config)
    config_path = Path('configs/rags')/f"v7_e_{task['id']}.json"
    freeze(config_path, config)
    marker = run/'v7_e_task_complete.json'
    if marker.is_file():
        row = read(marker)
        if row.get('version') != VERSION or row.get('task') != task or row.get('shape_arms') != list(shape_arms):
            raise ValueError('Existing V7-e completion identity differs')
        if not all(complete(config, run, model) for model in MODELS):
            raise ValueError('Completed V7-e task has missing native cells')
        audit_shape(parent, run, config, shape_arms)
        return row
    prepare(config)
    if not (run/'task_queue_complete.json').is_file():
        if all(complete(config, run, model) for model in MODELS):
            finalize_task(config, config_path)
        else:
            run_task(config, config_path)
    receipt = read(run/'task_queue_complete.json')
    native_count = len(config['arms']) * len(MODELS) * 2
    if not receipt.get('complete') or receipt.get('cells') != native_count:
        raise ValueError('V7-e native task barrier differs')
    gate(config, run, 'assets')
    gate(config, run, 'select')
    for arm in shape_arms:
        run_shape_phase(parent, run, config_path, shape_arms, arm)
    audit_shape(parent, run, config, shape_arms)
    result = dict(version=VERSION, complete=True, task=task, shape_arms=list(shape_arms),
        native_arms=list(config['arms']), seeds=[0, 1], models=list(MODELS),
        cells=native_count + len(shape_arms)*len(MODELS)*2,
        zero_new_bp_shape_selection=True, time=time.time())
    atomic_json(marker, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--wait-shape-pid', type=int)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--student', choices=MODELS)
    parser.add_argument('--arm', choices=SHAPE_ARMS)
    args = parser.parse_args()
    parent = args.parent.resolve()
    if args.worker_config:
        if not args.student or not args.arm:
            parser.error('Shape worker needs student and arm')
        config = read(args.worker_config)
        validate(config)
        selected = config['task_extension']['v7_e_shape_arms']
        if args.arm not in selected:
            raise ValueError('Shape worker arm is outside the frozen V7-e choice')
        run = Path(config['run_root']).resolve()
        with checked_model_lease(run, args.student):
            run_model(parent, [run], [config], args.student, args.arm)
        return
    if args.student or args.arm:
        parser.error('Worker parameters require --worker-config')
    state = parent/'v7_e_20260926'
    state.mkdir(parents=True, exist_ok=True)
    with (state/'queue.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        chosen = frozen_shape_choice(parent, args.wait_shape_pid)
        for position, task in enumerate(V7_E_CANDIDATES, 1):
            if (state/'STOP').exists():
                atomic_json(state/'status.json', dict(state='stopped_at_task_boundary',
                    next_task=task['id'], position=position, time=time.time()))
                return
            atomic_json(state/'status.json', dict(state='running', task=task['id'],
                position=position, total_pinned=len(V7_E_CANDIDATES), shape_arms=list(chosen), time=time.time()))
            result = run_one(parent, task, chosen)
            atomic_json(state/'status.json', dict(state='task_complete', task=task['id'],
                position=position, total_pinned=len(V7_E_CANDIDATES), cells=result['cells'], time=time.time()))
        atomic_json(state/'status.json', dict(state='candidate_list_exhausted',
            completed_tasks=len(V7_E_CANDIDATES), action='add independently justified TIME tasks; never repeat cells',
            time=time.time()))


if __name__ == '__main__':
    sys.exit(main())
