"""Real-data development selection plus online training curves, never test tuning."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rags.execution import SelectionExecution, selection_execution
from .arms import WindowIndex, RANDOM_SAMPLING_REVISION
from .calibration import build_student, online_curve
from .development_inputs import load_fragments
from .native_probe import NativeProbe
from .native_selection import ARMS, select_arm
from .regions import save_selection, load_selection
from .selection_metrics import SelectionMetrics
from .students import atomic_json
from .train_eval import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student', required=True, choices=['chronos-small', 'timesfm-2.5', 'timemoe-200m'])
    p.add_argument('--inputs', required=True)
    p.add_argument('--validation-dirs', nargs='+', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--arms', nargs='+', choices=(*ARMS, 'full_pool'), default=list(ARMS))
    p.add_argument('--selection-run', help='Reuse completed selections from this matching development run')
    p.add_argument('--steps', type=int, nargs='+', help='Explicit update counts, independent of selection budget')
    args = p.parse_args()
    torch.set_num_threads(4)
    out, inputs = Path(args.out), Path(args.inputs)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(arguments=vars(args), inputs=json.loads((inputs/'protocol.json').read_text()),
                    random_sampling_revision=RANDOM_SAMPLING_REVISION,
                    purpose='real development pilot; not main-table or unseen-test results')
    selection_run = Path(args.selection_run) if args.selection_run else None
    if selection_run:
        source_protocol = json.loads((selection_run/'protocol.json').read_text())
        if (source_protocol['inputs'] != protocol['inputs'] or
            source_protocol['arguments']['dataset'] != args.dataset or
            source_protocol['arguments']['n'] != args.n):
            raise ValueError('Transfer requires the same source pool, reference boundaries, task and N')
        for arm in args.arms:
            if not (selection_run/'selection'/(arm+'.npz')).exists():
                raise FileNotFoundError(f'Selection not completed: {arm}')
        protocol['selection_source_protocol'] = source_protocol
    atomic_json(out/'protocol.json', protocol)
    pool = load_fragments(inputs/'pool.npz')
    reference = load_fragments(inputs/(args.dataset+'.npz'))
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    engine = build_student(args.student)
    if args.student == 'timemoe-200m':
        engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    probe = None
    if selection_run is None:
        probe = (engine if args.student == 'chronos-small' or args.arms in [['full_pool'], ['random']] else
                 NativeProbe(engine, model_key=args.student))
        probe.execution = SelectionExecution(gradient_batch=1, inference_batch=4)
    base = {k: v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
    validation = []
    for directory in args.validation_dirs:
        with np.load(Path(directory)/(args.dataset+'.npz')) as d:
            validation.append(dict(test_windows=d['windows'], test_entities=d['entities'].tolist(),
                                   test_scales=d['scales'], context_length=512))
    def assess():
        scores = [evaluate(engine, batch_size=16, **v) for v in validation]
        return dict(by_time_fold=scores, mean_mase=float(np.mean([s['mase'] for s in scores])))
    atomic_json(out/'original.json', assess())
    metrics = SelectionMetrics(out, device=engine.device, protocol=protocol)
    for arm in args.arms:
        path = out/'selection'/(arm+'.npz')
        cached = path.exists()
        if selection_run:
            selected = metrics.measure(args.dataset, arm,
                lambda: load_selection(selection_run/'selection'/(arm+'.npz')), selection_cached=True)
        else:
            with selection_execution(probe):
                selected = metrics.measure(args.dataset, arm,
                    lambda: load_selection(path) if cached else select_arm(arm, probe, reference, pool, index, n_select=args.n),
                    selection_cached=cached)
        if arm == 'random' and selected.diagnostics.get('random_sampling_revision') != RANDOM_SAMPLING_REVISION:
            raise ValueError('Legacy fixed-subset Random cannot be reused; create a new Random run')
        if not cached:
            save_selection(path, selected)
        for seed in args.seeds:
            result_path = out/f'{arm}-seed-{seed}.json'
            if result_path.exists():
                continue
            result = online_curve(engine, selected.selected, base_state=base, learning_rate=args.lr,
                seed=seed, checkpoints=([(s*32/args.n, s) for s in args.steps] if args.steps else
                                       [(e, math.ceil(e*args.n/32)) for e in [1,2,4]]),
                batch_size=32, evaluate_fn=assess)
            atomic_json(result_path, result)
            print(arm, seed, 'DEVELOPMENT_CURVE_DONE', flush=True)
    print('NATIVE_DEVELOPMENT_DONE', flush=True)


if __name__ == '__main__':
    main()
