"""Diagnostic resampling of fixed audited windows, not a new formal selector."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .arms import WindowIndex, streaming_random_select
from .calibration import build_student, online_curve
from .development_inputs import load_fragments
from .regions import load_selection
from .students import atomic_json
from .train_eval import evaluate


def matching_probabilities(labels, target):
    labels, target = np.asarray(labels), np.asarray(target, dtype=float)
    counts = np.bincount(labels, minlength=len(target))
    missing_mass = float(target[counts == 0].sum())
    probabilities = target[labels] / counts[labels]
    if probabilities.sum() == 0:
        raise ValueError('No audited source group overlaps target mass')
    probabilities /= probabilities.sum()
    achieved = np.bincount(labels, weights=probabilities, minlength=len(target))
    return probabilities, dict(missing_target_mass=missing_mass,
        achieved_group_shares=achieved.tolist(), target_group_shares=target.tolist(),
        effective_sample_size=float(1 / np.square(probabilities).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--steps', nargs='+', type=int, default=[4,8,16,32,128])
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    args = p.parse_args()
    torch.set_num_threads(4)
    audit = json.loads(Path(args.audit).read_text())
    original = audit['source_protocol']['arguments']
    inputs = Path(original['inputs'])
    run = Path(audit['arguments']['selection_run'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'protocol.json', dict(arguments=vars(args), audit_arguments=audit['arguments'],
        purpose='Fixed 512-instance audit subset; uniform vs reference-mixture sampling, NOT formal Random or RAGE',
        absent_groups='Target distribution conditional on groups present in audit subset; missing mass disclosed'))
    engine = build_student(audit['arguments']['student'])
    if audit['arguments']['student'] == 'timemoe-200m':
        engine.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base = {k:v.detach().cpu().clone() for k,v in engine.model.state_dict().items()}
    validations = []
    for directory in original['validation_dirs']:
        with np.load(Path(directory)/(original['dataset']+'.npz')) as data:
            validations.append(dict(test_windows=data['windows'], test_entities=data['entities'].tolist(),
                test_scales=data['scales'], context_length=512))
    def assess():
        scores = [evaluate(engine, batch_size=16, **v) for v in validations]
        return dict(by_time_fold=scores, mean_mase=float(np.mean([s['mase'] for s in scores])))
    atomic_json(out/'original.json', assess())
    for arm, record in audit['arms'].items():
        if arm == 'before_random_pool':
            index = WindowIndex(load_fragments(inputs/'pool.npz'), context_length=512, prediction_length=64, stride=1)
            selected = streaming_random_select(index, n_select=128).selected
        else:
            selected = load_selection(run/'selection'/(arm+'.npz')).selected
        windows = selected[np.asarray(record['sampled_rows'])]
        mixture = record['direction_mixtures'][str(args.groups)]
        probabilities, diagnostics = matching_probabilities(mixture['source_labels'], mixture['reference_shares'])
        atomic_json(out/(arm+'-sampling.json'), dict(diagnostics=diagnostics,
            sampled_rows=record['sampled_rows'], probabilities=probabilities.tolist()))
        for variant, weights in [('uniform', None), ('target_mixture', probabilities)]:
            for seed in args.seeds:
                path = out/f'{arm}-{variant}-seed-{seed}.json'
                if path.exists():
                    continue
                result = online_curve(engine, windows, base_state=base, learning_rate=args.lr, seed=seed,
                    checkpoints=[(s*32/original['n'],s) for s in args.steps], batch_size=32,
                    evaluate_fn=assess, sampling_probabilities=weights)
                atomic_json(path, result)
                print(arm, variant, seed, 'MIXTURE_INTERVENTION_DONE', flush=True)


if __name__ == '__main__':
    main()
