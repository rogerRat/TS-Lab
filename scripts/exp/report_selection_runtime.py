"""Report baseline-unoptimized estimates; keep actual optimized time secondary.

Reads only cost JSON members of published selections, not the window arrays.
Shared framework acceleration (MPS) remains in both sides of the comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from baselines.ltsv.runtime import comparison_seconds
from rags.experiments.students import atomic_json


def report(run):
    config = json.loads((run / 'protocol.json').read_text())
    rows, missing = [], []
    for student in config['students']:
        for dataset in config['datasets']:
            for arm in config['arms']:
                path = run / 'students' / student / 'selections' / f'{dataset}__{arm}.npz'
                if not path.exists():
                    missing.append(dict(student=student, dataset=dataset, arm=arm))
                    continue
                with np.load(path, allow_pickle=False) as data:
                    cost = json.loads(str(data['cost']))
                notes = cost.get('notes', {})
                adjustment = notes.get('runtime_adjustment')
                actual = cost['wall_seconds']
                reported = comparison_seconds(actual, adjustment)
                warm = any(notes.get(k, False) for k in ('pool_cached', 'proxy_cached', 'scores_cached'))
                gradient = notes.get('gradient_cost')
                cold = ((gradient.get('estimated_cold_selection_seconds')
                         if gradient.get('estimate_complete_method_stages') else None)
                        if gradient else None if warm else reported)
                budget = bool(adjustment and adjustment.get('report_basis') ==
                              'estimated_original_full_budget_serial_implementation')
                execution = notes.get('execution', {})
                rows.append(dict(student=student, dataset=dataset, arm=arm,
                    method_label=execution.get('label', arm),
                    quality_protocol='budget_variant' if execution else 'configured_method',
                    original_cost_is_counterfactual=budget,
                    original_full_budget_seconds_estimate=(reported if budget else None),
                    reported_selection_body_seconds=reported,
                    reported_cold_selection_seconds=cold,
                    actual_execution_seconds=actual,
                    report_basis=(adjustment.get('report_basis', 'baseline-unoptimized estimate')
                                  if adjustment else 'no method-specific adjustment'),
                    estimated_native_to_actual_ratio=reported / actual if actual else None,
                    runtime_adjustment=adjustment, gradient_cost=gradient,
                    cache_state='warm' if warm else 'cold'))
    return dict(schema='selection-runtime-comparison-v1', rows=rows, missing=missing,
        complete=not missing,
        policy='Primary report reverses baseline-specific optimization only; common MPS retained',
        warning='Do not compare selection-body time against another method cold time; warm/cache reads are not cold costs. Gradient cold estimates retain their recorded calibration limits. Budget-variant accuracy belongs with its actual time; original full-budget time is a labeled counterfactual with no original accuracy claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report(args.run))


if __name__ == '__main__':
    main()
