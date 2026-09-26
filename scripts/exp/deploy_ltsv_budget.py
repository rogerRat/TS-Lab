"""Install validated LTSV budget settings and retain counterfactual cost evidence.

No formal job is launched, no existing selection is overwritten. The caller
must explicitly select the run and the independent pilot report directory.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from baselines.ltsv.fast import REVISION
from rags.experiments.students import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    args=p.parse_args()
    protocol=json.loads((args.run/'protocol.json').read_text())
    names={'chronos-base':'chronos','timesfm-2.5':'timesfm','timemoe-200m':'timemoe'}
    rows, installs=[],[]
    for student,short in names.items():
        path=args.pilot_root/f'{short}-stride576-ref64.json'
        d=json.loads(path.read_text())
        identity=json.loads((args.run/'students'/student/'model.json').read_text())
        if d['model']!=identity or not d['exact_concurrency_scores']:
            raise ValueError(f'{student}: model or concurrency gate failed')
        if not all(np.isfinite(t['probability_tv']) and t['rank_correlation']>0 and
                   t['probability_tv']<=.1 for t in d['tasks']):
            raise ValueError(f'{student}: stride-only pilot exceeds disclosed 10% TV screen')
        records={r['workers']:r for r in d['concurrency']}
        # Favor two workers unless four provide >15% additional active throughput.
        workers=4 if records[2]['active_span_seconds']/records[4]['active_span_seconds']>1.15 else 2
        if records[1]['active_span_seconds']/records[workers]['active_span_seconds']<1.1:
            workers=1
        chosen=records[workers]
        baseline=d['baseline_seconds_per_block']
        active_per_block=chosen['active_span_seconds']/chosen['blocks']
        startup=chosen['wall_seconds']-chosen['active_span_seconds']
        estimated_actual=startup+d['budget_blocks_per_task']*active_per_block
        original=d['original_blocks_per_task']*baseline
        timing=[r['update_seconds']+r['forecast_seconds']+r['cpu_restore_seconds']
            for r in d['rows'] if r['mode']=='baseline' and not r.get('timing_reused')]
        gpu_timing=[r['update_seconds']+r['forecast_seconds']+r['gpu_restore_seconds']
            for r in d['rows'] if r['mode']=='baseline' and not r.get('timing_reused')]
        peak=max(r['max_worker_peak_gpu_gib'] for r in d['concurrency'] if r['workers']>1)
        execution=dict(schema=REVISION,model=identity,**d['execution'],workers=workers,
            worker_peak_gpu_gib=peak,baseline_seconds_per_block=baseline,
            validation_report=str(path),label='LTSV-Budget (stride576, Ref64, eager)',
            baseline_scope='same framework adapter, original stride288/Ref64 serial CPU restore',
            rng_protocol='canonical-original-block-seed-v1',
            numerical_tolerance='source-prefix budget screening, no downstream accuracy equivalence claim')
        cache=args.run/'students'/student/'method_cache'
        destination=cache/'ltsv_execution.json'
        if destination.exists() and json.loads(destination.read_text())!=execution:
            raise ValueError(f'Existing execution settings differ: {destination}')
        selections=list((args.run/'students'/student/'selections').glob('*__ltsv.npz'))
        if selections:
            raise ValueError(f'Existing LTSV selections need explicit migration: {selections}')
        installs.append((destination,execution))
        rows.append(dict(student=student,workers_requested=workers,
            original_blocks=d['original_blocks_per_task'],budget_blocks=d['budget_blocks_per_task'],
            original_seconds_per_block_mean=baseline,
            original_seconds_per_block_p10_p90=np.percentile(timing,[10,90]).tolist(),
            paired_original_timing_rows=len(timing),
            original_serial_seconds_estimate=original,
            original_gpu_restore_serial_seconds_estimate=d['original_blocks_per_task']*float(np.mean(gpu_timing)),
            budget_serial_seconds_estimate=d['budget_serial_seconds_estimate'],
            measured_pilot_parallel_wall_seconds=chosen['wall_seconds'],
            measured_pilot_parallel_active_seconds=chosen['active_span_seconds'],
            measured_pilot_parallel_blocks=chosen['blocks'],
            parallel_startup_seconds=startup,
            budget_parallel_seconds_estimate=estimated_actual,
            estimated_original_to_budget_parallel_ratio=original/estimated_actual,
            estimated_original_to_budget_serial_ratio=d['estimated_serial_speedup'],
            quality_pilot=d['tasks'],concurrency=d['concurrency'],source_report=str(path),
            actual_formal_seconds=None,actual_formal_ratio=None))
    evidence=dict(schema='ltsv-budget-cost-evidence-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
        run=str(args.run),method_label='LTSV-Budget',rows=rows,
        formula=dict(original='original_block_count * matched_original_mean_seconds_per_block',
            modified_estimate='measured_startup + budget_block_count * measured_active_seconds / measured_blocks',
            ratio='estimated_original_seconds / modified_seconds',
            completed_run='complete.json records actual session time and estimated original / actual ratio; partial resumes are not cold runs'),
        limitations=['32 source blocks from 8 trajectory prefixes and 3 safe Ref tasks per model',
            'Chronos reuses original score rows for non-sentinel blocks; current original timing uses 8 sentinel blocks per task',
            'MPS contention and source-prefix representativeness affect estimates; p10/p90 are timing spread, not confidence intervals',
            'Parallel extrapolation excludes sustained pipeline and long-run effects; memory pressure may reduce worker count',
            'Downstream accuracy has not been measured; final cells measure budget-variant accuracy',
            'Original estimated cost and budget-variant accuracy must be labeled as different protocols; original accuracy is unavailable'])
    for destination,execution in installs:
        destination.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(destination,execution)
    report_path=args.run/'reports/ltsv_budget_cost_evidence.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(report_path,evidence)
    atomic_json(args.pilot_root/'deployed_cost_evidence.json',evidence)
    print(json.dumps({r['student']:dict(workers=r['workers_requested'],
        original_hours=r['original_serial_seconds_estimate']/3600,
        budget_hours=r['budget_parallel_seconds_estimate']/3600,
        ratio=r['estimated_original_to_budget_parallel_ratio']) for r in rows},indent=2))

if __name__=='__main__': main()
