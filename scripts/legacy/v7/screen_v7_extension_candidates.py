"""Read only native geometry/finite exam feasibility; no model performance."""
import numpy as np
from rage.experiments.shared_formal_inputs import native_rows, build_task_arrays
from rage.experiments.v7_protocol import crb_for

TASKS=[('MetroPT-3/5T',288),('Smart_Manufacturing/H',24),('SG_PM25/H',24),
       ('Current_Velocity/H',24),('Coastal_T_S/H',24)]

def main():
    for dataset,seasonality in TASKS:
        task=dict(benchmark='time',dataset=dataset,
            evaluation_config='configs/eval/time/official_all.yaml',seasonality=seasonality)
        try:
            rows,provenance=native_rows(task)
            refs,exam,series=build_task_arrays(rows,seasonality=seasonality)
            lengths=[len(f.data) for f in refs]
            budget=sum(sum(crb_for('chronos-base').budgets(n-575)) for n in lengths)
            print(dataset,dict(reference_trajectories=len(refs),legal_reference_windows=sum(n-575 for n in lengths),
                rage_reference_bp=budget,exam_windows=len(exam['windows']),
                exam_entities=len(set(exam['entities'])),nonfinite_reference_points=sum(r['nonfinite_reference_points'] for r in series)),flush=True)
        except Exception as error:
            print(dataset,'INELIGIBLE',type(error).__name__,str(error),flush=True)

if __name__=='__main__':
    main()
