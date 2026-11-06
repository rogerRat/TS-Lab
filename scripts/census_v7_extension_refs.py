"""Reference-only budget census; no model evaluation or test-performance screening."""
import numpy as np
from rage.experiments.v7_extension import CANDIDATES
from rage.experiments.shared_formal_inputs import native_rows
from rage.experiments.v7_protocol import crb_for
from assets.gradient.cache import atomic_json


def main():
    report=[]
    for task in CANDIDATES:
        rows,provenance=native_rows(task)
        lengths=[];nonfinite=0;constant=0
        for entity,values,boundary in rows:
            x=np.asarray(values[:boundary]);finite=np.isfinite(x)
            nonfinite+=int((~finite).sum())
            edges=np.diff(np.r_[False,finite,False].astype(np.int8))
            for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                if hi-lo>=576:
                    lengths.append(int(hi-lo));constant+=int(np.ptp(x[lo:hi])==0)
        row=dict(task=task['id'],reference_trajectories=len(lengths),
            reference_legal_windows=sum(n-575 for n in lengths),
            rage_reference_bp=sum(sum(crb_for('chronos-base').budgets(n-575)) for n in lengths),
            nonfinite_reference_points=nonfinite,constant_reference_fragments=constant,
            selection_uses_test_performance=False)
        report.append(row);print(row,flush=True)
    atomic_json('/root/autodl-tmp/rage/outputs/main-v7-20260923/extensions/reference_queue_census.json',report)


if __name__=='__main__':
    main()
