"""CPU-only synthetic scale/performance smoke. No formal assets or model BP."""
import json
import resource
import time
import numpy as np
from rags.experiments.v7_scores import uniform_anchors,score_trajectory,interpolated_chunks
from rags.experiments.ranking import rank_mass

def main():
    rng=np.random.default_rng(42)
    n=1_000_000;dim=4096;k=15000
    anchors=uniform_anchors(n,k)
    g=rng.standard_normal((k,dim),dtype=np.float32)
    target=rng.normal(size=dim)
    started=time.perf_counter()
    scores=score_trajectory(n,anchors,g,target)
    scoring=time.perf_counter()-started
    started=time.perf_counter()
    _,prob,audit=rank_mass(scores['grad-dot'],np.ones(n,dtype=np.int64))
    ranking=time.perf_counter()-started
    # Independent dense oracle only for a bounded prefix in a separate fixture.
    p=uniform_anchors(10000,150);gg=g[:150]
    started=time.perf_counter()
    dense=np.concatenate([v for _,v in interpolated_chunks(10000,p,gg)])
    oracle=np.linalg.norm(dense,axis=1)
    oracle_seconds=time.perf_counter()-started
    started=time.perf_counter();actual=score_trajectory(10000,p,gg,target)
    analytic_seconds=time.perf_counter()-started
    np.testing.assert_allclose(actual['grand-static'],oracle,rtol=5e-13,atol=5e-13)
    print(json.dumps(dict(windows=n,dimension=dim,anchors=k,scoring_seconds=scoring,
        ranking_seconds=ranking,probability_sum=float(prob.sum()),positive_support=bool(np.all(prob>0)),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        bounded_oracle_windows=10000,oracle_norm_seconds=oracle_seconds,
        analytic_all_three_seconds=analytic_seconds,gradient_bp=0,
        complexity='O(KD+W) scoring, O(W log W) ranking; no W x D or singleton list'),indent=2))

if __name__=='__main__':main()
