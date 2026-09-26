"""Bitwise equivalence + timing benchmark: resident-source vs per-batch upload."""
import time, sys
import numpy as np
sys.path.insert(0, '/root/TS3/src')
from rags.experiments.reference_opinions import reference_marginal, REVISION

rng = np.random.default_rng(42)
def unit_rows(n, d=256):
    v = rng.standard_normal((n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

for n_ref, n_src in [(64, 3000), (200, 9000)]:
    refs, srcs = unit_rows(n_ref), unit_rows(n_src)
    w = np.full(n_ref, 1.0/n_ref)
    for agg in ['linearSum', 'zeroClipSum', 'maxSum', 'expSum']:
        b1, a1 = reference_marginal(refs, srcs, w, aggregation=agg,
                                    reference_batch=32, source_batch=1024, device='cpu')
        b2, a2 = reference_marginal(refs, srcs, w, aggregation=agg,
                                    reference_batch=32, source_batch=1024, device='cuda')
        same = np.array_equal(b1, b2)
        print(f'ref={n_ref} src={n_src} {agg}: bitwise(cpu vs gpu-resident)={same} maxdiff={np.abs(b1-b2).max():.2e}')

# GPU timing on realistic shape (dim 4096): old vs new cannot both run post-patch,
# so benchmark absolute throughput of new path
d = 4096
refs, srcs = unit_rows(150, d), unit_rows(40000, d)
w = np.full(150, 1/150)
t0 = time.perf_counter()
b, a = reference_marginal(refs, srcs, w, aggregation='zeroClipSum',
                          reference_batch=128, source_batch=2048, device='cuda')
t1 = time.perf_counter()
pairs = (150/128)*(40000/2048)
print(f'gpu-resident: {t1-t0:.2f}s for refs150 x src40000 x dim{d} '
      f'({pairs:.0f} tiles, {pairs*128*2048/t1/1e9:.1f}G cos-elem/s)')
