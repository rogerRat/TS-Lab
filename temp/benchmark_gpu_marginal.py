import json
import time
from pathlib import Path
import numpy as np
import torch
import rage.experiments.resampling as mod
from threadpoolctl import threadpool_limits

torch.set_num_threads(4)
rng = np.random.default_rng(72)
refs = rng.normal(size=(256, 1024)).astype(np.float32)
sources = rng.normal(size=(32768, 1024)).astype(np.float32)
refs /= np.linalg.norm(refs, axis=1, keepdims=True)
sources /= np.linalg.norm(sources, axis=1, keepdims=True)
n, m = rng.integers(1, 30, len(refs)), rng.integers(1, 30, len(sources))
original = mod._gpu_reference_marginal

def old_postprocess(refs, gs, pi, m, ids, temperature, row_batch, gating):
    beta = np.zeros(len(m))
    for begin in range(0, refs.shape[0], row_batch):
        scores = (torch.as_tensor(refs.batch(begin, min(begin+row_batch, refs.shape[0])),
                                 device=gs.device) @ gs.T).cpu().numpy()
        for offset, row in enumerate(scores):
            logits = (row-row.max())/temperature + np.log(m)
            weights = np.exp(logits-logits.max())
            beta += pi[begin+offset] * weights/weights.sum()
    return beta, []

results = []
with threadpool_limits(limits=4):
    for name, helper, batch in [('old_gpu_cpu', old_postprocess, 32),
                                 ('gpu_all_32', original, 32),
                                 ('gpu_all_128', original, 128)]:
        mod._gpu_reference_marginal = helper
        times = []
        for attempt in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            beta, _ = mod.rage_marginal(refs, sources, n, m, device='cuda',
                                        gating=False, temperature=.25, row_batch=batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if attempt:
                times.append(elapsed)
        if name == 'old_gpu_cpu':
            expected = beta.copy()
        else:
            np.testing.assert_allclose(beta, expected, rtol=1e-12, atol=1e-15)
        results.append(dict(method=name, seconds=times, median=float(np.median(times))))
mod._gpu_reference_marginal = original
output = dict(geometry=dict(references=256, sources=32768, dimension=1024),
              results=results, concurrent_production_workers=True)
Path('temp/gpu-marginal-benchmark.json').write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
