import numpy as np
import pytest
import torch

from rags.experiments.resampling import rags_marginal, InfeasibleReference


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('gating', [False, True])
@pytest.mark.parametrize('row_batch', [1, 7, 32, 128])
def test_gpu_softmax_audits_match_cpu(gating, row_batch):
    rng = np.random.default_rng(91)
    refs, sources = rng.normal(size=(19, 31)), rng.normal(size=(103, 31))
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    sources /= np.linalg.norm(sources, axis=1, keepdims=True)
    n, m = rng.integers(1, 100, 19), rng.integers(1, 100, 103)
    args = (refs, sources, n, m)
    expected, cpu = rags_marginal(*args, device='cpu', gating=gating, temperature=.25)
    actual, gpu = rags_marginal(*args, device='cuda', gating=gating,
                               temperature=.25, row_batch=row_batch)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    for c, g in zip(cpu['rows'], gpu['rows']):
        for key in ('reference_id', 'positive_pairs', 'nonpositive_pairs'):
            assert c[key] == g[key]
        for key in ('pi', 'nonpositive_mass', 'conditional_mass', 'joint_mass'):
            assert g[key] == pytest.approx(c[key], rel=1e-12, abs=1e-15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_gpu_rejects_infeasible_and_underflow():
    refs = np.array([[1., 0.]])
    sources = np.array([[-1., 0.], [-.5, .5]])
    with pytest.raises(InfeasibleReference):
        rags_marginal(refs, sources, [1], [1, 1], device='cuda', gating=True)
    with pytest.raises(ValueError, match='underflows alignment mass'):
        rags_marginal(refs, np.array([[1., 0.], [-1., 0.]]), [1], [1, 1],
                      device='cuda', gating=False, temperature=1e-9)
