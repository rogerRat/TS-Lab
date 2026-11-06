from types import SimpleNamespace
import numpy as np
import pytest
from rage.crb import CRBConfig
from rage.experiments import full_pool
from rage.experiments.resampling import rage_global_marginal, value_mass




def test_global_linear_aggregation_matches_explicit_pair_product():
    rng = np.random.default_rng(9)
    refs = rng.normal(size=(7, 13)); refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    sources = rng.normal(size=(11, 13)); sources /= np.linalg.norm(sources, axis=1, keepdims=True)
    n = np.arange(1, 8); m = np.arange(1, 12)
    scores = (n / n.sum()) @ (refs @ sources.T)
    expected = value_mass(scores, m, temperature=.25, gating=False)
    actual, audit = rage_global_marginal(refs, sources, n, m, temperature=.25, gating=False)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(audit['source_scores'], scores, rtol=1e-13, atol=1e-15)
