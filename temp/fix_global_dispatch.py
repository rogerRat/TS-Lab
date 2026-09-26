from pathlib import Path

p = Path('src/rags/experiments/full_pool.py')
s = p.read_text()
old = '''    elif name == 'rags_global':
        output = _rags_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif name == 'rags_traj_global':
        output = _rags_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
'''
assert s.count(old) == 1
s = s.replace(old, '', 1)
s = s.replace('    elif base == "rags":\n', old + '    elif base == "rags":\n', 1)
s = s.replace('    source_directions[np.linalg.norm(source_directions, axis=1) < config.mean_epsilon] = 0',
'''    source_norms = np.array([np.linalg.norm(t.mean_gradient) for t in source])
    source_directions[(source_norms < config.mean_epsilon) | (source_norms == 0)] = 0''')
p.write_text(s)

p = Path('src/rags/experiments/resampling.py')
s = p.read_text()
s = s.replace('''    pair_scores = refs @ sources.T
    scores = weights @ pair_scores
''', '''    # Linearity avoids materializing the reference-by-source product matrix.
    # Keep the aggregate's norm: normalizing it would change the cosine sum.
    aggregate = weights @ refs
    scores = sources @ aggregate
''', 1)
s = s.replace('''        positive_pairs_per_source=(pair_scores > 0).sum(axis=0).astype(int).tolist(),
        nonpositive_pairs=int((pair_scores <= 0).sum()),
''', '''        aggregation='reference_window_weighted_mean_cosine',
        global_nonpositive_scores=int((scores <= 0).sum()),
''', 1)
p.write_text(s)

p = Path('tests/test_full_pool_global.py')
p.write_text('''from types import SimpleNamespace
import numpy as np
import pytest
from rags.crb import CRBConfig
from rags.experiments import full_pool
from rags.experiments.resampling import rags_global_marginal, value_mass


@pytest.mark.parametrize('name,implementation', [
    ('rags_global', '_rags_global'), ('rags_traj_global', '_rags_traj_global')])
def test_formal_global_dispatch_does_not_call_original_kernel(monkeypatch, name, implementation):
    calls = []
    def selected_kernel(*args, **kwargs):
        calls.append(implementation)
        return SimpleNamespace(name=name, cost=SimpleNamespace(wall_seconds=0, notes={}), diagnostics={})
    def wrong_kernel(*args, **kwargs):
        pytest.fail('Global arm dispatched to the original per-reference RAGS kernel')
    monkeypatch.setattr(full_pool, implementation, selected_kernel)
    monkeypatch.setattr(full_pool, '_rags', wrong_kernel)
    monkeypatch.setattr(full_pool, '_rags_traj', wrong_kernel)
    assets = SimpleNamespace(cache=SimpleNamespace(path='test', computed_windows=0,
                                                   manifest={'format':'test'}))
    out = full_pool.select_full_pool(name, None, reference_fragments=[], pool=[], index=None,
        pool_manifest={}, model_identity={}, crb_config=CRBConfig(), gradient_assets=assets)
    assert calls == [implementation]
    assert out.name == name


def test_global_linear_aggregation_matches_explicit_pair_product():
    rng = np.random.default_rng(9)
    refs = rng.normal(size=(7, 13)); refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    sources = rng.normal(size=(11, 13)); sources /= np.linalg.norm(sources, axis=1, keepdims=True)
    n = np.arange(1, 8); m = np.arange(1, 12)
    scores = (n / n.sum()) @ (refs @ sources.T)
    expected = value_mass(scores, m, temperature=.25, gating=False)
    actual, audit = rags_global_marginal(refs, sources, n, m, temperature=.25, gating=False)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(audit['source_scores'], scores, rtol=1e-13, atol=1e-15)
''')
print('Global dispatch, bounded-memory aggregation and integration tests installed')
