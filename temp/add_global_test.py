from pathlib import Path
p=Path("tests/test_full_pool_resampling.py")
s=p.read_text()
if "test_global_rags_aggregates_before_competition" not in s:
    s += '''

def test_global_rags_aggregates_before_competition():
    from rags.experiments.resampling import rags_global_marginal, rags_marginal
    import numpy as np
    refs = np.array([[1., 0.], [-1., 0.]])
    sources = np.array([[1., 0.], [-1., 0.], [0., 1.]])
    global_mass, global_audit = rags_global_marginal(
        refs, sources, [1, 1], [1, 1, 1], temperature=.25, gating=False)
    row_mass, _ = rags_marginal(
        refs, sources, [1, 1], [1, 1, 1], temperature=.25, gating=False)
    assert np.isclose(global_mass.sum(), 1.)
    assert np.isclose(row_mass.sum(), 1.)
    assert not np.allclose(global_mass, row_mass)
    assert len(global_audit["source_scores"]) == len(sources)
'''
p.write_text(s)
