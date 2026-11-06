from pathlib import Path
p=Path("src/rage/experiments/resampling.py")
s=p.read_text()
needle='''    # Do not renormalize across reference rows: their separate budgets are final.
    return beta, dict(pi=pi.tolist(), rows=audits, temperature=temperature,
                      mapping=("exp(cosine/temperature), pairwise cosine>0" if gating else
                               "exp(cosine/temperature), all finite cosines (no hard gating)"),
                      marginal_mass=float(beta.sum()))


def value_mass'''
insert='''    # Do not renormalize across reference rows: their separate budgets are final.
    return beta, dict(pi=pi.tolist(), rows=audits, temperature=temperature,
                      mapping=("exp(cosine/temperature), pairwise cosine>0" if gating else
                               "exp(cosine/temperature), all finite cosines (no hard gating)"),
                      marginal_mass=float(beta.sum()))


def rage_global_marginal(reference_directions, source_directions, reference_counts,
                         source_counts, *, reference_ids=None, temperature=0.1,
                         gating=True):
    """Aggregate all reference alignment into one global source competition."""
    refs = np.asarray(reference_directions, dtype=np.float64)
    sources = np.asarray(source_directions, dtype=np.float64)
    n = np.asarray(reference_counts, dtype=np.float64)
    m = np.asarray(source_counts, dtype=np.float64)
    if (refs.ndim != 2 or sources.ndim != 2 or refs.shape[1] != sources.shape[1]
            or len(n) != len(refs) or len(m) != len(sources)
            or not len(n) or not len(m) or np.any(n <= 0) or np.any(m <= 0)):
        raise ValueError("CRB directions require matching nonempty legal-start counts")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    weights = n / n.sum()
    pair_scores = refs @ sources.T
    scores = weights @ pair_scores
    beta = value_mass(scores, m, temperature=temperature, gating=gating)
    audit = dict(reference_ids=None if reference_ids is None else list(reference_ids),
        reference_weights=weights.tolist(), source_scores=scores.tolist(),
        positive_pairs_per_source=(pair_scores > 0).sum(axis=0).astype(int).tolist(),
        nonpositive_pairs=int((pair_scores <= 0).sum()),
        mapping=("exp(weighted_mean_cosine/temperature), global positive gating" if gating
                 else "exp(weighted_mean_cosine/temperature), all finite cosines"),
        marginal_mass=float(beta.sum()))
    return beta, audit


def value_mass'''
if needle not in s: raise SystemExit("resampling needle missing")
p.write_text(s.replace(needle,insert,1))

p=Path("src/rage/experiments/full_pool.py")
s=p.read_text()
s=s.replace('from .resampling import REVISION, ResampledWindows, rage_marginal, random_distribution',
'''from .resampling import (REVISION, ResampledWindows, rage_global_marginal,
                         rage_marginal, random_distribution)''')
s=s.replace('GATELESS_ARMS = ("rage_gateless", "rage_traj_gls")',
'''GATELESS_ARMS = ("rage_gateless", "rage_traj_gls")
GLOBAL_ARMS = ("rage_global", "rage_traj_global")''')
s=s.replace('TEMPERATURE_DEFAULT_FORM = ("rage_gateless", "random", "lessvg_t025", "tsds", "lpv_t025", "rage_traj_gls")',
'TEMPERATURE_DEFAULT_FORM = ("rage_gateless", "random", "lessvg_t025", "tsds", "lpv_t025", "rage_traj_gls", "rage_global", "rage_traj_global")')
s=s.replace('rage_gateless="rage", rage_traj_gls="rage_traj")',
'rage_gateless="rage", rage_traj_gls="rage_traj", rage_global="rage_global", rage_traj_global="rage_traj_global")')
s=s.replace('GATELESS_ARMS + VALUE_ARMS:', 'GATELESS_ARMS + GLOBAL_ARMS + VALUE_ARMS:')
needle='''    elif base == 'rage_uniform':
'''
insert='''    elif base == 'rage_global':
        output = _rage_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif base == 'rage_traj_global':
        output = _rage_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
    elif base == 'rage_uniform':
'''
if needle not in s: raise SystemExit("full pool branch needle missing")
s=s.replace(needle,insert,1)
s += '''

def _rage_global(engine, references, pool, index, manifest, config, assets, metadata, temperature):
    source, counters = assets.source_blocks, assets.source_counters
    ref, results = assets.reference_blocks, assets.reference_results
    if not ref or not source:
        raise ValueError("CRB retained no usable reference/source blocks; full-pool distribution infeasible")
    beta, audit = rage_global_marginal(
        np.stack([b.direction for b in ref]), np.stack([b.direction for b in source]),
        [len(b.start_positions) for b in ref], [len(b.start_positions) for b in source],
        reference_ids=[[int(b.fragment_id), int(b.start_positions[0]), int(b.start_positions[-1])] for b in ref],
        temperature=temperature, gating=True)
    audit["reference_coverage"] = [dict(fragment_id=int(f.fragment_id),
        legal_starts=len(result.valid_starts), retained_starts=sum(len(b.start_positions) for b in result.blocks))
        for f, result in zip(references, results)]
    selected = ResampledWindows.from_index(index, [b.start_positions for b in source], beta,
        pool_manifest=manifest, metadata=dict(metadata, scoring_unit="CRB_block_global",
                                               reference_audit=audit))
    return ArmOutput("rage_global", selected, [], SelectionCost(gradient_computations=0,
        index_bytes=sum(b.direction.nbytes for b in source) + beta.nbytes,
        notes={"pool_cached": True, "reference_cached": True, "pool": counters}),
        {"reference_audit": audit, "resampling": selected.coverage()})


def _rage_traj_global(index, manifest, config, assets, metadata, temperature):
    refs, source = assets.reference_trajectories, assets.source_trajectories
    if not refs or not source:
        raise ValueError('RAGE-Traj-Global requires retained RAGE trajectory assets')
    ref_counts = np.array([len(t.start_positions) for t in refs])
    source_counts = np.array([len(t.start_positions) for t in source])
    if np.any(np.array([np.linalg.norm(t.mean_gradient) for t in refs]) < config.mean_epsilon):
        raise ValueError('near-zero reference trajectory aggregate')
    source_directions = np.stack([t.direction for t in source])
    source_directions[np.linalg.norm(source_directions, axis=1) < config.mean_epsilon] = 0
    beta, audit = rage_global_marginal(np.stack([t.direction for t in refs]), source_directions,
        ref_counts, source_counts, reference_ids=[int(t.fragment_id) for t in refs],
        temperature=temperature)
    audit.update(support='union-of-retained-RAGE-block-owned-starts',
        aggregation='raw-linear-anchor-contribution-weighted',
        reference_trajectory_shares=(ref_counts / ref_counts.sum()).tolist(),
        source_retained_windows=int(source_counts.sum()), reference_retained_windows=int(ref_counts.sum()))
    selected = ResampledWindows.from_index(index, [t.start_positions for t in source], beta,
        pool_manifest=manifest, metadata=dict(metadata, scoring_unit='RAGE_trajectory_global',
                                               reference_audit=audit))
    return ArmOutput('rage_traj_global', selected, [], SelectionCost(gradient_computations=0,
        index_bytes=sum(t.mean_gradient.nbytes for t in source) + beta.nbytes,
        notes=dict(pool_cached=True, aggregation_incremental_only=True)),
        dict(reference_audit=audit, resampling=selected.coverage()))
'''
p.write_text(s)
