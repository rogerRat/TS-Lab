"""Bounded-memory geometry of exhaustive, ordered reference gradients.

All vectors are native-loss Count-Sketch gradients in the frozen selection
space, not full parameter gradients. Undefined directions stay undefined.
"""
from __future__ import annotations

import numpy as np


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denom, -1, 1)) if denom > 0 else None


def unit_rows(g):
    g = np.asarray(g, dtype=np.float64)
    norms = np.linalg.norm(g, axis=1)
    return np.divide(g, norms[:, None], out=np.zeros_like(g), where=norms[:, None] > 0), norms


def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return dict(n=0, mean=None, p50=None, p90=None, p99=None, maximum=None)
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.quantile(x, .5)),
                p90=float(np.quantile(x, .9)), p99=float(np.quantile(x, .99)), maximum=float(x.max()))


def summarize_trajectory(read, n, *, lags=(1, 64, 576), chunk=1024, anchors=None):
    """read(lo,hi) returns true gradients at every consecutive legal start.

    O(chunk*D + n) memory; never forms a window-by-window Gram matrix.
    Anchor keys are zero-based legal-start indices from the frozen CRB replay.
    """
    if n < 1 or chunk < 1 or any(k < 1 for k in lags):
        raise ValueError('positive window count, chunk and lags required')
    norms = np.empty(n, dtype=np.float64)
    raw_sum = unit_sum = first = last = None
    energy = 0.
    tail = max(1, n // 4)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        g = np.asarray(read(lo, hi), dtype=np.float64)
        if g.ndim != 2 or len(g) != hi-lo or not np.isfinite(g).all():
            raise ValueError('expected finite independent gradients [N,D]')
        u, norms[lo:hi] = unit_rows(g)
        if raw_sum is None:
            raw_sum = np.zeros(g.shape[1]); unit_sum = raw_sum.copy()
            first = raw_sum.copy(); last = raw_sum.copy()
        raw_sum += g.sum(0); unit_sum += u.sum(0)
        energy += float(np.square(g).sum())
        if lo < tail:
            first += g[:min(hi, tail)-lo].sum(0)
        if hi > n-tail:
            last += g[max(0, n-tail-lo):].sum(0)
    mean, mean_unit = raw_sum / n, unit_sum / n
    positive = int(np.count_nonzero(norms))
    mean_norm = float(norms.mean())
    mean_energy = energy / n
    variance = max(0., mean_energy - float(mean @ mean))
    out = dict(windows=n, nonzero_windows=positive, zero_fraction=1-positive/n,
        mean_raw=mean.tolist(), mean_unit=mean_unit.tolist(), mean_norm=mean_norm,
        mean_squared_norm=mean_energy, raw_variance=variance,
        relative_raw_variance=variance/mean_energy if mean_energy else None,
        directional_variance=float(max(0., positive/n - mean_unit @ mean_unit)),
        nonzero_directional_variance=float(max(0., 1-np.square(unit_sum/positive).sum())) if positive else None,
        cancellation=1-float(np.linalg.norm(mean))/mean_norm if mean_norm else None,
        norm_cv=float(norms.std()/mean_norm) if mean_norm else None,
        norm_top1_mass=float(np.sort(norms)[-max(1, int(np.ceil(.01*n))):].sum()/norms.sum()) if mean_norm else None,
        norms=distribution(norms),
        early_late_cosine=cosine(first, last) if n >= 2 else None,
        early_late_relative_shift=float(np.linalg.norm(last/tail-first/tail)/mean_norm) if mean_norm and n >= 2 else None,
        drift_tail_windows=tail)
    for lag in sorted(set(lags)):
        changes, flips, relative = [], [], []
        for lo in range(0, max(0, n-lag), chunk):
            hi = min(n-lag, lo+chunk)
            a, b = np.asarray(read(lo, hi), dtype=np.float64), np.asarray(read(lo+lag, hi+lag), dtype=np.float64)
            ua, na = unit_rows(a); ub, nb = unit_rows(b)
            valid = (na > 0) & (nb > 0)
            cos = np.clip(np.einsum('ij,ij->i', ua[valid], ub[valid]), -1, 1)
            changes.extend((1-cos).tolist()); flips.extend((cos < 0).tolist())
            denom = na+nb
            relative.extend((2*np.linalg.norm(b-a, axis=1)[denom > 0]/denom[denom > 0]).tolist())
        out[f'lag{lag}_direction_change'] = distribution(changes)
        out[f'lag{lag}_negative_fraction'] = float(np.mean(flips)) if flips else None
        out[f'lag{lag}_relative_step'] = distribution(relative)
    curvature = []
    for lo in range(1, n-1, chunk):
        hi = min(n-1, lo+chunk)
        g = np.asarray(read(lo-1, hi+1), dtype=np.float64)
        denom = np.linalg.norm(g[:-2], axis=1)+2*np.linalg.norm(g[1:-1], axis=1)+np.linalg.norm(g[2:], axis=1)
        valid = denom > 0
        curvature.extend((np.linalg.norm(g[2:]-2*g[1:-1]+g[:-2], axis=1)[valid]/denom[valid]).tolist())
    out['relative_curvature'] = distribution(curvature)
    if anchors:
        keys = np.asarray(sorted(anchors), dtype=int)
        if keys[0] != 0 or keys[-1] != n-1:
            raise ValueError('CRB anchors must cover both trajectory endpoints')
        values = np.asarray([anchors[int(k)] for k in keys], dtype=np.float64)
        error, recon_sum, held_error, held_energy, held_n = 0., np.zeros_like(mean), 0., 0., 0
        directional = []
        for lo in range(0, n, chunk):
            hi = min(n, lo+chunk); rows = np.arange(lo, hi)
            left = np.maximum(0, np.searchsorted(keys, rows, side='right')-1)
            right = np.minimum(len(keys)-1, left+1)
            width = keys[right]-keys[left]
            alpha = np.divide(rows-keys[left], width, out=np.zeros(len(rows)), where=width > 0)
            approx = values[left]*(1-alpha[:, None])+values[right]*alpha[:, None]
            actual = np.asarray(read(lo, hi), dtype=np.float64)
            errors = np.square(actual-approx).sum(1)
            error += float(errors.sum()); recon_sum += approx.sum(0)
            held = ~np.isin(rows, keys)
            held_error += float(errors[held].sum()); held_energy += float(np.square(actual[held]).sum()); held_n += int(held.sum())
            ua, na = unit_rows(actual); ub, nb = unit_rows(approx)
            valid = (na > 0) & (nb > 0)
            directional.extend((1-np.clip(np.einsum('ij,ij->i', ua[valid], ub[valid]), -1, 1)).tolist())
        anchor_mean = values.mean(0)
        out['crb'] = dict(anchors=len(keys), relative_rmse=float(np.sqrt(error/energy)) if energy else None,
            nonanchor_windows=held_n, nonanchor_relative_rmse=float(np.sqrt(held_error/held_energy)) if held_energy else None,
            direction_error=distribution(directional), reconstructed_mean_cosine=cosine(recon_sum, mean),
            anchor_mean_cosine=cosine(anchor_mean, mean),
            anchor_mean_relative_bias=float(np.linalg.norm(anchor_mean-mean)/mean_norm) if mean_norm else None)
    return out


def summarize_task(rows):
    """Equal reference-fragment primary weighting matches frozen V5 scoring.

    Windows are NOT inferential replicates. Keep the complete fragment table
    for window-weighted and native-entity sensitivity analyses.
    """
    if not rows:
        raise ValueError('no reference trajectories')
    means = np.asarray([r['mean_raw'] for r in rows])
    units = np.asarray([r['mean_unit'] for r in rows])
    center, ucenter = means.mean(0), units.mean(0)
    within = float(np.mean([r['raw_variance'] for r in rows]))
    between = float(np.square(means-center).sum(1).mean())
    dwithin = float(np.mean([r['directional_variance'] for r in rows]))
    dbetween = float(np.square(units-ucenter).sum(1).mean())
    total_energy = float(np.mean([r['mean_squared_norm'] for r in rows]))
    out = dict(trajectories=len(rows), windows=sum(r['windows'] for r in rows),
        mean_raw=center.tolist(), mean_unit=ucenter.tolist(),
        within_directional_variance=dwithin, between_directional_variance=dbetween,
        total_directional_variance=dwithin+dbetween,
        within_raw_variance_fraction=within/total_energy if total_energy else None,
        between_raw_variance_fraction=between/total_energy if total_energy else None,
        between_fraction_of_variance=between/(within+between) if within+between else None)
    def flatten(value, prefix=''):
        for key, v in value.items():
            name = prefix+key
            if isinstance(v, dict):
                yield from flatten(v, name+'_')
            elif v is None or isinstance(v, (float, int)):
                yield name, v
    metrics = [dict(flatten(r)) for r in rows]
    for key in sorted(set().union(*(m.keys() for m in metrics))):
        vals = [m[key] for m in metrics if m.get(key) is not None]
        if key not in ('fragment_id',) and vals:
            out['trajectory_mean_'+key] = float(np.mean(vals))
    return out


def alignment_summary(source, reference):
    """MC source draws versus exact equal-trajectory reference expectation.

    E_source E_reference cos = E_source(unit(g)) dot E_reference(unit(g)).
    No reference subsampling and no quadratic pairwise matrix are needed.
    """
    source = np.asarray(source, dtype=np.float64)
    u, norms = unit_rows(source)
    r = np.asarray(reference['mean_unit'])
    observations = u @ r
    return dict(mean_pair_cosine=float(observations.mean()),
        mean_pair_cosine_mc_se=float(observations.std(ddof=1)/np.sqrt(len(source))) if len(source)>1 else None,
        cosine_of_raw_means=cosine(source.mean(0), reference['mean_raw']),
        source_mean_norm=float(norms.mean()), source_zero_fraction=float(np.mean(norms == 0)),
        samples=len(source), reference_scope='all_windows_equal_reference_fragment',
        source_scope='iid_draws_from_actual_frozen_selection_distribution')
