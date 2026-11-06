"""Bounded pairwise reference opinion aggregation; raw cosines, no rank mapping."""
from __future__ import annotations
import numpy as np

AGGREGATIONS = ('linearSum', 'zeroClipSum', 'maxSum', 'expSum')
COMPETITIONS = ('global', 'local')
REVISION = 'reference-opinion-v1'
FAMILIES = {'less': 'less', 'rage-main': 'rage', 'rage-traj': 'rage_traj',
            'rage-uniform': 'rage_uniform'}
OPINION_ARMS = tuple(f'{family}-{mode}{suffix}' for family in FAMILIES
    for mode in AGGREGATIONS for suffix in ('', '-global', '-local'))
OPINION_BASES = {name: base for family, base in FAMILIES.items()
    for name in OPINION_ARMS if name.startswith(family + '-')}


def selection_options(name, config=None):
    """Frozen run protocols win over defaults; suffixes are explicit identities."""
    for family in sorted(FAMILIES, key=len, reverse=True):
        if name.startswith(family + '-'):
            suffix = name[len(family)+1:].split('-')
            if suffix[0] in AGGREGATIONS:
                if len(suffix) > 2 or (len(suffix) == 2 and suffix[1] not in COMPETITIONS):
                    raise ValueError('Invalid opinion arm suffix')
                return dict(aggregation=suffix[0], competition=suffix[1] if len(suffix)==2 else 'global')
    protocol = (config or {}).get('method_protocols', {}).get(name, {})
    if protocol:
        mode = protocol.get('reference_aggregation', 'linearSum')
        if mode in ('mean_cosine', 'mean_reference_cosines'):
            mode = 'linearSum'
        return dict(aggregation=mode, competition=protocol.get('competition', 'global'))
    if config is not None:  # Historical runs without explicit method definitions.
        return dict(aggregation='linearSum', competition='global')
    return dict(aggregation=('maxSum' if name == 'less-main' else
        'zeroClipSum' if name in ('rage-main', 'rage-traj') else 'linearSum'), competition='global')


def probability_revision(legacy, aggregation, competition):
    return legacy if (aggregation, competition)==('linearSum', 'global') else (
        f'{legacy}:{REVISION}:{aggregation}:{competition}')


def _transform(scores, aggregation):
    if aggregation == 'zeroClipSum':
        return np.maximum(scores, 0.)
    if aggregation == 'expSum':
        # User-defined expSum: exp(raw cosine), weighted sum, then outer Softmax.
        # NOT log-sum-exp or exp(cosine/T) followed by direct normalization.
        return np.exp(scores)
    return scores


def reference_marginal(reference_directions, source_directions, reference_weights,
                       *, aggregation='zeroClipSum', competition='global', temperature=.05,
                       reference_batch=128, source_batch=2048, device='auto'):
    """Equal source-unit base measure; local uses fixed weighted row mixtures.

    maxSum-global is an unweighted maximum over judges, not max(alpha*cos).
    maxSum-local degenerates to linearSum-local: each row has one opinion per
    source, so no cross-reference max exists before independent row Softmax.
    All source/ref directions are expected to be unit rows (zero rows allowed).
    """
    from .resampling import _DirectionRows, value_mass
    if aggregation not in AGGREGATIONS or competition not in COMPETITIONS:
        raise ValueError('Unknown reference aggregation/competition')
    refs, sources = _DirectionRows(reference_directions), _DirectionRows(source_directions)
    weights = np.asarray(reference_weights, dtype=np.float64)
    if (weights.shape != (refs.shape[0],) or refs.shape[1] != sources.shape[1]
            or not len(weights) or not sources.shape[0] or not np.isfinite(weights).all()
            or np.any(weights <= 0) or not np.isclose(weights.sum(),1.,rtol=0.,atol=1e-12)
            or temperature <= 0 or not np.isfinite(temperature)
            or reference_batch < 1 or source_batch < 1):
        raise ValueError('Invalid reference weights/dimensions/temperature/batches')
    target = None
    if device != 'cpu':
        import torch
        large = refs.shape[0]*sources.shape[0]*refs.shape[1] >= 10_000_000
        if device != 'auto' or (large and torch.cuda.is_available()):
            target = torch.device('cuda' if device=='auto' else device)
    scores = np.zeros(sources.shape[0], dtype=np.float64)
    if competition=='global' and aggregation=='maxSum':
        scores.fill(-np.inf)
    beta = np.zeros_like(scores)
    for rb in range(0,refs.shape[0],reference_batch):
        re = min(rb+reference_batch,refs.shape[0])
        ref = refs.batch(rb,re)
        ref_gpu = None
        if target is not None:
            import torch
            ref_gpu = torch.as_tensor(ref,device=target)
        def pair(sb,se):
            src = sources.batch(sb,se)
            if target is None:
                values = ref @ src.T
            else:
                import torch
                values = (ref_gpu @ torch.as_tensor(src,device=target).T).cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite reference cosine')
            return values
        row_max = np.full(re-rb,-np.inf)
        row_sum = np.zeros(re-rb)
        for sb in range(0,sources.shape[0],source_batch):
            se = min(sb+source_batch,sources.shape[0])
            values = _transform(pair(sb,se), aggregation)
            if competition=='global' and aggregation=='maxSum':
                scores[sb:se] = np.maximum(scores[sb:se],values.max(axis=0))
            else:
                scores[sb:se] += weights[rb:re] @ values
            if competition=='local':
                logits = values/temperature
                new_max = np.maximum(row_max, logits.max(axis=1))
                row_sum = row_sum*np.exp(row_max-new_max) + np.exp(logits-new_max[:,None]).sum(axis=1)
                row_max = new_max
        if competition=='local':
            for sb in range(0,sources.shape[0],source_batch):
                se = min(sb+source_batch,sources.shape[0])
                values = _transform(pair(sb,se), aggregation)
                conditional = np.exp(values/temperature-row_max[:,None])/row_sum[:,None]
                beta[sb:se] += weights[rb:re] @ conditional
    if competition=='global':
        beta = value_mass(scores, temperature=temperature, gating=False)
    else:
        beta /= beta.sum()  # Roundoff correction only, not another Softmax.
    audit = dict(reference_weights=weights.tolist(), reference_weight_sum=float(weights.sum()),
        source_scores=scores.tolist(), source_scores_role='selection_score' if competition=='global'
        else 'diagnostic_only_not_a_softmax_score', aggregation=aggregation, competition=competition,
        reference_aggregation_revision=REVISION, temperature=float(temperature),
        exp_inner_scale=1. if aggregation=='expSum' else None,
        reference_weights_apply='ignored_by_global_max' if aggregation=='maxSum' and competition=='global'
        else 'score_sum' if competition=='global' else 'post_row_softmax_mixture',
        fixed_reference_mass=competition=='local', source_length_weighting=False,
        local_max_degenerates_to_linear=aggregation=='maxSum' and competition=='local',
        score_workspace_shape=[min(reference_batch,refs.shape[0]),min(source_batch,sources.shape[0])],
        compute_device=str(target) if target is not None else 'cpu')
    return beta, audit
