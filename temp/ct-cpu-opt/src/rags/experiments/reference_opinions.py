"""Frozen reference-weighted scores, with bounded CPU/GPU pairwise tiles."""
from __future__ import annotations
import numpy as np

AGGREGATIONS = ('linearSum', 'zeroClipSum', 'maxSum', 'expSum')
REVISION = 'reference-weighted-scores-v5'
FAMILIES = {'less-main': 'less', 'rags-block': 'rags', 'rags-traj': 'rags_traj'}
# Intra-trajectory uniformized variants of the two strongest block aggregations.
# Scoring is identical to the base block arm; only the final block probability
# vector is transformed (trajectory mass = sum of its blocks, uniform interior).
UNI_MODES = ('expSum', 'linearSum')
BLOCK_UNI_ARMS = tuple(f'rags-block-{mode}-block-uni' for mode in UNI_MODES)
WIN_UNI_ARMS = tuple(f'rags-block-{mode}-win-uni' for mode in UNI_MODES)
LEGACY_UNI_ARMS = tuple(f'rags-block-{mode}-uni' for mode in UNI_MODES)
UNI_ARMS = LEGACY_UNI_ARMS + BLOCK_UNI_ARMS + WIN_UNI_ARMS
OPINION_ARMS = (tuple(f'{family}-{mode}' for family in ('rags-block','rags-traj') for mode in AGGREGATIONS)
                + UNI_ARMS)
OPINION_BASES = {f'{family}-{mode}': base for family,base in FAMILIES.items()
                if family != 'less-main' for mode in AGGREGATIONS}
OPINION_BASES.update({arm: 'rags' for arm in UNI_ARMS})
OPINION_BASES['less-main'] = 'less'


def is_uni_arm(name):
    return name in UNI_ARMS


def is_block_uni_arm(name):
    return name in LEGACY_UNI_ARMS + BLOCK_UNI_ARMS


def is_win_uni_arm(name):
    return name in WIN_UNI_ARMS


def selection_options(name, config=None):
    if name == 'less-main':
        return dict(aggregation='maxSum', competition='global')
    for mode in UNI_MODES:
        if name in (f'rags-block-{mode}-uni', f'rags-block-{mode}-block-uni',
                    f'rags-block-{mode}-win-uni'):
            return dict(aggregation=mode, competition='global')
    for family in ('rags-block','rags-traj'):
        if name.startswith(family+'-') and name[len(family)+1:] in AGGREGATIONS:
            return dict(aggregation=name[len(family)+1:], competition='global')
    return dict(aggregation='linearSum', competition='global')


def weight_placement_for(name, config=None):
    return 'after_transform' if selection_options(name)['aggregation']=='expSum' else 'before_transform'


def probability_revision(legacy, aggregation, competition='global', weight_placement=None):
    return f'{REVISION}:{aggregation}:object-rank-exponential-interval-v5'


def reference_scores(reference_directions, source_directions, reference_weights,
                     *, aggregation='linearSum', reference_batch=128,
                     source_batch=2048, device='auto'):
    return reference_scores_many(reference_directions, source_directions, reference_weights,
        aggregations=(aggregation,), reference_batch=reference_batch,
        source_batch=source_batch, device=device)[aggregation]


def reference_scores_many(reference_directions, source_directions, reference_weights,
                          *, aggregations=AGGREGATIONS, reference_batch=128,
                          source_batch=2048, device='auto'):
    """Return scores only; no Softmax, probability mixture, gating or BP.

    max=max(alpha*cos); linear=sum(alpha*cos); relu=sum(alpha*relu(cos));
    exp=sum(alpha*exp(cos)), inner temperature 1. Raw representatives must
    have been averaged/interpolated before their direction normalization.
    """
    from .resampling import _DirectionRows
    aggregations = tuple(dict.fromkeys(aggregations))
    refs, sources = _DirectionRows(reference_directions), _DirectionRows(source_directions)
    weights = np.asarray(reference_weights, dtype=np.float64)
    if (not aggregations or any(mode not in AGGREGATIONS for mode in aggregations) or weights.shape != (refs.shape[0],)
            or not len(weights) or not sources.shape[0] or refs.shape[1]!=sources.shape[1]
            or not np.isfinite(weights).all() or np.any(weights<=0)
            or not np.isclose(weights.sum(),1.,rtol=0.,atol=1e-12)
            or reference_batch<1 or source_batch<1):
        raise ValueError('Invalid reference aggregation/weights/dimensions/batches')
    target = None
    if device != 'cpu':
        import torch
        large = refs.shape[0]*sources.shape[0]*refs.shape[1] >= 10_000_000
        if device != 'auto' or (large and torch.cuda.is_available()):
            target = torch.device('cuda' if device=='auto' else device)
    scores_by_mode = {mode: np.full(sources.shape[0], -np.inf if mode=='maxSum' else 0.,dtype=np.float64)
                      for mode in aggregations}
    source_gpu = None
    if target is not None:
        import torch
        source_gpu = torch.empty(sources.shape,dtype=torch.float64,device=target)
        for sb in range(0,sources.shape[0],source_batch):
            se=min(sb+source_batch,sources.shape[0])
            source_gpu[sb:se]=torch.from_numpy(sources.batch(sb,se))
    for rb in range(0,refs.shape[0],reference_batch):
        re=min(rb+reference_batch,refs.shape[0]);ref=refs.batch(rb,re)
        if target is not None:
            ref_gpu=torch.as_tensor(ref,device=target)
        for sb in range(0,sources.shape[0],source_batch):
            se=min(sb+source_batch,sources.shape[0])
            values=(ref @ sources.batch(sb,se).T if target is None else
                    (ref_gpu @ source_gpu[sb:se].T).cpu().numpy())
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite reference cosine')
            # Share only the immutable cosine tile. Keep each arm's exact
            # FP64 transform, reduction, tile boundaries and summation order.
            for aggregation, scores in scores_by_mode.items():
                if aggregation=='maxSum':
                    scores[sb:se]=np.maximum(scores[sb:se],(weights[rb:re,None]*values).max(axis=0))
                else:
                    transformed = (np.maximum(values,0.) if aggregation=='zeroClipSum' else
                                   np.exp(values) if aggregation=='expSum' else values)
                    scores[sb:se]+=weights[rb:re] @ transformed
    return {aggregation: (scores, dict(reference_aggregation_revision=REVISION, aggregation=aggregation,
        competition='global', source_scores=scores.tolist(), source_scores_role='selection_score',
        reference_weights=weights.tolist(),reference_weight_sum=float(weights.sum()),
        reference_weights_stage='score_aggregation',fixed_reference_mass=False,
        exp_inner_temperature=1. if aggregation=='expSum' else None,
        score_workspace_shape=[min(reference_batch,refs.shape[0]),min(source_batch,sources.shape[0])],
        compute_device=str(target) if target is not None else 'cpu', additional_bp_allowed=False))
        for aggregation, scores in scores_by_mode.items()}
