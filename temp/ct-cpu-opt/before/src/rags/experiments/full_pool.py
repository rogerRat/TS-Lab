"""Canonical V5 full-pool selection from frozen, previously prepared assets."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .reference_opinions import (OPINION_ARMS,OPINION_BASES,UNI_ARMS,is_block_uni_arm,is_win_uni_arm,selection_options,
                                reference_scores,weight_placement_for)
from .ranking import (rank_mass,frozen_rank_protocol,validate_frozen_rank_protocol,
                      DEFAULT_TEMPERATURE,uniformize_intra_trajectory,uniformize_trajectory_windows,
                      UNIFORMIZATION_REVISION,WINDOW_UNIFORMIZATION_REVISION)
from .arms import ArmOutput,SelectionCost
from .resampling import REVISION,ResampledWindows,random_distribution

ARMS = ('rand','orig','less-main') + OPINION_ARMS + ('tsds','lp-mirror')
ARM_BASE = dict(OPINION_BASES,rand='random',orig='original',tsds='tsds',**{'lp-mirror':'lp_mirror'})
RAGS_PROBABILITY_REVISION = 'rags-reference-score-rank-volume-v5'
COSINE_NUMERICS = 'float64-norm-exact-unit-v1'


def ranked_selection(index,units,scores,*,pool_manifest,metadata,temperature=DEFAULT_TEMPERATURE):
    if sum(len(unit) for unit in units)!=len(index):raise ValueError("Source units must own the full legal-start pool")
    rank,probabilities,audit=rank_mass(scores,[len(unit) for unit in units],temperature=temperature)
    return ResampledWindows.from_index(index,units,probabilities,pool_manifest=pool_manifest,
        metadata=dict(metadata,native_scores=np.asarray(scores).tolist(),rank_probability=audit,
            frozen_rank_protocol=frozen_rank_protocol(),probability_mapping=audit['rank_probability_revision'],
            source_weighting='legal_window_counts',temperature=float(temperature)))


def validate_cached_selection(metadata,*,arm,identity,manifest,crb,temperature):
    if metadata.get('pool_manifest')!=manifest or metadata.get('model',identity)!=identity:
        raise ValueError('Cached probability artifact belongs to another pool/model')
    if ARM_BASE[arm] in ('rags','rags_traj','less'):
        validate_frozen_rank_protocol(metadata.get('frozen_rank_protocol'))
        if metadata.get('scoring_revision')!=REVISION or metadata.get('probability_mapping')!='object-rank-exponential-interval-v5':
            raise ValueError('Cached scoring/probability identity differs')
        if (metadata.get('crb')!=asdict(crb) or metadata.get('source_weighting')!='legal_window_counts'
                or metadata.get('cosine_numerics')!=COSINE_NUMERICS
                or metadata.get('opinion_aggregation')!=selection_options(arm)['aggregation']
                or metadata.get('opinion_weight_placement')!=weight_placement_for(arm)
                or metadata.get('competition')!='global'):
            raise ValueError('Cached V5 method/CRB/probability differs; use a new run')
        expected_uniformization = (WINDOW_UNIFORMIZATION_REVISION if is_win_uni_arm(arm)
            else UNIFORMIZATION_REVISION if is_block_uni_arm(arm) else None)
        if metadata.get('intra_trajectory_uniformization') != expected_uniformization:
            raise ValueError('Cached uni method identity differs; use a new run')
    if metadata.get('temperature') is not None and metadata['temperature']!=temperature:
        raise ValueError('Cached selection temperature differs; use a new run')


def select_full_pool(name,engine,*,reference_fragments,pool,index,pool_manifest,
                     model_identity,crb_config,cache_dir=None,temperature=DEFAULT_TEMPERATURE,
                     dataset_tag='',seed=0,run_dir=None,gradient_assets=None,
                     reference_aggregation=None,competition=None,reference_weight_placement=None):
    if name not in ARMS or name=='orig':raise ValueError(f'Unknown training arm: {name}')
    base=ARM_BASE[name];options=selection_options(name)
    aggregation=reference_aggregation or options['aggregation']
    placement=reference_weight_placement or weight_placement_for(name)
    if (aggregation!=options['aggregation'] or (competition or 'global')!='global'
            or placement!=weight_placement_for(name)):
        raise ValueError('Named V5 method does not match requested aggregation')
    if base in ('rags','rags_traj','less'):
        if gradient_assets is None:
            raise ValueError('Prepared gradients required; additional BP in selection is forbidden')
        if temperature!=DEFAULT_TEMPERATURE:
            raise ValueError('Frozen V5 ranking temperature cannot be overridden')
    started=time.perf_counter()
    metadata=dict(model=model_identity,temperature=float(temperature),crb=asdict(crb_config),
        cosine_numerics=COSINE_NUMERICS,task=dataset_tag,scoring_revision=REVISION,
        opinion_aggregation=aggregation,competition='global',opinion_weight_placement=placement,
        additional_bp_allowed=False)
    if base=='random':
        selected=random_distribution(index,pool_manifest=pool_manifest)
        output=ArmOutput(name,selected,[],SelectionCost(),{'resampling':selected.coverage()})
    elif base in ('rags','rags_traj'):
        output=_rags(name,index,pool_manifest,gradient_assets,metadata,temperature,
                     trajectory=base=='rags_traj',uniformize=("win" if is_win_uni_arm(name) else "block" if is_block_uni_arm(name) else None))
    else:
        common=dict(reference_fragments=reference_fragments,index=index,
            context_length=index.context_length,prediction_length=index.prediction_length,
            pool_manifest=pool_manifest,cache_dir=Path(cache_dir) if cache_dir else None,resampling=metadata)
        density=dict(cover_density=crb_config.cover_density,error_density=crb_config.error_density,
                     split_density=crb_config.split_density)
        if base=='less':
            from baselines.less.runner import less_select
            output=less_select(engine,gradient_assets=gradient_assets,**common,**density,
                seed=seed,aggregation=aggregation)
        elif base=='tsds':
            from baselines.tsds.runner import tsds_select
            output=tsds_select(engine,**common,**density,seed=seed)
        else:
            from baselines.lp_mirror.runner import lp_mirror_select
            output=lp_mirror_select(engine,**common,seed=seed,mapping='value')
    output.name=name;output.cost.wall_seconds=time.perf_counter()-started
    if 'source' in output.cost.notes:output.cost.notes['pool_cached']=output.cost.notes['source']['pool_cached']
    output.diagnostics['resampling_revision']=REVISION
    if gradient_assets is not None:
        output.diagnostics['gradient_assets']=dict(path=str(gradient_assets.cache.path),
            computed_windows=gradient_assets.cache.computed_windows,format=gradient_assets.cache.manifest['format'])
    return output


def _rags(name,index,manifest,assets,metadata,temperature,*,trajectory,uniformize=None):
    if uniformize and trajectory:
        raise ValueError('Intra-trajectory uniformization applies to the block representation only')
    from baselines.less.algorithms import unit_rows
    if trajectory:
        refs,source=assets.reference_trajectories,assets.source_trajectories
        if len(refs)!=len(assets.reference_results):raise ValueError('Every reference trajectory must have a representative')
        weights=np.full(len(refs),1./len(refs)) if refs else []
        ref_directions=[unit_rows(t.mean_gradient[None,:])[0] for t in refs]
        source_directions=[unit_rows(t.mean_gradient[None,:])[0] for t in source]
        representation='arithmetic_mean_of_distinct_raw_CRB_anchors'
    else:
        refs,source=assets.reference_blocks,assets.source_blocks
        legal={int(r.fragment_id):len(r.valid_starts) for r in assets.reference_results}
        owned={fid:0 for fid in legal}
        for block in refs:owned[int(block.fragment_id)]+=len(block.start_positions)
        if not legal or any(n<=0 or owned[fid]!=n for fid,n in legal.items()):
            raise ValueError('All reference legal starts must be owned exactly once')
        weights=[len(b.start_positions)/(len(legal)*legal[int(b.fragment_id)]) for b in refs]
        ref_directions=[b.direction for b in refs];source_directions=[b.direction for b in source]
        representation='raw_endpoint_geometric_midpoint_interpolation'
    if not refs or not source:raise ValueError('Prepared reference/source representatives are missing')
    scores,audit=reference_scores(ref_directions,source_directions,weights,
        aggregation=metadata['opinion_aggregation'])
    audit.update(reference_count=len(assets.reference_results),reference_ids=[int(r.fragment_id) for r in refs],
                 representative=representation)
    selected=ranked_selection(index,[b.start_positions for b in source],scores,
        pool_manifest=manifest,metadata=dict(metadata,scoring_unit='trajectory' if trajectory else 'CRB_block',
            reference_audit=audit,rags_probability_revision=RAGS_PROBABILITY_REVISION,
            trajectory_representation=representation),temperature=temperature)
    if uniformize:
        trajectory_ids=[]
        for block in source:
            owners=np.unique(index.fragment_ids[np.asarray(block.start_positions,dtype=np.int64)])
            if len(owners)!=1:
                raise ValueError('Each CRB block must belong to exactly one trajectory for RAGS-uni')
            trajectory_ids.append(int(owners[0]))
        if uniformize == "block":
            selected=uniformize_intra_trajectory(selected,np.asarray(trajectory_ids,dtype=np.int64))
        elif uniformize == "win":
            selected=uniformize_trajectory_windows(selected,np.asarray(trajectory_ids,dtype=np.int64))
        else:
            raise ValueError("Unknown intra-trajectory deployment")
    return ArmOutput(name,selected,[],SelectionCost(gradient_computations=0,index_bytes=scores.nbytes,
        notes=dict(pool_cached=True,prepared_assets_only=True)),
        dict(reference_audit=audit,resampling=selected.coverage()))
