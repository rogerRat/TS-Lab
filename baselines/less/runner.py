"""V5 adapted LESS: raw uniform-anchor means, weighted subtask cosine Max."""
from __future__ import annotations
import numpy as np
from rags.experiments.arms import ArmOutput,SelectionCost,WindowIndex
from baselines.trajectory import budget_config,trajectory_features
from .algorithms import unit_rows
from rags.experiments.reference_opinions import reference_scores
from rags.experiments.full_pool import ranked_selection

PROBABILITY_REVISION='less-subtask-max-rank-volume-v5'


def less_select(engine,*,reference_fragments,index,context_length,prediction_length,
                cover_density=.02,error_density=.01,split_density=.01,cache_dir=None,
                pool_manifest=None,seed=0,resampling=None,gradient_assets=None,
                aggregation='maxSum'):
    if gradient_assets is None or resampling is None:
        raise ValueError('LESS requires prepared assets and the V5 probability protocol; no additional BP')
    if aggregation!='maxSum':raise ValueError('LESS-Main uses weighted subtask Max')
    config=budget_config(index,cover_density,error_density,split_density)
    source,source_info=trajectory_features(engine,index,config,kind='less',gradient_assets=gradient_assets)
    ref_index=WindowIndex(reference_fragments,context_length=context_length,
        prediction_length=prediction_length,stride=index.stride)
    reference,ref_info=trajectory_features(engine,ref_index,config,kind='less',
        gradient_assets=gradient_assets,dataset=gradient_assets.reference_dataset)
    scores,audit=reference_scores(unit_rows(reference),unit_rows(source),
        np.full(len(reference),1./len(reference)),aggregation='maxSum')
    units=[np.arange(*index.bounds(int(fid))) for fid in source_info['trajectory_ids']]
    selected=ranked_selection(index,units,scores,pool_manifest=pool_manifest,
        metadata=dict(resampling,reference_audit=audit,reference_count=len(reference),
            less_probability_revision=PROBABILITY_REVISION,
            trajectory_representation='arithmetic_mean_of_distinct_raw_uniform_anchors'),
        temperature=resampling['temperature'])
    return ArmOutput('less-main',selected,[],SelectionCost(gradient_computations=0,
        index_bytes=source.nbytes,notes=dict(source=source_info,reference=ref_info)),
        dict(source_sampling=source_info,reference_sampling=ref_info,reference_audit=audit,
             resampling=selected.coverage()))
