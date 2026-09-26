from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from rags.experiments.reference_opinions import (
    AGGREGATIONS, reference_marginal, selection_options,
)
from rags.experiments.resampling import value_mass
from rags.experiments.full_pool import select_full_pool
from rags.experiments.v3_protocol import opinion_method_protocols, validate_v3_config
from test_full_pool_resampling import Probe, inputs


def directions():
    refs=np.array([[1.,0.],[-1.,0.],[0.,1.]])
    src=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    return refs,src,np.array([.2,.3,.5])


@pytest.mark.parametrize('mode',AGGREGATIONS)
@pytest.mark.parametrize('competition',['global','local'])
def test_batched_matches_explicit_pair_matrix(mode,competition):
    refs,src,w=directions(); c=refs@src.T
    transformed=np.maximum(c,0) if mode=='zeroClipSum' else np.exp(c) if mode=='expSum' else c
    scores=c.max(axis=0) if mode=='maxSum' and competition=='global' else w@transformed
    expected=value_mass(scores,temperature=.05,gating=False)
    if competition=='local':
        expected=w@np.stack([value_mass(row,temperature=.05,gating=False) for row in transformed])
    beta,audit=reference_marginal(refs,src,w,aggregation=mode,competition=competition,
        reference_batch=1,source_batch=2,device='cpu')
    np.testing.assert_allclose(beta,expected,atol=1e-15)
    np.testing.assert_allclose(audit['source_scores'],scores,atol=1e-15)
    assert audit['score_workspace_shape']==[1,2]


def test_same_pair_kernel_local_is_global_with_changed_ref_prior():
    refs,src,w=directions();kernel=np.exp((refs@src.T)/.25); z=kernel.sum(axis=1)
    local=w@(kernel/z[:,None])
    prior=w/z
    equivalent=prior@kernel;equivalent/=equivalent.sum()
    np.testing.assert_allclose(local,equivalent,atol=1e-15)
    global_mass=w@kernel;global_mass/=global_mass.sum()
    assert not np.allclose(local,global_mass)
    # Current linearSum-global exponentiates AFTER aggregation, another distinction.
    linear=value_mass(w@(refs@src.T),temperature=.25,gating=False)
    assert not np.allclose(linear,global_mass)


@pytest.mark.parametrize('mode',AGGREGATIONS)
def test_duplicate_same_direction_blocks_preserves_weighted_methods(mode):
    refs,src,w=directions()
    a,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu')
    b,_=reference_marginal(np.repeat(refs,3,axis=0),src,np.repeat(w/3,3),aggregation=mode,device='cpu')
    np.testing.assert_allclose(a,b,atol=1e-15)


def test_zero_clip_all_negative_is_uniform_not_zero_mass():
    p,_=reference_marginal([[1.,0.]],np.array([[-1.,0.],[-.6,.8]]),[1.],
        aggregation='zeroClipSum',device='cpu')
    np.testing.assert_allclose(p,[.5,.5])


def test_global_max_ignores_alpha_but_zero_clip_keeps_it():
    refs,src,w=directions(); other=np.array([.8,.1,.1])
    for mode in ['maxSum','zeroClipSum']:
        a,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu')
        b,_=reference_marginal(refs,src,other,aggregation=mode,device='cpu')
        assert np.allclose(a,b) if mode=='maxSum' else not np.allclose(a,b)


def test_new_defaults_and_frozen_v3_are_distinct():
    config=json.loads(Path('configs/rags/main_v3_t005.json').read_text())
    for name,mode in [('less-main','maxSum'),('rags-main','zeroClipSum'),('rags-traj','zeroClipSum')]:
        assert selection_options(name)['aggregation']==mode
        assert selection_options(name,config)['aggregation']=='linearSum'
    assert selection_options('rags-main-expSum-local')==dict(aggregation='expSum',competition='local')
    validate_v3_config(config)
    config['opinion_aggregation_protocol']='reference-opinion-v1'
    config['method_protocols']=opinion_method_protocols(config['arms'])
    validate_v3_config(config)
    changed=deepcopy(config);changed['method_protocols']['rags-main']['reference_aggregation']='linearSum'
    with pytest.raises(ValueError,match='method_protocols'):validate_v3_config(changed)


@pytest.mark.parametrize('name',['less-main','less-linearSum','rags-main','rags-traj',
    'rags-main-maxSum','rags-traj-expSum','rags-main-linearSum-local','less-zeroClipSum-local'])
def test_production_dispatch_records_mode_without_changing_windows(name,tmp_path):
    out=select_full_pool(name,Probe(),run_dir=tmp_path,temperature=.05,**inputs())
    options=selection_options(name)
    assert out.selected.metadata['opinion_aggregation']==options['aggregation']
    assert out.selected.metadata['competition']==options['competition']
    assert out.name==name
    np.testing.assert_allclose(out.selected.probabilities.sum(),1.)
    assert out.selected.metadata['source_weighting'] in ('equal_block','equal_trajectory')


def test_exp_sum_is_exp_then_sum_then_outer_softmax_not_logsumexp():
    refs,src,w=directions();c=refs@src.T
    p,_=reference_marginal(refs,src,w,aggregation='expSum',temperature=.05,device='cpu')
    expected=value_mass(w@np.exp(c),temperature=.05,gating=False)
    np.testing.assert_allclose(p,expected)
    kernel=w@np.exp(c/.05);kernel/=kernel.sum()
    assert not np.allclose(p,kernel)


@pytest.mark.parametrize('filename',['main_reference_opinion_20260917.json','ablation_reference_opinion_20260917.json'])
def test_generated_configs_are_valid_and_historical_default_stays_linear(filename):
    config=json.loads((Path('configs/rags')/filename).read_text())
    validate_v3_config(config)
    assert config['preparation_status']=='definition_only_not_launched'
    old=json.loads(Path('configs/rags/main_v3_t005.json').read_text())
    validate_v3_config(old)
    assert selection_options('less-main',old)['aggregation']=='linearSum'


@pytest.mark.parametrize('mode',AGGREGATIONS)
def test_gpu_bounded_pairs_match_cpu(mode):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA not available')
    refs,src,w=directions()
    cpu,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu',reference_batch=1,source_batch=2)
    gpu,audit=reference_marginal(refs,src,w,aggregation=mode,device='cuda',reference_batch=1,source_batch=2)
    np.testing.assert_allclose(gpu,cpu,atol=1e-14)
    assert audit['compute_device']=='cuda'


def test_report_labels_do_not_rename_historical_artifact_slugs():
    from rags.experiments.v3_report import display_method
    old=json.loads(Path('configs/rags/main_v3_t005.json').read_text())
    new=json.loads(Path('configs/rags/main_reference_opinion_20260917.json').read_text())
    assert display_method('less-main',old)=='less-linearSum'
    assert display_method('rags-main',old)=='rags-main-linearSum'
    assert display_method('rags-traj',old)=='rags-traj-linearSum'
    assert display_method('less-main',new)=='less-main'
    assert old['arms']==['orig','rand','tsds','lp-mirror','less-main','rags-main','rags-traj']
