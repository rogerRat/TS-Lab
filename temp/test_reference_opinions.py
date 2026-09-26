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
        reference_batch=1,source_batch=2,device='cpu',weight_placement='after_transform')
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
    a,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu',weight_placement='after_transform')
    b,_=reference_marginal(np.repeat(refs,3,axis=0),src,np.repeat(w/3,3),aggregation=mode,device='cpu',weight_placement='after_transform')
    np.testing.assert_allclose(a,b,atol=1e-15)


def test_zero_clip_all_negative_is_uniform_not_zero_mass():
    p,_=reference_marginal([[1.,0.]],np.array([[-1.,0.],[-.6,.8]]),[1.],
        aggregation='zeroClipSum',device='cpu')
    np.testing.assert_allclose(p,[.5,.5])


def test_global_max_ignores_alpha_but_zero_clip_keeps_it():
    refs,src,w=directions(); other=np.array([.8,.1,.1])
    for mode in ['maxSum','zeroClipSum']:
        a,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu',weight_placement='after_transform')
        b,_=reference_marginal(refs,src,other,aggregation=mode,device='cpu',weight_placement='after_transform')
        assert np.allclose(a,b) if mode=='maxSum' else not np.allclose(a,b)


def test_new_defaults_and_frozen_v3_are_distinct():
    config=json.loads(Path('configs/rags/main_v3_t005.json').read_text())
    for name,mode in [('less-main','maxSum'),('rags-main','linearSum'),('rags-traj','linearSum')]:
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
    p,_=reference_marginal(refs,src,w,aggregation='expSum',temperature=.05,device='cpu',weight_placement='after_transform')
    expected=value_mass(w@np.exp(c),temperature=.05,gating=False)
    np.testing.assert_allclose(p,expected)
    kernel=w@np.exp(c/.05);kernel/=kernel.sum()
    assert not np.allclose(p,kernel)


@pytest.mark.parametrize('filename',['main_reference_opinion_20260917.json','ablation_reference_opinion_20260917.json'])
def test_generated_configs_are_valid_and_historical_default_stays_linear(filename):
    config=json.loads((Path('configs/rags')/filename).read_text())
    validate_v3_config(config)
    assert config['preparation_status'] in ('definition_only_not_launched','superseded_not_launched')
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
    assert display_method('rags-main',old)=='rags-block-linearSum'
    assert display_method('rags-traj',old)=='rags-traj-linearSum'
    assert display_method('less-main',new)=='less-main'
    assert old['arms']==['orig','rand','tsds','lp-mirror','less-main','rags-main','rags-traj']



def test_neutral_matrix_exact_variants_and_no_main_method():
    from rags.experiments.v3_protocol import NEUTRAL_MATRIX
    config=json.loads(Path('configs/rags/matrix_reference_opinion_20260917.json').read_text())
    validate_v3_config(config)
    assert len(NEUTRAL_MATRIX)==13
    assert config['arms']==list(NEUTRAL_MATRIX)
    assert config['rags_main_method'] is None
    for name in config['arms']:
        if name.startswith('rags-'):
            p=config['method_protocols'][name]
            assert p['main_method_selected'] is False
            assert p['method_role']=='unranked_variant'
            assert p['competition']=='global'
    for student in config['students']:
        assert set(config['settings'][student]['temperature'].values())=={.05}
    changed=deepcopy(config);changed['rags_main_method']='rags-block-zeroClipSum'
    with pytest.raises(ValueError,match='main method'):validate_v3_config(changed)
    changed=deepcopy(config);changed['arms'].append('less-linearSum')
    with pytest.raises(ValueError,match='thirteen'):validate_v3_config(changed)


@pytest.mark.parametrize('mode',AGGREGATIONS)
def test_neutral_block_name_matches_explicit_historical_block_alias(mode,tmp_path):
    current=select_full_pool(f'rags-block-{mode}',Probe(),run_dir=tmp_path,
        temperature=.05,**inputs())
    previous=select_full_pool(f'rags-main-{mode}',Probe(),run_dir=tmp_path,
        temperature=.05,**inputs())
    np.testing.assert_allclose(current.selected.probabilities,previous.selected.probabilities)
    np.testing.assert_array_equal(current.selected.unit_offsets,previous.selected.unit_offsets)
    assert current.name==f'rags-block-{mode}'



@pytest.mark.parametrize('mode',AGGREGATIONS)
def test_weighted_first_exact_user_definition(mode):
    refs,src,w=directions();q=w[:,None]*(refs@src.T)
    expected_scores=(q.sum(axis=0) if mode=='linearSum' else
        np.maximum(q,0).sum(axis=0) if mode=='zeroClipSum' else
        q.max(axis=0) if mode=='maxSum' else np.exp(q).sum(axis=0))
    p,audit=reference_marginal(refs,src,w,aggregation=mode,device='cpu',
        reference_batch=1,source_batch=2,weight_placement='before_transform')
    np.testing.assert_allclose(audit['source_scores'],expected_scores,atol=1e-14)
    np.testing.assert_allclose(p,value_mass(expected_scores,temperature=.05,gating=False),atol=1e-14)
    assert audit['weight_placement']=='before_transform'


def test_weighted_max_and_exp_differ_from_previous_order():
    refs,src,w=directions()
    for mode in ('maxSum','expSum'):
        before,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu',weight_placement='before_transform')
        after,_=reference_marginal(refs,src,w,aggregation=mode,device='cpu',weight_placement='after_transform')
        assert not np.allclose(before,after)
    before,_=reference_marginal(refs,src,w,aggregation='zeroClipSum',device='cpu')
    after,_=reference_marginal(refs,src,w,aggregation='zeroClipSum',device='cpu',weight_placement='after_transform')
    np.testing.assert_allclose(before,after)


def test_many_small_weight_exponentials_use_stable_common_shift():
    refs=np.tile(np.array([[1.,0.]]),(10000,1));src=np.array([[1.,0.],[-1.,0.]])
    p,audit=reference_marginal(refs,src,np.full(10000,.0001),aggregation='expSum',device='cpu',weight_placement='before_transform')
    centered=10000*np.expm1(np.array([.0001,-.0001]))
    np.testing.assert_allclose(audit['source_scores_for_softmax'],centered,atol=1e-13)
    np.testing.assert_allclose(p,value_mass(centered,temperature=.05,gating=False),atol=1e-15)
    assert audit['source_score_common_offset']==10000


def test_current_config_weight_order_and_less_baseline_exception():
    from rags.experiments.reference_opinions import weight_placement_for
    c=json.loads(Path('configs/rags/matrix_reference_opinion_20260917.json').read_text())
    assert c['opinion_aggregation_protocol']=='reference-opinion-v4-mixed-weight-order'
    for name in c['arms']:
        if name.startswith('rags-'):
            expected='after_transform' if name.endswith('-expSum') else 'before_transform'
            assert c['method_protocols'][name]['weight_placement']==expected
            assert weight_placement_for(name,c)==expected
    assert weight_placement_for('less-main',c)=='after_transform'
    changed=deepcopy(c);changed['method_protocols']['rags-block-expSum']['weight_placement']='before_transform'
    with pytest.raises(ValueError,match='method definitions'):validate_v3_config(changed)


def test_weighted_traj_max_scales_less_raw_max_when_anchors_match(tmp_path):
    kwargs=inputs()
    less=select_full_pool('less-main',Probe(),run_dir=tmp_path,temperature=.05,**kwargs)
    traj=select_full_pool('rags-traj-maxSum',Probe(),run_dir=tmp_path,temperature=.05,**kwargs)
    n=len(kwargs['reference_fragments'])
    np.testing.assert_allclose(traj.selected.metadata['reference_audit']['source_scores'],
        np.asarray(less.selected.metadata['native_scores'])/n,atol=1e-7)
    assert traj.selected.metadata['opinion_weight_placement']=='before_transform'
    assert less.selected.metadata['opinion_weight_placement']=='after_transform'


def test_mixed_exp_temperature_and_duplicate_direction_invariance():
    refs,src,w=directions(); c=refs@src.T
    p,a=reference_marginal(refs,src,w,aggregation='expSum',temperature=.05,device='cpu')
    np.testing.assert_allclose(a['source_scores'],w@np.exp(c),atol=1e-14)
    np.testing.assert_allclose(p,value_mass(w@np.exp(c),temperature=.05,gating=False),atol=1e-14)
    assert a['exp_inner_temperature']==1.
    assert a['weight_placement']=='after_transform'
    repeated,_=reference_marginal(np.repeat(refs,4,axis=0),src,np.repeat(w/4,4),
        aggregation='expSum',temperature=.05,device='cpu')
    np.testing.assert_allclose(p,repeated,atol=1e-14)
    old,_=reference_marginal(refs,src,w,aggregation='expSum',temperature=.05,
        device='cpu',weight_placement='before_transform')
    assert not np.allclose(p,old)


def test_mixed_weight_dispatch_preserves_frozen_protocols():
    from rags.experiments.reference_opinions import weight_placement_for
    current=json.loads(Path('configs/rags/matrix_reference_opinion_20260917.json').read_text())
    previous=json.loads(Path('configs/rags/matrix_reference_opinion_v3_weighted_first_20260917.json').read_text())
    historical=json.loads(Path('configs/rags/main_v3_t005.json').read_text())
    for c in (current,previous,historical):validate_v3_config(c)
    assert weight_placement_for('rags-block-expSum',current)=='after_transform'
    assert weight_placement_for('rags-block-expSum',previous)=='before_transform'
    assert weight_placement_for('rags-block-maxSum',current)=='before_transform'
    assert weight_placement_for('rags-main',historical)=='after_transform'
    assert selection_options('less-main',historical)['aggregation']=='linearSum'


@pytest.mark.parametrize('family',['rags-block','rags-traj'])
def test_production_exp_uses_after_transform(family,tmp_path):
    out=select_full_pool(f'{family}-expSum',Probe(),run_dir=tmp_path,
        temperature=.05,**inputs())
    a=out.selected.metadata['reference_audit']
    assert out.selected.metadata['opinion_weight_placement']=='after_transform'
    assert a['exp_inner_temperature']==1.
    assert a['temperature']==.05
