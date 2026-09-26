import numpy as np
import pytest
from rags.experiments.reference_opinions import reference_scores,AGGREGATIONS

@pytest.mark.parametrize('mode',AGGREGATIONS)
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_frozen_weighted_scores_match_dense_oracle(mode,device):
    import torch
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    rng=np.random.default_rng(199)
    ref=rng.normal(size=(37,64));ref/=np.linalg.norm(ref,axis=1)[:,None]
    src=rng.normal(size=(103,64));src/=np.linalg.norm(src,axis=1)[:,None]
    w=rng.uniform(.01,1,37);w/=w.sum();c=ref@src.T
    expected={'linearSum':lambda:(w[:,None]*c).sum(0),
        'zeroClipSum':lambda:(w[:,None]*np.maximum(c,0)).sum(0),
        'maxSum':lambda:(w[:,None]*c).max(0),
        'expSum':lambda:(w[:,None]*np.exp(c)).sum(0)}[mode]()
    actual,audit=reference_scores(ref,src,w,aggregation=mode,reference_batch=7,source_batch=19,device=device)
    np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=1e-13)
    assert audit['additional_bp_allowed'] is False
    assert audit['reference_weights_stage']=='score_aggregation'
    assert audit['score_workspace_shape']==[7,19]

@pytest.mark.parametrize('mode',AGGREGATIONS)
def test_zero_opinions_and_negative_max_are_not_gated(mode):
    scores,_=reference_scores([[1.,0.],[0.,1.]],[[0.,0.],[-1.,-1.]], [.25,.75],aggregation=mode,device='cpu')
    assert scores[0]==(1. if mode=='expSum' else 0.)
    if mode=='maxSum':assert scores[1]==-.25
