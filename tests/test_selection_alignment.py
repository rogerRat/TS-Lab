import numpy as np
from rage.experiments.selection_alignment import alignment_metrics


def test_alignment_detects_gain_and_reports_zero_vectors():
    reference=np.array([[1.,0.],[1.,0.]])
    before=alignment_metrics(np.array([[1.,0.],[-1.,0.],[0.,0.]]),reference)
    after=alignment_metrics(np.array([[1.,0.],[1.,0.]]),reference)
    assert before['mean_pair_cosine']==0
    assert before['source_zero_gradient_fraction']==1/3
    assert before['cosine_between_mean_raw_gradients'] is None
    assert after['mean_pair_cosine']==1
    assert after['mean_best_reference_cosine']==1
    assert after['cosine_between_mean_raw_gradients']==1


def test_cosine_is_scale_invariant_for_tiny_nonzero_gradients():
    source = np.array([[1., 2.], [-2., 1.]])
    reference = np.array([[2., 1.], [1., -3.]])
    normal = alignment_metrics(source, reference)
    tiny = alignment_metrics(source * 1e-30, reference * 1e-25)
    for key in ['mean_pair_cosine', 'mean_best_reference_cosine', 'cosine_between_mean_raw_gradients']:
        np.testing.assert_allclose(tiny[key], normal[key], atol=1e-14)


def test_direction_mixture_detects_shift_despite_perfect_local_matching():
    from rage.experiments.selection_alignment import direction_groups, mixture_metrics
    ref = np.repeat(np.eye(3), [50,25,25], axis=0)
    selected = np.repeat(np.eye(3), [10,80,10], axis=0)
    centers = direction_groups(ref, 3)
    before = mixture_metrics(ref, ref, centers)
    after = mixture_metrics(selected, ref, centers)
    assert before['total_variation'] == 0
    np.testing.assert_allclose(after['total_variation'], .55)
    assert alignment_metrics(selected, ref)['mean_best_reference_cosine'] == 1
    zero = mixture_metrics(np.zeros((2,3)), ref, centers)
    assert zero['source_shares'][-1] == 1


def test_rage_max_score_recall_does_not_preserve_query_frequency():
    from rage.experiments.arms import _streamed_top_b
    reference = np.repeat(np.eye(4, dtype=np.float32)[:3], [50,25,25], axis=0)
    candidates = np.array([[.8,0,0,.6], [0,1,0,0], [0,0,.9,np.sqrt(.19)]], dtype=np.float32)
    scores, indices = _streamed_top_b(reference, lambda lo, hi: candidates[lo:hi], 3, 1, device='cpu')
    unique, inverse = np.unique(indices.ravel(), return_inverse=True)
    best = np.full(len(unique), -np.inf)
    np.maximum.at(best, inverse, scores.ravel())
    ranked = unique[np.argsort(-best)]
    assert ranked[0] == 1  # B wins the first-unit budget despite only 25% target mass.
