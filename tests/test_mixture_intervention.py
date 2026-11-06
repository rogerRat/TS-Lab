import numpy as np
from rage.experiments.mixture_intervention import matching_probabilities


def test_matches_target_without_adding_samples():
    labels = np.repeat([0,1,2], [10,80,10])
    weights, report = matching_probabilities(labels, [.5,.25,.25])
    np.testing.assert_allclose(np.bincount(labels, weights=weights), [.5,.25,.25])
    assert report['missing_target_mass'] == 0


def test_missing_group_is_disclosed_not_fabricated():
    weights, report = matching_probabilities(np.array([0,0,1]), [.5,.25,.25])
    assert report['missing_target_mass'] == .25
    np.testing.assert_allclose(report['achieved_group_shares'], [2/3,1/3,0])


def test_weighted_training_uses_requested_replacement_draws():
    import torch
    from types import SimpleNamespace
    from rage.experiments.train_eval import fine_tune
    model = torch.nn.Linear(2,1)
    windows = np.arange(6, dtype=np.float32).reshape(3,2)
    seen = []
    def loss(batch):
        seen.append(batch.copy())
        return model(torch.from_numpy(batch)).square().mean()
    engine = SimpleNamespace(model=model, device='cpu', training_loss=loss)
    p = np.array([.1,.2,.7])
    fine_tune(engine, windows, steps=2, batch_size=16, learning_rate=1e-6, seed=3,
              sampling_probabilities=p)
    rng = np.random.default_rng(3)
    for batch in seen:
        np.testing.assert_array_equal(batch, windows[rng.choice(3, size=16, replace=True, p=p)])
