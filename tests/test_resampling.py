from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rage.core import TimeSeriesFragment
from rage.experiments.arms import ArmOutput, SelectionCost, WindowIndex
from rage.experiments.regions import save_selection, load_selection
from rage.experiments.resampling import (
    ResampledWindows, InfeasibleReference, rage_marginal, ranked_mass,
)
from rage.experiments.schedule import exposure_checkpoints
from rage.experiments.train_eval import fine_tune


def pool_index():
    return WindowIndex([TimeSeriesFragment(np.arange(n, dtype=np.float32) + 100 * i,
                        entity_id=str(i), fragment_id=i) for i, n in enumerate([10, 12])],
                       context_length=4, prediction_length=2)


def distribution(index=None):
    index = index or pool_index()
    return ResampledWindows.from_index(index, [np.arange(5), np.arange(5, 12)], [.2, .8],
        pool_manifest={"total_equivalents": 3}, metadata={"model": "test-only", "unit": "trajectory"})


def test_ref_mass_shared_sources_and_hierarchical_equivalence():
    refs = np.eye(3)
    src = np.array([[1., 1., 0.], [0., 1., 1.], [1., -1., 1.], [-1., -1., -1.]])
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    n, m = np.array([2, 1, 1]), np.array([2, 4, 3, 1])
    beta, audit = rage_marginal(refs, src, n, m)
    pi = n / n.sum()
    scores = refs @ src.T
    weights = np.where(scores > 0, np.exp(scores / .1) * m, 0.)
    conditional = weights / weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(beta, pi @ conditional, rtol=1e-14)
    np.testing.assert_allclose([r["joint_mass"] for r in audit["rows"]], [.5, .25, .25])
    # Same window probabilities, not identical RNG sequences.
    np.testing.assert_allclose(np.repeat(beta / m, m),
                               np.repeat((pi[:, None] * conditional / m).sum(axis=0), m))
    assert beta[-1] == 0 and np.all(beta[:-1] > 0)


def test_tiny_positive_zero_negative_and_infeasible_reference():
    beta, audit = rage_marginal([[1., 0.]], [[1e-300, 1.], [0., 1.], [-1., 0.]], [7], [1, 2, 3])
    np.testing.assert_array_equal(beta, [1., 0., 0.])
    assert audit["rows"][0]["nonpositive_pairs"] == 2
    with pytest.raises(InfeasibleReference) as error:
        rage_marginal(np.eye(2), [[1., 0.]], [3, 1], [1], reference_ids=["a", "missing"])
    assert error.value.reference_id == "missing" and error.value.share == .25
    with pytest.raises(ValueError, match="finite"):
        rage_marginal([[1.]], [[np.nan]], [1], [1])


def test_temperature_cannot_silently_drop_positive_pairs():
    with pytest.raises(ValueError, match="underflows"):
        rage_marginal([[1.]], [[.1], [.9]], [1], [1, 1], temperature=1e-6)


def test_ranks_keep_ties_native_signs_and_length_factor():
    q = ranked_mass([1., 1., -1., 0.], [2, 4, 3, 8], gradient_alignment=True)
    np.testing.assert_allclose(q, [1/3, 2/3, 0, 0])
    assert np.all(ranked_mass([-3., -2., 0., 1.]) > 0)  # proxy losses are not cosines


def test_lazy_storage_roundtrip_and_cache_identity(tmp_path):
    selected = distribution()
    path = tmp_path / "probability.npz"
    save_selection(path, ArmOutput("test", selected, [], SelectionCost(), {}))
    loaded = load_selection(path).selected
    np.testing.assert_array_equal(loaded.window_probabilities(), selected.window_probabilities())
    np.testing.assert_array_equal(loaded[np.arange(12)], pool_index().windows_at(np.arange(12)))
    assert loaded.pool_n == 3 and len(loaded) == 12
    assert loaded.metadata["model"] == "test-only"
    assert loaded.coverage()["supported_legal_windows"] == 12
    np.testing.assert_array_equal(loaded.sample_indices(np.random.default_rng(7), 100),
                                  selected.sample_indices(np.random.default_rng(7), 100))


def test_sampling_mass_not_multiplied_by_length_twice():
    selected = distribution()
    draws = selected.sample_indices(np.random.default_rng(9), 100000)
    frequency = np.bincount(draws, minlength=len(selected)) / len(draws)
    np.testing.assert_allclose(frequency, selected.window_probabilities(), atol=.003)
    assert abs((draws < 5).mean() - .2) < .003


def test_manifest_and_unique_start_ownership():
    index = pool_index()
    with pytest.raises(ValueError, match="manifest"):
        ResampledWindows.from_index(index, [np.arange(12)], [1.],
            pool_manifest={"total_equivalents": 4}, metadata={})
    with pytest.raises(ValueError, match="unique"):
        ResampledWindows.from_index(index, [np.arange(7), np.arange(5, 12)], [.5, .5],
            pool_manifest={"total_equivalents": 3}, metadata={})


def test_training_draws_static_distribution_and_mean_loss():
    selected = distribution()
    model = torch.nn.Linear(6, 1)
    seen = []
    def loss(windows):
        seen.append(windows.copy())
        return model(torch.from_numpy(windows)).square().mean()
    engine = SimpleNamespace(model=model, device="cpu", training_loss=loss)
    fine_tune(engine, selected, steps=3, batch_size=32, learning_rate=1e-7, seed=7)
    rng = np.random.default_rng(7)
    for batch in seen:
        np.testing.assert_array_equal(batch, selected[selected.sample_indices(rng, 32)])
    with pytest.raises(ValueError, match="twice"):
        fine_tune(engine, selected, steps=1, batch_size=2, learning_rate=1e-7, seed=7,
                  sampling_probabilities=selected.window_probabilities())


def test_half_epoch_full_pool_accounting():
    points = exposure_checkpoints(16284, 32, [.5, 1., 1.5, 4.])
    assert points[0]["step"] == 255 and points[-1]["step"] == 2036
    assert points[-1]["draws"] == 65152
    assert points[-1]["actual_epoch"] == 65152 / 16284
    assert all(0 <= p["rounding_draws"] < 32 for p in points)
    assert len(exposure_checkpoints(3, 32, [.5, 1.])) == 2


def test_training_reports_actual_exposure_and_repeats():
    selected = distribution()
    model = torch.nn.Linear(6, 1)
    engine = SimpleNamespace(model=model, device="cpu",
        training_loss=lambda w: model(torch.from_numpy(w)).square().mean())
    result = fine_tune(engine, selected, steps=2, batch_size=32, learning_rate=1e-7, seed=7)
    assert result["sample_exposures"] == 64
    assert result["actual_epochs"] == 64 / 3
    assert result["unique_window_exposures"] + result["repeated_window_exposures"] == 64
    assert result["unique_window_exposures"] <= 12


def test_tiny_positive_mass_is_not_lost_in_sampler_cdf():
    selected = distribution()
    selected.probabilities = np.array([1., 1e-20])
    selected._prepare_sampling()
    represented = selected.sampler_probabilities()
    assert represented[1] > 0
    np.testing.assert_allclose(represented, selected.probabilities, rtol=1e-14, atol=0)
    class TinyEvent:
        def random(self, size):
            return np.zeros(size)
        def integers(self, low, high, size=None):
            return np.ones(size, dtype=np.int64) if size is not None else np.zeros_like(high)
    # A rare alias-column event remains a real route to the second block.
    np.testing.assert_array_equal(selected.sample_indices(TinyEvent(), 3), [5, 5, 5])


def test_alias_retains_zeros_and_mixed_tiny_masses():
    from rage.experiments.resampling import alias_table
    probabilities = np.array([.7, 1e-30, 0., .3, 1e-20, 0.])
    acceptance, aliases = alias_table(probabilities)
    represented = acceptance.copy()
    np.add.at(represented, aliases, 1 - acceptance)
    represented /= len(probabilities)
    np.testing.assert_allclose(represented, probabilities, rtol=1e-14, atol=0)
    assert np.all(acceptance[probabilities > 0] > 0)


def test_cdf_roundoff_does_not_give_zero_mass_tail_a_ticket():
    selected = distribution()
    selected.probabilities = np.r_[np.full(10, .1), 0.]
    selected.unit_offsets = np.arange(12)
    selected._prepare_sampling()
    assert selected.sampler_probabilities()[-1] == 0
