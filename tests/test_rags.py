"""Tests for the RAGS package (core, quadrature, perturbation, selection, pipeline)."""

from __future__ import annotations

import numpy as np
import pytest

from rags import (
    RAGSConfig,
    TimeSeriesFragment,
    AdaptiveInterval,
    GradientTrajectory,
    AdaptiveQuadratureEngine,
    MidpointResidualComputer,
    MovingBlockBootstrap,
    ReferencePerturbationEngine,
    StableRankingSelector,
    RAGSPipeline,
    compute_cosine_similarity,
    compute_inner_product,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic gradient fields with controllable local smoothness
# ---------------------------------------------------------------------------

def _low_frequency_features(window: np.ndarray, num_harmonics: int = 4) -> np.ndarray:
    """Low-frequency content of the window; smooth under one-step shifts."""
    spectrum = np.fft.rfft(window.astype(np.float64))
    truncated = spectrum[1 : num_harmonics + 1]
    return np.concatenate([truncated.real, truncated.imag])


def smooth_gradient_field(dimension: int = 8, num_harmonics: int = 4, seed: int = 0):
    """A gradient computer whose output varies smoothly with window position:
    g(W) = A @ lowfreq(W). High-frequency measurement noise in the data is
    filtered out, so the induced gradient trajectory is locally smooth."""
    matrices: dict[int, np.ndarray] = {}

    def matrix_for(window_length: int) -> np.ndarray:
        if window_length not in matrices:
            local = np.random.default_rng(seed + window_length)
            width = 2 * min(num_harmonics, window_length // 2)
            matrices[window_length] = local.normal(size=(dimension, width)) / np.sqrt(width)
        return matrices[window_length]

    def computer(window: np.ndarray, context_length: int, prediction_length: int) -> np.ndarray:
        features = _low_frequency_features(window, num_harmonics)
        return (matrix_for(len(window)) @ features).astype(np.float32)

    return computer


def abrupt_gradient_field(dimension: int = 8, num_harmonics: int = 4, regime_threshold: float = 0.0, seed: int = 1):
    """A gradient field that is piecewise smooth but jumps when the window's
    low-frequency content crosses a threshold — a state-change surrogate."""
    matrices: dict[tuple[int, int], np.ndarray] = {}

    def matrix_for(window_length: int, regime: int) -> np.ndarray:
        key = (window_length, regime)
        if key not in matrices:
            local = np.random.default_rng(seed + 1000 * regime + window_length)
            width = 2 * min(num_harmonics, window_length // 2)
            matrices[key] = local.normal(size=(dimension, width)) / np.sqrt(width)
        return matrices[key]

    def computer(window: np.ndarray, context_length: int, prediction_length: int) -> np.ndarray:
        features = _low_frequency_features(window, num_harmonics)
        regime = 1 if features[0] > regime_threshold else 0
        return (matrix_for(len(window), regime) @ features).astype(np.float32)

    return computer


def jittery_gradient_field(dimension: int = 8, num_harmonics: int = 4, noise: float = 5.0, seed: int = 2):
    """A gradient field with per-window independent noise: no local smoothness."""

    def computer(window: np.ndarray, context_length: int, prediction_length: int) -> np.ndarray:
        features = _low_frequency_features(window, num_harmonics)
        local = np.random.default_rng(seed + len(window))
        width = len(features)
        matrix = local.normal(size=(dimension, width)) / np.sqrt(width)
        local_rng = np.random.default_rng(abs(hash(tuple(np.round(window[:8], 6)))) % (2**32))
        return (matrix @ features + noise * local_rng.normal(size=dimension)).astype(np.float32)

    return computer


def make_fragment(length: int, seed: int = 0, freq: float = 1.0 / 600, noise: float = 0.01) -> TimeSeriesFragment:
    """Slow trend + mild noise: long locally-smooth stretches by construction."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    values = np.sin(2 * np.pi * freq * t) + noise * rng.normal(size=length)
    return TimeSeriesFragment(data=values.astype(np.float32), fragment_id=seed)


def make_fragment_with_spike(length: int, spike_center: int, spike_width: int = 12,
                             spike_height: float = 6.0, seed: int = 0) -> TimeSeriesFragment:
    """Slow trend plus one localized transient — the abrupt-field test bed."""
    fragment = make_fragment(length=length, seed=seed)
    t = np.arange(length, dtype=np.float64)
    spike = spike_height * np.exp(-((t - spike_center) / spike_width) ** 2)
    values = fragment.data.astype(np.float64) + spike
    return TimeSeriesFragment(data=values.astype(np.float32), fragment_id=seed)


def small_config(**overrides) -> RAGSConfig:
    defaults = dict(
        context_length=32,
        prediction_length=8,
        min_fragment_length=40,
        direction_tolerance=0.05,
        magnitude_tolerance=0.05,
        min_interval_size=4,
        block_length=4,
        num_reference_versions=5,
        selection_budget=5,
        ranking_budget=5,
        stability_threshold=0.4,
        lower_quantile=0.25,
        projection_dimension=8,
        perturbation_seed=7,
    )
    defaults.update(overrides)
    return RAGSConfig(**defaults)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class TestRAGSConfig:
    def test_window_length(self) -> None:
        config = small_config()
        assert config.window_length == config.context_length + config.prediction_length

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            small_config(direction_tolerance=0.0)
        with pytest.raises(ValueError):
            small_config(lower_quantile=0.9)
        with pytest.raises(ValueError):
            small_config(min_interval_size=1)
        with pytest.raises(ValueError):
            small_config(min_fragment_length=10)


class TestFragment:
    def test_valid_starts_matches_protocol(self) -> None:
        fragment = make_fragment(length=100, seed=1)
        starts = fragment.valid_starts(context_length=32, prediction_length=8)
        assert len(starts) == 100 - 40 + 1
        assert starts[0] == 1 and starts[-1] == 61

    def test_fragment_shorter_than_window_has_no_starts(self) -> None:
        fragment = make_fragment(length=20, seed=2)
        assert len(fragment.valid_starts(32, 8)) == 0

    def test_extract_window_is_exclusive_end(self) -> None:
        fragment = make_fragment(length=100, seed=3)
        window = fragment.extract_window(1, 32, 8)
        assert window.shape == (40,)
        np.testing.assert_allclose(window, fragment.data[:40])

    def test_extract_window_rejects_out_of_range(self) -> None:
        fragment = make_fragment(length=100, seed=4)
        with pytest.raises(ValueError):
            fragment.extract_window(62, 32, 8)
        with pytest.raises(ValueError):
            fragment.extract_window(0, 32, 8)

    def test_rejects_non_finite_data(self) -> None:
        with pytest.raises(ValueError):
            TimeSeriesFragment(data=np.array([1.0, np.nan, 3.0]))


class TestGradientTrajectory:
    def test_weighted_average(self) -> None:
        positions = np.array([1, 2, 3])
        gradients = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = np.array([1.0, 1.0, 2.0])
        trajectory = GradientTrajectory(
            fragment_index=0, positions=positions, gradients=gradients, weights=weights
        )
        expected = (1 * gradients[0] + 1 * gradients[1] + 2 * gradients[2]) / 4.0
        np.testing.assert_allclose(trajectory.weighted_average(), expected)

    def test_interpolation_between_anchors(self) -> None:
        positions = np.array([1, 5])
        gradients = np.array([[0.0, 0.0], [4.0, 8.0]])
        trajectory = GradientTrajectory(
            fragment_index=0, positions=positions, gradients=gradients,
            weights=np.ones(2),
        )
        midpoint = trajectory.interpolate_at(3, 1, 5)
        np.testing.assert_allclose(midpoint, [2.0, 4.0])

    def test_gradient_at_returns_none_for_missing(self) -> None:
        trajectory = GradientTrajectory(
            fragment_index=0, positions=np.array([1, 3]),
            gradients=np.ones((2, 4)), weights=np.ones(2),
        )
        assert trajectory.gradient_at(2) is None
        assert trajectory.gradient_at(3) is not None

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            GradientTrajectory(
                fragment_index=0, positions=np.array([1, 2, 3]),
                gradients=np.ones((2, 4)), weights=np.ones(3),
            )


class TestSimilarity:
    def test_cosine_of_identical_vectors(self) -> None:
        vector = np.array([1.0, 2.0, 3.0])
        assert compute_cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_of_orthogonal_vectors(self) -> None:
        assert compute_cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_of_zero_vector_is_zero(self) -> None:
        assert compute_cosine_similarity(np.zeros(3), np.ones(3)) == 0.0

    def test_inner_product(self) -> None:
        assert compute_inner_product(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

class TestMidpointResidualComputer:
    def test_zero_residual_for_perfect_interpolation(self) -> None:
        computer = MidpointResidualComputer(small_config())
        mid = np.array([1.0, 1.0])
        interp = np.array([1.0, 1.0])
        residuals = computer.compute_residuals(mid, interp)
        assert residuals["direction"] == pytest.approx(0.0, abs=1e-6)
        assert residuals["magnitude"] == pytest.approx(0.0, abs=1e-6)
        assert not computer.should_subdivide(mid, interp)

    def test_positive_residual_for_mismatch(self) -> None:
        computer = MidpointResidualComputer(small_config())
        mid = np.array([1.0, 0.0])
        interp = np.array([0.0, 1.0])
        residuals = computer.compute_residuals(mid, interp)
        assert residuals["direction"] == pytest.approx(1.0, abs=1e-6)
        assert residuals["magnitude"] > 1.0
        assert computer.should_subdivide(mid, interp)


class TestAdaptiveQuadrature:
    def test_smooth_trajectory_uses_few_anchors(self) -> None:
        config = small_config()
        fragment = make_fragment(length=200, seed=5)
        engine = AdaptiveQuadratureEngine(config, smooth_gradient_field())
        result = engine.process_fragment(fragment, 0)
        total = len(fragment.valid_starts(config.context_length, config.prediction_length))
        assert result.total_positions == total
        assert result.computed_gradients < total
        assert result.compression_ratio > 1.0

    def test_anchors_concentrate_near_change_points(self) -> None:
        config = small_config()
        fragment = make_fragment_with_spike(length=400, spike_center=200, seed=6)
        engine = AdaptiveQuadratureEngine(config, abrupt_gradient_field())
        result = engine.process_fragment(fragment, 0)
        # The abrupt field must spend more anchors than the smooth field
        smooth_engine = AdaptiveQuadratureEngine(config, smooth_gradient_field(seed=6))
        smooth_result = smooth_engine.process_fragment(fragment, 0)
        assert result.computed_gradients > smooth_result.computed_gradients
        # And the anchors should concentrate around the spike, not uniformly
        anchors = sorted({
            position
            for interval in result.intervals
            for position in interval.computed_anchors
        })
        near = [p for p in anchors if abs(p - 200) <= 60]
        far = [p for p in anchors if abs(p - 200) > 60]
        near_span = 121  # positions 140..260
        far_span = result.total_positions - near_span
        density_near = len(near) / near_span
        density_far = len(far) / far_span
        assert density_near > 1.5 * density_far

    def test_full_computation_when_everything_varies(self) -> None:
        config = small_config(direction_tolerance=1e-9, magnitude_tolerance=1e-9, min_interval_size=2)
        fragment = make_fragment(length=100, seed=7)
        engine = AdaptiveQuadratureEngine(config, jittery_gradient_field())
        result = engine.process_fragment(fragment, 0)
        assert result.compression_ratio <= 1.5  # near-full scan

    def test_anchor_cache_skips_gradient_computer(self) -> None:
        config = small_config()
        computer = smooth_gradient_field(seed=18)
        fragment = make_fragment(length=150, seed=19)

        calls = {"count": 0}
        def counting_computer(window, context_length, prediction_length):
            calls["count"] += 1
            return computer(window, context_length, prediction_length)

        plain_engine = AdaptiveQuadratureEngine(config, counting_computer)
        plain_result = plain_engine.process_fragment(fragment, 0)
        assert calls["count"] == plain_result.computed_gradients

        # Precompute every start's gradient and feed as cache: zero new calls,
        # identical anchors, identical estimated gradient.
        starts = fragment.valid_starts(config.context_length, config.prediction_length)
        cache = {}
        for start in starts:
            window = fragment.extract_window(int(start), config.context_length, config.prediction_length)
            cache[(fragment.fragment_id, int(start))] = computer(window, config.context_length, config.prediction_length)

        calls["count"] = 0
        cached_engine = AdaptiveQuadratureEngine(config, counting_computer, anchor_cache=cache)
        cached_result = cached_engine.process_fragment(fragment, 0)
        assert calls["count"] == 0
        assert cached_result.computed_gradients == plain_result.computed_gradients

        plain_G, _ = plain_engine.compute_full_reference_gradient([plain_result], [fragment])
        cached_G, _ = cached_engine.compute_full_reference_gradient([cached_result], [fragment])
        np.testing.assert_allclose(plain_G, cached_G, rtol=1e-5, atol=1e-6)

    def test_fragment_without_valid_starts(self) -> None:
        config = small_config()
        fragment = make_fragment(length=config.min_fragment_length - 10, seed=8)
        engine = AdaptiveQuadratureEngine(config, smooth_gradient_field())
        result = engine.process_fragment(fragment, 0)
        assert result.intervals == []
        assert result.total_positions == 0

    def test_intervals_partition_the_start_range(self) -> None:
        config = small_config()
        fragment = make_fragment(length=200, seed=9)
        engine = AdaptiveQuadratureEngine(config, smooth_gradient_field())
        result = engine.process_fragment(fragment, 0)
        starts = fragment.valid_starts(config.context_length, config.prediction_length)
        covered = np.concatenate([interval.positions for interval in result.intervals])
        # Subdivision shares boundary anchors, so positions may repeat once;
        # every valid start must be covered and no foreign position appears.
        assert set(covered.tolist()) == set(starts.tolist())
        for interval in result.intervals:
            assert len(interval.computed_anchors) >= 2

    def test_full_reference_gradient_approximates_dense_scan(self) -> None:
        """The core Proposition-1 experiment: quadrature estimate vs full scan."""
        config = small_config()
        computer = smooth_gradient_field(dimension=8)
        fragments = [make_fragment(length=200, seed=10), make_fragment(length=150, seed=11)]

        # Full scan (ground truth)
        full_G = np.zeros(config.projection_dimension, dtype=np.float64)
        total = 0
        for fragment in fragments:
            for position in fragment.valid_starts(config.context_length, config.prediction_length):
                window = fragment.extract_window(int(position), config.context_length, config.prediction_length)
                full_G += computer(window, config.context_length, config.prediction_length)
                total += 1
        full_G /= total

        engine = AdaptiveQuadratureEngine(config, computer)
        results = [engine.process_fragment(f, i) for i, f in enumerate(fragments)]
        estimated_G, _ = engine.compute_full_reference_gradient(results, fragments)

        error = np.linalg.norm(full_G - estimated_G) / (np.linalg.norm(full_G) + 1e-12)
        computed = sum(r.computed_gradients for r in results)
        assert computed < total
        assert error < 0.2, f"relative error {error:.4f} too large with {computed}/{total} anchors"

    def test_score_error_bound_and_topk_preservation(self) -> None:
        """Proposition 1: |s_i - ŝ_i| ≤ ε_Q and top-k preservation under rank gap."""
        config = small_config()
        computer = smooth_gradient_field(dimension=8)
        fragments = [make_fragment(length=200, seed=12)]

        full_G = np.zeros(config.projection_dimension)
        gradients_by_position = {}
        for position in fragments[0].valid_starts(config.context_length, config.prediction_length):
            window = fragments[0].extract_window(int(position), config.context_length, config.prediction_length)
            g = computer(window, config.context_length, config.prediction_length)
            gradients_by_position[int(position)] = g
            full_G += g
        full_G /= len(gradients_by_position)

        engine = AdaptiveQuadratureEngine(config, computer)
        result = engine.process_fragment(fragments[0], 0)
        estimated_G, _ = engine.compute_full_reference_gradient([result], fragments)

        positions = sorted(gradients_by_position)
        candidate_matrix = np.stack([gradients_by_position[p] for p in positions]).astype(np.float64)
        normalized = candidate_matrix / (np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-8)
        full_G = full_G.astype(np.float64)
        estimated_G = estimated_G.astype(np.float64)

        true_scores = normalized @ full_G
        approx_scores = normalized @ estimated_G
        epsilon_Q = np.linalg.norm(full_G - estimated_G)

        score_errors = np.abs(true_scores - approx_scores)
        assert score_errors.max() <= epsilon_Q + 1e-6

        k = 10
        true_top = set(np.argsort(-true_scores)[:k].tolist())
        gap = np.sort(true_scores)[::-1][k - 1] - np.sort(true_scores)[::-1][k]
        if gap > 2 * epsilon_Q:
            approx_top = set(np.argsort(-approx_scores)[:k].tolist())
            assert true_top == approx_top


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

class TestMovingBlockBootstrap:
    def test_resample_preserves_block_structure(self) -> None:
        config = small_config(block_length=4)
        bootstrap = MovingBlockBootstrap(config)
        positions = np.arange(1, 21, dtype=np.int32)
        resampled, weights = bootstrap.resample_positions(positions)
        assert len(resampled) >= len(positions)
        assert (weights > 0).all()
        # Each element of the resample must be a real position
        assert np.isin(resampled, positions).all()

    def test_resample_contains_continuous_runs(self) -> None:
        config = small_config(block_length=4)
        bootstrap = MovingBlockBootstrap(config)
        positions = np.arange(1, 33, dtype=np.int32)
        resampled, _ = bootstrap.resample_positions(positions)
        # Within each block of size 4, consecutive elements differ by 1
        for block_start in range(0, len(resampled) - 3, 4):
            block = resampled[block_start:block_start + 4]
            assert (np.diff(block) == 1).all()

    def test_block_delete_keeps_subset(self) -> None:
        config = small_config(block_length=4)
        bootstrap = MovingBlockBootstrap(config)
        positions = np.arange(1, 41, dtype=np.int32)
        kept, _ = bootstrap.block_delete_positions(positions, delete_fraction=0.5)
        assert len(kept) < len(positions)
        assert len(kept) > 0
        assert np.isin(kept, positions).all()

    def test_empty_input(self) -> None:
        bootstrap = MovingBlockBootstrap(small_config())
        resampled, weights = bootstrap.resample_positions(np.array([], dtype=np.int32))
        assert len(resampled) == 0 and len(weights) == 0


class TestReferencePerturbationEngine:
    def _setup(self, config: RAGSConfig):
        computer = smooth_gradient_field(seed=13)
        fragments = [make_fragment(length=150, seed=14), make_fragment(length=120, seed=15)]
        engine = AdaptiveQuadratureEngine(config, computer)
        results = [engine.process_fragment(f, i) for i, f in enumerate(fragments)]
        valid_starts = [
            f.valid_starts(config.context_length, config.prediction_length) for f in fragments
        ]
        return fragments, results, valid_starts

    def test_versions_are_distinct_but_finite(self) -> None:
        config = small_config(num_reference_versions=6)
        fragments, results, valid_starts = self._setup(config)
        engine = ReferencePerturbationEngine(config)
        versions = engine.generate_reference_versions(results, fragments, valid_starts)
        assert len(versions) == 6
        gradients = np.stack([v.reference_gradient for v in versions])
        assert np.isfinite(gradients).all()
        assert len({tuple(np.round(g, 6)) for g in gradients}) > 1

    def test_version_weights_are_nonnegative(self) -> None:
        config = small_config(num_reference_versions=4)
        fragments, results, valid_starts = self._setup(config)
        engine = ReferencePerturbationEngine(config)
        versions = engine.generate_reference_versions(results, fragments, valid_starts)
        for version in versions:
            assert version.total_weight > 0
            for fragment_weights in version.interval_weights.values():
                for weight in fragment_weights.values():
                    assert weight >= 0.0

    def test_versions_reuse_anchors_without_new_computation(self) -> None:
        """Perturbation must not trigger any new gradient computation."""
        config = small_config(num_reference_versions=5)
        computer = smooth_gradient_field(seed=16)
        fragments = [make_fragment(length=150, seed=17)]
        quadrature = AdaptiveQuadratureEngine(config, computer)
        results = [quadrature.process_fragment(fragments[0], 0)]
        anchors_before = dict(results[0].intervals[0].computed_anchors)

        calls = {"count": 0}
        def counting_computer(window, context_length, prediction_length):
            calls["count"] += 1
            return computer(window, context_length, prediction_length)

        engine = ReferencePerturbationEngine(config)
        valid_starts = [fragments[0].valid_starts(config.context_length, config.prediction_length)]
        engine.generate_reference_versions(results, fragments, valid_starts)
        assert calls["count"] == 0
        assert results[0].intervals[0].computed_anchors.keys() == anchors_before.keys()

    def test_boundary_perturbations_produce_versions(self) -> None:
        config = small_config()
        fragments, results, valid_starts = self._setup(config)
        engine = ReferencePerturbationEngine(config)
        versions = engine.generate_boundary_perturbations(results, fragments, valid_starts)
        assert len(versions) == 4  # shifts of -2, -1, +1, +2
        for version in versions:
            assert np.isfinite(version.reference_gradient).all()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestStableRankingSelector:
    def test_stable_candidates_are_preferred(self) -> None:
        config = small_config(num_reference_versions=10, ranking_budget=3, stability_threshold=0.5)
        selector = StableRankingSelector(config)

        dimension = 4
        rng = np.random.default_rng(0)
        # Candidate 0 points consistently along +x (stable, high utility)
        # Candidate 1 points along a rotating direction (unstable)
        gradients = np.zeros((2, dimension), dtype=np.float32)
        gradients[0] = np.array([1.0, 0.0, 0.0, 0.0])
        gradients[1] = np.array([0.0, 1.0, 0.0, 0.0])

        versions = []
        for r in range(10):
            direction = np.array([1.0, 0.0, 0.0, 0.0])
            if r % 2 == 0:
                direction = direction + rng.normal(scale=0.01, size=dimension)
            else:
                direction = np.array([0.0, 1.0, 0.0, 0.0]) + rng.normal(scale=0.01, size=dimension)
            from rags.perturbation import ReferenceVersion
            versions.append(ReferenceVersion(
                version_id=r, reference_gradient=direction.astype(np.float32),
                interval_weights={}, total_weight=1.0,
            ))

        scores = selector.score_candidates(gradients, [(0, 1), (1, 1)], versions)
        assert len(scores) == 2
        # Candidate 0 is in the top-3 only in half the versions (odd versions
        # point along y), so frequency should be measurable and finite
        for score in scores:
            assert 0.0 <= score.stability_frequency <= 1.0
            assert -1.001 <= score.lower_quantile_score <= 1.001

    def test_selection_respects_stability_constraint(self) -> None:
        config = small_config(
            num_reference_versions=8, ranking_budget=2, selection_budget=2,
            stability_threshold=0.75, lower_quantile=0.25,
        )
        selector = StableRankingSelector(config)

        from rags.selection import CandidateScore
        candidates = [
            CandidateScore(0, 0, 1, np.full(8, 0.9, dtype=np.float32), 1.0, 0.9, 0.9),   # stable + high
            CandidateScore(1, 0, 2, np.full(8, 0.5, dtype=np.float32), 1.0, 0.5, 0.5),   # stable + low
            CandidateScore(2, 1, 1, np.linspace(0.2, 1.0, 8, dtype=np.float32), 0.25, 0.35, 0.6),  # unstable
        ]
        selection = selector.select(candidates)
        selected_ids = {c.candidate_id for c in selection.selected}
        assert selected_ids == {0, 1}
        assert selection.coverage == 1.0
        assert selection.num_backfilled == 0

    def test_backfill_when_stable_candidates_insufficient(self) -> None:
        config = small_config(
            num_reference_versions=8, ranking_budget=2, selection_budget=3,
            stability_threshold=0.75,
        )
        selector = StableRankingSelector(config)

        from rags.selection import CandidateScore
        candidates = [
            CandidateScore(0, 0, 1, np.full(8, 0.9, dtype=np.float32), 1.0, 0.9, 0.9),
            CandidateScore(1, 1, 1, np.linspace(0.2, 1.0, 8, dtype=np.float32), 0.25, 0.35, 0.6),
            CandidateScore(2, 1, 2, np.full(8, 0.4, dtype=np.float32), 0.0, 0.4, 0.4),
        ]
        selection = selector.select(candidates)
        assert len(selection.selected) == 3
        assert selection.num_backfilled == 2
        assert 0.0 < selection.coverage < 1.0

    def test_version_norms_flag_degenerate_versions(self) -> None:
        config = small_config()
        selector = StableRankingSelector(config)
        from rags.perturbation import ReferenceVersion
        versions = [
            ReferenceVersion(0, np.ones(4, dtype=np.float32), {}, 1.0),
            ReferenceVersion(1, np.zeros(4, dtype=np.float32), {}, 1.0),
        ]
        norms = selector.version_norms(versions)
        assert norms[0] > 1.0
        assert norms[1] == 0.0

    def test_score_candidates_requires_matching_inputs(self) -> None:
        selector = StableRankingSelector(small_config())
        with pytest.raises(ValueError):
            selector.score_candidates(np.ones((3, 4)), [(0, 1)], [])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestRAGSPipeline:
    def test_end_to_end_run(self) -> None:
        config = small_config(
            context_length=32, prediction_length=8, projection_dimension=8,
            num_reference_versions=5, selection_budget=4, ranking_budget=4,
        )
        computer = smooth_gradient_field(seed=20)
        reference = [make_fragment(length=150, seed=21)]
        candidates = [make_fragment(length=120, seed=22), make_fragment(length=100, seed=23)]

        pipeline = RAGSPipeline(config, computer)
        report = pipeline.run(reference, candidates)

        num_candidates = sum(
            len(f.valid_starts(config.context_length, config.prediction_length)) for f in candidates
        )
        assert len(report.candidate_scores) == num_candidates
        assert len(report.selection.selected) == min(config.selection_budget, num_candidates)
        assert report.reference_backward_count < report.total_start_positions
        assert np.isfinite(report.estimated_reference_gradient).all()
        assert len(report.versions) == config.num_reference_versions

    def test_candidates_keep_fragment_and_position_identity(self) -> None:
        config = small_config()
        computer = smooth_gradient_field(seed=24)
        reference = [make_fragment(length=100, seed=25)]
        candidates = [make_fragment(length=90, seed=26)]

        report = RAGSPipeline(config, computer).run(reference, candidates)
        identities = {(c.fragment_id, c.position) for c in report.candidate_scores}
        expected = {
            (0, int(p))
            for p in candidates[0].valid_starts(config.context_length, config.prediction_length)
        }
        assert identities == expected

    def test_reproducible_with_same_seed(self) -> None:
        config = small_config()
        computer = smooth_gradient_field(seed=27)
        reference = [make_fragment(length=100, seed=28)]
        candidates = [make_fragment(length=90, seed=29)]

        first = RAGSPipeline(config, computer).run(reference, candidates)
        second = RAGSPipeline(config, computer).run(reference, candidates)
        np.testing.assert_array_equal(
            first.estimated_reference_gradient, second.estimated_reference_gradient
        )
        np.testing.assert_array_equal(
            np.stack([v.reference_gradient for v in first.versions]),
            np.stack([v.reference_gradient for v in second.versions]),
        )
        assert [c.candidate_id for c in first.selection.selected] == [
            c.candidate_id for c in second.selection.selected
        ]

    def test_separate_candidate_computer(self) -> None:
        config = small_config()
        reference = [make_fragment(length=100, seed=30)]
        candidates = [make_fragment(length=90, seed=31)]

        calls = {"count": 0}
        def candidate_computer(window, context_length, prediction_length):
            calls["count"] += 1
            return smooth_gradient_field(seed=32)(window, context_length, prediction_length)

        report = RAGSPipeline(
            config, smooth_gradient_field(seed=33)
        ).run(reference, candidates, candidate_gradient_computer=candidate_computer)
        num_candidates = len(
            candidates[0].valid_starts(config.context_length, config.prediction_length)
        )
        assert calls["count"] == num_candidates
        assert len(report.candidate_scores) == num_candidates
