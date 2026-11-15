import numpy as np
import pytest

from rage.ads import (
    ADSConfig,
    ADSPipeline,
    aggregate_source_scores,
    cosine_similarity_matrix,
    global_top_k,
    normalize_sampling_probabilities,
    SourceProbabilitySampler,
    sample_source_probability_descriptor,
    source_probability_descriptor,
    spread_anchor_scores,
    uniform_anchor_positions,
)
from rage.core import TimeSeriesFragment
from rage.experiments.arms import WindowIndex, ads_select
from rage.experiments.main_table import parse_args as parse_main_args
from rage.experiments.run import _step_argv, parse_args as parse_run_args
from rage.ads import _TrajectoryState, _fine_anchor_plan, _regions_for_hits


def test_uniform_coarse_anchors_include_endpoints_and_scale_density():
    sparse = uniform_anchor_positions(10, 4)
    dense = uniform_anchor_positions(10, 2)
    assert sparse[0] == 0 and sparse[-1] == 9
    assert dense[0] == 0 and dense[-1] == 9
    assert len(dense) >= len(sparse)
    assert uniform_anchor_positions(1, 2) == [0]
    assert uniform_anchor_positions(2, 2) == [0, 1]


def test_uniform_anchor_positions_match_nearest_scan_rule():
    for length in range(1, 65):
        for spacing in (0.25, 0.9, 1.0, 1.25, 2.0, 4.0, 8.0, 32.0, 100.0):
            if length == 1:
                expected = [0]
            elif length == 2:
                expected = [0, 1]
            else:
                interior_count = length - 2
                wanted = min(interior_count, int(np.ceil(interior_count / spacing)))
                candidates = set(range(1, length - 1))
                if wanted == 1:
                    targets = [(length - 1) / 2.0]
                else:
                    targets = [
                        1.0 + index * (interior_count - 1) / (wanted - 1)
                        for index in range(wanted)
                    ]
                chosen = []
                for target in targets:
                    position = min(candidates,
                                   key=lambda candidate: (abs(candidate - target), candidate))
                    chosen.append(position)
                    candidates.remove(position)
                expected = sorted({0, length - 1, *chosen})
            assert uniform_anchor_positions(length, spacing) == expected



def test_single_total_q_derives_both_side_spacings_and_default_ratio():
    config = ADSConfig(q=16, eta=1.0)
    assert config.source_spacing == pytest.approx(32.0)
    assert config.reference_spacing == pytest.approx(32.0)
    assert config.top_k is None
    assert config.top_ratio == pytest.approx(0.2)
    unequal = ADSConfig(q=16, eta=2.0, top_k=3)
    assert unequal.source_spacing == pytest.approx(48.0)
    assert unequal.reference_spacing == pytest.approx(24.0)


def test_eta_is_required_by_ads_config():
    with pytest.raises(TypeError):
        ADSConfig()

def test_zero_norm_cosines_and_deterministic_global_ties():
    matrix, zero_pairs = cosine_similarity_matrix(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert np.array_equal(matrix, np.asarray([[0.0, 0.0], [0.0, 1.0]]))
    assert zero_pairs == 3
    tied = global_top_k(
        np.ones((2, 2), dtype=np.float32), 2,
        reference_keys=[("r2", 1), ("r1", 1)],
        source_keys=[("s2", 1), ("s1", 1)],
    )
    assert [(row, column) for row, column, _ in tied] == [(1, 1), (1, 0)]


def test_candidate_regions_expand_from_nearest_unmatched_anchors():
    fragment = TimeSeriesFragment(np.arange(14), fragment_id=4)
    starts = fragment.valid_starts(1, 1, 1)
    state = _TrajectoryState(
        fragment=fragment,
        valid_starts=starts,
        anchors={position: np.ones(2) for position in (0, 4, 8, 12)},
    )
    assert _regions_for_hits(state, {4, 8}) == [(2, 10)]


def test_fine_plan_forces_endpoints_and_splits_longest_gaps():
    fragment = TimeSeriesFragment(np.arange(14), fragment_id=4)
    state = _TrajectoryState(
        fragment=fragment,
        valid_starts=fragment.valid_starts(1, 1, 1),
        anchors={position: np.ones(2) for position in (0, 4, 8, 12)},
    )
    plan = _fine_anchor_plan([state], {id(state): {4, 8}}, target_spacing=2.0)
    additions = plan[id(state)]
    assert len(additions) == 5
    assert {2, 10, 6}.issubset(additions)
    assert 3 in additions


def test_column_aggregation_and_finite_max_decay():
    similarities = np.asarray([[0.1, 0.7], [0.4, -0.2]], dtype=np.float32)
    assert np.allclose(aggregate_source_scores(similarities, "linear"), [0.5, 0.5])
    assert np.allclose(aggregate_source_scores(similarities, "max"), [0.4, 0.7])
    assert np.allclose(aggregate_source_scores(similarities, "relu"), [0.5, 0.7])
    assert np.allclose(aggregate_source_scores(similarities, "exp"),
                       np.exp(similarities).sum(axis=0))
    weights = spread_anchor_scores(11, {5: 0.8}, decay_distance=4)
    assert np.allclose(weights[[2, 3, 4, 5, 6, 7, 8]], [0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2])
    assert weights[1] == 0 and weights[9] == 0
    probabilities, fallback = normalize_sampling_probabilities(weights)
    assert not fallback
    assert probabilities.sum() == pytest.approx(1.0)



def test_overlapping_anchor_neighborhoods_take_maximum():
    weights = spread_anchor_scores(11, {4: 0.8, 6: 0.4}, decay_distance=4)
    assert weights[4] == pytest.approx(0.8)
    assert weights[5] == pytest.approx(0.8 * 0.75)
    assert weights[6] == pytest.approx(0.8 * 0.5)
    assert weights[0] == 0.0

def test_zero_weight_fallback_is_uniform_and_reported():
    probabilities, fallback = normalize_sampling_probabilities(np.zeros(5))
    assert fallback
    assert np.allclose(probabilities, np.full(5, 0.2))


def test_ads_pipeline_reuses_anchors_and_returns_normalized_probabilities():
    reference = TimeSeriesFragment(np.arange(14), entity_id="ref", fragment_id=1)
    source = TimeSeriesFragment(np.arange(14), entity_id="src", fragment_id=2)
    requests = {}

    def gradient_computer(fragment, starts):
        rows = []
        for start in starts:
            key = (fragment.fragment_id, int(start))
            requests[key] = requests.get(key, 0) + 1
            if int(start) == 1:
                rows.append([0.0, 0.0])
            else:
                rows.append([1.0, float(start) / 10.0])
        return np.asarray(rows, dtype=np.float32)

    report = ADSPipeline(
        ADSConfig(q=4, eta=1.5, top_k=3, decay_distance=4),
        gradient_computer,
        context_length=1,
        prediction_length=1,
    ).run([reference], [source])

    assert report.actual_gradient_counts["coarse"]["reference"] == 4
    assert report.actual_gradient_counts["coarse"]["source"] == 4
    assert report.nominal_gradient_counts["coarse_non_endpoint"]["reference"] == 2
    assert report.nominal_gradient_counts["coarse_non_endpoint"]["source"] == 2
    assert report.selected_top_k == [3, 3]
    assert [len(matches) for matches in report.matches_by_round] == [3, 3]
    assert sum(result.probabilities.sum() for result in report.source_results) == pytest.approx(1.0)
    assert all(count == 1 for count in requests.values())
    assert report.zero_norm_anchors["reference"] >= 1
    assert report.zero_norm_anchors["source"] >= 1
    assert report.zero_norm_pairs >= 1
    source_result = report.source_results[0]
    assert set(source_result.anchors).issubset(set(source_result.valid_starts.tolist()))



def test_compact_probability_descriptor_samples_only_legal_starts():
    reference = TimeSeriesFragment(np.arange(20), fragment_id=10)
    source = TimeSeriesFragment(np.arange(20), fragment_id=20)

    def gradient_computer(fragment, starts):
        return np.asarray([[1.0, float(start + fragment.fragment_id)] for start in starts],
                          dtype=np.float32)

    report = ADSPipeline(
        ADSConfig(q=2, eta=1.0, top_k=3, decay_distance=4),
        gradient_computer, context_length=2, prediction_length=2,
    ).run([reference], [source])
    descriptor = source_probability_descriptor(report)
    assert descriptor["mode"] == "anchor_neighbor_max_linear"
    assert descriptor["trajectories"][0]["candidate_count"] == len(
        report.source_results[0].valid_starts)
    rng = np.random.default_rng(0)
    fragment_ids, starts = sample_source_probability_descriptor(descriptor, rng, 200)
    sampler = SourceProbabilitySampler(descriptor)
    repeated_ids, repeated_starts = sampler.sample(np.random.default_rng(0), 200)
    assert np.array_equal(fragment_ids, repeated_ids)
    assert np.array_equal(starts, repeated_starts)
    legal = set(report.source_results[0].valid_starts.tolist())
    assert set(fragment_ids.tolist()) == {20}
    assert set(starts.tolist()) <= legal


def test_compact_probability_descriptor_uses_uniform_fallback():
    reference = TimeSeriesFragment(np.arange(8), fragment_id=1)
    source = TimeSeriesFragment(np.arange(8), fragment_id=2)

    def zero_gradient(_fragment, starts):
        return np.zeros((len(starts), 2), dtype=np.float32)

    report = ADSPipeline(
        ADSConfig(q=2, eta=1.0, top_k=1), zero_gradient,
        context_length=1, prediction_length=1,
    ).run([reference], [source])
    descriptor = source_probability_descriptor(report)
    assert descriptor["mode"] == "uniform"
    ids, starts = sample_source_probability_descriptor(
        descriptor, np.random.default_rng(4), 100)
    assert set(ids.tolist()) == {2}
    assert set(starts.tolist()) <= set(report.source_results[0].valid_starts.tolist())

def test_target_density_search_and_arm_sampling_interface(tmp_path):
    reference = TimeSeriesFragment(np.arange(12), fragment_id=10)
    source = TimeSeriesFragment(np.arange(12), fragment_id=20)

    def gradient_computer(fragment, starts):
        return np.asarray([[1.0, float(start)] for start in starts], dtype=np.float32)

    config = ADSConfig(
        q=2, eta=1.0, top_k=None,
        top_ratio=None, target_fine_density=0.2, rounds=1,
    )
    report = ADSPipeline(
        config, gradient_computer, context_length=1, prediction_length=1,
    ).run([reference], [source])
    assert report.selected_top_k[0] >= 1
    assert report.achieved_fine_density is not None
    assert report.actual_gradient_counts["fine"]["total"] >= 0

    index = WindowIndex([source], context_length=1, prediction_length=1)
    selected = ads_select(
        [reference], [source], index,
        gradient_computer=gradient_computer,
        ads_config=ADSConfig(q=2, eta=1.0, top_k=4),
        n_select=20,
        seed=7,
        sample_with_replacement=True,
    )
    assert selected.name == "ads"
    assert selected.selected.shape == (20, 2)
    assert len(selected.window_ids) == 20
    assert selected.cost.gradient_computations > 0
    assert selected.diagnostics["sampling"] == "weighted with replacement"
    from rage.experiments.regions import load_selection, save_selection
    selection_path = tmp_path / "ads-selection.npz"
    save_selection(selection_path, selected)
    restored = load_selection(selection_path)
    assert np.array_equal(restored.selected, selected.selected)
    assert restored.window_ids == selected.window_ids


def test_ads_config_and_cli_values_flow_through_main_runner(tmp_path):
    args = parse_main_args([
        "--selector", "ads", "--ads-q", "12", "--ads-eta", "2",
        "--ads-top-ratio", "0.1",
        "--ads-aggregation", "relu", "--ads-sampling-replacement",
    ])
    assert args.selector == "ads"
    assert args.ads_q == 12
    assert args.ads_top_ratio == pytest.approx(0.1)
    assert args.ads_sampling_replacement

    run_args = parse_run_args([
        "--config", "configs/rage/ads.yaml", "--selector", "ads",
        "--ads-q", "12", "--ads-top-ratio", "0.1",
    ])
    assert run_args.selector == "ads"
    assert run_args.ads_q == 12
    from rage.experiments.run import apply_cli_overrides
    config_path = __import__("pathlib").Path("configs/rage/ads.yaml")
    yaml_config = __import__("yaml").safe_load(config_path.read_text())
    merged = apply_cli_overrides(yaml_config, run_args)
    assert merged["selector"] == "ads"
    assert merged["ads_q"] == 12
    assert merged["ads_top_k"] is None
    assert merged["ads_top_ratio"] == pytest.approx(0.1)
    argv = _step_argv("main", merged, tmp_path / "run")
    joined = " ".join(argv)
    assert "--ads-top-ratio 0.1" in joined
    assert "--ads-top-k" not in joined
    config = {
        "steps": ["main"],
        "selector": "ads",
        "ads_q": 12,
        "ads_eta": 2.0,
        "ads_top_ratio": 0.1,
        "ads_aggregation": "relu",
        "with_baselines": [],
    }
    argv = _step_argv("main", config, tmp_path / "run")
    joined = " ".join(argv)
    assert "--selector ads" in joined
    assert "--ads-q 12" in joined
    assert "--ads-top-ratio 0.1" in joined
    assert "--ads-aggregation relu" in joined
    assert "--with-baselines" in joined
    assert "--with-baselines less" not in joined


@pytest.mark.parametrize("kwargs", [
    {"top_k": 3, "top_ratio": 0.5},
    {"top_k": 3, "top_ratio": 0.2},
    {"top_k": None, "target_fine_density": 0},
    {"rounds": 1.5},
    {"decay_distance": 2.5},
    {"top_k": 2.5},
])
def test_ads_config_rejects_conflicting_or_invalid_budget(kwargs):
    defaults = {"q": 4, "eta": 1, "top_k": 4}
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        ADSConfig(**defaults)



def test_freeze_run_document_records_ads_protocol(tmp_path):
    from rage.experiments import run as runner

    config = {
        "selector": "ads",
        "ads_q": 12,
        "ads_eta": 2.0,
        "ads_top_ratio": 0.1,
        "ads_top_k": None,
        "ads_rounds": 2,
        "ads_decay_distance": 4,
        "ads_aggregation": "relu",
        "ads_sampling_replacement": True,
        "steps": ["main"],
    }
    run_dir = tmp_path / "ads-run"
    commands = {"main": _step_argv("main", config, run_dir)}
    runner.freeze_run_doc("ads-test", config, ["main"], commands, run_dir)
    document = (run_dir / "RUN.md").read_text(encoding="utf-8")
    assert "ADS Configuration" in document
    assert "q | 12 |" in document
    assert "source=36.0, reference=18.0" in document
    assert "Top-K mode | rho=0.1" in document
    assert "Sampling replacement | True" in document
    assert "K_cover" not in document



def test_real_gradient_cache_is_reused_and_not_charged_as_new_compute():
    reference = TimeSeriesFragment(np.arange(9), fragment_id=30)
    source = TimeSeriesFragment(np.arange(9), fragment_id=40)
    starts = reference.valid_starts(1, 1, 1)
    cache = {}
    for side, fragment in (("reference", reference), ("source", source)):
        for start in starts:
            cache[(side, fragment.fragment_id, int(start))] = np.asarray(
                [1.0, float(start)], dtype=np.float32)

    def unexpected_compute(_fragment, _starts):
        raise AssertionError("cached gradients should be reused")

    report = ADSPipeline(
        ADSConfig(q=3, eta=1.0, top_k=2),
        unexpected_compute,
        context_length=1,
        prediction_length=1,
        gradient_cache=cache,
    ).run([reference], [source])
    assert report.cached_gradient_counts == {
        "reference": len(report.reference_results[0].anchors),
        "source": len(report.source_results[0].anchors),
    }
    assert report.actual_gradient_counts["coarse"]["total"] == 0
    assert report.actual_gradient_counts["fine"]["total"] == 0
    assert 0 < len(report.source_results[0].anchors) < len(starts)


def test_ads_uses_supplied_coarse_anchors_and_only_loads_requested_cache_rows():
    reference = TimeSeriesFragment(np.arange(12), fragment_id=30)
    source = TimeSeriesFragment(np.arange(12), fragment_id=40)
    starts = reference.valid_starts(1, 1, 1)
    cache = {
        (side, fragment.fragment_id, int(start)): np.asarray(
            [1.0, float(start)], dtype=np.float32)
        for side, fragment in (("reference", reference), ("source", source))
        for start in starts
    }

    def unexpected_compute(_fragment, _starts):
        raise AssertionError("cached gradients should be reused")

    report = ADSPipeline(
        ADSConfig(q=3, eta=1.0, top_k=1),
        unexpected_compute,
        context_length=1,
        prediction_length=1,
        gradient_cache=cache,
        coarse_anchor_positions={("source", 40): [0, len(starts) - 1]},
    ).run([reference], [source])
    source_result = report.source_results[0]
    assert {int(starts[0]), int(starts[-1])}.issubset(source_result.anchors)
    assert report.cached_gradient_counts["source"] == len(source_result.anchors)
    assert report.actual_gradient_counts["coarse"]["total"] == 0
    assert report.actual_gradient_counts["fine"]["total"] == 0
    assert len(source_result.anchors) < len(starts)


def test_run_streams_step_output_and_marks_completion(tmp_path, monkeypatch, capsys):
    import sys
    from pathlib import Path
    from rage.experiments import run as runner

    project = tmp_path / "project"
    config_path = project / "ads.json"
    project.mkdir()
    config_path.write_text('{"batch": "ads-stream-test", "steps": ["main"], "selector": "ads", "ads_eta": 1.0}')
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        runner, "_step_argv",
        lambda _step, _config, _run_dir: [
            sys.executable, "-u", "-c", "print('live-ads-output')",
        ],
    )

    def freeze(_run_id, _config, _steps, _commands, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "RUN.md").write_text("test run")

    monkeypatch.setattr(runner, "freeze_run_doc", freeze)
    code = runner.main(["--config", str(config_path), "--out-root", "outputs"])

    output = capsys.readouterr().out
    assert code == 0
    assert "live-ads-output" in output
    run_dir = next((project / "outputs").glob("rage-ads-stream-test-*"))
    log_path = project / "logs" / "rage_runs" / run_dir.name / "main.log"
    assert "live-ads-output" in log_path.read_text()
    assert (run_dir / ".complete").is_file()
