"""Trajectory sampling/means and pinned-upstream TSDS retrieval parity."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.trajectory import budget_config, uniform_positions, trajectory_features
from baselines.less.runner import less_select
from baselines.less.algorithms import less_influence_scores
from rags.core import TimeSeriesFragment
from rags.experiments.arms import WindowIndex
from rags.execution import SelectionExecution


def fragments(count=4, length=30):
    return [TimeSeriesFragment((np.sin(np.arange(length + i) / (i + 2)) + i).astype(np.float32),
                               fragment_id=10 + i) for i in range(count)]


class Probe:
    device = "cpu"
    config = SimpleNamespace(model_path="fixture", projection_dimension=3, projection_seed=42)
    execution = SelectionExecution(gradient_batch=3, inference_batch=3)

    def __init__(self):
        self.seen = []

    def batch_gradient_sketch(self, windows):
        self.seen.extend(windows.copy())
        return np.zeros(len(windows)), self.features(windows)

    @staticmethod
    def features(windows):
        return np.stack([windows[:, 0], windows.mean(axis=1), windows[:, -1]], axis=1)


@pytest.mark.parametrize("size", [0, 1, 2, 3, 21, 234, 10001])
def test_uniform_spreading_matches_rounded_budget(size):
    index = WindowIndex(fragments(), context_length=4, prediction_length=2)
    config = budget_config(index)
    positions = uniform_positions(size, config)
    assert len(positions) == sum(config.budgets(size))
    assert len(np.unique(positions)) == len(positions)
    if size:
        assert positions[0] == 0 and positions[-1] == size - 1
    if len(positions) > 2:
        assert np.ptp(np.diff(positions)) <= 1


def test_feature_mean_before_normalization_and_cache(tmp_path, monkeypatch):
    index = WindowIndex(fragments(), context_length=4, prediction_length=2)
    config = budget_config(index, .2, .1, .05)
    engine = Probe()
    manifest = {"seed": 0, "total_points": 126}
    expected = []
    for fid in index.fragment_lengths:
        positions = index.bounds(fid)[0] + uniform_positions(index.fragment_lengths[fid], config)
        expected.append(engine.features(index.windows_at(positions)).mean(axis=0))
    # Both baseline adapters consume precisely the same sampled windows.
    monkeypatch.setattr("rags.execution.encoder_embeddings", lambda e, w, **kw: e.features(w))
    for kind in ("tsds",):
        actual, info = trajectory_features(engine, index, config, kind=kind,
                                           cache_dir=tmp_path, pool_manifest=manifest)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        assert actual.dtype == np.float32
        assert info["executed_windows"] == sum(info["sample_counts"])
        cached, warm = trajectory_features(engine, index, config, kind=kind,
                                           cache_dir=tmp_path, pool_manifest=manifest)
        np.testing.assert_array_equal(cached, actual)
        assert warm["pool_cached"] and warm["executed_windows"] == 0
        changed = budget_config(index, .3, .1, .05)
        with pytest.raises(ValueError,match="cache fields differ"):
            trajectory_features(engine,index,changed,kind=kind,cache_dir=tmp_path,pool_manifest=manifest)




def test_ivf_and_kde_match_literal_upstream():
    import faiss
    from baselines.tsds.retrieval import trajectory_neighbours
    # Enough candidates for genuine IVF approximation (nlist > nprobe).
    rng = np.random.RandomState(8)
    xb = rng.normal(size=(1024, 8)).astype(np.float32)
    xq = rng.normal(size=(20, 8)).astype(np.float32)

    def upstream_index(data):
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(data.shape[-1]), data.shape[-1],
                                   int(np.sqrt(data.shape[0])) // 2)
        index.train(data)
        index.add(data)
        index.nprobe = 10
        return index

    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(4)
    try:
        distances, ids = upstream_index(xb).search(xq, 64)
        order = np.argsort(distances, axis=-1)
        distances = np.sqrt(np.take_along_axis(distances, order, axis=1))
        ids = np.take_along_axis(ids, order, axis=1)
        unique = list(set(ids.reshape(-1)))
        features = xb[unique]
        d2, _ = upstream_index(features).search(features, 16)
        kernel = 1 - d2 / 4. ** 2
        kde = (kernel * (kernel > 0)).sum(axis=-1)
        lookup = dict(zip(unique, kde))
        expected_kde = np.vectorize(lookup.__getitem__)(ids)
        actual_d, actual_i, actual_kde, info = trajectory_neighbours(xb, xq)
        np.testing.assert_array_equal(actual_d, distances)
        np.testing.assert_array_equal(actual_i, ids)
        np.testing.assert_array_equal(actual_kde, expected_kde)
        assert info["nlist"] == 16 and info["nprobe"] == 10
        _, _, unit_kde, _ = trajectory_neighbours(xb, xq, sigma=0)
        assert np.all(unit_kde == 1)
    finally:
        faiss.omp_set_num_threads(previous)


def test_old_run_and_changed_budget_cannot_reuse_student_cells(tmp_path):
    from rags.experiments.main_table import parse_args, check_baseline_resume
    from baselines.trajectory import REVISION
    args = parse_args([])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"protocol": vars(args)}))
    with pytest.raises(ValueError, match="new --out"):
        check_baseline_resume(tmp_path, args)
    args.language_baseline_revision = REVISION
    from rags.experiments.regions import REVISION as DELIVERY_REVISION
    args.delivery_revision = DELIVERY_REVISION
    from rags.experiments.arms import RANDOM_SAMPLING_REVISION
    args.random_sampling_revision = RANDOM_SAMPLING_REVISION
    path.write_text(json.dumps({"protocol": vars(args)}))
    check_baseline_resume(tmp_path, args)
    args.cover_density = .2
    with pytest.raises(ValueError, match="new --out"):
        check_baseline_resume(tmp_path, args)




def test_tsds_multinomial_and_duplicate_artifact(tmp_path, monkeypatch):
    from baselines.tsds import runner
    from rags.experiments.main_table import _save_selection, _load_selection
    index = WindowIndex(fragments(40, 12), context_length=4, prediction_length=2)
    engine = Probe()
    monkeypatch.setattr("rags.execution.encoder_embeddings", lambda e, w, **kw: e.features(w))
    probs = np.zeros(40)
    probs[[5, 13]] = [.7, .3]
    monkeypatch.setattr(runner, "tsds_probability_assignment", lambda *a, **kw: probs)
    kwargs = dict(reference_fragments=fragments(8, 20), index=index, context_length=4,
                  prediction_length=2, n_select=100, seed=7,
                  cache_dir=tmp_path, pool_manifest={"seed": 0, "total_points": 1260})
    out = runner.tsds_select(engine, **kwargs)
    assert out.diagnostics["parameters"] == dict(max_k=64, kde_k=16, sigma=4., alpha=.1,
                                                  c=5., sample_size=100, seed=7)
    assert out.diagnostics["parameter_origin"] == "experiment_frozen_not_upstream_defaults"
    expected = np.random.RandomState(7).multinomial(100, probs)
    np.testing.assert_array_equal(out.diagnostics["trajectory_sample_counts"], expected)
    assert set(out.selected.regions[:, 0]) == {15, 23}
    assert out.selected.coverage()["repeated_points"] > 0
    copies = np.repeat(np.arange(10, 50), expected)
    np.random.default_rng(7).shuffle(copies)
    lengths = np.asarray([index.by_fragment[int(fid)].length for fid in copies])
    count = np.searchsorted(np.cumsum(lengths), 600) + 1
    np.testing.assert_array_equal(out.selected.regions[:, 0], copies[:count])
    path = tmp_path / "task__tsds.npz"
    _save_selection(path, out)
    loaded = _load_selection(path)
    assert loaded.window_ids == out.window_ids
    np.testing.assert_array_equal(loaded.selected[:], out.selected[:])
    np.testing.assert_array_equal(loaded.selected.regions, out.selected.regions)
    warm = runner.tsds_select(engine, **kwargs)
    assert warm.window_ids == out.window_ids
    assert warm.cost.notes["pool_cached"]
    assert warm.cost.feature_computations == warm.cost.notes["reference_features"]


def test_shared_thread_policy_also_caps_and_restores_faiss():
    import faiss
    from rags.execution import selection_execution
    previous = faiss.omp_get_max_threads()
    engine = Probe()
    engine.execution = SelectionExecution(cpu_threads=2)
    with selection_execution(engine):
        assert faiss.omp_get_max_threads() == 2
    assert faiss.omp_get_max_threads() == previous




@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_chronos_512_64_language_baselines_cold_warm(tmp_path):
    from test_selection_execution import native_engine
    from rags.engines import ChronosT5Engine, ChronosT5Config, _FullModelProjector
    from rags.execution import selection_execution
    from baselines.runners import run_baseline
    engine = ChronosT5Engine.__new__(ChronosT5Engine)
    engine.__dict__.update(native_engine().__dict__)
    engine.config = ChronosT5Config(context_length=512, prediction_length=64,
                                    projection_dimension=32)
    engine._named_trainable = list(engine.model.named_parameters())
    engine._parameters = [p for _, p in engine._named_trainable]
    engine.parameter_count = sum(p.numel() for p in engine._parameters)
    engine.projector = _FullModelProjector(engine._named_trainable, 32, 42, engine.device)
    engine.execution = SelectionExecution(gradient_batch=4, inference_batch=8)
    rng = np.random.default_rng(52)
    pool = [TimeSeriesFragment(rng.normal(size=600).astype(np.float32), fragment_id=i)
            for i in range(40)]
    refs = [TimeSeriesFragment(rng.normal(size=600).astype(np.float32), fragment_id=100 + i)
            for i in range(8)]
    index = WindowIndex(pool, context_length=512, prediction_length=64)
    base = {k: v.clone() for k, v in engine.model.state_dict().items()}
    with selection_execution(engine):
        for name in ("tsds",):
            kwargs = dict(engine=engine, reference_fragments=refs, index=index,
                          context_length=512, prediction_length=64, n_select=40,
                          cache_dir=tmp_path, pool_manifest={"seed": 52, "total_points": 24000}, seed=7)
            cold = run_baseline(name, **kwargs)
            warm = run_baseline(name, **kwargs)
            assert len(cold.selected.regions) == 39
            assert cold.selected.coverage()["budget_points"] == 40 * 576
            assert cold.window_ids == warm.window_ids
            np.testing.assert_array_equal(cold.selected[:], warm.selected[:])
            assert not cold.cost.notes["pool_cached"] and warm.cost.notes["pool_cached"]
            assert cold.diagnostics["source_sampling"]["sample_counts"] == [2] * 40
            assert warm.cost.gradient_computations + warm.cost.feature_computations == 16
            if name == "tsds":
                assert cold.selected.coverage()["repeated_points"] > 0
                copies = np.repeat(np.arange(40), cold.diagnostics["trajectory_sample_counts"])
                np.random.default_rng(7).shuffle(copies)
                np.testing.assert_array_equal(cold.selected.regions[:, 0], copies[:39])
            for key, value in engine.model.state_dict().items():
                torch.testing.assert_close(value, base[key], rtol=0, atol=0)
        from rags.experiments.train_eval import run_arm_training
        result = run_arm_training(engine, source_windows=cold.selected.prefix(2), source_epochs=4,
            batch_size=2, learning_rate=1e-5, seed=0, checkpoint_dir=tmp_path / "native-training",
            test_windows=index.windows_at(np.array([0, 1])), test_entities=["a", "b"],
            context_length=512, test_scales=np.ones(2), base_state=base)
        assert [row["step"] for row in result["checkpoints"]] == [1, 2, 3, 4]
        assert result["train"]["source"]["legal_window_occurrences"] == 50
