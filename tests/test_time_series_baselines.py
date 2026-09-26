"""Method fidelity and small end-to-end tests, without downloaded weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rags.core import TimeSeriesFragment
from rags.experiments.arms import WindowIndex
from baselines.algorithms import ltsv_window_scores
from baselines.lp_mirror import runner as lp
from baselines.ltsv import runner as ltsv
from baselines.runners import run_baseline


def fixture_data():
    pool = [TimeSeriesFragment(np.linspace(-2, 3, 12), fragment_id=1)]
    ref = [TimeSeriesFragment(np.linspace(0, 2, 10), fragment_id=9)]
    return WindowIndex(pool, context_length=3, prediction_length=2), ref


def test_ltsv_averages_points_not_coverage_weights():
    result = ltsv_window_scores(np.array([0., 2.]), np.array([1, 2]), np.array([0]), 2)
    np.testing.assert_allclose(result, [0.5])  # Old implementation returned 2/3.


def test_ltsv_aggregation_matches_literal_paper_on_overlapping_blocks():
    sums, counts = np.zeros(9), np.zeros(9, dtype=int)
    blocks = [(0, 4, -2.), (2, 6, 3.), (5, 9, 1.)]
    for lo, hi, value in blocks:
        sums[lo:hi] += value
        counts[lo:hi] += 1
    expected_points = [np.mean([v for lo, hi, v in blocks if lo <= t < hi]) for t in range(9)]
    expected = [np.mean(expected_points[t:t + 4]) for t in range(6)]
    np.testing.assert_allclose(ltsv_window_scores(sums, counts, np.arange(6), 4), expected)


def test_ltsv_uncovered_windows_are_not_scored():
    result = ltsv_window_scores(np.ones(3), np.array([1, 0, 1]), np.array([0]), 3)
    assert np.isneginf(result[0])


def test_lp_proxy_visits_every_source_once_per_epoch(monkeypatch):
    index, _ = fixture_data()
    seen = []
    original = index.windows_at
    def record(positions):
        seen.extend(positions.tolist())
        return original(positions)
    monkeypatch.setattr(index, 'windows_at', record)
    steps = lp._train_lp_proxy(torch.nn.Linear(3, 2), index, 'cpu',
                             context_length=3, epochs=2, batch=3, lr=.01, seed=0)
    assert seen == list(range(len(index))) * 2
    assert steps == 6  # Includes the partial last batch, no overshoot or replacement.


def test_lp_matches_independent_forward_inf_training_and_scores(monkeypatch):
    index, ref = fixture_data()
    captured = []
    original_score = lp.lp_mirror_scores
    def capture(before, after):
        score = original_score(before, after)
        captured.extend(score.tolist())
        return score
    monkeypatch.setattr(lp, 'lp_mirror_scores', capture)
    actual = lp.lp_mirror_select(SimpleNamespace(device='cpu'), reference_fragments=ref,
        index=index, context_length=3, prediction_length=2, n_select=4,
        proxy_batch=3, adapt_batch=2, reference_stride=1, seed=7)

    # Independent implementation of upstream Adam -> SGD -> before/after MSE.
    torch.manual_seed(7)
    proxy = torch.nn.Linear(3, 2)
    def scaled_windows(fragment):
        data = fragment.data.astype(np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(fragment.data, 5).copy()
        return torch.from_numpy(((windows - float(data.mean())) / float(data.std())).astype(np.float32))
    source = scaled_windows(index.by_fragment[1])
    target = scaled_windows(ref[0])
    adam = torch.optim.Adam(proxy.parameters(), lr=.01)
    for z in source.split(3):
        adam.zero_grad()
        torch.nn.functional.mse_loss(proxy(z[:, :3]), z[:, 3:]).backward()
        adam.step()
    with torch.no_grad():
        before = (proxy(source[:, :3]) - source[:, 3:]).square().mean(1)
    sgd = torch.optim.SGD(proxy.parameters(), lr=.001)
    for z in torch.utils.data.DataLoader(target, batch_size=2, shuffle=True):
        sgd.zero_grad()
        torch.nn.functional.mse_loss(proxy(z[:, :3]), z[:, 3:]).backward()
        sgd.step()
    with torch.no_grad():
        after = (proxy(source[:, :3]) - source[:, 3:]).square().mean(1)
    np.testing.assert_allclose(captured, (before - after).numpy(), atol=2e-7)
    expected = np.argsort(captured)[::-1][:4]
    assert actual.window_ids == [(1, int(i + 1)) for i in expected]
    assert actual.diagnostics['adapt_steps'] == 3
    assert actual.cost.proxy_training_steps == 6
    assert len(actual.selected) == 4  # Non-overlap could fit only two here.


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.seen = []

    def forward(self, input_ids, labels, **kwargs):
        value = self.weight * input_ids.mean(1)
        logits = torch.stack([value, -value], dim=-1)[:, None, :].expand(-1, 2, -1)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, **kwargs):
        assert 'labels' not in kwargs
        self.seen.append(input_ids.clone())
        prediction = (input_ids.mean(1) * self.weight)[:, None].expand(-1, 2)
        return torch.cat([torch.zeros(len(input_ids), 1), prediction], dim=1)


def toy_engine():
    return SimpleNamespace(
        device='cpu', model=ToyModel(),
        config=SimpleNamespace(context_length=3, prediction_length=2),
        tokenizer=SimpleNamespace(
            context_input_transform=lambda x: (x, torch.ones_like(x), torch.ones(len(x))),
            output_transform=lambda x, scale: x,
        ),
        batch_tensors=lambda w: dict(input_ids=torch.tensor(w[:, :3]),
                                    labels=torch.zeros(len(w), 2, dtype=torch.long)),
    )


def test_reference_mse_uses_context_only_and_numeric_forecasts():
    engine = toy_engine()
    windows = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    assert ltsv._ltsv_reference_mse(engine, windows) == pytest.approx(((4-.4)**2+(5-.4)**2)/2)
    np.testing.assert_array_equal(engine.model.seen[0], windows[:, :3])


@pytest.mark.parametrize('block_length', [5, 7])
def test_ltsv_end_to_end_restores_model_and_retains_top_k(block_length, tmp_path):
    index, ref = fixture_data()
    engine = toy_engine()
    before = engine.model.weight.detach().clone()
    output = run_baseline('ltsv', engine=engine, reference_fragments=ref,
        index=index, context_length=3, prediction_length=2, n_select=4,
        block_length=block_length, block_stride=2, context_windows=2, cache_dir=tmp_path)
    torch.testing.assert_close(engine.model.weight, before, rtol=0, atol=0)
    assert len(output.selected) == 4
    assert output.diagnostics['reference_loss'] == 'forecast_mse'
    assert output.diagnostics['static_framework_verified']
    assert output.cost.gradient_computations >= 3
    assert output.cost.notes['scores_cached'] is False
    forecasts = len(engine.model.seen)
    cached = run_baseline('ltsv', engine=engine, reference_fragments=ref,
        index=index, context_length=3, prediction_length=2, n_select=4,
        block_length=block_length, block_stride=2, context_windows=2, cache_dir=tmp_path)
    assert cached.cost.notes['scores_cached'] is True
    assert len(engine.model.seen) == forecasts
    np.testing.assert_array_equal(cached.selected, output.selected)


def test_ltsv_restores_model_if_reference_evaluation_fails(monkeypatch):
    index, ref = fixture_data()
    engine = toy_engine()
    before = engine.model.weight.detach().clone()
    calls = []
    def fail_after_update(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('reference failed')
        return 1.
    monkeypatch.setattr(ltsv, '_ltsv_reference_mse', fail_after_update)
    with pytest.raises(RuntimeError, match='reference failed'):
        ltsv.ltsv_select(engine, reference_fragments=ref, index=index,
                      context_length=3, prediction_length=2, n_select=2)
    torch.testing.assert_close(engine.model.weight, before, rtol=0, atol=0)
