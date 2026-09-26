"""CRBEngineMulti must reproduce CRBEngine exactly per fragment."""

from dataclasses import replace

import zlib

import numpy as np

from rags.core import TimeSeriesFragment
from rags.crb import CRBConfig, CRBEngine, CRBEngineMulti


class SyntheticField:
    """Deterministic synthetic gradients addressed by window content.

    The value depends only on the extracted window (as with the real engine
    computer), so the sequential and multi engines -- which request the same
    windows through different fragment views -- see identical gradients.
    """

    def __init__(self, dim=6, seed=0):
        self.seed = seed
        self.table = {}

    def __call__(self, fragment, physical_starts):
        rows = []
        for start in np.asarray(physical_starts):
            window = np.asarray(fragment.extract_window(int(start), 8, 4),
                                dtype=np.float32)
            fingerprint = zlib.crc32(window.tobytes())
            key = (int(fingerprint), self.seed)
            if key not in self.table:
                rng = np.random.default_rng([self.seed, int(fingerprint)])
                base = rng.normal(size=6).astype(np.float32)
                wave = np.sin(np.arange(6) * float(fingerprint % 97) * 0.37
                              ).astype(np.float32)
                self.table[key] = base + 0.7 * wave
            rows.append(self.table[key])
        return np.stack(rows).astype(np.float32)


def _fragments():
    lengths = [700, 1300, 900, 640]
    return [TimeSeriesFragment(
        np.arange(length, dtype=np.float32) * (1.0 + 0.1 * i),
        fragment_id=i + 1) for i, length in enumerate(lengths)]


CONFIGS = [
    CRBConfig(context_length=8, prediction_length=4, window_stride=1,
              cover_density=0.10, error_density=0.05, split_density=0.03,
              mean_epsilon=0.4, endpoint_epsilon=0.35, curvature_cap=1.0),

]


def _assert_same_results(seq_results, multi_results):
    assert len(seq_results) == len(multi_results)
    for a, b in zip(seq_results, multi_results):
        assert sorted(a.anchors) == sorted(b.anchors)
        for key in a.anchors:
            assert np.allclose(a.anchors[key], b.anchors[key], atol=1e-6)
        assert (a.cover_computations, a.error_computations, a.split_computations) == \
               (b.cover_computations, b.error_computations, b.split_computations)
        assert (a.discarded_no_signal, a.discarded_antagonistic, a.banned_budget) == \
               (b.discarded_no_signal, b.discarded_antagonistic, b.banned_budget)
        assert len(a.blocks) == len(b.blocks)
        for block_a, block_b in zip(a.blocks, b.blocks):
            assert np.array_equal(block_a.start_positions, block_b.start_positions)
            assert np.allclose(block_a.direction, block_b.direction, atol=1e-6)


def test_multi_matches_sequential_all_variants():
    for config in CONFIGS:
        field = SyntheticField(seed=3)
        seq = [CRBEngine(config, field).process_fragment(f) for f in _fragments()]
        field_multi = SyntheticField(seed=3)
        multi = CRBEngineMulti(config, field_multi).process_fragments(_fragments())
        _assert_same_results(seq, multi)


def test_multi_batches_across_fragments():
    """The multi engine must merge probes from several fragments per call."""
    calls: list[int] = []

    def computer(fragment, physical_starts):
        calls.append(len(physical_starts))
        return np.stack([np.ones(4, dtype=np.float32) for _ in physical_starts])

    config = CRBConfig(context_length=8, prediction_length=4,
                       cover_density=0.08, error_density=0.04,
                       mean_epsilon=1e-8, endpoint_epsilon=1e-8)
    fragments = _fragments()
    sequential_calls = []
    for fragment in fragments:
        before = len(calls)
        CRBEngine(config, computer).process_fragment(fragment)
        sequential_calls.append(len(calls) - before)
    calls.clear()
    CRBEngineMulti(config, computer).process_fragments(fragments)
    # far fewer computer invocations than one-per-fragment rounds
    assert len(calls) <= max(sequential_calls) + 5
    assert max(calls) > 1  # probes from different fragments share batches


def test_multi_handles_empty_and_single_start_fragments():
    config = CRBConfig(context_length=8, prediction_length=4,
                      cover_density=0.1, error_density=0.05, mean_epsilon=1e-8)
    field = SyntheticField(seed=1)
    empty = TimeSeriesFragment(np.arange(5, dtype=np.float32), fragment_id=9)
    tiny = TimeSeriesFragment(np.arange(12, dtype=np.float32), fragment_id=10)
    normal = _fragments()[0]
    multi = CRBEngineMulti(config, field)
    results = multi.process_fragments([empty, tiny, normal])
    seq_empty = CRBEngine(config, field).process_fragment(empty)
    seq_tiny = CRBEngine(config, field).process_fragment(tiny)
    assert len(results) == 3
    assert len(results[0].blocks) == 0 and seq_empty.blocks == []
    assert len(results[1].blocks) == 1  # retain the single legal start
    assert results[1].computed_gradients == seq_tiny.computed_gradients
    assert len(results[2].blocks) > 0
