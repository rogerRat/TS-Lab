from collections import Counter

import numpy as np

from rage.core import TimeSeriesFragment
from rage.experiments.development_inputs import load_fragments, save_fragments


def test_packed_development_fragments_preserve_identity_and_values(tmp_path):
    original = [TimeSeriesFragment(np.arange(n, dtype=np.float32), entity_id=f'task::{i}', fragment_id=i)
                for i,n in [(7,600),(3,800)]]
    path = tmp_path/'fragments.npz'
    save_fragments(path, original)
    restored = load_fragments(path)
    for a,b in zip(original,restored):
        assert a.entity_id == b.entity_id and a.fragment_id == b.fragment_id
        np.testing.assert_array_equal(a.data,b.data)


def test_packed_values_are_decompressed_once_for_all_trajectories(tmp_path, monkeypatch):
    fragments = [TimeSeriesFragment(np.arange(10 + i, dtype=np.float32),
        entity_id=f"entity-{i}", fragment_id=i + 20) for i in range(4)]
    path = tmp_path / "pool.npz"
    save_fragments(path, fragments)
    accesses = Counter()
    original = np.lib.npyio.NpzFile.__getitem__
    def read(archive, key):
        accesses[key] += 1
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read)
    restored = load_fragments(path)
    assert accesses == dict(values=1, offsets=1, entities=1, ids=1)
    for actual, expected in zip(restored, fragments):
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.entity_id == expected.entity_id
        assert actual.fragment_id == expected.fragment_id
