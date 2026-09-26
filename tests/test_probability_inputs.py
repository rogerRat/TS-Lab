import json

import numpy as np

from rags.core import TimeSeriesFragment
from rags.experiments.development_inputs import load_fragments, save_fragments
from rags.experiments.students import atomic_json
from scripts.exp.probability_inputs import prepare


def test_fixed_development_series_keep_source_pool_and_exclude_exam(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    refs = [TimeSeriesFragment(np.arange(600, dtype=np.float32) + i,
        entity_id=f"task::{i}", fragment_id=i) for i in range(12)]
    save_fragments(inputs / "task__reference.npz", refs)
    save_fragments(inputs / "pool.npz", refs[:1])
    atomic_json(inputs / "pool_manifest.json", dict(total_equivalents=1))
    np.savez(inputs / "task__exam.npz", windows=np.ones((1, 576)))
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare(inputs, a, ["task"], series=8)
    assert prepare(inputs, b, ["task"], series=8) == first
    chosen = load_fragments(a / "task__reference.npz")
    assert len(chosen) == len({f.fragment_id for f in chosen}) == 8
    for f in chosen:
        np.testing.assert_array_equal(f.data, refs[f.fragment_id].data)
    assert (a / "pool.npz").read_bytes() == (inputs / "pool.npz").read_bytes()
    assert not (a / "task__exam.npz").exists()
    assert not json.loads((a / "development_subset.json").read_text())["formal_exam_copied"]
