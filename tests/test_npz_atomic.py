import json

import numpy as np

from rags.core import TimeSeriesFragment
from rags.experiments.arms import GradientBlock, load_pool_blocks, save_pool_blocks
from rags.experiments.regions import save_selection


def test_gradient_npz_is_atomic_and_has_format_field(tmp_path):
    path = tmp_path / "blocks.npz"
    block = GradientBlock(1, 0, 1, np.array([0, 1]), np.array([1., 0.]), np.array([1., 0.]))
    save_pool_blocks(path, [block], {"gradients": 1})
    assert path.exists() and not (tmp_path / "blocks.npz.tmp").exists()
    with np.load(path) as payload:
        meta = json.loads(str(payload["meta"]))
    assert meta["format"] == "gradient-blocks-npz-v1"
    assert len(load_pool_blocks(path)[0]) == 1


def test_selection_npz_is_atomic(tmp_path):
    from types import SimpleNamespace
    output = SimpleNamespace(name="random", selected=np.ones((1, 2), dtype=np.float32),
        window_ids=[(1, 0)], cost=SimpleNamespace(), diagnostics={})
    save_selection(tmp_path / "selection.npz", output)
    assert (tmp_path / "selection.npz").exists()
    assert not (tmp_path / "selection.npz.tmp").exists()
