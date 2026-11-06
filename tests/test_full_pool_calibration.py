import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rage.experiments import full_pool_calibration as module


def test_full_scan_p10_and_fragment_resume(tmp_path, monkeypatch):
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    module.import_inputs(inputs, dict(
        reference=[dict(entity="ref", fragment_id=1000, values=np.arange(578).tolist())],
        source=[dict(entity="source", fragment_id=0, values=np.arange(579).tolist())],
        provenance={"fixture": "not real calibration"}))
    seen = []
    def gradient(windows):
        seen.append(windows.copy())
        return np.zeros(len(windows)), np.tile(np.array([1., 2.], dtype=np.float32), (len(windows), 1))
    probe = SimpleNamespace(device="cpu", model=torch.nn.Linear(1, 1),
        config=SimpleNamespace(projection_dimension=2), batch_gradient_sketch=gradient)
    identity = {"model_id": "fixture"}
    monkeypatch.setattr(module, "build_model", lambda *a, **kw: (probe, probe, identity))
    module.calibrate("chronos-base", inputs, out)
    report = json.loads((out / "epsilon.json").read_text())
    assert report["count"] == sum(len(batch) for batch in seen) == 7
    np.testing.assert_array_equal(np.concatenate(seen)[:,0], [0,1,2,0,1,2,3])
    assert report["epsilon"] == np.sqrt(5.)
    assert (out / "reference-1000.npz").exists() and (out / "source-0.npz").exists()
    seen.clear()
    module.calibrate("chronos-base", inputs, out)
    assert not seen
    identity["model_id"] = "different"
    with pytest.raises(ValueError, match="changed"):
        module.calibrate("chronos-base", inputs, out)
