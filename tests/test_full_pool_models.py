import json
from types import SimpleNamespace

import torch

from rage.experiments import full_pool_models as module


def test_live_model_identity_equals_persisted_identity(monkeypatch):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(to_dict=lambda: dict(id2label={0: "LABEL_0"}, shape=(1, 2)))
    def engine(config, device):
        return SimpleNamespace(model=model, config=config, device=device)
    monkeypatch.setattr("gradient.chronos.ChronosT5Engine", engine)
    monkeypatch.setitem(module.PARAMETER_COUNTS, "chronos-base", 2)
    _, _, identity = module.build_model("chronos-base", device="cpu")
    assert identity == json.loads(json.dumps(identity))
    assert identity["config"]["id2label"] == {"0": "LABEL_0"}
