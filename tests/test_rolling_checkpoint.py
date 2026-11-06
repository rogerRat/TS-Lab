from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rage.experiments.train_eval import fine_tune


def test_previous_checkpoint_survives_until_new_evaluation_succeeds(tmp_path):
    model = torch.nn.Linear(3, 1)
    engine = SimpleNamespace(model=model, device="cpu")
    engine.training_loss = lambda w: model(torch.tensor(w[:, :3])).square().mean()
    windows = np.ones((4, 4), dtype=np.float32)
    path = tmp_path / "resume.pt"
    def observe(step):
        assert (tmp_path / "resume.pt.tmp").exists()
        if step == 2:
            assert torch.load(path, weights_only=False)["step"] == 1
            raise RuntimeError("evaluation failure")
    kwargs = dict(steps=3, batch_size=1, learning_rate=1e-4, seed=0,
                  resume_path=path, checkpoint_steps=[1, 2, 3])
    with pytest.raises(RuntimeError, match="evaluation failure"):
        fine_tune(engine, windows, **kwargs, on_step=observe)
    assert torch.load(path, weights_only=False)["step"] == 1
    result = fine_tune(engine, windows, **kwargs)
    assert result["resumed_from_step"] == 1
    assert torch.load(path, weights_only=False)["step"] == 3
    assert not (tmp_path / "resume.pt.tmp").exists()
