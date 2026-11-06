import copy
import torch
from rage.experiments.precision import make_adamw


def test_bf16_optimizer_has_no_master_weights_and_resumes():
    model = torch.nn.Linear(2, 1).bfloat16()
    optimizer = make_adamw(model, lr=.01)
    def step(model, optimizer):
        optimizer.zero_grad()
        model(torch.ones(2, 2, dtype=torch.bfloat16)).square().mean().backward()
        optimizer.step()
    step(model, optimizer)
    assert not hasattr(optimizer, "master_parameters")
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16
    clone = copy.deepcopy(model)
    resumed = make_adamw(clone, lr=.01)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(clone, resumed)
    for original, restored in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
