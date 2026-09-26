"""BF16 model, gradients and AdamW moments; no FP32 master parameters."""
import torch

POLICY = "bf16-model-gradient-adamw-v2"


def make_adamw(model, **kwargs):
    # Accept a module or pre-split parameter groups (probe weight-decay split);
    # both keep model parameters, gradients and AdamW moments in BF16.
    parameters = model if isinstance(model, (list, tuple)) else model.parameters()
    return torch.optim.AdamW(parameters, foreach=False, **kwargs)
