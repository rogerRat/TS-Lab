"""Official hyperparameter references, adapted to the project's full-parameter loop.

TimesFM's published example is LoRA: its optimizer settings are a starting
reference here, not a claim that this full-parameter run reproduces that recipe.
"""
from copy import deepcopy

MODELS = ("chronos-base", "timesfm-2.5", "timemoe-200m")
RECIPES = {
    "chronos-base": {
        "source": "https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/scripts/README.md",
        "source_config": "https://github.com/amazon-science/chronos-forecasting/blob/4dbf163c2734c089cdf7da2b86fde48862ff9c6f/scripts/training/configs/chronos-t5-base.yaml",
        "reference_scope": "Original T5 fine-tuning example (Small); Base config has the same LR. Not Chronos-2.",
        "official_duration": "1000 steps in fine-tuning example; not 1000 epochs",
        "learning_rate": 1e-3, "batch_size": 32, "weight_decay": 0.0,
        "optimizer": {"betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0,
                      "schedule": "linear", "warmup_ratio": 0.0, "min_lr_ratio": 0.0,
                      "micro_batch_size": 32},
    },
    "timesfm-2.5": {
        "source": "https://github.com/google-research/timesfm/blob/8cb0628371af142e16b8c232cc9fbf667ffb12f9/timesfm-forecasting/examples/finetuning/finetune_lora.py",
        "reference_scope": "Official LoRA example optimizer transferred to full parameters; not an official full-parameter recipe",
        "official_duration": "10 epochs on 5000 pre-sampled windows in retail example",
        "learning_rate": 1e-4, "batch_size": 32, "weight_decay": 0.01,
        "optimizer": {"betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0,
                      "schedule": "cosine", "warmup_ratio": 0.0, "min_lr_ratio": 0.0,
                      "micro_batch_size": 32},
    },
    "timemoe-200m": {
        "source": "https://github.com/Time-MoE/Time-MoE/blob/915bfda4c78a544d62a2bec6ab22948423059236/main.py",
        "reference_scope": "Official fine-tuning CLI defaults; native multi-horizon loss retained",
        "official_duration": "1 epoch in upstream data-loader units",
        "learning_rate": 1e-4, "batch_size": 64, "weight_decay": 0.1,
        "optimizer": {"betas": [0.9, 0.95], "eps": 1e-8, "max_grad_norm": 1.0,
                      "schedule": "cosine", "warmup_ratio": 0.0, "min_lr_ratio": 0.5,
                      "micro_batch_size": 16, "exclude_bias_norm": True},
    },
}


def recipe(model):
    return deepcopy(RECIPES[model])
