"""Explicit AdamW controls used by development hyperparameter probes."""
import math


def optimizer_options(value, *, batch_size, steps):
    defaults = dict(betas=[0.9, 0.999], eps=1e-8, max_grad_norm=None,
                    schedule="constant", warmup_ratio=0.0, min_lr_ratio=0.0,
                    schedule_steps=steps, micro_batch_size=batch_size, exclude_bias_norm=False)
    if set(value) - set(defaults):
        raise ValueError("Unknown optimizer options: %s" % sorted(set(value) - set(defaults)))
    result = {**defaults, **value}
    if type(result["exclude_bias_norm"]) is not bool:
        raise ValueError("exclude_bias_norm must be boolean")
    if len(result["betas"]) != 2 or any(not 0 <= b < 1 for b in result["betas"]):
        raise ValueError("Adam betas must lie in [0, 1)")
    if not math.isfinite(result["eps"]) or result["eps"] <= 0:
        raise ValueError("Adam eps must be positive")
    norm = result["max_grad_norm"]
    if norm is not None and (not math.isfinite(norm) or norm <= 0):
        raise ValueError("max_grad_norm must be positive or null")
    if result["schedule"] not in {"constant", "linear", "cosine"}:
        raise ValueError("schedule must be constant, linear or cosine")
    if not 0 <= result["warmup_ratio"] < 1 or not 0 <= result["min_lr_ratio"] <= 1:
        raise ValueError("Invalid warmup/minimum LR ratio")
    for key in ("schedule_steps", "micro_batch_size"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] <= 0:
            raise ValueError("%s must be a positive integer" % key)
    if result["schedule_steps"] < steps or result["micro_batch_size"] > batch_size:
        raise ValueError("schedule must cover the trial; micro batch cannot exceed effective batch")
    return result


def learning_rate_factor(step, options):
    """Zero-based update index; schedule horizon stays fixed during bisection."""
    warmup = int(options["schedule_steps"] * options["warmup_ratio"])
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, options["schedule_steps"] - warmup))
    if options["schedule"] == "constant":
        return 1.0
    decay = 1 - progress if options["schedule"] == "linear" else (1 + math.cos(math.pi * progress)) / 2
    return options["min_lr_ratio"] + (1 - options["min_lr_ratio"]) * decay
