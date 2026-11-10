"""Optional PEFT LoRA setup for supported forecasting students."""
from __future__ import annotations


_STUDENT_TARGETS = {
    "chronos-tiny": ("q", "v"),
    "chronos-small": ("q", "v"),
    "chronos-base": ("q", "v"),
    "timesfm-2.5": ("q_proj", "v_proj"),
    "timerxl-84m": ("q_proj", "v_proj"),
}


def apply_student_lora(engine, student: str, config: dict | None) -> dict | None:
    """Wrap the student model with attention LoRA when the manifest enables it."""
    if config is None or config is False:
        return None
    if student not in _STUDENT_TARGETS:
        raise ValueError("LoRA is unsupported for student: " + student)
    if not isinstance(config, dict):
        raise TypeError("training.lora must be an object or null")
    allowed = {"r", "alpha", "dropout", "target_modules"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("unknown LoRA options: " + ", ".join(sorted(unknown)))

    rank = config.get("r", 8)
    alpha = config.get("alpha", 16)
    dropout = config.get("dropout", 0.0)
    raw_targets = config.get("target_modules", _STUDENT_TARGETS[student])
    if not isinstance(raw_targets, (list, tuple)):
        raise TypeError("LoRA target_modules must be a list of module suffixes")
    targets = tuple(raw_targets)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA r must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha <= 0:
        raise ValueError("LoRA alpha must be a positive integer")
    if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if not targets or any(not isinstance(name, str) or not name for name in targets):
        raise ValueError("LoRA target_modules must be a non-empty string list")

    module_names = tuple(name for name, _module in engine.model.named_modules())
    missing = [target for target in targets
               if not any(name == target or name.endswith("." + target)
                          for name in module_names)]
    if missing:
        raise ValueError("LoRA targets not found in " + student + ": " + ", ".join(missing))

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install the model profile with PEFT to enable LoRA") from exc

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(dropout),
        target_modules=list(targets),
        bias="none",
        task_type=None,
        inference_mode=False,
    )
    wrapped = get_peft_model(engine.model, lora_config)
    base_dtype = next(wrapped.parameters()).dtype
    wrapped.to(dtype=base_dtype)
    engine.model = wrapped
    adapter = getattr(engine, "adapter", None)
    if adapter is not None:
        adapter._model = wrapped
    summary = {
        "method": "peft_lora",
        "r": rank,
        "alpha": alpha,
        "dropout": float(dropout),
        "target_modules": list(targets),
        "trainable_parameters": sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in wrapped.parameters()),
        "parameter_dtype": str(base_dtype),
        "embedding_and_output_heads_adapted": False,
    }
    engine.training_adapter_config = summary
    return summary
