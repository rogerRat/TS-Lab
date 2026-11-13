"""Load the actual production student and its own frozen selection probe."""
from dataclasses import replace
import json

from .students import STUDENTS, HISTORICAL_STUDENTS, PARAMETER_COUNTS, adapter_engine
from gradient.timesfm import TimesFMGradientEngine
from gradient.timemoe import TimeMoEGradientEngine

CHRONOS_BASE_REVISION = "ad294eaacead15db499b740ea4122266dd2a81a2"


def _gradient_probe(student, engine, *, model_key, checkpoint):
    """Frozen per-student selection probe; new students use native batched VJP."""
    if student in ("chronos-2-small", "chronos-2", "chronos-bolt-tiny",
                   "chronos-bolt-mini", "chronos-bolt-small"):
        from gradient.new_foundation import ChronosFamilyGradientEngine
        return ChronosFamilyGradientEngine(engine, model_key=model_key)
    if student == "moirai-2.0-small":
        from gradient.new_foundation import Moirai2GradientEngine
        return Moirai2GradientEngine(engine, model_key=model_key)
    if student == "timesfm-2.5":
        return TimesFMGradientEngine(engine, model_key=model_key)
    if student == "timemoe-200m":
        return TimeMoEGradientEngine(engine, model_key=model_key, checkpoint_path=checkpoint)
    if student == "timerxl-84m":
        from gradient.timerxl import TimerXLGradientEngine
        return TimerXLGradientEngine(engine, model_key=model_key)
    if student == "sundial-base-128m":
        from gradient.sundial import SundialGradientEngine
        return SundialGradientEngine(engine, model_key=model_key)
    raise ValueError("no gradient probe registered for student: " + student)


def build_model(student, *, model_path=None, device="cuda", context_length=512, prediction_length=64):
    if student not in STUDENTS + HISTORICAL_STUDENTS:
        raise ValueError("Full-pool production requires a registered student: " + student)
    if student in ("chronos-tiny", "chronos-small", "chronos-base"):
        from gradient.chronos import (
            ChronosT5Config, ChronosT5Engine, DEFAULT_CHRONOS_T5_TINY,
            DEFAULT_CHRONOS_T5_SMALL, DEFAULT_CHRONOS_T5_BASE,
        )
        chronos = {
            "chronos-tiny": ("amazon/chronos-t5-tiny",
                             "29d808298f1a62493e7b9a5e08529d0d930fa189",
                             DEFAULT_CHRONOS_T5_TINY),
            "chronos-small": ("amazon/chronos-t5-small",
                              "a971ba21945c4f1796b17a91fe69214b5f4ad472",
                              DEFAULT_CHRONOS_T5_SMALL),
            "chronos-base": ("amazon/chronos-t5-base",
                             CHRONOS_BASE_REVISION, DEFAULT_CHRONOS_T5_BASE),
        }
        model_id, revision, default_path = chronos[student]
        engine = ChronosT5Engine(ChronosT5Config(
            model_path=model_path or default_path, context_length=context_length,
            prediction_length=prediction_length,
        ), device=device)
        probe = engine
    else:
        from adapters.foundation import build_adapter
        adapter = build_adapter(student, device=device)
        model_id, revision = adapter.spec.model_id, adapter.spec.revision
        if model_path:
            adapter.spec = replace(adapter.spec, model_id=model_path)
        engine = adapter_engine(adapter, context_length, prediction_length)
        probe = _gradient_probe(student, engine, model_key=f"{model_id}@{revision}",
                                checkpoint=model_path or model_id)
    import torch
    if student not in ("sundial-base-128m", "moirai-2.0-small"):
        # Moirai 2.0 and Sundial retain the released FP32 runtime path.
        engine.model.to(dtype=torch.bfloat16)
    parameters = sum(p.numel() for p in engine.model.parameters())
    if parameters != PARAMETER_COUNTS[student]:
        raise ValueError(f"{student} weight identity mismatch: loaded {parameters} parameters")
    # Hugging Face configs include integer-keyed id2label maps and tuples.
    # Canonicalize once so the live identity equals its persisted JSON form.
    model_config = getattr(engine.model, "config", None)
    if model_config is not None and hasattr(model_config, "to_dict"):
        config = json.loads(json.dumps(model_config.to_dict()))
    else:
        module = engine.model
        config = {"module": type(module).__name__,
                  "patch_size": getattr(module, "patch_size", None),
                  "max_seq_len": getattr(module, "max_seq_len", None),
                  "num_predict_token": getattr(module, "num_predict_token", None),
                  "quantile_levels": list(getattr(module, "quantile_levels", ())) }
    identity = dict(student=student, model_id=model_id, revision=revision,
                    ckpt=model_path or model_id, ckpt_revision=revision,
                    loaded_path=model_path or getattr(probe.config, "model_path", model_id),
                    parameter_count=parameters,
                    trainable_parameter_count=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
                    model_class=type(engine.model).__name__, config=config,
                    loaded_parameter_dtypes=sorted({str(p.dtype) for p in engine.model.parameters()}),
                    projection_dimension=probe.config.projection_dimension,
                    projection_seed=probe.config.projection_seed)
    routed = sum(p.numel() for name, p in engine.model.named_parameters() if ".experts." in name)
    identity["active_parameters_per_token"] = (parameters - routed +
        routed * config["num_experts_per_tok"] // config["num_experts"] if routed else parameters)
    identity["active_parameter_definition"] = "dense/shared parameters and all output heads plus top-k routed experts per token"
    from .precision import POLICY
    identity["precision_policy"] = POLICY
    parameter_dtype = str(next(engine.model.parameters()).dtype)
    identity["training_parameter_dtype"] = parameter_dtype
    identity["scoring_parameter_dtype"] = parameter_dtype
    return engine, probe, identity
