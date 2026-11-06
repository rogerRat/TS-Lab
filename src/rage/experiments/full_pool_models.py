"""Load the actual production student and its own frozen selection probe."""
from dataclasses import replace
import json

from .students import STUDENTS, PARAMETER_COUNTS, adapter_engine
from gradient.timesfm import TimesFMGradientEngine
from gradient.timemoe import TimeMoEGradientEngine

CHRONOS_BASE_REVISION = "ad294eaacead15db499b740ea4122266dd2a81a2"


def _gradient_probe(student, engine, *, model_key, checkpoint):
    """Frozen per-student selection probe; new students use native batched VJP."""
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
    if student not in STUDENTS:
        raise ValueError("Full-pool production requires a registered student: " + student)
    if student == "chronos-base":
        from gradient.chronos import ChronosT5Config, ChronosT5Engine, DEFAULT_CHRONOS_T5_BASE
        engine = ChronosT5Engine(ChronosT5Config(model_path=model_path or DEFAULT_CHRONOS_T5_BASE,
            context_length=context_length, prediction_length=prediction_length), device=device)
        probe = engine
        model_id, revision = "amazon/chronos-t5-base", CHRONOS_BASE_REVISION
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
    if student != "sundial-base-128m":
        # Sundial keeps its native float32 checkpoint precision (bf16 breaks
        # its patch-embedding mask concat); other students follow BF16 policy.
        engine.model.to(dtype=torch.bfloat16)
    parameters = sum(p.numel() for p in engine.model.parameters())
    if parameters != PARAMETER_COUNTS[student]:
        raise ValueError(f"{student} weight identity mismatch: loaded {parameters} parameters")
    # Hugging Face configs include integer-keyed id2label maps and tuples.
    # Canonicalize once so the live identity equals its persisted JSON form.
    config = json.loads(json.dumps(engine.model.config.to_dict()))
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
