"""Check one released foundation checkpoint and its TS-Lab training paths."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from adapters.foundation import build_adapter, get_model_spec
from configs.paths import TRAIN_OUTPUTS, output_path
from gradient.new_foundation import ChronosFamilyGradientEngine, Moirai2GradientEngine
from rage.experiments.lora import apply_student_lora
from rage.experiments.students import PARAMETER_COUNTS, adapter_engine


def log(stage: str, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{stamp}-[smoke_foundation_models]-[{stage}]-{message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(PARAMETER_COUNTS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.model not in {
        "chronos-2-small", "chronos-2", "chronos-bolt-tiny",
        "chronos-bolt-mini", "chronos-bolt-small", "moirai-2.0-small",
    }:
        parser.error("model is outside the new foundation batch")
    started = time.monotonic()
    torch.set_num_threads(2)
    out_dir = output_path("train", TRAIN_OUTPUTS / "v2-foundation-expansion" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".complete").unlink(missing_ok=True)
    spec = get_model_spec(args.model)
    context_length, prediction_length = 512, 64
    context = np.linspace(-1, 1, context_length, dtype=np.float32)[None]
    future = np.linspace(1, 1.5, prediction_length, dtype=np.float32)[None]
    windows = np.concatenate([context, future], axis=1)
    log("config", f"model={args.model} revision={spec.revision} device={args.device} context=512 horizon=64")
    log("load-start", "loading pinned checkpoint")
    adapter = build_adapter(args.model, device=args.device)
    model = adapter.load()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PARAMETER_COUNTS[args.model]:
        raise ValueError(f"parameter count mismatch: {count} versus {PARAMETER_COUNTS[args.model]}")
    log("load-end", f"parameters={count}")
    model.eval()
    forecast = adapter.forecast(context, prediction_length=prediction_length)
    if forecast.shape != (1, prediction_length) or not np.isfinite(forecast).all():
        raise ValueError("invalid point forecast")
    log("forecast-end", f"first={float(forecast[0, 0]):.6g}")
    native_loss = adapter.training_loss(
        windows, context_length=context_length, prediction_length=prediction_length,
    )
    if native_loss.ndim != 0 or not torch.isfinite(native_loss):
        raise ValueError("invalid native loss")
    log("loss-end", f"loss={float(native_loss):.8g}")
    engine = adapter_engine(adapter, context_length, prediction_length)
    lora = apply_student_lora(engine, args.model, {"r": 8, "alpha": 16, "dropout": 0.0})
    if lora is None or lora["trainable_parameters"] < 1:
        raise ValueError("LoRA trainable parameter set is empty")
    log("lora-end", f"trainable={lora['trainable_parameters']}")
    loss = engine.training_loss(windows)
    loss.backward()
    gradients = [p.grad for p in engine.model.parameters() if p.requires_grad]
    if not any(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) > 0 for g in gradients):
        raise ValueError("LoRA did not receive a finite nonzero gradient")
    engine.model.zero_grad(set_to_none=True)
    log("backward-end", f"loss={float(loss):.8g}")
    probe_type = Moirai2GradientEngine if args.model == "moirai-2.0-small" else ChronosFamilyGradientEngine
    probe = probe_type(engine, model_key=f"{spec.model_id}@{spec.revision}", projection_dimension=128)
    losses, sketches = probe.batch_gradient_sketch(windows, batch_size=1)
    if losses.shape != (1,) or sketches.shape != (1, 128) or not np.isfinite(sketches).all():
        raise ValueError("invalid projected gradient")
    if not np.any(sketches):
        raise ValueError("projected gradient is zero")
    if abs(float(losses[0]) - float(loss)) > 1e-3 * max(1.0, abs(float(loss))):
        raise ValueError("probe loss differs from training loss")
    log("gradient-end", f"loss={float(losses[0]):.8g} norm={float(np.linalg.norm(sketches[0])):.8g}")
    second = windows.copy()
    second[:, :context_length] *= 1.2
    second[:, context_length:] += 0.3
    pair = np.concatenate((windows, second), axis=0)
    pair_losses, pair_sketches = probe.batch_gradient_sketch(pair, batch_size=2)
    serial_losses, serial_sketches = probe.batch_gradient_sketch(pair, batch_size=1)
    if not np.allclose(pair_losses, serial_losses, rtol=1e-4, atol=1e-4):
        raise ValueError("batched gradient losses differ from serial losses")
    if not np.allclose(pair_sketches, serial_sketches, rtol=2e-3, atol=2e-3):
        raise ValueError("batched projected gradients differ from serial gradients")
    log("batch-gradient-end", f"max_error={float(np.max(np.abs(pair_sketches - serial_sketches))):.8g}")
    trainable = [p for p in engine.model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    engine.model.train()
    optimizer.zero_grad(set_to_none=True)
    update_loss = engine.training_loss(windows)
    update_loss.backward()
    optimizer.step()
    if not any(not torch.equal(prior, current.detach()) for prior, current in zip(before, trainable)):
        raise ValueError("optimizer step did not update any LoRA weights")
    updated_forecast = engine.forecast(context)
    if updated_forecast.shape != forecast.shape or not np.isfinite(updated_forecast).all():
        raise ValueError("LoRA checkpoint cannot forecast after an optimizer step")
    log("train-step-end", f"loss={float(update_loss):.8g}")
    result = {"model": args.model, "revision": spec.revision, "parameters": count,
              "lora_trainable": lora["trainable_parameters"],
              "native_loss": float(native_loss), "lora_loss": float(loss),
              "probe_loss": float(losses[0]), "gradient_norm": float(np.linalg.norm(sketches[0])),
              "forecast_first": float(forecast[0, 0]), "elapsed_seconds": time.monotonic() - started}
    (out_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / ".complete").write_text("complete\n")
    log("end", f"seconds={result['elapsed_seconds']:.1f} output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
