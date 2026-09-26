"""Top-layer CLI for inspecting or smoke-testing foundation-model adapters."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from adapters.foundation import build_adapter, list_model_specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect revision-pinned TS3 model adapters")
    parser.add_argument("--model", choices=[spec.key for spec in list_model_specs()])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-forecast", action="store_true")
    parser.add_argument("--smoke-loss", action="store_true")
    parser.add_argument("--smoke-gradient", action="store_true")
    parser.add_argument("--projection-dimension", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    args = parser.parse_args(argv)
    if args.model is None:
        print(json.dumps([spec.as_mapping() for spec in list_model_specs()], indent=2))
        return 0
    adapter = build_adapter(args.model, device=args.device)
    payload = adapter.describe()
    if args.smoke_forecast:
        context = np.sin(np.linspace(0.0, 24.0, args.context_length, dtype=np.float32))[None]
        forecast = adapter.forecast(context, prediction_length=args.prediction_length)
        payload["smoke_forecast_shape"] = list(forecast.shape)
        payload["smoke_forecast_finite"] = bool(np.isfinite(forecast).all())
    if args.smoke_loss:
        total = args.context_length + args.prediction_length
        window = np.sin(np.linspace(0.0, 24.0, total, dtype=np.float32))[None]
        loss = adapter.training_loss(
            window,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
        )
        payload["smoke_loss"] = float(loss.detach().float().cpu())
        payload["smoke_loss_finite"] = bool(np.isfinite(payload["smoke_loss"]))
    if args.smoke_gradient:
        from gradient.foundation import FoundationGradientConfig, FoundationGradientEngine
        engine = FoundationGradientEngine(
            FoundationGradientConfig(
                model_key=args.model,
                context_length=args.context_length,
                prediction_length=args.prediction_length,
                projection_dimension=args.projection_dimension,
            ),
            device=args.device,
            adapter=adapter,
        )
        total = args.context_length + args.prediction_length
        window = np.sin(np.linspace(0.0, 24.0, total, dtype=np.float32))[None]
        losses, sketches = engine.batch_gradient_sketch(window)
        payload["smoke_gradient_layer"] = engine.layer_name
        payload["smoke_gradient_loss"] = float(losses[0])
        payload["smoke_gradient_shape"] = list(sketches.shape)
        payload["smoke_gradient_norm"] = float(np.linalg.norm(sketches[0]))
        payload["smoke_gradient_finite"] = bool(np.isfinite(sketches).all())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
