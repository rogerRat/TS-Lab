"""Top-layer CLI for static W2048/S2048 TimesFM valuation selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from gradient.foundation import FoundationGradientConfig, FoundationGradientEngine
from rags.experiments.foundation_selection import select_long_context_candidates


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="npz with source/reference windows, IDs, and group labels")
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregation", choices=["window", "segment"], required=True)
    parser.add_argument("--model", default="timesfm-2.5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--projection-dimension", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-select", type=int, default=4096)
    parser.add_argument("--deployment-window-length", type=int, default=576)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite existing output: %s" % output)
    source = np.load(args.input, allow_pickle=False)
    required = {
        "source_windows", "source_ids", "source_groups",
        "reference_windows", "reference_groups",
    }
    missing = required - set(source.files)
    if missing:
        raise SystemExit("input archive lacks: %s" % ", ".join(sorted(missing)))
    engine = FoundationGradientEngine(
        FoundationGradientConfig(
            model_key=args.model,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            projection_dimension=args.projection_dimension,
        ),
        device=args.device,
    )
    result = select_long_context_candidates(
        engine,
        source_windows=source["source_windows"], source_ids=source["source_ids"],
        source_groups=source["source_groups"], reference_windows=source["reference_windows"],
        reference_groups=source["reference_groups"],
        n_select=args.n_select, deployment_window_length=args.deployment_window_length,
        aggregation=args.aggregation, batch_size=args.batch_size, seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, selected_rows=result.selected_rows, selected_ids=result.selected_ids,
        event_scores=result.event_scores,
        protocol=json.dumps({
            "model": engine.adapter.spec.as_mapping(),
            "gradient_layer": engine.layer_name,
            "projection_seed": engine.config.projection_seed,
            "input": str(Path(args.input).resolve()),
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "projection_dimension": args.projection_dimension,
            "aggregation": result.aggregation,
            "deployment_window_length": args.deployment_window_length,
            "seed": args.seed, **result.diagnostics,
        }),
    )
    print(json.dumps(result.diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
