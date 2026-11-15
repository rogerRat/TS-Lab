"""Select source windows with uniform, ADS or CRB scoring."""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parents[2])
for _source_path in (_PROJECT_ROOT, str(_Path(_PROJECT_ROOT) / "src")):
    if _source_path not in sys.path:
        sys.path.insert(0, _source_path)


import argparse
import json
from pathlib import Path
import time

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from scripts.tools._common import apply_overrides, dispatch_model_environment, load_config, required, resolve


def _load_fragments(path: Path, *, id_key: str, entity_key: str):
    from rage.core import TimeSeriesFragment

    with np.load(path, allow_pickle=False) as archive:
        required_fields = {"values", "offsets"}
        missing = required_fields - set(archive.files)
        if missing:
            raise ValueError("fragment archive lacks: " + ", ".join(sorted(missing)))
        values = np.asarray(archive["values"], dtype=np.float32)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        ids = (np.asarray(archive[id_key]) if id_key in archive else
               np.arange(len(offsets) - 1, dtype=np.int64))
        entities = (np.asarray(archive[entity_key]) if entity_key in archive else ids)
    if values.ndim != 1 or offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("fragment values and offsets have invalid dimensions")
    if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) <= 0):
        raise ValueError("fragment offsets must partition the values array")
    if len(ids) != len(offsets) - 1 or len(entities) != len(ids):
        raise ValueError("fragment IDs and entities must align with offsets")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("fragment IDs must be unique within an archive")
    fragments = []
    for index, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
        fragments.append(TimeSeriesFragment(
            values[int(left):int(right)], entity_id=str(entities[index]),
            fragment_id=int(ids[index])))
    return fragments


def _gradient_cache(config, engine, identity, source_path, reference_path):
    from assets.gradient import GradientCache

    cache_root = output_path("grad_cache", config.get(
        "gradient_cache_dir", Path("outputs/grad_cache") / str(config["run_id"])))
    cache = GradientCache(
        cache_root, engine, model_identity=identity,
        data_fields={
            "source_archive": str(source_path.resolve()),
            "reference_archive": str(reference_path.resolve()),
            "source_id": str(config["source_id"]),
            "reference_id": str(config["reference_id"]),
        },
        context_length=int(config["context_length"]),
        prediction_length=int(config["prediction_length"]),
    )
    return cache


def run(config: dict, method: str) -> Path:
    from rage.experiments.arms import (
        WindowIndex, streaming_random_select, ads_select, rage_crb_select,
    )
    from rage.experiments.regions import save_selection
    from rage.ads import ADSConfig
    from rage.execution import selection_execution
    from rage.experiments.full_pool_models import build_model
    from assets.gradient.cache import FragmentGradients

    model = str(required(config, "model"))
    source_path = resolve(required(config, "source_archive"))
    reference_path = resolve(required(config, "reference_archive"))
    source = _load_fragments(source_path,
        id_key=str(config.get("fragment_id_key", "fragment_ids")),
        entity_key=str(config.get("entity_id_key", "entity_ids")))
    reference = _load_fragments(reference_path,
        id_key=str(config.get("fragment_id_key", "fragment_ids")),
        entity_key=str(config.get("entity_id_key", "entity_ids")))
    context = int(required(config, "context_length"))
    horizon = int(required(config, "prediction_length"))
    stride = int(config.get("window_stride", 1))
    selection = config.get("selection", {})
    count = int(required(selection, "n_select"))
    run_id = str(required(config, "run_id"))
    destination = output_path("selection", required(config, "output_dir"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("selection output is not empty: %s" % destination)
    destination.mkdir(parents=True, exist_ok=True)
    logger = configure_logger("tsl.tools.select", "tools/select", run_id, "SELECT")
    started = time.monotonic()
    engine, probe, identity = build_model(
        model, model_path=config.get("model_path"), device=str(config.get("device", "cuda")),
        context_length=context, prediction_length=horizon)
    index = WindowIndex(source, context_length=context, prediction_length=horizon,
                        stride=stride)
    cache = _gradient_cache(config, probe, identity, source_path, reference_path)
    source_gradients = FragmentGradients(cache, str(config["source_id"]))
    reference_gradients = FragmentGradients(cache, str(config["reference_id"]))
    owner_tags = {id(fragment): source_gradients for fragment in source}
    owner_tags.update({id(fragment): reference_gradients for fragment in reference})

    def gradient_computer(fragment, starts):
        try:
            reader = owner_tags[id(fragment)]
        except KeyError as exc:
            raise ValueError("selector requested a gradient for an unregistered fragment") from exc
        return reader(fragment, starts)

    matrix_path = destination / "score_matrix.npz"
    logger.info("stage=start method=%s model=%s source_fragments=%d reference_fragments=%d candidates=%d n_select=%d",
                method, model, len(source), len(reference), len(index), count)
    try:
        with selection_execution(probe):
            if method == "uni":
                result = streaming_random_select(
                    index, n_select=count, seed=int(selection.get("seed", 0)))
                score_record = {"kind": "not_applicable", "reason": "uniform selection has no score matrix"}
            elif method == "ads":
                ads_values = dict(selection.get("ads", {}))
                ads_config = ADSConfig(**ads_values)
                result = ads_select(
                    reference, source, index, gradient_computer=gradient_computer,
                    ads_config=ads_config, n_select=count,
                    seed=int(selection.get("seed", 0)),
                    sample_with_replacement=bool(selection.get("replacement", False)),
                    gradient_batch_size=int(selection.get("gradient_batch_size", 128)))
                score_record = {"kind": "adaptive_score_matrix_summaries",
                                "summaries": result.diagnostics.get("score_matrix_summaries", [])}
            else:
                result = rage_crb_select(
                    reference, source, index, gradient_computer=gradient_computer,
                    context_length=context, prediction_length=horizon,
                    window_stride=stride,
                    cover_density=float(selection.get("cover_density", 0.02)),
                    error_density=float(selection.get("error_density", 0.02)),
                    split_density=float(selection.get("split_density", 0.005)),
                    block_budget=int(selection.get("block_budget", 8)),
                    n_select=count,
                    mean_epsilon=float(selection.get("mean_epsilon", 1e-8)),
                    endpoint_epsilon=float(selection.get("endpoint_epsilon", 1e-8)),
                    tie_seed=int(selection.get("seed", 0)),
                    curvature_cap=float(selection.get("curvature_cap", 1.0)),
                    pool_cache_dir=destination / "cache",
                    pool_manifest={"source": str(source_path), "fragments": len(source)},
                    score_matrix_path=matrix_path)
                with np.load(matrix_path, allow_pickle=False) as matrix:
                    matrix_shape = list(matrix["scores"].shape)
                score_record = {
                    "kind": "reference_by_top_source_block_cosine",
                    "path": str(matrix_path),
                    "shape": matrix_shape,
                }
        selection_path = destination / "selection.npz"
        save_selection(selection_path, result)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "method": method,
            "model": identity,
            "source_archive": str(source_path),
            "reference_archive": str(reference_path),
            "source_id": str(config["source_id"]),
            "reference_id": str(config["reference_id"]),
            "context_length": context,
            "prediction_length": horizon,
            "window_stride": stride,
            "selected_count": len(result.selected),
            "score_matrix": score_record,
            "selection_file": str(selection_path),
            "gradient_cache": str(cache.path),
            "diagnostics": result.diagnostics,
            "cost": vars(result.cost),
        }
        temporary = destination / "manifest.json.partial"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="ascii")
        temporary.replace(destination / "manifest.json")
        (destination / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=complete elapsed=%s output=%s",
                    format_duration(time.monotonic() - started), destination)
    finally:
        cache.close()
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("uni", "ads", "crb"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    method = args.method or str(required(config, "method"))
    if method not in ("uni", "ads", "crb"):
        raise ValueError("method must be one of uni, ads, crb")
    model = str(required(config, "model"))
    dispatched = dispatch_model_environment(
        model, [str(Path(__file__).resolve()), "--config", str(resolve(args.config)),
                "--method", method, *sum((["--set", value] for value in args.set), [])])
    if dispatched is not None:
        return dispatched
    run(config, method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
