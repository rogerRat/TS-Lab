"""Field-validated, single-writer source CRB checkpoints, one NPZ per trajectory.

Lives under the locked gradient cache. No pickle, hashes, or new gradient work.
Stores raw representatives as well as directions; they are not interchangeable.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from rage.crb import CRBResult, GradientBlock
from .cache import atomic_json
from .variants import TrajectoryGradient, aggregate_anchors


def source_cache_directory(cache, config, pool, index):
    fields = dict(format="source-crb-trajectories-v1", gradient=cache.manifest,
                  crb=asdict(config), context_length=index.context_length,
                  prediction_length=index.prediction_length, stride=index.stride,
                  fragments=[dict(fragment_id=int(f.fragment_id), entity_id=str(f.entity_id),
                                  length=int(f.length), offset=(int(index.bounds(f.fragment_id)[0])
                                      if f.fragment_id in index.fragment_lengths else None))
                             for f in pool])
    # JSON normalization matches GradientCache's manifest convention.
    fields = json.loads(json.dumps(fields))
    root = cache.path / "source-crb"
    root.mkdir(exist_ok=True)
    version = 0
    while True:
        path = root / f"fields-{version}"
        marker = path / "cache_manifest.json"
        if not path.exists():
            path.mkdir()
            atomic_json(marker, fields)
            return path
        if marker.exists() and json.loads(marker.read_text()) == fields:
            return path
        version += 1


def save_source_result(path, result, aggregate):
    blocks = result.blocks
    offsets = np.cumsum([0] + [len(b.start_positions) for b in blocks])
    dimension = len(next(iter(result.anchors.values()))) if result.anchors else 0
    meta = {key: getattr(result, key) for key in (
        "fragment_id", "discarded_no_signal", "discarded_antagonistic", "banned_budget",
        "cover_computations", "error_computations", "split_computations",
        "epsilon_audit", "split_audit")}
    meta["has_aggregate"] = aggregate is not None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, meta=json.dumps(meta), valid_starts=result.valid_starts,
                bounds=np.array([(b.left_index, b.right_index) for b in blocks], dtype=np.int64).reshape(-1, 2),
                offsets=offsets,
                positions=np.concatenate([b.start_positions for b in blocks]) if blocks else np.empty(0, dtype=np.int64),
                means=np.stack([b.mean_gradient for b in blocks]) if blocks else np.empty((0, dimension), dtype=np.float32),
                directions=np.stack([b.direction for b in blocks]) if blocks else np.empty((0, dimension), dtype=np.float32),
                aggregate_positions=aggregate.start_positions if aggregate else np.empty(0, dtype=np.int64),
                aggregate_mean=aggregate.mean_gradient if aggregate else np.empty(0, dtype=np.float32))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_source_result(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        has_aggregate = meta.pop("has_aggregate")
        positions, offsets = data["positions"], data["offsets"]
        means, directions, bounds = data["means"], data["directions"], data["bounds"]
        blocks = [GradientBlock(meta["fragment_id"], int(left), int(right),
                               positions[offsets[i]:offsets[i + 1]], means[i], directions[i])
                  for i, (left, right) in enumerate(bounds)]
        result = CRBResult(valid_starts=data["valid_starts"], anchors={}, blocks=blocks, **meta)
        aggregate = (TrajectoryGradient(meta["fragment_id"], data["aggregate_positions"],
                                        data["aggregate_mean"]) if has_aggregate else None)
    return result, aggregate


def source_results(engine, group, config, cache_dir):
    """Only missing trajectories enter CRB; cached rows never request gradients."""
    paths = [cache_dir / f"trajectory-{int(f.fragment_id)}.npz" for f in group]
    missing = [f for f, path in zip(group, paths) if not path.exists()]
    computed = iter(engine.process_fragments(missing)) if missing else iter(())
    for fragment, path in zip(group, paths):
        if path.exists():
            result, aggregate = load_source_result(path)
        else:
            result = next(computed)
            aggregate = aggregate_anchors(result,
                expected_count=sum(config.budgets(len(result.valid_starts))))
            save_source_result(path, result, aggregate)
        yield fragment, result, aggregate
