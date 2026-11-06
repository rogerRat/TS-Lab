"""Shared trajectory adapters for language-side baselines (decisions 11/12)."""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
from rage.crb import CRBConfig
from rage.execution import execution_policy
from .common import _pool_fingerprint

REVISION = "trajectory-0912-repeat-v1"


def budget_config(index, cover_density=.02, error_density=.02, split_density=.005):
    return CRBConfig(context_length=index.context_length,
                     prediction_length=index.prediction_length,
                     window_stride=index.stride, cover_density=cover_density,
                     error_density=error_density, split_density=split_density)


def uniform_positions(size, config):
    """Spread the rounded CRB budget ceiling uniformly, including endpoints.

    All CRB phase budgets are capped to legal-start capacity and exhausted.
    """
    count = sum(config.budgets(size))
    return np.rint(np.linspace(0, size - 1, count)).astype(np.int64)


def trajectory_features(engine, index, config, *, kind, cache_dir=None,
                        pool_manifest=None, gradient_assets=None, dataset="source"):
    """Batch across trajectories; average raw FP32 features BEFORE normalization.

    Cache identity uses metadata, never scans/hashes model weights or data.
    """
    fids = list(index.fragment_lengths)
    counts = [sum(config.budgets(index.fragment_lengths[fid])) for fid in fids]
    if kind == "less" and gradient_assets is not None:
        from assets.gradient import index_request
        positions = trajectory_positions(index, config)
        owners = np.repeat(np.arange(len(fids)), counts)
        features = None
        for begin in range(0, len(positions), 1024):
            values = gradient_assets.cache.get(index_request(dataset, index, positions[begin:begin + 1024]))
            if features is None:
                features = np.zeros((len(fids), values.shape[-1]), dtype=np.float32)
            np.add.at(features, owners[begin:begin + len(values)], values)
        features /= np.asarray(counts, dtype=np.float32)[:, None]
        return features, dict(pool_cached=True, executed_windows=0, sampled_windows=len(positions),
            sample_counts=counts, trajectory_ids=fids, assets_path=str(gradient_assets.cache.path),
            sampling="uniform-endpoints; sum of rounded CRB budget ceilings")
    if kind == "less":
        raise ValueError("LESS requires prepared gradients; additional BP is forbidden")
    probe = getattr(engine, "config", None)
    identity = {"revision": REVISION, "pool": _pool_fingerprint(pool_manifest),
                "trajectories": [[fid, index.fragment_lengths[fid]] for fid in fids],
                "protocol": [config.context_length, config.prediction_length, config.window_stride],
                "density": [config.cover_density, config.error_density, config.split_density],
                "probe": {k: str(getattr(probe, k, None)) for k in
                          ("model_path", "projection_dimension", "projection_seed")}}
    cache = None
    if cache_dir is not None and pool_manifest is not None:
        from assets.gradient.cache import readable, atomic_json
        cache = Path(cache_dir) / (readable(kind)+"-trajectory-features.npy")
        manifest_path=cache.with_suffix('.json')
        if cache.exists() and (not manifest_path.exists() or json.loads(manifest_path.read_text())!=identity):
            raise ValueError('Trajectory feature cache fields differ; use a new cache directory')
        cache.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(manifest_path,identity)
    info = {"pool_cached": bool(cache is not None and cache.exists()),
            "sampled_windows": int(sum(counts)), "sample_counts": counts,
            "trajectory_ids": fids, "budget_density": identity["density"],
            "sampling": "uniform-endpoints; sum of rounded CRB budget ceilings"}
    if info["pool_cached"]:
        info["executed_windows"] = 0
        return np.load(cache, mmap_mode="r"), info
    positions = np.concatenate([index.bounds(fid)[0] + uniform_positions(
        index.fragment_lengths[fid], config) for fid in fids])
    owners = np.repeat(np.arange(len(fids)), counts)
    mode = "gradient" if kind == "less" else "encoder"
    batch = execution_policy(engine).batch_size(engine, mode, fallback=32, count=len(positions))
    features = None
    started = time.monotonic()
    next_log = 0
    for begin in range(0, len(positions), batch):
        windows = index.windows_at(positions[begin:begin + batch])
        if kind == "less":
            _, values = engine.batch_gradient_sketch(windows)
        else:
            from rage.execution import encoder_embeddings
            values = encoder_embeddings(engine, windows, batch_size=batch)
        values = np.asarray(values, dtype=np.float32)
        if features is None:
            features = np.zeros((len(fids), values.shape[-1]), dtype=np.float32)
        np.add.at(features, owners[begin:begin + batch], values)
        done = min(begin + batch, len(positions))
        if done >= next_log or done == len(positions):
            print(f"  [{kind} trajectory features] {done}/{len(positions)} windows "
                  f"({time.monotonic() - started:.0f}s)", flush=True)
            next_log = done + 50000
    features /= np.asarray(counts, dtype=np.float32)[:, None]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary=cache.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:np.save(stream,features)
        temporary.replace(cache)
    info["executed_windows"] = len(positions)
    return features, info


def trajectory_positions(index, config):
    """Method requirement expressed as source/reference index positions."""
    return np.concatenate([index.bounds(fid)[0] + uniform_positions(
        index.fragment_lengths[fid], config) for fid in index.fragment_lengths])
