"""Prepare native TSDS OT and LP-Mirror value resampling distributions."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/exp/main-crb-v0"))
import run_directional_time1 as base
from prepare_eight_task import TASKS, load_task_reference
from configs.logger import configure_logger, format_duration
from rage.experiments.arms import WindowIndex
from rage.experiments.resampling import ResampledWindows
from rage.experiments.full_pool_models import build_model
from rage.execution import encoder_embeddings
from baselines.tsds.retrieval import trajectory_neighbours
from baselines.tsds.algorithms import tsds_probability_assignment
from baselines.lp_mirror.runner import lp_mirror_select
BATCH = "q16-src256-eight-task-resample"
OLD = ROOT / "outputs/selection/main-crb-v0/q16-src256-time1-q8/uni/source_anchors.npz"
ASSET_ROOT = ROOT / "outputs/selection/main-crb-v0" / BATCH / "native_assets"
REF_ROOT = ROOT / "outputs/selection/main-crb-v0/q16-src256-eight-task"
REF_CACHE = ROOT / "outputs/grad_cache/main-crb-v0/q16-src256-eight-task/components"
OUT = ROOT / "outputs/selection/main-crb-v0" / BATCH
TRAIN_ROOT = ROOT / "outputs/train/main-crb-v0" / BATCH
TASK_CODES = tuple(task["code"] for task in TASKS)

def save_payload(method, task, probabilities, metadata, layout="window", offsets=None):
    root = OUT / method / task
    root.mkdir(parents=True, exist_ok=True)
    arrays = {"probabilities": np.asarray(probabilities, dtype=np.float64),
              "unit_layout": np.asarray(layout),
              "resampling_metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    if offsets is not None: arrays["unit_offsets"] = np.asarray(offsets, dtype=np.int64)
    path = root / "resampling_payload.npz"
    np.savez_compressed(path, **arrays)
    (root / ".complete").write_text("complete\n", encoding="ascii")
    return path

def pool_index(logger):
    rows, fragments, _values, _offsets, counts = base.load_source_fragments()
    index = WindowIndex(fragments, context_length=base.CONTEXT, prediction_length=base.HORIZON, stride=1)
    if len(index) != 6382020:
        raise ValueError("src-256 legal window index differs")
    logger.info("stage=source-index-ready trajectories=%d windows=%d", len(fragments), len(index))
    pool_manifest = {"seed": 0, "equivalents_target": len(index),
        "total_points": int(sum(fragment.length for fragment in fragments)),
        "total_equivalents": len(index), "window_length": base.WINDOW}
    return rows, fragments, np.asarray(counts, dtype=np.int64), index, pool_manifest

def anchor_positions(archive_path, legal_counts, q):
    with np.load(archive_path, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"], dtype=np.int64)
    counts = np.asarray([len(base.exact_uniform_positions(int(n), q)) for n in legal_counts], dtype=np.int64)
    if int(counts.sum()) != len(positions):
        raise ValueError("stored anchor positions differ from registered q")
    return positions, counts

def encode_positions(engine, index, positions, logger, label, output_path=None):
    output = None
    cursor = 0
    started = time.monotonic()
    for first in range(0, len(positions), 128):
        stop = min(first + 128, len(positions))
        windows = index.windows_at(positions[first:stop])
        batch_features = np.asarray(encoder_embeddings(engine, windows, batch_size=64), dtype=np.float32)
        if output is None:
            if output_path is None:
                output = np.empty((len(positions), batch_features.shape[1]), dtype=np.float32)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32,
                                                   shape=(len(positions), batch_features.shape[1]))
        output[first:stop] = batch_features
        cursor = stop
        if cursor == len(positions) or cursor % 8192 < 128:
            elapsed = time.monotonic() - started
            eta = elapsed * (len(positions) - cursor) / max(cursor, 1)
            logger.info("stage=encoder-progress label=%s windows=%d/%d elapsed=%s eta=%s",
                label, cursor, len(positions), format_duration(elapsed), format_duration(eta))
    if isinstance(output, np.memmap): output.flush()
    return output

def mean_rows_by_unit(features, counts):
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    means = np.stack([np.asarray(features[offsets[i]:offsets[i + 1]], dtype=np.float64).mean(axis=0)
                      for i in range(len(counts))])
    norms = np.linalg.norm(means, axis=1)
    return np.divide(means, norms[:, None], out=np.zeros_like(means), where=norms[:, None] > 0)

def prepare_tsds(engine, rows, source_fragments, source_counts, index, pool_manifest, logger):
    positions, counts = anchor_positions(OLD, source_counts, base.SOURCE_Q)
    global_positions = np.concatenate([index.bounds(fid)[0] + positions[
        int(np.sum(counts[:i])):int(np.sum(counts[:i + 1]))] for i, fid in enumerate(index.fragment_lengths)])
    source_cache = ASSET_ROOT / "tsds-source-q16-embeddings.npy"
    source_meta = ASSET_ROOT / "tsds-source-q16-embeddings.json"
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"model": str(base.BASE_MODEL), "source": "src-256", "anchors": "UNI q16", "windows": len(global_positions)}
    if source_cache.is_file() and source_meta.is_file():
        saved = json.loads(source_meta.read_text(encoding="ascii"))
        if saved != identity: raise ValueError("TSDS source feature cache identity differs")
        source_features = np.load(source_cache, mmap_mode="r", allow_pickle=False)
        logger.info("stage=tsds-source-cache-hit rows=%d dimensions=%d", *source_features.shape)
    else:
        source_features = encode_positions(engine, index, global_positions, logger, "tsds-source-q16", source_cache)
        source_meta.write_text(json.dumps(identity, indent=2) + "\n", encoding="ascii")
    source_vectors = mean_rows_by_unit(source_features, counts)
    for task in TASKS:
        task_dir = REF_ROOT / task["code"] / "reference"
        ref_manifest, ref_fragments, ref_counts, _ = load_task_reference(task)
        ref_positions, ref_anchor_counts = anchor_positions(task_dir / "reference_anchors.npz", ref_counts, base.REFERENCE_Q)
        ref_index = WindowIndex(ref_fragments, context_length=base.CONTEXT, prediction_length=base.HORIZON, stride=1)
        ref_global_positions = np.concatenate([ref_index.bounds(fragment.fragment_id)[0] + ref_positions[
            int(np.sum(ref_anchor_counts[:i])):int(np.sum(ref_anchor_counts[:i + 1]))]
            for i, fragment in enumerate(ref_fragments)])
        ref_features = encode_positions(engine, ref_index, ref_global_positions, logger,
                                        "tsds-ref-q8-" + task["code"])
        ref_vectors = mean_rows_by_unit(ref_features, ref_anchor_counts)
        distances, neighbours, kdes, retrieval = trajectory_neighbours(
            source_vectors, ref_vectors, max_k=64, kde_k=16, sigma=4.0)
        probabilities = tsds_probability_assignment(distances, neighbours, kdes,
            candidate_count=len(source_vectors), alpha=0.1, c_param=5.0)
        offsets = np.r_[0, np.cumsum(source_counts, dtype=np.int64)]
        payload = save_payload("tsds", task["code"], probabilities,
            {"method": "TSDS", "scoring_unit": "trajectory", "native_kernel": "IVF-KDE-OT",
             "source_features": "TimerXL encoder mean at UNI q16 anchor windows",
             "reference_features": "TimerXL encoder mean at task q8 anchor windows",
             "max_k": 64, "effective_max_k": retrieval["max_k"], "kde_k": 16,
             "sigma": 4.0, "alpha": 0.1, "c": 5.0, "nprobe": retrieval["nprobe"],
             "nlist": retrieval["nlist"], "hard_fraction_cutoff": False,
             "window_draw": "uniform legal window within sampled trajectory", "seed": 0},
            layout="trajectory", offsets=offsets)
        logger.info("stage=tsds-distribution-complete task=%s reference_windows=%d anchor_count=%d retrieval=%s payload=%s",
            task["code"], ref_manifest["legal_windows"], len(ref_positions), retrieval, payload)

def prepare_lp(engine, source_fragments, source_counts, index, pool_manifest, logger):
    cache_dir = OUT / "native_assets/lp-mirror-proxy"
    lp_pool_manifest = dict(pool_manifest)
    lp_pool_manifest["total_equivalents"] = sum(fragment.length // base.WINDOW for fragment in source_fragments)
    for task in TASKS:
        _manifest, ref_fragments, _counts, _positions = load_task_reference(task)
        result = lp_mirror_select(engine, reference_fragments=ref_fragments, index=index,
            context_length=base.CONTEXT, prediction_length=base.HORIZON,
            proxy_epochs=1, proxy_batch=4096, proxy_lr=0.01,
            adaptation_epochs=1, adapt_batch=256, adapt_lr=0.001,
            reference_stride=8, reference_cap=4096, cache_dir=cache_dir,
            pool_manifest=lp_pool_manifest, seed=0,
            resampling={"temperature": 0.05, "method": "lp-mirror"}, mapping="value")
        selected = result.selected
        payload = save_payload("lp_mirror", task["code"], selected.probabilities,
            {"method": "LP-Mirror", "scoring_unit": "legal_window", "native_kernel": "ForwardINF proxy loss drop",
             "mapping": "value_mass(exp(loss_drop/0.05))", "temperature": 0.05,
             "proxy_epochs": 1, "proxy_batch": 4096, "proxy_lr": 0.01,
             "adaptation_epochs": 1, "adapt_batch": 256, "adapt_lr": 0.001,
             "reference_stride": 8, "reference_cap": 4096, "hard_top_k": False,
             "proxy_cache_reused": result.cost.notes.get("proxy_cached", False), "seed": 0}, layout="window")
        logger.info("stage=lp-mirror-distribution-complete task=%s windows=%d proxy_cached=%s payload=%s",
            task["code"], len(index), result.cost.notes.get("proxy_cached", False), payload)

def main():
    logger = configure_logger("main_crb_resample_native_prepare", "main-crb-v0", BATCH, "native-tsds-lp", enable_heartbeat=True)
    started = time.monotonic()
    rows, fragments, source_counts, index, pool_manifest = pool_index(logger)
    engine, _probe, identity = build_model("timerxl-84m", model_path=str(base.BASE_MODEL), device="cuda",
        context_length=base.CONTEXT, prediction_length=base.HORIZON)
    engine.model.eval()
    engine = _probe
    logger.info("stage=model-ready identity=%s", identity.get("model_id", "timerxl-84m"))
    prepare_tsds(engine, rows, fragments, source_counts, index, pool_manifest, logger)
    prepare_lp(engine, fragments, source_counts, index, pool_manifest, logger)
    (OUT / "native-methods.complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete elapsed=%s output=%s", format_duration(time.monotonic() - started), OUT)

if __name__ == "__main__":
    main()
