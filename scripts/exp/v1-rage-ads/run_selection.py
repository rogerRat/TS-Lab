"""Build rolling ADS probability samplers and trajectory control selections."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.ads import ADSConfig, ADSPipeline, cosine_similarity_matrix
from rage.ads import global_top_k_hit_masks, source_probability_descriptor
from rage.ads import summarize_similarity_matrix, uniform_anchor_positions
from rage.core import TimeSeriesFragment
from rage.experiments.development_inputs import load_fragments

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "scripts/exp/v1-crb-redo"))
from rr_features import record_anchors, source_layout

TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ARMS = ("fullRand", "rage", "ancUni", "Traj")
CONTEXT, PREDICTION = 480, 96
SOURCE_DATA = ROOT / "data/src/shard_dataset/src-96"
REFERENCE_ROOT = ROOT / "data/ref/shard_dataset/v1-rage-ads-rolling"
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling"
GRADIENT_ROOT = ROOT / "outputs/train/v1-rage-ads/rolling/gradients"
WARMUP_ROOT = ROOT / "outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000"
SOURCE_CACHE_ROOT = ROOT / "outputs/train/v1-crb-redo/grad-prepare/gradients/timerxl-84m"
PROJECTION_DIMENSION, PROJECTION_SEED = 4096, 42
Q, ETA, RHO, DECAY_DISTANCE = 16.0, 1.0, 0.2, 4
GRADIENT_WINDOW_BATCH, GRADIENT_VJP_BATCH = 32, 32


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


def atomic_npz(path, **items):
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **items)
    os.replace(temporary, path)


class ReadOnlyGradientCache:
    def __init__(self):
        self.rows = {}

    def add(self, key, cache, dataset, fragment_id, start):
        self.rows.setdefault(tuple(key), (cache, dataset, int(fragment_id), int(start)))

    def add_array(self, key, gradient):
        self.rows[tuple(key)] = np.asarray(gradient, dtype=np.float32)

    def __contains__(self, key):
        return tuple(key) in self.rows

    def __getitem__(self, key):
        value = self.rows[tuple(key)]
        if isinstance(value, np.ndarray):
            return value
        cache, dataset, fragment_id, start = value
        return np.asarray(cache.lookup(dataset, np.asarray([[fragment_id, start]],
                                                           dtype=np.int64))[0],
                          dtype=np.float32)


def load_source_pool():
    config = yaml.safe_load((ROOT / "configs/data/src-96.yaml").read_text(encoding="ascii"))
    lengths = np.asarray([int(row["length"]) for row in config["trajectories"]], dtype=np.int64)
    legal = lengths - CONTEXT - PREDICTION + 1
    manifest = json.loads((SOURCE_DATA / "manifest.json").read_text(encoding="ascii"))
    offsets = np.asarray(np.load(SOURCE_DATA / "offsets.npy", mmap_mode="r"), dtype=np.int64)
    values = np.load(SOURCE_DATA / "values.npy", mmap_mode="r")
    if (manifest.get("name") != "src-96" or manifest.get("parent") != "src-4096"
            or int(manifest.get("trajectory_count", -1)) != 96
            or int(manifest.get("total_points", -1)) != int(offsets[-1])
            or len(lengths) != 96 or np.any(legal <= 0) or int(legal.sum()) != 2666312
            or len(offsets) != 97 or not np.array_equal(np.diff(offsets), lengths)
            or int(offsets[-1]) != len(values)):
        raise ValueError("src-96 packed source pool does not match its frozen config")
    fragments = []
    for fid, row in enumerate(config["trajectories"]):
        data = np.asarray(values[int(offsets[fid]):int(offsets[fid + 1])],
                          dtype=np.float32).copy()
        if not np.isfinite(data).all():
            raise ValueError("src-96 contains non-finite values")
        fragments.append(TimeSeriesFragment(data, entity_id=str(row["series_id"]), fragment_id=fid))
    return fragments, legal


def source_cache_inputs(source_info, source_fragments, legal):
    _config, _old_legal, _old_offsets, files = source_info
    positions, cache = {}, ReadOnlyGradientCache()
    for fid, fragment in enumerate(source_fragments):
        starts = fragment.valid_starts(CONTEXT, PREDICTION)
        record, dataset = files[("uniform", fid)]
        anchors = record_anchors(record, "uniform")
        if (len(anchors) < 2 or anchors[0] != 0 or anchors[-1] != legal[fid] - 1
                or np.any(np.diff(anchors) <= 0)
                or not np.array_equal(starts[anchors], anchors + 1)):
            raise ValueError("source q32 uniform cache anchors fail position checks")
        positions[("source", fid)] = anchors.tolist()
        for anchor in anchors:
            cache.add(("source", fid, int(anchor) + 1), dataset, "src-96",
                      fid, int(anchor) + 1)
        record, dataset = files[("crb", fid)]
        for anchor in record_anchors(record, "crb"):
            cache.add(("source", fid, int(anchor) + 1), dataset, "src-96",
                      fid, int(anchor) + 1)
    return positions, cache


def validate_reference_cache(path):
    with np.load(path, allow_pickle=False) as data:
        fields = {"gradients", "fragment_ids", "starts", "context_length",
                  "prediction_length", "warmup_steps", "projection_dimension",
                  "projection_seed", "warmup_checkpoint", "model_id", "gradient_protocol"}
        if not fields.issubset(data.files):
            raise ValueError("reference coarse gradient file lacks protocol fields")
        if (int(data["context_length"]) != CONTEXT
                or int(data["prediction_length"]) != PREDICTION
                or int(data["warmup_steps"]) != 1000
                or int(data["projection_dimension"]) != PROJECTION_DIMENSION
                or int(data["projection_seed"]) != PROJECTION_SEED
                or str(Path(str(data["warmup_checkpoint"].item())).resolve()) != str(WARMUP_ROOT.resolve())
                or str(data["model_id"].item()) != "thuml/timer-base-84m"
                or str(data["gradient_protocol"].item()) != "independent-native-loss-bf16-v2"):
            raise ValueError("reference coarse gradient protocol mismatch")
        gradients = np.asarray(data["gradients"], dtype=np.float32)
        fragment_ids = np.asarray(data["fragment_ids"], dtype=np.int64)
        starts = np.asarray(data["starts"], dtype=np.int64)
    if (gradients.ndim != 2 or gradients.shape[1] != PROJECTION_DIMENSION
            or len(gradients) != len(fragment_ids) or len(gradients) != len(starts)):
        raise ValueError("reference coarse gradient shape or identity mismatch")
    return gradients, fragment_ids, starts


def prepare_reference_gradients(task, fragments, probe, logger):
    path = output_path("train", GRADIENT_ROOT / task / "reference_coarse.npz")
    if path.is_file():
        return validate_reference_cache(path)
    gradients, ids, physical_starts = [], [], []
    for fragment in fragments:
        starts = fragment.valid_starts(CONTEXT, PREDICTION)
        positions = uniform_anchor_positions(len(starts), Q * (1.0 + ETA) / ETA)
        selected = starts[np.asarray(positions, dtype=np.int64)]
        windows = np.stack([fragment.extract_window(int(s), CONTEXT, PREDICTION)
                            for s in selected])
        _losses, sketches = probe.batch_gradient_sketch(windows, batch_size=GRADIENT_WINDOW_BATCH)
        gradients.append(np.asarray(sketches, dtype=np.float32))
        ids.extend([int(fragment.fragment_id)] * len(selected))
        physical_starts.extend(selected.astype(np.int64).tolist())
        logger.info("stage=reference-coarse-progress task=%s fragment=%d anchors=%d",
                    task, fragment.fragment_id, len(selected))
    values = np.concatenate(gradients, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path, gradients=values,
               fragment_ids=np.asarray(ids, dtype=np.int64),
               starts=np.asarray(physical_starts, dtype=np.int64),
               context_length=np.asarray(CONTEXT), prediction_length=np.asarray(PREDICTION),
               warmup_steps=np.asarray(1000), projection_dimension=np.asarray(PROJECTION_DIMENSION),
               projection_seed=np.asarray(PROJECTION_SEED),
               warmup_checkpoint=np.asarray(str(WARMUP_ROOT.resolve())),
               model_id=np.asarray("thuml/timer-base-84m"),
               gradient_protocol=np.asarray("independent-native-loss-bf16-v2"))
    logger.info("stage=reference-coarse-complete task=%s anchors=%d output=%s",
                task, len(values), path)
    return values, np.asarray(ids, dtype=np.int64), np.asarray(physical_starts, dtype=np.int64)


def add_reference_rows(cache, fragments, gradients, fragment_ids, starts):
    lookup = {(int(fid), int(start)): row for row, (fid, start)
              in enumerate(zip(fragment_ids, starts))}
    for fragment in fragments:
        fid = int(fragment.fragment_id)
        valid = fragment.valid_starts(CONTEXT, PREDICTION)
        positions = uniform_anchor_positions(len(valid), Q * (1.0 + ETA) / ETA)
        for position in positions:
            start = int(valid[position])
            cache.add_array(("reference", fid, start), gradients[lookup[(fid, start)]])


def unit_rows(gradients):
    values = np.asarray(gradients, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values),
                     where=norms[:, None] > 0)


def trajectory_directions(fragments, positions, cache, side):
    output = []
    for fragment in fragments:
        fid = int(fragment.fragment_id)
        starts = fragment.valid_starts(CONTEXT, PREDICTION)
        grads = np.stack([cache[(side, fid, int(starts[p]))] for p in positions[(side, fid)]])
        mean = unit_rows(grads).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        output.append(mean / norm if norm > 0 else mean)
    return np.asarray(output, dtype=np.float32)


def run_task(task, source_fragments, legal, source_info, source_positions,
             source_cache, probe, identity, model_fields, logger):
    reference_dir = REFERENCE_ROOT / task
    if not (reference_dir / ".complete").is_file():
        raise FileNotFoundError("rolling reference input is incomplete: " + task)
    ref_manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="ascii"))
    if ref_manifest.get("answer_values_included") is not False:
        raise ValueError("rolling reference input includes answer values")
    references = load_fragments(reference_dir / "references.npz")
    if len(references) != int(ref_manifest["fragments"]):
        raise ValueError("rolling reference fragment count differs from manifest")
    ref_gradients, ref_ids, ref_starts = prepare_reference_gradients(task, references, probe, logger)
    reference_positions = {
        ("reference", int(fragment.fragment_id)): uniform_anchor_positions(
            len(fragment.valid_starts(CONTEXT, PREDICTION)), Q * (1.0 + ETA) / ETA)
        for fragment in references
    }
    task_cache = ReadOnlyGradientCache()
    task_cache.rows.update(source_cache.rows)
    add_reference_rows(task_cache, references, ref_gradients, ref_ids, ref_starts)
    task_root = output_path("selection", SELECTION_ROOT / task)
    task_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for arm in ARMS:
        arm_dir = task_root / arm
        if (arm_dir / ".complete").is_file():
            logger.info("stage=arm-skip task=%s arm=%s reason=complete", task, arm)
            continue
        arm_dir.mkdir(parents=True, exist_ok=True)
        score_summaries, details = [], {}
        if arm == "fullRand":
            sampling = dict(schema=1, mode="uniform_all", replacement=True,
                            candidate_count=int(legal.sum()), seed=0)
        elif arm in ("rage", "ancUni"):
            rounds = 1 if arm == "rage" else 0
            config = ADSConfig(q=Q, eta=ETA, top_ratio=RHO, rounds=rounds,
                               decay_distance=DECAY_DISTANCE, aggregation="max")
            logger.info("stage=ads-start task=%s arm=%s q=%s eta=%s rho=%s rounds=%d h=%d",
                        task, arm, Q, ETA, RHO, rounds, DECAY_DISTANCE)
            def gradient_computer(fragment, starts):
                windows = np.stack([fragment.extract_window(int(s), CONTEXT, PREDICTION)
                                    for s in starts])
                _losses, sketches = probe.batch_gradient_sketch(windows, batch_size=GRADIENT_WINDOW_BATCH)
                return np.asarray(sketches, dtype=np.float32)
            def progress(event):
                logger.info("stage=ads-progress task=%s arm=%s details=%s",
                            task, arm, json.dumps(event, sort_keys=True))
            report = ADSPipeline(
                config, gradient_computer, context_length=CONTEXT,
                prediction_length=PREDICTION, window_stride=1,
                gradient_batch_size=GRADIENT_WINDOW_BATCH,
                progress_callback=progress, gradient_cache=task_cache,
                coarse_anchor_positions={**source_positions, **reference_positions},
            ).run(references, source_fragments)
            descriptor = source_probability_descriptor(report)
            write_json(arm_dir / "probability.json", descriptor)
            score_summaries = report.score_matrix_summaries
            details = dict(
                ads_config=dict(q=Q, eta=ETA, rho=RHO, rounds=rounds,
                                decay_distance=DECAY_DISTANCE, aggregation="max", overlap="max"),
                actual_gradient_counts=report.actual_gradient_counts,
                nominal_gradient_counts=report.nominal_gradient_counts,
                cached_gradient_counts=report.cached_gradient_counts,
                zero_norm_anchors=report.zero_norm_anchors,
                zero_norm_pairs=report.zero_norm_pairs,
                probability_fallback=report.probability_fallback,
                selected_top_k=report.selected_top_k,
                matches_by_round=[
                    dict(pair_count=x.pair_count,
                         reference_anchor_hits=x.reference_anchor_hits,
                         source_anchor_hits=x.source_anchor_hits, threshold=x.threshold)
                    for x in report.matches_by_round
                ],
            )
            sampling = dict(schema=1, mode="probability_descriptor",
                            descriptor_path="probability.json", replacement=True,
                            candidate_count=int(legal.sum()), seed=0)
            logger.info("stage=ads-finished task=%s arm=%s cached=%s new=%s matrices=%d",
                        task, arm, report.cached_gradient_counts,
                        report.actual_gradient_counts, len(score_summaries))
        elif arm == "Traj":
            ref_vectors = trajectory_directions(references, reference_positions,
                                                task_cache, "reference")
            src_vectors = trajectory_directions(source_fragments, source_positions,
                                                task_cache, "source")
            matrix, zero_pairs = cosine_similarity_matrix(ref_vectors, src_vectors)
            top_k = int(np.ceil(matrix.size * RHO))
            ref_hits, src_hits, threshold = global_top_k_hit_masks(
                matrix, top_k,
                reference_keys=[(i,) for i in range(len(references))],
                source_keys=[(i,) for i in range(len(source_fragments))])
            chosen = np.flatnonzero(src_hits).astype(np.int64)
            if not len(chosen):
                raise RuntimeError("trajectory matching selected no source trajectory")
            probabilities = np.full(len(chosen), 1.0 / len(chosen), dtype=np.float64)
            summary = summarize_similarity_matrix(matrix, top_k, threshold=threshold)
            summary["stage"] = "trajectory_final"
            score_summaries = [summary]
            trajectory_best = matrix.max(axis=0)
            best_order = np.lexsort((np.arange(len(source_fragments)), -trajectory_best))
            best_trajectory = int(best_order[0])
            details = dict(matching="global_top_rho_trajectory_direction_pairs",
                           rho=RHO, top_k_pairs=top_k, threshold=threshold,
                           matched_reference_trajectories=int(ref_hits.sum()),
                           selected_source_trajectories=int(len(chosen)),
                           zero_norm_pairs=zero_pairs,
                           top_one_percent_source_trajectory_count=max(
                               1, int(np.ceil(len(source_fragments) * 0.01))),
                           best_source_trajectory_id=best_trajectory,
                           best_source_trajectory_score=float(trajectory_best[best_trajectory]),
                           trajectory_top_one_percent_note=(
                               "one source trajectory at this pool size; no within-trajectory clustering inference"))
            sampling = dict(schema=1, mode="trajectory_uniform",
                            fragment_ids=chosen.tolist(),
                            trajectory_probabilities=probabilities.tolist(),
                            replacement=True, candidate_count=int(legal.sum()), seed=0)
        else:
            raise ValueError("unsupported arm: " + arm)
        selection = dict(
            schema="v1-rage-ads-selection-v1", experiment="v1-rage-ads",
            batch="rolling-eight-task", task=task, arm=arm,
            reference_manifest=str(reference_dir / "manifest.json"),
            reference_gradient_file=str(GRADIENT_ROOT / task / "reference_coarse.npz"),
            source_gradient_cache=str(SOURCE_CACHE_ROOT),
            warmup_checkpoint=str(WARMUP_ROOT.resolve()), warmup_steps=1000,
            gradient_window_batch_size=GRADIENT_WINDOW_BATCH,
            gradient_vjp_batch_size=GRADIENT_VJP_BATCH,
            model=identity, model_fields=model_fields,
            context_length=CONTEXT, prediction_length=PREDICTION,
            source_windows=int(legal.sum()), source_trajectories=len(source_fragments),
            reference_trajectories=len(references),
            score_matrix_summaries=score_summaries, diagnostics=details,
            sampling_rule=sampling["mode"], elapsed_seconds=time.monotonic() - started,
        )
        if arm in ("rage", "ancUni"):
            selection["probability_descriptor"] = str(arm_dir / "probability.json")
        write_json(arm_dir / "sampling.json", sampling)
        write_json(arm_dir / "selection.json", selection)
        (arm_dir / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=arm-complete task=%s arm=%s elapsed=%s output=%s",
                    task, arm, format_duration(time.monotonic() - started), arm_dir)

    if any(not (task_root / arm / ".complete").is_file() for arm in ARMS):
        raise RuntimeError("one or more selection arms are incomplete for " + task)
    write_json(task_root / "manifest.json", dict(
        experiment="v1-rage-ads", batch="rolling-eight-task", task=task,
        arms=list(ARMS), q=Q, eta=ETA, rho=RHO, rounds=1,
        decay_distance=DECAY_DISTANCE, aggregation="max", overlap="max",
        gradient_window_batch_size=GRADIENT_WINDOW_BATCH,
        gradient_vjp_batch_size=GRADIENT_VJP_BATCH,
        context_length=CONTEXT, prediction_length=PREDICTION, seed=0,
        warmup_steps=1000, projection_dimension=PROJECTION_DIMENSION,
        projection_seed=PROJECTION_SEED, source_windows=int(legal.sum()),
        reference_manifest=str(reference_dir / "manifest.json"),
        elapsed_seconds=time.monotonic() - started,
    ))
    (task_root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-complete task=%s elapsed=%s output=%s",
                task, format_duration(time.monotonic() - started), task_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    args = parser.parse_args()
    selected = TASKS if args.task == "all" else (args.task,)
    logger = configure_logger("v1_rage_ads_selection", "v1-rage-ads",
                              "rolling-eight-task", "selection")
    started = time.monotonic()
    logger.info("stage=config tasks=%s arms=%s context=%d prediction=%d q=%s eta=%s rho=%s rounds=1 h=%d aggregation=max overlap=max warmup=%s window_batch=%d vjp_batch=%d",
                ",".join(selected), ",".join(ARMS), CONTEXT, PREDICTION,
                Q, ETA, RHO, DECAY_DISTANCE, WARMUP_ROOT,
                GRADIENT_WINDOW_BATCH, GRADIENT_VJP_BATCH)
    if not (WARMUP_ROOT / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("step-1000 warmup adapter is missing")
    from metrics import build_gradient_probe
    from audit_metrics import validate_source_gradient_caches
    validate_source_gradient_caches()
    source_info = source_layout()
    source_fragments, legal = load_source_pool()
    source_positions, source_cache = source_cache_inputs(source_info, source_fragments, legal)
    model_engine, probe, identity, model_fields = build_gradient_probe()
    probe.vjp_batch_size = GRADIENT_VJP_BATCH
    if (identity["model_id"] != "thuml/timer-base-84m"
            or probe.config.projection_dimension != PROJECTION_DIMENSION
            or probe.config.projection_seed != PROJECTION_SEED
            or model_fields["warmup_steps"] != 1000):
        raise ValueError("warmed TimerXL gradient probe identity mismatch")
    for index, task in enumerate(selected, start=1):
        run_task(task, source_fragments, legal, source_info, source_positions,
                 source_cache, probe, identity, model_fields, logger)
        logger.info("stage=task-progress task=%s progress=%d/%d eta=%s",
                    task, index, len(selected),
                    format_duration((time.monotonic() - started) * (len(selected) - index) / index))
    if args.task == "all":
        root = output_path("selection", SELECTION_ROOT)
        if any(not (root / task / ".complete").is_file() for task in TASKS):
            raise RuntimeError("rolling ADS selection set is incomplete")
        (root / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete tasks=%d elapsed=%s output=%s",
                len(selected), format_duration(time.monotonic() - started), SELECTION_ROOT)


if __name__ == "__main__":
    main()

