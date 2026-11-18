"""Run the controlled main-crb-supp-chronos2-small queue one target at a time."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "third_parties/TIME/src", ROOT / "scripts/exp/main-crb-v0", ROOT / "scripts/exp/v1-crb-redo"):
    sys.path.insert(0, str(path))

import run_directional_time1 as base
import prepare_eight_task as old_prepare
import prepare_resample_core as old_pool
import ref_inputs
from assets.gradient import GradientCache
from configs.logger import configure_logger, format_duration
from configs.paths import output_path
from rage.experiments.development_inputs import load_fragments
from rage.experiments.ranking import rank_mass
from rage.experiments.v7_shape import shape_rows

PYTHON = Path(sys.executable)
SOURCE_ROOT = ROOT / "outputs/selection/main-crb-v0/q16-src256-time1-q8"
SOURCE_COMPONENTS = ROOT / "outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/components"
POOL_ROOT = ROOT / "outputs/selection/main-crb-v0/q16-src256-eight-task-resample/shared/pool"
OLD_RAND = ROOT / "outputs/train/main-crb-v0/q16-src256-eight-task-resample/shared/rand/checkpoint"
FEV_SOURCE = ROOT / "configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml"
FEV_DATA = ROOT / "data/bench/fev/datasets/main-crb-supp"
TIME_DATA = ROOT / "data/bench/time/v1-rage-trev/dataset"
REF_ROOT = ROOT / "data/ref/shard_dataset/main-crb-supp-chronos2-small"
CONFIG_ROOT = ROOT / "configs/experiments/main-crb-supp-chronos2-small"
DOC_ROOT = ROOT / "docs/exp/main-crb-supp-chronos2-small"
ARMS = ("e1-uniform-mean", "e2-uniform-shape", "e3-crb-mean", "e4-crb-shape", "e5-target-only", "e0-rand")
TASKS = (
    {"code": "time-crypto-d", "kind": "time", "key": "Crypto/D"},
    {"code": "time-supply-chain-customer-d", "kind": "time", "key": "Supply_Chain_Customer/D"},
    {"code": "time-supply-chain-location-d", "kind": "time", "key": "Supply_Chain_Location/D"},
    {"code": "time-ne-china-wind-h", "kind": "time", "key": "NE_China_Wind/H"},
    {"code": "time-coastal-t-s-20t", "kind": "time", "key": "Coastal_T_S/20T"},
    {"code": "time-australia-solar-h", "kind": "time", "key": "Australia_Solar/H"},
    {"code": "time-finland-traffic-15t", "kind": "time", "key": "Finland_Traffic/15T"},
    {"code": "fev-jena-weather-1h", "kind": "fev", "key": 10},
    {"code": "fev-ett-1h", "kind": "fev", "key": 6},
    {"code": "time-metropt-3-5t", "kind": "time", "key": "MetroPT-3/5T"},
    {"code": "time-current-velocity-5t", "kind": "time", "key": "current_velocity/5T"},
    {"code": "time-azure2019-u-5t", "kind": "time", "key": "azure2019_U/5T"},
    {"code": "time-sg-pm25-h", "kind": "time", "key": "SG_PM25/H"},
    {"code": "time-water-quality-darwin-15t", "kind": "time", "key": "Water_Quality_Darwin/15T"},
    {"code": "time-open-electricity-nem-5t", "kind": "time", "key": "OpenElectricity_NEM/5T"},
    {"code": "time-eweld-load-15t", "kind": "time", "key": "EWELD_Load/15T"},
    {"code": "time-port-activity-d", "kind": "time", "key": "Port_Activity/D"},
    {"code": "time-current-velocity-10t", "kind": "time", "key": "current_velocity/10T"},
    {"code": "time-coastal-t-s-5t", "kind": "time", "key": "Coastal_T_S/5T"},
    {"code": "time-epf-electricity-price-h", "kind": "time", "key": "epf_electricity_price/H"},
    {"code": "time-current-velocity-15t", "kind": "time", "key": "current_velocity/15T"},
    {"code": "time-coastal-t-s-15t", "kind": "time", "key": "Coastal_T_S/15T"},
    {"code": "time-azure2019-d-5t", "kind": "time", "key": "azure2019_D/5T"},
    {"code": "fev-hospital-admissions-1d", "kind": "fev", "key": 9},
)
ACTIVE = []
STOPPED = False


def roots(validation):
    name = "validation" if validation else "run"
    return {
        "name": name,
        "reference": REF_ROOT / name,
        "cache": output_path("grad_cache", ROOT / "outputs/grad_cache/main-crb-supp-chronos2-small" / name),
        "selection": output_path("selection", ROOT / "outputs/selection/main-crb-supp-chronos2-small" / name),
        "train": output_path("train", ROOT / "outputs/train/main-crb-supp-chronos2-small" / name),
        "eval": output_path("eval", ROOT / "outputs/eval/main-crb-supp-chronos2-small" / name),
        "configs": CONFIG_ROOT / name,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    temp.replace(path)


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=False), encoding="ascii")


def prepare_eval(task, validation):
    r = roots(validation)
    code = task["code"]
    if task["kind"] == "time":
        old = ROOT / "configs/eval/time/v1-rage-ads-time-01.yaml"
        cfg = yaml.safe_load(old.read_text(encoding="ascii"))
        target = task["key"]
        source = TIME_DATA / target
        if not source.is_dir():
            raise FileNotFoundError(source)
        subset = ROOT / "data/bench/time/main-crb-supp-chronos2-small" / code
        link = subset / "dataset" / target
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise ValueError("TIME data link differs")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(source.resolve(), target_is_directory=True)
        inherited = json.loads((ROOT / "data/bench/time/v1-rage-trev/data_manifest.json").read_text())
        inherited["verified_tasks"] = [{"dataset": target, "term": "short"}]
        inherited["validation"] = "main-crb-supp-chronos2-small safe input and schema field validation"
        write_json(subset / "data_manifest.json", inherited)
        cfg["adapter"]["config"]["model"]["model_key"] = "chronos-2-small"
        cfg["bench_config"]["dataset"]["root"] = str((subset / "dataset").relative_to(ROOT))
        cfg["bench_config"]["tasks"]["items"] = [{"dataset": target, "terms": ["short"]}]
        path = ROOT / "configs/eval/time" / ("main-crb-supp-chronos2-small-" + code + ".yaml")
    else:
        cfg = yaml.safe_load((ROOT / "configs/eval/fev/v1-rage-ads-fev-09.yaml").read_text())
        cfg["adapter"]["config"]["model"]["model_key"] = "chronos-2-small"
        cfg["bench_config"]["task_file"] = "main-crb-supp-chronos2-small-tasks.yaml"
        cfg["bench_config"]["task_indices"] = [int(task["key"])]
        path = ROOT / "configs/eval/fev" / ("main-crb-supp-chronos2-small-" + code + ".yaml")
    write_yaml(path, cfg)
    return path


def prepare_fev_task_file():
    cfg = yaml.safe_load(FEV_SOURCE.read_text(encoding="ascii"))
    for index, relative in ((6, "ETT/1H"), (10, "jena_weather/1H")):
        source = FEV_DATA / relative / "train-00000-of-00001.parquet"
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        cfg["tasks"][index]["dataset_path"] = str(source)
    path = ROOT / "configs/eval/fev/main-crb-supp-chronos2-small-tasks.yaml"
    write_yaml(path, cfg)
    return path


def prepare_source(logger):
    manifest = old_pool.prepare_pool()
    if int(manifest["total_equivalents"]) != 6382020:
        raise ValueError("Source pool size differs")
    for method in ("crb", "uni"):
        anchor = SOURCE_ROOT / method / "source_anchors.npz"
        index = SOURCE_ROOT / method / (method + "_block_index.npz")
        if not anchor.is_file() or not index.is_file():
            raise FileNotFoundError(index)
        with np.load(anchor, allow_pickle=False) as rows:
            if len(rows["gradient_rows"]) != 399044:
                raise ValueError("Source K differs for " + method)
        with np.load(index, allow_pickle=False) as blocks:
            if len(blocks["fragment_ids"]) != 398788:
                raise ValueError("Source block count differs for " + method)
    for name in ("directions_fp16.npy", "magnitudes_fp32.npy"):
        if not (SOURCE_COMPONENTS / name).is_file():
            raise FileNotFoundError(SOURCE_COMPONENTS / name)
    if not (OLD_RAND / "rage_checkpoint_manifest.json").is_file():
        raise FileNotFoundError(OLD_RAND)
    logger.info("stage=source-ready pool_windows=%d anchors_per_placement=399044 blocks_per_placement=398788 q=16 cover=longest split=min_kappa budget=1:0:1", manifest["total_equivalents"])
    return manifest


def prepare_reference(task, validation, logger):
    r = roots(validation)
    code = task["code"]
    complete = r["selection"] / code / "reference" / ".complete"
    if complete.is_file():
        return json.loads((complete.parent / "manifest.json").read_text())
    eval_path = prepare_eval(task, validation)
    ref_inputs.prepare_one(code, task["kind"], task["key"], logger,
                           config_path=eval_path if task["kind"] == "time" else prepare_fev_task_file(),
                           data_root=r["reference"])
    source = r["reference"] / code
    input_manifest = json.loads((source / "manifest.json").read_text())
    if input_manifest["answer_values_included"] is not False:
        raise ValueError("Reference includes answer values")
    fragments = load_fragments(source / "references.npz")
    counts = np.asarray([len(f.valid_starts(base.CONTEXT, base.HORIZON)) for f in fragments], dtype=np.int64)
    positions = [base.exact_uniform_positions(int(n), 8) for n in counts]
    dataset = "main-crb-supp-chronos2-small-reference:" + code
    cache_root = r["cache"] / code
    component_root = cache_root / "components"
    engine = probe = cache = None
    started = time.monotonic()
    try:
        engine, probe, identity = base.build_probe(logger)
        os.environ["RAGE_GRADIENT_CACHE_ROOT"] = str(cache_root / "raw_bf16")
        cache = GradientCache(cache_root, probe,
                              model_identity={**identity, "probe_checkpoint": str(base.PROBE_ADAPTER)},
                              data_fields={"experiment": "main-crb-supp-chronos2-small", "task": code,
                                           "dataset": dataset, "reference_manifest": str(source / "manifest.json"),
                                           "q": 8, "projection_dimension": base.PROJECTION_DIM,
                                           "projection_seed": base.PROJECTION_SEED},
                              context_length=base.CONTEXT, prediction_length=base.HORIZON)
        base.prepare_cache_positions(cache, dataset, fragments, positions, logger)
        cache.freeze()
        old_component_root = base.COMPONENT_ROOT
        base.COMPONENT_ROOT = component_root
        try:
            lookup, direction_path, magnitude_path, _ = base.write_direction_components(
                cache, base.collect_union_keys(cache), np.empty(0, dtype=np.int64), [],
                counts, fragments, logger)
        finally:
            base.COMPONENT_ROOT = old_component_root
        rows = []
        for fragment, selected in zip(fragments, positions):
            starts = fragment.valid_starts(base.CONTEXT, base.HORIZON)
            rows.extend(lookup[(dataset, int(fragment.fragment_id), int(starts[int(p)]))] for p in selected)
        ref_root = r["selection"] / code / "reference"
        ref_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ref_root / "reference_anchors.npz",
                            fragment_counts=counts, positions=np.concatenate(positions),
                            gradient_rows=np.asarray(rows, dtype=np.int64))
        old_prepare.materialize_windows(fragments, counts, ref_root / "training_windows.npz", logger, code)
        result = {"task": code, "kind": task["kind"], "key": task["key"],
                  "safe_history_manifest": str(source / "manifest.json"),
                  "eval_config": str(eval_path.relative_to(ROOT)), "fragments": len(fragments),
                  "entities": len({f.entity_id for f in fragments}),
                  "legal_windows": int(counts.sum()), "q8_anchors": len(rows),
                  "reference_dataset": dataset, "answer_values_included": False,
                  "direction_path": str(direction_path), "magnitude_path": str(magnitude_path)}
        write_json(ref_root / "manifest.json", result)
        complete.write_text("complete\n", encoding="ascii")
        logger.info("stage=reference-ready task=%s fragments=%d q8_anchors=%d elapsed=%s", code, len(fragments), len(rows), format_duration(time.monotonic() - started))
        return result
    finally:
        if cache is not None:
            cache.close()
        del engine, probe


def raw_rows(directions, magnitudes, indices):
    ids = np.asarray(indices, dtype=np.int64)
    return np.asarray(directions[ids], dtype=np.float32) * np.asarray(magnitudes[ids], dtype=np.float32)[:, None]


def normalize_rows(values):
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] > 0)


def reference_vector(task, representation, manifest, validation):
    r = roots(validation)
    ref_root = r["selection"] / task["code"] / "reference"
    with np.load(ref_root / "reference_anchors.npz", allow_pickle=False) as a:
        counts = np.asarray(a["fragment_counts"], dtype=np.int64)
        positions = np.asarray(a["positions"], dtype=np.int64)
        row_ids = np.asarray(a["gradient_rows"], dtype=np.int64)
    fragments = load_fragments(r["reference"] / task["code"] / "references.npz")
    directions = np.load(manifest["direction_path"], mmap_mode="r")
    magnitudes = np.load(manifest["magnitude_path"], mmap_mode="r")
    sums = {}
    totals = {}
    cursor = 0
    for fragment, count in zip(fragments, counts):
        n = len(base.exact_uniform_positions(int(count), 8))
        left = raw_rows(directions, magnitudes, row_ids[cursor:cursor + n])
        cursor += n
        if representation == "mean":
            vectors = left
        else:
            vectors = shape_rows(left[:-1], left[1:])
        key = fragment.entity_id
        value = vectors.sum(axis=0, dtype=np.float64)
        sums[key] = sums.get(key, np.zeros_like(value)) + value
        totals[key] = totals.get(key, 0) + len(vectors)
    if cursor != len(positions):
        raise ValueError("Reference anchor indexing differs")
    opinions = []
    for key in sorted(sums):
        vector = sums[key] / totals[key]
        norm = np.linalg.norm(vector)
        opinions.append(vector / norm if norm > 0 else np.zeros_like(vector))
    result = np.mean(opinions, axis=0)
    if not np.isfinite(result).all():
        raise ValueError("Reference vector is nonfinite")
    return result.astype(np.float32)


def source_scores(method, representation, ref_vector, pool):
    with np.load(SOURCE_ROOT / method / (method + "_block_index.npz"), allow_pickle=False) as a:
        fids = np.asarray(a["fragment_ids"], dtype=np.int64)
        left_positions = np.asarray(a["left_positions"], dtype=np.int64)
        right_positions = np.asarray(a["right_positions"], dtype=np.int64)
        left_rows = np.asarray(a["left_gradient_rows"], dtype=np.int64)
        right_rows = np.asarray(a["right_gradient_rows"], dtype=np.int64)
    directions = np.load(SOURCE_COMPONENTS / "directions_fp16.npy", mmap_mode="r")
    magnitudes = np.load(SOURCE_COMPONENTS / "magnitudes_fp32.npy", mmap_mode="r")
    counts = np.asarray(pool["trajectory_legal_windows"], dtype=np.int64)
    if len(counts) != 256 or len(fids) != 398788:
        raise ValueError("Source block geometry differs")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for source block scoring")
    scores = np.empty(len(fids), dtype=np.float64)
    target = torch.as_tensor(ref_vector, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        for first in range(0, len(fids), 2048):
            stop = min(first + 2048, len(fids))
            left = torch.as_tensor(raw_rows(directions, magnitudes, left_rows[first:stop]), device="cuda", dtype=torch.float64)
            right = torch.as_tensor(raw_rows(directions, magnitudes, right_rows[first:stop]), device="cuda", dtype=torch.float64)
            midpoint = 0.5 * (left + right)
            if representation == "mean":
                norm = torch.linalg.vector_norm(midpoint, dim=1)
                vectors = midpoint / norm.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
            else:
                slope = (right - left) / np.sqrt(12.0)
                energy = (midpoint * midpoint).sum(dim=1) + (slope * slope).sum(dim=1)
                vectors = torch.cat((midpoint, slope), dim=1) / torch.sqrt(
                    energy.clamp_min(torch.finfo(torch.float64).tiny))[:, None]
            scores[first:stop] = (vectors @ target).cpu().numpy()
    offsets = [0]
    cursor = 0
    for fid, legal in enumerate(counts):
        stop = int(np.searchsorted(fids, fid, side="right"))
        left = left_positions[cursor:stop]
        right = right_positions[cursor:stop]
        if not len(left) or int(left[0]) != 0 or int(right[-1]) != int(legal) - 1:
            raise ValueError("Source block endpoints do not span the trajectory")
        if np.any(left[1:] != right[:-1]):
            raise ValueError("Source blocks do not form adjacent native intervals")
        owned = right - left
        owned[-1] += 1
        if np.any(owned <= 0) or int(owned.sum()) != int(legal):
            raise ValueError("Source native block ownership differs")
        offsets.extend((offsets[-1] + np.cumsum(owned)).tolist())
        cursor = stop
    if cursor != len(fids) or offsets[-1] != 6382020:
        raise ValueError("Source pool partition differs")
    return scores, np.asarray(offsets, dtype=np.int64)


def prepare_arm_payload(task, arm, ref_manifest, pool, validation, logger):
    r = roots(validation)
    path = r["selection"] / task["code"] / arm / "resampling_payload.npz"
    if path.is_file():
        return path
    method = "uni" if "uniform" in arm else "crb"
    representation = "shape" if "shape" in arm else "mean"
    vector = reference_vector(task, representation, ref_manifest, validation)
    scores, offsets = source_scores(method, representation, vector, pool)
    rank, probabilities, audit = rank_mass(scores, np.ones(len(scores), dtype=np.int64))
    if not np.allclose(rank, probabilities, rtol=0, atol=1e-12):
        raise ValueError("Unit source prior changed rank mass")
    metadata = {"experiment": "main-crb-supp-chronos2-small", "arm": arm, "task": task["code"],
                "sampling_unit": "block", "within_unit_sampling": "uniform_owned_legal_start",
                "source_placement": method, "representation": representation,
                "source_q": 16, "reference_q": 8, "source_anchor_evaluations": 399044,
                "source_length_prior": False, "reference_length_prior": False,
                "reference_aggregation": "mean_q8_anchor_or_shape_by_entity_then_equal_entities",
                "rank_probability_revision": audit["rank_probability_revision"],
                "rank_curve": audit["curve"], "rank_temperature": audit["temperature"],
                "rank_ess": audit["rank_ess"], "block_count": len(scores),
                "legal_windows": int(offsets[-1])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probabilities, unit_layout=np.asarray("interval"),
                        unit_offsets=offsets, resampling_metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    logger.info("stage=selection-ready task=%s arm=%s blocks=%d rank_ess=%.1f", task["code"], arm, len(scores), audit["rank_ess"])
    return path


def stage_config(task, arm, ref_manifest, validation, steps):
    r = roots(validation)
    code = task["code"]
    cdir = r["configs"] / code
    template = yaml.safe_load((ROOT / "configs/experiments/main-crb-v0/resample/TIME1/reft-train.yaml").read_text())
    template["run_id"] = "main-crb-supp-chronos2-small-" + r["name"] + "-" + code + "-" + arm
    template["model"] = "chronos-2-small"
    template["model_path"] = None
    template["training"]["lora"]["target_modules"] = ["self_attention.q", "self_attention.v"]
    template["training"]["steps"] = steps
    template["training"]["schedule_total_steps"] = steps
    template["training"]["recipe"]["schedule_epochs"] = steps
    milestones = (1, 2, 3) if validation else (1000, 2000, 4000)
    if arm != "e0-rand":
        template["training"]["evaluation_steps"] = list(milestones)
    template["output_dir"] = str(r["train"] / code / arm)
    if arm == "e5-target-only":
        template["data"] = {"id": code + "-safe-reference-all-windows",
                            "windows": str(r["selection"] / code / "reference/training_windows.npz"),
                            "windows_key": "windows", "context_length": 480, "prediction_length": 96}
    else:
        template["data"] = {"id": "src-256-" + code + "-" + arm,
                            "resampling_payload": str(r["selection"] / code / arm / "resampling_payload.npz"),
                            "resampling_pool": str(POOL_ROOT),
                            "context_length": 480, "prediction_length": 96}
    train_path = cdir / (arm + "-train.yaml")
    write_yaml(train_path, template)
    train_pipeline = cdir / (arm + "-train-pipeline.yaml")
    write_yaml(train_pipeline, {"schema_version": 1, "run_id": template["run_id"] + "-train",
                               "stages": [{"name": "train", "config": str(train_path.relative_to(ROOT))}]})
    evaluations = []
    for step in ((steps,) if arm == "e0-rand" else milestones):
        if step == steps:
            checkpoint = r["train"] / code / arm / "checkpoint"
        else:
            checkpoint = r["train"] / code / arm / "checkpoints" / ("step-%d" % step) / "checkpoint"
        eval_cfg = {"schema_version": 1, "checkpoint": str(checkpoint),
                    "evaluation_config": ref_manifest["eval_config"],
                    "output_dir": str(r["eval"] / code / arm / ("step-%d" % step))}
        eval_path = cdir / ("%s-step-%d-eval.yaml" % (arm, step))
        write_yaml(eval_path, eval_cfg)
        pipeline_path = cdir / ("%s-step-%d-eval-pipeline.yaml" % (arm, step))
        write_yaml(pipeline_path, {"schema_version": 1,
                                  "run_id": template["run_id"] + "-step-%d-eval" % step,
                                  "stages": [{"name": "eval", "config": str(eval_path.relative_to(ROOT))}]})
        evaluations.append((step, pipeline_path))
    return train_pipeline, evaluations


def child_environment():
    env = dict(os.environ)
    env["TSL_TOOL_ENV_DISPATCHED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "third_parties/TIME/src") + os.pathsep + str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "4"
    return env


def run_pipeline(path, from_eval, logger):
    command = [str(PYTHON), "-u", "-m", "scripts.tools.run", "--config", str(path.relative_to(ROOT))]
    if from_eval:
        command += ["--from-stage", "eval"]
    logger.info("stage=pipeline-start config=%s from_eval=%s", path, from_eval)
    proc = subprocess.Popen(command, cwd=ROOT, env=child_environment(), start_new_session=True)
    ACTIVE.append(proc)
    try:
        code = proc.wait()
    finally:
        ACTIVE.remove(proc)
    if code or STOPPED:
        raise RuntimeError("Pipeline failed or interrupted: " + str(path))
    logger.info("stage=pipeline-complete config=%s", path)


def run_train_parallel(pipelines, logger):
    if not pipelines:
        return
    running = []
    try:
        for arm, path in pipelines:
            command = [str(PYTHON), "-u", "-m", "scripts.tools.run", "--config", str(path.relative_to(ROOT))]
            proc = subprocess.Popen(command, cwd=ROOT, env=child_environment(), start_new_session=True)
            ACTIVE.append(proc)
            running.append((arm, proc))
            logger.info("stage=train-arm-start arm=%s pid=%d config=%s", arm, proc.pid, path)
        while running:
            if STOPPED:
                raise RuntimeError("Parallel training interrupted")
            for arm, proc in list(running):
                code = proc.poll()
                if code is None:
                    continue
                running.remove((arm, proc))
                ACTIVE.remove(proc)
                logger.info("stage=train-arm-end arm=%s code=%d", arm, code)
                if code:
                    raise RuntimeError("Parallel training failed: %s code=%d" % (arm, code))
            if running:
                time.sleep(1)
    except BaseException:
        for _arm, proc in running:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for _arm, proc in running:
            proc.wait()
            if proc in ACTIVE:
                ACTIVE.remove(proc)
        raise


def result_metric(task, path):
    if task["kind"] == "time":
        metric_path = path / task["key"] / "short" / "metrics.npz"
        key = "MASE"
    else:
        metric_path = path / "summary.csv"
        if not metric_path.is_file():
            return None, None
        with metric_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0].get("task") != "mean":
            raise ValueError("FEV summary must contain one aggregate task row")
        value = float(rows[0]["MASE"])
        if not np.isfinite(value):
            raise ValueError("FEV MASE is nonfinite")
        return value, str(metric_path)
    if not metric_path.is_file():
        return None, None
    with np.load(metric_path, allow_pickle=False) as a:
        if key not in a:
            return None, str(metric_path)
        value = float(np.nanmean(a[key]))
    if not np.isfinite(value):
        raise ValueError("TIME MASE is nonfinite")
    return value, str(metric_path)


def sync_results(tasks, validation=False):
    if validation:
        return
    r = roots(False)
    labels = json.loads((DOC_ROOT / "labels.json").read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        for arm in ARMS:
            for step in ((4000,) if arm == "e0-rand" else (1000, 2000, 4000)):
                out = r["eval"] / task["code"] / arm / ("step-%d" % step)
                metric, path = result_metric(task, out) if (out / ".complete").is_file() else (None, None)
                state = labels["completed"] if metric is not None else labels["pending"]
                rows.append((task["code"], task["kind"], str(task["key"]), arm, step, state, metric, path))
    exp_lines = [labels["title"], "", labels["notice"], "",
                 labels["result_header"], "| --- | --- | --- | ---: | --- | ---: | --- |"]
    index_lines = []
    global_lines = []
    for code, kind, key, arm, step, state, metric, path in rows:
        m = "-" if metric is None else "%.8g" % metric
        p = "-" if path is None else path
        exp_lines.append("| %s | %s | %s | %d | %s | %s | %s |" % (code, kind, arm, step, state, m, p))
        link = "docs/exp/main-crb-supp-chronos2-small/results.md"
        index_lines.append("| main-crb-supp-chronos2-small | %s | %s | %s step %d | %s | %s | %s |" % (code, arm, key, step, state, m, link))
        global_lines.append("| main-crb-supp-chronos2-small | %s | %s step %d | %s | %s | %s |" % (code, arm, step, state, m, p))
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = DOC_ROOT / "results.md"
    begin_result, end_result = "<!-- main-crb-supp-chronos2-small-results begin -->", "<!-- main-crb-supp-chronos2-small-results end -->"
    table = begin_result + "\n" + "\n".join(exp_lines) + "\n" + end_result
    if result_path.is_file():
        current = result_path.read_text(encoding="utf-8")
        if begin_result in current and end_result in current:
            current = current[:current.index(begin_result)] + table + current[current.index(end_result) + len(end_result):]
        else:
            current = current.rstrip() + "\n\n" + table + "\n"
    else:
        current = table + "\n"
    result_path.write_text(current, encoding="utf-8")
    for name, header, new_rows in (
        ("EXP.md", [labels["exp_header"], "| --- | --- | --- | --- | --- | --- | --- |"], index_lines),
        ("RESULTS.md", [labels["global_header"], "| --- | --- | --- | --- | ---: | --- |"], global_lines)):
        path = ROOT / "docs" / name
        original = path.read_text(encoding="utf-8")
        begin, end = "<!-- main-crb-supp-chronos2-small begin -->", "<!-- main-crb-supp-chronos2-small end -->"
        body = begin + "\n" + "\n".join(header + new_rows) + "\n" + end
        if begin in original and end in original:
            prefix = original[:original.index(begin)]
            suffix = original[original.index(end) + len(end):]
            original = prefix + body + suffix
        else:
            original = original.rstrip() + "\n\n" + body + "\n"
        path.write_text(original, encoding="utf-8")


def signal_children(_sig, _frame):
    global STOPPED
    STOPPED = True
    for proc in list(ACTIVE):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass


def process_task(task, validation, pool, logger):
    r = roots(validation)
    code = task["code"]
    logger.info("stage=task-start task=%s kind=%s key=%s", code, task["kind"], task["key"])
    manifest = prepare_reference(task, validation, logger)
    training = []
    evaluations = {}
    for arm in ARMS:
        if arm.startswith(("e1", "e2", "e3", "e4")):
            prepare_arm_payload(task, arm, manifest, pool, validation, logger)
        elif arm == "e0-rand":
            rand = ROOT / "outputs/selection/main-crb-v0/q16-src256-eight-task-resample/rand/shared/resampling_payload.npz"
            dst = r["selection"] / code / arm / "resampling_payload.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(rand.resolve())
        train_pipeline, eval_pipelines = stage_config(task, arm, manifest, validation, 3 if validation else 4000)
        if not (r["train"] / code / arm / ".complete").is_file():
            training.append((arm, train_pipeline))
        evaluations[arm] = eval_pipelines
    run_train_parallel(training, logger)
    for arm in ARMS:
        for step, eval_pipeline in evaluations[arm]:
            done = r["eval"] / code / arm / ("step-%d" % step) / ".complete"
            if not done.is_file():
                run_pipeline(eval_pipeline, False, logger)
            if not done.is_file():
                raise RuntimeError("Evaluation completion marker missing: " + str(done))
            metric, path = result_metric(task, done.parent)
            if metric is None or path is None:
                raise RuntimeError("Completed evaluation lacks primary metric: " + str(done.parent))
            logger.info("stage=arm-result task=%s arm=%s step=%d mase=%s metrics=%s",
                        code, arm, step, metric, path)
            sync_results(TASKS, validation)
        (r["eval"] / code / arm / ".complete").write_text("complete\n", encoding="ascii")
    (r["eval"] / code / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=task-complete task=%s", code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "validate", "run"), required=True)
    args = parser.parse_args()
    validation = args.mode == "validate"
    logger = configure_logger("main_crb_supp_queue", "main-crb-supp-chronos2-small", args.mode, "queue", enable_heartbeat=True)
    signal.signal(signal.SIGINT, signal_children)
    signal.signal(signal.SIGTERM, signal_children)
    started = time.monotonic()
    logger.info("stage=config mode=%s tasks=%d arms=%d model=chronos-2-small source=src-256 source_q=16 reference_q=8 steps=%d batch=32 seed=0",
                args.mode, len(TASKS), len(ARMS), 3 if validation else 4000)
    pool = prepare_source(logger)
    prepare_fev_task_file()
    if args.mode == "prepare":
        for task in TASKS:
            prepare_eval(task, False)
        sync_results(TASKS)
        return
    if validation:
        try:
            process_task(TASKS[0], True, pool, logger)
        finally:
            r = roots(True)
            for kind in ("reference", "cache", "selection", "train", "eval", "configs"):
                path = r[kind]
                if path.exists():
                    shutil.rmtree(path)
            logger.info("stage=validation-artifacts-cleaned")
    else:
        sync_results(TASKS)
        for task in TASKS:
            task_done = roots(False)["eval"] / task["code"] / ".complete"
            if task_done.is_file():
                logger.info("stage=task-skip task=%s reason=complete-marker", task["code"])
                continue
            process_task(task, False, pool, logger)
        (roots(False)["eval"] / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete mode=%s elapsed=%s output=%s", args.mode, format_duration(time.monotonic() - started), roots(validation)["eval"])


if __name__ == "__main__":
    main()
