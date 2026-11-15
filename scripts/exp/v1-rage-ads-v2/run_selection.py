"""Run task-local ADS-VIC and baseline anchor selection for v2."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from experiment_config import ARMS, TASKS
from rage.ads import SourceProbabilitySampler, normalize_sampling_probabilities
from rage.ads import spread_anchor_scores, uniform_anchor_positions
from rage.core import TimeSeriesFragment
from rage.experiments.development_inputs import load_fragments

ROOT = PROJECT_ROOT
SOURCE_CONFIG = ROOT / "configs/data/src-256.yaml"
SOURCE_DATA = ROOT / "data/src/shard_dataset/src-256"
REFERENCE_ROOT = ROOT / "data/ref/shard_dataset/v1-rage-ads-rolling"
OUTPUT_ROOT = ROOT / "outputs/selection/v1-rage-ads-v2"
GRAD_ROOT = ROOT / "outputs/train/v1-rage-ads/rolling/gradients"
CONTEXT, PREDICTION, Q0, THRESHOLD, H = 480, 96, 32, 0.1, 4
WINDOW_BATCH, VJP_BATCH, PROJECTION_DIM, PROJECTION_SEED = 32, 32, 4096, 42


def log(logger, stage, **fields):
    logger.info("stage=%s %s", stage, " ".join(f"{key}={value}" for key, value in fields.items()))


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    os.replace(temporary, path)


def load_sources():
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="ascii"))
    values = np.load(SOURCE_DATA / "values.npy", mmap_mode="r")
    offsets = np.asarray(np.load(SOURCE_DATA / "offsets.npy", mmap_mode="r"), dtype=np.int64)
    fragments = []
    legal = []
    for row in config["trajectories"]:
        fid = int(row["ordinal"])
        start, end = int(offsets[fid]), int(offsets[fid + 1])
        if end - start != int(row["length"]):
            raise ValueError(f"source offset mismatch for fragment={fid}")
        fragment = TimeSeriesFragment(np.asarray(values[start:end], dtype=np.float32).copy(),
                                      entity_id=str(row["series_id"]), fragment_id=fid)
        count = int(row["length"]) - CONTEXT - PREDICTION + 1
        if count <= 0:
            raise ValueError(f"source trajectory has no legal windows: {fid}")
        fragments.append(fragment)
        legal.append(count)
    if len(fragments) != 256 or len(offsets) != 257:
        raise ValueError("src-256 manifest or packed array count mismatch")
    return fragments, np.asarray(legal, dtype=np.int64)


def normalize_rows(matrix):
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] > 0)


def grid_region(left, right, spacing):
    length = right - left + 1
    positions = {left, right}
    positions.update(left + k * spacing - 1 for k in range(1, length // spacing + 1))
    return {int(value) for value in positions if left <= value <= right}


def get_new_gradients(states, plans, probe, side, logger, task, round_id):
    added = 0
    for index, fragment in enumerate(states):
        positions = sorted(plans[index] - set(fragment["vectors"]))
        if not positions:
            continue
        starts = np.asarray(positions, dtype=np.int64)
        windows = np.stack([fragment["fragment"].extract_window(int(start) + 1, CONTEXT, PREDICTION)
                            for start in starts])
        _loss, vectors = probe.batch_gradient_sketch(windows, batch_size=WINDOW_BATCH)
        for position, vector in zip(positions, vectors):
            fragment["vectors"][int(position)] = np.asarray(vector, dtype=np.float32)
        added += len(positions)
        log(logger, "gradient-progress", task=task, side=side, round=round_id,
            fragment=index, completed=len(positions), total=added)
    return added


def _similarity_hits_cpu(reference, source, threshold=THRESHOLD):
    ref_rows = [(owner, pos, vec) for owner, state in enumerate(reference)
                for pos, vec in sorted(state["vectors"].items())]
    src_rows = [(owner, pos, vec) for owner, state in enumerate(source)
                for pos, vec in sorted(state["vectors"].items())]
    ref_norms = np.asarray([np.linalg.norm(row[2]) for row in ref_rows], dtype=np.float32)
    src_norms = np.asarray([np.linalg.norm(row[2]) for row in src_rows], dtype=np.float32)
    ref_hit = set()
    src_hit = set()
    source_max = np.full(len(src_rows), -np.inf, dtype=np.float32)
    for left in range(0, len(ref_rows), 128):
        ref_end = min(left + 128, len(ref_rows))
        rblock = np.stack([row[2] for row in ref_rows[left:ref_end]]).astype(np.float32, copy=False)
        rblock /= np.maximum(ref_norms[left:ref_end, None], 1e-30)
        for right in range(0, len(src_rows), 2048):
            src_end = min(right + 2048, len(src_rows))
            sblock = np.stack([row[2] for row in src_rows[right:src_end]]).astype(np.float32, copy=False)
            sblock /= np.maximum(src_norms[right:src_end, None], 1e-30)
            block = rblock @ sblock.T
            source_max[right:src_end] = np.maximum(source_max[right:src_end], block.max(axis=0))
            rows, columns = np.where(block > threshold)
            ref_hit.update((ref_rows[left + int(row)][0], ref_rows[left + int(row)][1])
                           for row in np.unique(rows))
            src_hit.update((src_rows[right + int(column)][0], src_rows[right + int(column)][1])
                           for column in np.unique(columns))
    scores = {}
    for index, row in enumerate(src_rows):
        scores.setdefault(row[0], {})[row[1]] = float(source_max[index])
    return ref_hit, src_hit, scores, len(ref_rows) * len(src_rows)



def _similarity_hits_cuda(reference, source, threshold=THRESHOLD):
    import torch

    ref_rows = [(owner, pos, vec) for owner, state in enumerate(reference)
                for pos, vec in sorted(state["vectors"].items())]
    src_rows = [(owner, pos, vec) for owner, state in enumerate(source)
                for pos, vec in sorted(state["vectors"].items())]
    if not ref_rows or not src_rows:
        raise ValueError("similarity inputs must be non-empty")
    dimension = int(np.asarray(ref_rows[0][2]).size)
    if any(np.asarray(row[2]).size != dimension for row in ref_rows + src_rows):
        raise ValueError("gradient vector dimensions differ")
    required = (len(ref_rows) + len(src_rows)) * dimension * np.dtype(np.float32).itemsize
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    reserve_bytes = 2 * 1024**3 + 1024 * 8192 * 4
    if required + reserve_bytes > free_bytes:
        raise MemoryError("CUDA similarity matrices exceed the guarded free-memory budget")

    ref_values = np.stack([row[2] for row in ref_rows]).astype(np.float32, copy=False)
    src_values = np.stack([row[2] for row in src_rows]).astype(np.float32, copy=False)
    ref_norms = np.linalg.norm(ref_values, axis=1)
    src_norms = np.linalg.norm(src_values, axis=1)
    np.divide(ref_values, ref_norms[:, None], out=ref_values,
              where=ref_norms[:, None] > 0)
    np.divide(src_values, src_norms[:, None], out=src_values,
              where=src_norms[:, None] > 0)

    device = torch.device("cuda")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref_tensor = torch.from_numpy(ref_values).to(device=device)
        src_tensor = torch.from_numpy(src_values).to(device=device)
        del ref_values, src_values
        ref_hit = torch.zeros(len(ref_rows), dtype=torch.bool, device=device)
        src_hit = torch.zeros(len(src_rows), dtype=torch.bool, device=device)
        source_max = torch.full((len(src_rows),), -torch.inf, dtype=torch.float32,
                                device=device)
        for left in range(0, len(ref_rows), 1024):
            ref_end = min(left + 1024, len(ref_rows))
            rblock = ref_tensor[left:ref_end]
            for right in range(0, len(src_rows), 8192):
                src_end = min(right + 8192, len(src_rows))
                block = torch.matmul(rblock, src_tensor[right:src_end].T)
                hits = block > threshold
                ref_hit[left:ref_end] |= hits.any(dim=1)
                src_hit[right:src_end] |= hits.any(dim=0)
                source_max[right:src_end] = torch.maximum(
                    source_max[right:src_end], block.max(dim=0).values)
        ref_flags = ref_hit.cpu().numpy()
        src_flags = src_hit.cpu().numpy()
        maxima = source_max.cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    ref_hits = {(row[0], row[1]) for row, hit in zip(ref_rows, ref_flags) if hit}
    src_hits = {(row[0], row[1]) for row, hit in zip(src_rows, src_flags) if hit}
    scores = {}
    for row, value in zip(src_rows, maxima):
        scores.setdefault(row[0], {})[row[1]] = float(value)
    return ref_hits, src_hits, scores, len(ref_rows) * len(src_rows)


def similarity_hits(reference, source, threshold=THRESHOLD):
    import torch

    if not torch.cuda.is_available():
        return _similarity_hits_cpu(reference, source, threshold)
    try:
        return _similarity_hits_cuda(reference, source, threshold)
    except (MemoryError, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
        return _similarity_hits_cpu(reference, source, threshold)


def refine_states(states, hits, spacing):
    hits_by_owner = {}
    for owner, position in hits:
        hits_by_owner.setdefault(owner, set()).add(position)
    plans = []
    for owner, state in enumerate(states):
        anchors = sorted(state["vectors"])
        hit_positions = hits_by_owner.get(owner, set())
        indices = [index for index, position in enumerate(anchors)
                   if position in hit_positions]
        if not indices:
            plans.append(set())
            continue
        runs = []
        first = previous = indices[0]
        for index in indices[1:]:
            if index != previous + 1:
                runs.append((first, previous))
                first = index
            previous = index
        runs.append((first, previous))
        plan = set()
        for first, last in runs:
            plan.update(grid_region(anchors[first], anchors[last], spacing))
        plans.append(plan)
    return plans


def descriptor(states, scores):
    trajectories = []
    mass = 0.0
    for owner, state in enumerate(states):
        weights = spread_anchor_scores(state["count"], scores.get(owner, {}), H)
        mass += float(weights.sum())
        trajectories.append({
            "fragment_id": owner, "first_start": 0, "stride": 1,
            "candidate_count": state["count"],
            "anchor_positions": [int(position) for position in scores.get(owner, {})],
            "anchor_scores": [float(value) for value in scores.get(owner, {}).values()],
        })
    return {"schema": 1, "mode": "anchor_neighbor_max_linear" if mass > 0 else "uniform",
            "decay_distance": H, "normalization_mass": mass,
            "population": sum(state["count"] for state in states),
            "probability_fallback": mass <= 0, "trajectories": trajectories}


def task_state(fragments, legal, spacing):
    states = []
    plans = []
    for fragment, count in zip(fragments, legal):
        positions = uniform_anchor_positions(int(count), spacing)
        states.append({"fragment": fragment, "count": int(count),
                       "vectors": {}, "legal_positions": positions})
        plans.append(set(positions))
    return states, plans


def run_task(task, sources, legal, probe, identity, logger):
    reference_dir = REFERENCE_ROOT / task
    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="ascii"))
    if not (reference_dir / ".complete").is_file() or manifest.get("answer_values_included") is not False:
        raise ValueError(f"reference input is incomplete or includes answers: {task}")
    references = load_fragments(reference_dir / "references.npz")
    ref_legal = np.asarray([len(item.valid_starts(CONTEXT, PREDICTION)) for item in references], dtype=np.int64)
    if np.any(ref_legal <= 0):
        raise ValueError(f"reference trajectory has no legal windows: {task}")
    source, source_plans = task_state(sources, legal, Q0)
    reference, reference_plans = task_state(references, ref_legal, Q0)
    start = time.monotonic()
    get_new_gradients(source, source_plans, probe, "source", logger, task, 0)
    get_new_gradients(reference, reference_plans, probe, "reference", logger, task, 0)
    ref_hit, src_hit, ads_scores, pair_count = similarity_hits(reference, source)
    ads_ref_count = sum(len(state["vectors"]) for state in reference)
    ads_src_count = sum(len(state["vectors"]) for state in source)
    for round_id, spacing in enumerate((8,), start=1):
        source_plans = refine_states(source, src_hit, spacing)
        reference_plans = refine_states(reference, ref_hit, spacing)
        new_src = get_new_gradients(source, source_plans, probe, "source", logger, task, round_id)
        new_ref = get_new_gradients(reference, reference_plans, probe, "reference", logger, task, round_id)
        if new_src + new_ref == 0:
            break
        ref_hit, src_hit, ads_scores, pair_count = similarity_hits(reference, source)
    ads_ref_budget = sum(len(state["vectors"]) for state in reference)
    ads_src_budget = sum(len(state["vectors"]) for state in source)

    out_task = output_path("selection", OUTPUT_ROOT / task)
    out_task.mkdir(parents=True, exist_ok=True)
    ads_desc = descriptor(source, ads_scores)
    for arm in ARMS:
        arm_dir = out_task / arm
        if (arm_dir / ".complete").is_file():
            continue
        arm_dir.mkdir(parents=True, exist_ok=True)
        if arm in ("ads-vic", "uni"):
            if arm == "ads-vic":
                current_source, current_reference = source, reference
                current_scores = ads_scores
                mode = "probability_descriptor"
            else:
                def uniform_budget(states, budget):
                    minimums = np.asarray([min(2, state["count"]) for state in states], dtype=np.int64)
                    min_budget = int(minimums.sum())
                    if budget < min_budget:
                        raise ValueError("uniform budget cannot cover mandatory trajectory endpoints")
                    capacities = np.asarray([state["count"] - minimum
                                             for state, minimum in zip(states, minimums)], dtype=np.float64)
                    remaining = budget - min_budget
                    raw = (remaining * capacities / capacities.sum()
                           if capacities.sum() > 0 else np.zeros_like(capacities))
                    quotas = np.floor(raw).astype(np.int64)
                    remainder = remaining - int(quotas.sum())
                    for index in np.argsort(-(raw - quotas), kind="stable")[:remainder]:
                        quotas[index] += 1
                    plans = []
                    for state, count in zip(states, quotas + minimums):
                        plans.append(set(np.rint(np.linspace(0, state["count"] - 1, int(count))).astype(int).tolist()))
                    return plans
                current_source = [{"fragment": x["fragment"], "count": x["count"], "vectors": {}}
                                  for x in source]
                current_reference = [{"fragment": x["fragment"], "count": x["count"], "vectors": {}}
                                     for x in reference]
                source_uniform_plans = uniform_budget(current_source, ads_src_budget)
                reference_uniform_plans = uniform_budget(current_reference, ads_ref_budget)
                for state, plan, cached in zip(current_source, source_uniform_plans, source):
                    state["vectors"].update({position: cached["vectors"][position]
                                             for position in plan if position in cached["vectors"]})
                for state, plan, cached in zip(current_reference, reference_uniform_plans, reference):
                    state["vectors"].update({position: cached["vectors"][position]
                                             for position in plan if position in cached["vectors"]})
                source_uni_new = get_new_gradients(current_source, source_uniform_plans,
                                                   probe, "source", logger, task, "uni")
                reference_uni_new = get_new_gradients(current_reference, reference_uniform_plans,
                                                      probe, "reference", logger, task, "uni")
                _, _, current_scores, _ = similarity_hits(current_reference, current_source)
                mode = "probability_descriptor"
            write_json(arm_dir / "probability.json", descriptor(current_source, current_scores))
            sampling = dict(schema=1, mode=mode, descriptor_path="probability.json",
                            replacement=True, candidate_count=int(legal.sum()), seed=0)
        elif arm == "traj":
            ref_vectors = np.stack([normalize_rows(np.stack(list(state["vectors"].values()))).mean(axis=0)
                                    for state in reference])
            src_vectors = np.stack([normalize_rows(np.stack(list(state["vectors"].values()))).mean(axis=0)
                                    for state in source])
            matrix = normalize_rows(ref_vectors) @ normalize_rows(src_vectors).T
            trajectory_scores = np.maximum(matrix.max(axis=0), 0)
            probabilities, fallback = normalize_sampling_probabilities(trajectory_scores)
            sampling = dict(schema=1, mode="trajectory_uniform", fragment_ids=list(range(len(sources))),
                            trajectory_probabilities=probabilities.tolist(), replacement=True,
                            candidate_count=int(legal.sum()), seed=0)
            write_json(arm_dir / "trajectory.json", dict(scores=trajectory_scores.tolist(), fallback=fallback))
        elif arm == "orig":
            sampling = dict(schema=1, mode="uniform_all", replacement=True,
                            candidate_count=int(legal.sum()), seed=0)
        else:
            sampling = dict(schema=1, mode="reference_uniform", replacement=True,
                            candidate_count=int(sum(ref_legal)), seed=0,
                            reference_path=str(reference_dir / "references.npz"))
        write_json(arm_dir / "sampling.json", sampling)
        write_json(arm_dir / "selection.json", dict(
            schema="v1-rage-ads-v2-selection-v1", experiment="v1-rage-ads-v2",
            batch="baseline-comparison", task=task, arm=arm,
            source_pool="src-256", source_trajectories=len(sources),
            source_windows=int(legal.sum()), reference_trajectories=len(references),
            reference_manifest=str(reference_dir / "manifest.json"), model=identity,
            method=dict(q0=Q0, threshold=THRESHOLD, total_rounds=2,
                        q1=8, eta=1, neighborhood=H, aggregation="max"),
            ads_budget=dict(source=ads_src_budget, reference=ads_ref_budget),
            pair_count=pair_count, elapsed_seconds=time.monotonic() - start,
            uni_actual_new_gradients=(dict(source=source_uni_new, reference=reference_uni_new)
                                      if arm == "uni" else None)))
        (arm_dir / ".complete").write_text("complete\n", encoding="ascii")
    write_json(out_task / "manifest.json", dict(experiment="v1-rage-ads-v2",
        batch="baseline-comparison", task=task, arms=list(ARMS), source_pool="src-256",
        source_trajectories=len(sources), source_windows=int(legal.sum()),
        reference_trajectories=len(references), coarse_source_anchors=ads_src_count,
        coarse_reference_anchors=ads_ref_count, ads_source_budget=ads_src_budget,
        ads_reference_budget=ads_ref_budget, pairs_evaluated=pair_count,
        elapsed_seconds=time.monotonic() - start))
    (out_task / ".complete").write_text("complete\n", encoding="ascii")
    log(logger, "task-complete", task=task, ads_src_budget=ads_src_budget,
        ads_ref_budget=ads_ref_budget, elapsed=format_duration(time.monotonic() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    logger = configure_logger("v1_rage_ads_v2_selection", "v1-rage-ads-v2",
                              "baseline-comparison", "selection-" + args.task)
    start = time.monotonic()
    log(logger, "config", tasks=",".join(tasks), arms=",".join(ARMS),
        source_pool="src-256", q0=Q0, threshold=THRESHOLD, q1=8,
        window_batch=WINDOW_BATCH, vjp_batch=VJP_BATCH, workers=len(tasks))
    if len(tasks) == len(TASKS) and any(not (REFERENCE_ROOT / task / ".complete").is_file() for task in tasks):
        raise FileNotFoundError("one or more rolling reference inputs are incomplete")
    sources, legal = load_sources()
    sys.path.insert(0, str(ROOT / "scripts/exp/v1-crb-redo"))
    from metrics import build_gradient_probe
    model_engine, probe, identity, model_fields = build_gradient_probe()
    probe.vjp_batch_size = VJP_BATCH
    if (identity["model_id"] != "thuml/timer-base-84m"
            or probe.config.projection_dimension != PROJECTION_DIM
            or probe.config.projection_seed != PROJECTION_SEED
            or model_fields["warmup_steps"] != 1000):
        raise ValueError("warmed TimerXL gradient identity mismatch")
    for index, task in enumerate(tasks, start=1):
        run_task(task, sources, legal, probe, identity, logger)
        log(logger, "progress", completed=f"{index}/{len(tasks)}",
            eta=format_duration((time.monotonic() - start) * (len(tasks) - index) / index))
    if args.task == "all":
        (output_path("selection", OUTPUT_ROOT) / ".complete").write_text("complete\n", encoding="ascii")
    log(logger, "complete", elapsed=format_duration(time.monotonic() - start), output=OUTPUT_ROOT)


if __name__ == "__main__":
    main()
