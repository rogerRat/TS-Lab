"""Save BF16 full gradients, then continuously replay CRB budget searches on CPU."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from configs.logger import configure_logger, format_bytes, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig, CRBEngine
from rage.experiments.data import _read_series, load_bundle_index

LOGGER = logging.getLogger("v0.crb_reconst_test")
PARAM_COUNT = 8_394_496
CONTEXT = 512
PREDICTION = 64
WINDOW = CONTEXT + PREDICTION
DISK_RESERVE_BYTES = 1 * 1024 ** 3
METRIC_CHUNK_ROWS = 4
GRADIENT_FLUSH_ROWS = 16
STOP_REQUESTED = False
STOP_PATH: Path | None = None


def stop_requested() -> bool:
    global STOP_REQUESTED
    if STOP_PATH is not None and STOP_PATH.is_file():
        STOP_REQUESTED = True
    return STOP_REQUESTED


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                         encoding="ascii")
    temporary.replace(path)


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="ascii"))
    if config.get("experiment") != "v0-crb-reconst-test":
        raise ValueError("unexpected experiment config")
    if config.get("gradient_dtype") != "bfloat16":
        raise ValueError("this experiment requires BF16 full-gradient storage")
    if not 1 <= int(config.get("trajectory_count", 0)) <= 20:
        raise ValueError("trajectory_count must be between one and twenty")
    denominators = tuple(int(value) for value in config.get("budget_denominators", []))
    if denominators != (1, 4, 16, 64, 128, 256, 512, 576):
        raise ValueError("unexpected standard total-budget denominator set")
    return config


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_space(path: Path, needed: int, stage: str) -> None:
    available = free_bytes(path)
    required = int(needed) + DISK_RESERVE_BYTES
    if available < required:
        raise OSError(
            "insufficient data-disk space before %s: available=%s required_with_reserve=%s" %
            (format_bytes(available), format_bytes(required), stage)
        )


def load_trajectories(config: dict, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    rows = manifest.get("trajectories", [])
    if len(rows) < int(config["trajectory_count"]):
        raise ValueError("v0-grad-mech-test input manifest has fewer than ten trajectories")
    bundle = load_bundle_index(Path(config["bundle_data"]) / "load")
    selected = []
    source_ordinals = [int(value) for value in config.get(
        "trajectory_ordinals", range(int(config["trajectory_count"])))
    ]
    if len(source_ordinals) != int(config["trajectory_count"]):
        raise ValueError("trajectory ordinal list length differs from trajectory_count")
    if len(source_ordinals) != len(set(source_ordinals)):
        raise ValueError("trajectory ordinals must be unique")
    cache_ordinals = [int(value) for value in config.get(
        "cache_ordinals", range(len(source_ordinals)))
    ]
    if len(cache_ordinals) != len(source_ordinals):
        raise ValueError("cache ordinal list length differs from trajectory_count")
    for ordinal, source_ordinal in enumerate(source_ordinals):
        if not 0 <= source_ordinal < len(rows):
            raise ValueError("trajectory ordinal is outside the input manifest")
        row = rows[source_ordinal]
        source_row = int(row["source_row"])
        if bool(bundle.index["dropped"][source_row]):
            raise ValueError("manifest trajectory is marked dropped")
        values = np.asarray(_read_series(bundle, source_row), dtype=np.float32)
        if not np.isfinite(values).all() or len(values) != int(row["length"]):
            raise ValueError("source trajectory no longer matches the fixed manifest")
        if str(bundle.meta["series_id"][source_row]) != str(row["series_id"]):
            raise ValueError("source trajectory ID differs from the fixed manifest")
        if len(values) < WINDOW:
            raise ValueError("trajectory has no legal window")
        selected.append({
            "ordinal": ordinal,
            "cache_ordinal": cache_ordinals[ordinal],
            "source_ordinal": source_ordinal,
            "source_row": source_row,
            "series_id": str(row["series_id"]),
            "values": values,
            "window_count": len(values) - WINDOW + 1,
        })
    LOGGER.info("Fixed input loaded: trajectories=%d windows=%d",
                len(selected), sum(item["window_count"] for item in selected))
    return selected


def load_bf16_engine(checkpoint: Path):
    from rage.engines import ChronosT5Config, ChronosT5Engine
    engine = ChronosT5Engine(ChronosT5Config(
        model_path=str(checkpoint),
        context_length=CONTEXT,
        prediction_length=PREDICTION,
        torch_dtype="bfloat16",
        use_vmap_gradients=False,
    ), device="cuda")
    if sum(parameter.numel() for parameter in engine._parameters) != PARAM_COUNT:
        raise ValueError("loaded model parameter count differs from the pinned checkpoint")
    return engine


def compute_one_gradient(engine, window: np.ndarray) -> torch.Tensor:
    tensors = engine.batch_tensors(window.reshape(1, -1))
    output = engine.model(**engine._logits_inputs(tensors), return_dict=True)
    from rage.engines import per_sample_cross_entropy
    loss = per_sample_cross_entropy(output.logits, tensors["labels"])[0]
    gradients = torch.autograd.grad(loss, engine._parameters, allow_unused=True)
    pieces = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(engine._parameters, gradients)
    ]
    flat = torch.cat([piece.detach().reshape(-1) for piece in pieces])
    if flat.numel() != PARAM_COUNT:
        raise ValueError("flattened gradient dimension differs from the pinned model")
    return flat.to(dtype=torch.bfloat16).detach()


def open_bf16_map(path: Path, rows: int):
    elements = int(rows) * PARAM_COUNT
    expected_bytes = elements * 2
    if path.exists() and path.stat().st_size != expected_bytes:
        raise ValueError("gradient map has unexpected size: %s" % path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(expected_bytes)
    return torch.from_file(
        str(path), shared=True, size=elements, dtype=torch.bfloat16
    ).view(rows, PARAM_COUNT)


def fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDWR)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_all_gradients(engine, trajectories, gradient_dir: Path, state_path: Path) -> None:
    gradient_dir.mkdir(parents=True, exist_ok=True)
    remaining_bytes = 0
    for item in trajectories:
        ordinal = item.get("cache_ordinal", item["ordinal"])
        marker = gradient_dir / ("trajectory-%02d.complete" % ordinal)
        if marker.is_file():
            continue
        progress_path = gradient_dir / ("trajectory-%02d.progress.json" % ordinal)
        completed_rows = 0
        if progress_path.exists():
            state = json.loads(progress_path.read_text(encoding="ascii"))
            completed_rows = int(state.get("completed_rows", 0))
        remaining_bytes += max(0, item["window_count"] - completed_rows) * PARAM_COUNT * 2
    require_space(gradient_dir, remaining_bytes, "remaining full-gradient phase")

    total_windows = sum(item["window_count"] for item in trajectories)
    progress = tqdm(total=total_windows, desc="BF16 full gradients", unit="window",
                    mininterval=10)
    for item in trajectories:
        ordinal = item.get("cache_ordinal", item["ordinal"])
        name = "trajectory-%02d.bf16" % ordinal
        path = gradient_dir / name
        marker = gradient_dir / ("trajectory-%02d.complete" % ordinal)
        progress_path = gradient_dir / ("trajectory-%02d.progress.json" % ordinal)
        n_windows = item["window_count"]
        if marker.is_file():
            progress.update(n_windows)
            continue
        state = {"completed_rows": 0}
        if progress_path.exists():
            state = json.loads(progress_path.read_text(encoding="ascii"))
        completed_rows = int(state.get("completed_rows", 0))
        if not 0 <= completed_rows <= n_windows:
            raise ValueError("invalid gradient resume cursor")
        require_space(
            gradient_dir, (n_windows - completed_rows) * PARAM_COUNT * 2,
            "remaining trajectory gradient file",
        )
        mapped = open_bf16_map(path, n_windows)
        progress.update(completed_rows)
        LOGGER.info("Gradient trajectory start: ordinal=%d/10 id=%s windows=%d dtype=BF16 bytes=%s",
                    ordinal + 1, item["series_id"], n_windows,
                    format_bytes(n_windows * PARAM_COUNT * 2))
        batch_started = time.monotonic()
        try:
            with torch.enable_grad():
                for start in range(completed_rows, n_windows):
                    if stop_requested():
                        break
                    if free_bytes(gradient_dir) <= DISK_RESERVE_BYTES:
                        raise OSError("data-disk reserve reached during gradient write")
                    gradient = compute_one_gradient(
                        engine, item["values"][start:start + WINDOW])
                    mapped[start].copy_(gradient)
                    progress.update(1)
                    if (start + 1) % GRADIENT_FLUSH_ROWS == 0 or start + 1 == n_windows:
                        del gradient
                        fsync_file(path)
                        write_json_atomic(progress_path, {
                            "trajectory_ordinal": ordinal,
                            "series_id": item["series_id"],
                            "window_count": n_windows,
                            "completed_rows": start + 1,
                            "dtype": "bfloat16",
                        })
                        LOGGER.info("Gradient checkpoint: trajectory=%d/10 rows=%d/%d elapsed=%s disk_free=%s",
                                    ordinal + 1, start + 1, n_windows,
                                    format_duration(time.monotonic() - batch_started),
                                    format_bytes(free_bytes(gradient_dir)))
            if stop_requested():
                fsync_file(path)
                write_json_atomic(progress_path, {
                    "trajectory_ordinal": ordinal,
                    "series_id": item["series_id"],
                    "window_count": n_windows,
                    "completed_rows": start + 1,
                    "dtype": "bfloat16",
                })
                write_json_atomic(state_path, {
                    "phase": "full_gradient",
                    "completed_trajectories": ordinal,
                    "stop_requested": True,
                })
                return
            fsync_file(path)
            write_json_atomic(gradient_dir / ("trajectory-%02d.json" % ordinal), {
                "trajectory_ordinal": ordinal,
                "source_row": item["source_row"],
                "series_id": item["series_id"],
                "window_count": n_windows,
                "parameter_count": PARAM_COUNT,
                "dtype": "bfloat16",
                "file": name,
            })
            marker.write_text("complete\n", encoding="ascii")
            mapped = None
            LOGGER.info("Gradient trajectory complete: ordinal=%d/10 id=%s windows=%d disk_free=%s",
                        ordinal + 1, item["series_id"], n_windows,
                        format_bytes(free_bytes(gradient_dir)))
        except OSError:
            LOGGER.exception("Gradient write stopped at trajectory=%d id=%s",
                             ordinal + 1, item["series_id"])
            write_json_atomic(state_path, {
                "phase": "full_gradient",
                "completed_trajectories": ordinal,
                "disk_free_bytes": free_bytes(gradient_dir),
                "stop_reason": "disk_limit",
            })
            return
    progress.close()
    write_json_atomic(state_path, {
        "phase": "full_gradient_complete",
        "completed_trajectories": len(trajectories),
        "window_count": total_windows,
        "dtype": "bfloat16",
    })


def uniform_indices(count: int, total: int) -> np.ndarray:
    if not 1 <= count <= total:
        raise ValueError("uniform anchor count must be between one and the number of windows")
    values = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    if len(np.unique(values)) != count:
        raise ValueError("uniform anchor rounding produced duplicate positions")
    return values


def interpolate_metrics(mapped, n_windows: int, anchors: dict[int, np.ndarray]):
    positions = np.asarray(sorted(anchors), dtype=np.int64)
    if len(positions) == 0 or positions[0] != 0 or positions[-1] != n_windows - 1:
        raise ValueError("reconstruction anchors must include both legal-start endpoints")
    l2_error = np.empty(n_windows, dtype=np.float32)
    cosine = np.empty(n_windows, dtype=np.float32)
    if len(positions) == 1:
        intervals = [(0, 0)]
    else:
        intervals = list(zip(positions[:-1], positions[1:]))
    for left, right in intervals:
        left_vector = torch.from_numpy(np.asarray(anchors[int(left)], dtype=np.float32))
        right_vector = torch.from_numpy(np.asarray(anchors[int(right)], dtype=np.float32))
        width = max(1, int(right - left))
        for lo in range(int(left), int(right) + 1, METRIC_CHUNK_ROWS):
            hi = min(int(right) + 1, lo + METRIC_CHUNK_ROWS)
            rows = torch.arange(lo, hi, dtype=torch.float32)
            alpha = ((rows - float(left)) / float(width)).unsqueeze(1)
            estimate = (1.0 - alpha) * left_vector + alpha * right_vector
            truth = mapped[lo:hi].to(dtype=torch.float32)
            delta = estimate - truth
            error = torch.linalg.vector_norm(delta, dim=1)
            truth_norm = torch.linalg.vector_norm(truth, dim=1)
            estimate_norm = torch.linalg.vector_norm(estimate, dim=1)
            dot = torch.sum(truth * estimate, dim=1)
            score = torch.where(
                (truth_norm > 0) & (estimate_norm > 0),
                dot / torch.clamp(truth_norm * estimate_norm, min=1e-30),
                torch.zeros_like(dot),
            )
            l2_error[lo:hi] = error.numpy()
            cosine[lo:hi] = score.numpy()
        del left_vector, right_vector
    return l2_error, cosine


def uniform_anchors(mapped, n_windows: int, count: int) -> dict[int, np.ndarray]:
    indices = uniform_indices(count, n_windows)
    values = mapped[torch.as_tensor(indices, dtype=torch.long)].to(dtype=torch.float32).numpy()
    return {int(index): row.copy() for index, row in zip(indices, values)}


def allocate_counts(total_budget: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total_budget < 2:
        raise ValueError("CRB coverage requires at least two endpoint gradients")
    discretionary = total_budget - 2
    shares = np.asarray(ratios, dtype=np.float64)
    shares = shares / shares.sum()
    raw = discretionary * shares
    extras = np.floor(raw).astype(np.int64)
    remainder = discretionary - int(extras.sum())
    order = np.argsort(-(raw - extras), kind="stable")
    for index in order[:remainder]:
        extras[index] += 1
    return 2 + int(extras[0]), int(extras[1]), int(extras[2])


def densities_for_counts(counts: tuple[int, int, int], total: int) -> CRBConfig:
    cover, error, split = counts
    config = CRBConfig(
        context_length=CONTEXT,
        prediction_length=PREDICTION,
        window_stride=1,
        cover_density=cover / total,
        error_density=error / total,
        split_density=split / total,
    )
    if config.budgets(total) != counts:
        raise ValueError("CRB density rounding changed the requested integer allocation")
    return config


def aggregate_metrics(rows):
    l2 = np.concatenate([row["l2"] for row in rows])
    cosine = np.concatenate([row["cosine"] for row in rows])
    return {
        "mean_l2": float(np.mean(l2, dtype=np.float64)),
        "median_l2": float(np.median(l2)),
        "p95_l2": float(np.quantile(l2, 0.95)),
        "worst_l2": float(np.max(l2)),
        "mean_cosine": float(np.mean(cosine, dtype=np.float64)),
        "median_cosine": float(np.median(cosine)),
        "p05_cosine": float(np.quantile(cosine, 0.05)),
        "worst_cosine": float(np.min(cosine)),
    }


def save_metric_file(path: Path, records, metadata: dict) -> None:
    offsets = [0]
    for record in records:
        offsets.append(offsets[-1] + len(record["l2"]))
    arrays = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "trajectory_ids": np.asarray([record["series_id"] for record in records]),
        "l2_error": np.concatenate([record["l2"] for record in records]),
        "cosine": np.concatenate([record["cosine"] for record in records]),
    }
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    write_json_atomic(path.with_suffix(".json"), metadata)


def load_metric_file(path: Path, trajectories):
    with np.load(path, allow_pickle=False) as data:
        offsets = data["offsets"]
        ids = data["trajectory_ids"]
        l2 = data["l2_error"]
        cosine = data["cosine"]
    records = []
    for i, item in enumerate(trajectories):
        if str(ids[i]) != item["series_id"]:
            raise ValueError("metric cache trajectory order mismatch")
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        records.append({
            "series_id": item["series_id"],
            "l2": l2[lo:hi].copy(),
            "cosine": cosine[lo:hi].copy(),
        })
    return records


def uniform_key(q: int, counts: tuple[int, ...]) -> str:
    return "uniform-q%02d-k%s" % (q, "-".join(str(value) for value in counts))


def evaluate_uniform(q: int, counts: tuple[int, ...], trajectories, maps, search_dir: Path):
    key = uniform_key(q, counts)
    path = search_dir / (key + ".npz")
    if path.is_file():
        records = load_metric_file(path, trajectories)
    else:
        records = []
        for item, mapped, count in zip(trajectories, maps, counts):
            if count == item["window_count"]:
                l2 = np.zeros(item["window_count"], dtype=np.float32)
                cosine = np.ones(item["window_count"], dtype=np.float32)
            else:
                anchors = uniform_anchors(mapped, item["window_count"], count)
                l2, cosine = interpolate_metrics(mapped, item["window_count"], anchors)
            records.append({"series_id": item["series_id"], "l2": l2, "cosine": cosine})
        summary = aggregate_metrics(records)
        save_metric_file(path, records, {
            "method": "baseline-uniform",
            "budget_group_denominator": q,
            "actual_gradient_counts": list(counts),
            "summary": summary,
        })
        LOGGER.info("Uniform baseline complete: q=%d actual_gradients=%d summary=%s",
                    q, sum(counts), summary)
    return key, path, records, aggregate_metrics(records)


def crb_candidate(q: int, round_index: int, sample_index: int,
                  ratios: tuple[float, float, float], trajectories, maps,
                  search_dir: Path, seen_budgets: dict):
    nominal = tuple(max(2, math.ceil(item["window_count"] / q)) for item in trajectories)
    allocations = tuple(allocate_counts(k, ratios) for k in nominal)
    budget_key = str(q) + "|" + "|".join(",".join(str(value) for value in item) for item in allocations)
    candidate_id = "q%02d-r%05d-s%02d" % (q, round_index, sample_index)
    path = search_dir / (candidate_id + ".npz")
    if budget_key in seen_budgets:
        prior = seen_budgets[budget_key]
        if prior.is_file() and prior.with_suffix(".json").is_file():
            reused = json.loads(prior.with_suffix(".json").read_text(encoding="ascii"))
            reused["duplicate_of"] = prior.stem
            reused["candidate_id"] = candidate_id
            reused["candidate_path"] = str(prior)
            reused["ratios_cover_error_split"] = list(ratios)
            reused["reused"] = True
            return reused
    if path.is_file() and path.with_suffix(".json").is_file():
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="ascii"))
        seen_budgets[budget_key] = path
        return metadata

    actual_counts = []
    rows = []
    if q == 1:
        actual_counts = list(nominal)
        for item in trajectories:
            rows.append({
                "series_id": item["series_id"],
                "l2": np.zeros(item["window_count"], dtype=np.float32),
                "cosine": np.ones(item["window_count"], dtype=np.float32),
            })
    else:
        for item, mapped, allocation in zip(trajectories, maps, allocations):
            n_windows = item["window_count"]
            config = densities_for_counts(allocation, n_windows)
            fragment = TimeSeriesFragment(
                np.zeros(len(item["values"]), dtype=np.float32),
                entity_id=item["series_id"],
                fragment_id=item["ordinal"],
            )
            def gradient_computer(_fragment, physical_starts, store=mapped):
                indices = np.asarray(physical_starts, dtype=np.int64) - 1
                index_tensor = torch.as_tensor(indices, dtype=torch.long)
                return store.index_select(0, index_tensor).to(dtype=torch.float32).numpy()

            result = CRBEngine(config, gradient_computer).process_fragment(fragment)
            actual = int(result.computed_gradients)
            if actual > max(2, math.ceil(n_windows / q)):
                raise ValueError("CRB replay exceeded the total-gradient budget")
            actual_counts.append(actual)
            anchors = result.anchors
            l2, cosine = interpolate_metrics(mapped, n_windows, anchors)
            rows.append({"series_id": item["series_id"], "l2": l2, "cosine": cosine})
            LOGGER.info("CPU replay trajectory=%s q=%d actual=%d/%d anchors=%d",
                        item["series_id"], q, actual, math.ceil(n_windows / q), len(anchors))
            del result, anchors, fragment
    actual_counts = tuple(actual_counts)
    matched_key, matched_path, baseline_rows, baseline_summary = evaluate_uniform(
        q, actual_counts, trajectories, maps, search_dir)
    crb_summary = aggregate_metrics(rows)
    metadata = {
        "candidate_id": candidate_id,
        "method": "CRB",
        "total_budget_denominator": q,
        "q_points_per_gradient": q,
        "nominal_gradient_counts": list(nominal),
        "expected_gradient_counts_before_endpoint_floor": [
            math.ceil(item["window_count"] / q) for item in trajectories
        ],
        "allocations_cover_error_split": [list(item) for item in allocations],
        "ratios_cover_error_split": list(ratios),
        "actual_gradient_counts": list(actual_counts),
        "actual_gradient_count_total": sum(actual_counts),
        "matched_uniform_baseline": matched_key,
        "matched_uniform_path": str(matched_path),
        "crb_summary": crb_summary,
        "uniform_summary": baseline_summary,
        "mean_l2_gap_crb_minus_uniform": crb_summary["mean_l2"] - baseline_summary["mean_l2"],
        "mean_cosine_gap_crb_minus_uniform": crb_summary["mean_cosine"] - baseline_summary["mean_cosine"],
        "candidate_path": str(path),
        "reused": False,
    }
    save_metric_file(path, rows, metadata)
    seen_budgets[budget_key] = path
    return metadata


def pareto_insert(front: list[dict], row: dict) -> list[dict]:
    score = row["crb_summary"]
    for old in front:
        old_score = old["crb_summary"]
        if old_score["mean_l2"] <= score["mean_l2"] and old_score["mean_cosine"] >= score["mean_cosine"]:
            return front
    kept = []
    for old in front:
        old_score = old["crb_summary"]
        dominated = score["mean_l2"] <= old_score["mean_l2"] and score["mean_cosine"] >= old_score["mean_cosine"]
        if not dominated:
            kept.append(old)
    kept.append(row)
    return kept


def uniform_simplex_ratios():
    ratios = []
    denominator = 4
    for cover_units in range(denominator, -1, -1):
        for error_units in range(denominator - cover_units, -1, -1):
            split_units = denominator - cover_units - error_units
            ratios.append((cover_units / denominator,
                           error_units / denominator,
                           split_units / denominator))
    ratios.append((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    if len(ratios) != 16 or len(set(ratios)) != 16:
        raise RuntimeError("uniform simplex design must contain 16 unique ratios")
    if any(min(ratio) < 0.0 or max(ratio) > 1.0 or abs(sum(ratio) - 1.0) > 1e-12
           for ratio in ratios):
        raise RuntimeError("uniform simplex ratio violates the unit simplex constraints")
    return ratios


def run_search(trajectories, gradient_dir: Path, search_dir: Path,
               progress_path: Path, config: dict):
    search_dir.mkdir(parents=True, exist_ok=True)
    maps = []
    for item in trajectories:
        maps.append(open_bf16_map(
            gradient_dir / ("trajectory-%02d.bf16" % item.get("cache_ordinal", item["ordinal"])),
            item["window_count"],
        ))
    eval_log = search_dir / "evaluations.jsonl"
    completed = {}
    seen_budgets = {}
    denominators = tuple(int(value) for value in config["budget_denominators"])
    fronts = {q: [] for q in denominators}
    if eval_log.exists():
        for line in eval_log.read_text(encoding="ascii").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed[row["candidate_id"]] = row
            if row.get("candidate_path") and row.get("allocations_cover_error_split"):
                allocations = row["allocations_cover_error_split"]
                key = str(row["q_points_per_gradient"]) + "|" + "|".join(",".join(str(v) for v in values) for values in allocations)
                seen_budgets[key] = Path(row["candidate_path"])
            if row.get("crb_summary"):
                q = int(row["q_points_per_gradient"])
                fronts[q] = pareto_insert(fronts[q], row)
    round_index = 1
    if progress_path.exists():
        prior = json.loads(progress_path.read_text(encoding="ascii"))
        prior_round = int(prior.get("round", 0))
        round_index = prior_round + 1 if prior.get("round_complete") else max(1, prior_round)
    LOGGER.info("CPU replay search starting: budget_denominators=%s search_strategy=%s per_group=baseline_uniform+16_configs",
                list(denominators), config.get("search_strategy", "uniform_simplex_lattice_d4_plus_centroid"))
    while not stop_requested():
        round_started = time.monotonic()
        for q in denominators:
            if stop_requested():
                break
            if free_bytes(search_dir) <= DISK_RESERVE_BYTES:
                raise OSError("data-disk reserve reached before the next budget group")
            nominal = tuple(max(2, math.ceil(item["window_count"] / q)) for item in trajectories)
            uniform_key_name = uniform_key(q, nominal)
            evaluate_uniform(q, nominal, trajectories, maps, search_dir)
            ratios = uniform_simplex_ratios()
            with tqdm(total=16, desc="CPU replay q=%d round=%d" % (q, round_index),
                      unit="config", mininterval=10) as progress:
                for sample_index, ratio in enumerate(ratios):
                    if stop_requested():
                        break
                    if free_bytes(search_dir) <= DISK_RESERVE_BYTES:
                        raise OSError("data-disk reserve reached during parameter search")
                    candidate_id = "q%02d-r%05d-s%02d" % (q, round_index, sample_index)
                    if candidate_id in completed:
                        progress.update(1)
                        continue
                    row = crb_candidate(
                        q, round_index, sample_index, ratio, trajectories, maps,
                        search_dir, seen_budgets,
                    )
                    row["candidate_id"] = candidate_id
                    row["q_points_per_gradient"] = q
                    row["round"] = round_index
                    row["sample_index"] = sample_index
                    row.setdefault("ratios_cover_error_split", list(ratio))
                    if not row.get("reused") and row.get("crb_summary"):
                        fronts[q] = pareto_insert(fronts[q], row)
                    with eval_log.open("a", encoding="ascii") as handle:
                        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    completed[candidate_id] = row
                    write_json_atomic(search_dir / ("pareto-q%02d.json" % q), {
                        "q_points_per_gradient": q,
                        "metric_directions": {"mean_l2": "minimize", "mean_cosine": "maximize"},
                        "front": fronts[q],
                    })
                    progress.update(1)
                    write_json_atomic(progress_path, {
                        "phase": "cpu_replay_search",
                        "round": round_index,
                        "round_complete": False,
                        "current_budget_denominator": q,
                        "current_candidate": candidate_id,
                        "candidate_count": len(completed),
                        "disk_free_bytes": free_bytes(search_dir),
                        "stop_requested": STOP_REQUESTED,
                    })
        if STOP_REQUESTED:
            break
        write_json_atomic(progress_path, {
            "phase": "cpu_replay_search",
            "round": round_index,
            "round_complete": True,
            "candidate_count": len(completed),
            "elapsed_round_seconds": time.monotonic() - round_started,
            "disk_free_bytes": free_bytes(search_dir),
        })
        LOGGER.info("Search round complete: round=%d candidates=%d elapsed=%s disk_free=%s",
                    round_index, len(completed),
                    format_duration(time.monotonic() - round_started),
                    format_bytes(free_bytes(search_dir)))
        max_rounds = int(config.get("max_rounds", 0))
        if max_rounds > 0 and round_index >= max_rounds:
            LOGGER.info("Configured round limit reached: max_rounds=%d", max_rounds)
            break
        round_index += 1


def run(config_path: Path) -> None:
    config = read_config(config_path)
    predecessor = output_path("train", "outputs/train/v0-grad-mech-test")
    predecessor_complete = (predecessor / ".complete").is_file()
    if not predecessor_complete:
        if not config.get("allow_cached_gradients_before_predecessor_complete", False):
            raise RuntimeError("v0-grad-mech-test must complete before CRB reconstruction begins")
        LOGGER.info("Starting CRB preparation while the source gradient probe continues")
    output = output_path("train", config.get(
        "output_root", "outputs/train/v0-crb-reconst-test"))
    gradient_dir = PROJECT_ROOT / config.get(
        "full_gradient_output", "outputs/train/v0-crb-reconst-test/gradients")
    search_dir = output / "search"
    output.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "logs" / "v0-crb-reconst-test"
    log_dir.mkdir(parents=True, exist_ok=True)
    global STOP_PATH
    STOP_PATH = PROJECT_ROOT / config["stop_sentinel"]
    manifest_path = predecessor / "input_manifest.json"
    trajectories = load_trajectories(config, manifest_path)
    if not predecessor_complete:
        while not all((gradient_dir / ("trajectory-%02d.complete" % item.get("cache_ordinal", item["ordinal"]))).is_file()
                      for item in trajectories):
            if stop_requested():
                LOGGER.info("Stop requested while waiting for retained gradient caches")
                return
            alive = subprocess.run(["tmux", "has-session", "-t", "v0-grad-run"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not alive:
                raise RuntimeError("source probe ended before all retained gradients were complete")
            LOGGER.info("Waiting for retained gradient caches: %d/%d complete",
                        sum((gradient_dir / ("trajectory-%02d.complete" % item.get("cache_ordinal", item["ordinal"]))).is_file()
                            for item in trajectories), len(trajectories))
            time.sleep(30)
        LOGGER.info("All retained BF16 gradients are complete; starting CPU CRB replay")
    total_windows = sum(item["window_count"] for item in trajectories)
    expected_bytes = total_windows * PARAM_COUNT * 2
    missing_bytes = sum(item["window_count"] * PARAM_COUNT * 2 for item in trajectories
                        if not (gradient_dir / ("trajectory-%02d.complete" % item.get("cache_ordinal", item["ordinal"]))).is_file())
    LOGGER.info("Configuration: trajectories=%d windows=%d parameters=%d dtype=BF16 full_gradient_bytes=%s budget_denominators=%s search_strategy=%s samples_per_budget=16",
                len(trajectories), total_windows, PARAM_COUNT, format_bytes(expected_bytes),
                list(config["budget_denominators"]),
                config.get("search_strategy", "uniform_simplex_lattice_d4_plus_centroid"))
    require_space(output, missing_bytes, "remaining BF16 gradient cache")
    checkpoint = predecessor / "chronos-tiny-warmup-100"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError("fixed post-warm-up Chronos checkpoint is missing")
    write_json_atomic(output / "input_manifest.json", {
        "source_manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "trajectory_count": len(trajectories),
        "window_count": total_windows,
        "parameter_count": PARAM_COUNT,
        "gradient_dtype": "bfloat16",
        "budget_denominators": list(config["budget_denominators"]),
        "search_strategy": config.get("search_strategy", "uniform_simplex_lattice_d4_plus_centroid"),
        "max_rounds": int(config.get("max_rounds", 0)),
        "trajectories": [{
            "ordinal": item["ordinal"],
            "cache_ordinal": item.get("cache_ordinal", item["ordinal"]),
            "source_ordinal": item["source_ordinal"],
            "source_row": item["source_row"],
            "series_id": item["series_id"],
            "window_count": item["window_count"],
        } for item in trajectories],
    })

    gradient_dir.mkdir(parents=True, exist_ok=True)
    complete = all((gradient_dir / ("trajectory-%02d.complete" % item.get("cache_ordinal", item["ordinal"]))).is_file()
                   for item in trajectories)
    if not complete:
        engine = load_bf16_engine(checkpoint)
        LOGGER.info("Full-gradient stage starting: checkpoint=%s dtype=BF16", checkpoint)
        save_all_gradients(engine, trajectories, gradient_dir, output / "progress.json")
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        complete = all((gradient_dir / ("trajectory-%02d.complete" % item.get("cache_ordinal", item["ordinal"]))).is_file()
                       for item in trajectories)
        if not complete:
            if STOP_REQUESTED:
                LOGGER.info("Stop requested during full-gradient stage; partial rows retained for resume")
                return
            LOGGER.warning("Full-gradient stage stopped before all selected trajectories were complete")
            return
    if config.get("mark_gradient_dir_complete", True):
        (gradient_dir / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("Full-gradient stage complete: trajectories=%d windows=%d dtype=BF16 output=%s",
                len(trajectories), total_windows, gradient_dir)

    requested_threads = int(config.get("cpu_threads", 32))
    torch.set_num_threads(max(1, min(requested_threads, os.cpu_count() or 4)))
    try:
        run_search(trajectories, gradient_dir, search_dir, output / "progress.json", config)
    except OSError as exc:
        if "data-disk reserve reached" not in str(exc):
            raise
        LOGGER.warning("Search reached the data-disk reserve: %s", exc)
    max_rounds = int(config.get("max_rounds", 0))
    search_state = (json.loads((search_dir / "progress.json").read_text(encoding="ascii"))
                    if (search_dir / "progress.json").is_file() else {})
    round_limit_reached = (max_rounds > 0 and search_state.get("round_complete")
                           and int(search_state.get("round", 0)) >= max_rounds)
    reason = ("user_stop" if stop_requested() else
              ("round_limit" if round_limit_reached else "disk_limit"))
    write_json_atomic(output / "search_termination.json", {
        "reason": reason,
        "candidate_count": sum(1 for line in (search_dir / "evaluations.jsonl").read_text(encoding="ascii").splitlines() if line.strip())
        if (search_dir / "evaluations.jsonl").exists() else 0,
        "disk_free_bytes": free_bytes(output),
        "full_gradients_retained": True,
    })
    if reason == "disk_limit":
        (output / ".stopped-disk-limit").write_text("stopped near data-disk exhaustion\n", encoding="ascii")
    elif reason == "user_stop":
        (output / ".stopped-user-request").write_text("stopped by STOP sentinel or signal\n", encoding="ascii")
    else:
        (output / ".stopped-round-limit").write_text("completed configured search rounds\n", encoding="ascii")
    (output / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("Search safely terminated: reason=%s full_gradients_retained=true output=%s",
                reason, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rage/v0-crb-reconst-test.json"))
    args = parser.parse_args()
    configure_logger(
        "v0.crb_reconst_test", "v0-crb-reconst-test", "experiment",
        "BF16-GRADIENT-CACHE",
    )
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    run(args.config)


if __name__ == "__main__":
    main()
