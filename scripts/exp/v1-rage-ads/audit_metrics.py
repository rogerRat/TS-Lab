"""Audit source alignment, arm sampling, rolling reference continuity, and ADS scores."""
from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.ads import SourceProbabilitySampler, uniform_anchor_positions
from rage.experiments.development_inputs import load_fragments

ROOT = PROJECT_ROOT
for _dependency_path in (ROOT / "src", ROOT / "third_parties/fev/src",
                         ROOT / "third_parties/TIME/src"):
    if str(_dependency_path) not in sys.path:
        sys.path.insert(0, str(_dependency_path))
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ARMS = ("fullRand", "rage", "ancUni", "Traj")
HORIZONS = {"fev-09": 28, "time-01": 12, "time-02": 12, "time-03": 24,
            "time-04": 30, "time-06": 24, "time-07": 12, "time-08": 12}
CONTEXT = 480
SELECTOR_HORIZON = 96
PROJECTION_DIMENSION = 4096
PROJECTION_SEED = 42
SAMPLE_COUNT = 4000
AUDIT_SEED = 42
QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling"
GRADIENT_ROOT = ROOT / "outputs/train/v1-rage-ads/rolling/gradients"
INPUT_ROOT = ROOT / "data/ref/shard_dataset/v1-rage-ads-rolling"
SOURCE_DATA = ROOT / "data/src/shard_dataset/src-96"
SOURCE_CACHE_ROOT = ROOT / "outputs/train/v1-crb-redo/grad-prepare/gradients/timerxl-84m"
SOURCE_UNIT_CACHE_ROOT = ROOT / "outputs/selection/v1-rage-ads/parallelism/source-q32-unit-cache"
WARMUP_ROOT = ROOT / "outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000"
FEV_TASK_FILE = ROOT / "configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml"
TIME_CONFIGS = {task: ROOT / ("configs/eval/time/v1-rage-ads-" + task + ".yaml")
                for task in TASKS if task.startswith("time-")}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


def upsert_jsonl(path: Path, row: dict, keys: tuple[str, ...]) -> None:
    lock_path = path.with_name(path.name + ".lock")
    temporary = path.with_name(path.name + ".partial.%d" % os.getpid())
    with lock_path.open("a", encoding="ascii") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = read_jsonl(path)
        identity = tuple(row[key] for key in keys)
        rows = [item for item in rows if tuple(item.get(key) for key in keys) != identity]
        rows.append(row)
        temporary.write_text("".join(
            json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in rows),
            encoding="ascii")
        os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()
            if line.strip()]


def unit_rows_with_norms(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("gradient rows must be a matrix")
    units = np.zeros_like(rows)
    norms = np.empty(len(rows), dtype=np.float32)
    if torch.cuda.is_available() and rows.nbytes >= 64 * 1024 * 1024:
        with torch.inference_mode():
            for begin in range(0, len(rows), 8192):
                end = min(begin + 8192, len(rows))
                tensor = torch.as_tensor(rows[begin:end], dtype=torch.float32, device="cuda")
                batch_norms = torch.linalg.vector_norm(tensor, dim=1)
                normalized = torch.where(batch_norms[:, None] > 0,
                                         tensor / batch_norms[:, None].clamp_min(1e-30),
                                         torch.zeros_like(tensor))
                units[begin:end] = normalized.cpu().numpy()
                norms[begin:end] = batch_norms.cpu().numpy()
                del tensor, batch_norms, normalized
    else:
        norms[:] = np.linalg.norm(rows, axis=1)
        units = np.divide(rows, norms[:, None], out=units, where=norms[:, None] > 0)
    return units, norms


def unit_rows(values: np.ndarray) -> tuple[np.ndarray, int]:
    units, norms = unit_rows_with_norms(values)
    return units, int(np.count_nonzero(norms == 0))


def score_summary(scores: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("alignment scores must be finite and nonempty")
    quantiles = np.quantile(values, QUANTILE_LEVELS)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {"p%02d" % int(level * 100): float(value)
                      for level, value in zip(QUANTILE_LEVELS, quantiles)},
    }


def p1_unit_rows(values: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    rows = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1)
    units = np.divide(rows, norms[:, None], out=np.zeros_like(rows),
                      where=norms[:, None] > 0)
    units[norms <= threshold] = 0.0
    return units, int(np.count_nonzero(norms <= threshold))


def pairwise_distribution(left: np.ndarray, right: np.ndarray, seed: int = AUDIT_SEED,
                          sample_size: int = 2_000_000, *,
                          left_are_units: bool = False,
                          right_are_units: bool = False,
                          left_zero_count: int | None = None,
                          right_zero_count: int | None = None) -> dict:
    if left_are_units:
        left_units = np.asarray(left, dtype=np.float32)
        left_zero = (int(left_zero_count) if left_zero_count is not None else
                     int(np.count_nonzero(np.linalg.norm(left_units, axis=1) == 0)))
    else:
        left_units, left_zero = unit_rows(left)
    if right_are_units:
        right_units = np.asarray(right, dtype=np.float32)
        right_zero = (int(right_zero_count) if right_zero_count is not None else
                      int(np.count_nonzero(np.linalg.norm(right_units, axis=1) == 0)))
    else:
        right_units, right_zero = unit_rows(right)
    pair_count = int(len(left_units) * len(right_units))
    if pair_count == 0:
        raise ValueError("pairwise alignment requires nonempty matrices")
    exact = pair_count <= sample_size
    if exact:
        left_index = np.repeat(np.arange(len(left_units), dtype=np.int64), len(right_units))
        right_index = np.tile(np.arange(len(right_units), dtype=np.int64), len(left_units))
    else:
        rng = np.random.default_rng(seed)
        left_index = rng.integers(0, len(left_units), size=sample_size, dtype=np.int64)
        right_index = rng.integers(0, len(right_units), size=sample_size, dtype=np.int64)
    use_cuda = pair_count >= 1_000_000 and torch.cuda.is_available()
    if use_cuda:
        scores = np.empty(len(left_index), dtype=np.float32)
        left_tensor = torch.as_tensor(left_units, dtype=torch.float32, device="cuda")
        right_tensor = torch.as_tensor(right_units, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            left_mean = left_tensor.mean(dim=0)
            right_mean = right_tensor.mean(dim=0)
            exact_mean = float(torch.dot(left_mean, right_mean).item())
            for begin in range(0, len(scores), 8192):
                end = min(begin + 8192, len(scores))
                left_ids = torch.as_tensor(left_index[begin:end], dtype=torch.long, device="cuda")
                right_ids = torch.as_tensor(right_index[begin:end], dtype=torch.long, device="cuda")
                left_batch = left_tensor.index_select(0, left_ids)
                right_batch = right_tensor.index_select(0, right_ids)
                score_batch = torch.sum(left_batch * right_batch, dim=1)
                scores[begin:end] = score_batch.cpu().numpy()
                del left_ids, right_ids, left_batch, right_batch, score_batch
        del left_tensor, right_tensor, left_mean, right_mean
        torch.cuda.empty_cache()
    else:
        scores = np.empty(len(left_index), dtype=np.float32)
        for begin in range(0, len(scores), 32768):
            end = min(begin + 32768, len(scores))
            scores[begin:end] = np.sum(
                left_units[left_index[begin:end]] * right_units[right_index[begin:end]],
                axis=1, dtype=np.float32)
        exact_mean = float(np.dot(left_units.mean(axis=0), right_units.mean(axis=0)))
    quantiles = np.quantile(scores, QUANTILE_LEVELS)
    return dict(
        pair_count=pair_count, mean=exact_mean, quantile_sample_count=int(len(scores)),
        quantile_sampling="exact" if exact else "uniform_with_replacement",
        quantile_seed=int(seed), zero_left=int(left_zero), zero_right=int(right_zero),
        positive_fraction=float(np.mean(scores > 0)),
        abs_le_1e_3_fraction=float(np.mean(np.abs(scores) <= 1e-3)),
        quantiles={"p%02d" % int(level * 100): float(value)
                   for level, value in zip(QUANTILE_LEVELS, quantiles)},
    )


def p99_source_clustering(scores: np.ndarray, ids: np.ndarray, seed: int = AUDIT_SEED,
                           permutations: int = 500) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    identities = np.asarray(ids, dtype=np.int64)
    if values.ndim != 1 or identities.shape != (len(values), 2) or not len(values):
        raise ValueError("P99 clustering inputs are invalid")
    top_count = max(1, int(np.ceil(len(values) * 0.01)))
    order = np.lexsort((identities[:, 1], identities[:, 0], -values))
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:top_count]] = True
    per_fragment = []
    observed_adjacent = 0
    longest_run = 0
    counts = np.zeros(96, dtype=np.int64)
    for fragment_id in range(96):
        rows = np.flatnonzero(identities[:, 0] == fragment_id)
        rows = rows[np.argsort(identities[rows, 1], kind="stable")]
        hits = np.flatnonzero(selected[rows])
        counts[fragment_id] = len(hits)
        if len(hits) > 1:
            observed_adjacent += int(np.count_nonzero(np.diff(hits) == 1))
        run = current = 0
        previous = None
        for position in hits:
            current = current + 1 if previous is not None and position == previous + 1 else 1
            run = max(run, current)
            previous = int(position)
        longest_run = max(longest_run, run)
        per_fragment.append((len(rows), len(hits)))
    rng = np.random.default_rng(seed)
    null_adjacent = np.zeros(permutations, dtype=np.int64)
    for draw in range(permutations):
        pairs = 0
        for length, count in per_fragment:
            if count < 2:
                continue
            chosen = np.sort(rng.choice(length, size=count, replace=False))
            pairs += int(np.count_nonzero(np.diff(chosen) == 1))
        null_adjacent[draw] = pairs
    proportions = counts / float(top_count)
    return dict(
        unit_count=len(values), top_fraction=0.01, top_unit_count=top_count,
        top_score_threshold=float(values[order[top_count - 1]]),
        trajectories_with_hits=int(np.count_nonzero(counts)),
        trajectory_count=96,
        effective_trajectory_count=float(1.0 / np.sum(proportions ** 2)),
        top_10_trajectory_share=float(np.sort(counts)[-10:].sum() / top_count),
        adjacent_top_units=observed_adjacent, longest_consecutive_run=longest_run,
        random_adjacent_mean=float(null_adjacent.mean()),
        randomization_upper_tail_probability=float(
            (1 + np.count_nonzero(null_adjacent >= observed_adjacent)) / (permutations + 1)),
        randomization_draws=int(permutations), randomization_seed=int(seed),
    )


def best_reference_scores(source_units: np.ndarray, reference_units: np.ndarray,
                          logger, task: str, label: str) -> np.ndarray:
    source = np.asarray(source_units, dtype=np.float32)
    reference = np.asarray(reference_units, dtype=np.float32)
    if source.ndim != 2 or reference.ndim != 2 or source.shape[1] != reference.shape[1]:
        raise ValueError("source/reference gradient dimensions do not match")
    source_gpu = torch.as_tensor(source, dtype=torch.float32, device="cuda")
    best = torch.full((len(source),), -float("inf"), dtype=torch.float32, device="cuda")
    started = time.monotonic()
    with torch.inference_mode():
        for begin in range(0, len(reference), 16):
            end = min(begin + 16, len(reference))
            ref = torch.as_tensor(reference[begin:end], dtype=torch.float32, device="cuda")
            best = torch.maximum(best, torch.matmul(ref, source_gpu.T).max(dim=0).values)
            if begin == 0 or end == len(reference) or end % 512 == 0:
                elapsed = time.monotonic() - started
                eta = elapsed * (len(reference) - end) / max(end, 1)
                logger.info("stage=alignment-progress task=%s label=%s reference_rows=%d/%d eta=%s",
                            task, label, end, len(reference), format_duration(eta))
    values = best.cpu().numpy()
    del source_gpu, best
    torch.cuda.empty_cache()
    return values


def row_scores_to_vector(rows: np.ndarray, vector: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    target = np.asarray(vector, dtype=np.float32)
    if values.ndim != 2 or target.shape != (values.shape[1],):
        raise ValueError("row-vector alignment shapes do not match")
    if torch.cuda.is_available() and values.nbytes >= 64 * 1024 * 1024:
        output = np.empty(len(values), dtype=np.float32)
        vector_tensor = torch.as_tensor(target, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            for begin in range(0, len(values), 8192):
                end = min(begin + 8192, len(values))
                rows_tensor = torch.as_tensor(values[begin:end], dtype=torch.float32, device="cuda")
                output[begin:end] = torch.mv(rows_tensor, vector_tensor).cpu().numpy()
                del rows_tensor
        del vector_tensor
        torch.cuda.empty_cache()
        return output
    return values @ target


def validate_source_gradient_caches() -> None:
    expected_warmup = str(WARMUP_ROOT.resolve())
    for worker in range(8):
        matches = list((SOURCE_CACHE_ROOT / ("worker-%d" % worker) / "grad_cache").glob(
            "*/cache_manifest.json"))
        if len(matches) != 1:
            raise ValueError("source worker must have exactly one gradient cache manifest")
        manifest = json.loads(matches[0].read_text(encoding="ascii"))
        model = manifest.get("model", {})
        data = manifest.get("data", {})
        probe = manifest.get("probe", {})
        checks = (
            manifest.get("context_length") == CONTEXT,
            manifest.get("prediction_length") == SELECTOR_HORIZON,
            manifest.get("gradient_protocol") == "independent-native-loss-bf16-v2",
            model.get("model_id") == "thuml/timer-base-84m",
            model.get("warmup_steps") == 1000,
            str(Path(model.get("warmup_checkpoint", "")).resolve()) == expected_warmup,
            model.get("lora", {}).get("target_modules") == ["q_proj", "v_proj"],
            model.get("lora", {}).get("r") == 8,
            model.get("lora", {}).get("alpha") == 16,
            probe.get("projection_dimension") == PROJECTION_DIMENSION,
            probe.get("projection_seed") == PROJECTION_SEED,
            data.get("dataset") == "src-96",
            data.get("q") == 32,
            "uniform" in data.get("method", ""),
        )
        if not all(checks):
            raise ValueError("source gradient cache protocol mismatch for worker %d" % worker)


def source_cache_gradients(source_info) -> tuple[np.ndarray, np.ndarray]:
    from rr_features import fetch, record_anchors

    _config, _legal, _offsets, source_files = source_info
    rows, ids = [], []
    for fragment_id in range(96):
        record, cache = source_files[("uniform", fragment_id)]
        anchors = record_anchors(record, "uniform")
        rows.append(fetch(cache, "src-96", fragment_id, anchors).astype(np.float32))
        ids.extend((fragment_id, int(start) + 1) for start in anchors)
    return np.concatenate(rows), np.asarray(ids, dtype=np.int64)


def _source_unit_manifest() -> dict:
    return dict(schema_version=1, dataset="src-96", source_grid="uniform_q32_cache",
                row_count=83330, gradient_dimension=PROJECTION_DIMENSION,
                dtype="float32", normalization="L2_unit_rows_zero_preserved",
                context_length=CONTEXT, prediction_length=SELECTOR_HORIZON,
                warmup_checkpoint=str(WARMUP_ROOT.resolve()), warmup_steps=1000,
                model_id="thuml/timer-base-84m", projection_dimension=PROJECTION_DIMENSION,
                projection_seed=PROJECTION_SEED, source_cache_root=str(SOURCE_CACHE_ROOT))


def load_source_unit_cache(cache_root: Path):
    marker = cache_root / ".complete"
    manifest_path = cache_root / "manifest.json"
    expected = _source_unit_manifest()
    if not marker.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("source unit cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("source unit cache protocol fields do not match")
    units = np.load(cache_root / "unit_vectors.npy", mmap_mode="r", allow_pickle=False)
    norms = np.load(cache_root / "norms.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(cache_root / "anchor_ids.npy", mmap_mode="r", allow_pickle=False)
    if (units.shape != (expected["row_count"], expected["gradient_dimension"])
            or units.dtype != np.float32 or norms.shape != (expected["row_count"],)
            or norms.dtype != np.float32 or ids.shape != (expected["row_count"], 2)
            or ids.dtype != np.int64 or not np.isfinite(norms).all()
            or np.any(norms < 0)):
        raise ValueError("source unit cache array fields are invalid")
    return units, norms, ids


def prepare_source_unit_cache(logger=None) -> tuple[Path, float, bool]:
    cache_root = output_path("selection", SOURCE_UNIT_CACHE_ROOT)
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        load_source_unit_cache(cache_root)
        return cache_root, 0.0, False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    started = time.monotonic()
    if logger is not None:
        logger.info("stage=source-unit-cache-start source=%s output=%s",
                    SOURCE_CACHE_ROOT, cache_root)
    cache_reader_path = str(ROOT / "scripts/exp/v1-crb-redo")
    if cache_reader_path not in sys.path:
        sys.path.insert(0, cache_reader_path)
    source_info = source_layout()
    gradients, ids = source_cache_gradients(source_info)
    if gradients.shape != (83330, PROJECTION_DIMENSION) or ids.shape != (83330, 2):
        raise ValueError("cached source gradient population differs from q32 contract")
    unit_temp = cache_root / "unit_vectors.partial.npy"
    norm_temp = cache_root / "norms.partial.npy"
    unit_output = np.lib.format.open_memmap(
        unit_temp, mode="w+", dtype=np.float32,
        shape=(len(gradients), PROJECTION_DIMENSION))
    norm_output = np.lib.format.open_memmap(norm_temp, mode="w+", dtype=np.float32,
                                            shape=(len(gradients),))
    for begin in range(0, len(gradients), 8192):
        end = min(begin + 8192, len(gradients))
        tensor = torch.as_tensor(gradients[begin:end], dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            batch_norms = torch.linalg.vector_norm(tensor, dim=1)
            normalized = torch.where(batch_norms[:, None] > 0,
                                     tensor / batch_norms[:, None].clamp_min(1e-30),
                                     torch.zeros_like(tensor))
            unit_output[begin:end] = normalized.cpu().numpy()
            norm_output[begin:end] = batch_norms.cpu().numpy()
        del tensor, batch_norms, normalized
        if logger is not None and (end == len(gradients) or end % 32768 == 0):
            elapsed = time.monotonic() - started
            logger.info("stage=source-unit-cache-progress rows=%d/%d eta=%s",
                        end, len(gradients),
                        format_duration(elapsed * (len(gradients) - end) / max(end, 1)))
    unit_output.flush()
    norm_output.flush()
    del unit_output, norm_output
    os.replace(unit_temp, cache_root / "unit_vectors.npy")
    os.replace(norm_temp, cache_root / "norms.npy")
    ids_temp = cache_root / "anchor_ids.partial.npy"
    np.save(ids_temp, ids, allow_pickle=False)
    os.replace(ids_temp, cache_root / "anchor_ids.npy")
    manifest = _source_unit_manifest()
    manifest.update(created_at=time.time(), source_gradient_file_count=83330,
                    arrays={"unit_vectors": "unit_vectors.npy", "norms": "norms.npy",
                            "anchor_ids": "anchor_ids.npy"})
    write_json(cache_root / "manifest.json", manifest)
    (cache_root / ".complete").write_text("complete\n", encoding="ascii")
    load_source_unit_cache(cache_root)
    elapsed = time.monotonic() - started
    if logger is not None:
        logger.info("stage=source-unit-cache-complete rows=%d elapsed=%s output=%s",
                    len(gradients), format_duration(elapsed), cache_root)
    del gradients, ids
    return cache_root, elapsed, True


def source_layout():
    from rr_features import source_layout as load_source_layout
    return load_source_layout()


def reference_coarse_gradients(task: str) -> tuple[np.ndarray, np.ndarray]:
    path = GRADIENT_ROOT / task / "reference_coarse.npz"
    if not path.is_file():
        raise FileNotFoundError("missing ADS reference coarse gradients: " + str(path))
    with np.load(path, allow_pickle=False) as data:
        required = {"gradients", "fragment_ids", "starts", "context_length",
                    "prediction_length", "warmup_steps", "projection_dimension",
                    "projection_seed", "warmup_checkpoint", "model_id",
                    "gradient_protocol"}
        if not required.issubset(data.files):
            raise ValueError("reference coarse gradient file lacks protocol fields")
        if (int(data["context_length"]) != CONTEXT
                or int(data["prediction_length"]) != SELECTOR_HORIZON
                or int(data["warmup_steps"]) != 1000
                or int(data["projection_dimension"]) != PROJECTION_DIMENSION
                or int(data["projection_seed"]) != PROJECTION_SEED
                or str(Path(str(data["warmup_checkpoint"].item())).resolve()) != str(WARMUP_ROOT.resolve())
                or str(data["model_id"].item()) != "thuml/timer-base-84m"
                or str(data["gradient_protocol"].item()) != "independent-native-loss-bf16-v2"):
            raise ValueError("reference coarse gradient protocol mismatch")
        gradients = np.asarray(data["gradients"], dtype=np.float32)
        ids = np.column_stack((data["fragment_ids"], data["starts"])).astype(np.int64)
    return gradients, ids


def source_population():
    config = yaml.safe_load((ROOT / "configs/data/src-96.yaml").read_text(encoding="ascii"))
    lengths = np.asarray([int(row["length"]) for row in config["trajectories"]], dtype=np.int64)
    legal = lengths - CONTEXT - SELECTOR_HORIZON + 1
    manifest = json.loads((SOURCE_DATA / "manifest.json").read_text(encoding="ascii"))
    offsets = np.asarray(np.load(SOURCE_DATA / "offsets.npy", mmap_mode="r"), dtype=np.int64)
    if (manifest.get("name") != "src-96" or manifest.get("parent") != "src-4096"
            or int(manifest.get("trajectory_count", -1)) != 96
            or int(manifest.get("total_points", -1)) != int(offsets[-1])):
        raise ValueError("src-96 manifest does not match the registered dataset")
    if len(lengths) != 96 or np.any(legal <= 0) or int(legal.sum()) != 2666312:
        raise ValueError("src-96 legal-window population changed")
    if len(offsets) != 97 or not np.array_equal(np.diff(offsets), lengths):
        raise ValueError("src-96 values offsets do not match the source config")
    legal_offsets = np.cumsum(np.r_[0, legal])
    values = np.load(SOURCE_DATA / "values.npy", mmap_mode="r")
    if values.ndim != 1 or int(offsets[-1]) != len(values):
        raise ValueError("src-96 source values layout is not the expected packed univariate form")
    return config, lengths, legal, legal_offsets, offsets, values


def positions_to_ids(positions: np.ndarray, legal: np.ndarray,
                     legal_offsets: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= int(legal_offsets[-1])):
        raise ValueError("source global position outside legal window population")
    owners = np.searchsorted(legal_offsets[1:], positions, side="right")
    local = positions - legal_offsets[owners]
    return np.column_stack((owners, local + 1)).astype(np.int64)


def sample_arm_windows(task: str, arm: str, rng: np.random.Generator,
                       legal: np.ndarray, legal_offsets: np.ndarray,
                       task_dir: Path, sample_count: int) -> np.ndarray:
    manifest_path = task_dir / arm / "sampling.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing actual sampling manifest: " + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    mode = manifest.get("mode")
    if mode == "uniform_all":
        positions = rng.integers(0, int(legal_offsets[-1]), size=sample_count)
        return positions_to_ids(positions, legal, legal_offsets)
    if mode == "uniform_positions":
        positions_path = task_dir / arm / manifest["positions_path"]
        positions = np.asarray(np.load(positions_path, allow_pickle=False), dtype=np.int64)
        if positions.ndim != 1 or not len(positions):
            raise ValueError("uniform_positions must contain a nonempty position vector")
        draws = rng.integers(0, len(positions), size=sample_count)
        return positions_to_ids(positions[draws], legal, legal_offsets)
    if mode == "probability_descriptor":
        descriptor_path = task_dir / arm / manifest["descriptor_path"]
        descriptor = json.loads(descriptor_path.read_text(encoding="ascii"))
        sampler = SourceProbabilitySampler(descriptor)
        fragments, starts = sampler.sample(rng, sample_count)
        return np.column_stack((fragments, starts)).astype(np.int64)
    if mode == "trajectory_uniform":
        fragment_ids = np.asarray(manifest["fragment_ids"], dtype=np.int64)
        if not len(fragment_ids) or np.any(fragment_ids < 0) or np.any(fragment_ids >= len(legal)):
            raise ValueError("trajectory_uniform fragment_ids are invalid")
        probabilities = np.asarray(manifest.get(
            "trajectory_probabilities",
            np.full(len(fragment_ids), 1.0 / len(fragment_ids))), dtype=np.float64)
        if probabilities.shape != (len(fragment_ids),) or np.any(probabilities < 0):
            raise ValueError("trajectory probabilities are invalid")
        mass = float(probabilities.sum())
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("trajectory probabilities must have positive finite mass")
        probabilities /= mass
        owners = rng.choice(fragment_ids, size=sample_count, replace=True, p=probabilities)
        local = np.asarray([rng.integers(0, legal[fid]) for fid in owners], dtype=np.int64)
        return np.column_stack((owners, local + 1)).astype(np.int64)
    raise ValueError("unsupported sampling mode: " + str(mode))


def gather_source_windows(ids: np.ndarray, values: np.ndarray,
                          value_offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    unique_ids, inverse = np.unique(ids, axis=0, return_inverse=True)
    windows = np.empty((len(unique_ids), CONTEXT + SELECTOR_HORIZON), dtype=np.float32)
    for row, (fragment_id, start) in enumerate(unique_ids):
        begin = int(value_offsets[fragment_id]) + int(start) - 1
        windows[row] = values[begin:begin + CONTEXT + SELECTOR_HORIZON]
    if not np.isfinite(windows).all():
        raise ValueError("sampled source windows contain non-finite values")
    return unique_ids, inverse, windows


def rolling_test_windows(task_id: str, logger):
    if task_id == "fev-09":
        import fev
        from datasets import Dataset
        from evaluate.fev import load_task_declarations
        from adapters.chronos.eval.fev import configure_evaluation_windows
        from adapters.foundation.eval.common import clean_context

        declarations = load_task_declarations(FEV_TASK_FILE, required_indices=(9,))
        spec = declarations[9]
        task = fev.Task(**spec)
        source = Path(spec["dataset_path"])
        task._full_dataset = (Dataset.from_parquet(str(source)) if source.suffix == ".parquet"
                              else Dataset.from_file(str(source)))
        configure_evaluation_windows(task, "rolling")
        values, ids, masks = [], [], []
        for window_index in range(int(task.num_windows)):
            window = task.get_window(window_index, num_proc=1)
            past, _ = window.get_input_data()
            truth = window.get_ground_truth().with_format("numpy")
            by_id = {str(row[task.id_column]): row for row in truth}
            for row in past.with_format("numpy"):
                entity = str(row[task.id_column])
                for channel in task.target_columns:
                    if entity not in by_id:
                        raise ValueError("FEV rolling input and truth entity mismatch")
                    context = clean_context(np.asarray(row[channel], dtype=np.float32))[-CONTEXT:]
                    target = np.asarray(by_id[entity][channel], dtype=np.float32).reshape(-1)[:HORIZONS[task_id]]
                    if len(context) != CONTEXT or len(target) != HORIZONS[task_id]:
                        continue
                    mask = np.isfinite(target)
                    if not mask.any():
                        continue
                    values.append(np.concatenate((context, np.where(mask, target, 0.0))))
                    masks.append(mask)
                    ids.append("%d:%s:%s" % (window_index, entity, channel))
        if not values:
            raise ValueError("FEV rolling task has no valid diagnostic windows")
        logger.info("stage=rolling-test-inputs task=%s windows=%d cutoffs=%d",
                    task_id, len(values), int(task.num_windows))
        return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool), ids

    from adapters.foundation.eval.common import clean_context
    from evaluate.time import _validate_bench_config, evaluation_test_length
    from timebench.evaluation.data import Dataset

    config_path = TIME_CONFIGS[task_id]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checked = _validate_bench_config(config["bench_config"])
    if len(checked["tasks"]) != 1 or checked["tasks"][0]["term"] != "short":
        raise ValueError("TIME task config must contain one short task")
    item = checked["tasks"][0]
    name = item["dataset"]
    settings = checked["datasets"][name]
    horizon = int(settings["short"]["prediction_length"])
    if horizon != HORIZONS[task_id]:
        raise ValueError("TIME task horizon mismatch")
    test_length = evaluation_test_length(settings["test_length"], horizon, "rolling")
    kwargs = dict(name=name, term="short", prediction_length=horizon, test_length=test_length,
                  val_length=settings.get("val_length", 0), storage_path=checked["dataset_root"])
    native = Dataset(to_univariate=False, **kwargs)
    dataset = native if native.target_dim == 1 else Dataset(to_univariate=True, **kwargs)
    values, masks, ids = [], [], []
    for ordinal, (inp, label) in enumerate(dataset.test_data):
        context = clean_context(np.asarray(inp["target"], dtype=np.float32).reshape(-1))
        target = np.asarray(label["target"], dtype=np.float32).reshape(-1)[:horizon]
        if len(context) < CONTEXT or len(target) != horizon:
            continue
        context = context[-CONTEXT:]
        mask = np.isfinite(target)
        if not mask.any():
            continue
        values.append(np.concatenate((context, np.where(mask, target, 0.0))))
        masks.append(mask)
        ids.append(str(inp.get("item_id", ordinal)) + ":" + str(ordinal))
    if not values:
        raise ValueError("TIME rolling task has no valid diagnostic windows")
    logger.info("stage=rolling-test-inputs task=%s windows=%d test_length=%d",
                task_id, len(values), test_length)
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool), ids


def continuity_reference_windows(task_id: str, horizon: int):
    fragments = load_fragments(INPUT_ROOT / task_id / "references.npz")
    rows, ids = [], []
    for fragment in fragments:
        starts = fragment.valid_starts(CONTEXT, horizon)
        positions = uniform_anchor_positions(len(starts), 32.0)
        for position in positions:
            start = int(starts[position])
            rows.append(fragment.extract_window(start, CONTEXT, horizon))
            ids.append((int(fragment.fragment_id), start))
    values = np.asarray(rows, dtype=np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("safe-history continuity windows are invalid")
    return values, np.asarray(ids, dtype=np.int64)


def audit_gradient_sketches(probe, windows: np.ndarray, logger, name: str,
                            window_batch_size: int, vjp_batch_size: int,
                            target_masks: np.ndarray | None = None):
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("audit gradient windows must be a nonempty matrix")
    if window_batch_size <= 0 or vjp_batch_size <= 0:
        raise ValueError("audit gradient batch sizes must be positive")
    if target_masks is not None:
        masks = np.asarray(target_masks, dtype=bool)
        if masks.ndim != 2 or masks.shape[0] != len(values):
            raise ValueError("audit target masks do not match gradient windows")
    else:
        masks = None
    previous_vjp = getattr(probe, "vjp_batch_size", None)
    if previous_vjp is not None:
        probe.vjp_batch_size = vjp_batch_size
    losses, sketches = [], []
    started = time.monotonic()
    try:
        for left in range(0, len(values), window_batch_size):
            end = min(left + window_batch_size, len(values))
            batch = values[left:end]
            if masks is not None:
                probe.active_target_mask = masks[left:end]
            try:
                batch_losses, batch_sketches = probe.batch_gradient_sketch(
                    batch, batch_size=window_batch_size)
            finally:
                if masks is not None:
                    del probe.active_target_mask
            losses.append(batch_losses)
            sketches.append(batch_sketches)
            done = end
            if done == len(values) or done % 256 == 0:
                elapsed = time.monotonic() - started
                eta = elapsed * (len(values) - done) / max(done, 1)
                logger.info("stage=gradient-progress name=%s rows=%d/%d window_batch=%d vjp_batch=%d eta=%s",
                            name, done, len(values), window_batch_size, vjp_batch_size,
                            format_duration(eta))
    finally:
        if previous_vjp is not None:
            probe.vjp_batch_size = previous_vjp
    return np.concatenate(losses), np.concatenate(sketches)


def quantile_alignment(task: str, label: str, sources: np.ndarray,
                       references: np.ndarray, logger, *, sources_are_units: bool = False,
                       source_zero_count: int | None = None) -> tuple[np.ndarray, dict]:
    if sources_are_units:
        source_units = np.asarray(sources, dtype=np.float32)
        source_zero = int(source_zero_count or 0)
    else:
        source_units, source_zero = unit_rows(sources)
    reference_units, reference_zero = unit_rows(references)
    values = best_reference_scores(source_units, reference_units, logger, task, label)
    summary = score_summary(values)
    summary.update(source_zero_gradients=source_zero,
                   reference_zero_gradients=reference_zero,
                   reference_anchor_count=int(len(references)),
                   estimator="per-window maximum cosine to reference anchors")
    return values, summary


def read_score_matrix_summaries(task: str) -> list[dict]:
    rows = []
    task_dir = SELECTION_ROOT / task
    for path in task_dir.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (OSError, ValueError):
            continue
        summaries = value.get("score_matrix_summaries")
        if summaries is None and isinstance(value.get("diagnostics"), dict):
            summaries = value["diagnostics"].get("score_matrix_summaries")
        if not isinstance(summaries, list):
            continue
        for summary in summaries:
            rows.append(dict(task=task, artifact=str(path),
                             selection_arm=value.get("arm", value.get("method")),
                             **summary))
    return rows


def task_complete(output: Path, task: str, sample_count: int, seed: int,
                  window_batch_size: int, vjp_batch_size: int) -> bool:
    task_dir = output / "tasks" / task
    manifest_path = task_dir / "manifest.json"
    marker = task_dir / ".complete"
    if not marker.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return False
    return (manifest.get("sample_count") == sample_count
            and manifest.get("seed") == seed
            and manifest.get("gradient_window_batch_size") == window_batch_size
            and manifest.get("gradient_vjp_batch_size") == vjp_batch_size)


def finalize_audit(output: Path, sample_count: int, seed: int,
                   window_batch_size: int, vjp_batch_size: int,
                   model=None, model_fields=None) -> None:
    expected_source = len(TASKS)
    expected_arms = len(TASKS) * len(ARMS)
    current_source = [r for r in read_jsonl(output / "source-alignment.jsonl")
                      if r.get("task") in TASKS]
    current_arms = [r for r in read_jsonl(output / "arm-alignment.jsonl")
                    if r.get("task") in TASKS and r.get("sample_count") == sample_count
                    and r.get("seed") == seed]
    current_continuity = [r for r in read_jsonl(output / "reference-continuity.jsonl")
                          if r.get("task") in TASKS]
    current_p99 = [r for r in read_jsonl(output / "source-p99-clustering.jsonl")
                   if r.get("task") in TASKS]
    task_markers = [output / "tasks" / task / ".complete" for task in TASKS]
    if (len({r["task"] for r in current_source}) != expected_source
            or len({(r["task"], r["arm"]) for r in current_arms}) != expected_arms
            or len({r["task"] for r in current_continuity}) != expected_source
            or len({r["task"] for r in current_p99}) != expected_source
            or not all(path.is_file() for path in task_markers)):
        raise RuntimeError("one or more registered audit rows are missing")
    if model_fields is None:
        manifests = list((SOURCE_CACHE_ROOT / "worker-0" / "grad_cache").glob(
            "*/cache_manifest.json"))
        if len(manifests) != 1:
            raise FileNotFoundError("source gradient cache manifest is missing for audit finalization")
        model_fields = json.loads(manifests[0].read_text(encoding="ascii")).get("model", {})
    if model is None:
        model = dict(model_id=model_fields.get("model_id"), revision=model_fields.get("revision"),
                     warmup_checkpoint=str(WARMUP_ROOT), warmup_steps=1000)
    score_rows = [r for r in read_jsonl(output / "score-matrix.jsonl")
                  if r.get("task") in TASKS]
    write_json(output / "manifest.json", dict(
        experiment="v1-rage-ads", batch="rolling-eight-task", tasks=list(TASKS),
        arms=list(ARMS), sample_count=sample_count, seed=seed,
        source_baseline="all unique q32 uniform cached source gradient anchors",
        alignment_score="per-window maximum cosine plus p1-filtered pairwise direction mean",
        pairwise_quantiles="exact up to 2000000 pairs, otherwise 2000000 uniform draws with replacement, seed 42",
        p99_cluster="top one percent source best-reference anchors, 500 within-trajectory randomizations, seed 42",
        continuity_score="rolling test-window best-history cosine plus pairwise history-test mean and distribution",
        selector_context_length=CONTEXT, selector_prediction_length=SELECTOR_HORIZON,
        warmup_steps=1000, projection_dimension=PROJECTION_DIMENSION,
        gradient_window_batch_size=window_batch_size,
        gradient_vjp_batch_size=vjp_batch_size,
        source_unit_cache=os.environ.get("V1_RAGE_ADS_SOURCE_UNIT_CACHE") or None,
        projection_seed=PROJECTION_SEED, model=model, model_fields=model_fields,
        score_matrix_rows=len(score_rows), output=str(output)))
    (output / ".complete").write_text("complete\n", encoding="ascii")


def main():
    started = time.monotonic()
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=AUDIT_SEED)
    parser.add_argument("--gradient-window-batch", type=int, default=32)
    parser.add_argument("--vjp-batch-size", type=int, default=16)
    parser.add_argument("--source-unit-cache", default=os.environ.get("V1_RAGE_ADS_SOURCE_UNIT_CACHE", ""))
    parser.add_argument("--output-tag", default="production")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    if args.gradient_window_batch <= 0 or args.vjp_batch_size <= 0:
        raise ValueError("gradient batch sizes must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.output_tag):
        raise ValueError("output-tag must contain only letters, digits, underscore, and hyphen")
    output_name = "audit" if args.output_tag == "production" else "audit-" + args.output_tag
    output = output_path("selection", SELECTION_ROOT / output_name)
    output.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        if args.task != "all":
            raise ValueError("--finalize-only requires --task all")
        finalize_audit(output, args.sample_count, args.seed, args.gradient_window_batch,
                       args.vjp_batch_size)
        return
    logger = configure_logger("v1_rage_ads_audit_%s_%s" % (args.task, args.output_tag),
                              "v1-rage-ads", "audit-%s-%s" % (args.task, args.output_tag), "audit")
    selected_tasks = TASKS if args.task == "all" else (args.task,)
    logger.info("stage=config tasks=%s arms=%s sample_count=%d seed=%d warmup=%s window_batch=%d vjp_batch=%d output=%s",
                ",".join(selected_tasks), ",".join(ARMS), args.sample_count, args.seed,
                WARMUP_ROOT, args.gradient_window_batch, args.vjp_batch_size, output)
    if not (WARMUP_ROOT / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("warmup adapter is missing")
    validate_source_gradient_caches()
    for task in selected_tasks:
        reference_coarse_gradients(task)
        task_dir = SELECTION_ROOT / task
        for arm in ARMS:
            if not (task_dir / arm / "sampling.json").is_file():
                raise FileNotFoundError("missing sampling manifest for %s/%s" % (task, arm))
        if not read_score_matrix_summaries(task):
            raise FileNotFoundError("ADS score matrix summaries are missing for " + task)
    source_info = source_layout()
    if args.source_unit_cache:
        source_units, source_norms, source_ids = load_source_unit_cache(Path(args.source_unit_cache))
    else:
        source_gradients, source_ids = source_cache_gradients(source_info)
        source_units, source_norms = unit_rows_with_norms(source_gradients)
        del source_gradients
    model_engine, probe, identity, model_fields = __import__("metrics").build_gradient_probe()
    if (model_fields["warmup_steps"] != 1000
            or probe.config.projection_dimension != PROJECTION_DIMENSION
            or probe.config.projection_seed != PROJECTION_SEED):
        raise ValueError("warmed gradient probe identity mismatch")
    source_config, source_lengths, legal, legal_offsets, value_offsets, source_values = source_population()
    alignment_rows, continuity_rows, score_rows = [], [], []
    work_tasks = tuple(task for task in selected_tasks
                       if not task_complete(output, task, args.sample_count, args.seed,
                                            args.gradient_window_batch, args.vjp_batch_size))
    if args.task == "all" and not work_tasks:
        finalize_audit(output, args.sample_count, args.seed, args.gradient_window_batch,
                       args.vjp_batch_size, identity, model_fields)
        logger.info("stage=finalize-only reason=all task markers complete")
        return
    for task_index, task in enumerate(work_tasks, start=1):
        task_dir = SELECTION_ROOT / task
        reference_gradients, reference_ids = reference_coarse_gradients(task)
        if len(reference_gradients) != len(reference_ids):
            raise ValueError("reference gradient IDs and rows differ")
        source_scores, source_summary = quantile_alignment(
            task, "source_q32_uniform_cache", source_units,
            reference_gradients, logger, sources_are_units=True,
            source_zero_count=int(np.count_nonzero(source_norms == 0)))
        reference_norms = np.linalg.norm(reference_gradients, axis=1)
        source_p1 = float(np.quantile(source_norms, 0.01))
        reference_p1 = float(np.quantile(reference_norms, 0.01))
        source_p1_units = np.asarray(source_units, dtype=np.float32).copy()
        source_p1_units[source_norms <= source_p1] = 0.0
        reference_p1_units, reference_p1_zero = p1_unit_rows(reference_gradients, reference_p1)
        source_p1_window_mean = row_scores_to_vector(
            source_p1_units, reference_p1_units.mean(axis=0))
        source_pairwise = pairwise_distribution(
            source_p1_units, reference_p1_units, seed=AUDIT_SEED,
            left_are_units=True, right_are_units=True,
            left_zero_count=int(np.count_nonzero(source_norms <= source_p1)),
            right_zero_count=reference_p1_zero)
        source_row = dict(task=task, source_anchor_count=len(source_ids),
                          source_anchor_grid="q32_uniform_cache",
                          reference_gradient_file=str(GRADIENT_ROOT / task / "reference_coarse.npz"),
                          p1_source_norm_threshold=source_p1,
                          p1_reference_norm_threshold=reference_p1,
                          p1_filtered_pairwise_mean=source_pairwise["mean"],
                          p1_filtered_pairwise_distribution=source_pairwise,
                          p1_filtered_per_window_mean=score_summary(source_p1_window_mean),
                          **source_summary)
        upsert_jsonl(output / "source-alignment.jsonl", source_row, ("task",))
        p99_row = dict(task=task, source_gradient_grid="q32_uniform_cache",
                       reference_gradient_file=str(GRADIENT_ROOT / task / "reference_coarse.npz"),
                       **p99_source_clustering(source_scores, source_ids))
        upsert_jsonl(output / "source-p99-clustering.jsonl", p99_row, ("task",))
        logger.info("stage=source-alignment-complete task=%s mean=%.8f p50=%.8f p95=%.8f",
                    task, source_summary["mean"], source_summary["quantiles"]["p50"],
                    source_summary["quantiles"]["p95"])
        for arm_index, arm in enumerate(ARMS, start=1):
            stable_task_index = TASKS.index(task) + 1
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, stable_task_index, arm_index]))
            draw_ids = sample_arm_windows(
                task, arm, rng, legal, legal_offsets, task_dir, args.sample_count)
            unique_ids, inverse, windows = gather_source_windows(draw_ids, source_values, value_offsets)
            _, unique_gradients = audit_gradient_sketches(
                probe, windows, logger, "audit-%s-%s" % (task, arm),
                args.gradient_window_batch, args.vjp_batch_size)
            draw_gradients = unique_gradients[inverse]
            arm_scores, arm_summary = quantile_alignment(
                task, arm, draw_gradients, reference_gradients, logger)
            quantile_differences = {
                key: float(arm_summary["quantiles"][key] - source_summary["quantiles"][key])
                for key in source_summary["quantiles"]
            }
            arm_p1_units, arm_p1_zero = p1_unit_rows(draw_gradients, source_p1)
            arm_p1_window_mean = arm_p1_units @ reference_p1_units.mean(axis=0)
            arm_pairwise = pairwise_distribution(
                arm_p1_units, reference_p1_units, seed=args.seed,
                left_are_units=True, right_are_units=True,
                left_zero_count=arm_p1_zero, right_zero_count=reference_p1_zero)
            row = dict(task=task, arm=arm, sample_count=len(draw_ids),
                       unique_sampled_windows=len(unique_ids), seed=args.seed,
                       sampling_manifest=str(task_dir / arm / "sampling.json"),
                       sampled_window_ids_path=str(output / (task + "-" + arm + "-sample-ids.npz")),
                       source_q32_baseline_mean=source_summary["mean"],
                       mean_improvement_vs_source_q32=float(arm_summary["mean"] - source_summary["mean"]),
                       quantile_improvements_vs_source_q32=quantile_differences,
                       p1_source_norm_threshold=source_p1,
                       p1_reference_norm_threshold=reference_p1,
                       p1_filtered_pairwise_mean=arm_pairwise["mean"],
                       p1_filtered_pairwise_distribution=arm_pairwise,
                       p1_filtered_per_window_mean=score_summary(arm_p1_window_mean),
                       **arm_summary)
            ids_path = output / (task + "-" + arm + "-sample-ids.npz")
            temp_path = ids_path.with_name(ids_path.stem + ".partial.npz")
            np.savez_compressed(temp_path, draw_fragment_ids=draw_ids[:, 0],
                                draw_starts=draw_ids[:, 1], unique_fragment_ids=unique_ids[:, 0],
                                unique_starts=unique_ids[:, 1], inverse=inverse, seed=np.asarray(args.seed))
            os.replace(temp_path, ids_path)
            upsert_jsonl(output / "arm-alignment.jsonl", row, ("task", "arm"))
            logger.info("stage=arm-alignment-complete task=%s arm=%s draws=%d unique=%d mean=%.8f delta=%.8f",
                        task, arm, len(draw_ids), len(unique_ids), arm_summary["mean"],
                        row["mean_improvement_vs_source_q32"])
            del draw_gradients, unique_gradients, windows, arm_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        horizon = HORIZONS[task]
        reference_task_windows, reference_task_ids = continuity_reference_windows(task, horizon)
        continuity_probe = __import__("metrics").HorizonProbe(
            model_engine, horizon, identity["model_id"] + "@" + identity["revision"])
        _, history_gradients = audit_gradient_sketches(
            continuity_probe, reference_task_windows, logger, "continuity-history-" + task,
            args.gradient_window_batch, args.vjp_batch_size)
        test_windows, test_masks, test_ids = rolling_test_windows(task, logger)
        test_losses, test_gradients = audit_gradient_sketches(
            continuity_probe, test_windows, logger, "continuity-test-" + task,
            args.gradient_window_batch, args.vjp_batch_size, target_masks=test_masks)
        test_scores, continuity_summary = quantile_alignment(
            task, "rolling_test_task_horizon", test_gradients, history_gradients, logger)
        history_test_pairwise = pairwise_distribution(
            test_gradients, history_gradients, seed=args.seed)
        continuity_ids_path = output / (task + "-continuity-window-ids.npz")
        continuity_ids_tmp = continuity_ids_path.with_name(continuity_ids_path.stem + ".partial.npz")
        np.savez_compressed(continuity_ids_tmp, reference_fragment_ids=reference_task_ids[:, 0],
                            reference_starts=reference_task_ids[:, 1],
                            rolling_test_ids=np.asarray(test_ids, dtype="U"))
        os.replace(continuity_ids_tmp, continuity_ids_path)
        continuity_row = dict(task=task, horizon=horizon,
                              rolling_test_windows=len(test_gradients),
                              window_ids_path=str(continuity_ids_path),
                              valid_target_count=int(test_masks.sum()), mean_task_h_mse=float(np.mean(test_losses)),
                              mean_history_test_cosine=history_test_pairwise["mean"],
                              history_test_pairwise_distribution=history_test_pairwise,
                              warmup_steps=1000, context_length=CONTEXT,
                              **continuity_summary)
        upsert_jsonl(output / "reference-continuity.jsonl", continuity_row, ("task",))
        logger.info("stage=continuity-complete task=%s windows=%d refs=%d mean=%.8f",
                    task, len(test_gradients), len(history_gradients), continuity_summary["mean"])
        matrix_rows = read_score_matrix_summaries(task)
        if not matrix_rows:
            raise FileNotFoundError("ADS score matrix summaries are missing for " + task)
        for row in matrix_rows:
            upsert_jsonl(output / "score-matrix.jsonl", row, ("task", "artifact", "stage"))
            score_rows.append(row)
        alignment_rows.extend([source_row] + [row for row in read_jsonl(output / "arm-alignment.jsonl")
                                              if row.get("task") == task])
        continuity_rows.append(continuity_row)
        task_output = output / "tasks" / task
        task_output.mkdir(parents=True, exist_ok=True)
        write_json(task_output / "manifest.json", dict(
            task=task, sample_count=args.sample_count, seed=args.seed,
            gradient_window_batch_size=args.gradient_window_batch,
            gradient_vjp_batch_size=args.vjp_batch_size,
            source_unit_cache=(str(Path(args.source_unit_cache)) if args.source_unit_cache else None),
            completed_at=time.time()))
        (task_output / ".complete").write_text("complete\n", encoding="ascii")
        del continuity_probe, reference_task_windows, history_gradients, test_windows, test_gradients, test_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("stage=task-complete task=%s progress=%d/%d eta=%s",
                    task, task_index, len(selected_tasks),
                    format_duration((time.monotonic() - started)
                                    * (len(selected_tasks) - task_index) / task_index))
    if args.task == "all":
        finalize_audit(output, args.sample_count, args.seed, args.gradient_window_batch,
                       args.vjp_batch_size, identity, model_fields)
    logger.info("stage=complete tasks=%d source_rows=%d arm_rows=%d continuity_rows=%d matrix_rows=%d elapsed=%s output=%s",
                len(selected_tasks), len(alignment_rows), len(read_jsonl(output / "arm-alignment.jsonl")),
                len(continuity_rows), len(score_rows), format_duration(time.monotonic() - started), output)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts/exp/v1-crb-redo"))
    main()
