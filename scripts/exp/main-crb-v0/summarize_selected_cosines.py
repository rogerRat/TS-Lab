"""Estimate cosine-score distribution over sampled selected source windows."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs.logger import configure_logger, format_duration

BATCH = "q16-src256-eight-task"
TASKS = ("FEV09", "TIME1", "TIME2", "TIME3", "TIME4", "TIME6", "TIME7", "TIME8")
ARMS = ("crb", "uni")
SELECTION_ROOT = ROOT / "outputs/selection/main-crb-v0" / BATCH
REPORT_ROOT = SELECTION_ROOT / "cosine-summary"
SAMPLE_SIZE = 4096
BASE_SEED = 20261116
SCORE_CHUNK = 256


def interpolate_directions(centers, directions, positions):
    positions = np.asarray(positions, dtype=np.float32)
    right = np.searchsorted(centers, positions, side="right")
    left = np.clip(right - 1, 0, len(centers) - 1)
    right = np.clip(right, 0, len(centers) - 1)
    denominator = centers[right] - centers[left]
    alpha = np.divide(positions - centers[left], denominator,
                      out=np.zeros_like(positions), where=denominator > 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    values = directions[left] * (1.0 - alpha[:, None]) + directions[right] * alpha[:, None]
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values),
                     where=norms[:, None] > 0)


def summarize(scores):
    quantiles = np.quantile(scores, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "sample_count": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "min": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p90": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def score_sample(task, arm, task_index, arm_index, logger):
    task_root = SELECTION_ROOT / task
    arm_root = task_root / arm
    with (arm_root / "selection.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected_ids = np.load(arm_root / "selected_window_ids.npy", mmap_mode="r")
    rng = np.random.default_rng(BASE_SEED + task_index * 10 + arm_index)
    n = min(SAMPLE_SIZE, len(selected_ids))
    chosen = np.sort(rng.choice(len(selected_ids), size=n, replace=False))
    sample_ids = np.asarray(selected_ids[chosen], dtype=np.int64)

    source_direction_path = Path(manifest["block_directions"])
    source_index_path = Path(manifest["block_index"])
    reference_direction_path = task_root / "reference/reference_block_directions_fp16.npy"
    if not source_direction_path.is_absolute():
        source_direction_path = ROOT / source_direction_path
    if not source_index_path.is_absolute():
        source_index_path = ROOT / source_index_path
    source_directions = np.load(source_direction_path, mmap_mode="r")
    reference_directions = np.load(reference_direction_path, mmap_mode="r").astype(np.float32)
    with np.load(source_index_path, allow_pickle=False) as archive:
        fragment_ids = np.asarray(archive["fragment_ids"])
        centers = np.asarray(archive["centers"], dtype=np.float32)

    interpolated = np.empty((n, source_directions.shape[1]), dtype=np.float32)
    fragment_values = sample_ids[:, 0]
    for fragment_id in np.unique(fragment_values):
        row_mask = fragment_values == fragment_id
        block_rows = np.flatnonzero(fragment_ids == fragment_id)
        if not len(block_rows):
            raise ValueError("source fragment has no direction blocks: %d" % fragment_id)
        block_centers = centers[block_rows]
        block_reps = source_directions[block_rows].astype(np.float32)
        local_positions = sample_ids[row_mask, 1].astype(np.float32) - 1.0
        interpolated[row_mask] = interpolate_directions(
            block_centers, block_reps, local_positions)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected cosine summary requires CUDA BF16 support")
    ref_vectors = torch.as_tensor(reference_directions, dtype=torch.bfloat16, device="cuda")
    sample_scores = np.empty(n, dtype=np.float32)
    for first in range(0, n, SCORE_CHUNK):
        stop = min(first + SCORE_CHUNK, n)
        source_vectors = torch.as_tensor(
            interpolated[first:stop], dtype=torch.bfloat16, device="cuda")
        scores = torch.matmul(source_vectors, ref_vectors.T).float()
        sample_scores[first:stop] = scores.max(dim=1).values.cpu().numpy()
        if stop == n or stop % 1024 == 0:
            logger.info("stage=cosine-progress task=%s arm=%s sampled=%d/%d",
                        task, arm, stop, n)
        del source_vectors, scores

    result = {
        "task": task,
        "method": arm,
        "score_definition": "max BF16 cosine to task-specific reference block directions",
        "selected_count": int(len(selected_ids)),
        "selection_cutoff": float(manifest["score_cutoff"]),
        "selection_budget_fraction": float(manifest["selection_budget_fraction"]),
        "sample_seed": int(BASE_SEED + task_index * 10 + arm_index),
        "statistics": summarize(sample_scores),
    }
    del ref_vectors, reference_directions, source_directions, interpolated
    torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected cosine summary requires CUDA BF16 support")
    logger = configure_logger(
        "main_crb_v0_selected_cosines", "main-crb-v0", BATCH,
        "selected-cosine-summary", enable_heartbeat=True)
    started = time.monotonic()
    logger.info("stage=config batch=%s tasks=%d methods=%s sample_size=%d seed=%d chunk=%d device=cuda",
                BATCH, len(TASKS), ",".join(ARMS), SAMPLE_SIZE, BASE_SEED, SCORE_CHUNK)
    results = []
    for task_index, task in enumerate(TASKS):
        for arm_index, arm in enumerate(ARMS):
            logger.info("stage=arm-start task=%s arm=%s", task, arm)
            result = score_sample(task, arm, task_index, arm_index, logger)
            results.append(result)
            stats = result["statistics"]
            logger.info("stage=arm-complete task=%s arm=%s mean=%.6f median=%.6f p10=%.6f p90=%.6f cutoff=%.6f",
                        task, arm, stats["mean"], stats["median"], stats["p10"],
                        stats["p90"], result["selection_cutoff"])

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch": BATCH,
        "sample_size_per_task_method": SAMPLE_SIZE,
        "base_seed": BASE_SEED,
        "method": "uniform sample without replacement from selected top-10-percent ids",
        "results": results,
    }
    target = REPORT_ROOT / "selected_cosine_sample.json"
    temporary = target.with_name(target.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(target)
    (REPORT_ROOT / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete elapsed=%s output=%s",
                format_duration(time.monotonic() - started), REPORT_ROOT)


if __name__ == "__main__":
    main()
