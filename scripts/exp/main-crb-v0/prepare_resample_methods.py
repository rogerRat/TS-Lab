"""Prepare full-support CRB/UNI window and trajectory probability arms."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/exp/main-crb-v0"))
import run_directional_time1 as base
from prepare_eight_task import TASKS, load_task_reference
from rage.experiments.ranking import rank_mass
from rage.experiments.resampling import ResampledWindows
BATCH = "q16-src256-eight-task-resample"
OLD = ROOT / "outputs/selection/main-crb-v0/q16-src256-time1-q8"
REF_ROOT = ROOT / "outputs/selection/main-crb-v0/q16-src256-eight-task"
CACHE = ROOT / "outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/components"
REF_CACHE = ROOT / "outputs/grad_cache/main-crb-v0/q16-src256-eight-task/components"
OUT = ROOT / "outputs/selection/main-crb-v0" / BATCH
CHUNK = 8192

def save_payload(method, task, probabilities, metadata, layout="window", offsets=None):
    root = OUT / method / task
    root.mkdir(parents=True, exist_ok=True)
    arrays = {"probabilities": np.asarray(probabilities, dtype=np.float64),
              "unit_layout": np.asarray(layout),
              "resampling_metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    if offsets is not None:
        arrays["unit_offsets"] = np.asarray(offsets, dtype=np.int64)
    path = root / "resampling_payload.npz"
    np.savez_compressed(path, **arrays)
    (root / ".complete").write_text("complete\n", encoding="ascii")
    return path

def reference_linear_sum(task, logger=None):
    manifest, fragments, counts, _positions = load_task_reference(task)
    ref_root = REF_ROOT / task["code"] / "reference"
    with np.load(ref_root / "reference_block_index.npz", allow_pickle=False) as archive:
        block_fids = np.asarray(archive["fragment_ids"], dtype=np.int64)
        centers = np.asarray(archive["centers"], dtype=np.float64)
    directions = np.load(ref_root / "reference_block_directions_fp16.npy", mmap_mode="r", allow_pickle=False)
    weights = np.zeros(len(block_fids), dtype=np.float64)
    for fragment, count in zip(fragments, counts):
        rows = np.flatnonzero(block_fids == int(fragment.fragment_id))
        local_centers = centers[rows]
        if len(rows) == 0 or np.any(np.diff(local_centers) <= 0):
            raise ValueError("reference block centers are missing or unordered")
        boundaries = (local_centers[:-1] + local_centers[1:]) * 0.5
        owner = np.searchsorted(boundaries, np.arange(int(count)), side="left")
        owned = np.bincount(owner, minlength=len(rows))
        if int(owned.sum()) != int(count) or np.any(owned <= 0):
            raise ValueError("reference blocks do not partition all legal starts")
        weights[rows] = owned / (len(fragments) * int(count))
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("linearSum reference weights do not sum to one")
    aggregate = np.zeros(directions.shape[1], dtype=np.float64)
    for first in range(0, len(directions), CHUNK):
        stop = min(first + CHUNK, len(directions))
        aggregate += weights[first:stop] @ directions[first:stop].astype(np.float64)
    return aggregate, manifest, fragments, counts, len(directions)

def prepare_window_arm(method, task, source_counts, source_total, logger):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CRB/UNI probability scoring requires CUDA")
    ref_vector, ref_manifest, _fragments, _counts, ref_blocks = reference_linear_sum(task)
    method_root = OLD / method
    with np.load(method_root / (method + "_block_index.npz"), allow_pickle=False) as archive:
        block_fids = np.asarray(archive["fragment_ids"], dtype=np.int64)
        centers = np.asarray(archive["centers"], dtype=np.float32)
    directions = np.load(method_root / (method + "_block_directions_fp16.npy"), mmap_mode="r", allow_pickle=False)
    score_root = OUT / method / task["code"]
    score_root.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(score_root / "window_scores.npy", mode="w+", dtype=np.float32, shape=(source_total,))
    aggregate = torch.as_tensor(ref_vector, dtype=torch.float32, device="cuda")
    cursor = 0
    started = time.monotonic()
    for fid, count in enumerate(source_counts):
        rows = np.flatnonzero(block_fids == fid)
        if len(rows) == 0:
            raise ValueError("source trajectory has no q16 blocks")
        local_centers = torch.as_tensor(centers[rows], dtype=torch.float32, device="cuda")
        local_directions = torch.as_tensor(directions[rows].astype(np.float32), dtype=torch.float32, device="cuda")
        for first in range(0, int(count), CHUNK):
            stop = min(first + CHUNK, int(count))
            positions = torch.arange(first, stop, dtype=torch.float32, device="cuda")
            right = torch.searchsorted(local_centers, positions, right=True)
            left = torch.clamp(right - 1, min=0, max=len(rows) - 1)
            right = torch.clamp(right, min=0, max=len(rows) - 1)
            denominator = local_centers[right] - local_centers[left]
            alpha = torch.where(denominator > 0, (positions - local_centers[left]) / denominator, torch.zeros_like(positions)).clamp_(0.0, 1.0)
            interpolated = local_directions[left] * (1.0 - alpha[:, None]) + local_directions[right] * alpha[:, None]
            norms = torch.linalg.vector_norm(interpolated, dim=1)
            interpolated = torch.where(norms[:, None] > 0, interpolated / norms[:, None].clamp_min(1e-20), torch.zeros_like(interpolated))
            scores[cursor + first:cursor + stop] = (interpolated @ aggregate).cpu().numpy()
        cursor += int(count)
        if cursor == source_total or (fid + 1) % 32 == 0:
            elapsed = time.monotonic() - started
            eta = elapsed * (256 - fid - 1) / max(fid + 1, 1)
            print("score-progress method=%s task=%s trajectories=%d/256 windows=%d/%d elapsed=%s eta=%s" % (
                method, task["code"], fid + 1, cursor, source_total, base.format_duration(elapsed), base.format_duration(eta)), flush=True)
        del local_centers, local_directions
    scores.flush()
    _, probabilities, audit = rank_mass(np.asarray(scores), np.ones(source_total, dtype=np.int64))
    payload = save_payload(method, task["code"], probabilities,
        {"method": method, "scoring_unit": "legal_window", "reference_aggregation": "linearSum",
         "reference_block_count": ref_blocks, "reference_task": task["label"],
         "probability_mapping": audit["curve"], "temperature": audit["temperature"],
         "hard_subset": False, "seed": 0})
    torch.cuda.empty_cache()
    del scores, probabilities
    print("prepared method=%s task=%s windows=%d payload=%s" % (method, task["code"], source_total, payload), flush=True)

def raw_trajectory_vectors(anchor_path, directions_path, magnitudes_path, source=False):
    with np.load(anchor_path, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"], dtype=np.int64)
        gradient_rows = np.asarray(archive["gradient_rows"], dtype=np.int64)
        legal_counts = np.asarray(archive["fragment_counts"], dtype=np.int64)
    q = base.SOURCE_Q if source else base.REFERENCE_Q
    anchor_counts = np.asarray([len(base.exact_uniform_positions(int(n), q)) for n in legal_counts], dtype=np.int64)
    if int(anchor_counts.sum()) != len(positions):
        raise ValueError("anchor counts differ from q registration")
    directions = np.load(directions_path, mmap_mode="r", allow_pickle=False)
    magnitudes = np.load(magnitudes_path, mmap_mode="r", allow_pickle=False)
    vectors = np.zeros((len(anchor_counts), directions.shape[1]), dtype=np.float64)
    cursor = 0
    for i, count in enumerate(anchor_counts):
        rows = gradient_rows[cursor:cursor + int(count)]
        accumulator = np.zeros(directions.shape[1], dtype=np.float64)
        for first in range(0, len(rows), 4096):
            local_rows = rows[first:first + 4096]
            unit = directions[local_rows].astype(np.float64)
            norm = np.asarray(magnitudes[local_rows], dtype=np.float64)
            accumulator += (unit * norm[:, None]).sum(axis=0)
        length = float(np.linalg.norm(accumulator))
        if length > 0:
            vectors[i] = accumulator / length
        cursor += int(count)
    return vectors, legal_counts

def prepare_trajectory_arm(method, task, source_counts):
    source_method = "crb" if method == "crb-traj" else "uni"; source_anchor = OLD / source_method / "source_anchors.npz"
    # Shared source directions and magnitudes are read-only cache assets.
    source_vectors, source_legal = raw_trajectory_vectors(source_anchor,
        CACHE / "directions_fp16.npy", CACHE / "magnitudes_fp32.npy", source=True)
    ref_anchor = REF_ROOT / task["code"] / "reference/reference_anchors.npz"
    ref_vectors, _ref_legal = raw_trajectory_vectors(ref_anchor,
        REF_CACHE / "directions_fp16.npy", REF_CACHE / "magnitudes_fp32.npy", source=False)
    if not len(ref_vectors) or not len(source_vectors):
        raise ValueError("trajectory representations are empty")
    scores = source_vectors @ ref_vectors.mean(axis=0)
    _, probabilities, audit = rank_mass(scores, source_legal)
    offsets = np.r_[0, np.cumsum(source_legal, dtype=np.int64)]
    payload = save_payload(method, task["code"], probabilities,
        {"method": method, "scoring_unit": "trajectory", "reference_aggregation": "linearSum",
         "trajectory_representation": "mean raw projected anchor gradient, then L2 normalize",
         "probability_mapping": audit["curve"], "temperature": audit["temperature"],
         "hard_subset": False, "window_draw": "uniform legal window within sampled trajectory", "seed": 0},
        layout="trajectory", offsets=offsets)
    print("prepared method=%s task=%s trajectories=%d payload=%s" % (method, task["code"], len(probabilities), payload), flush=True)

def main():
    rows, source_fragments, _values, _offsets, source_counts = base.load_source_fragments()
    source_counts = np.asarray(source_counts, dtype=np.int64)
    source_total = int(source_counts.sum())
    if source_total != 6382020:
        raise ValueError("src-256 pool size changed")
    for task in TASKS:
        for method in ("crb", "uni"):
            prepare_window_arm(method, task, source_counts, source_total, None)
        for method in ("crb-traj", "uni-traj"):
            prepare_trajectory_arm(method, task, source_counts)

if __name__ == "__main__":
    main()
