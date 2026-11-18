"""Build compact full-pool random and GraNd-static resampling assets."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/exp/main-crb-v0"))
import run_directional_time1 as base
from rage.experiments.ranking import rank_mass
BATCH = "q16-src256-eight-task-resample"
POOL_ROOT = ROOT / "outputs/selection/main-crb-v0" / BATCH / "shared/pool"
SELECTION_ROOT = ROOT / "outputs/selection/main-crb-v0" / BATCH
SOURCE_ANCHORS = ROOT / "outputs/selection/main-crb-v0/q16-src256-time1-q8/uni/source_anchors.npz"
MAGNITUDES = ROOT / "outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/components/magnitudes_fp32.npy"

def prepare_pool():
    POOL_ROOT.mkdir(parents=True, exist_ok=True)
    locations_path = POOL_ROOT / "window_locations.npy"
    ids_path = POOL_ROOT / "window_ids.npy"
    manifest_path = POOL_ROOT / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="ascii") as stream:
            manifest = json.load(stream)
        if manifest.get("total_equivalents") != 6382020:
            raise ValueError("cached src-256 pool manifest differs")
        return manifest
    rows = __import__("yaml").safe_load(base.SOURCE_CONFIG.read_text(encoding="ascii"))["trajectories"]
    source_values = base.SOURCE_DATA / "values.npy"
    if not source_values.is_file():
        raise FileNotFoundError("registered src-256 values are missing")
    values_link = POOL_ROOT / "values.npy"
    if not values_link.exists():
        values_link.symlink_to(source_values.resolve())
    source_offsets = np.load(base.SOURCE_DATA / "offsets.npy", mmap_mode="r", allow_pickle=False)
    counts = np.asarray([int(row["length"]) - base.WINDOW + 1 for row in rows], dtype=np.int64)
    total = int(counts.sum())
    if len(rows) != 256 or total != 6382020:
        raise ValueError("src-256 trajectory/window count differs from registration")
    locations = np.lib.format.open_memmap(locations_path, mode="w+", dtype=np.int64, shape=(total,))
    ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int64, shape=(total, 2))
    cursor = 0
    for fid, count in enumerate(counts):
        stop = cursor + int(count)
        starts = np.arange(1, int(count) + 1, dtype=np.int64)
        locations[cursor:stop] = int(source_offsets[fid]) + starts - 1
        ids[cursor:stop, 0] = fid
        ids[cursor:stop, 1] = starts
        cursor = stop
    locations.flush(); ids.flush()
    del locations, ids
    manifest = {"schema_version": 1, "source": "registered src-256", "trajectories": 256,
        "total_equivalents": total, "legal_windows": total,
        "context_length": base.CONTEXT, "prediction_length": base.HORIZON,
        "window_length": base.WINDOW, "stride": 1,
        "trajectory_legal_windows": counts.tolist(),
        "values_path": str(values_link.resolve()),
        "window_locations_path": str(locations_path), "window_ids_path": str(ids_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest

def save_payload(method, scope, probabilities, metadata, layout="window", offsets=None):
    root = SELECTION_ROOT / method / scope
    root.mkdir(parents=True, exist_ok=True)
    path = root / "resampling_payload.npz"
    arrays = {"probabilities": np.asarray(probabilities, dtype=np.float64),
              "unit_layout": np.asarray(layout),
              "resampling_metadata": np.asarray(json.dumps(metadata, sort_keys=True))}
    if offsets is not None:
        arrays["unit_offsets"] = np.asarray(offsets, dtype=np.int64)
    np.savez_compressed(path, **arrays)
    return path

def prepare_rand(manifest):
    count = int(manifest["total_equivalents"])
    probabilities = np.full(count, 1.0 / count, dtype=np.float64)
    return save_payload("rand", "shared", probabilities,
        {"method": "rand", "sampling_unit": "legal_window", "probability_rule": "uniform over all src-256 legal windows", "seed": 0, "hard_subset": False})

def prepare_grand(manifest):
    with np.load(SOURCE_ANCHORS, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"], dtype=np.int64)
        grad_rows = np.asarray(archive["gradient_rows"], dtype=np.int64)
        legal_counts = np.asarray(archive["fragment_counts"], dtype=np.int64)
    anchor_counts = np.asarray([len(base.exact_uniform_positions(int(n), base.SOURCE_Q)) for n in legal_counts], dtype=np.int64)
    if int(anchor_counts.sum()) != len(positions) or len(grad_rows) != len(positions):
        raise ValueError("UNI source anchor split differs from q16 registration")
    magnitudes = np.load(MAGNITUDES, mmap_mode="r", allow_pickle=False)
    lengths = np.asarray(manifest["trajectory_legal_windows"], dtype=np.int64)
    scores, interval_counts = [], []
    flat_offsets = [0]
    cursor = 0
    for fid, anchor_count in enumerate(anchor_counts):
        stop = cursor + int(anchor_count)
        anchor_positions = positions[cursor:stop]
        anchor_rows = grad_rows[cursor:stop]
        if len(anchor_positions) < 2 or np.any(np.diff(anchor_positions) <= 0):
            raise ValueError("GraNd requires strictly ordered adjacent q16 anchors")
        centers = (anchor_positions[:-1].astype(np.float64) + anchor_positions[1:].astype(np.float64)) * 0.5
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        owner = np.searchsorted(boundaries, np.arange(int(lengths[fid])), side="left")
        counts = np.bincount(owner, minlength=len(centers)).astype(np.int64)
        if np.any(counts <= 0) or int(counts.sum()) != int(lengths[fid]):
            raise ValueError("GraNd interval ownership does not partition a trajectory")
        norms_left = np.asarray(magnitudes[anchor_rows[:-1]], dtype=np.float64)
        norms_right = np.asarray(magnitudes[anchor_rows[1:]], dtype=np.float64)
        scores.extend(((norms_left + norms_right) * 0.5).tolist())
        interval_counts.extend(counts.tolist())
        flat_offsets.extend((flat_offsets[-1] + np.cumsum(counts)).tolist())
        cursor = stop
    if cursor != len(positions):
        raise ValueError("not all q16 anchors were consumed")
    scores = np.asarray(scores, dtype=np.float64)
    interval_counts = np.asarray(interval_counts, dtype=np.int64)
    _, probabilities, audit = rank_mass(scores, interval_counts)
    metadata = {"method": "grand-static", "sampling_unit": "adjacent UNI q16 anchor interval",
        "interval_score": "arithmetic mean of endpoint projected-gradient L2 norms",
        "interval_window_ownership": "nearest interval center, Voronoi boundaries, tie to lower interval",
        "probability_mapping": audit["curve"], "temperature": audit["temperature"],
        "interval_count": len(scores), "legal_windows": int(interval_counts.sum()),
        "seed": 0, "support": "all intervals with positive rank mass"}
    return save_payload("grand-static", "shared", probabilities, metadata,
                        layout="interval", offsets=flat_offsets)

def main():
    manifest = prepare_pool()
    rand_path = prepare_rand(manifest)
    grand_path = prepare_grand(manifest)
    print("prepared src-256 pool windows=%d path=%s" % (manifest["total_equivalents"], POOL_ROOT), flush=True)
    print("prepared rand payload=%s" % rand_path, flush=True)
    print("prepared grand-static payload=%s" % grand_path, flush=True)

if __name__ == "__main__":
    main()
