"""Run v0-grad-mech-test warm-up and full-gradient overlap analysis."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from configs.logger import configure_logger
from configs.paths import PROJECT_ROOT, output_path
from rage.experiments.data import _read_series, load_bundle_index
from rage.experiments.train_eval import fine_tune

LOGGER = logging.getLogger("v0.grad_mech_test")
WINDOW = 576
MAX_LAG = WINDOW - 1
RING_SIZE = WINDOW
BATCH_GRADIENTS = 8
HISTORY_TILE = 32
HISTOGRAM_BINS = 128


def read_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="ascii"))
    if value.get("experiment") != "v0-grad-mech-test":
        raise ValueError("unexpected experiment config")
    return value


def select_trajectories(config: dict) -> tuple[list[tuple[str, np.ndarray]], list[dict]]:
    bundle = load_bundle_index(Path(config["bundle_data"]) / "load")
    selected = []
    manifest_rows = []
    for row, raw_length in enumerate(bundle.index["length"]):
        if len(selected) >= int(config["selection"]["count"]):
            break
        if bundle.index["dropped"][row] or int(raw_length) < WINDOW:
            continue
        values = np.asarray(_read_series(bundle, row), dtype=np.float32)
        if not np.isfinite(values).all() or float(np.std(values)) == 0.0:
            continue
        series_id = str(bundle.meta["series_id"][row])
        selected.append((series_id, values))
        manifest_rows.append({"source_row": row, "series_id": series_id, "length": len(values)})
    if len(selected) != int(config["selection"]["count"]):
        raise RuntimeError("only %d eligible trajectories found; expected %d" % (
            len(selected), int(config["selection"]["count"])))
    return selected, manifest_rows


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
                         encoding="ascii")
    temporary.replace(path)


def prepare_inputs(config: dict, out: Path) -> tuple[list[tuple[str, np.ndarray]], list[dict], np.ndarray]:
    manifest_path = out / "input_manifest.json"
    selected, rows = select_trajectories(config)
    manifest_config = {key: value for key, value in config.items()
                       if key != "retain_gradient_ordinals"}
    manifest = {"config": manifest_config, "trajectory_count": len(rows), "trajectories": rows,
                "selection_rule": "first eligible rows in Chronos m5 source order; no random sampling"}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="ascii"))
        if prior != manifest:
            raise ValueError("input manifest changed; use a new experiment output directory")
    else:
        write_json_atomic(manifest_path, manifest)

    data_root = PROJECT_ROOT / "data" / "processed" / "v0-grad-mech-test"
    data_root.mkdir(parents=True, exist_ok=True)
    windows_path = data_root / "warmup_windows.npy"
    if windows_path.exists():
        warmup_windows = np.load(windows_path, mmap_mode="r", allow_pickle=False)
    else:
        rows_out = []
        for _series_id, values in selected:
            max_start = len(values) - WINDOW
            starts = sorted(set((0, max_start // 3, (2 * max_start) // 3, max_start)))
            for start in starts:
                rows_out.append(values[start:start + WINDOW])
        warmup_windows = np.asarray(rows_out, dtype=np.float32)
        temporary = windows_path.with_suffix(".npy.partial")
        with temporary.open("wb") as handle:
            np.save(handle, warmup_windows, allow_pickle=False)
        temporary.replace(windows_path)
    if len(warmup_windows) < 32:
        raise ValueError("warm-up set must contain at least one full batch")
    return selected, rows, warmup_windows


def load_engine(model_path: str, dtype: str):
    from rage.engines import ChronosT5Config, ChronosT5Engine
    return ChronosT5Engine(ChronosT5Config(
        model_path=model_path, context_length=512, prediction_length=64,
        torch_dtype=dtype, use_vmap_gradients=False,
    ), device="cuda")


def warmup(engine, windows: np.ndarray, out: Path, config: dict) -> Path:
    checkpoint = out / "chronos-tiny-warmup-100"
    marker = checkpoint / ".complete"
    if marker.is_file():
        LOGGER.info("Warm-up checkpoint already complete: %s", checkpoint)
        return checkpoint
    LOGGER.info("Warm-up starting: steps=100 batch_size=32 learning_rate=1e-6 trajectories=1000")
    started = time.monotonic()
    def on_step(step: int) -> None:
        if step % 10 == 0 or step == 100:
            LOGGER.info("Warm-up optimizer step %d/100 elapsed_seconds=%.1f", step, time.monotonic() - started)
    result = fine_tune(engine, windows, steps=100, batch_size=32, learning_rate=1e-6,
                       seed=0, weight_decay=0.0, on_step=on_step)
    checkpoint.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(checkpoint), safe_serialization=True)
    write_json_atomic(out / "warmup_summary.json", {
        "steps": result.get("steps", 100), "initial_loss": result.get("initial_loss"),
        "final_loss": result.get("final_loss"), "elapsed_seconds": time.monotonic() - started,
        "batch_size": 32, "learning_rate": 1e-6, "seed": 0,
        "model": config["model"], "training_windows": int(len(windows)),
    })
    marker.write_text("complete\n", encoding="ascii")
    LOGGER.info("Warm-up complete: elapsed_seconds=%.1f checkpoint=%s", time.monotonic() - started, checkpoint)
    return checkpoint


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _summarize_pairs(pair_values: np.ndarray, metric: str) -> tuple[np.ndarray, ...]:
    # pair_values has [window, lag] geometry and NaN where no pair exists.
    n_lags = pair_values.shape[1]
    count = np.zeros(n_lags, dtype=np.int64)
    total = np.zeros(n_lags, dtype=np.float64)
    total_sq = np.zeros(n_lags, dtype=np.float64)
    minimum = np.full(n_lags, np.inf, dtype=np.float64)
    maximum = np.full(n_lags, -np.inf, dtype=np.float64)
    hist = np.zeros((n_lags, HISTOGRAM_BINS), dtype=np.int64)
    if metric == "cosine":
        lower, upper = -1.0, 1.0
    elif metric == "relative_norm_difference":
        lower, upper = 0.0, 2.0
    else:
        lower, upper = -6.0, 6.0
    edges = np.linspace(lower, upper, HISTOGRAM_BINS + 1)
    for lag_index in range(n_lags):
        values = pair_values[:, lag_index]
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        count[lag_index] = len(values)
        total[lag_index] = values.sum(dtype=np.float64)
        total_sq[lag_index] = np.square(values, dtype=np.float64).sum(dtype=np.float64)
        minimum[lag_index] = values.min()
        maximum[lag_index] = values.max()
        hist[lag_index], _ = np.histogram(values, bins=edges)
    minimum[~np.isfinite(minimum)] = np.nan
    maximum[~np.isfinite(maximum)] = np.nan
    return count, total, total_sq, minimum, maximum, hist


def _gradient(engine, window: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = engine.batch_tensors(window.reshape(1, -1))
    output = engine.model(**engine._logits_inputs(tensors), return_dict=True)
    from rage.engines import per_sample_cross_entropy
    loss = per_sample_cross_entropy(output.logits, tensors["labels"])[0]
    params = engine._parameters
    gradients = torch.autograd.grad(loss, params, allow_unused=True)
    pieces = [torch.zeros_like(parameter) if grad is None else grad
              for parameter, grad in zip(params, gradients)]
    flat = torch.cat([piece.detach().reshape(-1) for piece in pieces])
    norm = torch.linalg.vector_norm(flat.float())
    return flat.detach(), norm.detach()


def process_trajectory(engine, series_id: str, values: np.ndarray, precision: str,
                       destination: Path, progress: tqdm,
                       cache_destination: Path | None = None) -> dict:
    device = engine.device
    model = engine.model
    model.eval()
    n_windows = len(values) - WINDOW + 1
    pair_cos = np.full((n_windows, MAX_LAG), np.nan, dtype=np.float32)
    pair_ratio = np.full_like(pair_cos, np.nan)
    pair_rel = np.full_like(pair_cos, np.nan)
    cache_dtype = torch.float32 if precision == "float32" else torch.bfloat16
    parameter_count = sum(parameter.numel() for parameter in engine._parameters)
    ring = torch.empty((RING_SIZE, parameter_count), dtype=cache_dtype, device=device)
    norm_ring = torch.empty((RING_SIZE,), dtype=torch.float32, device=device)
    gradient_map = None
    gradient_partial = None
    if cache_destination is not None:
        cache_destination.parent.mkdir(parents=True, exist_ok=True)
        gradient_partial = cache_destination.with_name(cache_destination.name + ".partial")
        if gradient_partial.exists():
            gradient_partial.unlink()
        gradient_map = np.memmap(gradient_partial, dtype=np.uint16, mode="w+",
                                 shape=(n_windows, parameter_count))
        LOGGER.info("Retaining full BF16 gradients: path=%s windows=%d bytes=%d",
                    cache_destination, n_windows, n_windows * parameter_count * 2)
    LOGGER.info("Trajectory start: id=%s points=%d windows=%d precision=%s ring_gib=%.2f",
                series_id, len(values), n_windows, precision,
                RING_SIZE * parameter_count * torch.empty((), dtype=cache_dtype).element_size() / (1024 ** 3))
    gradients = []
    norms = []
    done = 0
    total_before_batch = 0

    def flush_batch() -> None:
        nonlocal done, total_before_batch, gradients, norms
        if not gradients:
            return
        batch = torch.stack(gradients)
        batch_norm = torch.stack(norms).float()
        batch_size = len(batch)
        if gradient_map is not None:
            gradient_map[done:done + batch_size] = (
                batch.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            )
        history_count = min(MAX_LAG, done)
        if history_count:
            history_start = done - history_count
            slots = np.arange(history_start, done, dtype=np.int64) % RING_SIZE
            gradient_batch_fp32 = batch.float()
            for tile_start in range(0, history_count, HISTORY_TILE):
                tile_stop = min(history_count, tile_start + HISTORY_TILE)
                tile_slots = torch.as_tensor(slots[tile_start:tile_stop], dtype=torch.long, device=device)
                old = ring.index_select(0, tile_slots)
                old_norm = norm_ring.index_select(0, tile_slots)
                dots = torch.matmul(gradient_batch_fp32, old.float().transpose(0, 1))
                dot_np = dots.detach().cpu().numpy()
                old_norm_np = old_norm.detach().cpu().numpy()
                cols = np.arange(tile_start, tile_stop, dtype=np.int64)
                current_norm_np = batch_norm.detach().cpu().numpy()
                for row in range(batch_size):
                    current_index = done + row
                    lags = row + history_count - cols
                    valid = (lags >= 1) & (lags <= MAX_LAG)
                    if not np.any(valid):
                        continue
                    lag_idx = lags[valid] - 1
                    prev_norm = old_norm_np[valid]
                    cosine = dot_np[row, valid] / np.maximum(current_norm_np[row] * prev_norm, 1e-30)
                    ratio = current_norm_np[row] / np.maximum(prev_norm, 1e-30)
                    relative = np.abs(current_norm_np[row] - prev_norm) / np.maximum(
                        np.maximum(current_norm_np[row], prev_norm), 1e-30)
                    pair_cos[current_index, lag_idx] = cosine
                    pair_ratio[current_index, lag_idx] = np.log10(np.maximum(ratio, 1e-30))
                    pair_rel[current_index, lag_idx] = relative
                del old, old_norm, dots
            del gradient_batch_fp32
        batch_fp32 = batch.float()
        within_dots = torch.matmul(batch_fp32, batch_fp32.transpose(0, 1)).detach().cpu().numpy()
        batch_norm_np = batch_norm.detach().cpu().numpy()
        for row in range(batch_size):
            current_index = done + row
            earlier = np.arange(row, dtype=np.int64)
            if len(earlier):
                lags = row - earlier
                previous_norms = batch_norm_np[earlier]
                cosine = within_dots[row, earlier] / np.maximum(batch_norm_np[row] * previous_norms, 1e-30)
                ratio = batch_norm_np[row] / np.maximum(previous_norms, 1e-30)
                relative = np.abs(batch_norm_np[row] - previous_norms) / np.maximum(
                    np.maximum(batch_norm_np[row], previous_norms), 1e-30)
                pair_cos[current_index, lags - 1] = cosine
                pair_ratio[current_index, lags - 1] = np.log10(np.maximum(ratio, 1e-30))
                pair_rel[current_index, lags - 1] = relative
        for row in range(batch_size):
            slot = (done + row) % RING_SIZE
            ring[slot].copy_(batch[row])
            norm_ring[slot].copy_(batch_norm[row])
        done += batch_size
        total_before_batch = done
        progress.update(batch_size)
        gradients = []
        norms = []
        del batch, batch_fp32, within_dots

    with torch.enable_grad():
        for start in range(n_windows):
            gradient, norm = _gradient(engine, values[start:start + WINDOW])
            gradients.append(gradient)
            norms.append(norm)
            if len(gradients) >= BATCH_GRADIENTS:
                flush_batch()
    flush_batch()
    if gradient_map is not None:
        gradient_map.flush()
        del gradient_map
        gradient_partial.replace(cache_destination)
        marker = cache_destination.with_suffix(".complete")
        marker.write_text("complete\n", encoding="ascii")
    del ring, norm_ring
    torch.cuda.empty_cache()

    summaries = {}
    for metric, matrix in (("cosine", pair_cos), ("log10_norm_ratio", pair_ratio),
                           ("relative_norm_difference", pair_rel)):
        summaries[metric] = _summarize_pairs(matrix, metric)
    _atomic_npz(destination, series_id=np.asarray(series_id), trajectory_windows=np.asarray(n_windows),
                precision=np.asarray(precision),
                **{metric + "_" + field: value for metric, result in summaries.items()
                   for field, value in zip(("count", "sum", "sum_sq", "min", "max", "hist"), result)})
    LOGGER.info("Trajectory complete: id=%s windows=%d pair_count=%d precision=%s",
                series_id, n_windows, int(summaries["cosine"][0].sum()), precision)
    return {"series_id": series_id, "windows": n_windows, "precision": precision,
            "pairs": int(summaries["cosine"][0].sum())}


def aggregate_trajectories(directory: Path, expected: int, output: Path) -> None:
    paths = sorted(directory.glob("trajectory-*.npz"))
    if len(paths) != expected:
        raise RuntimeError("trajectory summaries incomplete: %d/%d" % (len(paths), expected))
    global_state = {}
    metrics = ("cosine", "log10_norm_ratio", "relative_norm_difference")
    for metric in metrics:
        global_state[metric] = {field: None for field in ("count", "sum", "sum_sq", "min", "max", "hist")}
    for path in paths:
        with np.load(path, allow_pickle=False) as item:
            for metric in metrics:
                for field in global_state[metric]:
                    value = item[metric + "_" + field]
                    current = global_state[metric][field]
                    if current is None:
                        global_state[metric][field] = value.copy()
                    elif field == "min":
                        global_state[metric][field] = np.fmin(current, value)
                    elif field == "max":
                        global_state[metric][field] = np.fmax(current, value)
                    else:
                        global_state[metric][field] = current + value
    lags = np.arange(1, MAX_LAG + 1, dtype=np.int32)
    report = {"trajectories": expected, "lags": lags.tolist(), "overlap_ratio": ((WINDOW - lags) / WINDOW).tolist()}
    out_arrays = {"lags": lags}
    for metric in metrics:
        state = global_state[metric]
        count = state["count"]
        total = state["sum"]
        total_sq = state["sum_sq"]
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        variance = np.divide(total_sq, count, out=np.full_like(total_sq, np.nan), where=count > 0) - mean ** 2
        report[metric] = {"count": count.tolist(), "mean": mean.tolist(),
                          "std": np.sqrt(np.maximum(variance, 0)).tolist(),
                          "min": state["min"].tolist(), "max": state["max"].tolist()}
        for field in ("count", "sum", "sum_sq", "min", "max", "hist"):
            out_arrays[metric + "_" + field] = state[field]
        out_arrays[metric + "_mean"] = mean
        out_arrays[metric + "_std"] = np.sqrt(np.maximum(variance, 0))
    # Spearman correlation across lag-level means summarizes the overlap trend.
    from scipy.stats import spearmanr
    valid = np.isfinite(out_arrays["cosine_mean"])
    corr_cos = spearmanr(lags[valid], out_arrays["cosine_mean"][valid])
    valid_rel = np.isfinite(out_arrays["relative_norm_difference_mean"])
    corr_rel = spearmanr(lags[valid_rel], out_arrays["relative_norm_difference_mean"][valid_rel])
    report["lag_level_spearman"] = {"lag_vs_cosine": float(corr_cos.statistic),
                                     "lag_vs_relative_norm_difference": float(corr_rel.statistic)}
    write_json_atomic(output.with_suffix(".json"), report)
    _atomic_npz(output.with_suffix(".npz"), **out_arrays)


def run(config_path: Path) -> None:
    config = read_config(config_path)
    out = output_path("train", "outputs/train/v0-grad-mech-test")
    probe_dir = out / "probe" / "trajectories"
    probe_dir.mkdir(parents=True, exist_ok=True)
    selected, manifest_rows, warmup_windows = prepare_inputs(config, out)
    total_windows = sum(len(values) - WINDOW + 1 for _series_id, values in selected)
    LOGGER.info("Experiment config: model=%s data=%s subset=%s trajectories=%d total_windows=%d warmup_steps=100 batch=32 lr=1e-6",
                config["model"], config["source"], config["subset"], len(selected), total_windows)
    model_path = str(PROJECT_ROOT / "cache/pretrained_models/amazon/chronos-t5-tiny/29d808298f1a62493e7b9a5e08529d0d930fa189")
    checkpoint = out / "chronos-tiny-warmup-100"
    engine = load_engine(model_path if not (checkpoint / "config.json").is_file() else str(checkpoint), "float32")
    checkpoint = warmup(engine, warmup_windows, out, config)
    if str(engine.checkpoint_path) != str(checkpoint):
        del engine
        torch.cuda.empty_cache()
        engine = load_engine(str(checkpoint), "float32")
    engine.model.eval()
    precision = "float32"
    starts = sum(len(values) - WINDOW + 1 for _series_id, values in selected)
    retain_ordinals = [int(value) for value in config.get("retain_gradient_ordinals", [])]
    if len(retain_ordinals) != len(set(retain_ordinals)):
        raise ValueError("retained gradient ordinals must be unique")
    retain_map = {source_ordinal: cache_ordinal
                  for cache_ordinal, source_ordinal in enumerate(retain_ordinals)}
    gradient_dir = PROJECT_ROOT / "outputs/train/v0-crb-reconst-test/gradients"
    with tqdm(total=starts, desc="Full-gradient windows", unit="window", mininterval=10) as progress:
        for index, ((series_id, values), manifest_row) in enumerate(zip(selected, manifest_rows)):
            destination = probe_dir / ("trajectory-%04d.npz" % index)
            cache_ordinal = retain_map.get(index)
            cache_destination = (gradient_dir / ("trajectory-%02d.bf16" % cache_ordinal)
                                 if cache_ordinal is not None else None)
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    if str(saved["series_id"].item()) != series_id:
                        raise ValueError("completed trajectory identity mismatch")
                    progress.update(int(saved["trajectory_windows"]))
                continue
            started = time.monotonic()
            try:
                result = process_trajectory(engine, series_id, values, precision, destination,
                                            progress, cache_destination)
            except torch.cuda.OutOfMemoryError:
                LOGGER.exception("FP32 OOM for trajectory %s; switching full gradients/cache to BF16", series_id)
                torch.cuda.empty_cache()
                del engine
                engine = load_engine(str(checkpoint), "bfloat16")
                precision = "bfloat16"
                result = process_trajectory(engine, series_id, values, precision, destination,
                                            progress, cache_destination)
            LOGGER.info("Trajectory batch recorded: ordinal=%d/%d id=%s windows=%d pairs=%d elapsed_seconds=%.1f eta_hours=%.2f",
                        index + 1, len(selected), series_id, result["windows"], result["pairs"],
                        time.monotonic() - started,
                        (time.monotonic() - started) * (len(selected) - index - 1) / 3600.0)
            state_path = out / "probe" / "progress.json"
            write_json_atomic(state_path, {"completed_trajectories": index + 1,
                                           "total_trajectories": len(selected),
                                           "last_series_id": series_id,
                                           "precision_mode": precision,
                                           "elapsed_seconds": time.monotonic() - starts_at})
    summary_path = out / "probe" / "lag_summary"
    aggregate_trajectories(probe_dir, len(selected), summary_path)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("Experiment complete: trajectories=%d windows=%d output=%s", len(selected), total_windows, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rage/v0-grad-mech-test.json"))
    args = parser.parse_args()
    configure_logger("v0.grad_mech_test", "v0-grad-mech-test", "experiment", "WARMUP/GRADIENT-PROBE")
    starts_at = time.monotonic()
    run(args.config)
