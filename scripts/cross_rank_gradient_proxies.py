"""Compare cross-track proxy ranking with full-gradient cosine ranking."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

from configs import logger as logging_config
from configs.logger import configure_logger
from configs.paths import PROJECT_ROOT, output_path

LOGGER = logging.getLogger("v0.gradient_proxy_cross_rank")
WINDOW_SPAN = 576


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty result table")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_tracks(root: Path, cache_root: Path, dimension: int, source_tracks: list[int]) -> dict[int, dict]:
    tracks = {}
    for track in range(10):
        directory = root / ("trajectory-%02d" % track)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="ascii"))
        count = int(manifest["windows"])
        features = np.load(directory / "window_features.npz")
        starts = np.asarray(features["start"], dtype=np.int64)
        if len(starts) != count or not np.array_equal(starts, np.arange(count, dtype=np.int64)):
            raise ValueError("unexpected window start layout for track %d" % track)
        head = np.load(directory / "head_gradients.npy", mmap_mode="r")
        embedding = np.load(directory / "window_features.npz")["embedding"]
        loss = np.load(directory / "window_features.npz")["loss"]
        if head.shape != (count, 1048576):
            raise ValueError("unexpected head-gradient shape for track %d" % track)
        full_path = cache_root / ("trajectory-%02d.bf16" % track)
        complete = cache_root / ("trajectory-%02d.complete" % track)
        if not complete.exists() or full_path.stat().st_size != count * dimension * 2:
            raise ValueError("incomplete full-gradient cache for track %d" % track)
        full = np.memmap(full_path, mode="r", dtype=np.uint16, shape=(count, dimension))
        tracks[track] = {"manifest": manifest, "starts": starts, "head": head,
                         "embedding": embedding, "loss": loss, "full": full}
        if track in source_tracks:
            series_length = count + WINDOW_SPAN - 1
            candidate_count = series_length // WINDOW_SPAN
            if candidate_count < 1:
                raise ValueError("source track has no full-span window")
            selected_starts = [int(math.floor(i * series_length / candidate_count))
                               for i in range(candidate_count)]
            selected = []
            for start in selected_starts:
                if start < 0 or start + WINDOW_SPAN > series_length or start >= count:
                    raise ValueError("invalid source start %d for track %d" % (start, track))
                selected.append(start)
            if any(selected[i + 1] - selected[i] < WINDOW_SPAN for i in range(len(selected) - 1)):
                raise ValueError("source windows overlap in track %d" % track)
            tracks[track]["source_indices"] = np.asarray(selected, dtype=np.int64)
    return tracks


def to_gpu_bf16(array: np.ndarray, device: torch.device) -> torch.Tensor:
    copied = np.array(array, dtype=np.uint16, copy=True)
    return torch.from_numpy(copied).view(torch.bfloat16).to(device=device, dtype=torch.float32)


def safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = pearsonr(x, y).statistic if kind == "pearson" else spearmanr(x, y).statistic
    return float(result) if np.isfinite(result) else None


def rank_row(ref_track: int, ref_index: int, ref_start: int, full: np.ndarray,
             head: np.ndarray, embed: np.ndarray, loss: np.ndarray,
             source_meta: list[tuple[int, int, int]], top_k: int) -> tuple[dict, list[dict]]:
    full_order = np.argsort(-full, kind="stable")
    methods = {
        "head_cosine": head,
        "embedding_cosine": embed,
        "negative_loss_difference": -loss,
    }
    output = {"ref_track": ref_track, "ref_window_index": ref_index, "ref_start": ref_start,
              "candidate_count": len(full), "full_cosine_mean": float(np.mean(full))}
    pair_rows = []
    full_by_key = {}
    for source_position, (source_track, source_index, source_start) in enumerate(source_meta):
        pair_rows.append({"ref_track": ref_track, "ref_window_index": ref_index, "ref_start": ref_start,
                          "source_track": source_track, "source_window_index": source_index,
                          "source_start": source_start, "full_cosine": float(full[source_position]),
                          "head_cosine": float(head[source_position]),
                          "embedding_cosine": float(embed[source_position]),
                          "absolute_loss_difference": float(loss[source_position])})
        full_by_key[(source_track, source_index)] = source_position
    for name, values in methods.items():
        order = np.argsort(-values, kind="stable")
        output[name + "_pearson"] = safe_corr(full, values, "pearson")
        output[name + "_spearman"] = safe_corr(full, values, "spearman")
        tau = kendalltau(full, values).statistic if len(full) >= 2 else np.nan
        output[name + "_kendall_tau_b"] = float(tau) if np.isfinite(tau) else None
        full_top = {full_order[i] for i in range(min(top_k, len(full_order)))}
        proxy_top = {order[i] for i in range(min(top_k, len(order)))}
        output[name + "_top%d_overlap" % top_k] = len(full_top & proxy_top) / max(1, min(top_k, len(full_order)))
    head_delta = head - full
    embed_delta = embed - full
    output.update({"head_cosine_mae": float(np.mean(np.abs(head_delta))),
                   "head_cosine_rmse": float(np.sqrt(np.mean(head_delta ** 2))),
                   "head_cosine_bias": float(np.mean(head_delta)),
                   "embedding_cosine_mae": float(np.mean(np.abs(embed_delta))),
                   "embedding_cosine_rmse": float(np.sqrt(np.mean(embed_delta ** 2))),
                   "embedding_cosine_bias": float(np.mean(embed_delta))})
    return output, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-gradient-proxy-cross-rank.json")
    args = parser.parse_args()
    cfg = json.loads((PROJECT_ROOT / args.config).read_text(encoding="ascii"))
    if cfg.get("part_code") != "Cross-Rank" or int(cfg["window_span"]) != WINDOW_SPAN:
        raise ValueError("unexpected Cross-Rank protocol configuration")
    log_path = PROJECT_ROOT / "logs/v0-gradient-proxy-test/cross-rank.log"
    LOGGER = configure_logger("v0.gradient_proxy_cross_rank", "v0-gradient-proxy-test",
                              "cross-rank.log", "initialization", log_directory=log_path.parent)
    out = output_path("train", cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".complete").exists():
        LOGGER.info("[complete] existing part marker found; refusing to overwrite results")
        return
    source_tracks = [int(x) for x in cfg["source_tracks"]]
    reference_tracks = [int(x) for x in cfg["reference_tracks"]]
    if set(source_tracks) & set(reference_tracks) or sorted(source_tracks + reference_tracks) != list(range(10)):
        raise ValueError("reference and source tracks must partition all ten fixed tracks")
    proxy_root = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
    cache_root = PROJECT_ROOT / "outputs/train/v0-crb-reconst-test/gradients"
    layout = json.loads((proxy_root / "parameter_layout.json").read_text(encoding="ascii"))
    dimension = sum(int(item["count"]) for item in layout)
    started = time.monotonic()
    torch.set_num_threads(int(cfg["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Cross-Rank requires the configured CUDA device")
    LOGGER.info("[configuration] part=Cross-Rank ref_tracks=%s src_tracks=%s span=%d dimension=%d device=%s output=%s",
                reference_tracks, source_tracks, WINDOW_SPAN, dimension, device, out)
    LOGGER.info("[data] start validating ten manifests, proxy features, and full-gradient caches")
    tracks = load_tracks(proxy_root, cache_root, dimension, source_tracks)
    LOGGER.info("[data] end; all ten tracks validated")
    source_meta = [(track, int(index), int(tracks[track]["starts"][index]))
                   for track in source_tracks for index in tracks[track]["source_indices"]]
    if len(source_meta) < 3:
        raise ValueError("at least three source candidates are required")
    source_full_np = np.concatenate([np.asarray(tracks[t]["full"][tracks[t]["source_indices"]], dtype=np.uint16)
                                     for t in source_tracks], axis=0)
    source_full = torch.from_numpy(source_full_np).view(torch.bfloat16).to(device=device, dtype=torch.float32)
    del source_full_np
    source_full_norm = torch.linalg.vector_norm(source_full, dim=1)
    source_head_np = np.concatenate([np.asarray(tracks[t]["head"][tracks[t]["source_indices"]], dtype=np.float32)
                                     for t in source_tracks], axis=0)
    source_head = torch.from_numpy(source_head_np).to(device)
    del source_head_np
    source_head_norm = torch.linalg.vector_norm(source_head, dim=1)
    source_embed_np = np.concatenate([np.asarray(tracks[t]["embedding"][tracks[t]["source_indices"]], dtype=np.float32)
                                      for t in source_tracks], axis=0)
    source_embed = torch.from_numpy(source_embed_np).to(device)
    source_embed_norm = torch.linalg.vector_norm(source_embed, dim=1)
    source_loss = np.concatenate([np.asarray(tracks[t]["loss"][tracks[t]["source_indices"]], dtype=np.float64)
                                  for t in source_tracks])
    source_starts = np.asarray([item[2] for item in source_meta], dtype=np.int64)
    source_loss_gpu = torch.from_numpy(source_loss).to(device=device, dtype=torch.float64)
    LOGGER.info("[source] candidates=%d track_counts=%s all_spans_nonoverlap=true", len(source_meta),
                {str(t): int(len(tracks[t]["source_indices"])) for t in source_tracks})
    query_rows: list[dict] = []
    pair_rows: list[dict] = []
    reference_count = sum(len(tracks[t]["starts"]) for t in reference_tracks)
    batch_size = int(cfg["reference_batch_size"])
    processed = 0
    LOGGER.info("[ranking] start full-coordinate cross-track score computation")
    for ref_track in reference_tracks:
        track = tracks[ref_track]
        full = track["full"]
        head = track["head"]
        embedding = track["embedding"]
        loss = track["loss"]
        for begin in range(0, len(track["starts"]), batch_size):
            end = min(begin + batch_size, len(track["starts"]))
            ref_full = to_gpu_bf16(full[begin:end], device)
            ref_full_norm = torch.linalg.vector_norm(ref_full, dim=1)
            full_cos = (ref_full @ source_full.T) / (ref_full_norm[:, None] * source_full_norm[None, :]).clamp_min(1e-30)
            ref_head = torch.from_numpy(np.array(head[begin:end], dtype=np.float32, copy=True)).to(device)
            head_cos = (ref_head @ source_head.T) / (torch.linalg.vector_norm(ref_head, dim=1)[:, None] * source_head_norm[None, :]).clamp_min(1e-30)
            ref_embed = torch.from_numpy(np.array(embedding[begin:end], dtype=np.float32, copy=True)).to(device)
            embed_cos = (ref_embed @ source_embed.T) / (torch.linalg.vector_norm(ref_embed, dim=1)[:, None] * source_embed_norm[None, :]).clamp_min(1e-30)
            ref_loss = torch.from_numpy(np.asarray(loss[begin:end], dtype=np.float64).copy()).to(device)
            loss_difference = torch.abs(ref_loss[:, None] - source_loss_gpu[None, :])
            full_np = full_cos.clamp(-1, 1).cpu().numpy()
            head_np = head_cos.clamp(-1, 1).cpu().numpy()
            embed_np = embed_cos.clamp(-1, 1).cpu().numpy()
            loss_np = loss_difference.cpu().numpy()
            for offset in range(end - begin):
                ref_index = begin + offset
                qrow, p_rows = rank_row(ref_track, ref_index, int(track["starts"][ref_index]),
                                        full_np[offset], head_np[offset], embed_np[offset], loss_np[offset],
                                        source_meta, int(cfg["top_k"]))
                query_rows.append(qrow)
                pair_rows.extend(p_rows)
                processed += 1
            del ref_full, ref_head, ref_embed, ref_loss, full_cos, head_cos, embed_cos, loss_difference
            if time.monotonic() - started >= 30 and processed % 64 < batch_size:
                elapsed = time.monotonic() - started
                eta = elapsed * (reference_count - processed) / max(processed, 1)
                LOGGER.info("[progress] references=%d/%d pairs=%d elapsed=%s eta=%s",
                            processed, reference_count, len(pair_rows), logging_config.format_duration(elapsed),
                            logging_config.format_duration(eta))
    torch.cuda.synchronize()
    del source_full, source_head, source_embed
    if len(pair_rows) != reference_count * len(source_meta):
        raise RuntimeError("pair output count does not match the fixed design")
    LOGGER.info("[ranking] end; references=%d pairs=%d", reference_count, len(pair_rows))
    query_path = out / "query_ranking.csv"
    pair_path = out / "cross_track_pair_scores.csv"
    write_csv(query_path, query_rows)
    write_csv(pair_path, pair_rows)
    aggregate = {"part_code": "Cross-Rank", "reference_count": reference_count,
                 "source_candidate_count": len(source_meta), "pair_count": len(pair_rows),
                 "mean_full_cosine": float(np.mean([r["full_cosine"] for r in pair_rows])),
                 "proxy_summary": {}}
    for method in ("head_cosine", "embedding_cosine", "negative_loss_difference"):
        aggregate["proxy_summary"][method] = {}
        for metric in (method + "_pearson", method + "_spearman", method + "_kendall_tau_b", method + "_top%d_overlap" % int(cfg["top_k"])):
            finite = [float(row[metric]) for row in query_rows if row[metric] is not None and np.isfinite(row[metric])]
            aggregate["proxy_summary"][method][metric] = {
                "mean_query_weighted": float(np.mean(finite)),
                "median_query_weighted": float(np.median(finite)),
                "query_count": len(finite),
                "by_reference_track": {}
            }
            for ref_track in reference_tracks:
                values = [float(row[metric]) for row in query_rows if row["ref_track"] == ref_track and row[metric] is not None]
                aggregate["proxy_summary"][method][metric]["by_reference_track"][str(ref_track)] = {
                    "mean": float(np.mean(values)), "median": float(np.median(values)), "count": len(values)}
    pair_values = {
        "head_cosine": np.asarray([row["head_cosine"] for row in pair_rows], dtype=np.float64),
        "embedding_cosine": np.asarray([row["embedding_cosine"] for row in pair_rows], dtype=np.float64),
        "negative_loss_difference": -np.asarray([row["absolute_loss_difference"] for row in pair_rows], dtype=np.float64),
    }
    full_values = np.asarray([row["full_cosine"] for row in pair_rows], dtype=np.float64)
    for method, values in pair_values.items():
        aggregate["proxy_summary"][method]["pair_weighted_correlation"] = {
            "pearson": safe_corr(full_values, values, "pearson"),
            "spearman": safe_corr(full_values, values, "spearman"),
            "pair_count": len(values)}
    for method in ("head_cosine", "embedding_cosine"):
        errors = np.asarray([row[method.replace("_cosine", "_cosine_mae")] for row in query_rows], dtype=np.float64)
        biases = np.asarray([row[method.replace("_cosine", "_cosine_bias")] for row in query_rows], dtype=np.float64)
        aggregate["proxy_summary"][method]["calibration"] = {
            "mean_query_mae": float(np.mean(errors)), "mean_query_bias": float(np.mean(biases))}
    manifest = {"protocol": cfg, "reference_source_ordinals": [int(tracks[t]["manifest"]["ordinal"]) for t in reference_tracks],
                "reference_window_counts": {str(t): int(len(tracks[t]["starts"])) for t in reference_tracks},
                "source_tracks": {str(t): {"ordinal": int(tracks[t]["manifest"]["ordinal"]),
                                            "selected_starts": [int(x) for x in tracks[t]["starts"][tracks[t]["source_indices"]]],
                                            "candidate_count": int(len(tracks[t]["source_indices"]))}
                                   for t in source_tracks},
                "source_candidate_count": len(source_meta), "pair_count": len(pair_rows),
                "reference_rows_are_overlapping": True,
                "source_candidates_are_pairwise_nonoverlapping_within_track": True,
                "source_tracks_are_distinct_from_reference_tracks": True,
                "full_gradients": "retained BF16 copies of FP32 BP; all 8,394,496 coordinates used",
                "primary_proxy": "output-head gradient cosine",
                "secondary_proxies": ["decoder token-mean embedding cosine", "negative absolute CE loss difference"],
                "selection_metrics": ["per-reference Spearman", "per-reference Kendall tau-b", "per-reference top-5 overlap"],
                "calibration_metrics": ["head and embedding cosine MAE, RMSE, bias"],
                "elapsed_seconds": time.monotonic() - started}
    write_json(out / "input_manifest.json", manifest)
    write_json(out / "aggregate.json", aggregate)
    (out / ".complete").write_text("complete\n", encoding="ascii")
    LOGGER.info("[complete] references=%d candidates=%d pairs=%d elapsed=%s output=%s",
                reference_count, len(source_meta), len(pair_rows),
                logging_config.format_duration(time.monotonic() - started), out)


if __name__ == "__main__":
    main()
