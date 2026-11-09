"""Measure matched per-window wall time for full BP and gradient proxies."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.engines import per_sample_cross_entropy
from scripts.exp.v0_grad_mech_test import load_engine, select_trajectories, _gradient


def run_method(engine, window, method, capture):
    torch.cuda.synchronize()
    started = time.perf_counter()
    if method == "full_bp":
        vector, _norm = _gradient(engine, window)
        result = vector.to(torch.bfloat16).cpu()
        del vector
    else:
        tensors = engine.batch_tensors(window.reshape(1, -1))
        capture.clear()
        kwargs = engine._logits_inputs(tensors)
        if method == "embedding_mean":
            kwargs["output_hidden_states"] = True
        with torch.no_grad():
            output = engine.model(**kwargs, return_dict=True)
            loss = per_sample_cross_entropy(output.logits, tensors["labels"])
            valid = tensors["labels"].ne(-100)
            if method == "forward_loss":
                result = loss.detach().cpu()
            elif method == "embedding_mean":
                hidden = output.decoder_hidden_states[-1]
                result = ((hidden * valid.unsqueeze(-1)).sum(1)
                          / valid.sum(1, keepdim=True)).detach().cpu()
            else:
                result = None
        if method == "head_gradient":
            local_logits = output.logits.detach().requires_grad_(True)
            local_loss = per_sample_cross_entropy(local_logits, tensors["labels"]).sum()
            logits_gradient = torch.autograd.grad(local_loss, local_logits)[0]
            head = torch.bmm(logits_gradient.transpose(1, 2), capture["head_input"].float())
            result = head.detach().to(torch.bfloat16).cpu()
            del local_logits, local_loss, logits_gradient, head
        elif method not in {"forward_loss", "embedding_mean"}:
            raise ValueError("unknown method: " + method)
        del output, tensors
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rage/v0-gradient-proxy-test.json")
    parser.add_argument("--windows-per-trajectory", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="ascii"))
    threads = int(config["cpu_threads"])
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    logger = configure_logger("proxy_cost", "v0-gradient-proxy-test", "cost", "cost")
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s-[proxy_cost]-[timing]-[%(message)s]"))
    output = output_path("train", "outputs/train/v0-gradient-proxy-cost")
    output.mkdir(parents=True, exist_ok=True)
    logger.info("[setup] start windows_per_trajectory=%d repeats=%d warmup=%d methods=%s",
                args.windows_per_trajectory, args.repeats, args.warmup,
                ["full_bp", "head_gradient", "embedding_mean", "forward_loss"])
    source = json.loads((PROJECT_ROOT / config["source_config"]).read_text(encoding="ascii"))
    trajectories, _manifest = select_trajectories(source)
    checkpoint = PROJECT_ROOT / config["checkpoint"]
    engine = load_engine(str(checkpoint), "float32")
    engine.model.eval()
    capture = {}
    hook = engine.model.lm_head.register_forward_pre_hook(
        lambda _module, inputs: capture.update(head_input=inputs[0].detach()))
    methods = ["full_bp", "head_gradient", "embedding_mean", "forward_loss"]
    windows = []
    for track, ordinal in enumerate(config["source_ordinals"]):
        series_id, values = trajectories[ordinal]
        n = len(values) - 575
        meta = json.loads((PROJECT_ROOT / ("outputs/train/v0-crb-reconst-test/batches/trajectory-%02d/config.json" % track)).read_text())
        cache_path = PROJECT_ROOT / config["cache_root"] / ("trajectory-%02d.bf16" % track)
        if meta["trajectory_ordinals"] != [ordinal] or meta["cache_ordinals"] != [track]:
            raise ValueError("source trajectory configuration mismatch")
        if not cache_path.with_suffix(".complete").is_file():
            raise ValueError("full-gradient cache is incomplete")
        starts = np.unique(np.linspace(0, n - 1, args.windows_per_trajectory, dtype=np.int64))
        for start in starts:
            windows.append({"track": track, "ordinal": ordinal, "series_id": series_id,
                            "start": int(start), "values": values[int(start):int(start)+576]})
    # Validate proxy implementations against the completed feature and head caches.
    validation = []
    for row in windows[::max(1, len(windows)//10)][:10]:
        dest = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test" / ("trajectory-%02d" % row["track"])
        features = np.load(dest / "window_features.npz")
        head_cache = np.load(dest / "head_gradients.npy", mmap_mode="r")
        local = {}
        for method in ("head_gradient", "embedding_mean", "forward_loss"):
            _elapsed, value = run_method(engine, row["values"], method, capture)
            if method == "head_gradient":
                expected = torch.from_numpy(np.asarray(head_cache[row["start"]]).copy()).to(torch.bfloat16)
                cosine = float(torch.nn.functional.cosine_similarity(value.flatten().float(), expected.flatten().float(), dim=0))
                if cosine < 0.999:
                    raise ValueError("head-gradient proxy does not match saved proxy")
                local["head_cache_cosine"] = cosine
            elif method == "embedding_mean":
                local["embedding_max_abs_error"] = float(np.max(np.abs(value.numpy()[0] - features["embedding"][row["start"]])))
            else:
                local["loss_abs_error"] = abs(float(value.numpy()[0]) - float(features["loss"][row["start"]]))
        validation.append({"track": row["track"], "start": row["start"], **local})
    # Warm every path, then interleave methods in a seeded order for each window.
    for method in methods:
        for row in windows[:args.warmup]:
            run_method(engine, row["values"], method, capture)
    order_rng = random.Random(config["seed"])
    measurements = []
    total = len(windows) * args.repeats * len(methods)
    logger.info("[measure] start sampled_windows=%d total_calls=%d",
                len(windows), total)
    with tqdm(total=total, desc="matched proxy cost", unit="call", mininterval=5) as bar:
        for repeat in range(args.repeats):
            order = list(windows)
            order_rng.shuffle(order)
            for row in order:
                method_order = list(methods)
                order_rng.shuffle(method_order)
                for method in method_order:
                    elapsed, _result = run_method(engine, row["values"], method, capture)
                    measurements.append({"track": row["track"], "ordinal": row["ordinal"],
                                         "series_id": row["series_id"], "start": row["start"],
                                         "repeat": repeat, "method": method,
                                         "seconds": elapsed})
                    bar.update(1)
    hook.remove()
    with (output / "per_window_timings.csv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    window_times = {}
    for row in measurements:
        key = (row["track"], row["start"], row["method"])
        window_times.setdefault(key, []).append(row["seconds"])
    window_medians = {key: float(np.median(values)) for key, values in window_times.items()}
    per_track = {}
    summaries = []
    full_median_by_window = {(track, start): window_medians[(track, start, "full_bp")]
                             for track, start in {(k[0], k[1]) for k in window_medians}}
    for method in methods:
        samples = np.array([value for (track, start, name), value in window_medians.items()
                            if name == method])
        track_medians = []
        for track in range(10):
            track_values = np.array([value for (track_id, _start, name), value in window_medians.items()
                                     if name == method and track_id == track])
            track_medians.append(float(np.median(track_values)))
        per_track[method] = track_medians
        paired_speedups = np.array([full_median_by_window[(track, start)] / value
                                    for (track, start, name), value in window_medians.items()
                                    if name == method])
        summaries.append({"method": method, "timed_calls": sum(1 for row in measurements if row["method"] == method),
                          "sampled_windows": len(samples),
                          "median_seconds": float(np.median(samples)),
                          "mean_seconds": float(np.mean(samples)),
                          "p90_seconds": float(np.quantile(samples, .9)),
                          "median_ms": float(np.median(samples)*1000),
                          "median_speedup_vs_full_bp": float(np.median(paired_speedups)),
                          "per_window_paired_speedup_min": float(np.min(paired_speedups)),
                          "per_window_paired_speedup_max": float(np.max(paired_speedups)),
                          "per_track_median_ms": [1000*v for v in track_medians]})
    report = {"experiment": "v0-gradient-proxy-cost", "sampled_windows": len(windows),
              "source_ordinals": config["source_ordinals"], "series_count": 10,
              "repeats_per_window": args.repeats, "warmup_calls_per_method": args.warmup,
              "batch_size": 1, "timer": "end-to-end Python wall time including CPU window tokenization, model forward, requested backward/proxy operation, CPU result copy, and CUDA synchronization",
              "device": torch.cuda.get_device_name(), "checkpoint": str(checkpoint),
              "validation": validation, "methods": summaries,
              "note": "Per-window latency is the median of three repeats. Speedup is paired by the same track and window. Per-track medians show across-track variation; selected windows and trajectories are not treated as independent population replicates."}
    temporary = output / "summary.json.partial"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    temporary.replace(output / "summary.json")
    (output / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("[measure] end summaries=%s", summaries)
    logger.info("[complete] duration=%s output=%s", format_duration(time.monotonic()-start_time), output)


if __name__ == "__main__":
    start_time = time.monotonic()
    try:
        main()
    except Exception:
        raise
