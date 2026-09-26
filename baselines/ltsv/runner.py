"""ltsv model/data adapter and native selection workflow."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rags.core import TimeSeriesFragment
from rags.experiments.arms import ArmOutput, SelectionCost, WindowIndex
from baselines.common import _pool_fingerprint, reference_windows
from .algorithms import ltsv_window_scores
from rags.execution import forecast_mse, prepare_contexts
from .runtime import load_restore_timing, accounting


def _restore_snapshot_on_device(base_flat, device):
    device = torch.device(device)
    if device.type != 'cuda':
        return base_flat
    free, _ = torch.cuda.mem_get_info(device)
    if base_flat.numel() * base_flat.element_size() > free * .05:
        return base_flat
    try:
        return base_flat.to(device=device)
    except torch.cuda.OutOfMemoryError:
        return base_flat

def _ltsv_reference_mse(engine, windows, *, batch_size=32):
    """Context-only autoregressive forecast; future labels are only MSE targets.

    Chronos adapter uses deterministic greedy token decoding in original units.
    No teacher-forced future values are passed to the forecasting model.
    """
    if callable(getattr(engine, "forecast", None)):
        windows = np.asarray(windows, dtype=np.float32)
        context = engine.config.context_length
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = windows[start:start + batch_size]
                forecast = np.asarray(engine.forecast(batch[:, :context]))
                total += float(np.square(forecast - batch[:, context:]).sum())
        return total / (len(windows) * engine.config.prediction_length)
    return forecast_mse(engine, windows, batch_size=batch_size)


def _ltsv_training_loss(engine, windows):
    if callable(getattr(engine, "training_loss", None)):
        return engine.training_loss(windows)
    from rags.engines import per_sample_cross_entropy
    tensors = engine.batch_tensors(windows)
    output = engine.model(**tensors, return_dict=True)
    return per_sample_cross_entropy(output.logits, tensors["labels"]).mean()


def ltsv_select(engine, *, reference_fragments, index: WindowIndex,
                context_length: int, prediction_length: int,
                n_select: int | None = None, block_length: int | None = None,
                block_stride: int | None = None, icf_lr: float = 1e-5,
                context_windows: int = 64, cache_dir: Path | None = None,
                pool_manifest: dict | None = None, dataset_tag: str = "",
                seed: int = 0, resampling=None) -> ArmOutput:
    """LTSV: context-loss change after one-step in-context fine-tuning.

    Overlapping fixed-length blocks on every pool trajectory; per block one
    Adam step (lr 1e-5, weight decay 0.1, grad clip 1.0) on the block's
    native probe training loss; block value = forecast MSE before - after.
    Point values average covering blocks (Eq. 10), window values average
    covered points (Eq. 11, vectorised via prefix sums).  Because the context
    loss is reference-dependent, per-dataset window scores are cached.
    """
    execution_path = Path(cache_dir) / 'ltsv_execution.json' if cache_dir is not None else None
    if execution_path is not None and execution_path.exists():
        from .fast import optimized_select
        return optimized_select(engine, reference_fragments=reference_fragments, index=index,
            context_length=context_length, prediction_length=prediction_length,
            n_select=n_select, block_length=block_length, block_stride=block_stride,
            icf_lr=icf_lr, context_windows=context_windows, cache_dir=cache_dir,
            pool_manifest=pool_manifest, dataset_tag=dataset_tag, seed=seed,
            resampling=resampling, execution=json.loads(execution_path.read_text()))
    started = time.time()
    model = engine.model
    window_length = context_length + prediction_length
    block_length = block_length or window_length
    block_stride = block_stride or block_length // 2
    device = engine.device
    if block_length < window_length or block_stride < 1 or block_stride > block_length:
        raise ValueError("LTSV requires block_length >= C+H and 1 <= stride <= block_length")
    if next(model.parameters()).dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("LTSV requires a supported valuation dtype")

    cache = None
    if cache_dir is not None:
        tag = "%s_%s_%s" % (_pool_fingerprint(pool_manifest),
                            hashlib.sha1(dataset_tag.encode()).hexdigest()[:8],
                            hashlib.sha1(json.dumps(
                                {"bl": block_length, "bs": block_stride,
                                 "lr": icf_lr, "cw": context_windows}).encode()
                            ).hexdigest()[:8])
        cache = Path(cache_dir) / ("ltsv_scores_v2_mse_%s.npz" % tag)
    scores_cached = cache is not None and cache.exists()
    runtime_adjustment = None
    if scores_cached:
        with np.load(cache) as payload:
            window_scores = payload["scores"]
            block_info = json.loads(str(payload["meta"]))
        icf_steps = block_info["blocks"]
        print("  [LTSV] loaded cached window scores: %s" % cache, flush=True)
    else:
        val_windows, _ids = reference_windows(
            reference_fragments, context_length=context_length,
            prediction_length=prediction_length, stride=16,
            max_windows=context_windows, seed=seed)
        if not callable(getattr(engine, "forecast", None)):
            val_windows = prepare_contexts(engine, val_windows)

        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        # Immutable restoration weights live on CPU, as in student training.
        # Keeping a second full TimeMoE model on GPU crowds out Adam moments.
        base_flat = torch.cat([p.detach().cpu().reshape(-1) for _n, p in named])
        # Enable this method-specific acceleration only with explicit paired
        # timings, so reports can restore the unoptimized baseline's cost.
        calibration = load_restore_timing(cache_dir, (resampling or {}).get('model'))
        setup_seconds = 0.
        if calibration is not None and torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
            setup_started = time.perf_counter()
            base_flat = _restore_snapshot_on_device(base_flat, device)
            torch.cuda.synchronize(device)
            setup_seconds = time.perf_counter() - setup_started
        if base_flat.device.type != 'cuda':
            calibration = None
        print('  [LTSV] restore snapshot=%s; runtime basis=%s' % (
            base_flat.device, 'unoptimized estimate (calibrated)' if calibration else
            'unoptimized execution'), flush=True)
        offsets = np.cumsum([0] + [p.numel() for _n, p in named])

        def restore_base() -> None:
            cursor = 0
            for (_name, p), stop in zip(named, offsets[1:]):
                count = stop - cursor
                with torch.no_grad():
                    p.copy_(base_flat[cursor:stop].view_as(p))
                cursor = stop

        def context_loss() -> float:
            model.eval()
            return _ltsv_reference_mse(engine, val_windows)

        model.eval()
        base_context_loss = context_loss()

        window_scores = np.full(len(index), -np.inf, dtype=np.float32)
        icf_steps = 0
        block_count = 0
        for fragment in index.by_fragment.values():
            data = fragment.data
            max_start = fragment.length - block_length
            if max_start < 0:
                continue
            lo = index.bounds(fragment.fragment_id)[0]
            sums = np.zeros(fragment.length + 1, dtype=np.float64)
            counts = np.zeros(fragment.length + 1, dtype=np.int64)
            starts = np.arange(0, max_start + 1, block_stride)
            if starts[-1] != max_start:
                starts = np.append(starts, max_start)
            for block_start in starts:
                window = np.asarray(
                    data[int(block_start):int(block_start) + block_length],
                    dtype=np.float32)
                if len(window) < window_length:
                    continue
                # A longer block supplies all C/H subwindows to one update.
                block_windows = np.lib.stride_tricks.sliding_window_view(
                    window, window_length).copy()
                optimizer = torch.optim.Adam(model.parameters(), lr=icf_lr, foreach=False,
                                             betas=(0.9, 0.999), eps=1e-8,
                                             weight_decay=0.1)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _ltsv_training_loss(engine, block_windows)
                try:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    icf_steps += 1
                    block_count += 1
                    value = base_context_loss - context_loss()
                finally:
                    restore_base()
                    model.eval()
                stop = min(int(block_start) + block_length, fragment.length)
                sums[int(block_start):stop] += value
                counts[int(block_start):stop] += 1
                if block_count % 500 == 0:
                    print("  [LTSV] %d blocks (%.0fs)" % (
                        block_count, time.time() - started), flush=True)
            # Eqs. 10-11: point average over covering blocks, then window
            # average over covered points (prefix sums, vectorised)
            legal_starts = fragment.valid_starts(context_length,
                                                 prediction_length, index.stride)
            window_scores[lo:lo + len(legal_starts)] = ltsv_window_scores(
                sums, counts, legal_starts - 1, window_length)
        runtime_adjustment = accounting(calibration, icf_steps, setup_seconds)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, scores=window_scores,
                     meta=json.dumps({"blocks": icf_steps}))
    if resampling is not None:
        from rags.experiments.resampling import window_distribution
        selected = window_distribution(index, window_scores, pool_manifest=pool_manifest,
            metadata=resampling, temperature=resampling.get("temperature", .1))
        return ArmOutput("ltsv", selected, [], SelectionCost(
            gradient_computations=icf_steps, feature_computations=icf_steps * context_windows,
            notes={"scores_cached": scores_cached, "runtime_adjustment": runtime_adjustment}),
            {"blocks": int(icf_steps), "context_windows": context_windows,
             "block_length": block_length, "block_stride": block_stride,
             "reference_loss": "forecast_mse", "resampling": selected.coverage(),
             "native_kernel": "block_to_point_mean_to_window_mean"})
    scored = np.nonzero(np.isfinite(window_scores))[0]
    chosen_local = np.argsort(window_scores[scored])[::-1][:n_select]
    chosen = scored[chosen_local]
    return ArmOutput(
        name="ltsv",
        selected=index.windows_at(chosen),
        window_ids=[(int(f), int(s)) for f, s in zip(*index.fragment_arrays(chosen))],
        cost=SelectionCost(
            gradient_computations=icf_steps,
            feature_computations=icf_steps * context_windows,
            wall_seconds=0.0,
            notes={"scores_cached": scores_cached, "runtime_adjustment": runtime_adjustment},
        ),
        diagnostics={
            "selected": int(len(chosen)),
            "blocks": int(icf_steps),
            "context_windows": context_windows,
            "block_length": block_length,
            "block_stride": block_stride,
            "wall_seconds": round(time.time() - started, 1),
            "deployment": "native_top_k_overlap_allowed",
            "reference_loss": "forecast_mse",
            "probe_model": str(getattr(engine.config, "model_path", "unspecified")),
            "adaptation": ("native probe loss for block training; context-only deterministic "
                           "probe forecasts for reference MSE; longer blocks use "
                           "mean subwindow training loss in one optimizer step"),
        },
    )
