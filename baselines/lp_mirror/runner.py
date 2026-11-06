"""lp_mirror model/data adapter and native selection workflow."""
from __future__ import annotations

import hashlib
import json
import math
import time
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rage.core import TimeSeriesFragment
from rage.experiments.arms import ArmOutput, SelectionCost, WindowIndex
from baselines.common import _pool_fingerprint, reference_windows
from .algorithms import lp_mirror_scores

def lp_mirror_select(engine, *, reference_fragments, index: WindowIndex,
                     context_length: int, prediction_length: int,
                     n_select: int | None = None, proxy_epochs: int = 1,
                     proxy_batch: int = 4096, proxy_lr: float = 0.01,
                     adapt_lr: float = 1e-3, adaptation_epochs: int = 1,
                     adapt_batch: int = 256, reference_stride: int = 8,
                     reference_cap: int = 4096, cache_dir: Path | None = None,
                     pool_manifest: dict | None = None,
                     seed: int = 0, resampling=None, mapping="rank",
                     prepared_proxy=None, prepare_only=False) -> ArmOutput:
    """LP-Mirror ForwardINF: proxy loss drop after target adaptation.

    Port of ``ForwardINFSelector`` (train a linear proxy on the pool, SGD
    adapt on target windows, score = loss_orig - loss_adapted).  Windows are
    standardised by safe trajectory statistics; pool windows are extracted on the fly
    so the pool is never materialised; the proxy is cached across datasets.
    """
    torch.manual_seed(seed)
    started = time.time()
    device = engine.device
    window_length = context_length + prediction_length

    dtype = (next(engine.model.parameters()).dtype if hasattr(engine, "model")
             else (torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32))
    proxy = nn.Linear(context_length, prediction_length).to(device=device, dtype=dtype)
    if prepared_proxy is None:
        cache = None
        if cache_dir is not None:
            cache = Path(cache_dir) / ("lp_mirror_proxy_v2_%s_%s_%s_%s_%s_%s_%s.pt"
                                       % (_pool_fingerprint(pool_manifest), context_length,
                                          prediction_length, proxy_epochs, proxy_batch, proxy_lr, seed))
            if cache.exists():
                proxy.load_state_dict(torch.load(cache, map_location=device))
                proxy.eval()
                proxy_steps = -1  # cached
            else:
                proxy_steps = _train_lp_proxy(proxy, index, device,
                                              context_length=context_length,
                                              epochs=proxy_epochs,
                                              batch=proxy_batch, lr=proxy_lr, seed=seed)
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                # A killed preparation must not leave a loadable-looking partial proxy.
                with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        torch.save(proxy.state_dict(), stream)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                os.replace(temporary, cache)
        else:
            proxy_steps = _train_lp_proxy(proxy, index, device,
                                          context_length=context_length,
                                          epochs=proxy_epochs, batch=proxy_batch,
                                          lr=proxy_lr, seed=seed)
        proxy.eval()
        original_state = {k: v.detach().clone() for k, v in proxy.state_dict().items()}

        val_windows, _ids = reference_windows(
            reference_fragments, context_length=context_length,
            prediction_length=prediction_length, stride=reference_stride,
            max_windows=reference_cap, seed=seed)
        val_z = torch.from_numpy(_trajectory_standardise(
            val_windows, np.asarray([fid for fid, _ in _ids]),
            {f.fragment_id: f for f in reference_fragments})).to(device=device, dtype=dtype)
        adapt_optimizer = torch.optim.SGD(proxy.parameters(), lr=adapt_lr)
        proxy.train()
        adapt_steps = 0
        target_loader = torch.utils.data.DataLoader(val_z, batch_size=adapt_batch, shuffle=True)
        for _epoch in range(adaptation_epochs):
            for z in target_loader:
                x, y = z[:, :context_length], z[:, context_length:]
                adapt_optimizer.zero_grad()
                loss = nn.functional.mse_loss(proxy(x).float(), y.float())
                loss.backward()
                adapt_optimizer.step()
                adapt_steps += 1
        adapted_state = {k: v.detach().clone() for k, v in proxy.state_dict().items()}
        proxy.eval()

    else:
        if prepare_only:
            raise ValueError('prepare_only and prepared_proxy are mutually exclusive')
        original_state = prepared_proxy['original_state']
        adapted_state = prepared_proxy['adapted_state']
        proxy_steps = prepared_proxy['proxy_steps']
        adapt_steps = prepared_proxy['adapt_steps']
    if prepare_only:
        return dict(original_state={k:v.cpu() for k,v in original_state.items()},
                    adapted_state={k:v.cpu() for k,v in adapted_state.items()},
                    proxy_steps=proxy_steps, adapt_steps=adapt_steps)

    scores = np.empty(len(index), dtype=np.float32)
    eval_chunk = proxy_batch
    with torch.no_grad():
        for begin in range(0, len(index), eval_chunk):
            stop = min(len(index), begin + eval_chunk)
            batch = torch.from_numpy(_lp_windows(
                index, np.arange(begin, stop, dtype=np.int64))).to(device=device, dtype=dtype)
            x, y = batch[:, :context_length], batch[:, context_length:]
            proxy.load_state_dict(original_state)
            loss_orig = nn.functional.mse_loss(proxy(x).float(), y.float(), reduction="none").mean(dim=1)
            proxy.load_state_dict(adapted_state)
            loss_adapt = nn.functional.mse_loss(proxy(x).float(), y.float(), reduction="none").mean(dim=1)
            scores[begin:stop] = lp_mirror_scores(
                loss_orig.cpu().numpy(), loss_adapt.cpu().numpy(),
            )

    if resampling is not None:
        from rage.experiments.resampling import window_distribution
        selected = window_distribution(index, scores, pool_manifest=pool_manifest,
            metadata=resampling, temperature=resampling.get("temperature", .1), mapping=mapping)
        return ArmOutput("lp_mirror", selected, [], SelectionCost(
            proxy_training_steps=max(0, proxy_steps) + adapt_steps,
            feature_computations=len(index), index_bytes=scores.nbytes,
            wall_seconds=time.time() - started, notes={"proxy_cached": proxy_steps < 0}),
            {"proxy_steps": proxy_steps, "adapt_steps": adapt_steps,
             "resampling": selected.coverage(), "native_kernel": "proxy_loss_drop"})
    chosen = np.argsort(scores)[::-1][:n_select]
    chosen_ids = [(int(f), int(s)) for f, s in zip(*index.fragment_arrays(chosen))]
    trained_steps = proxy_steps if proxy_steps >= 0 else 0
    return ArmOutput(
        name="lp_mirror",
        selected=index.windows_at(chosen),
        window_ids=chosen_ids,
        cost=SelectionCost(
            proxy_training_steps=trained_steps + adapt_steps,
            feature_computations=len(index),
            index_bytes=int(scores.nbytes),
            wall_seconds=time.time() - started,
            notes={"proxy_cached": proxy_steps < 0},
        ),
        diagnostics={
            "selected": len(chosen_ids),
            "proxy_steps": proxy_steps,
            "adapt_steps": adapt_steps,
            "adaptation_epochs": adaptation_epochs,
            "deployment": "native_top_k_overlap_allowed",
            "pool_windows_scored": int(len(index)),
            "adaptation": ("univariate windows standardised by safe trajectory statistics; "
                           "channel-independent linear proxy over the time axis; "
                           "pool windows extracted on the fly"),
        },
    )


def _trajectory_standardise(windows, fragment_ids, fragments):
    """Dataset/channel scaling fitted only on the supplied safe histories."""
    result = windows.astype(np.float32, copy=True)
    for fid in np.unique(fragment_ids):
        data = np.asarray(fragments[int(fid)].data, dtype=np.float64)
        scale = float(data.std()) or 1.0
        mask = fragment_ids == fid
        result[mask] = (result[mask] - float(data.mean())) / scale
    return result


def _lp_windows(index, positions):
    return _trajectory_standardise(index.windows_at(positions),
                                   index.fragment_arrays(positions)[0], index.by_fragment)


def _train_lp_proxy(proxy, index, device, *, context_length: int, epochs: int,
                    batch: int, lr: float, seed: int) -> int:
    optimizer = torch.optim.Adam(proxy.parameters(), lr=lr, foreach=False)
    criterion = nn.MSELoss()
    steps = 0
    pool_size = len(index)
    for _epoch in range(epochs):
        for begin in range(0, pool_size, batch):
            positions = np.arange(begin, min(begin + batch, pool_size), dtype=np.int64)
            z = _lp_windows(index, positions)
            batch_t = torch.from_numpy(z).to(device=device, dtype=next(proxy.parameters()).dtype)
            x, y = batch_t[:, :context_length], batch_t[:, context_length:]
            optimizer.zero_grad()
            loss = criterion(proxy(x).float(), y.float())
            loss.backward()
            optimizer.step()
            steps += 1
    return steps
