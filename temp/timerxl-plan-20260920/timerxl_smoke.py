"""Timer-XL GPU smoke (PLAN.md T0, 15-30 min budget): native loss, parameter update,
single-vs-batched per-window gradient consistency, finite deterministic forecast,
throughput and memory. Uses only safe-history prefixes of the boomlet dev inputs.

Run with the ts3-timerxl env python and HF_HUB_CACHE=/root/autodl-tmp/hf-integration/hub.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rags.experiments.development_inputs import load_fragments
from rags.experiments.full_pool_models import build_model
from rags.experiments.train_eval import fine_tune

INPUTS = Path("/root/autodl-tmp/TS3/outputs/timerxl-calibration-20260920/inputs-dev")
EXPECTED_PARAMS = 84_142_080


def history_windows(n, *, length=576):
    """First n non-overlapping windows from the stage-1 training-history prefix."""
    fragments = load_fragments(INPUTS / "boomlet_1676__reference.npz")
    windows = []
    for fragment in fragments:
        history = fragment.data[:5215]  # stage-1 reference_stop_exclusive; validation never touched
        for start in range(0, len(history) - length + 1, length):
            windows.append(history[start:start + length])
            if len(windows) >= n:
                return np.stack(windows).astype(np.float32)
    raise ValueError("not enough history windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"checks": {}}

    engine, probe, identity = build_model("timerxl-84m", device="cuda")
    params = sum(p.numel() for p in engine.model.parameters())
    assert params == EXPECTED_PARAMS, params
    report["identity"] = {k: identity[k] for k in
                          ("student", "model_id", "revision", "parameter_count",
                           "loaded_parameter_dtypes", "model_class")}
    patch = engine.model.config.input_token_len
    report["patch"] = patch
    report["context_semantics"] = ("training windows 576 = 6 patches (input 5, labels shifted 1); "
                                   "forecast tail-truncates 512 -> 480 with revin on (native generate)")

    windows = history_windows(96)

    # 1. Native training loss finite.
    loss = engine.training_loss(windows[:32])
    assert torch.isfinite(loss).all() if isinstance(loss, torch.Tensor) else np.isfinite(loss)
    report["checks"]["native_loss_finite"] = float(loss)

    # 2. Parameters actually update through the production fine_tune path.
    base = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
    log = fine_tune(engine, windows[:64], steps=3, batch_size=32, learning_rate=1e-5,
                    seed=0, weight_decay=0.01)
    changed = sum(1 for k, v in engine.model.state_dict().items()
                  if not torch.equal(v.detach().cpu(), base[k]))
    assert changed > 0 and log["steps"] == 3, (changed, log)
    report["checks"]["parameters_updated"] = {"tensors_changed": changed, "log": log}
    engine.model.load_state_dict(base)
    engine.model.eval()

    # 3. Single vs batched per-window gradient losses agree (eval mode, bf16 tolerance).
    batched = probe.per_window_losses(windows[:8]).detach().float().cpu().numpy()
    singles = np.stack([probe.per_window_losses(windows[i:i + 1]).detach().float().cpu().numpy()[0]
                        for i in range(8)])
    scale = np.maximum(np.abs(batched), 1e-3)
    max_rel = float(np.max(np.abs(batched - singles) / scale))
    assert max_rel < 2e-2, max_rel
    report["checks"]["per_window_loss_batch_consistency_max_rel"] = max_rel

    # 4. Finite deterministic forecast under the native generate semantics.
    contexts = np.stack([w[:512] for w in windows[:4]])
    forecast1 = engine.forecast(contexts)
    forecast2 = engine.forecast(contexts)
    assert forecast1.shape == (4, 64) and np.isfinite(forecast1).all()
    assert np.array_equal(forecast1, forecast2), "forecast not deterministic"
    report["checks"]["forecast_finite_deterministic"] = True

    # 5. Throughput and memory at the calibration batch size.
    engine.model.load_state_dict(base)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    log = fine_tune(engine, windows, steps=10, batch_size=32, learning_rate=1e-5,
                    seed=0, weight_decay=0.01)
    elapsed = time.monotonic() - started
    engine.model.load_state_dict(base)
    engine.model.eval()
    report["throughput"] = {"steps": 10, "batch_size": 32, "seconds_per_step": elapsed / 10,
                            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                            "final_loss": log["final_loss"]}

    (args.out / "smoke.json").write_text(json.dumps(report, indent=1) + "\n")
    print("TIMERXL_SMOKE_OK", json.dumps(report["throughput"], indent=1))


if __name__ == "__main__":
    main()
