"""Diagnose non-finite Timer-XL forecasts on the stage-1 validation windows."""
import json
import numpy as np

from rage.experiments.full_pool_exploration import development_split
from rage.experiments.full_pool_models import build_model

INPUTS = "/root/autodl-tmp/TS3/outputs/timerxl-calibration-20260920/inputs-dev"

validation, boundaries, history = development_split(
    INPUTS, ["boomlet_1676", "boomlet_619"], context=512, horizon=64, count=8)
engine, probe, identity = build_model("timerxl-84m", device="cuda")

report = {}
for name, kwargs in validation.items():
    windows = kwargs["test_windows"]
    contexts = windows[:, :512]
    bad = []
    for start in range(0, len(contexts), 32):
        batch = contexts[start:start + 32]
        forecast = engine.forecast(batch)
        finite = np.isfinite(forecast).all(axis=1)
        for i, ok in enumerate(finite):
            if not ok:
                ctx = batch[i]
                bad.append(dict(index=start + i, entity=kwargs["test_entities"][start + i],
                                ctx_min=float(ctx.min()), ctx_max=float(ctx.max()),
                                ctx_std=float(ctx.std()),
                                out_nan=int(np.isnan(forecast[i]).sum()),
                                out_inf=int(np.isinf(forecast[i]).sum())))
    report[name] = dict(n_windows=len(contexts), n_bad=len(bad), bad=bad[:10])
    print(name, "windows:", len(contexts), "bad:", len(bad))
    for row in bad[:5]:
        print("  ", json.dumps(row))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "bad"} for k, v in report.items()}))
