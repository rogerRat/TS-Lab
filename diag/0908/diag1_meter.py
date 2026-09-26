"""Diagnostic 1: is the evaluation meter working at all?

Two direction-guaranteed checks on boomlet_1676:
  A. determinism — the same pretrained model evaluated twice must give
     bit-identical numbers (and reproduce the archived original cell).
  B. canary      — weights corrupted (x10) must eval dramatically worse.

Pass criteria: identical twice AND canary MASE >> original (>= 2x or NaN).
"""
import json
import time

import numpy as np
import torch

from rags.engines import ChronosT5Config, ChronosT5Engine
from rags.experiments.tasks import build_task_data, build_task_specs
from rags.experiments.train_eval import evaluate

OUT_PATH = "/root/TS3/diag/0908/diag1_meter.json"
EXPECT_ORIGINAL_MASE = 0.3297  # archived formal-batch original cell (seed0)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    spec = {s.dataset: s for s in build_task_specs(series_cap=None)}["boomlet_1676"]
    task = build_task_data(spec, context_length=512, prediction_length=64,
                           series_cap=None, pool_fragments=[])
    log("task: %d ref series, %d test windows" % (
        len(task.reference_fragments), len(task.test_windows)))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    log("engine ready: %d params" % engine.parameter_count)

    ev_kw = dict(test_entities=task.test_entities, context_length=512,
                 test_scales=task.test_scales)

    # --- A. determinism ------------------------------------------------------
    t0 = time.time()
    m1 = evaluate(engine, task.test_windows, **ev_kw)
    m2 = evaluate(engine, task.test_windows, **ev_kw)
    identical = (m1["mase"] == m2["mase"]) and (m1["sql"] == m2["sql"])
    log("A determinism: mase %.6f vs %.6f identical=%s (%.0fs)" % (
        m1["mase"], m2["mase"], identical, time.time() - t0))

    # --- B. canary: corrupt weights, must degrade ----------------------------
    with torch.no_grad():
        for p in engine.model.parameters():
            if p.dtype.is_floating_point:
                p.mul_(10.0)
    m3 = evaluate(engine, task.test_windows, **ev_kw)
    canary_mase = float(m3["mase"])
    degraded = (not np.isfinite(canary_mase)) or canary_mase >= 2 * m1["mase"]
    log("B canary x10: mase %r sql %r degraded=%s" % (
        canary_mase, m3["sql"], degraded))

    result = {
        "dataset": "boomlet_1676",
        "determinism": {
            "mase_run1": m1["mase"], "mase_run2": m2["mase"],
            "sql_run1": m1["sql"], "sql_run2": m2["sql"],
            "identical": bool(identical),
        },
        "reproduce_archived_original": {
            "mase": m1["mase"], "archived": EXPECT_ORIGINAL_MASE,
            "abs_diff": abs(m1["mase"] - EXPECT_ORIGINAL_MASE),
            "matches": abs(m1["mase"] - EXPECT_ORIGINAL_MASE) < 1e-6,
        },
        "canary_x10": {"mase": canary_mase, "sql": float(m3["sql"]),
                       "degraded": bool(degraded)},
        "verdict": "METER OK" if (identical and degraded) else "METER BROKEN",
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(result, fh, indent=1)
    log("VERDICT: %s -> %s" % (result["verdict"], OUT_PATH))
    return 0 if identical and degraded else 1


if __name__ == "__main__":
    raise SystemExit(main())
