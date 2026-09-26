"""Diagnostic 2: step ladder — is the 3,000-step schedule simply over-training?

Rebuild the formal batch's random-arm selection exactly (pool seed 0, slot
grid, seed 101), cross-check the window ids against the archived run, then
train ONE seed-0 pass to 3,000 steps with checkpoints at
{250, 500, 1000, 2000, 3000} and evaluate each on the native exam.

Built-in validations:
  * ids match the archive  -> pool rebuild is identical to the formal batch;
  * the 3,000 checkpoint   -> should reproduce archived random seed0 MASE.

Readings:
  * some early checkpoint >= original (0.3297) -> schedule over-training;
  * flat-degraded at every step               -> training harmful from ~0.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rags.engines import ChronosT5Config, ChronosT5Engine
from rags.experiments.arms import WindowIndex, random_select
from rags.experiments.tasks import build_ratio_pool, build_task_data, build_task_specs
from rags.experiments.train_eval import evaluate

OUT_DIR = Path("/root/TS3/diag/0908")
MARKS = (250, 500, 1000, 2000, 3000)
ARCHIVE = ("/root/autodl-tmp/TS3/outputs_archive/"
           "rags-main-0908-INVALID-20260908/results.json")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def main():
    pool, manifest = build_ratio_pool(
        context_length=512, prediction_length=64, equivalents=20480,
        seed=0, cache_dir=OUT_DIR / "cache")
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    sel = random_select(index, n_select=4096, seed=101)
    windows = np.asarray(sel.selected, dtype=np.float32)
    log("pool rebuilt: %d trajectories; random selection: %d windows" % (
        len(pool), len(windows)))

    with open(ARCHIVE) as fh:
        archived_ids = [tuple(x) for x in
                        json.load(fh)["selection_records"]["boomlet_1676__random"]["ids"]]
    ids_match = [tuple(x) for x in sel.window_ids] == archived_ids
    log("ids match archive: %s" % ids_match)

    spec = {s.dataset: s for s in build_task_specs(series_cap=None)}["boomlet_1676"]
    task = build_task_data(spec, context_length=512, prediction_length=64,
                           series_cap=None, pool_fragments=[])

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    from rags.engines import per_sample_cross_entropy

    model = engine.model
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # value-range audit (train windows vs exam windows)
    audit = {
        "train_abs_p50": float(np.percentile(np.abs(windows), 50)),
        "train_abs_p99": float(np.percentile(np.abs(windows), 99)),
        "test_abs_p50": float(np.percentile(np.abs(task.test_windows), 50)),
        "test_abs_p99": float(np.percentile(np.abs(task.test_windows), 99)),
    }
    log("range audit: %s" % audit)

    # ---- one seed-0 training pass with checkpoints (mirrors fine_tune) -----
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    rng = np.random.default_rng(0)
    checkpoints = {}
    losses = []
    model.train()
    t0 = time.time()
    for step in range(1, MARKS[-1] + 1):
        indices = rng.integers(0, len(windows), size=min(32, len(windows)))
        tensors = engine.batch_tensors(windows[indices])
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors, return_dict=True)
        loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step in MARKS:
            checkpoints[step] = {k: v.detach().clone()
                                 for k, v in model.state_dict().items()}
            log("step %d: loss %.3f (%.0fs elapsed)" % (
                step, losses[-1], time.time() - t0))
    model.eval()

    ev_kw = dict(test_entities=task.test_entities, context_length=512,
                 test_scales=task.test_scales)
    ladder = {}
    for mark in (0,) + MARKS:
        if mark == 0:
            model.load_state_dict(base, strict=True)
        else:
            model.load_state_dict(checkpoints[mark], strict=True)
        model.eval()
        m = evaluate(engine, task.test_windows, **ev_kw)
        ladder[mark] = {"mase": m["mase"], "sql": m["sql"], "mae": m["mae"]}
        log("eval @ %4d steps: MASE %.4f SQL %.4f" % (
            mark, m["mase"], m["sql"]))

    model.load_state_dict(base, strict=True)
    result = {
        "dataset": "boomlet_1676", "arm": "random", "seed": 0,
        "ids_match_archive": ids_match,
        "range_audit": audit,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_curve_every50": losses[::50],
        "ladder": ladder,
        "original_mase": 0.3297,
        "archived_random_seed0_mase": 0.3479,
    }
    with open(OUT_DIR / "diag2_steps.json", "w") as fh:
        json.dump(result, fh, indent=1)
    log("done -> %s" % (OUT_DIR / "diag2_steps.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
