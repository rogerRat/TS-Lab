"""Diagnostic 5: merged six-dataset task + regime grid (user direction
2026-09-08).

The six FEV tasks are merged into ONE heterogeneous target: training data =
even disjoint 576-point slots of every dataset's safe history, probe = odd
slots, exam = the union of all six native exams (equal-weight per-dataset
MASE, main-table aggregation).  Merging raises reference heterogeneity so
the task is no longer trivially "learnable at a glance".

Grid: mechanism {full, decoder-only (encoder frozen), head-only (tied
embeddings/output head)} x lr x step ladder with checkpoints.  After the
grid, retrain the RANDOM pool selection under the best regime and evaluate
on the merged exam (the actual selection setting).

Success criterion: in-domain regime with merged-exam MASE clearly below the
no-train baseline (>= ~2%, above the ~0.2-0.5% seed-noise band), then pool
under that regime at least neutral.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rags.engines import ChronosT5Config, ChronosT5Engine, per_sample_cross_entropy
from rags.experiments.arms import WindowIndex, random_select
from rags.experiments.tasks import build_ratio_pool, build_task_data, build_task_specs
from rags.experiments.train_eval import evaluate

OUT_DIR = Path("/root/TS3/diag/0908")
WL = 576
MARKS = (25, 50, 100, 200, 400, 800)
MECH_LRS = {
    "full": (3e-5, 1e-4, 3e-4),
    "decoder": (1e-4, 3e-4, 1e-3),
    "head": (3e-4, 1e-3, 3e-3),
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def dataset_of(entity):
    return entity.split("::", 1)[0]


def exam_report(metrics):
    per = {}
    for entity, value in metrics["per_series"]["mase"].items():
        per.setdefault(dataset_of(entity), []).append(value)
    per_dataset = {d: float(np.mean(v)) for d, v in per.items()}
    overall = float(np.mean(list(per_dataset.values())))
    return overall, per_dataset


def probe_ce(engine, windows, batch_size=256):
    model = engine.model
    model.eval()
    out = []
    with torch.no_grad():
        for begin in range(0, len(windows), batch_size):
            tensors = engine.batch_tensors(windows[begin:begin + batch_size])
            output = model(**tensors, return_dict=True)
            out.append(per_sample_cross_entropy(
                output.logits, tensors["labels"]).cpu().numpy())
    return float(np.concatenate(out).mean())


def trainable_params(model, mechanism):
    if mechanism == "full":
        return [p for p in model.parameters() if p.requires_grad]
    named = list(model.named_parameters())
    if mechanism == "decoder":
        keep = {id(p) for n, p in named if not n.startswith("encoder.")}
    elif mechanism == "head":
        keep = {id(p) for n, p in named
                if n in ("shared.weight", "lm_head.weight")}
    else:
        raise ValueError(mechanism)
    return [p for p in model.parameters() if id(p) in keep]


def train_checkpoints(engine, windows, *, params, lr, seed=0):
    model = engine.model
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    windows = np.asarray(windows, dtype=np.float32)
    checkpoints = {}
    losses = []
    model.train()
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
            checkpoints[step] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
    model.eval()
    return checkpoints, losses


def main():
    # ---- merged six-dataset task -------------------------------------------
    train_rows, probe_rows, exam_rows, exam_ent, exam_scales = [], [], [], [], []
    for spec in build_task_specs(series_cap=None):
        task = build_task_data(spec, context_length=512, prediction_length=64,
                               series_cap=None, pool_fragments=[])
        scale_of = dict(zip(task.test_entities, task.test_scales))
        n_slots = 0
        for frag in task.reference_fragments:
            data = np.asarray(frag.data, dtype=np.float32)
            for j in range(len(data) // WL):
                w = data[j * WL:(j + 1) * WL]
                if j % 2 == 0:
                    train_rows.append(w)
                else:
                    probe_rows.append(w)
                n_slots += 1
        exam_rows.append(task.test_windows)
        exam_ent.extend(task.test_entities)
        exam_scales.extend(scale_of[e] for e in task.test_entities)
        log("%s: %d series, %d slots, %d exam windows" % (
            spec.dataset, len(task.reference_fragments), n_slots,
            len(task.test_windows)))
    train_windows = np.stack(train_rows).astype(np.float32, copy=False)
    probe_windows = np.stack(probe_rows).astype(np.float32, copy=False)
    exam_windows = np.concatenate(exam_rows).astype(np.float32, copy=False)
    exam_scales = np.asarray(exam_scales, dtype=np.float64)
    log("merged: %d train / %d probe slots / %d exam windows" % (
        len(train_windows), len(probe_windows), len(exam_windows)))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    model = engine.model
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ev_kw = dict(test_entities=exam_ent, context_length=512,
                 test_scales=exam_scales)

    model.load_state_dict(base, strict=True)
    m0 = evaluate(engine, exam_windows, **ev_kw)
    base_overall, base_per_ds = exam_report(m0)
    log("ORIGINAL merged exam: MASE %.4f | %s" % (
        base_overall, {d: round(v, 4) for d, v in sorted(base_per_ds.items())}))

    def eval_mark():
        m = evaluate(engine, exam_windows, **ev_kw)
        overall, per_ds = exam_report(m)
        return {"mase": overall,
                "per_dataset": {d: round(v, 4) for d, v in per_ds.items()},
                "probe_ce": probe_ce(engine, probe_windows)}

    grid = {}
    best = (None, None, None, 1e9)
    for mechanism, lrs in MECH_LRS.items():
        params = trainable_params(model, mechanism)
        log("mechanism %s: %d/%d params" % (
            mechanism, sum(p.numel() for p in params), engine.parameter_count))
        for lr in lrs:
            model.load_state_dict(base, strict=True)
            cps, losses = train_checkpoints(
                engine, train_windows, params=params, lr=lr)
            run = {}
            for mark in MARKS:
                model.load_state_dict(cps[mark], strict=True)
                model.eval()
                run[mark] = eval_mark()
                if run[mark]["mase"] < best[3]:
                    best = (mechanism, lr, mark, run[mark]["mase"])
            grid["%s_lr%g" % (mechanism, lr)] = {
                "train_loss_first": losses[0],
                "train_loss_last": losses[-1],
                "marks": {str(k): v for k, v in run.items()},
            }
            log("%s lr=%g: mase %s (probe_ce %.3f->%.3f)" % (
                mechanism, lr, {k: round(v["mase"], 4) for k, v in run.items()},
                run[MARKS[0]]["probe_ce"], run[MARKS[-1]]["probe_ce"]))

    log("BEST in-domain regime: %s lr=%g steps=%d MASE %.4f "
        "(baseline %.4f, %.1f%% gain)" % (
            best[0], best[1], best[2], best[3], base_overall,
            100 * (1 - best[3] / base_overall)))

    # ---- best regime on the random pool (actual selection setting) ----------
    pool, _manifest = build_ratio_pool(
        context_length=512, prediction_length=64, equivalents=20480,
        seed=0, cache_dir=OUT_DIR / "cache")
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    pool_windows = np.asarray(
        random_select(index, n_select=4096, seed=101).selected, dtype=np.float32)
    params = trainable_params(model, best[0])
    model.load_state_dict(base, strict=True)
    cps, losses = train_checkpoints(
        engine, pool_windows, params=params, lr=best[1])
    pool_run = {"train_loss_first": losses[0], "train_loss_last": losses[-1],
                "marks": {}}
    for mark in (best[2], MARKS[-1]):
        model.load_state_dict(cps[mark], strict=True)
        model.eval()
        pool_run["marks"][str(mark)] = eval_mark()
    log("POOL@best: %s | @800: %s" % (
        {k: round(v["mase"], 4) for k, v in pool_run["marks"].items()},
        round(pool_run["marks"][str(MARKS[-1])]["mase"], 4)))

    model.load_state_dict(base, strict=True)
    result = {
        "merged_task": {"n_train": len(train_windows),
                        "n_probe": len(probe_windows),
                        "n_exam": len(exam_windows)},
        "original": {"mase": base_overall,
                     "per_dataset": {k: round(v, 4)
                                     for k, v in base_per_ds.items()}},
        "grid": grid,
        "best_in_domain": {"mechanism": best[0], "lr": best[1],
                           "steps": best[2], "mase": best[3]},
        "pool_at_best": pool_run,
    }
    with open(OUT_DIR / "diag5_merged.json", "w") as fh:
        json.dump(result, fh, indent=1)
    log("done -> %s" % (OUT_DIR / "diag5_merged.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
