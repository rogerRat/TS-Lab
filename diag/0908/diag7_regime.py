"""Diagnostic 7: comprehensive regime search on the merged six-dataset task.

Phases (all on the merged task: even-slot train / odd-slot probe / merged
native exam; base MASE ~0.3409):
  A1  lr sweep {1e-5, 3e-5, 5e-5, 1e-4} @ batch 32, wd 0.01, constant,
      marks {200, 400, 800, 1600, 3200}
  A2  variants around the A1 winner: batch 128 / wd 0.1 / cosine decay /
      clip-1.0 rescue of lr 1e-4 / encoder-only mechanism
  B   seed robustness {1, 2} of the top-2 configs
  EXT champion extended to 6400 steps

Every run is cached under runs7/<key>.json and skipped on restart.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rags.engines import ChronosT5Config, ChronosT5Engine, per_sample_cross_entropy
from rags.experiments.tasks import build_task_data, build_task_specs
from rags.experiments.train_eval import evaluate

OUT_DIR = Path("/root/TS3/diag/0908")
RUNS = OUT_DIR / "runs7"
RUNS.mkdir(parents=True, exist_ok=True)
WL = 576
A1_MARKS = (200, 400, 800, 1600, 3200)
B_MARKS = (400, 800, 1600, 3200)
EXT_MARKS = (3200, 4800, 6400)


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def dataset_of(entity):
    return entity.split("::", 1)[0]


def exam_report(metrics):
    per = {}
    for entity, value in metrics["per_series"]["mase"].items():
        per.setdefault(dataset_of(entity), []).append(value)
    per_dataset = {d: float(np.mean(v)) for d, v in per.items()}
    return float(np.mean(list(per_dataset.values()))), per_dataset


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
    named = list(model.named_parameters())
    if mechanism == "full":
        return [p for p in model.parameters() if p.requires_grad]
    if mechanism == "decoder":
        keep = {id(p) for n, p in named if not n.startswith("encoder.")}
    elif mechanism == "encoder":
        keep = {id(p) for n, p in named if n.startswith("encoder.")}
    elif mechanism == "head":
        keep = {id(p) for n, p in named
                if n in ("shared.weight", "lm_head.weight")}
    else:
        raise ValueError(mechanism)
    return [p for p in model.parameters() if id(p) in keep]


def run_config(engine, base, ctx, *, key, mechanism="full", lr=3e-5,
               batch=32, wd=0.01, schedule="const", clip=0.0,
               marks=A1_MARKS, seed=0):
    path = RUNS / ("%s.json" % key)
    if path.exists():
        record = json.loads(path.read_text())
        log("skip (cached) %s best %.4f @%d" % (
            key, record["best_mase"], record["best_step"]))
        return record
    model = engine.model
    model.load_state_dict(base, strict=True)
    params = trainable_params(model, mechanism)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    windows = ctx["train_windows"]
    total = marks[-1]
    checkpoints = {}
    losses = []
    model.train()
    t0 = time.time()
    for step in range(1, total + 1):
        if schedule == "cosine":
            current = lr * min(step / 100.0, 1.0) * 0.5 * (
                1.0 + np.cos(np.pi * step / total))
            for group in optimizer.param_groups:
                group["lr"] = float(current)
        indices = rng.integers(0, len(windows), size=min(batch, len(windows)))
        tensors = engine.batch_tensors(windows[indices])
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors, return_dict=True)
        loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step in marks:
            checkpoints[step] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
    model.eval()

    ladder = {}
    for mark in marks:
        model.load_state_dict(checkpoints[mark], strict=True)
        model.eval()
        m = evaluate(engine, ctx["exam_windows"], ctx["exam_ent"],
                     context_length=512, test_scales=ctx["exam_scales"])
        overall, per_ds = exam_report(m)
        pm = evaluate(engine, ctx["probe_windows"], ctx["probe_ent"],
                      context_length=512, test_scales=ctx["probe_scales"])
        ladder[mark] = {"mase": overall,
                        "per_dataset": {d: round(v, 4) for d, v in per_ds.items()},
                        "probe_mase": pm["mase"],
                        "probe_ce": probe_ce(engine, ctx["probe_windows"])}
        log("  %s @%5d: exam %.4f probe %.4f ce %.3f (%.0fs)" % (
            key, mark, overall, pm["mase"], ladder[mark]["probe_ce"],
            time.time() - t0))
    del checkpoints
    best_step = min(ladder, key=lambda k: ladder[k]["mase"])
    record = {"key": key, "mechanism": mechanism, "lr": lr, "batch": batch,
              "wd": wd, "schedule": schedule, "clip": clip, "seed": seed,
              "loss_first": losses[0], "loss_last": losses[-1],
              "marks": {str(k): v for k, v in ladder.items()},
              "best_step": int(best_step), "best_mase": ladder[best_step]["mase"]}
    path.write_text(json.dumps(record, indent=1))
    model.load_state_dict(base, strict=True)
    return record


def main():
    # ---- merged task --------------------------------------------------------
    train_rows, probe_rows, exam_rows = [], [], []
    exam_ent, exam_scales = [], []
    probe_ent, probe_scales = [], []
    for spec in build_task_specs(series_cap=None):
        task = build_task_data(spec, context_length=512, prediction_length=64,
                               series_cap=None, pool_fragments=[])
        scale_of = dict(zip(task.test_entities, task.test_scales))
        for frag in task.reference_fragments:
            data = np.asarray(frag.data, dtype=np.float32)
            for j in range(len(data) // WL):
                w = data[j * WL:(j + 1) * WL]
                if j % 2 == 0:
                    train_rows.append(w)
                else:
                    probe_rows.append(w)
                    probe_ent.append(frag.entity_id)
                    probe_scales.append(scale_of.get(frag.entity_id, 0.0))
        exam_rows.append(task.test_windows)
        exam_ent.extend(task.test_entities)
        exam_scales.extend(scale_of[e] for e in task.test_entities)
    keep = [i for i, s in enumerate(probe_scales) if s and s > 0]
    ctx = {
        "train_windows": np.stack(train_rows).astype(np.float32, copy=False),
        "probe_windows": np.stack([probe_rows[i] for i in keep]),
        "probe_ent": [probe_ent[i] for i in keep],
        "probe_scales": np.asarray([probe_scales[i] for i in keep], np.float64),
        "exam_windows": np.concatenate(exam_rows).astype(np.float32, copy=False),
        "exam_ent": exam_ent,
        "exam_scales": np.asarray(exam_scales, np.float64),
    }
    log("merged task: %d train / %d probe / %d exam" % (
        len(ctx["train_windows"]), len(ctx["probe_windows"]),
        len(ctx["exam_windows"])))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    model = engine.model
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(base, strict=True)
    m0 = evaluate(engine, ctx["exam_windows"], ctx["exam_ent"],
                  context_length=512, test_scales=ctx["exam_scales"])
    base_overall, base_per_ds = exam_report(m0)
    pm0 = evaluate(engine, ctx["probe_windows"], ctx["probe_ent"],
                   context_length=512, test_scales=ctx["probe_scales"])
    log("BASELINE exam %.4f probe %.4f ce %.3f" % (
        base_overall, pm0["mase"],
        probe_ce(engine, ctx["probe_windows"])))
    (RUNS / "_baseline.json").write_text(json.dumps(
        {"exam_mase": base_overall,
         "per_dataset": {k: round(v, 4) for k, v in base_per_ds.items()},
         "probe_mase": pm0["mase"]}, indent=1))

    # ---- A1: lr sweep ---------------------------------------------------------
    a1 = []
    for lr in (1e-5, 3e-5, 5e-5, 1e-4):
        a1.append(run_config(engine, base, ctx,
                             key="full_lr%g_b32_wd0.01_const" % lr,
                             lr=lr, marks=A1_MARKS))
    best_lr = min(a1, key=lambda r: r["best_mase"])["lr"]
    log("A1 done; best lr = %g" % best_lr)

    # ---- A2: regime variants ---------------------------------------------------
    a2 = []
    a2.append(run_config(engine, base, ctx,
                         key="full_lr%g_b128_wd0.01_const" % best_lr,
                         lr=best_lr, batch=128, marks=A1_MARKS))
    a2.append(run_config(engine, base, ctx,
                         key="full_lr%g_b32_wd0.1_const" % best_lr,
                         lr=best_lr, wd=0.1, marks=A1_MARKS))
    a2.append(run_config(engine, base, ctx,
                         key="full_lr%g_b32_wd0.01_cosine" % best_lr,
                         lr=best_lr, schedule="cosine", marks=A1_MARKS))
    a2.append(run_config(engine, base, ctx,
                         key="full_lr0.0001_b32_wd0.01_const_clip1",
                         lr=1e-4, clip=1.0, marks=A1_MARKS))
    a2.append(run_config(engine, base, ctx,
                         key="encoder_lr0.0001_b32_wd0.01_const",
                         mechanism="encoder", lr=1e-4, marks=A1_MARKS))

    # ---- B: seed robustness of top-2 --------------------------------------------
    pool = a1 + a2
    top2 = sorted(pool, key=lambda r: r["best_mase"])[:2]
    log("top-2 for seeds: %s" % [r["key"] for r in top2])
    b_records = []
    for rec in top2:
        for seed in (1, 2):
            b_records.append(run_config(
                engine, base, ctx, key="%s_s%d" % (rec["key"], seed),
                mechanism=rec["mechanism"], lr=rec["lr"], batch=rec["batch"],
                wd=rec["wd"], schedule=rec["schedule"], clip=rec["clip"],
                marks=B_MARKS, seed=seed))

    # ---- EXT: champion to 6400 ----------------------------------------------------
    champion = min(pool, key=lambda r: r["best_mase"])
    ext = run_config(engine, base, ctx, key="%s_ext6400" % champion["key"],
                     mechanism=champion["mechanism"], lr=champion["lr"],
                     batch=champion["batch"], wd=champion["wd"],
                     schedule=champion["schedule"], clip=champion["clip"],
                     marks=EXT_MARKS)

    summary = {"baseline_exam_mase": base_overall,
               "runs": [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))
                        if not p.name.startswith("_")]}
    (OUT_DIR / "diag7_regime.json").write_text(json.dumps(summary, indent=1))
    log("diag7 complete -> %s" % (OUT_DIR / "diag7_regime.json"))
    for rec in sorted(summary["runs"], key=lambda r: r["best_mase"]):
        log("  %-42s best %.4f @%d (seed %d)" % (
            rec["key"], rec["best_mase"], rec["best_step"], rec["seed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
