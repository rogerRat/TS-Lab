"""Diagnostic 6: merged task with STACKED budget (user 2026-09-08):
N_total = 6 x 4096 = 24,576 windows, pool = 5 x N_total = 122,880 window
equivalents (A5 ratio kept).

Runs the random-pool selection under the best regime found by diag5's
in-domain grid (mechanism + lr read from diag5_merged.json) with a step
ladder — the larger N shifts the harm threshold right, so steps are
re-laddered: {100, 200, 400, 800, 1600}.

Evaluation: merged six-dataset native exam (equal-weight per-dataset MASE),
plus odd-slot probe CE as the overfit signal.
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
N_TOTAL = 6 * 4096          # 24,576 stacked
POOL_EQUIV = 5 * N_TOTAL    # 122,880
MARKS = (100, 200, 400, 800, 1600)


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
    elif mechanism == "head":
        keep = {id(p) for n, p in named
                if n in ("shared.weight", "lm_head.weight")}
    else:
        raise ValueError(mechanism)
    return [p for p in model.parameters() if id(p) in keep]


def train_checkpoints(engine, windows, *, params, lr, marks, seed=0):
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
    for step in range(1, marks[-1] + 1):
        indices = rng.integers(0, len(windows), size=min(32, len(windows)))
        tensors = engine.batch_tensors(windows[indices])
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors, return_dict=True)
        loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step in marks:
            checkpoints[step] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
    model.eval()
    return checkpoints, losses


def main():
    # ---- merged exam + probe (same construction as diag5) --------------------
    probe_rows, exam_rows, exam_ent, exam_scales = [], [], [], []
    for spec in build_task_specs(series_cap=None):
        task = build_task_data(spec, context_length=512, prediction_length=64,
                               series_cap=None, pool_fragments=[])
        scale_of = dict(zip(task.test_entities, task.test_scales))
        for frag in task.reference_fragments:
            data = np.asarray(frag.data, dtype=np.float32)
            for j in range(1, len(data) // WL, 2):
                probe_rows.append(data[j * WL:(j + 1) * WL])
        exam_rows.append(task.test_windows)
        exam_ent.extend(task.test_entities)
        exam_scales.extend(scale_of[e] for e in task.test_entities)
    probe_windows = np.stack(probe_rows).astype(np.float32, copy=False)
    exam_windows = np.concatenate(exam_rows).astype(np.float32, copy=False)
    exam_scales = np.asarray(exam_scales, dtype=np.float64)
    ev_kw = dict(test_entities=exam_ent, context_length=512,
                 test_scales=exam_scales)

    # ---- best regime from diag5 ---------------------------------------------
    with open(OUT_DIR / "diag5_merged.json") as fh:
        d5 = json.load(fh)
    mechanism = d5["best_in_domain"]["mechanism"]
    lr = float(d5["best_in_domain"]["lr"])
    log("regime from diag5: %s lr=%g (in-domain best %.4f vs base %.4f)" % (
        mechanism, lr, d5["best_in_domain"]["mase"], d5["original"]["mase"]))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    model = engine.model
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(base, strict=True)
    m0 = evaluate(engine, exam_windows, **ev_kw)
    base_overall, base_per_ds = exam_report(m0)
    log("ORIGINAL merged exam: MASE %.4f" % base_overall)

    # ---- stacked pool, random selection of N_total ---------------------------
    pool, manifest = build_ratio_pool(
        context_length=512, prediction_length=64, equivalents=POOL_EQUIV,
        seed=0, cache_dir=OUT_DIR / "cache")
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    t0 = time.time()
    sel = random_select(index, n_select=N_TOTAL, seed=101)
    pool_windows = np.asarray(sel.selected, dtype=np.float32)
    log("stacked pool: %d trajectories / %d equivalents; selected %d windows "
        "(%.0fs)" % (len(pool), manifest["total_equivalents"], len(pool_windows),
                     time.time() - t0))

    params = trainable_params(model, mechanism)
    cps, losses = train_checkpoints(engine, pool_windows, params=params, lr=lr,
                                    marks=MARKS)
    run = {}
    for mark in MARKS:
        model.load_state_dict(cps[mark], strict=True)
        model.eval()
        m = evaluate(engine, exam_windows, **ev_kw)
        overall, per_ds = exam_report(m)
        run[mark] = {"mase": overall,
                     "per_dataset": {d: round(v, 4) for d, v in per_ds.items()},
                     "probe_ce": probe_ce(engine, probe_windows)}
        log("POOL @ %4d steps: MASE %.4f (probe_ce %.3f) | %s" % (
            mark, overall, run[mark]["probe_ce"],
            {d: round(v, 4) for d, v in sorted(per_ds.items())}))

    model.load_state_dict(base, strict=True)
    result = {
        "design": {"n_total": N_TOTAL, "pool_equivalents": POOL_EQUIV,
                   "mechanism": mechanism, "lr": lr,
                   "pool_trajectories": len(pool)},
        "original": {"mase": base_overall,
                     "per_dataset": {k: round(v, 4)
                                     for k, v in base_per_ds.items()}},
        "loss_first": losses[0], "loss_last": losses[-1],
        "ladder": {str(k): v for k, v in run.items()},
    }
    with open(OUT_DIR / "diag6_stacked.json", "w") as fh:
        json.dump(result, fh, indent=1)
    log("done -> %s" % (OUT_DIR / "diag6_stacked.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
