"""Diagnostic 4: (a) does the pool/exam magnitude gap survive Chronos
per-window normalization?  (b) does ANY gentle (steps, lr) point exist?

A. Token-space comparison (what the model actually sees):
   - per-window raw context std, pathological fractions;
   - normalized |z| stats; spike-window fraction (max|z| > 10);
   - input/label token histograms: total-variation distance pool vs exam;
   - ORIGINAL model CE on pool windows vs exam windows.
   If token distributions are close and CE comparable, the magnitude gap is
   neutralised by normalization and "data mismatch" loses its mechanism.

B. Micro-grid below the previously tested 250-step floor:
   pool -> exam and in-domain -> probe, steps {10,25,50,100,250} x
   lr {1e-4, 1e-5}, fresh model per lr, checkpoint eval at every mark.
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
MARKS = (10, 25, 50, 100, 250)
LRS = (1e-4, 1e-5)
WL = 576


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def token_stats(engine, windows, tag):
    """Raw scale / normalized shape / token histograms for a window set."""
    contexts = np.asarray(windows, dtype=np.float32)[:, :512]
    futures = np.asarray(windows, dtype=np.float32)[:, 512:]
    c_std = contexts.std(axis=1)
    c_mean = contexts.mean(axis=1)
    z = (contexts - c_mean[:, None]) / np.maximum(c_std[:, None], 1e-8)
    ids_hist = np.zeros(4096, dtype=np.float64)
    lab_hist = np.zeros(4096, dtype=np.float64)
    for begin in range(0, len(windows), 256):
        tensors = engine.batch_tensors(windows[begin:begin + 256])
        ids = tensors["input_ids"].cpu().numpy()
        labs = tensors["labels"].cpu().numpy()
        ids_hist += np.bincount(ids[ids >= 0].ravel(), minlength=4096)
        labs_hist = labs
        lab_hist += np.bincount(labs[labs >= 0].ravel(), minlength=4096)
    for hist in (ids_hist, lab_hist):
        hist /= hist.sum()
    stats = {
        "raw_std_p50": float(np.percentile(c_std, 50)),
        "raw_std_p99": float(np.percentile(c_std, 99)),
        "frac_std_gt_100": float(np.mean(c_std > 100)),
        "frac_std_lt_1e_3": float(np.mean(c_std < 1e-3)),
        "abs_z_p50": float(np.percentile(np.abs(z), 50)),
        "abs_z_p99": float(np.percentile(np.abs(z), 99)),
        "frac_spike_maxz_gt_10": float(np.mean(np.abs(z).max(axis=1) > 10)),
    }
    return stats, ids_hist, lab_hist


def tv_distance(h1, h2):
    return float(0.5 * np.abs(h1 - h2).sum())


def mean_ce(engine, windows, batch_size=128):
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


def train_marks(engine, windows, *, lr, steps, marks, seed=0):
    """Fresh AdamW run from current state; snapshot state at marks."""
    model = engine.model
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    windows = np.asarray(windows, dtype=np.float32)
    checkpoints = {}
    model.train()
    for step in range(1, steps + 1):
        indices = rng.integers(0, len(windows), size=min(32, len(windows)))
        tensors = engine.batch_tensors(windows[indices])
        optimizer.zero_grad(set_to_none=True)
        output = model(**tensors, return_dict=True)
        loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
        loss.backward()
        optimizer.step()
        if step in marks:
            checkpoints[step] = {k: v.detach().clone()
                                 for k, v in model.state_dict().items()}
    model.eval()
    return checkpoints


def main():
    # ---- shared data --------------------------------------------------------
    pool, _manifest = build_ratio_pool(
        context_length=512, prediction_length=64, equivalents=20480,
        seed=0, cache_dir=OUT_DIR / "cache")
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    pool_windows = random_select(index, n_select=4096, seed=101).selected

    spec = {s.dataset: s for s in build_task_specs(series_cap=None)}["boomlet_1676"]
    task = build_task_data(spec, context_length=512, prediction_length=64,
                           series_cap=None, pool_fragments=[])
    scale_of = dict(zip(task.test_entities, task.test_scales))
    train_rows, probe_rows, probe_ent, probe_scales = [], [], [], []
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
    keep = [i for i, s in enumerate(probe_scales) if s and s > 0]
    probe_rows = np.stack([probe_rows[i] for i in keep])
    probe_ent = [probe_ent[i] for i in keep]
    probe_scales = np.asarray([probe_scales[i] for i in keep], dtype=np.float64)
    indomain_windows = np.stack(train_rows).astype(np.float32, copy=False)
    log("data: %d pool / %d exam / %d in-domain train / %d probe" % (
        len(pool_windows), len(task.test_windows), len(indomain_windows),
        len(probe_rows)))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    model = engine.model
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---- A. token-space comparison ------------------------------------------
    pool_stats, pool_ids, pool_lab = token_stats(engine, pool_windows, "pool")
    exam_stats, exam_ids, exam_lab = token_stats(engine, task.test_windows, "exam")
    ce_pool = mean_ce(engine, pool_windows)
    ce_exam = mean_ce(engine, task.test_windows)
    token_compare = {
        "pool": pool_stats, "exam": exam_stats,
        "tv_input_hist": tv_distance(pool_ids, exam_ids),
        "tv_label_hist": tv_distance(pool_lab, exam_lab),
        "orig_ce_pool": ce_pool, "orig_ce_exam": ce_exam,
    }
    log("A token-space: tv_input=%.3f tv_label=%.3f orig_ce pool=%.3f exam=%.3f" % (
        token_compare["tv_input_hist"], token_compare["tv_label_hist"],
        ce_pool, ce_exam))
    log("A pool stats: %s" % json.dumps(pool_stats))
    log("A exam stats: %s" % json.dumps(exam_stats))

    # ---- B. micro-grid: pool -> exam -----------------------------------------
    ev_kw = dict(test_entities=task.test_entities, context_length=512,
                 test_scales=task.test_scales)
    pool_grid = {}
    for lr in LRS:
        model.load_state_dict(base, strict=True)
        cps = train_marks(engine, pool_windows, lr=lr, steps=MARKS[-1], marks=MARKS)
        run = {}
        m0 = evaluate(engine, task.test_windows, **ev_kw)
        run[0] = {"mase": m0["mase"]}
        for mark in MARKS:
            model.load_state_dict(cps[mark], strict=True)
            model.eval()
            m = evaluate(engine, task.test_windows, **ev_kw)
            run[mark] = {"mase": m["mase"]}
        model.load_state_dict(base, strict=True)
        pool_grid["lr%g" % lr] = run
        log("B pool lr=%g: %s" % (lr, {k: round(v["mase"], 4)
                                       for k, v in run.items()}))

    # ---- C. micro-grid: in-domain -> probe ------------------------------------
    probe_kw = dict(test_entities=probe_ent, context_length=512,
                    test_scales=probe_scales)
    indomain_grid = {}
    for lr in LRS:
        model.load_state_dict(base, strict=True)
        cps = train_marks(engine, indomain_windows, lr=lr, steps=MARKS[-1],
                          marks=MARKS)
        run = {}
        m0 = evaluate(engine, probe_rows, **probe_kw)
        run[0] = {"mase": m0["mase"], "ce": mean_ce(engine, probe_rows)}
        for mark in MARKS:
            model.load_state_dict(cps[mark], strict=True)
            model.eval()
            m = evaluate(engine, probe_rows, **probe_kw)
            run[mark] = {"mase": m["mase"], "ce": mean_ce(engine, probe_rows)}
        model.load_state_dict(base, strict=True)
        indomain_grid["lr%g" % lr] = run
        log("C in-domain lr=%g: mase %s ce %s" % (
            lr, {k: round(v["mase"], 4) for k, v in run.items()},
            {k: round(v["ce"], 3) for k, v in run.items()}))

    result = {
        "dataset": "boomlet_1676",
        "token_space": token_compare,
        "micro_pool_to_exam": pool_grid,
        "micro_indomain_to_probe": indomain_grid,
    }
    with open(OUT_DIR / "diag4_scale.json", "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    log("done -> %s" % (OUT_DIR / "diag4_scale.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
