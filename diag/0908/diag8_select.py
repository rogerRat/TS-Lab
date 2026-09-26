"""Diagnostic 8: selection headroom — can informed selection beat random?

Under the champion regime from diag7, compare on the merged native exam:
  sim-max    top-24,576 pool windows by max cosine to any reference
             (even-slot) window, Chronos encoder mean-pool embeddings;
  sim-cent   top-24,576 by cosine to the reference centroid;
  random     disjoint-grid random 24,576 (seed 101), paired control.

References: diag6 random under lr 3e-5 = 0.3398 (neutral); diag5 in-domain
best = 0.3311 (-2.9%).  If sim selection moves clearly toward in-domain,
the selection premise of the main experiment is validated.
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
RUNS7 = OUT_DIR / "runs7"
WL = 576
N_TOTAL = 6 * 4096
POOL_EQUIV = 5 * N_TOTAL
MARKS = (400, 800, 1600, 3200)
EMBED_CHUNK = 512


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


def embed_windows(engine, windows):
    """Chronos encoder mean-pool context embedding, fp16 [N, d] on GPU."""
    model = engine.model
    encoder = model.get_encoder()
    model.eval()
    chunks = []
    with torch.no_grad():
        for begin in range(0, len(windows), EMBED_CHUNK):
            tensors = engine.batch_tensors(windows[begin:begin + EMBED_CHUNK])
            out = encoder(input_ids=tensors["input_ids"],
                          attention_mask=tensors["attention_mask"])
            hidden = out.last_hidden_state  # [B, T, d]
            mask = tensors["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.to(torch.float16))
    return torch.cat(chunks)


def normalized(x):
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-6)


def train_marks(engine, windows, *, params, lr, batch, wd, schedule, clip,
                seed=0):
    model = engine.model
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
    windows = np.asarray(windows, dtype=np.float32)
    checkpoints = {}
    losses = []
    model.train()
    for step in range(1, MARKS[-1] + 1):
        if schedule == "cosine":
            current = lr * min(step / 100.0, 1.0) * 0.5 * (
                1.0 + np.cos(np.pi * step / MARKS[-1]))
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
        if step in MARKS:
            checkpoints[step] = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
    model.eval()
    return checkpoints, losses


def main():
    # ---- merged task (reference even slots for embeddings; exam/probe) --------
    ref_rows, probe_rows, exam_rows = [], [], []
    exam_ent, exam_scales, probe_ent, probe_scales = [], [], [], []
    for spec in build_task_specs(series_cap=None):
        task = build_task_data(spec, context_length=512, prediction_length=64,
                               series_cap=None, pool_fragments=[])
        scale_of = dict(zip(task.test_entities, task.test_scales))
        for frag in task.reference_fragments:
            data = np.asarray(frag.data, dtype=np.float32)
            for j in range(len(data) // WL):
                w = data[j * WL:(j + 1) * WL]
                if j % 2 == 0:
                    ref_rows.append(w)
                else:
                    probe_rows.append(w)
                    probe_ent.append(frag.entity_id)
                    probe_scales.append(scale_of.get(frag.entity_id, 0.0))
        exam_rows.append(task.test_windows)
        exam_ent.extend(task.test_entities)
        exam_scales.extend(scale_of[e] for e in task.test_entities)
    keep = [i for i, s in enumerate(probe_scales) if s and s > 0]
    ref_windows = np.stack(ref_rows).astype(np.float32, copy=False)
    probe_windows = np.stack([probe_rows[i] for i in keep])
    probe_ent = [probe_ent[i] for i in keep]
    probe_scales = np.asarray([probe_scales[i] for i in keep], np.float64)
    exam_windows = np.concatenate(exam_rows).astype(np.float32, copy=False)
    exam_scales = np.asarray(exam_scales, np.float64)
    log("merged: %d reference / %d probe / %d exam" % (
        len(ref_windows), len(probe_windows), len(exam_windows)))

    # ---- champion regime from diag7 -------------------------------------------
    records = [json.loads(p.read_text()) for p in RUNS7.glob("*.json")]
    records = [r for r in records if r.get("seed", 0) == 0 and "marks" in r]
    champ = min(records, key=lambda r: r["best_mase"])
    regime = {k: champ[k] for k in ("mechanism", "lr", "batch", "wd",
                                    "schedule", "clip")}
    log("champion regime: %s (%s best %.4f @%d)" % (
        champ["key"], regime, champ["best_mase"], champ["best_step"]))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    model = engine.model
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- candidate pool windows -----------------------------------------------
    pool, manifest = build_ratio_pool(
        context_length=512, prediction_length=64, equivalents=POOL_EQUIV,
        seed=0, cache_dir=OUT_DIR / "cache")
    index = WindowIndex(pool, context_length=512, prediction_length=64, stride=1)
    stride = max(1, len(index) // 200000)
    cand_pos = np.arange(0, len(index), stride)
    log("pool: %d legal windows; %d candidates (stride %d)" % (
        len(index), len(cand_pos), stride))

    t0 = time.time()
    ref_emb = normalized(embed_windows(engine, ref_windows))       # [R, d]
    cand_emb_chunks = []
    score_max = torch.zeros(len(cand_pos), dtype=torch.float32, device=engine.device)
    centroid = normalized(ref_emb.mean(dim=0, keepdim=True))
    score_cent_list = []
    for cbeg in range(0, len(cand_pos), 20000):
        positions = cand_pos[cbeg:cbeg + 20000]
        windows = index.windows_at(positions)
        emb = normalized(embed_windows(engine, windows))           # [C, d]
        score_max[cbeg:cbeg + 20000] = (emb @ ref_emb.T).max(dim=1).values.float()
        score_cent_list.append((emb @ centroid.T).squeeze(1).float().cpu())
        del emb, windows
    score_cent = torch.cat(score_cent_list)
    log("embeddings+scores done in %.0fs" % (time.time() - t0))

    selections = {}
    top_pos = torch.topk(score_max, k=N_TOTAL).indices.cpu().numpy()
    order = np.sort(top_pos)
    selections["sim_max"] = index.windows_at(order)
    top_pos_c = torch.topk(score_cent, k=N_TOTAL).indices.cpu().numpy()
    selections["sim_cent"] = index.windows_at(np.sort(top_pos_c))
    selections["random"] = np.asarray(
        random_select(index, n_select=N_TOTAL, seed=101).selected, np.float32)
    frag_div = {}
    for name in ("sim_max", "sim_cent"):
        positions = np.sort(top_pos if name == "sim_max" else top_pos_c)
        frag_div[name] = {
            "unique_fragments": int(len(set(
                index.fragment_ids[positions].tolist()))),
            "score_p50_p95_max": [float(np.percentile(
                score_max.cpu().numpy()[positions], q)) for q in (50, 95)],
        }
    log("diversity: %s" % json.dumps(frag_div))

    # ---- train each selection under champion regime ---------------------------
    trainable = None
    named = list(model.named_parameters())
    if regime["mechanism"] == "full":
        trainable = [p for p in model.parameters() if p.requires_grad]
    elif regime["mechanism"] == "decoder":
        keep = {id(p) for n, p in named if not n.startswith("encoder.")}
        trainable = [p for p in model.parameters() if id(p) in keep]
    elif regime["mechanism"] == "encoder":
        keep = {id(p) for n, p in named if n.startswith("encoder.")}
        trainable = [p for p in model.parameters() if id(p) in keep]
    results = {}
    for name, windows in selections.items():
        model.load_state_dict(base, strict=True)
        cps, losses = train_marks(
            engine, windows, params=trainable, lr=regime["lr"],
            batch=regime["batch"], wd=regime["wd"],
            schedule=regime["schedule"], clip=regime["clip"])
        run = {"loss_first": losses[0], "loss_last": losses[-1], "marks": {}}
        for mark in MARKS:
            model.load_state_dict(cps[mark], strict=True)
            model.eval()
            m = evaluate(engine, exam_windows, exam_ent,
                         context_length=512, test_scales=exam_scales)
            overall, per_ds = exam_report(m)
            run["marks"][str(mark)] = {
                "mase": overall,
                "per_dataset": {d: round(v, 4) for d, v in per_ds.items()},
                "probe_ce": probe_ce(engine, probe_windows)}
            log("%s @%4d: MASE %.4f (ce %.3f)" % (
                name, mark, overall, run["marks"][str(mark)]["probe_ce"]))
        del cps
        model.load_state_dict(base, strict=True)
        results[name] = run

    outcome = {
        "regime": regime, "champion_key": champ["key"],
        "n_total": N_TOTAL, "candidates": int(len(cand_pos)),
        "diversity": frag_div,
        "references": {"diag6_random_lr3e-5": 0.3398,
                       "diag5_indomain_best": 0.3311,
                       "baseline": 0.3409},
        "selections": results,
    }
    (OUT_DIR / "diag8_select.json").write_text(json.dumps(outcome, indent=1))
    log("done -> %s" % (OUT_DIR / "diag8_select.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
