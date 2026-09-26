"""Diagnostic 3: does train->eval transmit signal at all (in-distribution)?

No exam involved, no "should be best" assumption.  Split each boomlet_1676
reference fragment (the safe history) into disjoint 576-point slots; even
slots train, odd slots are held-out probes from the SAME distribution.
Train one seed-0 pass with checkpoints and, at each mark, measure on the
probes both MASE (deployment-style metric) and mean CE (the quantity SGD
actually minimises — direction guaranteed in-distribution before
overfitting).

Readings:
  * probe CE falls early AND probe MASE improves  -> pipeline transmits;
  * probe CE falls but probe MASE never improves   -> readout/metric mismatch;
  * probe CE flat or rising everywhere             -> training not transmitting.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rags.engines import ChronosT5Config, ChronosT5Engine
from rags.experiments.tasks import build_task_data, build_task_specs
from rags.experiments.train_eval import evaluate

OUT_DIR = Path("/root/TS3/diag/0908")
MARKS = (250, 500, 1000, 2000, 3000)
WL = 576


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def probe_ce(engine, windows, batch_size=128):
    from rags.engines import per_sample_cross_entropy
    model = engine.model
    model.eval()
    totals = []
    with torch.no_grad():
        for begin in range(0, len(windows), batch_size):
            tensors = engine.batch_tensors(windows[begin:begin + batch_size])
            output = model(**tensors, return_dict=True)
            totals.append(per_sample_cross_entropy(
                output.logits, tensors["labels"]).cpu().numpy())
    return float(np.concatenate(totals).mean())


def main():
    spec = {s.dataset: s for s in build_task_specs(series_cap=None)}["boomlet_1676"]
    task = build_task_data(spec, context_length=512, prediction_length=64,
                           series_cap=None, pool_fragments=[])
    scale_of = dict(zip(task.test_entities, task.test_scales))

    train_rows, probe_rows, probe_ent, probe_scales = [], [], [], []
    for frag in task.reference_fragments:
        data = np.asarray(frag.data, dtype=np.float32)
        for j in range(len(data) // WL):
            window = data[j * WL:(j + 1) * WL]
            if j % 2 == 0:
                train_rows.append(window)
            else:
                probe_rows.append(window)
                probe_ent.append(frag.entity_id)
                probe_scales.append(scale_of.get(frag.entity_id, 0.0))
    keep = [i for i, s in enumerate(probe_scales) if s and s > 0]
    probe_rows = np.stack([probe_rows[i] for i in keep])
    probe_ent = [probe_ent[i] for i in keep]
    probe_scales = np.asarray([probe_scales[i] for i in keep], dtype=np.float64)
    train_windows = np.stack(train_rows).astype(np.float32, copy=False)
    log("split: %d train / %d probe windows (zero-scale probes dropped: %d)" % (
        len(train_windows), len(probe_rows), len(keep) != len(probe_scales)))

    engine = ChronosT5Engine(
        ChronosT5Config(context_length=512, prediction_length=64,
                        projection_dimension=4096), device="cuda")
    from rags.engines import per_sample_cross_entropy

    model = engine.model
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}

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
        indices = rng.integers(0, len(train_windows), size=min(32, len(train_windows)))
        tensors = engine.batch_tensors(train_windows[indices])
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

    ev_kw = dict(test_entities=probe_ent, context_length=512,
                 test_scales=probe_scales)
    ladder = {}
    for mark in (0,) + MARKS:
        model.load_state_dict(base if mark == 0 else checkpoints[mark],
                              strict=True)
        model.eval()
        m = evaluate(engine, probe_rows, **ev_kw)
        ce = probe_ce(engine, probe_rows)
        ladder[mark] = {"mase": m["mase"], "sql": m["sql"], "probe_ce": ce}
        log("probe @ %4d steps: MASE %.4f SQL %.4f CE %.4f" % (
            mark, m["mase"], m["sql"], ce))

    model.load_state_dict(base, strict=True)
    result = {
        "dataset": "boomlet_1676", "split": "even slots train / odd slots probe",
        "n_train": len(train_windows), "n_probe": len(probe_rows),
        "loss_first": losses[0], "loss_last": losses[-1],
        "ladder": ladder,
    }
    with open(OUT_DIR / "diag3_transfer.json", "w") as fh:
        json.dump(result, fh, indent=1)
    log("done -> %s" % (OUT_DIR / "diag3_transfer.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
