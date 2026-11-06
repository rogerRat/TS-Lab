"""Reusable real-model mixed-precision and checkpoint-resume acceptance tool."""
import argparse
from pathlib import Path

import numpy as np
import torch

from rage.experiments.checkpoint_retention import cleanup_completed_weights
from rage.experiments.full_pool_models import build_model
from rage.experiments.students import atomic_json
from rage.experiments.train_eval import evaluate, fine_tune

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
torch.set_num_threads(4)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
engine, probe, identity = build_model(args.student, model_path=args.model_path)
windows = np.random.default_rng(91).normal(size=(2, 576)).astype(np.float32)
assert {p.dtype for p in engine.model.parameters()} == {torch.float32}
losses, sketches = probe.batch_gradient_sketch(windows[:1])
assert sketches.dtype == np.float32 and np.isfinite(sketches).all()
engine.model.to(dtype=torch.bfloat16)
assert {p.dtype for p in engine.model.parameters()} == {torch.bfloat16}
if hasattr(engine, "training_loss"):
    loss = engine.training_loss(windows[:1])
else:
    from rage.engines import per_sample_cross_entropy
    tensors = engine.batch_tensors(windows[:1])
    inputs = getattr(engine, "_logits_inputs", lambda value: value)(tensors)
    output = engine.model(**inputs, return_dict=True)
    loss = per_sample_cross_entropy(output.logits, tensors["labels"]).mean()
    del output, tensors, inputs
assert loss.dtype == torch.float32 and torch.isfinite(loss)
del loss
exam = dict(test_windows=windows[:1], test_entities=["smoke"], test_scales=np.ones(1), context_length=512)
before = evaluate(engine, **exam)
kwargs = dict(steps=2, batch_size=1, learning_rate=1e-5, seed=1,
              resume_path=out / "resume.pt", checkpoint_steps=[1, 2])
trained = fine_tune(engine, windows, **kwargs)
state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
assert {p.dtype for p in state["model"].values() if p.is_floating_point()} == {torch.bfloat16}
assert {p.dtype for p in state["optimizer"]["master_parameters"]} == {torch.float32}
assert all(v.dtype == torch.float32 for row in state["optimizer"]["optimizer"]["state"].values()
           for v in row.values() if isinstance(v, torch.Tensor))
del state
resumed = fine_tune(engine, windows, **kwargs)
assert resumed["resumed_from_step"] == 2
after = evaluate(engine, **exam)
assert np.isfinite(before["mase"]) and np.isfinite(after["mase"])
atomic_json(out / "result.json", dict(identity=identity, train=trained,
    resumed_from_step=resumed["resumed_from_step"], loss_dtype="torch.float32",
    probe_gradient_dtype="torch.float32", training_model_dtype="torch.bfloat16",
    optimizer_dtype="torch.float32", probe_sketch_norm=float(np.linalg.norm(sketches)),
    before_mase=before["mase"], after_mase=after["mase"], purpose="synthetic_real_model_smoke"))
cleanup_completed_weights(out / "result.json", out)
print(args.student, "MIXED_PRECISION_SMOKE_PASS", flush=True)
