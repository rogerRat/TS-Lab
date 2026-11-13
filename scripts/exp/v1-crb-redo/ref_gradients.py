"""Prepare matched CRB and Uniform TimerXL gradients for safe references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from safetensors.torch import load_file
from peft.utils.save_and_load import set_peft_model_state_dict

from assets.gradient import GradientCache, FragmentGradients, WindowRequest
from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from gradient.timerxl import TimerXLGradientEngine
from rage.crb import CRBConfig, CRBEngine
from rage.experiments.development_inputs import load_fragments
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora

ROOT = PROJECT_ROOT
INPUTS = ROOT / "data/ref/shard_dataset/v1-crb-redo-12tasks"
OUT = ROOT / "outputs/train/v1-crb-redo/12tasks-ref-grad-prepare"
SRC_CACHE = ROOT / "outputs/train/v1-crb-redo/grad-prepare/gradients/timerxl-84m/worker-0/grad_cache/thuml-timer-base-84m--local"
WARM = ROOT / "outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000"


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(path)


def build_warmed_probe():
    source = json.loads((SRC_CACHE / "cache_manifest.json").read_text())
    model_fields = source["model"]
    if source["context_length"] != 480 or source["prediction_length"] != 96:
        raise ValueError("Source gradient window protocol differs")
    if source["probe"]["projection_dimension"] != 4096 or source["probe"]["projection_seed"] != 42:
        raise ValueError("Source projection identity differs")
    engine, _, identity = build_model("timerxl-84m", model_path=model_fields["ckpt"],
                                      device="cuda", context_length=480, prediction_length=96)
    lora = apply_student_lora(engine, "timerxl-84m", dict(r=8, alpha=16, dropout=0.0,
                                                          target_modules=["q_proj", "v_proj"]))
    if lora != model_fields["lora"]:
        raise ValueError("LoRA identity differs from source cache")
    adapter = load_file(WARM / "adapter_model.safetensors", device="cpu")
    incompatible = set_peft_model_state_dict(engine.model, adapter)
    if getattr(incompatible, "unexpected_keys", []):
        raise ValueError("Unexpected warmup adapter keys")
    engine.model.eval()
    probe = TimerXLGradientEngine(engine, model_key=f"{identity['model_id']}@{identity['revision']}")
    if (probe.config.projection_dimension, probe.config.projection_seed) != (4096, 42):
        raise ValueError("Probe projection differs")
    return probe, model_fields


def prepare_task(task, probe, model_fields, logger):
    source = INPUTS / task
    if not (source / ".complete").is_file():
        raise FileNotFoundError("Safe reference input incomplete: " + task)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["answer_values_included"] or manifest["context_length"] != 480 or manifest["prediction_length"] != 96:
        raise ValueError("Reference input safety or window identity differs")
    refs = load_fragments(source / "references.npz")
    if len(refs) != manifest["fragments"]:
        raise ValueError("Reference fragment count differs")
    out = output_path("train", OUT / task)
    if (out / ".complete").is_file():
        logger.info("stage=skip task=%s reason=complete", task)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "crb").mkdir(exist_ok=True)
    (out / "uniform").mkdir(exist_ok=True)
    config = CRBConfig(context_length=480, prediction_length=96,
                       cover_density=1/64, error_density=0, split_density=1/64)
    dataset = "reference:" + task
    cache = GradientCache(out, probe, model_identity=model_fields,
        data_fields=dict(dataset=dataset, reference_manifest=str(source / "manifest.json"),
                         method="crb_1_0_1_and_uniform", q=32),
        context_length=480, prediction_length=96)
    computer = FragmentGradients(cache, dataset)
    planned = sum(sum(config.budgets(f.length - 575)) for f in refs)
    logger.info("stage=task-start task=%s fragments=%d legal_windows=%d anchors_per_method=%d output=%s",
                task, len(refs), manifest["legal_windows"], planned, out)
    started = time.monotonic()
    try:
        for number, fragment in enumerate(refs, 1):
            crb_path = out / "crb" / ("%05d.json" % fragment.fragment_id)
            uniform_path = out / "uniform" / ("%05d.json" % fragment.fragment_id)
            if crb_path.is_file() and uniform_path.is_file():
                continue
            starts = fragment.valid_starts(480, 96)
            budget = sum(config.budgets(len(starts)))
            crb = CRBEngine(config, computer).process_fragment(fragment)
            anchors = sorted(int(position) for position in crb.anchors)
            if len(anchors) != budget or crb.computed_gradients != budget:
                raise ValueError("CRB anchor budget mismatch")
            uniform = np.unique(np.rint(np.linspace(0, len(starts)-1, budget)).astype(np.int64))
            if len(uniform) != budget:
                raise ValueError("Uniform anchor budget mismatch")
            ids = np.column_stack((np.full(budget, fragment.fragment_id), starts[uniform]))
            request = WindowRequest(dataset, ids, lambda positions, f=fragment, ss=starts[uniform]:
                np.stack([f.extract_window(int(ss[int(i)]), 480, 96) for i in positions]))
            cache.prepare([request])
            write_json(crb_path, dict(fragment_id=int(fragment.fragment_id), count=budget,
                cover_error_split=list(config.budgets(len(starts))), anchor_positions=anchors,
                blocks=[dict(left_index=int(b.left_index), right_index=int(b.right_index),
                             legal_windows=int(b.size)) for b in crb.blocks]))
            write_json(uniform_path, dict(fragment_id=int(fragment.fragment_id), count=budget,
                window_ids=ids.tolist()))
            if number == 1 or number % 10 == 0 or number == len(refs):
                eta=(len(refs)-number)*(time.monotonic()-started)/number
                logger.info("stage=progress task=%s fragments=%d/%d cached_windows=%d eta=%s",
                            task, number, len(refs), len(cache.rows), format_duration(eta))
        cache.freeze()
        if len(list((out / "crb").glob("*.json"))) != len(refs) or len(list((out / "uniform").glob("*.json"))) != len(refs):
            raise ValueError("Reference method manifests incomplete")
        counts=[json.loads(path.read_text())["count"] for path in sorted((out / "crb").glob("*.json"))]
        if sum(counts) != planned:
            raise ValueError("Reference total budget mismatch")
        write_json(out / "manifest.json", dict(task=task, input_manifest=str(source / "manifest.json"),
            fragments=len(refs), legal_windows=manifest["legal_windows"],
            anchors_per_method=planned, cache=str(cache.path), model=model_fields,
            method="crb_1_0_1_and_uniform"))
        (out / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=task-finished task=%s fragments=%d anchors_per_method=%d elapsed=%s output=%s",
                    task, len(refs), planned, format_duration(time.monotonic()-started), out)
    finally:
        cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    logger = configure_logger("v1_crb_redo_ref_gradients", "v1-crb-redo",
                              "12tasks-ref-grad-prepare-" + args.task, "gradients")
    logger.info("stage=config task=%s q=32 crb=1:0:1 uniform=matched context=480 prediction=96 output=%s",
                args.task, OUT / args.task)
    probe, model_fields = build_warmed_probe()
    prepare_task(args.task, probe, model_fields, logger)


if __name__ == "__main__":
    main()
