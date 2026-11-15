"""Train and rolling-evaluate one v1-rage-ads task/arm cell."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from peft import PeftModel
from transformers import GenerationConfig

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from rage.ads import SourceProbabilitySampler
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora
from rage.experiments.regions import RegionWindows
from rage.experiments.train_eval import TrainingRecipe, fine_tune

ROOT = PROJECT_ROOT
TASKS = ("fev-09", "time-01", "time-02", "time-03", "time-04", "time-06", "time-07", "time-08")
ARMS = ("fullRand", "rage", "ancUni", "Traj")
SELECTION_ROOT = ROOT / "outputs/selection/v1-rage-ads/rolling"
TRAIN_ROOT = ROOT / "outputs/train/v1-rage-ads/rolling"
EVAL_ROOT = ROOT / "outputs/eval/v1-rage-ads/rolling"
SOURCE_DATA = ROOT / "data/src/shard_dataset/src-96"
MODEL_PATH = Path("/root/autodl-tmp/pretrained_models/timer-base-84m-84m")
CONTEXT, PREDICTION = 480, 96
STEPS, BATCH_SIZE, SEED = 4000, 32, 0


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="ascii")
    os.replace(temporary, path)


class SourceWindowView(RegionWindows):
    """Lazy source-pool window view, compatible with fine_tune sampling."""

    def __init__(self, values, value_offsets, legal, global_positions=None, sampler=None):
        self.values = values
        self.value_offsets = np.asarray(value_offsets, dtype=np.int64)
        self.legal = np.asarray(legal, dtype=np.int64)
        self.legal_offsets = np.cumsum(np.r_[0, self.legal])
        self.global_positions = (None if global_positions is None else
                                 np.asarray(global_positions, dtype=np.int64))
        self.sampler = sampler
        self.window_length = CONTEXT + PREDICTION
        sampler_mode = "uniform_all" if sampler is None else sampler["mode"]
        self.training_contract = dict(
            sampler_population=int(self.legal_offsets[-1]),
            sampler_positions_count=(None if self.global_positions is None
                                     else int(len(self.global_positions))),
            sampler_algorithm=sampler_mode,
            window_protocol="480+96",
        )

    def sample_indices(self, rng, count):
        if self.global_positions is not None:
            return rng.integers(0, len(self.global_positions), size=count)
        if self.sampler is None:
            return rng.integers(0, int(self.legal_offsets[-1]), size=count)
        if self.sampler["mode"] == "ads_probability_descriptor":
            compact = self.sampler["sampler"]
            fragment_ids, starts = compact.sample(rng, count)
            owners = np.searchsorted(compact.fragment_ids, fragment_ids)
            if (np.any(owners >= len(compact.fragment_ids))
                    or not np.array_equal(compact.fragment_ids[owners], fragment_ids)):
                raise ValueError("ADS sampler returned an unknown source trajectory")
            local = ((starts - compact.first_starts[owners])
                     // compact.strides[owners]).astype(np.int64)
            if np.any(local < 0) or np.any(local >= self.legal[fragment_ids]):
                raise ValueError("ADS sampler returned an invalid source start")
            return self.legal_offsets[fragment_ids] + local
        if self.sampler["mode"] == "trajectory_uniform":
            ids = self.sampler["fragment_ids"]
            probabilities = self.sampler["trajectory_probabilities"]
            chosen = rng.choice(len(ids), size=count, replace=True, p=probabilities)
            fragment_ids = ids[chosen]
            local = rng.integers(0, self.legal[fragment_ids])
            return self.legal_offsets[fragment_ids] + local
        raise ValueError("unsupported compact source sampler")

    def __len__(self):
        return int(self.legal_offsets[-1] if self.global_positions is None
                   else len(self.global_positions))

    def __getitem__(self, indices):
        rows = np.asarray(indices, dtype=np.int64).reshape(-1)
        global_rows = rows if self.global_positions is None else self.global_positions[rows]
        if np.any(global_rows < 0) or np.any(global_rows >= self.legal_offsets[-1]):
            raise IndexError("source global window position out of range")
        owners = np.searchsorted(self.legal_offsets[1:], global_rows, side="right")
        local = global_rows - self.legal_offsets[owners]
        begins = self.value_offsets[owners] + local
        return self.values[begins[:, None] + np.arange(self.window_length)[None, :]]


def load_source():
    config = yaml.safe_load((ROOT / "configs/data/src-96.yaml").read_text(encoding="ascii"))
    lengths = np.asarray([int(row["length"]) for row in config["trajectories"]], dtype=np.int64)
    legal = lengths - CONTEXT - PREDICTION + 1
    values = np.load(SOURCE_DATA / "values.npy", mmap_mode="r")
    value_offsets = np.asarray(np.load(SOURCE_DATA / "offsets.npy", mmap_mode="r"),
                               dtype=np.int64)
    if (len(lengths) != 96 or np.any(legal <= 0) or int(legal.sum()) != 2666312
            or len(value_offsets) != 97 or not np.array_equal(np.diff(value_offsets), lengths)
            or int(value_offsets[-1]) != len(values)):
        raise ValueError("src-96 source data no longer matches its registered layout")
    return values, legal, value_offsets, np.cumsum(np.r_[0, legal])


def sampling_probabilities(task_dir, arm, legal, legal_offsets):
    manifest = json.loads((task_dir / "sampling.json").read_text(encoding="ascii"))
    mode = manifest.get("mode")
    total = int(legal_offsets[-1])
    if int(manifest.get("candidate_count", -1)) != total:
        raise ValueError("sampler population differs from source pool")
    if mode == "uniform_all":
        return None, None, manifest, None
    if mode == "probability_descriptor":
        descriptor_path = task_dir / manifest["descriptor_path"]
        descriptor = json.loads(descriptor_path.read_text(encoding="ascii"))
        sampler = SourceProbabilitySampler(descriptor)
        if (int(sampler.offsets[-1]) != total
                or not np.array_equal(sampler.fragment_ids, np.arange(len(legal)))
                or not np.array_equal(sampler.counts, legal)):
            raise ValueError("ADS descriptor candidate counts differ from source data")
        if sampler.mode == "uniform":
            return None, None, manifest, None
        return None, None, manifest, dict(mode="ads_probability_descriptor", sampler=sampler)
    if mode == "trajectory_uniform":
        fragment_ids = np.asarray(manifest["fragment_ids"], dtype=np.int64)
        probabilities = np.asarray(manifest["trajectory_probabilities"], dtype=np.float64)
        if (not len(fragment_ids) or probabilities.shape != fragment_ids.shape
                or np.any(fragment_ids < 0) or np.any(fragment_ids >= len(legal))
                or np.any(probabilities < 0) or not np.isfinite(probabilities).all()
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError("trajectory sampler probabilities are invalid")
        return None, None, manifest, dict(mode="trajectory_uniform",
                                          fragment_ids=fragment_ids,
                                          trajectory_probabilities=probabilities)
    raise ValueError("unsupported training sampler mode: " + str(mode))


def export_model(adapter_dir, destination, task, arm):
    base, _probe, _identity = build_model(
        "timerxl-84m", model_path=str(MODEL_PATH), device="cpu",
        context_length=CONTEXT, prediction_length=PREDICTION)
    wrapped = PeftModel.from_pretrained(base.model, adapter_dir, is_trainable=False)
    merged = wrapped.merge_and_unload()
    destination.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size="5GB")
    GenerationConfig.from_pretrained(MODEL_PATH, local_files_only=True).save_pretrained(destination)
    record = dict(schema_version=1, experiment_version="v1-rage-ads/" + task + "/" + arm,
                  model_key="timerxl-84m", model="thuml/timer-base-84m",
                  training_step=STEPS, precision="bfloat16",
                  evaluation_mode="rolling")
    if task == "fev-09":
        record["reference_task_record"] = dict(benchmark="fev", task_indices=[9])
    write_json(destination / "rage_checkpoint_manifest.json", record)
    del base, wrapped, merged


def evaluate_model(checkpoint, task, arm, logger):
    folder = "fev" if task == "fev-09" else "time"
    config = ROOT / "configs/eval" / folder / ("v1-rage-ads-" + task + ".yaml")
    eval_config = yaml.safe_load(config.read_text(encoding="ascii"))
    mode = eval_config["adapter"]["config"]["inference"]["evaluation_mode"]
    if mode != "rolling":
        raise ValueError("v1-rage-ads evaluation config must use rolling mode")
    output = output_path("eval", EVAL_ROOT / task / arm)
    if (output / ".complete").is_file():
        return output
    command = [sys.executable, "-u", "-m", "evaluate.run",
               "--checkpoint", str(checkpoint), "--config", str(config),
               "--output-dir", str(output)]
    if output.exists():
        command.append("--resume")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:third_parties/fev/src:third_parties/TIME/src:."
    logger.info("stage=evaluation-start task=%s arm=%s mode=rolling config=%s",
                task, arm, config)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    primary = output / ("fev_results.csv" if folder == "fev" else "time_metadata.json")
    if not primary.is_file():
        raise RuntimeError("evaluation primary output is missing")
    if folder == "fev":
        with primary.open(newline="") as stream:
            if len(list(csv.DictReader(stream))) != 1:
                raise RuntimeError("FEV task result count differs")
        if not (output / "fev_metadata.json").is_file():
            raise RuntimeError("FEV evaluation metadata is missing")
    elif not json.loads(primary.read_text(encoding="ascii")).get("completed_at"):
        raise RuntimeError("TIME evaluation is incomplete")
    (output / ".complete").write_text("complete\n", encoding="ascii")
    return output


def run_cell(task, arm, values, legal, value_offsets, legal_offsets, logger):
    selection_dir = SELECTION_ROOT / task / arm
    if not (selection_dir / ".complete").is_file():
        raise FileNotFoundError("selection arm is incomplete: %s/%s" % (task, arm))
    selection = json.loads((selection_dir / "selection.json").read_text(encoding="ascii"))
    if selection.get("task") != task or selection.get("arm") != arm:
        raise ValueError("selection manifest task/arm mismatch")
    probabilities, global_positions, sampling_manifest, sampler = sampling_probabilities(
        selection_dir, arm, legal, legal_offsets)
    if global_positions is not None and probabilities is not None:
        raise ValueError("this run does not combine subset positions and weighted sampling")
    windows = SourceWindowView(values, value_offsets, legal, global_positions, sampler)
    if probabilities is not None and len(probabilities) != len(windows):
        raise ValueError("training probability vector does not match window view")
    phase = output_path("train", TRAIN_ROOT / task / arm)
    phase.mkdir(parents=True, exist_ok=True)
    if (phase / ".complete").is_file():
        logger.info("stage=cell-skip task=%s arm=%s reason=complete", task, arm)
        return
    logger.info("stage=cell-config task=%s arm=%s windows=%d sampler=%s steps=%d batch=%d seed=%d",
                task, arm, len(windows), sampling_manifest["mode"], STEPS, BATCH_SIZE, SEED)
    resume = phase / "resume.pt"
    adapter = phase / "adapter-step-4000"
    export = phase / "export-step-4000"
    training_manifest = phase / "training_manifest.json"
    if not training_manifest.is_file():
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        engine, _probe, identity = build_model(
            "timerxl-84m", model_path=str(MODEL_PATH), device="cuda",
            context_length=CONTEXT, prediction_length=PREDICTION)
        lora = apply_student_lora(engine, "timerxl-84m", dict(
            r=8, alpha=16, dropout=0.0, target_modules=["q_proj", "v_proj"]))
        recipe = TrainingRecipe(
            learning_rate=1e-4, epochs=1.0, batch_size=BATCH_SIZE,
            micro_batch_size=8, weight_decay=0.0, betas=(0.9, 0.999),
            eps=1e-8, max_grad_norm=1.0, warmup_ratio=0.03,
            schedule="cosine", min_lr_ratio=0.0, schedule_epochs=STEPS)
        windows.training_contract.update(
            selection_manifest=str(selection_dir / "selection.json"),
            sampler_mode=sampling_manifest["mode"], arm=arm, task=task)
        def on_step(step):
            if step == STEPS or step % 100 == 0:
                logger.info("stage=train-progress task=%s arm=%s step=%d/%d",
                            task, arm, step, STEPS)
        result = fine_tune(
            engine, windows, steps=STEPS, batch_size=BATCH_SIZE,
            learning_rate=recipe.learning_rate, seed=SEED, recipe=recipe,
            schedule_total_steps=STEPS, sampling_probabilities=probabilities,
            resume_path=resume, checkpoint_steps=list(range(500, STEPS + 1, 500)),
            on_step=on_step)
        if result["steps"] != STEPS:
            raise RuntimeError("training stopped before the registered update budget")
        engine.model.save_pretrained(adapter, safe_serialization=True)
        write_json(training_manifest, dict(
            experiment="v1-rage-ads", task=task, arm=arm, identity=identity,
            lora=lora, recipe=recipe.as_dict(), result=result,
            sampling=sampling_manifest, selection=str(selection_dir / "selection.json"),
            selection_config=selection.get("diagnostics", {}).get("ads_config"),
        ))
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not export.exists():
        export_model(adapter, export, task, arm)
    eval_output = evaluate_model(export, task, arm, logger)
    if not (eval_output / ".complete").is_file():
        raise RuntimeError("rolling evaluation did not complete")
    if resume.exists():
        resume.unlink()
    write_json(phase / "completion_manifest.json", dict(
        experiment="v1-rage-ads", task=task, arm=arm, steps=STEPS,
        selection=str(selection_dir / "selection.json"),
        training=str(training_manifest), evaluation=str(eval_output),
        adapter=str(adapter), exported_checkpoint=str(export)))
    (phase / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=cell-complete task=%s arm=%s output=%s", task, arm, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS + ("all",), default="all")
    parser.add_argument("--arm", choices=ARMS + ("all",), default="all")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    arms = ARMS if args.arm == "all" else (args.arm,)
    logger = configure_logger("v1_rage_ads_train_eval_%s_%s" % (args.task, args.arm),
                              "v1-rage-ads", "train-eval-%s-%s" % (args.task, args.arm),
                              "train-eval")
    started = time.monotonic()
    logger.info("stage=config tasks=%s arms=%s model=timerxl-84m steps=%d batch=%d lr=1e-4 seed=%d",
                ",".join(tasks), ",".join(arms), STEPS, BATCH_SIZE, SEED)
    values, legal, value_offsets, legal_offsets = load_source()
    completed = 0
    total = len(tasks) * len(arms)
    for task in tasks:
        for arm in arms:
            run_cell(task, arm, values, legal, value_offsets, legal_offsets, logger)
            completed += 1
            logger.info("stage=progress task=%s arm=%s cells=%d/%d eta=%s",
                        task, arm, completed, total,
                        format_duration((time.monotonic() - started)
                                        * (total - completed) / completed))
    if args.task == "all" and args.arm == "all":
        for task in TASKS:
            for arm in ARMS:
                if not (TRAIN_ROOT / task / arm / ".complete").is_file():
                    raise RuntimeError("training/evaluation set incomplete at %s/%s" % (task, arm))
        (TRAIN_ROOT / ".complete").write_text("complete\n", encoding="ascii")
        (EVAL_ROOT / ".complete").write_text("complete\n", encoding="ascii")
    logger.info("stage=complete cells=%d elapsed=%s train_output=%s eval_output=%s",
                completed, format_duration(time.monotonic() - started), TRAIN_ROOT, EVAL_ROOT)


if __name__ == "__main__":
    main()

