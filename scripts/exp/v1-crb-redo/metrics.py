"""Compute whole-source direction scores and safe-history to test-gradient alignment."""
from __future__ import annotations
import csv
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from peft.utils.save_and_load import set_peft_model_state_dict

from adapters.foundation.eval.common import clean_context
from assets.gradient.dataset import GradientDataset
from adapters.chronos.eval.fev import configure_evaluation_windows
from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from gradient.timerxl import TimerXLGradientEngine
from rage.experiments.development_inputs import load_fragments
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora
from rr_features import REF_ROOT, REF_INPUTS, SRC_ROOT, TASKS, record_anchors, fetch, source_layout

ROOT = PROJECT_ROOT
OUT = ROOT / "outputs/selection/v1-crb-redo/sgwdev/metrics"
SRC_CACHE = ROOT / "outputs/train/v1-crb-redo/grad-prepare/gradients/timerxl-84m/worker-0/grad_cache/thuml-timer-base-84m--local"
WARM = ROOT / "outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000"
FEV_TASKS = ROOT / "configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml"
TIME_CONFIGS = {task: ROOT / ("configs/eval/time/v1-crb-redo-" + task + ".yaml")
                for task in TASKS if task.startswith("time-")}
H = {"fev-09": 28, "time-01": 12, "time-02": 12, "time-03": 24,
     "time-04": 30, "time-06": 24, "time-07": 12, "time-08": 12}
METHODS = ("crb", "uniform")
SELECTION_BATCHES = (
    ("timerXL-base-uni-01-traj-rr", "uniform"),
    ("timerXL-base-uni-01-block-rr", "uniform"),
    ("timerXL-ours-rage101-01-traj-rr", "crb"),
    ("timerXL-ours-rage101-01-block-rr", "crb"),
    ("timerXL-base-uni-01-win-rr", "uniform"),
    ("timerXL-ours-rage101-01-win-rr", "crb"),
)
N_SOURCE = 2666312
CONTEXT = 480
HEAD = 96
SCORE_CHUNK = 16384


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    os.replace(tmp, path)


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with path.open("a", encoding="ascii") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path, rows):
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def score_mean(task, method, source_info, logger):
    _config, _legal, _offsets, source_files = source_info
    source_norms = []
    source_count = 0
    for fid in range(96):
        path, cache = source_files[(method, fid)]
        anchors = record_anchors(path, method)
        gradients = fetch(cache, "src-96", fid, anchors)
        source_norms.extend(np.linalg.norm(gradients, axis=1).tolist())
        source_count += len(gradients)
    source_threshold = float(np.quantile(source_norms, 0.01))
    source_sum = np.zeros(4096, dtype=np.float64)
    for fid in range(96):
        path, cache = source_files[(method, fid)]
        anchors = record_anchors(path, method)
        gradients = fetch(cache, "src-96", fid, anchors)
        norms = np.linalg.norm(gradients, axis=1)
        units = np.divide(gradients, norms[:, None], out=np.zeros_like(gradients),
                          where=norms[:, None] > 0)
        units[norms <= source_threshold] = 0.0
        source_sum += units.sum(axis=0, dtype=np.float64)

    ref_manifest = json.loads((REF_ROOT / task / "manifest.json").read_text(encoding="ascii"))
    ref_cache = GradientDataset(Path(ref_manifest["cache"]))
    fragments = load_fragments(REF_INPUTS / task / "references.npz")
    ref_records, ref_norms, ref_count = [], [], 0
    for fragment in fragments:
        anchors = record_anchors(REF_ROOT / task / method / ("%05d.json" % fragment.fragment_id), method)
        gradients = fetch(ref_cache, "reference:" + task, fragment.fragment_id, anchors)
        ref_records.append((fragment.fragment_id, anchors))
        ref_norms.extend(np.linalg.norm(gradients, axis=1).tolist())
        ref_count += len(gradients)
    ref_threshold = float(np.quantile(ref_norms, 0.01))
    ref_sum = np.zeros(4096, dtype=np.float64)
    for fragment_id, anchors in ref_records:
        gradients = fetch(ref_cache, "reference:" + task, fragment_id, anchors)
        norms = np.linalg.norm(gradients, axis=1)
        units = np.divide(gradients, norms[:, None], out=np.zeros_like(gradients),
                          where=norms[:, None] > 0)
        units[norms <= ref_threshold] = 0.0
        ref_sum += units.sum(axis=0, dtype=np.float64)
    mean = float(np.dot(ref_sum / ref_count, source_sum / source_count))
    logger.info("stage=cached-anchor-alignment task=%s method=%s src_anchors=%d ref_anchors=%d mean=%.10f",
                task, method, source_count, ref_count, mean)
    return dict(task=task, method=method, source_anchor_gradients=source_count,
                reference_anchor_gradients=ref_count, cosine_pairs=int(source_count * ref_count),
                score_dtype="float64_aggregation_from_bf16_gradient_cache",
                estimator="mean_of_cached_true_anchor_gradients",
                source_p1_threshold=source_threshold, reference_p1_threshold=ref_threshold,
                matrix_shape=[ref_count, source_count], score_elements=int(ref_count * source_count),
                mean_direction_cosine=mean, source_cache_root=str(SRC_ROOT),
                reference_cache=str(ref_manifest["cache"]))


def post_selection_row(task, selection_batch, method, source_info, direction_row, logger):
    _config, _legal, offsets, source_files = source_info
    selection_dir = ROOT / "outputs/selection/v1-crb-redo" / selection_batch / task
    indices_path = selection_dir / "window_indices.npy"
    if not indices_path.is_file() or not (selection_dir / ".complete").is_file():
        return None
    selected = np.load(indices_path, allow_pickle=False)
    if selected.ndim != 1 or len(selected) != 266631 or len(np.unique(selected)) != len(selected):
        raise ValueError("selected source index list failed the top10% contract")
    selected_source_sum = np.zeros(4096, dtype=np.float64)
    selected_anchor_count = 0
    for fid in range(96):
        record, cache = source_files[(method, fid)]
        anchors = record_anchors(record, method)
        chosen = np.isin(int(offsets[fid]) + anchors, selected, assume_unique=False)
        if not chosen.any():
            continue
        gradients = fetch(cache, "src-96", fid, anchors[chosen])
        norms = np.linalg.norm(gradients, axis=1)
        units = np.divide(gradients, norms[:, None], out=np.zeros_like(gradients),
                          where=norms[:, None] > 0)
        units[norms <= float(direction_row["source_p1_threshold"])] = 0.0
        selected_source_sum += units.sum(axis=0, dtype=np.float64)
        selected_anchor_count += len(gradients)
    if selected_anchor_count == 0:
        raise ValueError("top10% selected set contains no cached source gradient anchors")

    manifest = json.loads((REF_ROOT / task / "manifest.json").read_text(encoding="ascii"))
    ref_cache = GradientDataset(Path(manifest["cache"]))
    fragments = load_fragments(REF_INPUTS / task / "references.npz")
    ref_sum = np.zeros(4096, dtype=np.float64)
    reference_count = 0
    threshold = float(direction_row["reference_p1_threshold"])
    for fragment in fragments:
        anchors = record_anchors(REF_ROOT / task / method / ("%05d.json" % fragment.fragment_id), method)
        gradients = fetch(ref_cache, "reference:" + task, fragment.fragment_id, anchors)
        norms = np.linalg.norm(gradients, axis=1)
        units = np.divide(gradients, norms[:, None], out=np.zeros_like(gradients),
                          where=norms[:, None] > 0)
        units[norms <= threshold] = 0.0
        ref_sum += units.sum(axis=0, dtype=np.float64)
        reference_count += len(gradients)
    mean = float(np.dot(ref_sum / reference_count, selected_source_sum / selected_anchor_count))
    logger.info("stage=post-selection-alignment task=%s batch=%s method=%s selected_windows=%d selected_cached_anchors=%d mean=%.10f",
                task, selection_batch, method, len(selected), selected_anchor_count, mean)
    return dict(task=task, selection_batch=selection_batch, method=method,
                estimator="mean_of_cached_true_anchor_gradients_filtered_by_rr_top10_selection",
                selected_windows=len(selected), selection_fraction=len(selected) / N_SOURCE,
                selected_source_anchor_gradients=selected_anchor_count,
                eligible_source_anchor_gradients=int(direction_row["source_anchor_gradients"]),
                selected_anchor_coverage=selected_anchor_count / float(direction_row["source_anchor_gradients"]),
                reference_anchor_gradients=reference_count,
                cosine_pairs=int(selected_anchor_count * reference_count),
                mean_selected_direction_cosine=mean,
                source_p1_threshold=float(direction_row["source_p1_threshold"]),
                reference_p1_threshold=threshold,
                selection_file=str(indices_path), reference_cache=str(manifest["cache"]))


class HorizonProbe(TimerXLGradientEngine):
    def __init__(self, engine, horizon, model_key):
        super().__init__(engine, model_key=model_key)
        self.horizon = int(horizon)

    def per_window_losses(self, windows):
        import torch.nn.functional as F
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != CONTEXT + self.horizon:
            raise ValueError("gradient windows must have length 480 + task horizon")
        history, target = values[:, :CONTEXT], values[:, CONTEXT:]
        mean = history.mean(axis=1, keepdims=True)
        std = history.std(axis=1, keepdims=True)
        scale = np.where(std == 0, 1.0, std)
        parameter = next(self.model.parameters())
        x = torch.as_tensor((history - mean) / scale, device=parameter.device, dtype=parameter.dtype)
        valid = getattr(self, "active_target_mask", None)
        if valid is None:
            valid = np.isfinite(target)
        else:
            valid = np.asarray(valid, dtype=bool)
        if valid.shape != target.shape or np.any(valid.sum(axis=1) == 0):
            raise ValueError("test gradient target mask is invalid")
        normalized_target = np.where(valid, (target - mean) / scale, 0.0)
        y = torch.as_tensor(normalized_target, device=parameter.device, dtype=parameter.dtype)
        mask = torch.as_tensor(valid, device=parameter.device, dtype=torch.bool)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        hidden = base.model(input_ids=x, use_cache=False, return_dict=True).last_hidden_state
        try:
            head = base.lm_heads[base.output_token_len_map[HEAD]]
        except (AttributeError, KeyError):
            candidates = [(h, int(n)) for h, n in zip(base.lm_heads, base.config.output_token_lens)
                          if int(n) == HEAD]
            if len(candidates) != 1:
                raise ValueError("TimerXL 96-point output head is not unique")
            head = candidates[0][0]
        prediction = head(hidden[:, -1:, :]).float()[..., :self.horizon]
        squared = F.mse_loss(prediction, y[:, None, :].float(), reduction="none")
        mask = mask[:, None, :].to(dtype=squared.dtype)
        return (squared * mask).sum(dim=-1).squeeze(1) / mask.sum(dim=-1).squeeze(1)

    def raw_gradient(self, window):
        loss = self.per_window_losses(np.asarray(window, dtype=np.float32)[None])[0]
        grads = torch.autograd.grad(loss, self.parameters, allow_unused=True)
        values = [torch.zeros_like(p) if g is None else g.detach()
                  for p, g in zip(self.parameters, grads)]
        return float(loss.detach().cpu()), values


def build_probe():
    source = json.loads((SRC_CACHE / "cache_manifest.json").read_text(encoding="ascii"))
    model_fields = source["model"]
    if (source.get("context_length"), source.get("prediction_length")) != (480, 96):
        raise ValueError("source gradient cache protocol mismatch")
    if source["probe"].get("projection_dimension") != 4096 or source["probe"].get("projection_seed") != 42:
        raise ValueError("gradient projection mismatch")
    engine, _, identity = build_model("timerxl-84m", model_path=model_fields["ckpt"],
                                      device="cuda", context_length=480, prediction_length=96)
    lora = apply_student_lora(engine, "timerxl-84m",
                               dict(r=8, alpha=16, dropout=0.0, target_modules=["q_proj", "v_proj"]))
    if lora != model_fields["lora"]:
        raise ValueError("LoRA identity mismatch")
    adapter = load_file(WARM / "adapter_model.safetensors", device="cpu")
    incompatible = set_peft_model_state_dict(engine.model, adapter)
    if getattr(incompatible, "unexpected_keys", []):
        raise ValueError("warmup adapter contains unexpected keys")
    engine.model.eval()
    return engine, identity, model_fields


def build_gradient_probe():
    engine, identity, model_fields = build_probe()
    probe = TimerXLGradientEngine(engine, model_key=identity["model_id"] + "@" + identity["revision"])
    return engine, probe, identity, model_fields


def compute_sketches(probe, windows, logger, name, target_masks=None):
    losses, sketches = [], []
    started = time.monotonic()
    for left in range(0, len(windows), 32):
        batch = windows[left:left + 32]
        if target_masks is None:
            l, g = probe.batch_gradient_sketch(batch, batch_size=4)
        else:
            batch_losses, batch_sketches = [], []
            for offset, row in enumerate(batch):
                probe.active_target_mask = np.asarray(target_masks[left + offset], dtype=bool)[None, :]
                try:
                    loss, sketch = probe._batched_gradient_sketch(row[None, :])
                finally:
                    del probe.active_target_mask
                batch_losses.append(loss)
                batch_sketches.append(sketch)
            l, g = np.concatenate(batch_losses), np.concatenate(batch_sketches)
        losses.append(l)
        sketches.append(g)
        done = left + len(batch)
        if done == len(windows) or done % 256 == 0:
            elapsed = time.monotonic() - started
            logger.info("stage=gradient-progress name=%s rows=%d/%d eta=%s",
                        name, done, len(windows),
                        format_duration(elapsed * (len(windows) - done) / max(done, 1)))
    return np.concatenate(losses), np.concatenate(sketches)


def atomic_npz(path, **items):
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, **items)
    os.replace(tmp, path)


def fev_test_windows(logger):
    import fev
    from datasets import Dataset
    from evaluate.fev import load_task_declarations
    declarations = load_task_declarations(FEV_TASKS, required_indices=(9,))
    spec = declarations[9]
    if int(spec["horizon"]) != H["fev-09"]:
        raise ValueError("FEV 9 horizon mismatch")
    source = Path(spec["dataset_path"])
    task = fev.Task(**spec)
    task._full_dataset = (Dataset.from_parquet(str(source)) if source.suffix == ".parquet"
                          else Dataset.from_file(str(source)))
    configure_evaluation_windows(task, "single_window")
    window = task.get_window(0, num_proc=1)
    past, _ = window.get_input_data()
    truth = window.get_ground_truth().with_format("numpy")
    by_id = {str(row[task.id_column]): row for row in truth}
    values, ids = [], []
    for row in past.with_format("numpy"):
        entity = str(row[task.id_column])
        if entity not in by_id:
            raise ValueError("FEV input and truth entity mismatch")
        for col in task.target_columns:
            context = clean_context(np.asarray(row[col], dtype=np.float32))[-CONTEXT:]
            target = np.asarray(by_id[entity][col], dtype=np.float32).reshape(-1)[:task.horizon]
            if len(context) != CONTEXT or len(target) != task.horizon or not np.isfinite(target).all():
                raise ValueError("invalid FEV single-window diagnostic sample")
            values.append(np.concatenate((context, target)))
            ids.append(entity + ":" + col)
    if not values:
        raise ValueError("FEV 9 test set is empty")
    meta = dict(benchmark="fev", task_id="fev-09", horizon=task.horizon,
                source=str(source), cutoff=str(window.cutoff), evaluation_mode="single_window",
                sample_ids=ids)
    logger.info("stage=test-windows-ready task=fev-09 samples=%d cutoff=%s", len(values), window.cutoff)
    return np.asarray(values, dtype=np.float32), meta


def time_test_windows(task_id, logger):
    from evaluate.time import _validate_bench_config, evaluation_test_length
    from timebench.evaluation.data import Dataset
    config_path = TIME_CONFIGS[task_id]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checked = _validate_bench_config(config["bench_config"])
    if len(checked["tasks"]) != 1 or checked["tasks"][0]["term"] != "short":
        raise ValueError("TIME task config must contain one short task")
    item = checked["tasks"][0]
    name = item["dataset"]
    settings = checked["datasets"][name]
    horizon = int(settings["short"]["prediction_length"])
    if horizon != H[task_id]:
        raise ValueError("TIME task horizon mismatch")
    test_length = evaluation_test_length(settings["test_length"], horizon, "single_window")
    kwargs = dict(name=name, term="short", prediction_length=horizon, test_length=test_length,
                  val_length=settings.get("val_length", 0), storage_path=checked["dataset_root"])
    native = Dataset(to_univariate=False, **kwargs)
    dataset = native if native.target_dim == 1 else Dataset(to_univariate=True, **kwargs)
    values, ids, excluded, excluded_short, target_masks = [], [], [], [], []
    masked_target_count = 0
    for ordinal, (inp, label) in enumerate(dataset.test_data):
        context = clean_context(np.asarray(inp["target"], dtype=np.float32).reshape(-1))
        target = np.asarray(label["target"], dtype=np.float32).reshape(-1)[:horizon]
        sample_id = str(inp.get("item_id", ordinal))
        if len(target) != horizon:
            raise ValueError("invalid TIME single-window diagnostic target shape")
        if len(context) < CONTEXT:
            excluded_short.append(sample_id)
            continue
        context = context[-CONTEXT:]
        valid = np.isfinite(target)
        if not valid.any():
            excluded.append(sample_id)
            continue
        masked_target_count += int((~valid).sum())
        target_masks.append(valid)
        safe_target = np.where(valid, target, 0.0).astype(np.float32)
        values.append(np.concatenate((context, safe_target)))
        ids.append(sample_id)
    if not values:
        raise ValueError("TIME test set has no sample with valid task-horizon targets")
    meta = dict(benchmark="time", task_id=task_id, dataset=name, term="short",
                horizon=horizon, source=str(checked["dataset_root"]),
                config=str(config_path), test_length=test_length,
                evaluation_mode="single_window", sample_ids=ids,
                excluded_all_nan_target_ids=excluded,
                excluded_short_context_ids=excluded_short,
                masked_target_count=masked_target_count)
    if masked_target_count:
        meta["target_masks"] = np.asarray(target_masks, dtype=bool)
    logger.info("stage=test-windows-ready task=%s samples=%d excluded_all_nan=%d excluded_short_context=%d horizon=%d",
                task_id, len(values), len(excluded), len(excluded_short), horizon)
    return np.asarray(values, dtype=np.float32), meta


def reference_windows(task, method, horizon):
    fragments = load_fragments(REF_INPUTS / task / "references.npz")
    paths = REF_ROOT / task / method
    windows, ids = [], []
    for fragment in fragments:
        anchors = record_anchors(paths / ("%05d.json" % fragment.fragment_id), method)
        starts = fragment.valid_starts(CONTEXT, horizon)
        if not len(starts) or not len(anchors) or int(anchors.max()) >= len(starts):
            raise ValueError("reference anchors invalid for task horizon")
        for position in anchors:
            start = int(starts[int(position)])
            windows.append(fragment.extract_window(start, CONTEXT, horizon))
            ids.append((fragment.fragment_id, int(position), start))
    values = np.asarray(windows, dtype=np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("safe history windows are invalid")
    return values, np.asarray(ids, dtype=np.int64)


def unit_rows(x):
    values = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    return np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] > 0), int(np.sum(norms == 0))


def cached_reference_gradients(task, method):
    manifest = json.loads((REF_ROOT / task / "manifest.json").read_text(encoding="ascii"))
    cache = GradientDataset(Path(manifest["cache"]))
    fragments = load_fragments(REF_INPUTS / task / "references.npz")
    rows, ids = [], []
    for fragment in fragments:
        anchors = record_anchors(REF_ROOT / task / method / ("%05d.json" % fragment.fragment_id), method)
        rows.append(fetch(cache, "reference:" + task, fragment.fragment_id, anchors))
        ids.extend((int(fragment.fragment_id), int(position)) for position in anchors)
    return np.concatenate(rows).astype(np.float32), np.asarray(ids, dtype=np.int64)


def alignment_row(task, method, probe, test_values, test_meta, logger, test_masks=None):
    folder = OUT / "gradients" / task
    folder.mkdir(parents=True, exist_ok=True)
    test_file = folder / "test_gradients.npz"
    if test_file.is_file():
        with np.load(test_file, allow_pickle=False) as f:
            test_g, test_loss = f["gradients"], f["losses"]
            if int(f["horizon"]) != H[task] or f["sample_ids"].astype(str).tolist() != test_meta["sample_ids"]:
                raise ValueError("test gradient cache identity mismatch")
            if test_masks is not None and ("target_masks" not in f or
                    not np.array_equal(f["target_masks"], test_masks)):
                raise ValueError("test gradient target-mask cache mismatch")
    else:
        test_loss, test_g = compute_sketches(probe, test_values, logger, "test-" + task,
                                             target_masks=test_masks)
        payload = dict(gradients=test_g.astype(np.float32), losses=test_loss.astype(np.float32),
                       horizon=np.asarray(H[task]), sample_ids=np.asarray(test_meta["sample_ids"], dtype="U"))
        if test_masks is not None:
            payload["target_masks"] = np.asarray(test_masks, dtype=bool)
        atomic_npz(test_file, **payload)
    hist_file = folder / (method + "_history_gradients_from_cache.npz")
    if hist_file.is_file():
        with np.load(hist_file, allow_pickle=False) as f:
            hist_g, hist_ids = f["gradients"], f["anchor_ids"]
    else:
        hist_g, hist_ids = cached_reference_gradients(task, method)
        atomic_npz(hist_file, gradients=hist_g, anchor_ids=hist_ids,
                   horizon=np.asarray(96), source_horizon=np.asarray(96))
    hu, hz = unit_rows(hist_g)
    tu, tz = unit_rows(test_g)
    mean_cosine = float(np.dot(hu.mean(axis=0), tu.mean(axis=0)))
    return dict(task=task, method=method, horizon=H[task], reference_cache_horizon=96,
                safe_history_gradients=len(hist_g), true_test_gradients=len(test_g),
                cosine_pairs=int(len(hist_g) * len(test_g)), zero_safe_gradients=hz,
                zero_test_gradients=tz, mean_history_test_cosine=mean_cosine,
                mean_test_task_h_mse=float(np.mean(test_loss)),
                history_gradient_file=str(hist_file), test_gradient_file=str(test_file),
                test_metadata=test_meta)


def publish_docs(direction_rows, alignment_rows, post_rows):
    required = {(task, method) for task in H for method in METHODS}
    if ({(r["task"], r["method"]) for r in direction_rows} != required or
            {(r["task"], r["method"]) for r in alignment_rows} != required):
        raise ValueError("all metric rows are required before publishing")
    docs = ROOT / "docs"
    now = datetime.now(timezone(timedelta(hours=8)))
    stamp = now.strftime("%Y-%m-%d %H:%M CST")
    tick = chr(96)
    methods = {"crb": "CRB", "uniform": "Uniform"}
    names = {"fev-09": "FEV 9 hospital_admissions_1D",
             "time-01": "TIME 1 CPHL/15T", "time-02": "TIME 2 CPHL/30T",
             "time-03": "TIME 3 CPHL/H", "time-04": "TIME 4 ECDC_COVID/D",
             "time-06": "TIME 6 epf_electricity_price/H",
             "time-07": "TIME 7 Commodity_Import/M",
             "time-08": "TIME 8 Commodity_Production/M"}
    drows = ["| %s | %s | %d | %d | %.10f |" % (
        names[r["task"]], methods[r["method"]], r["reference_anchor_gradients"],
        r["source_anchor_gradients"], r["mean_direction_cosine"])
        for r in direction_rows]
    arows = ["| %s | %s | %d | %d | %d | %.8f | %.8f | %d | %d |" % (
        names[r["task"]], methods[r["method"]], r["horizon"], r["safe_history_gradients"],
        r["true_test_gradients"], r["mean_history_test_cosine"], r["mean_test_task_h_mse"],
        len(r.get("test_metadata", {}).get("excluded_all_nan_target_ids", [])),
        len(r.get("test_metadata", {}).get("excluded_short_context_ids", [])))
        for r in alignment_rows]
    selection_names = {
        "timerXL-base-uni-01-traj-rr": "Uniform traj",
        "timerXL-base-uni-01-block-rr": "Uniform block",
        "timerXL-ours-rage101-01-traj-rr": "CRB traj",
        "timerXL-ours-rage101-01-block-rr": "CRB block",
        "timerXL-base-uni-01-win-rr": "Uniform win",
        "timerXL-ours-rage101-01-win-rr": "CRB win",
    }
    prows = ["| %s | %s | %d | %d | %.4f%% | %d | %.8f |" % (
        names[r["task"]], selection_names[r["selection_batch"]], r["selected_windows"],
        r["selected_source_anchor_gradients"], 100.0 * r["selected_anchor_coverage"],
        r["reference_anchor_gradients"], r["mean_selected_direction_cosine"])
        for r in post_rows]
    body = [
        "## \u8865\u7b97\u65b9\u5411\u5bf9\u9f50\u6307\u6807\uff08sgwdev\uff09", "",
        "\u6279\u6b21\uff1asgwdev\uff08single-window-eval\uff09\uff1b\u6307\u6807\u4e00\u5b8c\u6210 16/16 \u7ec4\uff0c\u6307\u6807\u4e8c\u5b8c\u6210 %d/16 \u7ec4\uff0c\u7b5b\u9009\u540e\u6307\u6807\u8bb0\u5f55 %d \u4e2a\u5df2\u5b8c\u6210\u9009\u6837\u6279\u6b21\u4efb\u52a1\u7ec4\u5408\u3002\u5168\u6e90\u6307\u6807\u4e3a\u5df2\u8ba1\u7b97\u951a\u70b9\u5747\u503c\u4f30\u8ba1\uff0c\u7b5b\u9009\u540e\u6307\u6807\u4e5f\u9650\u4e8e\u73b0\u6709\u6e90\u951a\u70b9\u3002" % (len(alignment_rows), len(post_rows)),
        "", "### \u5168\u6e90\u6570\u636e\u96c6\u4efb\u52a1\u5e73\u5747\u65b9\u5411\u8bc4\u5206", "",
        "\u6e90\u4fa7\u76f4\u63a5\u8bfb\u53d6 src-96 \u5df2\u6709\u7f13\u5b58\uff0c\u53c2\u8003\u4fa7\u8bfb\u53d6\u5bf9\u5e94\u4efb\u52a1\u5df2\u6709\u5b89\u5168\u5386\u53f2\u7f13\u5b58\uff1bCRB \u4e0e Uniform \u4e24\u4fa7\u65b9\u6cd5\u5339\u914d\u3002\u5bf9\u5355\u4f4d\u65b9\u5411\u6267\u884c P1 \u8fc7\u6ee4\u540e\uff0c\u62a5\u544a\u7f13\u5b58\u951a\u70b9\u95f4\u5168\u90e8\u4e24\u4e24\u4f59\u5f26\u7684\u7b97\u672f\u5e73\u5747\uff1b\u4e0d\u91cd\u7b97\u6e90\u68af\u5ea6\u3001\u4e0d\u5bf9\u672a\u8ba1\u7b97\u7a97\u53e3\u63d2\u503c\u3002",
        "", "| \u4e0b\u6e38\u4efb\u52a1 | \u65b9\u6cd5 | \u53c2\u8003\u951a\u70b9\u6570 | \u6e90\u951a\u70b9\u6570 | \u5747\u503c\u4f30\u8ba1 cos |",
        "| --- | --- | ---: | ---: | ---: |", *drows, "",
        "### \u5b89\u5168\u5386\u53f2\u5bf9\u771f\u5b9e\u6d4b\u8bd5\u6837\u672c\u7684\u68af\u5ea6\u65b9\u5411\u4ee3\u8868\u6027", "",
        "\u5b89\u5168\u5386\u53f2\u4fa7\u4f7f\u7528\u5df2\u6709\u539f\u751f 480+96 \u68af\u5ea6\u7f13\u5b58\uff1b\u6d4b\u8bd5\u4fa7\u4f7f\u7528\u771f\u5b9e single-window \u8f93\u5165\u4e0e\u6807\u7b7e\uff0c\u6309\u4efb\u52a1 H \u8ba1\u7b97\u6837\u672c\u68af\u5ea6\u3002TIME \u8bc4\u6d4b\u6807\u7b7e\u4e2d\u7684 NaN \u4f4d\u7f6e\u6309\u8bc4\u6d4b\u5668\u6709\u6548\u6807\u7b7e\u63a9\u7801\u8ba1\u7b97\uff1bH \u5185\u5168\u7f3a\u5931\u7684\u6837\u672c\u4e0d\u4ea7\u751f\u68af\u5ea6\u5e76\u6392\u9664\u3002",
        "", "| \u4e0b\u6e38\u4efb\u52a1 | \u65b9\u6cd5 | H | \u5b89\u5168\u5386\u53f2\u7f13\u5b58\u68af\u5ea6\u6570 | \u771f\u5b9e\u6d4b\u8bd5\u6837\u672c\u6570 | \u5e73\u5747\u65b9\u5411 cos | \u771f\u5b9e\u6837\u672c task-H MSE | \u5168\u7f3a\u5931\u6807\u7b7e\u6392\u9664\u6570 | \u4e0d\u8db3 480 \u70b9\u4e0a\u4e0b\u6587\u6392\u9664\u6570 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |", *arows, "",
        "### \u7b5b\u9009\u540e\uff08top10%\uff09\u6e90\u4fa7\u65b9\u5411\u63a5\u8fd1\u5ea6", "",
        "\u5bf9\u6bcf\u4e2a\u5df2\u5b8c\u6210\u7684 top10% \u9009\u6837\u6279\u6b21\uff0c\u4ece\u73b0\u6709 src-96 \u6e90\u68af\u5ea6\u7f13\u5b58\u4e2d\u53d6\u51fa\u5168\u5c40\u7a97\u53e3\u7d22\u5f15\u843d\u5728\u9009\u4e2d\u96c6\u5408\u7684\u951a\u70b9\uff0c\u4e0e\u540c\u65b9\u6cd5\u5b89\u5168\u5386\u53f2\u951a\u70b9\u6309\u5168\u6e90\u6307\u6807\u76f8\u540c P1 \u89c4\u5219\u6c42\u5e73\u5747\u4e24\u4e24 cos\u3002\u6e90\u4fa7\u4e0d\u91cd\u7b97\u68af\u5ea6\uff1b\u8be5\u6307\u6807\u662f\u9009\u4e2d top10% \u4e2d\u7f13\u5b58\u771f\u5b9e\u951a\u70b9\u7684\u5bf9\u9f50\u4f30\u8ba1\uff0c\u540c\u65f6\u62a5\u544a\u5176\u5bf9\u5168\u90e8\u6e90\u951a\u70b9\u7684\u8986\u76d6\u7387\u3002",
        "", "| \u4efb\u52a1 | \u9009\u6837\u65b9\u6cd5 | \u9009\u4e2d\u6e90\u7a97\u53e3 | \u9009\u4e2d\u6e90\u951a\u70b9 | \u6e90\u951a\u70b9\u8986\u76d6\u7387 | \u53c2\u8003\u951a\u70b9 | \u5e73\u5747\u65b9\u5411 cos |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |", *prows, "",
        "\u6307\u6807\u4e8c\u635f\u5931\u53e3\u5f84\uff1a\u4ec5\u771f\u5b9e\u6d4b\u8bd5\u6837\u672c\u4fa7\u6309\u4efb\u52a1 H \u4f7f\u7528 480 \u70b9\u5386\u53f2\u548c H \u70b9\u771f\u5b9e\u76ee\u6807\u8ba1\u7b97 MSE \u68af\u5ea6\uff1b\u4f7f\u7528 TimerXL 96 \u70b9\u8f93\u51fa\u5934\u7684\u524d H \u4e2a\u9884\u6d4b\uff0c\u5f52\u4e00\u5316\u4ec5\u4f9d\u8d56\u5386\u53f2\u3002\u5b89\u5168\u5386\u53f2\u4fa7\u4e0d\u91cd\u7b97\uff0c\u76f4\u63a5\u4f7f\u7528\u540c\u4e00 step-1000 \u68c0\u67e5\u70b9\u7684\u65e2\u6709\u68af\u5ea6\u7f13\u5b58\u3002",
        "",
        "\u8f93\u5165\u4e0e\u4ea7\u7269\uff1asrc-96 \u68af\u5ea6\u7f13\u5b58\u3001\u5404\u6279\u6b21 window_indices.npy\u548c\u5b89\u5168\u5386\u53f2\u7f13\u5b58\uff1b\u7ed3\u679c\u6e05\u5355\u3001\u68af\u5ea6\u8349\u56fe\u4e0e manifest \u4f4d\u4e8e outputs/selection/v1-crb-redo/sgwdev/metrics/\u3002"
    ]
    body[3:3] = ["", "\u7edf\u4e00\u68af\u5ea6\u914d\u5bf9\u534f\u8bae\uff08\u7528\u6237\u786e\u8ba4\uff09\uff1a\u5168\u6e90\u4e0e top10% \u7b5b\u9009\u540e\u6307\u6807\u53ea\u914d\u5bf9\u5df2\u7f13\u5b58\u7684\u771f\u5b9e\u6e90\u951a\u70b9\u68af\u5ea6\u548c\u5df2\u7f13\u5b58\u7684\u540c\u65b9\u6cd5\u5b89\u5168\u5386\u53f2\u951a\u70b9\u68af\u5ea6\uff1b\u5b89\u5168\u5386\u53f2\u2014\u6d4b\u8bd5\u6307\u6807\u53ea\u914d\u5bf9\u5b89\u5168\u5386\u53f2\u7f13\u5b58\u951a\u70b9\u4e0e\u6309 H \u65b0\u7b97\u7684\u771f\u5b9e\u6d4b\u8bd5\u6837\u672c\u68af\u5ea6\u3002\u4e09\u9879\u90fd\u4e0d\u7528\u91cd\u5efa\u9010\u7a97\u53e3\u68af\u5ea6\uff1b\u5168\u6e90\u548c\u7b5b\u9009\u540e\u7ed3\u679c\u4ec5\u4ee3\u8868\u5bf9\u5e94\u7684\u7f13\u5b58\u951a\u70b9\u5b50\u96c6\u4f30\u8ba1\u3002\u5bf9\u5df2\u9009\u68af\u5ea6\u5355\u4f4d\u5316\u540e\u7684\u5168\u90e8\u4ea4\u53c9\u7ec4\u5408\u6c42\u4f59\u5f26\u7b97\u672f\u5747\u503c\uff0c\u4ee5\u53cc\u65b9\u5355\u4f4d\u68af\u5ea6\u5747\u503c\u7684\u70b9\u79ef\u7b49\u4ef7\u7d2f\u52a0\uff0c\u4e0d\u5b9e\u4f53\u5316\u4f59\u5f26\u77e9\u9635\u3002"]
    section = "\n".join(body) + "\n"
    results_path = docs / "exp/v1-crb-redo/results.md"
    text = results_path.read_text(encoding="utf-8")
    marker = "## \u8865\u7b97\u65b9\u5411\u5bf9\u9f50\u6307\u6807\uff08sgwdev\uff09"
    text = text[:text.index(marker)].rstrip() if marker in text else text.rstrip()
    text += "\n\n" + section
    core_batches = (
        "timerXL-base-uni-01-traj-rr",
        "timerXL-base-uni-01-block-rr",
        "timerXL-ours-rage101-01-traj-rr",
        "timerXL-ours-rage101-01-block-rr",
    )
    task_aliases = {
        "FEV hospital_admissions_1D": "fev-09",
        "TIME CPHL/15T": "time-01", "TIME CPHL/30T": "time-02",
        "TIME CPHL/H": "time-03", "TIME ECDC_COVID/D": "time-04",
        "TIME epf_electricity_price/H": "time-06",
        "TIME Commodity_Import/M": "time-07",
        "TIME Commodity_Production/M": "time-08",
    }
    selected_values = {}
    for task in task_aliases.values():
        vals = [float(r["mean_selected_direction_cosine"]) for r in post_rows
                if r["task"] == task and r["selection_batch"] in core_batches]
        selected_values[task] = (sum(vals) / len(vals), len(vals)) if vals else (None, 0)
    doc_lines = text.splitlines()
    in_summary = False
    for i, line in enumerate(doc_lines):
        if line.startswith("| \u4efb\u52a1 | baseline full |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")][:10]
            doc_lines[i] = "| " + " | ".join(cells + ["\u7b5b\u9009\u540e\u65b9\u5411 cos\uff08top10% RR \u56db arm \u5747\u503c\uff09"]) + " |"
            in_summary = True
            continue
        if in_summary and line.startswith("| ---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")][:10]
            doc_lines[i] = "| " + " | ".join(cells + ["---"]) + " |"
            continue
        if in_summary and not line.startswith("|"):
            in_summary = False
            continue
        if not in_summary:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] in task_aliases:
            val, count = selected_values[task_aliases[cells[0]]]
            extra = ("\u2014" if val is None else "%.8f (%d/4)" % (val, count))
            doc_lines[i] = "| " + " | ".join(cells[:10] + [extra]) + " |"
        elif cells[0] == "\u516b\u4efb\u52a1\u7b49\u6743\u5b8f\u5e73\u5747\uff08RR \u56db arm\uff09":
            vals = [v for v, n in selected_values.values() if n == 4]
            extra = ("\u2014" if len(vals) != 8 else "%.8f" % (sum(vals) / 8.0))
            doc_lines[i] = "| " + " | ".join(cells[:10] + [extra]) + " |"
    text = "\n".join(doc_lines) + "\n"
    text = re.sub(r"\u6700\u540e\u4fee\u6539\u65e5\u671f\uff1a[^\n]*", "\u6700\u540e\u4fee\u6539\u65e5\u671f\uff1a" + stamp, text, count=1)
    results_path.write_text(text, encoding="utf-8")

    exp_path = docs / "exp/v1-crb-redo/exp.md"
    text = exp_path.read_text(encoding="utf-8")
    old = ("\u8ba1\u7b97\u811a\u672c\uff1a" + tick + "scripts/exp/v1-crb-redo/metrics.py" + tick +
           "\u3002\u7ed3\u679c\u4e0e\u6e05\u5355\uff1a" + tick + "outputs/selection/v1-crb-redo/sgwdev/metrics/" + tick +
           "\uff1b\u65e5\u5fd7\uff1a" + tick + "logs/v1-crb-redo/" + tick +
           "\u3002\u6210\u529f\u5b8c\u6210\u540e\u5199\u5165 " + tick + ".complete" + tick +
           "\uff0c\u5e76\u5728\u672c\u5b9e\u9a8c results.md \u53ca\u5168\u5c40\u7d22\u5f15\u4e2d\u767b\u8bb0\u5b9e\u9645\u6570\u503c\u3001\u6837\u672c\u8ba1\u6570\u3001\u8f93\u5165\u8def\u5f84\u548c\u8ba1\u7b97\u914d\u7f6e\u3002")
    new = ("\u8ba1\u7b97\u811a\u672c\uff1a" + tick + "scripts/exp/v1-crb-redo/metrics.py" + tick +
           "\u3002\u8865\u7b97\u5df2\u5b8c\u6210\uff0c8 \u9879\u4efb\u52a1\u5404\u6709 CRB \u4e0e Uniform \u4e24\u884c\u7ed3\u679c\uff1b\u8f93\u51fa\u3001\u9010\u9879 CSV\u3001\u68af\u5ea6\u8349\u56fe\u4e0e\u6e05\u5355\u4f4d\u4e8e " +
           tick + "outputs/selection/v1-crb-redo/sgwdev/metrics/" + tick +
           "\uff0c\u6210\u529f\u6807\u8bb0\u4e3a\u8be5\u76ee\u5f55 " + tick + ".complete" + tick +
           "\u3002\u5b9e\u9645\u6570\u503c\u3001\u6837\u672c\u8ba1\u6570\u53ca\u8f93\u5165\u8def\u5f84\u5df2\u767b\u8bb0\u4e8e results.md \u4e0e\u5168\u5c40\u7d22\u5f15\u3002")
    text = text.replace(old, new, 1)
    text = re.sub(r"\u6700\u540e\u4fee\u6539\u65e5\u671f\uff1a[^\n]*", "\u6700\u540e\u4fee\u6539\u65e5\u671f\uff1a" + stamp, text, count=1)
    exp_path.write_text(text, encoding="utf-8")

    aliases = ("timerXL-base-uni-01-traj-rr", "timerXL-base-uni-01-block-rr",
               "timerXL-ours-rage101-01-traj-rr", "timerXL-ours-rage101-01-block-rr",
               "timerXL-base-uni-01-win-rr", "timerXL-ours-rage101-01-win-rr",
               "timerXL-baseline-full", "timerXL-baseline-rand01")
    dv = [float(r["mean_direction_cosine"]) for r in direction_rows]
    av = [float(r["mean_history_test_cosine"]) for r in alignment_rows]
    sv = [float(r["mean_selected_direction_cosine"]) for r in post_rows]
    summary = ("\u5168\u6e90\u65b9\u5411 cos \u8303\u56f4 %.8f\u2013%.8f\uff1b\u5b89\u5168\u5386\u53f2\u2014\u771f\u5b9e\u6d4b\u8bd5\u65b9\u5411 cos \u8303\u56f4 %.8f\u2013%.8f\uff1b"
               "top10%% \u7b5b\u9009\u540e cos \u8303\u56f4 %.8f\u2013%.8f\uff1b\u8be6\u7ec6\u6570\u503c\u89c1 results.md\u3002" % (min(dv), max(dv), min(av), max(av), min(sv), max(sv)))
    exp_index = docs / "EXP.md"
    text = exp_index.read_text(encoding="utf-8")
    for name in aliases:
        text = text.replace("| v1-crb-redo | " + name + " |",
                            "| v1-crb-redo | " + name + "-sgwdev |")
    old_count = "| " + tick + "v1-crb-redo" + tick + " | [v1-crb-redo.md](exp/v1-crb-redo/exp.md) | 14 |"
    new_count = "| " + tick + "v1-crb-redo" + tick + " | [v1-crb-redo.md](exp/v1-crb-redo/exp.md) | 15 |"
    text = text.replace(old_count, new_count, 1)
    exp_row = ("| v1-crb-redo | sgwdev-metrics | 8 \u4efb\u52a1\u7684\u5168\u6e90\u65b9\u5411 cos\u3001\u5b89\u5168\u5386\u53f2\u2014\u6d4b\u8bd5\u68af\u5ea6\u5bf9\u9f50\u548c top10% \u7b5b\u9009\u540e\u65b9\u5411\u63a5\u8fd1\u5ea6 | "
               "\u5b8c\u6210\uff1a16/16 \u4efb\u52a1-\u65b9\u6cd5\u7ec4\u5408\u6307\u6807\u5df2\u8bb0\u5f55\uff0c\u6279\u6b21 .complete \u5df2\u751f\u6210 | " + summary)
    anchor = next((x for x in text.splitlines()
                   if x.startswith("| v1-crb-redo | timerXL-baseline-rand01-sgwdev |")), None)
    existing = next((x for x in text.splitlines()
                     if x.startswith("| v1-crb-redo | sgwdev-metrics |")), None)
    if existing:
        text = text.replace(existing, exp_row, 1)
    elif anchor:
        text = text.replace(anchor, anchor + "\n" + exp_row, 1)
    else:
        raise ValueError("EXP.md insertion anchor missing")
    exp_index.write_text(text, encoding="utf-8")

    result_index = docs / "RESULTS.md"
    text = result_index.read_text(encoding="utf-8")
    for name in aliases:
        text = text.replace("| v1-crb-redo | " + name + " |",
                            "| v1-crb-redo | " + name + "-sgwdev |")
    result_row = ("| v1-crb-redo | sgwdev-metrics | \u5168\u6e90\u65b9\u5411 cos\u3001\u5b89\u5168\u5386\u53f2\u2014\u6d4b\u8bd5\u5bf9\u9f50\u3001top10% \u7b5b\u9009\u540e cos | "
                  "\u5b8c\u6210\uff1a16 \u7ec4\u5168\u6e90\u4f30\u8ba1\uff0c16 \u7ec4\u5386\u53f2-\u6d4b\u8bd5\u5bf9\u9f50\uff0c" + str(len(post_rows)) + " \u4e2a\u5b8c\u6210\u9009\u6837\u7ec4\u5408 | " + summary +
                  "\uff1b\u8be6\u7ec6\u6570\u503c\u4e0e MASE \u8868\u89c1 docs/exp/v1-crb-redo/results.md")
    anchor = next((x for x in text.splitlines()
                   if x.startswith("| v1-crb-redo | timerXL-baseline-rand01-sgwdev |")), None)
    existing = next((x for x in text.splitlines()
                     if x.startswith("| v1-crb-redo | sgwdev-metrics |")), None)
    if existing:
        text = text.replace(existing, result_row, 1)
    elif anchor:
        text = text.replace(anchor, anchor + "\n" + result_row, 1)
    else:
        raise ValueError("RESULTS.md insertion anchor missing")
    result_index.write_text(text, encoding="utf-8")

    log_dir = docs / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = now.strftime("%Y-%m-%d-%H%M%S") + "-v1-crb-redo-sgwdev-metrics.md"
    log_path = log_dir / log_name
    n = 1
    while log_path.exists():
        log_path = log_dir / (now.strftime("%Y-%m-%d-%H%M%S") +
                              "-v1-crb-redo-sgwdev-metrics-%02d.md" % n)
        n += 1
    log_text = (
        "# v1-crb-redo sgwdev \u6307\u6807\u8865\u7b97\n\n"
        "- \u64cd\u4f5c\u65f6\u95f4\uff1a" + stamp + "\n"
        "- \u6279\u6b21\uff1asgwdev\uff08single-window-eval\uff09\uff1b\u72b6\u6001\uff1a\u4e09\u9879\u6307\u6807\u5747\u5b8c\u6210\uff0c8 \u9879\u4efb\u52a1\u300116 \u4e2a\u4efb\u52a1-\u65b9\u6cd5\u7ec4\u5408\u3002\n"
        "- \u547d\u4ee4\uff1apython -u scripts/exp/v1-crb-redo/metrics.py\uff1b\u590d\u7528 step-1000 TimerXL LoRA\u3001\u65e2\u6709\u53c2\u8003\u951a\u70b9\u548c src-96 \u5168\u7a97\u53e3\u6c60\u3002\n"
        "- \u8f93\u51fa\uff1aoutputs/selection/v1-crb-redo/sgwdev/metrics/\uff1b\u4e3b\u7a0b\u5e8f\u53d1\u5e03\u6587\u6863\u540e\u5199\u5165 .complete\u3002\n"
        "- \u6587\u6863\uff1a\u66f4\u65b0\u672c\u5b9e\u9a8c exp.md\u3001results.md\u3001\u5168\u5c40 EXP.md\u3001RESULTS.md\uff1b\u73b0\u6709\u8bc4\u6d4b\u6279\u6b21\u903b\u8f91\u540d\u8ffd\u52a0 -sgwdev\uff0c\u539f\u4ea7\u7269\u8def\u5f84\u4fdd\u7559\u3002\n")
    with log_path.open("x", encoding="utf-8") as stream:
        stream.write(log_text)

def main():
    logger = configure_logger("v1_crb_redo_metrics", "v1-crb-redo", "sgwdev", "metrics")
    started = time.monotonic()
    out = output_path("selection", OUT)
    out.mkdir(parents=True, exist_ok=True)
    if (out / ".complete").is_file():
        logger.info("stage=skip reason=complete output=%s", out)
        return
    lock = out / ".run.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, (str(os.getpid()) + "\n").encode("ascii"))
    os.close(fd)
    try:
        logger.info("stage=config batch=sgwdev tasks=%s methods=%s source_windows=%d horizons=%s output=%s",
                    list(H), METHODS, N_SOURCE, H, out)
        direction_file = out / "direction_alignment.jsonl"
        archive = out / "direction_alignment_linear_partial.jsonl"
        if direction_file.is_file():
            prior = read_jsonl(direction_file)
            if prior and prior[0].get("estimator") != "mean_of_cached_true_anchor_gradients":
                if archive.exists():
                    raise FileExistsError("linear partial results already archived")
                os.replace(direction_file, archive)
        direction_rows = read_jsonl(direction_file)
        done = {(r["task"], r["method"]) for r in direction_rows}
        source_info = source_layout()
        for method in METHODS:
            for task in H:
                if (task, method) in done:
                    continue
                row = score_mean(task, method, source_info, logger)
                append_jsonl(direction_file, row)
                direction_rows.append(row)
                done.add((task, method))
        engine, identity, model_fields = build_probe()
        model_key = identity["model_id"] + "@" + identity["revision"]
        alignment_file = out / "history_test_alignment.jsonl"
        if alignment_file.is_file():
            prior = read_jsonl(alignment_file)
            if prior and prior[0].get("reference_cache_horizon") != 96:
                raise ValueError("existing alignment results use a different reference protocol")
        alignment_rows = read_jsonl(alignment_file)
        done = {(r["task"], r["method"]) for r in alignment_rows}
        for task in H:
            test_values, test_meta = fev_test_windows(logger) if task == "fev-09" else time_test_windows(task, logger)
            test_masks = test_meta.pop("target_masks", None)
            probe = HorizonProbe(engine, H[task], model_key)
            for method in METHODS:
                if (task, method) in done:
                    continue
                row = alignment_row(task, method, probe, test_values, test_meta, logger, test_masks)
                append_jsonl(alignment_file, row)
                alignment_rows.append(row)
                done.add((task, method))
            del probe, test_values
            torch.cuda.empty_cache()
        post_file = out / "post_selection_alignment.jsonl"
        post_rows = read_jsonl(post_file)
        post_done = {(r["task"], r["selection_batch"]) for r in post_rows}
        direction_lookup = {(r["task"], r["method"]): r for r in direction_rows}
        for selection_batch, method in SELECTION_BATCHES:
            for task in H:
                key = (task, selection_batch)
                if key in post_done:
                    continue
                row = post_selection_row(task, selection_batch, method, source_info,
                                         direction_lookup[(task, method)], logger)
                if row is not None:
                    append_jsonl(post_file, row)
                    post_rows.append(row)
                    post_done.add(key)

        keys = {(task, method) for task in H for method in METHODS}
        if ({(r["task"], r["method"]) for r in direction_rows} != keys or
                {(r["task"], r["method"]) for r in alignment_rows} != keys):
            raise ValueError("one or more task metrics are missing")
        order = {task: i for i, task in enumerate(H)}
        direction_rows.sort(key=lambda r: (order[r["task"]], r["method"]))
        alignment_rows.sort(key=lambda r: (order[r["task"]], r["method"]))
        write_csv(out / "direction_alignment.csv", direction_rows)
        safe_csv = [{k: json.dumps(v, ensure_ascii=True, sort_keys=True)
                     if isinstance(v, (dict, list)) else v for k, v in row.items()}
                    for row in alignment_rows]
        write_csv(out / "history_test_alignment.csv", safe_csv)
        write_csv(out / "post_selection_alignment.csv", post_rows)
        publish_docs(direction_rows, alignment_rows, post_rows)
        write_json(out / "manifest.json", dict(
            batch="sgwdev", full_name="single-window-eval", tasks=list(H), methods=list(METHODS),
            source_windows=N_SOURCE, score_estimator="mean_of_cached_true_anchor_gradients", context_length=CONTEXT,
            output_head_length=HEAD, task_horizons=H, model=identity,
            gradient_protocol=model_fields, warmup_adapter=str(WARM),
            script="scripts/exp/v1-crb-redo/metrics.py", elapsed_seconds=time.monotonic()-started,
            direction_csv=str(out / "direction_alignment.csv"),
            alignment_csv=str(out / "history_test_alignment.csv"),
            post_selection_csv=str(out / "post_selection_alignment.csv")))
        (out / ".complete").write_text("complete\n", encoding="ascii")
        logger.info("stage=complete direction_rows=%d alignment_rows=%d elapsed=%s output=%s",
                    len(direction_rows), len(alignment_rows),
                    format_duration(time.monotonic()-started), out)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
