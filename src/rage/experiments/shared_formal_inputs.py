"""Native held-out exams and complete pre-exam references for the shared tasks.

NPZ/JSON files are published atomically in the caller's TMP inputs directory.
Only native observations and legal windows are stored: no interpolated gradients.
The per-task manifest is the completion marker and records explicit identities.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from evaluate.quito_data import window_starts
from rage.core import TimeSeriesFragment
from .development_inputs import save_fragments


PROTOCOL = "shared-ten-native-rolling-h64-v1"


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def native_rows(task):
    """Return (entity, full native values, first test index) and provenance."""
    if task["benchmark"] == "quito":
        from evaluate.quito_data import iter_series, validate_snapshot, CUTOFF, REVISION, SEASONALITY
        root = Path(task["root"])
        validate_snapshot(root)
        return iter_series(root, task["subset"]), dict(
            benchmark="quito", revision=REVISION, cutoff=str(CUTOFF),
            seasonality=SEASONALITY[task["subset"]], source=str(root),
            channel_policy="native_channelwise")
    if task["benchmark"] != "time":
        raise ValueError("Shared formal tasks require TIME or Quito native data")
    import datasets
    import yaml
    from evaluate.time import _validate_bench_config
    config = yaml.safe_load(Path(task["evaluation_config"]).read_text())
    checked = _validate_bench_config(config["bench_config"])
    dataset = task["dataset"]
    if dataset not in {item["dataset"] for item in checked["tasks"]}:
        raise ValueError("TIME task is absent from the pinned evaluation protocol")
    reserve = int(checked["datasets"][dataset]["test_length"])
    if reserve <= 0:
        raise ValueError("Native exam tail must be positive")
    selected = task.get("variate_names")
    aliases = task.get("variate_name_aliases", {})
    if selected is not None and (not selected or len(set(selected)) != len(selected)):
        raise ValueError("variate_names must be unique and nonempty")
    if aliases and (selected is None or not set(aliases) <= set(selected)):
        raise ValueError("Aliases require explicitly selected variates")
    accepted = {name: [name] + aliases.get(name, []) for name in (selected or [])}
    flat = [name for variants in accepted.values() for name in variants]
    if len(flat) != len(set(flat)) or any(not isinstance(name, str) or not name for name in flat):
        raise ValueError("Variate aliases must be unique nonempty names")
    source = checked["dataset_root"] / dataset
    loaded = datasets.load_from_disk(str(source))
    provenance = dict(benchmark="time", revision=checked["dataset_values"]["revision"],
                      excluded_tail=reserve, seasonality=int(task["seasonality"]),
                      source=str(source), selected_variate_names=selected,
                      variate_name_aliases=aliases, resolved_alias_series=[],
                      channel_policy="native_channelwise")

    def channels():
        for index, row in enumerate(loaded):
            target = np.asarray(row["target"], dtype=np.float64)
            if target.ndim not in (1, 2):
                raise ValueError("TIME target must be scalar series or channels by time")
            target = target[None, :] if target.ndim == 1 else target
            if task.get("require_native_univariate", False) and len(target) != 1:
                raise ValueError("Expected native univariate TIME task")
            names = row.get("variate_names")
            selected_actual = None
            if selected is not None:
                if names is None or len(names) != len(target):
                    raise ValueError("Requested variate names are absent")
                selected_actual = []
                for canonical, variants in accepted.items():
                    matches = [name for name in names if name in variants]
                    if len(matches) != 1:
                        raise ValueError("Requested variate is absent or ambiguous: " + canonical)
                    selected_actual.extend(matches)
                    if matches[0] != canonical:
                        provenance["resolved_alias_series"].append(dict(
                            entity=str(row.get("item_id", index)), canonical=canonical, actual=matches[0]))
            for channel, values in enumerate(target):
                if selected_actual is None or names[channel] in selected_actual:
                    yield f"{row.get('item_id', index)}/{channel}", values, max(0, len(values) - reserve)
    return channels(), provenance


def build_task_arrays(rows, *, seasonality, context_length=512, prediction_length=64,
                      evaluation_mode="rolling", exam_stride=None):
    """Keep every legal finite reference segment and causal native exam window."""
    if (context_length, prediction_length) not in ((512, 64), (480, 96)) or evaluation_mode != "rolling":
        raise ValueError("Formal shared rolling protocol supports only C512/H64 or Timer V6 C480/H96")
    exam_stride = prediction_length if exam_stride is None else int(exam_stride)
    if type(seasonality) is not int or seasonality <= 0:
        raise ValueError("Native seasonality must be a positive integer")
    refs, windows, entities, scales, series = [], [], [], [], []
    seen = set()
    for entity, values, boundary in rows:
        entity = str(entity)
        if entity in seen:
            raise ValueError("Duplicate native channel identity: " + entity)
        seen.add(entity)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not 0 <= boundary <= len(values):
            raise ValueError("Invalid native scalar series or exam boundary")
        boundary = int(boundary)
        history = values[:boundary]
        finite = np.isfinite(history)
        transitions = np.diff(np.r_[False, finite, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))
        bounds = []
        for begin, end in runs:
            if end - begin < context_length + prediction_length:
                continue
            fragment_id = len(refs)
            refs.append(TimeSeriesFragment(history[begin:end].astype(np.float32),
                entity_id=entity, fragment_id=fragment_id))
            bounds.append(dict(fragment_id=fragment_id, start=int(begin), stop=int(end)))
        # Prefix sums of finite native seasonal differences provide exactly
        # the full true-history denominator at each forecast origin.
        differences = np.abs(values[seasonality:] - values[:-seasonality])
        valid = np.isfinite(differences)
        sums = np.r_[0.0, np.cumsum(np.where(valid, differences, 0.0))]
        counts = np.r_[0, np.cumsum(valid)]
        starts = window_starts(len(values), boundary, context_length, prediction_length,
                               evaluation_mode, exam_stride)
        origins, invalid = [], 0
        for origin in starts:
            window = values[origin-context_length:origin+prediction_length]
            if not np.isfinite(window).all():
                invalid += 1
                continue
            index = max(0, origin - seasonality)
            scale = float(sums[index] / counts[index]) if counts[index] else float("nan")
            windows.append(window.astype(np.float32))
            entities.append(entity)
            scales.append(scale)
            origins.append(int(origin))
        series.append(dict(entity=entity, length=len(values), exam_boundary=boundary,
                           reference_segments=bounds, exam_origins=origins,
                           candidate_exam_windows=len(starts), invalid_exam_windows=invalid,
                           nonfinite_reference_points=int((~finite).sum()),
                           unused_tail_points=(len(values)-boundary) % prediction_length))
    if not refs or not windows:
        raise ValueError("Formal task needs finite legal references and native exam windows")
    return refs, dict(windows=np.stack(windows), entities=np.asarray(entities),
                     scales=np.asarray(scales, dtype=np.float64)), series


def prepare_shared_formal_inputs(shared_decisions, out, *, evaluation_mode="rolling",
                                 context_length=512, prediction_length=64,
                                 protocol=PROTOCOL, exam_stride=None):
    """Export the approved tasks from native data, never their dev exam arrays.

    ``input_roots`` identifies historical task declaration manifests only.
    Existing completed exports must match every explicit protocol identity.
    """
    out = Path(out)
    tasks = shared_decisions["datasets"]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Shared task IDs must be nonempty and unique")
    if evaluation_mode != "rolling":
        raise ValueError("Formal shared exams require rolling")
    out.mkdir(parents=True, exist_ok=True)
    manifests = []
    for dataset in tasks:
        if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
            raise ValueError("Unsafe task file stem")
        declaration_path = Path(shared_decisions["input_roots"][dataset]) / f"{dataset}__manifest.json"
        declaration = json.loads(declaration_path.read_text())
        task = declaration["task"]
        if task["id"] != dataset:
            raise ValueError("Task declaration identity mismatch")
        stride = prediction_length if exam_stride is None else int(exam_stride)
        identity = dict(protocol=protocol, task=task, revision=declaration["revision"],
                        context_length=context_length, prediction_length=prediction_length,
                        evaluation_mode=evaluation_mode, exam_stride=stride, source_training_only=True,
                        formal_exam_excluded_from_reference=True, full_reference_history=True)
        done = out / f"{dataset}__manifest.json"
        reference_path = out / f"{dataset}__reference.npz"
        exam_path = out / f"{dataset}__exam.npz"
        if done.exists():
            saved = json.loads(done.read_text())
            if any(saved.get(k) != v for k, v in identity.items()):
                raise ValueError("Existing formal task identity mismatch: " + dataset)
            if not reference_path.is_file() or not exam_path.is_file():
                raise ValueError("Incomplete completed formal task: " + dataset)
            manifests.append(saved)
            continue
        rows, provenance = native_rows(task)
        if provenance["revision"] != identity["revision"]:
            raise ValueError("Native data revision differs from task declaration")
        refs, exam, series = build_task_arrays(rows, seasonality=int(provenance["seasonality"]),
                                               context_length=context_length,
                                               prediction_length=prediction_length,
                                               evaluation_mode=evaluation_mode,
                                               exam_stride=stride)
        partial = reference_path.with_name(reference_path.stem + ".partial.npz")
        save_fragments(partial, refs)
        os.replace(partial, reference_path)
        _atomic_npz(exam_path, **exam)
        manifest = dict(identity, **provenance, dataset=dataset, series=series,
                        reference_fragments=len(refs), exam_windows=len(exam["windows"]),
                        reference_lengths=[f.length for f in refs],
                        declaration_manifest=str(declaration_path),
                        selection_scope="user_approved_shared_ten_formal_migration")
        _atomic_json(done, manifest)
        manifests.append(manifest)
    _atomic_json(out / "shared_formal_inputs.json", dict(
        protocol=protocol, datasets=tasks, evaluation_mode=evaluation_mode,
        context_length=context_length, prediction_length=prediction_length, exam_stride=stride,
        formal_exam_used_for_parameter_selection=False))
    return manifests
