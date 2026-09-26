"""Read-only prerequisites for launching the document-frozen experiment."""
from __future__ import annotations

import shutil
from pathlib import Path

from configs.paths import PROJECT_ROOT
from rags.engines import DEFAULT_CHRONOS_T5_SMALL
from .data import SOURCE_LOAD, REFERENCE_ROOT
from .schedule import epoch_checkpoints
from .tasks import MAIN_DATASETS, build_task_specs
from .students import PARAMETER_COUNTS, STUDENTS, key_values


def check_deployment(config: dict, *, out_root: Path) -> dict:
    source = (PROJECT_ROOT / config.get("source_load", SOURCE_LOAD)).resolve()
    reference = (PROJECT_ROOT / config.get("reference_root", REFERENCE_ROOT)).resolve()
    model = (PROJECT_ROOT / (config.get("model_path") or DEFAULT_CHRONOS_T5_SMALL)).resolve()
    blockers = []
    for entry in (source / "index.arrow", source / "meta.arrow", source / "shards"):
        if not entry.exists():
            blockers.append("missing source data: %s" % entry)
    try:
        specs = build_task_specs(keywords=tuple(config.get("datasets") or MAIN_DATASETS),
                                 reference_root=reference)
        for spec in specs:
            for entry in (spec.load_dir / "index.arrow", spec.load_dir / "meta.arrow", spec.load_dir / "shards"):
                if not entry.exists():
                    blockers.append("missing reference data: %s" % entry)
    except FileNotFoundError as exc:
        blockers.append("reference data unavailable: %s" % exc)
    if not (model / "config.json").is_file() or not (
            list(model.glob("*.safetensors")) or list(model.glob("pytorch_model*.bin"))):
        blockers.append("missing Chronos-small checkpoint: %s" % model)
    epochs = int(config.get("source_epochs", 4))
    n = int(config.get("n_windows", 4096))
    batch = int(config.get("train_batch", 32))
    boundaries = epoch_checkpoints(n, batch, epochs)
    # Lower-bound estimate: main-table weights only, excluding all caches/curves.
    cells = len(config.get("datasets") or MAIN_DATASETS) * len(config.get("seeds", [0, 1, 2]))
    train_arms = 3 + len(config.get("with_baselines", ["less", "tsds", "lp_mirror", "ltsv"]))
    students = config.get("students", ["chronos-small"])
    checkpoint_bytes = cells * train_arms * epochs * sum(PARAMETER_COUNTS[s] for s in students) * 4
    model_paths = key_values(config.get("student_model_paths", []))
    model_sources = {s: model_paths.get(s, "pinned Hugging Face checkpoint (download on first load)") for s in students}
    if "chronos-small" in students:
        model_sources["chronos-small"] = str(model)
    for student, location in model_paths.items():
        if not (PROJECT_ROOT / location / "config.json").is_file():
            blockers.append("missing student checkpoint: %s at %s" % (student, location))
    existing = out_root.resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < checkpoint_bytes:
        blockers.append("main-table weights alone need ~%.1f GiB; %.1f GiB free" %
                        (checkpoint_bytes / 2**30, free / 2**30))
    return {"ready": not blockers, "blockers": blockers,
            "source_load": str(source), "reference_root": str(reference),
            "model_path": str(model), "checkpoint_steps": boundaries,
            "students": students, "student_model_sources": model_sources,
            "estimated_main_checkpoint_bytes": checkpoint_bytes, "free_bytes": free,
            "checkpoint_policy": "all epochs retained; final epoch reported",
            "forecast_protocol": "MASE: Chronos median / TimesFM mean_predictions / Time-MoE autoregressive"}
