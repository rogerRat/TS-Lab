"""Export only pre-exam references for task/parameter probes; no training run."""
import argparse
import json
from pathlib import Path

import numpy as np

from rags.core import TimeSeriesFragment
from .development_inputs import load_fragments, save_fragments
from .probing import write_json


def references(task):
    benchmark = task["benchmark"]
    if benchmark == "quito":
        from evaluate.quito_data import iter_series, validate_snapshot, SEASONALITY, CUTOFF, REVISION
        root = Path(task["root"])
        validate_snapshot(root)
        rows = ((entity, values[:boundary]) for entity, values, boundary in iter_series(root, task["subset"]))
        provenance = dict(benchmark=benchmark, cutoff=str(CUTOFF), revision=REVISION,
                          seasonality=SEASONALITY[task["subset"]], source_target_dims=[5],
                          channel_policy="multivariate_split_into_channels")
    elif benchmark == "time":
        import yaml
        import datasets
        from evaluate.time import _validate_bench_config
        config = yaml.safe_load(Path(task["evaluation_config"]).read_text())
        checked = _validate_bench_config(config["bench_config"])
        dataset = task["dataset"]
        if dataset not in {item["dataset"] for item in checked["tasks"]}:
            raise ValueError("TIME probe task is absent from the evaluation protocol")
        reserve = int(checked["datasets"][dataset]["test_length"])
        if reserve <= 0:
            raise ValueError("TIME must reserve a positive exam tail")
        loaded = datasets.load_from_disk(str(checked["dataset_root"] / dataset))
        selected = task.get("variate_names")
        if selected is not None and (not selected or len(set(selected)) != len(selected)):
            raise ValueError("variate_names must be non-empty and unique")
        aliases = task.get("variate_name_aliases", {})
        if aliases and (selected is None or not set(aliases) <= set(selected)):
            raise ValueError("Aliases must name explicitly selected variates")
        accepted = {name: [name] + aliases.get(name, []) for name in (selected or [])}
        flat = [name for variants in accepted.values() for name in variants]
        if len(flat) != len(set(flat)) or any(not isinstance(name, str) or not name for name in flat):
            raise ValueError("Variate aliases must be unique non-empty names")
        require_native = task.get("require_native_univariate", False)
        if type(require_native) is not bool:
            raise ValueError("require_native_univariate must be boolean")
        provenance = dict(benchmark=benchmark, excluded_tail=reserve,
                          revision=checked["dataset_values"]["revision"], seasonality=task["seasonality"],
                          source_target_dims=[], selected_variate_names=selected,
                          variate_name_aliases=aliases, resolved_alias_series=[],
                          channel_policy="native_univariate" if require_native else "channelwise")
        def channels():
            for index, row in enumerate(loaded):
                target = np.asarray(row["target"])
                if target.ndim not in (1, 2):
                    raise ValueError("TIME target must be a series or channels-by-time array")
                dimension = 1 if target.ndim == 1 else target.shape[0]
                if require_native and dimension != 1:
                    raise ValueError("Requested native univariate TIME task has multiple targets: " + dataset)
                if dimension not in provenance["source_target_dims"]:
                    provenance["source_target_dims"].append(dimension)
                    provenance["source_target_dims"].sort()
                names = row.get("variate_names")
                selected_actual = None
                if selected is not None:
                    if names is None or len(names) != dimension:
                        raise ValueError("Requested variate_names absent from TIME series: " + dataset)
                    selected_actual = []
                    for canonical, variants in accepted.items():
                        matches = [name for name in names if name in variants]
                        if len(matches) != 1:
                            raise ValueError("Requested variate_names absent or ambiguous in TIME series: " + dataset)
                        selected_actual.extend(matches)
                        if matches[0] != canonical:
                            provenance["resolved_alias_series"].append(dict(
                                entity=str(row.get("item_id", index)), canonical=canonical, actual=matches[0]))
                for channel, values in enumerate(target if target.ndim == 2 else target[None, :]):
                    if selected_actual is not None and names[channel] not in selected_actual:
                        continue
                    yield "%s/%d" % (row.get("item_id", index), channel), values[:-reserve]
        rows = channels()
    elif benchmark == "fev" and "dataset_file" in task:
        # Pinned local scalar-target FEV data; reserve the entire declared exam.
        import yaml
        import datasets
        declarations = yaml.safe_load(Path(task["evaluation_config"]).read_text())["tasks"]
        matches = [d for d in declarations if d["dataset_config"] == task["dataset_config"]]
        if len(matches) != 1:
            raise ValueError("FEV dataset must have exactly one declared task")
        declaration = matches[0]
        target = declaration.get("target", "target")
        if not isinstance(target, str):
            raise ValueError("This FEV export requires a native scalar target")
        if any(k in declaration for k in ("initial_cutoff", "window_step_size")):
            raise ValueError("Explicit FEV cutoff/stride needs a dedicated export")
        horizon, count = declaration["horizon"], declaration["num_windows"]
        if any(type(v) is not int or v <= 0 for v in (horizon, count)):
            raise ValueError("FEV exam horizon and count must be positive integers")
        reserve = horizon * count
        source = Path(task["dataset_file"])
        if source.suffix != ".arrow" or not task.get("revision"):
            raise ValueError("FEV export requires a local Arrow file and pinned revision")
        loaded = datasets.Dataset.from_file(str(source))
        def scalar_rows():
            for index, row in enumerate(loaded):
                values = np.asarray(row[target])
                if values.ndim != 1:
                    raise ValueError("FEV scalar target must be one dimensional")
                yield str(row.get(declaration.get("id_column", "id"), index)), values[:-reserve]
        rows = scalar_rows()
        provenance = dict(benchmark=benchmark, source=str(source), revision=task["revision"],
                          declaration=declaration, excluded_tail=reserve,
                          seasonality=declaration["seasonality"], source_target_dims=[1],
                          channel_policy="native_scalar_target_covariates_unused")
    elif benchmark == "fev":
        # Consume the existing FEV full-pool export, whose reference excludes exam.
        source = Path(task["reference_file"])
        manifest = json.loads(Path(task["manifest_file"]).read_text())
        if manifest.get("source_training_only") is not True and manifest.get("formal_exam_excluded") is not True:
            raise ValueError("FEV reference requires its safe-reference provenance manifest")
        rows = ((f.entity_id, f.data) for f in load_fragments(source))
        provenance = dict(benchmark=benchmark, source=str(source), source_manifest=manifest,
                          seasonality=task["seasonality"])
    else:
        raise ValueError("Probe inputs support fev, time and quito")
    return rows, provenance


def prepare(config, out):
    out = Path(out)
    tasks = config["tasks"]
    ids = [task["id"] for task in tasks]
    if not ids or len(set(ids)) != len(ids) or any(not name or Path(name).name != name or name in {".", ".."} for name in ids):
        raise ValueError("Probe task aliases must be unique safe file stems")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new probe input directory")
    context = config.get("context_length", 512)
    horizon = config.get("prediction_length", 64)
    count = config.get("validation_windows", 8)
    if any(type(v) is not int or v <= 0 for v in (context, horizon, count)):
        raise ValueError("Positive context/horizon/validation count required")
    minimum = 2 * context + (count + 1) * horizon
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        rows, provenance = references(task)
        kept, skipped, skip_reasons = [], [], {}
        for entity, values in rows:
            values = np.asarray(values, dtype=np.float32)
            if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
                skipped.append(str(entity))
                skip_reasons[str(entity)] = ("not_one_dimensional" if values.ndim != 1 else
                    "insufficient_history" if len(values) < minimum else "nonfinite_history")
                continue
            kept.append(TimeSeriesFragment(values.copy(), entity_id=str(entity), fragment_id=len(kept)))
        if not kept:
            raise ValueError("No finite pre-exam series long enough for probe: " + task["id"])
        save_fragments(out / (task["id"] + "__reference.npz"), kept)
        write_json(out / (task["id"] + "__manifest.json"), dict(provenance,
            formal_exam_excluded=True, task=task, kept_series=len(kept), skipped_series=skipped,
            skipped_reasons=skip_reasons,
            minimum_reference_length=minimum, context_length=context,
            prediction_length=horizon, validation_windows=count))
    write_json(out / "probe_inputs.json", dict(config=config, tasks=ids, formal_exam_excluded=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(json.loads(args.config.read_text()), args.out)


if __name__ == "__main__":
    main()
