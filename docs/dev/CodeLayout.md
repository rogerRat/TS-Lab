# Code layout (2026-09-11)

```text
baselines/     common dispatch + less/, tsds/, lp_mirror/, ltsv/ packages
src/           project-owned implementation packages
  rags/        RAGS selection and experiment services
  ts3/         shared shard-data schemas only (legacy selection core removed)
  adapters/    foundation-model adapters
  gradient/    gradient extraction and projection
  assets/      data-processing implementation (not runtime datasets)
  evaluate/ runtime/ hub/ visualize/  retained infrastructure
scripts/       orchestration and CLI entry points
configs/       configuration files and shared path/logging helpers
tests/         all regression tests
requ/          dependency and upstream-revision locks
```

Import names stay unchanged (`rags`, `ts3`, `gradient`, etc.); external methods
now use `baselines.algorithms`, `baselines.contracts`, and `baselines.runners`.
Each method owns its implementation under `baselines.<method>`; the top-level
modules only provide compatibility exports and unified dispatch/registry.
There are no root-level compatibility copies of the moved implementation trees.

Activate an existing environment, then run from the repository root:

```bash
python -m pip install --no-deps -e .
python -m scripts.exp.rags main_table --help
python -m scripts.exp.rags run --help
python -m pytest tests/test_time_series_baselines.py -q
```

The installer registers an editable installation for each environment. Pytest
also declares `src` and the repository root as import paths. Project data/output
paths come from `configs.paths`, never from the depth of a moved source file.
Existing `python -m rags.experiments.*` invocations continue to work; new CLI use
should go through `scripts/`. This migration moves implementation ownership,
not every legacy command parser or workflow in one pass.

Two separate commits: method fixes first, layout/packaging second. No datasets,
checkpoints, runtime outputs, or archived experiment artifacts were moved.

A subsequent cleanup removes the legacy OT/KDE/candidate/grid selection stack,
its training scheduler, descriptor production, analysis workflows and configs.
RAGS training remains in `rags.experiments.train_eval`; foundation-model training
remains in `adapters.foundation.training`. External baselines are retained.
Data preparation uses `python -m scripts.assets <stage> --help` (fetch/load/dedup).
Historical docs and experiment archives are records, not supported entry points.

After method changes use a new experiment output directory: completed selections
and result cells are resumable artifacts, not automatically recomputed by a source
layout change. Old artifacts have not been deleted or overwritten.
