# RAGE Project Layout

Last modified: 2026-11-07

## Main workflow

1. `assets` downloads, converts, and validates source data. Deduplication and gradient assets are optional stages.
2. `rage.static_selection` selects a fixed subset from prepared windows. `rr` cycles through groups in first-seen order, taking each group's highest-scoring remaining row first; `topk` sorts finite scores from highest to lowest and keeps input order for ties. `scripts/select_static.py` writes the selected rows, IDs, indices, protocol, and completion marker under `outputs/selection/`.
3. `rage.experiments.students` and `rage.experiments.train_eval` train models on shared selection artifacts and evaluate them.

The selection input is an NPZ with `windows` and `ids`. Both methods need a finite one-dimensional `scores` array. RoundRobin also needs a one-dimensional `groups` array. Use `--ratio` or `--count`, not both.

## Naming

`rage` names the project, algorithms, selection methods, training workflows, runtime variables, and checkpoints. `tsl` is the Python package for TimeseriesLab data structures and specifications; serialized specification identifiers use the `tsl.` prefix.
