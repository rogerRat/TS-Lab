# RAGS-only implementation cleanup — 2026-09-11

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

Executed after the two requested commits: `a958d27` (method fixes) and `582b515`
(baselines/src layout). This is a separate cleanup, not part of either change.

## Removed active implementations

- `src/ts3/core`: old OT/Sinkhorn, KDE, candidate-uniform/vote/ratio, matched
  concentration, transfer caches, exposure/sampling and ticket schedulers.
- `src/ts3/features`, feature/catalog production and legacy grid/descriptor
  analysis/reporting; old catalog/runtime/training protocol objects.
- Old `src/train` and Chronos policy/scheduler adapter; obsolete experiment
  runners, watchers, 113 old method configurations and 28 dedicated test files.
- Legacy random-source warmup loader tied to the deleted scheduler; generic
  gradient engines/projection and in-memory warmup utilities remain.
- Unused POT dependency and transfer-policy path dispatch.

Generic dedup coverage remains: its test now verifies shard preservation and
duplicate flags without invoking the removed feature/catalog stages.

## Preserved

- All RAGS selection, CRB, downstream training, ablations and evaluation code.
- The four external baselines. TSDS's own OT/KDE is part of that external
  method, not the deleted project OT/KDE implementation.
- Random/trajectory controls and the RAGS 2x2 ablation matrix (the word “grid”
  here denotes an experiment matrix, not the old feature-grid selection method).
- Source/reference fetch, load, dedup and shard schemas; model adapters,
  gradient projection, generic runtime/evaluation/plotting and CLI infrastructure.
- Existing data/model/result files and historical documents/archives. Shard
  schemas may still recognize historical feature metadata for file compatibility;
  this does not retain any feature selection algorithm.

Use `scripts.exp.rags` for experiments, `scripts.assets` for data stages, and
`scripts.exp.model_{probe,select,train}` for foundation-model workflows.
The duplicate `scripts.run` wrapper has been removed; use `python -m scripts.exp.rags run`.

## Verification scope

First post-cleanup full run: 188 passed, 3 skipped, 6 failed. All six failures
already occurred before cleanup: two unavailable local FEV datasets, two absent
TIME checkout fixtures, and two reference-fetch tests using paths rejected by
the existing configuration policy. Removed-method tests were retired with their
implementations; a smaller failure count is not a claim to have fixed them.

Deleted tracked content is recoverable from Git (the pre-cleanup tree is
`582b515`). Only bytecode caches in verified removed code directories were also
removed; these are generated artifacts. No experiment artifacts were deleted.
