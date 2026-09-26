# Experiment deployment — 2026-09-11

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

> The later decision 9 extends this to three students, MASE only, and forecast-origin
> history scales. Use [MultiStudent0911.md](MultiStudent0911.md) for current exp
> installation, launch, resume and storage planning; single-student estimates below
> describe the earlier development milestone.

## Authority and scope

The user explicitly restricts the current `dev` container to code development
and testing. Do not launch formal experiments here. Unit/integration tests,
small-model smoke tests and dry-runs are in scope; provisioning formal datasets,
model weights and checkpoint storage is deferred to a future experiment
environment and is not a blocker for current development.

The requested `001-iclr27/rags-md` path is absent locally. The available current
source is `/home/rogers/Projects/001-iclr17/rags-md/_exp-0907.md`, especially
decision 8 and D2, updated September 11. TODO entries still labelled pending
(including the proposed appendix/extra granularity arms) are not auto-promoted.

This conversation's approved baseline method-fidelity decisions supersede stale
paper wording that forces LP-Mirror/LTSV through non-overlap deployment or uses
CE for LTSV reference scoring. External native overlap is disclosed and actual
unique-window/point coverage is now recorded separately from selected N.

## Implemented

- Six tasks sequentially; N=4096/task, one shared 20480-equivalent source pool.
- Four exposure epochs; with-replacement batches of 32; AdamW lr=1e-5, wd=.01.
  Cumulative boundaries are `ceil(epoch*N/batch)`: 128/256/384/512 at main N.
  One continuous optimizer trajectory, not four independently restarted runs.
- Save all epoch weights, then evaluate every saved checkpoint. The final epoch
  determines the main table, never the best test checkpoint. Restore pretrained
  weights before/after each cell, including exceptional exits.
- Data curves use only the epoch schedule. Time-budget runs share the epoch cap,
  keep completed epoch checkpoints and any final partial-epoch state, and record
  actual timings (a wall-clock limit is checked between updates, not mid-update).
- Checkpoint CSV/JSON/HTML, cost/performance HTML and data/performance HTML.
  Cost curves use measured selection + checkpoint training time; exam/reporting
  overhead is separate and wall-clock jobs can exceed a training-only time cap.
- Formal and smoke YAML, CLI dispatch, frozen RUN.md and deployment preflight
  updated; obsolete fixed-step YAML is rejected rather than silently ignored.

## Forecast correction — do not mix old and new results

The former main-table evaluator supplied true future labels to the T5 decoder.
This teacher-forced preceding future values and was not a context-only forecast.
Evaluation now generates from history only and reads each horizon's bin CDF
using the official tokenizer's inverse mapping. Distributions are conditional
on the greedily generated prefix; they are not sampled marginal quantiles.

Forecast protocol ID: `autoregressive_greedy_prefix_bin_cdf`. Old reported
metrics and step-calibration measurements cannot be treated as equivalent to
this corrected exam. Keep the approved 4-epoch configuration, but rerun the
original/random sanity check on real data before interpreting formal results.
Use a new output directory; old selections/cells are not automatically deleted.

## Verification and deployment state

Official Chronos dummy checkpoint smoke (not a trained-small performance test):
`outputs/epoch-0911-official-dummy-smoke/`. Four checkpoint files and evaluations,
final-epoch selection, exact base restoration, future-label isolation and curve
artifacts passed. Unit tests independently check uninterrupted-optimizer parity,
epoch rounding, failure restoration, coverage, CLI config and report generation.

Full regression: 204 passed, 3 skipped, 6 failed. The six failures are unchanged
from the previous commit: two absent FEV resources, two absent TIME checkout
resources, and two pre-existing temporary-config path-policy failures. All six
orchestration stages passed command construction in a dry-run; this is not a
real-data execution test.

Deployment report: `outputs/deployment-0911.json`. Formal training is **not
started**. Missing: source shard bundle, six-task reference bundles, and the
Chronos-small checkpoint. Free storage is ~41.2 GiB; main-table FP32 weights
alone need ~86.4 GiB (504 checkpoints), excluding data, caches and other steps.

For a future formal experiment environment (not the current dev container),
configure input paths in YAML as `source_load`, `reference_root`,
`model_path`; use a sufficiently large `--out-root` for retained checkpoints.
`RAGS_SOURCE_LOAD` / `RAGS_REFERENCE_ROOT` also configure standalone processes.

```bash
python -m scripts.exp.deploy --config configs/rags/run.yaml --out-root /path/to/large/output
# Checks only. Do not use --launch on the current development container.
python -m scripts.exp.rags run --config configs/rags/run.yaml --dry-run
```

No automatic shutdown/watchdog was enabled. No full-data experiment has been
claimed complete merely because code deployment or a dummy-model smoke passed.
