# FEV OT/KDE Ten-Step Smoke Training Log

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

Date: 2026-08-04

## Objective

Run ten optimizer steps for the FEV OT and KDE training paths in order to
observe startup latency, planner solve time, steady-state training throughput,
and any blocking behavior.  The smoke configurations preserve the production
model, source catalog, feature settings, and `sampling.trajectory_quality:
count` setting.  They only reduce `training_steps` from 100000 to 10.

## Environment And Inputs

- Remote project root: `/root/TS3`
- Conda environment: `ts3`
- GPU: NVIDIA GeForce RTX 4080 SUPER; approximately 32 GiB free before runs
- Source catalog: `configs/catalog/src/1b.yaml`
- Source bundle: `/root/TS3/data/src/shard_dataset/profiles/1b`
- Original FEV reference parent: `configs/catalog/ref/fev/`
- Model checkpoint: `google/t5-efficient-tiny` snapshot resolved locally

The source runtime build reports 404579 catalog records and 303268 eligible
source trajectories under `context_length: 512`, `prediction_length: 64`, and
`missing_value_policy: reject`.

## Full FEV Catalog Blocker

The original OT and KDE FEV configurations point at all 20 reference catalogs.
`RuntimeReferenceDatasetBuilder` currently rejects a reference catalog when it
has zero trajectories satisfying its runtime validity requirements.  An audit
of all 20 FEV catalogs found three empty references:

| Reference | Records | Valid trajectories |
| --- | ---: | ---: |
| `jena_weather_1H` | 21 | 0 |
| `rohlik_orders_1D` | 7 | 0 |
| `uci_air_quality_1H` | 4 | 0 |

The remaining 17 references have at least one valid trajectory.  Consequently,
the current complete-FEV training configuration fails before OT or KDE
planning, first at `jena_weather_1H`, with:

```text
ValueError: Reference fev-autogluon%2Ffev_datasets%3Ajena_weather_1H has no eligible trajectories
```

This is a data/runtime-reference eligibility issue, not an OT solver or KDE
solver failure.  No change to the core protocol was made here.  A future
decision is required for full-FEV runs: repair/re-extract the empty reference
data or define and document a protocol rule that omits empty references before
planning.

## Smoke-Specific Changes

Two new configurations were added:

- `configs/train/chronos/ts3-profile-1b-fev-smoke-10.yaml`
- `configs/train/chronos/ts3-profile-1b-fev-kde-smoke-10.yaml`

Both retain `trajectory_quality: count`, `batch_size: 32`, and all respective
OT/KDE hyperparameters from the production FEV configurations.  They set:

```yaml
training_steps: 10
audit:
  flush_interval_steps: 1
save_checkpoint_steps: 0
```

The per-step audit flush is intentional: it makes I/O-related stalls visible
in a short run.  Ten optimizer steps therefore mean 320 delivered slots.

Reference discovery was extended as a non-core infrastructure change to accept
either a parent directory of per-asset headers or one per-asset catalog
directory containing its own `<directory-name>.yaml` header. This avoids
duplicating catalog data or maintaining an ignored smoke-only symlink.

`rossmann_1W` is valid and contains 1115 reference trajectories. The smoke
configurations directly point to its existing catalog directory. This is
adequate for observing an actual OT/KDE planner and model training run, but
its planner timing is not a direct measurement of the complete 17-reference
FEV workload.

## Attempts And Status

| Attempt | Outcome | Evidence |
| --- | --- | --- |
| OT with complete FEV parent | Failed during reference runtime build | `/root/autodl-tmp/ts3-fev-ot-smoke-10-20260804.log`; total wall time 33.542 s |
| OT with direct single-catalog directory | Failed during catalog discovery | `/root/autodl-tmp/ts3-fev-ot-smoke-10-20260804-v2.log`; total wall time 33.462 s |
| OT with temporary smoke discovery parent | Planner completed; failed at first scheduled slot audit, before optimizer step 1 | `/root/autodl-tmp/ts3-fev-ot-smoke-10-20260804-v3.log`; run output `/root/TS3/runs/fev-ot-smoke-10-20260804-v3`; total wall time 66.401 s |
| KDE with temporary smoke discovery parent | Planner completed; failed at first scheduled slot audit, before optimizer step 1 | `/root/autodl-tmp/ts3-fev-kde-smoke-10-20260804.log`; run output `/root/TS3/runs/fev-kde-smoke-10-20260804`; total wall time 51.244 s |

The OT v3 and planned KDE command are intentionally serialized on the single
GPU.  On completion, extract timestamp differences between `Source built`,
`Reference built`, `Exposure computed`, `Adapter configured`, `Training loop
started`, and every `Training progress` line.  This separates source build,
planner solve, adapter initialization, first-step latency, and steady-state
training rate.

## Measured Solver Timing And Second Blocker

Both smoke runs completed runtime construction and planner computation.  The
timestamps below have one-second resolution; total wall times were captured by
the shell with millisecond resolution.

| Stage | OT | KDE |
| --- | ---: | ---: |
| Source catalog open and runtime build | about 29 s | about 29 s |
| `rossmann_1W` reference build | less than 1 s | less than 1 s |
| Planner (`Reference built` to `Exposure computed`) | about 21 s | about 11 s |
| Adapter/core initialization after planner | about 12 s | about 7 s |
| Completed optimizer steps | 0 / 10 | 0 / 10 |
| Total wall time before failure | 66.401 s | 51.244 s |

The runs did not exhibit a hang: source build emitted continuous progress,
both planner calls returned, and the failure occurred immediately after
`Training loop started`.  Neither run reached `Training progress`, so no
model-step throughput can yet be reported.

The common failure is:

```text
ValueError: target_trajectory_probability must be a finite probability in (0, 1]
```

It is raised while `ScheduleCore.schedule_tickets()` writes the first
`slot_scheduled` audit event.  The planners' final sampling distributions are
not degenerate: all 303268 source trajectory probabilities are positive.

| Planner | Minimum final sampling probability | Zero final sampling probabilities |
| --- | ---: | ---: |
| OT | `1.6487067544157061e-06` | 0 |
| KDE | `1.6487067544119222e-06` | 0 |

The incompatible value is instead the planner's sparse target mass for a
trajectory selected from the positive final mixture.  Sparse OT/KDE target
masses can legitimately be zero, but the audit schema currently validates
`target_trajectory_probability` with `(0, 1]`.  This is a core
planner/audit-contract decision and was deliberately not changed during this
smoke investigation.  Resolving it is required before any OT or KDE model
training step can execute and before real training throughput can be measured.

## Audit Repair

The audit contract was subsequently corrected with the smallest field-level
change. `ts3/spec/audit.py` now has a non-negative finite probability validator
for `[0, 1]`, used only by:

- `target_trajectory_probability`
- `final_trajectory_probability`

The existing strict-positive `(0, 1]` validator remains in place for the
source base probability and `trajectory_selection_probability`. This preserves
the invariant that a trajectory actually recorded as selected cannot have zero
selection probability, while allowing a sparse target/final distribution to
represent zero mass. No planner, sampler, mixture formula, event field, or
audit schema version was changed.

`tests/test_random_core.py` now covers the previously failing case of a
scheduled slot with zero target probability and positive final/selection
probabilities, verifies that the final-probability validator accepts zero, and
verifies that zero selection probability is still rejected.
The focused remote test command passed with 24 tests:

```text
python -m pytest -q tests/test_random_core.py tests/test_ot_schema_and_discovery.py
24 passed
```

## Post-Repair Ten-Step Runs

The two smoke configurations were rerun after the audit repair, directly using
the existing `rossmann_1W` catalog directory. Both runs completed all 10
optimizer steps and wrote a final checkpoint; neither reproduced the audit
exception or showed a stall during the training loop.

| Method | Planner time | Training loop started | Training loop completed | Total wall time | Run output |
| --- | ---: | --- | --- | ---: | --- |
| OT | about 21 s | 23:21:12 | 23:21:17 | 71.003 s | `/root/TS3/runs/fev-ot-smoke-10-20260804-v4` |
| KDE | about 11 s | 23:22:36 | 23:22:41 | 53.441 s | `/root/TS3/runs/fev-kde-smoke-10-20260804-v2` |

The per-step progress timestamps place all ten optimizer updates within about
one to two seconds after loop start. The additional few seconds to the logged
loop-completed time include audit finalization and final checkpoint writing.
This smoke run is sufficient to establish that the prior blocker was the audit
validator, not the model update path. Its single-reference planner timing
should still not be treated as the complete 17-valid-reference FEV timing.

## Single-Task Public Evaluation Smoke

The public FEV evaluation entry now supports selecting actual declared task
indices through `bench_config.task_indices`. The smoke configuration
`configs/eval/fev/smoke-rossmann-1w.yaml` selects only task `13`, the
`rossmann_1W` / `Sales` task used by both smoke references. It retains
`expected_tasks: 1`; the evaluator rejects a count that does not equal the
explicit index list length.

OT and KDE training configurations now manually declare their reference task
set as `sampling.reference_task_record`. The declaration is copied into new run and
checkpoint manifests. The already-completed smoke checkpoints were supplemented
with the equivalent checkpoint metadata:

```json
{"reference_task_record": {"benchmark": "fev", "task_indices": [13]}}
```

The public command was run separately for each checkpoint:

```text
python -m evaluate.run --checkpoint runs/fev-ot-smoke-10-20260804-v4/checkpoints/checkpoint-final --config configs/eval/fev/smoke-rossmann-1w.yaml --output-dir runs/fev-ot-smoke-10-20260804-v4/evaluation/rossmann-1w-smoke-task013
python -m evaluate.run --checkpoint runs/fev-kde-smoke-10-20260804-v2/checkpoints/checkpoint-final --config configs/eval/fev/smoke-rossmann-1w.yaml --output-dir runs/fev-kde-smoke-10-20260804-v2/evaluation/rossmann-1w-smoke-task013
```

Both runs completed public config parsing, checkpoint validation, explicit
task-13 selection, and the reference-task check (`checkpoint FEV tasks=[13];
evaluation tasks=[13]`). Each spawned its isolated FEV child process. Neither
reached checkpoint model loading, OT/KDE inference, or metric computation.

The shared blocker is FEV dataset resolution: the child attempts a HEAD request
to `https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/README.md`
and receives `[Errno 101] Network is unreachable`, followed by the Hugging Face
client's exponential retry sequence. OT was stopped after repeated README and
dataset-script retries; KDE was stopped after reproducing the first identical
failure. This is a data/network precondition shared by both checkpoints, not a
solver, model-loading, or inference stall.

## Mirror Retry Result

The interrupted public evaluations were resumed with AutoDL's documented
network setup in the tmux command environment only:

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

Both completed through the same `python -m evaluate.run --resume` public entry
using task selector `[13]`. The OT run started at `00:10:06` and wrote its final
result at `00:10:33` (about 27 s); KDE started at `00:10:58` and completed at
`00:11:22` (about 24 s). Both wrote `tasks/task_013.json`, `fev_results.csv`,
and metadata with `num_tasks: 1` and `task_indices: [13]`. No download, model
load, inference, or result-writing stall remained once the mirror was enabled.
