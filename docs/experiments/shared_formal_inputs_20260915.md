# Shared-ten formal native inputs (2026-09-15)

The user authorized migration of the ten selected tasks to the formal experiment.
The development manifests supply task declarations only. Development validation
windows and their scores are not reused as formal exams.

Implementation: `src/rags/experiments/shared_formal_inputs.py`, function
`prepare_shared_formal_inputs(shared_decisions, out, evaluation_mode="rolling")`.
The caller loads `configs/rags/shared_tasks_10.json` as `shared_decisions`.

## Protocol

- C=512, H=64. TIME holds out its pinned native `test_length`; Quito holds out
  timestamps beginning at `2023-07-28 00:00:00+00:00`.
- References preserve all pre-exam observations. Nonfinite values split the
  history into contiguous finite fragments; fragments shorter than 576 points
  have no legal gradient windows and are omitted. No probe minimum of 1600,
  trajectory cap, reference-tail crop, or validation gap is applied.
- Formal rolling forecast origins begin at the native test boundary and advance
  by 64. Each exam contains 512 true-history context points and 64 held-out label
  points. The final remainder shorter than 64 is not a legal complete horizon.
  Native scalar/channel identities and selected variates remain unchanged.
- Exam windows containing nonfinite observations are counted and excluded;
  their presence does not cause otherwise valid reference history to be dropped.
- Seasonal scale is computed from all available true history strictly before
  each forecast origin, using native task seasonality. Finite seasonal
  differences are averaged; zero scales remain zero and must be handled by the
  evaluator's separate constant-series MAE reporting. There is no epsilon floor.
- Raw observations are stored as float32; scales use float64. These are input
  data storage types, not the model/gradient precision policy, which remains
  BF16. No model was loaded and no forecast performance was measured here.

## Storage and reuse

Output directory:
`/root/autodl-tmp/TS3/outputs/main-shared-ten-20260915/inputs`.

Per task, `dataset__reference.npz` stores packed values, offsets, fragment IDs and
native entities in the existing `development_inputs.load_fragments` format.
`dataset__exam.npz` stores `windows`, `entities`, and `scales` for existing formal
workers. `dataset__manifest.json` records the exact task, revision, native
boundary/cutoff, per-series reference bounds and exam origins, and counts.
The per-task manifest is the completion marker. NPZ and JSON files are written
through sibling temporary files followed by atomic replacement. No hashes are
calculated. A completed export is reused only when explicit task, revision,
C/H, mode, and stride identities match and both NPZ files exist.

`shared_formal_inputs.json` records the ordered ten-task protocol
`shared-ten-native-rolling-h64-v1`. `native_input_audit.json` records read-only
verification against native observations. Preserve these formal inputs and
audit results; do not delete them as disposable caches. The independent common
source pool is owned by the formal orchestrator and was not generated or
changed by this input export.

## Generated inputs and validation

All ten tasks were exported CPU-only in approximately 16 seconds. Compressed
reference/exam/manifests occupy approximately 65 MiB (before the source pool).

| Task | Native channels | Reference fragments | Exam windows |
|---|---:|---:|---:|
| time_nem_demand_5t | 5 | 5 | 315 |
| time_azure_i_cpu_5t | 492 | 492 | 6396 |
| time_darwin_water_15t | 42 | 42 | 924 |
| time_finland_15t | 1 | 1 | 46 |
| time_eweld_15t | 10 | 10 | 210 |
| quito_min | 3865 | 3865 | 197115 |
| time_cphl_h | 4 | 6 | 32 |
| time_nem_price_5t | 5 | 5 | 315 |
| time_solar_h | 3 | 1 | 39 |
| time_port_activity_d | 198 | 198 | 990 |

CPHL has nonfinite history, hence six finite reference fragments from four
native channels, and eight invalid exam windows. Solar has only one valid
reference/exam channel; 78 candidate exam windows are nonfinite. These are
native data validity outcomes, not performance-based task selection.
Quito has 51,459 zero-scale exam windows, retained for separate constant-history
handling. No task has a nonfinite stored scale.

Audit checks passed for all reference values and boundaries, all exam origins
and identities, first/last exam values per native channel, and full exam-array
finiteness. Seven model-free unit tests cover causal scales beyond context
length, finite segmentation, invalid exam isolation, constants, short channels,
atomic export and identity rejection, and external data paths. The combined
CRB/input test run passed 28 tests.

Necessary native-reader dependency ports: exact `evaluate/quito_data.py` from
the active probe worktree, and its existing external-disk path resolver in
`evaluate/time.py`. Pinned TIME source commit and native data revision/license
validation remain active.
