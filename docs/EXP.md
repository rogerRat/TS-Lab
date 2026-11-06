# Experiments

Last modified: 2026-11-07

## Current experiment

### rage-smoke-static-cycle

- First started: 2026-11-07
- Last updated: 2026-11-07
- Goal: verify a minimal prepared-assets to static-selection to training and evaluation cycle for `rr` and `topk`.
- Data: three deterministic synthetic source series, each with 20 float64 values. The smoke test uses context length 4, prediction length 4, and a 25% selection ratio.
- Preparation: write a source ShardDatasetBundle through `BundleLoadWriter`, then derive eligible length-8 windows and stable source IDs.
- Methods: `rr` cycles over series IDs in first-seen order and greedily chooses each series group’s highest remaining window score. `topk` ranks a supplied finite scalar score per window in descending order, with input-order ties.
- Training: CPU `torch.nn.Linear(4, 4)` fixture, one epoch, batch size 1, learning rate 0.001, seed 7. The fixture exercises the existing `run_arm_training` training and evaluation path without downloading a pretrained checkpoint.
- Evaluation: two held-out smoke windows, entity labels `a` and `b`, context length 4, unit test scales, MASE output.
- Completion criteria: preparation bundle validates and can be read; both selectors return the expected budget; both training runs complete and produce finite MASE; selected artifacts and logs stay under the standard project paths.
- Scripts and tests: `scripts/select_static.py`, `src/rage/static_selection.py`, `tests/test_static_selection.py`, `tests/test_minimal_cycle.py`.

| Batch | Work | Dependency | Status |
|---|---|---|---|
| 1 | Install and verify all six Conda environments | Conda and package indexes | Passed; all six pass `pip check`, imports, and CUDA smoke |
| 2 | Verify the preparation-selection-training-evaluation test | Batch 1 | Passed in integrated test |
| 3 | Run the full test suite and fix migration regressions | Batch 1 | Passed: 784 passed, 3 skipped |
| 4 | Verify the one-command installer across all profiles | Batch 1 | Passed with `bash _install.sh --verify-only` |

## History

- 2026-11-07: Initial CPU smoke run used 12 prepared NPZ windows and exercised `rr`, `topk`, and `run_arm_training`. `rr` selected 3 rows and completed 3 optimizer steps with MASE 4.5501. `topk` selected 3 rows and completed 3 optimizer steps with MASE 4.2771. The integrated bundle-to-selector-to-training test was subsequently added and passed (see Batch 2).
