# Operation Log

Last modified: 2026-11-07

## 2026-11-07 Project refactor and smoke cycle

- Confirmed remote checkout `/root/TS-Lab` is on branch `rage`. Preserved pre-existing deleted tracked files and the untracked data survey.
- Renamed the active project package and configuration paths to `rage`; migrated the shared specification package to `tsl` and changed checkpoint manifest naming to `rage_checkpoint_manifest.json`.
- Added static selection methods `rr` and `topk`, CLI entry `scripts/select_static.py`, output path category, architecture notes, and focused tests.
- Changed one-command installer environment names and install root to `rage` / `RAGE_INSTALL_ROOT`. Added per-environment completion markers.
- Linked project `data`, `outputs`, `logs`, `cache`, and `archive` to `/root/autodl-tmp/rage` after confirming the project paths did not exist and target storage was empty except for the installer log/cache.
- Completed `bash _install.sh` in tmux session `rage-install`; all six configured Conda environments were installed.
- Static selection CLI smoke: `rr` and `topk` each selected 3 of 12 windows (ratio 0.25). Outputs are under `outputs/selection/`; logs are under `logs/selection/`.
- Preliminary CPU training/evaluation smoke used a tiny linear fixture and the existing `run_arm_training` function. Both methods completed 3 optimizer steps and returned finite MASE. This did not exercise the ShardDatasetBundle writer; the integrated test is pending.
- A bundle-preparation attempt in the base Conda environment stopped before writing because base lacked pandas. No partial bundle was created. The installed `rage` environment will be used for the integrated test.
- Integrated test `tests/test_minimal_cycle.py` passed: bundle writing, `rr` and `topk` selection, training, and finite MASE evaluation all completed.
- Focused regression after updating outdated expectations passed: 119 passed, 1 skipped.
- Re-ran `bash _install.sh --verify-only`: all six environments passed `pip check`, required imports, and a CUDA matrix operation on the RTX 4080.
- Repaired the Docker environment name and base dependency setup to use the current `rage` profile and source tree.
- Full suite passed: 784 passed, 3 skipped in 150.28 seconds. The log and `.complete` marker are under `logs/tests/`.

## 2026-11-07 Final verification

- All-profile installer verification passed for six environments on RTX 4080.
- Focused static-selection and integrated workflow checks passed: 119 passed, 1 skipped.
- Full test suite passed: 784 passed, 3 skipped.
- `git diff --check` passed; active project/config/script/test paths contain no old project or specification identifiers.
- Confirmed data, outputs, logs, cache, and archive resolve through the project paths to `/root/autodl-tmp/rage/`.
