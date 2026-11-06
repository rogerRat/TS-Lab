# Task Orchestration

Last modified: 2026-11-07

## Current resources and tasks

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| One-command environment installation | CPU, network, data disk; no GPU | None | tmux `rage-install`, running `bash _install.sh` from the project root | All six profiles passed installation verification |
| Static selection and integrated cycle checks | CPU | Main `rage` environment | Focused regression command; 119 passed, 1 skipped | Passed |
| Integrated preparation and training test | CPU, small RAM | Main `rage` environment installed | `pytest -q tests/test_minimal_cycle.py` | Passed in focused test run |
| Full regression suite | CPU, GPU available for optional cases | Main `rage` environment installed | `logs/tests/full-suite-20261107.log` | Passed: 784 passed, 3 skipped |

The install log is written below `/root/autodl-tmp/rage/logs/install/`. Project data, outputs, logs, and caches use project symlinks to `/root/autodl-tmp/rage/`. No training task is queued while environment installation is using the network and disk.

## Scheduling

1. Installation and all-profile verification are complete.
2. Focused selector, asset preparation, and training/evaluation checks passed.
3. The full test suite passed (784 passed, 3 skipped); completion marker: `logs/tests/full-suite-20261107.complete`.

Last modified: 2026-11-07
