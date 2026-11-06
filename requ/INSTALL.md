# One-command environment installation

Run `bash _install.sh` from the project root to install and verify all six environments: `rage`, `rage-timesfm`, `rage-timemoe`, `rage-sundial`, `rage-timerxl`, and `time`.

Requirements: Linux, Conda, Git, package-index access, and GitHub access. On AutoDL, GitHub access enables `/etc/network_turbo` only in the Git subprocess; package downloads continue to use the configured package index. If a proxy is already set in the shell and domestic package downloads are slow, unset the proxy variables before installing.

```bash
bash _install.sh
bash _install.sh --env timemoe
bash _install.sh --verify-only
bash _install.sh --env main --verify-only
```

The wrappers in `scripts/_install_<profile>.sh` install one profile. Valid profiles are `main`, `timesfm`, `timemoe`, `sundial`, `timerxl`, and `time`. The installer can be rerun after interruption; it reuses existing environments and does not remove or migrate them.

Dependency pins live under `requ/`. `requirements.txt` is the main environment. Each `<profile>-requirements.txt` file is for an isolated model environment. `native-selection.txt` contains shared selection dependencies. Python 3.10 and PyTorch 2.7.1 with CUDA 12.8 are used across profiles. The TIME profile keeps its older NumPy, datasets, and PyArrow versions.

Model environments keep the required Transformers versions: 4.57.1 for the main environment, 4.40.1 for Time-MoE, Sundial, and Timer-XL, and 5.3.0 for TimesFM. The lock files pin third-party source revisions and model integrations.

New environments, source checkouts, package caches, temporary extraction files, logs, and package snapshots are stored under `/root/autodl-tmp/rage` by default. Set `RAGE_INSTALL_ROOT` to choose another data-disk location. The installer checks for at least 20 GiB of free space. Set `PYPI_INDEX_URL` or `PYTORCH_INDEX_URL` to use different package indexes.

Verification runs `pip check`, imports required modules, and performs a small CUDA operation when a GPU is available. Without a GPU, the installer reports that GPU verification was skipped. It does not download datasets or model checkpoints and does not start training. Set `CHECKPOINT` to add an optional TIME checkpoint-load check.

Activate the main environment with `conda activate rage`; model profiles use the corresponding `rage-<profile>` environment.
