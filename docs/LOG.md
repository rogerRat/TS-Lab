# Operation Log

Last modified: 2026-11-07

## 2026-11-07 v0-grad-mech-test second progress check

- Two trajectory summaries completed; each had 1,394 windows and 635,950 positive-overlap pairs. The third trajectory is processing.
- Verified the first trajectory summary: lag-1 count 1,393; cosine mean 0.677, range -0.315 to 0.9996. This is an early single-trajectory observation only, not an experiment conclusion. Lag-575 count is 819, matching the expected W-lag count.
- At 3,044 / 1,144,885 windows, throughput was 16-18 windows/s, estimated remaining runtime about 22 hours. Latest GPU sample: 22,040 MiB allocated, 41% utilization, 102 W; data disk 502 GB free. Job remains active; no OOM and cache remains FP32.

## 2026-11-07 v0-grad-mech-test first progress check

- End-to-end job is active in tmux `v0-grad-run`. The 100-step warm-up completed in 8.1 seconds and saved `outputs/train/v0-grad-mech-test/chronos-tiny-warmup-100`.
- The probe selected 1,000 eligible Chronos m5 trajectories in source order, totaling 1,144,885 stride-1 windows. First trajectory has 1,394 windows; FP32 576-gradient ring allocated at about 18.01 GiB.
- Last resource check: RTX 4080 used 22,040 MiB, 52% GPU utilization, 126 W; data disk retained 502 GB free. Progress was 656 / 1,144,885 windows at about 20 windows/s, estimated duration about 16 hours. No OOM; BF16 fallback not used.
- Console and configured log are under `logs/v0-grad-mech-test/`. Continue monitoring; keep the session alive and review subsequent trajectory summaries/errors.

## 2026-11-07 v0-grad-mech-test execution started

- Pinned ChronosTiny checkpoint downloaded and validated at the project cache path.
- Fetched the Chronos `m5` config (78,063,286-byte Parquet) and loaded all 30,490 trajectories into `data/src/shard_dataset/v0-grad-mech-test/m5/load/load`; no random trajectory sampling was used.
- Implemented `scripts/exp/v0_grad_mech_test.py`: select first 1,000 eligible trajectories in source order, create four deterministic warm-up windows per trajectory, run 100 AdamW steps at batch 32 / LR 1e-6, save the post-warm-up checkpoint, then compare all positive-overlap full-gradient pairs with GPU matrix operations and a 576-slot ring. Pair metrics aggregate by trajectory/lag; BF16 is the VRAM fallback.
- Started the end-to-end warm-up and probe in tmux `v0-grad-run`. Monitor the console/log and GPU; resume from completed trajectory summaries after interruption.

## 2026-11-07 v0-grad-mech-test data acquisition update

- The pinned ChronosTiny checkpoint download completed and registry validation passed.
- Attempted the planned Boomlet FEV fetch, but the project FEV adapter requires the complete 20-task local declaration even when selecting one task; the configured benchmark files are absent. No benchmark data was written.
- Re-evaluated source cardinality: the selected Boomlet task cannot support the required 1,000 trajectories. Switched to the existing Chronos `m5` configuration, whose Hub metadata reports 30,490 rows and 78,063,286 Parquet bytes. Selected the first 1,000 eligible rows in source order, without random sampling; pinned source metadata revision `eeecad0b82a8c237e212ce6f8d1abecb513e2cec`.
- Added `configs/rage/v0-grad-mech-test.json`; started raw m5 fetch to `data/src/raw/chronos/m5` in tmux `v0-grad-m5-fetch`. Next stage loads the raw rows into the dedicated bundle and freezes the exact 1,000 trajectory IDs.

## 2026-11-07 v0-grad-mech-test launch

- Re-read the active experiment, orchestration, and operation documents; checked current resource and process state. GPU was idle, data disk had 503 GB free, and the configured development input/source/reference bundle paths were absent.
- Started the pinned ChronosTiny checkpoint download in tmux session `v0-grad-model`; live console output is captured at `logs/v0-grad-mech-test/model-download.log`.
- Next: recover the configured `boomlet_1676` FEV bundle and source bundle through project fetch/build procedures, freeze the deterministic first 1,000 eligible trajectories, then implement and queue warm-up/probe.

## 2026-11-07 v0-grad-mech-test plan review

- Reviewed experiment/orchestration documents, Chronos model registry, development input configs, data/output paths, and current host resources.
- Selected `boomlet_1676` from `configs/rage/development_inputs_0914.json` as the single data configuration and specified stable first-1,000 eligible trajectory IDs; its configured input directory is absent and only the 954-byte smoke fixture currently exists under project data.
- Verified the official Hugging Face model API for `amazon/chronos-t5-tiny`: immutable revision `29d808298f1a62493e7b9a5e08529d0d930fa189`, T5 architecture, 8,394,496 parameters. Added it to the Chronos registry and registered `chronos-tiny` in the supported student-model map with its pinned checkpoint path and exact parameter count. The checkpoint was absent at preflight; its pinned download has now been started in tmux `v0-grad-model`.
- Current resource snapshot: 503 GB free on `/root/autodl-tmp`, 351 GiB host RAM available, RTX 4080 with 32 GiB VRAM, no active GPU process. Estimated 576-gradient ring is 18.02 GiB in FP32 or 9.01 GiB in BF16. The plan uses GPU-batched all-pairs comparisons for lags 1-575, bounded cache storage, online metric aggregation to avoid a pair-output explosion, and BF16 as the VRAM fallback with FP32 metric accumulation.
- Updated `docs/EXP.md`, `docs/ORCHE.md`, and the registry. No experiment or training job was started during planning.


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

## 2026-11-07 Experiment document translated

- Rewrote the prose and table headings in docs/EXP.md in Chinese, preserving experiment identifiers, paths, parameters, batch states, and historical details.
- Kept the active experiment status and all recorded measurements unchanged.

### 2026-11-07 07:23 UTC - v0-grad-mech-test status check

- The `v0-grad-run` tmux session is active. The gradient probe has completed 244/1000 trajectories and 287,236/1,144,885 windows (about 25.0%); the next trajectory is in progress. The progress file reports 15,266 seconds elapsed and FP32 mode.
- Preliminary aggregation over the 244 completed trajectory files: mean cosine similarity is 0.7483 at lag 1, 0.4554 at lag 8, 0.2380 at lag 32, 0.0901 at lag 64, and 0.0133 at lag 575. Mean relative norm difference at lag 1 is 0.1316 and at lag 575 is 0.3983. These are interim results only.
- GPU was active at 70% utilization, 22,040/32,760 MiB memory and 142.76 W. Data disk had 502 GiB free. No OOM or precision fallback observed.
- ETA from the active window progress was approximately 12-13 hours, subject to trajectory-length variation.

### 2026-11-07 07:26 UTC - interim lag-statistics plot

- Added one-off analysis entry point scripts/temp/v0_grad_mech_snapshot_plot.py. It reads only the first N completed trajectory summaries identified by the probe progress file, pools pair-level sufficient statistics at each lag, and writes CSV, JSON, PNG, and PDF under the standard output path.
- Snapshot captured 249/1000 completed trajectories. Population variance is computed as pooled E[x^2] - E[x]^2; the mean and variance are pair-weighted across trajectories. This measures pair-level spread, not between-trajectory uncertainty.
- Artifacts: outputs/train/v0-grad-mech-test/analysis/snapshot_20261106_233011_utc/. The chart shows the mean and +/-1 SD band plus a separate variance panel for cosine similarity and relative norm difference, over lags 1-575.
- Selected interim values: lag 1 cosine mean/SD 0.7485/0.2555 and relative norm difference mean/SD 0.1317/0.1380; lag 575 cosine mean/SD 0.0137/0.1867 and relative norm difference mean/SD 0.3994/0.2571.
- The analysis script initially had malformed string indexing in its final console summary due to shell quote handling. Corrected the summary block and reran successfully; all four artifacts exist. git diff --check passed.
- Probe remained active at 249/1000. GPU utilization was 35% at inspection, memory 22,040/32,760 MiB, power 123.40 W; data disk had 502 GiB free.

## 2026-11-07 v0-crb-reconst-test planning update

- Inspected the CRB implementation and confirmed the coverage, risk-driven error, and split-detection budgets; configured the task as a pending CPU replay after the active gradient probe.
- Added the first 100 fixed trajectories, full lag-1 gradient retention, equal-actual-budget uniform anchors, per-window L2/cosine metrics, staged budget search, and future extreme-trajectory analysis to docs/EXP.md and docs/ORCHE.md.
- The active probe log reports 124,378 windows for the first 100 trajectories. FP32 storage is estimated at about 4.18 TB versus about 502 GB free on the data disk; the batch is not scheduled until storage capacity is resolved.

## 2026-11-07 v0-crb-reconst-test scope and storage update

- Reduced the planned CRB reconstruction sample from 100 to the first 10 trajectories in the fixed v0-grad-mech-test selection.
- The active probe log gives 13,915 legal windows for those ten trajectories. FP32 full-gradient data is estimated at 467.24 GB (435.15 GiB); current data-disk availability is about 538.62 GB (501.6 GiB), leaving about 71.38 GB before metadata and filesystem overhead.
- The reduced scope fits nominally but uses about 86.7% of available disk space. Planned execution remains pending the active probe and a fresh capacity check; use per-trajectory writes and read-only memory mapping, with no duplicate full copy or whole-matrix RAM load.

## 2026-11-07 v0-crb-reconst-test precision update

- Set the full-gradient vector representation and persistent storage for v0-crb-reconst-test to BF16; do not create a full FP32 gradient copy.
- For 13,915 windows and 8,394,496 parameters, BF16 storage is estimated at 233.62 GB (217.57 GiB). Current data-disk availability is about 538.61 GB (501.6 GiB), leaving about 304.99 GB before metadata and filesystem overhead.
- Updated docs/EXP.md and docs/ORCHE.md. The batch remains pending completion of v0-grad-mech-test; no CRB computation has started.

### 2026-11-07 05:00 UTC - visualization path and Plotly update

- Removed the prior PNG and PDF files from the deprecated analysis path. Preserved the earlier CSV and JSON summaries under outputs/train/v0-grad-mech-test/vis/ with vis_ filename prefixes, then removed the analysis directory.
- Replaced the old one-off script with scripts/temp/vis_v0_grad_mech_snapshot.py. It pools sufficient statistics from exactly the completed trajectory ordinals and creates lag-level means, population variances, standard deviations, pair counts, and overlap ratios.
- The Plotly figure uses four aligned panels: mean with +/-1 SD band and population variance for cosine similarity and relative norm difference. The PNG export backend is configurable with --backend or PLOTLY_IMAGE_BACKEND; Kaleido is the default, with auto and Orca options. HTML and SVG exports are also configured.
- Plotly 6.8.0 and Kaleido 1.3.0 are installed. Kaleido requires a Chrome binary; no browser binary was present. Chrome for Testing is downloading into the data-disk cache at /root/autodl-tmp/rage/cache/chrome.
- At this check, the probe remained active at 595/1000 trajectories, in FP32, with about 59 percent of windows processed. GPU memory was 22,040/32,760 MiB; utilization fluctuated between samples. Data disk had 502 GiB free.
- The new script passed Python bytecode compilation and git diff --check. PNG/SVG generation is pending browser download completion.

## 2026-11-07 v0-crb-reconst-test standard protocol and queue

- Fixed the standard budget groups at q=1..8, with K=ceil(N/q) probes per trajectory. Each group runs baseline-uniform and 16 CRB ratio attempts, including forced 1:0:0 plus 15 seeded random settings per round.
- Implemented BF16 full-gradient retention for the first ten fixed trajectories, resumable per-trajectory files, exact per-window L2/cosine CPU replay, matched-budget uniform baselines, Pareto-guided continuing search, and STOP/data-disk reserve handling.
- Added the queue launcher to wait for v0-grad-mech-test/.complete and then launch/restart the CRB experiment in tmux. Full BF16 gradients and completed replay results are retained until user stop or the 1 GiB disk safety reserve.

### 2026-11-07 — 修正 v0-crb-reconst-test 总预算定义

按用户澄清，将总预算比固定为 q∈{1,4,16,64,128,256,512,576}，每条轨迹允许 ceil(N/q) 次梯度计算，首尾合法起点必须计算，预算不足两个时按两个执行。均匀基准在全部合法起点范围内均匀布置指定数量锚点；CRB 在相同总次数约束下由内部规则分配探测位置。q 不作为随机跳过概率。已同步配置、实验脚本、EXP 与 ORCHE。

### 2026-11-07 — 锁定预算比标准并验证排队状态

总预算标准修正为 q∈{1,4,16,64,128,256,512,576}，定义为每条轨迹合法起点数/梯度计算次数。统一锚点总数为 ceil(N/q)，并强制取首尾；当预算少于两个时执行两个端点。CRB 的点位由其内部调度规则决定，估计值只用最近左右相邻锚点做线性插值。每个 q 组运行 baseline-uniform，并尝试含固定 1:0:0 的 16 组比例。更新文件：docs/EXP.md、docs/ORCHE.md、configs/rage/v0-crb-reconst-test.json、scripts/exp/v0_crb_reconst_test.py。

验证：Python 3.12 脚本编译、JSON 配置解析、队列 shell 语法和 git diff --check 均通过。tmux 中 v0-crb-queue 正在等待 v0-grad-mech-test/.complete；前置实验仍运行（606/1000 条轨迹，691638/1144885 个窗口）；新实验尚未启动，STOP 和完成标记均不存在。数据盘可用约 502 GB。

### 2026-11-07 05:29 UTC - Plotly backend installation and visualization refresh

- Confirmed the Chrome download process had inherited the proxy enabled by source /etc/network_turbo. A route comparison against the same Google Chrome archive measured 1.33 MiB transferred in 15 seconds through the proxy versus 8 MiB in 8.09 seconds directly. Restarted the download without proxy variables; the complete 172,287,286-byte browser archive and extracted Chrome are stored under /root/autodl-tmp/rage/cache/chrome.
- Chrome initially lacked libgbm.so.1 and libasound.so.2. Installed libgbm1 and libasound2 from the domestic Ubuntu mirror; their small dependencies included libwayland-server0 and libasound2-data. Chrome then passed a headless startup check.
- Added scripts/lib/plotly_backend.sh and integrated it into the shared environment installer. It benchmarks proxy and direct routes, downloads Chrome to the data disk, checks Chrome shared libraries, maps common missing libraries to apt packages and installs them on apt-based systems, writes a Conda activation hook for BROWSER_PATH, and runs a Plotly PNG smoke check.
- Added Kaleido 1.3.0 to the TimesFM, Time-MoE, Sundial, and TimerXL requirement profiles that already use Plotly. Main already included Kaleido.
- Verified with bash _install.sh --env main --verify-only: pip check passed, CUDA smoke passed on the RTX 4080, and Plotly produced a valid 23,159-byte PNG in memory. Confirmed the Conda activation hook resolves BROWSER_PATH. Shell syntax checks and git diff --check passed.
- Deleted the old PNG/PDF plots and the deprecated analysis directory. Preserved prior CSV/JSON summaries under outputs/train/v0-grad-mech-test/vis with vis_ prefixes; removed the incomplete n609 snapshot.
- Generated the refreshed 618-trajectory snapshot at outputs/train/v0-grad-mech-test/vis/snapshot_20261107_052627_utc_n618. It contains a 3200x2100 PNG, vector SVG, interactive HTML, CSV, and JSON. The PNG and SVG were verified as valid files.
- At the latest status check, the probe was active at 619/1000 completed trajectories in FP32. GPU utilization was 32%, memory 22,040/32,760 MiB, and data disk free space was 502 GiB.

### 2026-11-07 05:36 UTC - refreshed Plotly snapshot and installer verification

- Updated the snapshot plot script so Plotly auto-selects its static export backend by default; explicit backend selection remains configurable. The final run completed without the deprecated-engine warning.
- The final current snapshot is outputs/train/v0-grad-mech-test/vis/snapshot_20261107_053022_utc_n622. It contains vis_lag_similarity_mean_variance.png (3200x2100), SVG, interactive HTML, CSV, and JSON. Removed the intermediate n618 snapshot and the failed n609 partial snapshot; only the latest generated figure remains.
- Snapshot aggregates 622 completed trajectories. At lag 1, pair-weighted cosine mean/SD is 0.7521/0.2529 and relative norm difference mean/SD is 0.1290/0.1355. At lag 575, these are 0.0117/0.1881 and 0.3997/0.2535.
- Environment verification passed via bash _install.sh --env main --verify-only: dependencies are consistent, CUDA smoke passed, and Plotly exported a valid PNG. The Conda activation hook resolves the Chrome binary from the data-disk cache.
- Latest probe state: 627/1000 trajectories completed, FP32; 720,895/1,144,885 windows (about 63 percent), with global progress ETA around 5 hours 53 minutes. GPU remained active at the last checks with 22,040 MiB memory usage. Data disk had 502 GiB free.
- Shell syntax checks and git diff --check passed.

### 2026-11-07 06:30 UTC - two-scale lag plots

- Reworked the visualization at the user's direction: there are exactly two figures in the current snapshot, one with a linear lag axis and one with a logarithmic lag axis. Each figure has cosine similarity and relative norm difference panels; each panel overlays the pair-weighted mean and population variance on separate left/right axes. There are no separate variance panels.
- Set all Plotly fonts to Times New Roman. The log-axis figure uses logarithmic lag ticks to expand the short-lag region. The mean and variance traces use a color-blind-safe palette and different line styles.
- Current output: outputs/train/v0-grad-mech-test/vis/snapshot_20261107_062858_utc_n680/. The two PNGs are 3000x2100; matching SVG and interactive HTML exports and pair-weighted CSV/JSON are included. Verified both PNG file types, the Times New Roman declaration in SVG, the log axis in HTML, and git diff --check.
- Deleted the prior four-panel n622 snapshot. Earlier CSV/JSON-only snapshots remain under the batch vis directory.
- The live probe is at 682/1000 trajectories and 785,288/1,144,885 windows (about 69 percent), with around 4 hours 43 minutes remaining by the active progress bar. GPU utilization was 44%, memory 22,040/32,760 MiB, and the process remained active.

### 2026-11-07 07:24 UTC - compact 2x2 lag figure

- Replaced the two separate linear/log images with one combined figure containing four panels: linear lag on the left, log lag on the right; cosine similarity on the top row, relative norm difference on the bottom row. Each panel overlays mean and population variance with separate left/right axes.
- Increased mean/variance line widths to 3.8/3.4, text to 24 px (the previous figure used 16 px), and added mirrored axis frames plus an outer frame. Removed the overlapping 500 tick from the log panel; the 575 endpoint remains labeled.
- Plotly's native log axis uses base 10. The new right column uses log-base-12 coordinates with 12 percent upper-range padding to modestly reduce the early-lag expansion. Log tick positions are transformed while displayed tick labels remain actual lags.
- The output uses a Times New Roman-first font stack. This remote Ubuntu host does not have Times New Roman; installed fonts-liberation2 and configured Liberation Serif as the metric-compatible fallback. The environment installer now checks and prepares this fallback.
- Current figure: outputs/train/v0-grad-mech-test/vis/snapshot_20261107_072202_utc_n736/vis_gradient_similarity_linear_log.png, with matching SVG and interactive HTML plus CSV/JSON. PNG is 4400x2900. Visual inspection confirmed the four-panel layout, frame, enlarged type, and non-overlapping log ticks.
- Removed the intermediate n730 and n735 figures. Environment verify-only passed after the font check; git diff --check and shell syntax checks passed.
- Probe remains active at 738/1000 trajectories and 849,581/1,144,885 windows (about 74 percent). GPU utilization was 100%, memory 19,036/32,760 MiB, and estimated remaining time was about 4 hours 27 minutes.

### 2026-11-07 07:31 UTC - lag boundary and outer frame correction

- Corrected the right-column lag transform to stop exactly at lag 575. The axis now spans log-base-12 coordinates from lag 1 through lag 575 with no range padding; its displayed endpoints align with the left-column linear axis, and no x-axis region is shown beyond 575.
- Moved the outer rectangle beyond the four subplot frames with a clear inset gap. Kept individual subplot frames. Checked the PNG visually: the outer line no longer crosses the y-axis or x-axis titles, and the log endpoint labels do not overlap.
- Deleted the intermediate n736 and n741 snapshots. Final current artifacts are under outputs/train/v0-grad-mech-test/vis/snapshot_20261107_072922_utc_n742/.
- Final PNG is 4400x2900. Shell syntax checks and git diff --check passed.
- Probe is active at 744/1000 trajectories and 858,114/1,144,885 windows (about 75 percent); ETA from the active progress bar is about 4 hours 20 minutes. GPU memory was 22,040/32,760 MiB and the GPU process remained active.


## 2026-11-07 Dependency manifests reorganized

- Split model, benchmark, and repository-owned Python requirements into dedicated manifests. Stable top-level Conda profile files now compose those categories.
- Split the mixed source lock into model/Chronos and benchmark/FEV locks. The TIME profile now reuses the model Chronos lock and has a separate benchmark source lock.
- Moved model integration and baseline provenance locks into their category directories. Updated installer, Dockerfile, FEV provenance, model provenance, baseline documentation, and dependency guide paths.
- Preserved Kaleido 1.3.0 in all Plotly-enabled model profiles.
- Verification results are recorded after the profile and regression checks complete.


### 2026-11-07 Dependency verification results

- Pip parsed all six top-level profile manifests and every included project/model/benchmark requirements file. The recursive Plotly detector found Plotly in the five relevant profiles and excluded TIME.
- Updated the TimesFM, Time-MoE, Sundial, and TimerXL environments from the composed manifests; installed the declared Kaleido 1.3.0 dependency where it was absent. Their profile install and verification commands passed.
- `bash _install.sh --verify-only` passed all six profiles: main, TimesFM, Time-MoE, Sundial, TimerXL, and TIME. CUDA smoke was disabled while the experiment GPU session remained active. Plotly PNG export checks passed for all five Plotly profiles.
- Full regression suite passed: 758 passed, 30 skipped, 21 warnings in 168.20 seconds.
- Shell syntax and `git diff --check` passed. Docker build context paths were reviewed; a container image build was not run.

### 2026-11-07 17:59 UTC - probe progress refresh and lag plot snapshot

- Reconnected to the `tsl` host after one SSH connection closed. The gradient probe remained active in tmux `v0-grad-run`.
- Latest progress: 896/1,000 trajectories and 1,031,483/1,144,885 windows (about 90 percent), FP32; the active log estimated about 1.98 hours remaining. GPU usage was 22,046/32,760 MiB.
- Regenerated lag statistics and figures from completed trajectory summaries. The first image export attempt did not load the Conda activation hook, so Kaleido could not find Chrome. Retried with the configured `BROWSER_PATH`; PNG, SVG, HTML, CSV, and JSON exports completed.
- Current snapshot: `outputs/train/v0-grad-mech-test/vis/snapshot_20261107_095920_utc_n896/`. PNG verified as 4400x2900 and SVG identified as valid vector graphics. Removed only the incomplete snapshot from the failed export attempt.
- Snapshot lag-1 pair-weighted cosine mean/SD: 0.754858/0.253708; relative norm difference mean/SD: 0.127870/0.135470. These summarize 896 completed trajectories, not the full experiment.
- Updated `docs/ORCHE.md` with the current progress. The next check should refresh the plot after additional trajectory summaries are complete.

Last modified: 2026-11-07

### 2026-11-07 18:01 UTC - refreshed plot after additional trajectory completions

- The probe advanced to 898/1,000 completed trajectories while the earlier plot was being rendered. Regenerated the snapshot from all 898 completed trajectory summaries.
- Current snapshot: outputs/train/v0-grad-mech-test/vis/snapshot_20261107_100114_utc_n898/. PNG, SVG, HTML, CSV, and JSON exports completed.
- Lag-1 pair-weighted cosine mean/SD: 0.754857/0.253776; relative norm difference mean/SD: 0.127926/0.135541.
- At the status check, the live progress bar showed 1,033,462/1,144,885 windows; the progress file recorded 898/1,000 fully completed trajectories. The active log estimated 1.94 hours remaining.
- Updated docs/ORCHE.md to the latest window and trajectory counts. The next plot refresh should use completed trajectory summaries after further progress.

Last modified: 2026-11-07


### 2026-11-07 18:07 UTC - final requested progress snapshot at 906 trajectories

- Generated the requested current plot using exactly the 906 completed trajectory summaries. No further plot refresh is planned until the user requests one near experiment completion.
- Snapshot: outputs/train/v0-grad-mech-test/vis/snapshot_20261107_100713_utc_n906/ with PNG, SVG, HTML, CSV, and JSON outputs.
- Lag-1 pair-weighted cosine mean/SD: 0.754576/0.253737; relative norm difference mean/SD: 0.128023/0.135499.
- At the check, the live progress bar showed 1,040,814/1,144,885 windows; the progress file recorded 906/1,000 completed trajectories. The latest trajectory ETA was about 1.08 hours.
- Updated docs/ORCHE.md and consolidated a duplicate log entry from the earlier documentation append.

Last modified: 2026-11-07


### 2026-11-07 19:15 UTC - resume probe with CRB gradient retention

- The user authorized a random trajectory subset for CRB and requested twenty if storage risk was acceptable, otherwise ten. Current free disk was about 502 GiB. A seed-0 sample of twenty remaining trajectories was estimated at 407.5 GB, leaving about 122 GiB; this uses most available space. The ten-trajectory sample was estimated at 215.5 GB, leaving about 300 GiB, so ten were selected.
- Selected zero-based input-manifest ordinals: 973, 980, 981, 984, 985, 987, 988, 996, 998, 999. Their complete BF16 gradients are retained under outputs/train/v0-crb-reconst-test/gradients/ and will be reused by CRB.
- Updated the probe to write selected BF16 gradient rows atomically, the CRB loader to accept the fixed sampled ordinals, and both experiment configs. The v0-grad manifest comparison ignores this storage-only setting, preserving the original data selection and warm-up protocol.
- Interrupted the active probe after trajectory 973 completed, then resumed it in tmux v0-grad-run. The first retained cache (trajectory-00) is complete at about 10.2 GB. The other nine are still pending.
- The probe source and CRB source passed Python bytecode compilation. Updated docs/EXP.md and docs/ORCHE.md with the storage estimate, random selection rationale, and reuse plan.

Last modified: 2026-11-07


### 2026-11-07 19:21 UTC - retained gradient cache progress

- Resumed the probe through trajectory 981/1,000. The first two selected trajectory cache files are complete; the third is being written. Cached data is BF16 converted from the FP32 probe gradients.
- Current free data-disk space is about 468 GB. The full ten-trajectory estimate remains 215.5 GB.
- Writing the retained vectors reduced observed throughput from about 20 windows/s to about 10-12 windows/s during larger trajectories. The latest per-trajectory ETA was about 0.65 hours for the remaining run.
- The CRB queue monitor is active again in tmux v0-crb-queue and waits for the predecessor completion marker.

Last modified: 2026-11-07


### 2026-11-07 19:35 UTC - start CRB preparation in parallel

- Per user direction, launched v0-crb-reconst-test before v0-grad-mech-test completion. The queue and CRB runner were updated to permit early CPU preparation while the source probe continues.
- CRB loaded the fixed ten selected series (12,836 legal windows) from the manifest and bundle. It is waiting for all ten retained BF16 completion markers; at launch, 7/10 were complete.
- The CRB process does not initialize a GPU model while waiting. Once all markers exist, it will skip gradient computation and start CPU replay. GPU memory remained about 22 GiB used by the source probe.
- Updated the experiment and orchestration documents. Python bytecode compilation and queue shell syntax checks passed.

Last modified: 2026-11-07


### 2026-11-07 19:51 UTC - start formal per-trajectory CRB batches

- The user requested immediate analysis of available cached gradients and specified ten formal single-trajectory batches, without a separate early-preview directory.
- Removed the incomplete early-7 output and preview config. Formal results are written under outputs/train/v0-crb-reconst-test/batches/trajectory-NN/.
- The queue starts each trajectory batch as soon as its retained cache marker exists. Each batch runs one initial round over all eight budget groups, with uniform baseline and 16 CRB configurations per group, then advances to the next ready cache. No GPU model is loaded by CRB.
- Fixed a CRB dispatch error where the uniform baseline helper was called with the candidate-search arguments. The first batch now creates baseline and candidate metric files; at the latest check it had completed q=1 and was evaluating q=4.
- All 1,000 source trajectories and all ten selected BF16 caches are complete. Data disk has about 301 GiB available. Python compilation and queue shell syntax checks passed.

Last modified: 2026-11-07


### 2026-11-07 20:05 UTC - launch all ten CRB trajectory batches concurrently

- The user clarified that formal CRB work should use ten individual trajectory batches and start analyzing available caches immediately; no separate early-preview output should remain.
- Removed the incomplete early-7 directory/config. The formal queue now writes to batches/trajectory-00 through trajectory-09 and starts each batch when its matching retained-gradient cache is complete.
- Started all ten CRB CPU batches concurrently with six threads each (60 requested threads total on a 96-core host). All ten source caches and the v0-grad-mech-test .complete marker are present; about 301 GiB data-disk space remains. No CRB task allocates GPU memory.
- Fixed a cache-index mapping error found on batches 01-09 during their first start. Each batch now maps its source ordinal to the correct shared cache ordinal; failed attempts produced no complete markers, and their progress restarts cleanly. Batch 00 progress was preserved and resumed.
- Fixed the uniform-baseline helper dispatch, confirmed each batch emits its own progress and evaluation files, and added the one-round-per-trajectory bound. At the latest check, all batches had completed q=1; batch 00 was evaluating q=4. No batch had a completion marker yet.
- Python bytecode compilation and queue shell syntax checks passed. Updated docs/EXP.md and docs/ORCHE.md to record the per-trajectory plan.

Last modified: 2026-11-07

### 2026-11-07 20:38 CST - full gradient snapshot

- Regenerated the lag visualization from all 1,000 completed trajectories in outputs/train/v0-grad-mech-test/vis/snapshot_20261107_123811_utc_n1000/. Verified 575 lags and PNG, SVG, HTML, CSV, and JSON outputs.
- Lag 1 cosine mean/SD: 0.753878/0.253848; lag 575: 0.013485/0.195253 across 581872 pairs.
- Removed six older incomplete snapshots after generating the full snapshot.

Last modified: 2026-11-07


### 2026-11-07 container recovery and bounded CRB replay

- After the user restarted the container, SSH recovered. The container had no active tmux sessions or CRB/probe processes. All ten retained BF16 caches and the source .complete marker were present; GPU was idle, host available RAM was about 405 GiB, and data-disk free space was about 301 GiB.
- Replaced randomized CRB ratios with 15 denominator-4 simplex lattice points plus the centroid (1/3, 1/3, 1/3). Each of the ten trajectories runs all eight budget groups, each with one uniform baseline and 16 CRB configurations, for one finite round.
- Reworked the queue to run at most two trajectory batches concurrently, with two CPU threads per batch. Old incomplete per-trajectory outputs from the random design are superseded; shared full-gradient caches and prior logs are preserved.
- Python bytecode compilation and queue shell syntax validation passed. The bounded queue was started in tmux; session and progress state are verified below.

Last modified: 2026-11-07


### 2026-11-07 20:40 CST - bounded CRB replay started

- Started tmux queue v0-crb-queue. It launched trajectory batches 00 and 01; the queue caps concurrency at two batches, and each process has OMP, MKL, OpenBLAS, and NumExpr thread limits set to two.
- Both active batches loaded their one-trajectory input and the shared BF16 cache without GPU use. Each completed q=1 uniform baseline plus 16 parameter attempts; q=1 is the exact identity case. They are now running the next budget group. No batch completion marker exists yet.
- At verification, host available RAM was about 403 GiB, data-disk free space about 301 GiB, and GPU utilization/memory were zero. Two Python workers used about 1.7 CPU cores each in the sampled process snapshot.

Last modified: 2026-11-07

### 2026-11-07 - gradient experiment completion and archive

- Recorded the full 1,000-trajectory lag results in docs/RESULTS.md and moved the experiment record into the history section of docs/EXP.md.
- Archived experiment-specific outputs, visualization, data, logs, configuration, and scripts under archive/v0-grad-mech-test/, retaining symbolic links at their original paths for the active CRB workflow. Shared inputs and CRB caches remain in place.

Last modified: 2026-11-07


### 2026-11-07 20:42 CST - CRB replay progress check

- tmux confirmed the queue and trajectory batches 00 and 01 are alive. Batch 00 finished q=1 and started the 16 q=4 CRB configurations; batch 01 finished q=1 and was computing the q=4 uniform baseline.
- GPU remained idle at 0% utilization and 0 MiB. The queue remains capped at two concurrent batches and two configured CPU threads per worker. No completion markers are present yet.

Last modified: 2026-11-07

### 2026-11-07 - narrow gradient experiment archive scope

- Restored experiment data, logs, configuration, scripts, checkpoint, and manifest to their original locations. Only probe results and visualization remain archived, with links at their original paths. CRB gradient caches remain at outputs/train/v0-crb-reconst-test/gradients/. This supersedes the prior archive scope.

Last modified: 2026-11-07


### 2026-11-07 20:43 CST - q=4 replay progress

- Both active batches completed the q=4 uniform baseline and started the 16 CRB ratio configurations. Queue reports 0/10 batches fully complete and 2/2 active.
- Host load average was 12.05, available RAM about 388 GiB, and data-disk free space about 301 GiB. GPU remained at 0% utilization and 0 MiB.

Last modified: 2026-11-07

### 2026-11-07 - archive trajectory-level lag results

- Copied the input manifest and generated an index of all 1,000 source rows and series IDs. Exported 575,000 trajectory-by-lag summary rows with cosine and relative norm metrics to archive/v0-grad-mech-test/results/. Verified pair counts and weighted cosine means against pooled summaries at all 575 lags. The first export was interrupted and discarded after a concurrent writer corrupted it; the final export was regenerated and verified. Documented sources, indices, metrics, plots, and the absence of individual pair records in docs/RESULTS.md. Gradient caches remained in place.

Last modified: 2026-11-07


### 2026-11-07 21:02 CST - provisional CRB visualization preview

- Generated three draft figures from completed per-window vector L2 and cosine arrays without interrupting the CRB queue. Snapshot: outputs/train/v0-crb-reconst-test/vis/preview_20261107_130219_utc/.
- Snapshot coverage: 44 completed CRB configurations and 4 uniform baselines across trajectory batches 00-01 and q=1,4. Missing heatmap cells are unfinished configurations, not failed runs.
- Figure 1 is a paired overview heatmap by trajectory/q and baseline/configuration. Figure 2 maps candidate-minus-uniform median errors onto the simplex for trajectory batch 00, q=4. Figure 3 plots per-window L2 and 1-cosine for S00 against the matching uniform baseline, with raw and 21-start rolling-median traces. PNG and SVG versions plus CSV metric/configuration tables and a manifest are saved.
- The preview reads the metric schema as currently saved (vector L2 and cosine similarity); it does not add a separate gradient-magnitude error. The one-time plotting source is scripts/temp/plot_v0_crb_preview.py.
- The active queue was still running after rendering; no replay sessions or results were changed.

Last modified: 2026-11-07


### 2026-11-07 21:34 CST - progress refresh and explicit baseline markers

- Latest queue count: 72/1,280 CRB parameter attempts (5.6%), 5/80 uniform baselines, and 0/10 completed trajectory batches. Batch 00 has completed q=1 and q=4 and is at q=16 (14/16); batch 01 has completed q=1 and is at q=4 (10/16). Remaining eight tracks are queued.
- Refreshed the three-figure snapshot under outputs/train/v0-crb-reconst-test/vis/preview_20261107_133243_utc/ from 71 completed CRB outputs and five baselines. Overview cells now show B/W/= relative to the matched uniform baseline; simplex nodes use B/W/= labels and distinct marker shapes; window profiles use green/red regions and explicit median-delta labels.
- Snapshot includes only completed files. It does not interrupt workers. Current host available RAM was about 367 GiB, data-disk free space about 301 GiB, and GPU remained idle.

Last modified: 2026-11-07

### 2026-11-07 - CRB redesign literature assessment

Last modified: 2026-11-07

- Inspected current CRB evaluation files, the frozen engine and reconstruction code, and all ten selected trajectory lag summaries. Current nontrivial results remain limited to the first two trajectories.
- Created docs/crb-reconstruction-redesign-20261107.md with evidence boundaries, smoothness-risk derivation, nine relevant primary references, four candidate hypotheses, and a staged validation proposal.
- Confirmed the source probe uses model.eval(); rapid gradient variation is not evidence of dropout noise or fundamental unpredictability.
- No experiment configuration, gradient cache, or running queue was changed. Proposed diagnostics and replacement methods have not been executed.


### 2026-11-07 23:01 CST - arrow comparison markers

- Updated the provisional plotting script to replace all B/W labels with down/up arrow text: down means lower error than the matched uniform baseline; up means higher error; equals denotes a tie. Figure 1 retains absolute-error color and adds arrow direction; Figure 2 labels each simplex node with its direction arrow; Figure 3 retains green/red local comparison shading and shows an arrow beside each median delta.
- Corrected preview discovery to read evaluations.jsonl, so completed attempts that reuse a prior metric file still appear as their own simplex parameter points. Current snapshot: outputs/train/v0-crb-reconst-test/vis/preview_20261107_150037_utc/. It contains 207 completed attempts, 158 unique metric files, and 14 baselines at render time.
- Latest queue check after rendering: 208/1,280 CRB attempts (16.25%), 14/80 baselines, and 1/10 track batches complete. Batch 01 is at q=64 (5/16); batch 02 is at q=4 (11/16). GPU idle; data-disk free space about 301 GiB.
- Python compilation passed. The plotting source contains no B/W labels. The replay queue was not modified.

Last modified: 2026-11-07


### 2026-11-07 23:40 CST - CRB completion status check

- CRB is not complete. There are 235/1,280 completed configuration attempts (18.36%%), 16/80 uniform baselines, and 1/10 completed trajectory batches.
- Trajectory batch 00 is complete. Batch 01 is at q=128 (6/16); batch 02 is at q=16 (5/16); batches 03-09 are queued. Two bounded workers and the queue tmux session remain active. The top-level .complete marker is absent.
- GPU is idle and data-disk free space is about 301 GiB.

Last modified: 2026-11-07


### 2026-11-07 23:47 CST - CRB visualization refresh

- Refreshed all three provisional charts from the latest completed per-window metrics. Snapshot: outputs/train/v0-crb-reconst-test/vis/preview_20261107_154520_utc/. The snapshot contains 240 completed CRB attempts, 191 unique metric files, and 16 uniform baselines across the observed trajectory/q pairs.
- Down/up arrow comparison markers remain in the heatmap and simplex map; the window profile retains local green/red comparison shading and the median-delta arrow. The plot script reads evaluations.jsonl so reused configurations remain represented.
- The CRB queue remained active and the plots were read-only. The post-render live count was 242/1,280 CRB attempts (18.91 percent), 16/80 uniform baselines, and 1/10 trajectory batches complete; data-disk free space remained about 301 GiB.

Last modified: 2026-11-07


### 2026-11-08 00:16 CST - q-sweep simplex plots

- Added q-sweep simplex pages to the latest preview at outputs/train/v0-crb-reconst-test/vis/preview_20261107_161124_utc/. Each page holds eight q panels in columns and two error metrics in rows, with one shared color scale per metric row. Every panel compares candidates with a matched-budget uniform baseline.
- The q-sweep panels show only direction arrows inside nodes, with larger circles; unfinished configurations are pale gray. No Sxx labels appear in these q-sweep pages. PNG and SVG pages were generated for each trajectory with available completed attempts.
- At render time there were 263 completed CRB attempts, 214 unique metric files, and 18 uniform baselines. Trajectory 00 had all eight q groups complete; trajectories 01 and 02 were partial. The replay queue continued running during plot generation.

Last modified: 2026-11-08


### 2026-11-08 08:28 CST - aggregate q-sweep simplex visualization

- Reworked the q-sweep simplex view to aggregate across available trajectories. For each q and simplex node, the plotted value is the equal-weight mean of each trajectory's median candidate error minus its matched uniform median error. The rows show vector L2 and cosine error; the eight columns show q.
- Removed node labels and arrow glyphs from this q-sweep figure. Black outer circles with green inner circles mark lower mean error than uniform; red marks higher mean error; gray marks ties; pale gray marks missing results. The number in each node is the contributing trajectory count. Circle areas are enlarged (outer s=300, inner s=225).
- Latest snapshot: outputs/train/v0-crb-reconst-test/vis/preview_20261107_162754_utc/. It contains 280 completed configuration attempts and 18/80 baselines at render time; the q-sweep CSV records 128 q/node means and contributor counts. Data are partial, so counts vary by q/node.
- Queue was left untouched: batch 00 complete; batches 01 and 02 active under the existing two-batch, two-CPU-thread-per-batch limit. The current run was read-only with respect to replay outputs.

Last modified: 2026-11-08


## 2026-11-08 - gradient proxy preparation

Last modified: 2026-11-08

Read source protocols, engine and caches. Prepared proxy experiment and protocol; confirmed rage environment with user. GPU idle, 439 GiB available RAM, 301 GiB free data disk. User authorized stopping CRB if resources become constrained; not needed at preparation. Direction is the primary outcome. No test or execution result is claimed yet.


### 2026-11-08 - gradient proxy execution

Last modified: 2026-11-08

Started v0-gradient-proxy in tmux with the confirmed rage Conda environment. Python compilation passed. Initial validation stopped on a 2.5e-5 relative difference between FP32 softmax-based analytic head derivatives and log-softmax autograd. The validation now uses the same batch shape and a recorded 1e-4 relative tolerance; no mathematical gradient definition changed. Track 00 completed with all three full-BP cache checks and head autograd check passing; track 01 is computing full-coordinate Gram matrices. CPU replay remains active, available RAM about 395 GiB and data-disk free space about 293 GiB. Source files and cached full gradients are unchanged.

### 2026-11-08 - container memory pressure and CRB pause

Last modified: 2026-11-08

- Container cgroup limit: 66,571,993,088 bytes (about 62 GiB). Before the stop, current charge was about 64.97 GB, memory PSI full avg10 about 42%, CPU quota 12 cores, and host CPU count 96. OOM counters remained zero. Host free memory alone was misleading.
- Proxy analysis completed all 10/10 trajectories and wrote its completion marker. Final summary and FP64 Gram spot checks completed; report is docs/results/v0-gradient-proxy-test.md.
- Per the user's authorization, created outputs/train/v0-crb-reconst-test/STOP and sent SIGUSR1 to batch workers 03 and 04. They remained for over 30 minutes inside in-flight candidate computations without progress updates or responding to the safe-stop signal. Sent SIGKILL to those two workers to release container memory. Their in-flight q=4 candidate s12 and q=1 candidate s15 were not committed; previously completed JSONL records, candidate arrays, and full BF16 gradients remain. Their per-batch completion markers are absent; the queue-level completion marker remains absent.
- After stopping the workers, cgroup memory charge fell to about 2.98 GB; OOM counters remained zero. The queue session remains alive but STOP prevents new batches from launching. Resume after reviewing the interrupted candidates and removing STOP under normal memory pressure.


### 2026-11-08 - CRB replay termination, redraw, and archive

Last modified: 2026-11-08

- Followed the user's decision to terminate the CRB replay because results were broadly clear and container resources were constrained. Confirmed no replay worker remained; stopped the idle tmux queue supervisor. The STOP marker remains.
- Replotted committed metrics with the rage environment. Snapshot: outputs/train/v0-crb-reconst-test/vis/preview_20261108_024904_utc/. It contains 429 committed configuration attempts, 326 unique metric files, and 28 uniform baselines.
- Archived all CRB batch directories (5.5 MiB) and the regenerated snapshot (5.2 MiB) under archive/v0-crb-reconst-test/. Removed the three older visualization snapshots from outputs/train/v0-crb-reconst-test/vis/ after confirming the new archive. Batch metric files and the ten full-gradient caches were preserved.
- Batch status at termination: trajectories 00-02 completed all eight q groups; trajectory 03 completed q=1 and 13/16 q=4 configurations; trajectory 04 completed q=1; trajectories 05-09 did not start. The experiment is partial, not complete.
- Updated docs/EXP.md to move CRB into history and identify proxy exploration as the current workstream; updated docs/RESULTS.md, docs/results/v0-crb-reconst-test.md, and docs/ORCHE.md with the final status and batch-level results.
- Remaining research state: the ten BF16 full-gradient caches remain at outputs/train/v0-crb-reconst-test/gradients/. Proxy exploration batch v0-gradient-proxy-test is complete; its next batch protocol is not yet fixed.


### 2026-11-08 - direct proxy cost benchmark

Last modified: 2026-11-08

Prepared a matched batch-size-one timing experiment for full BP, output-head gradients, decoder token-mean embeddings, and forward loss. It samples eight spaced windows on each of ten fixed tracks, warms the four methods, randomizes method order, and repeats each call three times. Results include tokenization, model compute, CPU result transfer, and CUDA sync. Current cgroup charge is about 3 GiB after stopping the two stalled CRB workers. The CRB STOP marker remains set.


### 2026-11-08 - gradient proxy cost benchmark completed

Last modified: 2026-11-08

Measured 240 calls per method on 80 spaced windows, three repeats each, at batch size one. Median times: full BP 42.80 ms, output head 14.21 ms, decoder token-mean embedding 11.35 ms, forward loss 11.28 ms. Paired median speedups: 2.93x, 3.72x, and 3.71x. Timing includes tokenizer preparation, model computation, CPU result transfer and CUDA synchronization. All ten feature-cache spot checks passed. Raw repeats and summary are under outputs/train/v0-gradient-proxy-cost/. The CRB STOP marker remains set; there are no active replay workers.

### 2026-11-08 03:34 UTC - relative lag-L2 supplement prepared

Last modified: 2026-11-08

- User requested a grad-mech supplement limited to the ten retained full-gradient caches. Registered v0-grad-mech-l2-norm in docs/EXP.md and docs/ORCHE.md, with config configs/rage/v0-grad-mech-l2-norm.json and entry scripts/v0_grad_mech_l2_norm.py.
- The existing full-coordinate Gram matrices were derived from the same BF16 caches and passed the prior proxy audit. This batch will verify three direct BF16 pair distances per trajectory, then compute relative pair L2 for every lag 1-575. No model gradients or CRB replay are scheduled.
- Remote resource check: data disk 250 GB free, host 341 GiB available, RTX 4080 idle. The rage Conda environment is available at /root/autodl-tmp/rage/envs/rage. Python syntax, JSON parse, and git diff whitespace checks passed. Execution and results remain pending.


### 2026-11-08 - Cross-Rank experiment protocol and implementation prepared

- Split v0-gradient-proxy-test into Local-Risk and Cross-Rank. Local-Risk records the existing block-32 proxy-dispersion result and its limit: descriptive association, not budgeted interval-selection evidence.
- Fixed Cross-Rank references to tracks 00 and 07 and sources to tracks 01-06, 08, and 09. All reference windows are used; each source track contributes three evenly placed, 576-point nonoverlapping windows, for 24 candidates and 45,048 pairs.
- Added configs/rage/v0-gradient-proxy-cross-rank.json and scripts/cross_rank_gradient_proxies.py. The run reuses all-coordinate BF16 full-gradient caches and saved proxy features; it performs no model backward passes.
- Updated docs/EXP.md, docs/gradient-proxy-protocol.md, and docs/ORCHE.md. Script bytecode compilation and configuration partition checks passed. Current state: implementation prepared; execution not yet reported.

Last modified: 2026-11-08

### 2026-11-08 03:37 UTC - relative lag-L2 supplement completed

Last modified: 2026-11-08

- Ran scripts/v0_grad_mech_l2_norm.py in tmux v0-grad-mech-l2-norm under the rage Conda environment. The session exited normally and outputs/train/v0-grad-mech-l2-norm/.complete is present.
- Read only the ten retained BF16 gradient caches for direct pair checks and their completed full-coordinate Gram matrices for the all-lag analysis. No model gradient was recomputed; the stopped CRB replay was not restarted.
- Validated 12,836 windows, 12,826 lag-1 pairs, 5750 trajectory-lag rows, 575 pooled lag rows, and 30 direct BF16 pair checks. Maximum direct-versus-Gram absolute L2 difference: 0.00001966. No squared distance clipping was needed.
- Lag-1 relative L2: pair-weighted mean 0.637249, median 0.762024, p95 1.241116; equal-trajectory mean 0.646177. Recorded full results and scope limits in docs/results/v0-grad-mech-l2-norm.md, linked docs/RESULTS.md, and moved the completed batch to EXP history. The ten source caches remain intact.

### 2026-11-08 03:38 UTC - entry-point repair and repeat check

Last modified: 2026-11-08

- Moved the reusable supplement entry to scripts/v0_grad_mech_l2_norm.py to satisfy the project entry-point rule. The scripts/ directory contains queue.py, which shadows the Python standard library queue module when PyTorch is imported from a root-level script. Replaced the three-per-track BF16 direct-check decoder with NumPy bit conversion; the all-lag Gram computation is unchanged.
- Reran the formal entry under rage with OPENBLAS_NUM_THREADS=2 and OMP_NUM_THREADS=2. The batch completed again for ten trajectories, 12,836 windows, and 12,826 lag-1 pairs. Lag-1 mean and median are unchanged at 0.6372489570 and 0.7620241576; no squared-distance clipping occurred. The .complete marker was recreated after successful output generation. Source caches remain unchanged.


### 2026-11-08 11:38 CST - Cross-Rank launched

- The first direct-file launch failed before logger setup because scripts/queue.py shadowed the standard-library queue module during PyTorch import. No result files or gradient caches were modified. Changed the launch form to python -m scripts.cross_rank_gradient_proxies.
- Started tmux v0-gradient-proxy-cross-rank with the confirmed rage environment. Ten trajectory manifests, all ten full-gradient cache sizes and completion markers, and proxy feature shapes validated.
- The run selected 24 source windows (three in each of eight source tracks), verified the selected 576-point spans do not overlap, and began reference-source scoring for 1,877 reference queries (45,048 pairs).
- At launch check, GPU usage was 15 percent with about 1.3 GiB allocated; the cross-rank process was active. Result completion is pending.

Last modified: 2026-11-08


### 2026-11-08 11:40 CST - Cross-Rank completed and results recorded

- Completed 1,877 reference queries against 24 pairwise nonoverlapping source candidates (45,048 score rows) in about 1.5 minutes. All ten manifests and full-gradient cache sizes/markers passed validation; all 8,394,496 full-gradient coordinates were used.
- Output-head cosine: pooled pair Pearson/Spearman 0.786/0.795; per-reference mean Spearman/Kendall 0.640/0.481; top-5 overlap 0.487 (random five-of-24 expectation 0.208); cosine MAE/RMSE/bias 0.071/0.091/-0.016.
- Embedding cosine ranking was near zero or negative; negative absolute loss difference was weak. Cross-Rank classification is Other: a qualified positive output-head ranking signal on this fixed pool, not evidence that full gradients can be replaced for general candidate selection.
- Local-Risk remains Other: positive block-dispersion association, but no budget-matched interval selection policy was evaluated.
- Updated docs/EXP.md, docs/RESULTS.md, docs/results/v0-gradient-proxy-test.md, docs/gradient-proxy-protocol.md, and docs/ORCHE.md with the two part codes, exact questions, result categories, quantitative outcomes, and limits. GPU idle after completion; CRB remains stopped.

Last modified: 2026-11-08

### 2026-11-08 03:48 UTC - relative lag-L2 conclusion and artifact archive

Last modified: 2026-11-08

- Recomputed lag-1 cosine from the same ten full-gradient Gram matrices used for the relative L2 analysis: pair-weighted mean 0.721466 across 12,826 pairs and equal-trajectory mean 0.717265. The original 1,000-trajectory lag-1 mean cosine of 0.753878 is a different analysis scope.
- Updated docs/EXP.md, docs/RESULTS.md, and docs/results/v0-grad-mech-l2-norm.md to state the supported empirical conclusion: high lag-1 raw-window overlap does not imply small relative full-gradient changes in the ten retained trajectories. Mean pairwise relative L2 is 0.637249, median 0.762024, and 7,056/12,826 pairs (55.01%) are at least 0.6. The conclusion is not a mathematical non-Lipschitz proof.
- Archived only the 1.5 MB v0-grad-mech-l2-norm result directory under archive/v0-grad-mech-test/outputs/train/. Left outputs/train/v0-grad-mech-l2-norm as a symbolic link so configured paths continue to resolve. The ten BF16 gradient caches under outputs/train/v0-crb-reconst-test/gradients/ were not moved, linked into the archive, or deleted.


### 2026-11-08 - Local-Risk sparse-sampling extension prepared

- Added an extension measuring output-head variance and direction-dispersion correlations against all-32-point full-gradient targets when only 32, 24, 16, 8, or 4 proxy gradients are observed per block.
- For counts below 32, fixed 20 uniform without-replacement sampling replicates with nested subsets and seed 0. Count 32 is the exact cached baseline.
- Reuse saved full/head Gram matrices and complete block targets; no model/GPU run or gradient recomputation is needed.
- Added configs/rage/v0-gradient-proxy-local-risk-sparse.json and scripts/analyze_sparse_proxy_risk.py. Python bytecode and JSON parsing checks passed. Current state: prepared; execution pending.

Last modified: 2026-11-08


### 2026-11-08 12:02 CST - Local-Risk sparse-sampling extension completed

- Reused ten saved full/head Gram matrices and existing complete block targets; analyzed 396 complete 32-start blocks. All 396 k=32 head variance and direction-dispersion values matched the original cache.
- For each block, used one exact 32-point baseline and 20 fixed-seed nested samples without replacement at k=24,16,8,4. The run wrote 32,076 sampled block records and 3,240 per-track/per-replicate correlation records in about 5 seconds.
- Spearman for head variance versus full variance was 0.948, 0.942, 0.935, 0.916, and 0.867 at k=32,24,16,8,4. For direction dispersion it was 0.816, 0.806, 0.783, 0.704, and 0.579.
- Classified the result as Other: variance ranking remains strong at low sample counts; direction ranking degrades more. This is retrospective subsampling of cached Gram matrices, not a runtime benchmark or an interval-selection policy test.
- Updated docs/EXP.md, docs/RESULTS.md, docs/results/v0-gradient-proxy-test.md, docs/gradient-proxy-protocol.md, and docs/ORCHE.md. The part-level .complete marker is present.

Last modified: 2026-11-08


### 2026-11-08 - Local-Risk 64-point extension prepared

- User corrected the requested proxy counts to {64,48,32,16,8,4} for consecutive 64-start blocks.
- Fixed the design to use all-64 full-gradient statistics as targets and saved full/head Gram matrices, with 20 fixed-seed nested uniform samples without replacement for each lower count.
- Added configs/rage/v0-gradient-proxy-local-risk-64.json and scripts/analyze_sparse_proxy_risk_64.py. Python compilation and config parsing passed. No full gradients are recalculated; execution is pending.

Last modified: 2026-11-08


### 2026-11-08 12:10 CST - Local-Risk 64-point extension completed

- Processed 194 complete blocks across ten retained tracks with k={64,48,32,16,8,4}. All 194 full and proxy targets used every point in their respective 64-start blocks; Gram shape, symmetry, and finite-value checks passed.
- Used 20 fixed-seed nested samples without replacement for each k below 64; wrote 19,594 sampled-block records and 4,040 per-track/per-replicate correlations in about 5 seconds.
- Spearman for head variance versus full variance was 0.935, 0.931, 0.924, 0.914, 0.886, and 0.837 at k=64,48,32,16,8,4. For direction dispersion it was 0.808, 0.808, 0.805, 0.771, 0.693, and 0.557.
- Classified as Other: overall-variance ranking remains strong at small counts; direction ranking degrades more, especially at four points. This is retrospective subsampling, not a runtime or interval-selection test.
- Updated docs/EXP.md, docs/RESULTS.md, docs/results/v0-gradient-proxy-test.md, docs/gradient-proxy-protocol.md, and docs/ORCHE.md. The .complete marker is present.

Last modified: 2026-11-08


### 2026-11-08 - Local-Risk long-block extension prepared

- Added 128-, 256-, and 576-start block sizes while preserving proxy fractions 100%, 75%, 50%, 25%, 12.5%, and 6.25%. Corresponding counts are {128,96,64,32,16,8}, {256,192,128,64,32,16}, and {576,432,288,144,72,36}.
- Fixed 20 seeded nested uniform samples without replacement for every lower count. The analysis reuses saved Gram matrices and does not recompute gradients.
- Recorded that 576-start blocks produce only 19 blocks overall and at most two per track, so report pooled correlations as exploratory only.
- Added configs/rage/v0-gradient-proxy-local-risk-long.json and scripts/analyze_sparse_proxy_risk_long.py. Python compilation and config parsing passed. Execution pending.

Last modified: 2026-11-08


### 2026-11-08 13:39 CST - Local-Risk long-block extension completed

- Processed block lengths 128, 256, and 576 start indices using the same proxy fractions 100%, 75%, 50%, 25%, 12.5%, and 6.25%, with 20 fixed-seed nested samples below full count.
- Complete block counts were 93, 46, and 19 respectively. The 256-start equal-track summary includes nine tracks because track 00 has only two complete blocks. The 576-start result is pooled over 19 blocks only; no per-track correlation is estimable and the pooled result is exploratory.
- At 128 starts, 8 proxy points gave variance/direction Spearman 0.809/0.702. At 256 starts, 16 points gave 0.867/0.772 and 32 points gave 0.867/0.811. At 576 starts, 36 points gave pooled 0.892/0.704, subject to the small-pool limitation.
- The run wrote 15,958 sampled-block records, 13,332 replicate correlations, and 144 summary rows in about 14 seconds from saved Gram matrices. No model/GPU computation or gradient recomputation occurred.
- Updated docs/EXP.md, docs/RESULTS.md, docs/results/v0-gradient-proxy-test.md, docs/gradient-proxy-protocol.md, and docs/ORCHE.md. Part .complete is present.

Last modified: 2026-11-08


### 2026-11-08 - Local-Risk target spread checked

- Read-only analysis deduplicated full-gradient variance and direction-dispersion targets from the completed 32/64/128/256/576-start outputs.
- After dividing each block's full variance by its own trajectory median, pooled p90/p10 was 7.44, 5.25, 4.42, and 3.88 for lengths 32, 64, 128, and 256. This confirms meaningful within-track target spread, not just between-track scale differences.
- At length 576, the normalized p90/p10 was 2.06 but only 19 blocks exist; treat this as exploratory. Median within-track CV is unstable at this length because tracks contribute only two blocks each (or one for track 00).
- Updated docs/EXP.md, docs/RESULTS.md, docs/results/v0-gradient-proxy-test.md, and docs/gradient-proxy-protocol.md. No experiment process or cached data was changed.

Last modified: 2026-11-08

### 2026-11-08 - v0-vage Static replay completed

- Implemented Bg=ceil(N/64), Bp=ceil(N/64), 512-start blocks, static output-head variance scores, n_min=2, marginal block priority, and largest-gap anchor insertion. Defaults and tie rules are in configs/rage/v0-vage.json and docs/results/v0-vage.md.
- Reused ten complete proxy Gram caches and retained BF16 gradient caches. CPU replay took about 1.6 seconds with no model or GPU work. Prospective counts: VAGE 204 full-gradient and 204 proxy calls; Uniform 204 full-gradient calls.
- VAGE-Static had equal-track mean L2 3.5437 versus Uniform 3.4824 and mean cosine 0.3651 versus 0.3971. It won 2/10 tracks on mean L2 and 1/10 on mean cosine. Primary metrics favor Uniform.
- Three direct BF16-cache interpolation errors were checked against Gram calculations; absolute differences were at most 0.00295 L2. Outputs and .complete are in outputs/train/v0-vage/. No CRB task was restarted.

Last modified: 2026-11-08

### 2026-11-08 - v0-vage n_min=1 sensitivity completed

- Preserved the n_min=2 result and ran a separate n_min=1 config at configs/rage/v0-vage-nmin1.json, with outputs/train/v0-vage-nmin1/. One anchor is the block center; the next selected point is the farther endpoint. Reconstruction outside the final global anchor range uses the nearest anchor.
- Equal-track mean L2 worsened from 3.5437 to 3.8747 and mean cosine fell from 0.3651 to 0.3156. Uniform remains 3.4824 L2 and 0.3971 cosine.
- All ten tracks lack the first start in the final n_min=1 anchor set; four also lack the last start. This is a plausible contributor, not an isolated causal explanation. No model or GPU work occurred; the CRB queue remains stopped.

Last modified: 2026-11-08

### 2026-11-08 - v0-vage n_min=1 method comparison completed

- Added scripts/v0_vage_methods.py and replayed Uniform, Static, Quota, Fusion, Residual diagnostic, and Oracle on the same ten caches and reconstruction evaluator. Outputs and completion marker are in outputs/train/v0-vage-methods-nmin1/.
- Equal-track mean L2: Uniform 3.4824, Quota 3.5031, Residual diagnostic 3.5159, Static 3.8747, Fusion 3.8762, Oracle 3.8774. Mean cosine also favors Uniform at 0.3971; the closest method is Residual diagnostic at 0.3801.
- Residual candidate anchors were limited to the observed proxy points to preserve Bp. Since Bg=Bp, all proxy positions become anchors; this row cannot test adaptive residual selection. Fusion used kappa=8 and leave-one-track-out variance scale calibration. Oracle used full cached truth only for scoring.
- Original Static and Uniform anchor sets were confirmed identical to the preceding n_min=1 replay. No new model or GPU calls occurred. The CRB queue remains stopped.

Last modified: 2026-11-08

### 2026-11-08 - v0-uni-reconst-scale completed

- Added configs/rage/v0-uni-reconst-scale.json and scripts/v0_uni_reconst_scale.py. Reused ten retained full-gradient caches through existing full-coordinate Gram matrices; checked cache markers, gradient file sizes, Gram shapes, and finite values.
- Replayed s={1,2,4,8,16,32,64,128}, K=ceil(N/s), whole-trajectory Uniform anchors, and the same linear interpolation evaluator. The 80 trajectory-scale rows completed in about 0.8 seconds without model or GPU work.
- Equal-track mean L2 increases from zero at s=1 to 3.8809 at s=128; mean cosine decreases from one to 0.2563. The s=64 row matches the preceding Uniform baseline at L2 3.4824 and cosine 0.3971. All ten s=1 trajectories have exact L2 zero and cosine one.
- Raw results and .complete are in outputs/train/v0-uni-reconst-scale/. Result report: docs/results/v0-uni-reconst-scale.md. The CRB queue remains stopped.

Last modified: 2026-11-08

### 2026-11-08 - v0-uni-data-evaluation prepared

- Added experiment config and GPU replay entry. Fixed seed 0, ten nonadjacent references per trajectory, 100 total, lag 1-575, and exclusion of exact reference starts from queries and lag endpoints.
- Relative score difference uses |a-b| divided by the mean absolute score magnitude because cosine and dot products can be negative. Absolute differences, sign flips, and zero denominators are retained to interpret the ratio.
- Existing full-gradient BF16 caches are read only. GPU scoring and lag aggregation are pending.

Last modified: 2026-11-08

### 2026-11-08 - v0-uni-data-evaluation first-run repair

- The first GPU run completed track 00 score computation but stopped before lag aggregation because lag_rows unpacked a three-dimensional score array into two variables. The track 00 score file remains intact; no completion marker or lag results were written.
- Corrected the shape unpacking in scripts/v0_uni_data_evaluation.py and restarted the same fixed protocol. The failed run is not a completed experiment result.

Last modified: 2026-11-08

### 2026-11-08 - v0-uni-data-evaluation completed

- The repaired run scored 12,736 non-reference starts against 100 seeded references, ten per trajectory, using complete BF16 gradient vectors on the RTX 4080. All references are nonadjacent within their source trajectories and are excluded from query and lag endpoints.
- Computed 11,500 per-track lag/metric rows for lags 1-575 and the equal-track summary. Lag-1 mean relative difference is 0.4856 for cosine and 0.5372 for dot product; sign flips are 9.40%. Lag-64 relative differences are 1.1524 and 1.2668. No zero denominators occurred.
- Saved all ten score matrices, reference manifest, summary tables, lag plot, and .complete under outputs/train/v0-uni-data-evaluation/. Recorded protocol and interpretation in docs/results/v0-uni-data-evaluation.md. GPU is idle; no CRB work restarted.

Last modified: 2026-11-08

### 2026-11-08 - v0-uni-data-evaluation score distribution amendment

- Read all ten saved score matrices, excluding the 100 reference starts. Computed signed means, population variances, mean absolute values, and median absolute values over 1,273,600 valid query-reference pairs, with pooled and equal-track summaries.
- Pooled cosine signed mean/variance are 0.0284/0.0283; mean and median absolute values are 0.1378/0.1207. Pooled dot signed mean/variance are 0.6101/9.1331; mean and median absolute values are 1.8782/1.0334.
- Updated docs/results/v0-uni-data-evaluation.md, docs/EXP.md, and docs/RESULTS.md. The experiment conclusion now distinguishes a near-zero signed mean from low typical cosine magnitude and relates both to the lag-1 difference. Saved scores and original lag outputs were not changed.

Last modified: 2026-11-08

### 2026-11-08 - v0-uni-data-evaluation median amendment

- Computed pooled signed medians from the saved valid score pairs: cosine 0.0284 and dot product 0.1635. Pooled absolute-value medians remain 0.1207 and 1.0334.
- Added both median types to the result report and updated the EXP and RESULTS summaries. The score and lag outputs were not modified.

Last modified: 2026-11-08

### 2026-11-08 - experiment index and narrative continuation

- Expanded the opening EXP overview to batch-level rows for all maintained experiments, including individual terminated and unstarted CRB trajectory batches. Corrected two stale completed-batch labels in the detailed sections.
- Continued the user-provided TEMP.md narrative in the project root using the existing question-led, plain-language structure. Added VAGE allocation and sensitivity results, the eight Uniform reconstruction budgets, and the 100-reference lag score study. Every non-smoke experiment in the current overview is now represented; no new experimental result was generated.

Last modified: 2026-11-08

### 2026-11-08 - VAGE method explanations added to narrative

- Expanded the VAGE section of TEMP.md with plain-language descriptions of Uniform, Static, Quota, Fusion, Oracle, and Residual. Clarified that all methods share the same final linear reconstruction and differ in anchor placement, while Oracle is nondeployable and the current Residual comparison is degenerate.
- No experimental results or code were changed.

Last modified: 2026-11-08

### 2026-11-08 - pair-mean reconstruction supplements completed

- Added a shared config and entry script to evaluate a constant representative equal to the mean of adjacent true-gradient anchors on each anchor gap. Queried anchors remain exact; n_min=1 uncovered global ends use the nearest anchor, as before. All anchor sets, proxy selections, and budgets are unchanged.
- Completed two CPU-only batches under outputs/train/v0-vage/pairmean/ and outputs/train/v0-uni-reconst-scale/pairmean/. Both .complete markers are present. Wrote 100 VAGE trajectory-method rows and 80 Uniform trajectory-scale rows.
- At s=64 Uniform, pair mean L2/cosine are 3.5445/0.3498 versus linear 3.4824/0.3971. VAGE-Static n_min=2 pair mean is 3.5880/0.3205 versus linear 3.5437/0.3651. Both primary metrics worsen on average for all compared VAGE methods.
- Recomputed linear metrics from the frozen anchors and matched all 80 previous Uniform scale rows and 20 primary VAGE rows. Three direct BF16-cache pair-mean L2 checks differed from Gram calculations by at most 0.00310. No model or GPU work occurred.

Last modified: 2026-11-08

### 2026-11-08 - pair-mean narrative integrated into original experiments

- Updated TEMP.md so the pair-mean supplement appears inside the original v0-vage and v0-uni-reconst-scale sections. Each section now compares linear and pair-mean reconstruction on the same anchors in one table and places the interpretation next to the original batch results.
- Removed the need for a separate reconstruction-supplement section in the narrative. No experiment code or outputs were changed.

Last modified: 2026-11-08

### 2026-11-08 - v0-var-proxy-estimate completed

Last modified: 2026-11-08

- Confirmed ssh tsl points to /root/TS-Lab. Read EXP, ORCHE, LOG, and the gradient-proxy protocol before execution. The rage Conda environment and ten complete full/head Gram caches were available; RTX 4080 was idle. The user-terminated CRB replay remained stopped.
- Added configs/rage/v0-var-proxy-estimate.json and scripts/v0_var_proxy_estimate.py. The first short replay exposed a wrong relative output path, placing only this new experiment output at the project root. Corrected the config, removed that new misplaced output, and reran into outputs/train/v0-var-proxy-estimate/ on the data disk. Source caches were not modified.
- Completed 46 full 256-start blocks across ten trajectories. Each block used eight even output-head proxy positions and four even full-gradient anchors. Ten terminal fragments were excluded and their lengths recorded. Results include block_metrics.csv, correlations.csv, input_manifest.json, config_snapshot.json, and .complete.
- Pooled Spearman correlations: proxy variance versus mean L2 0.8538, mean cosine 0.3977, and mean absolute norm difference 0.4970. Recorded the positive cosine association and the small per-track block counts in docs/results/v0-var-proxy-estimate.md, EXP.md, ORCHE.md, and RESULTS.md. No GPU work or gradient recomputation occurred.

### 2026-11-08 - true variance reconstruction supplement

Last modified: 2026-11-08

- Reused the 46 original 256-start blocks and their unchanged reconstruction metrics. Added scripts/v0_true_var_reconstruction.py to compute population vector variance from all 256 cached full gradients per block and correlate it with the three metrics. The source gradient caches were read only; no model or GPU calls occurred.
- Wrote outputs/train/v0-var-proxy-estimate/true_variance/block_metrics.csv, correlations.csv, input_manifest.json, and .complete. All 46 full variance values matched the prior 256-point full-variance study exactly (maximum absolute difference 0).
- Pooled Spearman: full variance versus mean L2 0.9463, mean cosine 0.3240, mean absolute norm difference 0.6381, and eight-proxy variance 0.9169. Recorded the oracle nature of full variance, the shared gradient-scale dependence, and the small within-track sample counts in the result report, EXP, RESULTS, and ORCHE.

### 2026-11-08 - four-anchor mean representative supplement

Last modified: 2026-11-08

- Reused the 46 fixed 256-start blocks, the same four full-gradient anchors, and the existing eight-proxy and all-256 true variance values. Added scripts/v0_four_anchor_mean.py to average the four full-gradient anchor vectors and use that one vector for all 256 positions, including the anchor starts. No model or GPU calls occurred.
- Wrote outputs/train/v0-var-proxy-estimate/four_anchor_mean/block_metrics.csv, correlations.csv, summary.json, input_manifest.json, and .complete. The pooled Spearman values for true variance versus mean L2/cosine/absolute norm difference are 0.9873/0.2670/0.8078; for proxy variance they are 0.9126/0.3374/0.6444.
- Compared aggregate metrics with the earlier linear reconstruction and documented the exact mean-squared-error variance decomposition in the result report. Updated EXP, ORCHE, and RESULTS. No cached source data were changed.

### 2026-11-08 - TEMP.md narrative updated for v0-var-proxy-estimate

Last modified: 2026-11-08

- Added a question-led prose section to the user-specified root TEMP.md immediately after the variance-spread discussion and before v0-vage. It covers the eight-proxy/four-anchor linear replay, the complete-variance comparison, and the four-anchor constant-mean supplement using the recorded 46-block results.
- Checked the reported Spearman coefficients and aggregate reconstruction metrics against docs/results/v0-var-proxy-estimate.md and the completed output summaries. No experiment code, parameters, or raw results were changed.

### 2026-11-08 - TEMP.md experiment moved and expanded

Last modified: 2026-11-08

- Moved the v0-var-proxy-estimate narrative section from between the variance-spread and VAGE sections to the end of the user-specified root TEMP.md, preserving chronological section order.
- Expanded it with three result tables: six linear-reconstruction correlations, six constant-four-anchor-mean correlations, and the paired aggregate reconstruction metrics. Included pooled Pearson and Spearman, equal-track Spearman, exact sampling rules, the oracle status of true variance, and the constant-representative variance decomposition.
- Cross-checked rounded table values against the completed correlation CSV files and four-anchor-mean summary JSON. No experiment code, parameters, or raw results changed.

### 2026-11-08 - trajectory mean-gradient norm comparison

Last modified: 2026-11-08

- Read current EXP, ORCHE, LOG, and gradient-proxy results before work. Confirmed the ten cached full/head Gram pairs and rage Conda environment; GPU was idle and the terminated CRB replay remained stopped.
- Added scripts/v0_grad_traj_level_norm_gap.py. Computed both the norm of the mean gradient vector and the mean of window gradient norms for full and output-head gradients across all 12,836 legal windows in the ten fixed tracks. Wrote 20 trajectory-kind rows, spread summary, paired associations, manifest, and .complete under outputs/train/v0-grad-traj-level-norm-gap/ on the data disk. No model or GPU work occurred.
- Primary norm-of-mean full range 0.7780-1.5050 (1.9345x); output-head range 0.0607-0.1515 (2.4960x). Ten-track full/head Spearman 0.8788. The secondary mean-of-window-norm spread is narrower. Recorded per-track values and interpretive limits in docs/results/v0-grad-traj-level-norm-gap.md, EXP, RESULTS, and ORCHE.

### 2026-11-08 - experiment renamed to v0-grad-traj-level-norm-gap

Last modified: 2026-11-08

- Applied the user-specified standalone experiment code v0-grad-traj-level-norm-gap to the entry script, output tree, logs, EXP, ORCHE, RESULTS, and result report.
- Re-ran the analysis under the final experiment name. All trajectory metrics, spread summaries, and paired associations matched the provisional run exactly; the new output .complete marker is present. Removed only this turns provisional output and log directory after verification.

### 2026-11-08 - v0-gradient-variance-decomposition completed

Last modified: 2026-11-08

- Read EXP, ORCHE, LOG, the prior v0-var-proxy-estimate report, and the reconstruction code. Confirmed the rage environment, ten full-gradient Gram caches, 46 fixed blocks, and idle GPU. The terminated CRB replay remained stopped.
- Added scripts/v0_gradient_variance_decomposition.py. Reused all 46 prior blocks and four anchors. Recomputed all and 252-nonanchor L2, absolute norm difference, cosine, and direction error from full Gram matrices; verified 46 true variances and original all-position reconstruction metrics against prior CSV values.
- Computed mean gradient norm, norm variance, full-gradient direction dispersion, relative vector variance, and exact amplitude/direction-related variance components. Wrote block_metrics.csv, correlations.csv, input_manifest.json, two PNG and two SVG track-colored raw/controlled scatter figures, and .complete under outputs/train/v0-gradient-variance-decomposition/.
- Pooled Spearman V_g versus mean norm 0.970, amplitude variance 0.569, and D_h -0.222. D_h versus nonanchor direction error 0.701, controlled rank 0.713. V_g versus cosine changes from pooled +0.324 to controlled rank -0.431. Proxy variance versus cosine changes from +0.398 to -0.202. Leave-one-track-out controlled signs held in all ten omissions. Exact variance decomposition and nonanchor scaling checks passed; plots were inspected. No new model or GPU work occurred.
- Updated EXP, ORCHE, RESULTS, and docs/results/v0-gradient-variance-decomposition.md. Interpret all associations as descriptive for the ten selected trajectories; no independent-window significance tests were performed.

### 2026-11-08 - TEMP.md narrative appended for variance decomposition

Last modified: 2026-11-08

- Appended a chronological, question-led v0-gradient-variance-decomposition section to the user-specified root TEMP.md. It explains the six new block statistics, raw and controlled associations, non-anchor comparison, and figure locations with two result tables.
- Verified the displayed rounded values against outputs/train/v0-gradient-variance-decomposition/correlations.csv and block_metrics.csv. Existing TEMP.md sections were not changed.

### 2026-11-08 - variance decomposition result-document cross-check

Last modified: 2026-11-08

- Confirmed TEMP.md, EXP, ORCHE, RESULTS, LOG, and the detailed result report already recorded v0-gradient-variance-decomposition and its completed 46-block output.
- Clarified in TEMP.md that the controlled sign reversal is a rank association; controlled Pearson for full variance versus cosine is -0.109 and for proxy variance versus cosine is about +0.001. Added direct PNG figure paths.
- Expanded the RESULTS.md summary into a compact table of raw and controlled associations and included anchor-exclusion levels and figure references. No code, inputs, metrics, or figures changed.

### 2026-11-08 - v0-directional-marginal-gain completed

Last modified: 2026-11-08

- Read current EXP, ORCHE, LOG, the prior variance-decomposition report, and the existing spaced/split-gap/reconstruction code. Confirmed rage Conda, ten cached full/head Gram matrices, 46 complete blocks, and idle GPU. The terminated CRB replay remained stopped.
- Added scripts/v0_directional_marginal_gain.py. Reused fixed 256-start blocks and calculated true direction dispersion, sparse eight-point output-head direction dispersion, full and relative vector variance. For n=2,4,8,16, replayed uniform full-gradient anchors followed by one leftmost largest-gap midpoint, at local positions 127,42,54,8.
- Wrote 184 block-budget rows, 64 per-budget correlation rows, 24 budget summaries, 46 proxy-dispersion rows, 160 leave-one-track-out sensitivity rows, manifests, and .complete under outputs/train/v0-directional-marginal-gain/. The source caches were read only; no model or GPU work occurred. The first completed replay was rerun after adding a persistent sensitivity table; the final tables and .complete were regenerated under the same output path.
- True direction dispersion versus starting error has pooled Spearman 0.616/0.701/0.547/0.395 by ascending n, but versus total marginal gain -0.074/0.142/0.256/0.060. Sparse proxy direction dispersion versus gain is -0.002/0.380/0.235/0.056. The n=4 proxy signal is moderate and budget-specific. Nonnew-position gain and within-track correlations were recorded separately. Updated EXP, ORCHE, RESULTS, and docs/results/v0-directional-marginal-gain.md.

### 2026-11-08 - TEMP.md directional marginal gain narrative

Last modified: 2026-11-08

- Appended the v0-directional-marginal-gain section to the user-specified root TEMP.md after the variance-decomposition section, preserving chronological order. It distinguishes starting difficulty from measured one-anchor gain, reports four fixed budgets in a table, and explains the nonnew-point check and limited n=4 proxy signal.
- Checked displayed values against the completed budget_summary.csv, correlations.csv, proxy_association.json, and leave_one_track_out.csv outputs. Earlier TEMP.md sections were not changed.

### 2026-11-08 - Directional error interpretation and TEMP coverage

Last modified: 2026-11-08

- Defined per-point and block-mean cosine direction error, its [0,2] interpretation, and the sign and units of one-anchor gain in docs/results/v0-directional-marginal-gain.md, docs/EXP.md, docs/RESULTS.md, and TEMP.md.
- Recomputed Spearman association between starting error and fixed one-anchor total/nonnew gain from the saved 184 block-budget rows. At n=16, trajectory 07 block 0 has the highest starting error (0.5299) but negative total gain (-0.00587); this fixed added anchor worsens its average direction error. Other anchor choices were not tested.
- Added the completed v0-grad-traj-level-norm-gap experiment to TEMP.md and made the three Local-Risk sampling extension codes explicit. TEMP.md now covers the research experiments recorded in EXP.md. The environment validation smoke test was excluded as requested. No experiment was rerun.

### 2026-11-09 - v0-active-value-rank planned

Last modified: 2026-11-09

- User requested the first two batches of active value-ranking experiments.
- Checked existing experiment documents, proxy and cost reports, cache formats, GPU, RAM, disk, and tmux. Existing worktree changes were retained. GPU idle; data disk about 250 GiB free.
- Registered 1-expanded-cross-rank and 2-active-correction. Use saved scores and gradients; do not restart CRB.
\n### 2026-11-09 - v0-active-value-rank batches 1 and 2 completed

Last modified: 2026-11-09

- Added scripts/v0_active_value_rank.py. Initial batch-1 launch failed because its output_path argument pointed to the project root. Corrected it to outputs/train/v0-active-value-rank/ and removed only the empty directories from that failed launch.
- The next validation attempt found NaN full scores at starts reserved for the 100 references. Excluded those reserved starts from the stride-16 candidate pool and checked finiteness only for cross-track reference columns. No source cache was modified.
- Batch 1 completed on the GPU in about 9 seconds, with 100 references and 714-766 cross-track candidates each. Mean per-reference Spearman was 0.7104. A spot-check against the older Cross-Rank output matched full, head, embedding, and loss values to float rounding.
- Batch 2 completed in about 6 seconds of CPU replay, producing 4,800 per-reference selection rows, 48 summary rows, query manifests, and a .complete marker. At top 10% and B=64, boundary regret was 0.04192 versus 0.05148 for highest-proxy and 0.07525 for direct proxy. At B=8 common calibration was worse than direct proxy; at top 5% and B=64 highest-proxy beat boundary.
- Updated EXP, ORCHE, RESULTS, and docs/results/v0-active-value-rank.md. Results are retrospective, with modeled call costs and no downstream training claim. Existing terminated CRB remained stopped.

### 2026-11-09 - TEMP.md active value ranking narrative

Last modified: 2026-11-09

- Updated the last-modified date and appended the two completed v0-active-value-rank batches to the user-specified root TEMP.md. The section explains the expanded candidate pool, ranking baseline, budgeted correction policies, top-10-percent results, the eight-label and top-5-percent exceptions, and modeled versus measured costs.
- Checked numbers against docs/results/v0-active-value-rank.md and the experiment overview. TEMP.md now covers every non-smoke experiment code in the current EXP overview. No experiment was rerun and no cached results were changed.

### 2026-11-09 - TEMP.md chronological experiment table

Last modified: 2026-11-09

- Added a three-column table at the start of the user-specified root TEMP.md with each experiment's latest recorded date, code, and plain-language research question. Dates follow docs/EXP.md and the individual result reports; same-day rows follow the article's experiment sequence. The smoke test remains excluded as requested.
- Verified 13 experiment rows, ascending recorded dates, and coverage of the non-smoke experiment codes in docs/EXP.md. No experiment or result data was changed.

### 2026-11-09 - v0-active-value-rank diagnostic batches 3 and 4

Last modified: 2026-11-09

- Connected to the specified tsl server and checked existing EXP, ORCHE, LOG, RESULTS, cached inputs, worktree, processes, GPU, memory, and data disk. Existing worktree changes were preserved. SSHFS failed because the local mount point is on a macFUSE volume, so remote files were edited through SSH.
- Added scripts/v0_active_value_diagnostics.py. Batch 3 reused 100 cached full/head cosine pools and produced 7,900 coverage rows, 7,270 true-top rank rows, 600 bin rows, 100 figures, and .complete in about 23 seconds. No gradient or model calls occurred.
- Batch 4 replayed three simple query policies with the same eight counted initial labels as the existing complex policy, plus 30 seeded random repeats. It reused the prior complex selection metrics, wrote 13,300 recall rows, query manifests, summary and plot, and .complete in about three seconds.
- At k,2k,3k,5k proxy coverage is 0.4023, 0.6480, 0.7950, 0.9265. At B=64, complex/pure-boundary recall is 0.5415/0.5064. The existing complex correction is worse at B=8. Updated EXP, ORCHE, RESULTS, and the detailed report. No CRB job was resumed, and no automatic inspection was enabled.

- Added the aggregate 100-reference curve and rank-bin figure plus a 120-row random budget-seed table. An initial attempt to patch the main script used unavailable plain python, so no patch was applied; both owned batches were replayed from their unchanged cached inputs and the aggregate script then completed. Validated every reference bin sum, full-pool recall, k-pool versus zero-budget agreement, exact reuse of 400 complex recall rows, output row counts, figure counts, and completion markers.

### 2026-11-09 - TEMP.md active-value diagnostic narrative

Last modified: 2026-11-09

- Added sections 5.4 and 5.5 after the existing active-value-rank narrative in the user-specified root TEMP.md. They state the two questions, give direct answers, summarize proxy coverage and equal-budget Recall@k, and point to per-reference figures and the detailed result report.
- Updated the active-value-rank row in the experiment index. The numeric claims were checked against the completed batch-3 and batch-4 summary tables. No experiment was rerun or source result was changed.


### 2026-11-09 - v0-crb-var completed

The ten-trajectory CPU replay completed using the retained BF16 full-gradient caches. Verified outputs/train/v0-crb-var/block_metrics.csv, trajectory_summary.json, and .complete. Equal-track mean relative variance was 0.6410 for CRB versus 0.6802 for Uniform; mean direction dispersion was 0.4051 versus 0.4402, with CRB lower on all ten tracks for both. Block-size Spearman correlations were stronger for CRB than Uniform. Absolute variance did not consistently favor CRB. Full result and limitations are documented in docs/results/v0-crb-var.md. No new gradients or GPU computations were used.


### 2026-11-09 - v0-crb-var budget audit and corrected replay

Audited the first completed output and found it contained 63 intervals per track, inconsistent with the intended ceil(window_count/64) budget. That output was discarded and not used for the result. Re-ran the entire batch in tmux with the corrected per-trajectory budget. Verified anchor budgets 10,22,21,22,21,22,22,20,22,22 for trajectories 00-09; CRB and Uniform each form budget minus one contiguous intervals covering every start exactly once. The corrected run has 388 block rows and .complete. Corrected means and correlations are in docs/results/v0-crb-var.md.

### 2026-11-09 22:19 CST - v0-proxy-dp-segmentation prepared

Registered a new independent cached-gradient experiment. It samples existing output-head proxy gradients every four starts, linearly reconstructs the proxy vectors, and solves the unweighted sum of vector trace variances with exact DP for K=ceil(N/64), minimum interval length 16, and no maximum interval length. A same-K evenly spaced partition is the baseline. Evaluation uses the retained full-gradient cache. GPU was idle and data-disk free space was 22 GiB at launch. No new gradients are computed.

### 2026-11-09 - Time-300B sharded BIN load fix

- The `energy/buildings_900k` fetch completed with 15 BIN shards and one metadata file. The Time-300B loader returned no rows because it required exactly one BIN file; raw input was present and the completion marker resolved all 16 files.
- Updated the source-specific adapter to validate metadata against ordered BIN shards, memory-map each shard, and read each trajectory by its global offset, including trajectories crossing shard boundaries.
- Re-ran Load successfully: 2,466,924 trajectories were written to `data/src/shard_dataset/selection24-time300b-fixed`. The output index contains 15,199,263,843 time points.
- The complete converted source set contains 3,008,537 trajectories and 19,725,286,219 time points across Chronos, LOTSA, Monash, UTSD, and Time-300B.
