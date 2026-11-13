# Operation Log

Last modified: 2026-11-10

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

Last modified: 2026-11-10

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


### 2026-11-09 - Output-head gradient cache support

Last modified: 2026-11-09

- Inspected the existing cache, Chronos Tiny engine, model registry, Timer-XL and TimesFM adapters, experiment documents, worktree, GPU, memory, data disk, tmux sessions, and active processes. The v0-proxy-dp-seg task was active on GPU; no validation job was started.
- Confirmed the v0-grad-mech-test warmup checkpoint has tie_word_embeddings=true and that input embedding and lm_head weights are the same Parameter and storage. ChronosTinyGradientEngine defaults to lm_head.weight and uses _fast_batch under torch.no_grad with captured head inputs and an analytic CE derivative. It excludes embedding-path terms. ChronosT5Engine.gradient_sketch instead calls torch.autograd.grad over every trainable parameter, so it computes the combined gradient for the tied shared weight and the rest of the model.
- Added an output_head cache mode. Chronos-T5 Tiny/Small/Base and TimerXL use analytic output-head derivatives. TimesFM 2.5 freezes backbone parameters and differentiates only its two nonlinear output projection blocks. Added output-head identity fields to the cache manifest so these resources cannot collide with full-model caches. Cache roots are restricted to the project outputs directory.
- Extended the cache runner to Chronos-T5 Tiny/Small/Base, TimesFM 2.5, and TimerXL 84M. TimesFM 3.0 remains excluded because its project capability is inference-only.
- Added docs/gradient-output-head.md with the model scope, loss semantics, tie handling, command shape, and required numerical validation. Read-only CPU loading confirmed the warmed ChronosTiny checkpoint ties the input and output weights. AST parsing and git diff --check passed. No model gradient cache was generated, and no tests or numeric gradient comparisons were run.

### 2026-11-10 00:06 CST - proxy-DP memory-limit interruption and rewrite

The first launch entered trajectory 01 proxy interpolation but produced no checkpoint before the user reported a container memory overflow and restarted it. The previous true-gradient evaluator converted an entire interval to float32; with up to 8,394,496 parameters per time point, a single long block could allocate multiple GiB, in addition to proxy reconstruction buffers. The 62 GiB cgroup limit is lower than host free-memory figures suggested. After restart, confirmed no replay process, tmux session, or trajectory checkpoint remained. Reworked proxy interpolation to BF16 with bounded 32-anchor host transfers; proxy Gram construction now converts 65,536 coordinates at a time and accumulates FP32 products; true-gradient metrics now use four-start CPU chunks with FP32 reductions. Added partition coverage checks and cgroup-memory logging. The selected cache files remain intact and data-disk space is unchanged.

### 2026-11-10 00:07 CST - proxy-DP memory-bounded batch restart

Restarted the unweighted then window-weighted longest-four batches in tmux `v0-proxy-dp-longest4` after Python bytecode compilation. The run reports cgroup memory at each trajectory start; inspect memory and completion checkpoints before allowing the next trajectory to continue if usage rises unexpectedly.

### 2026-11-10 - Optional LoRA support for the output-head model batches

Last modified: 2026-11-10

- Added opt-in PEFT LoRA support to the shared student worker for Chronos-T5 Tiny/Small/Base, TimerXL 84M, and TimesFM 2.5 200M. Default targets are Chronos T5 attention q/v and TimerXL/TimesFM attention q_proj/v_proj; embeddings and forecasting heads remain frozen.
- Kept existing manifests unchanged by enabling LoRA only when training.lora is present. Updated adapter-backed loss closures to reference the wrapped PEFT model, and added the normalized adapter configuration to run metrics and the optimizer-resume contract.
- Updated optimizer parameter selection and gradient clipping to operate only on trainable parameters when training options are used.
- Pinned PEFT 0.18.1 in the Chronos, TimerXL, and TimesFM model profiles and added import checks for those environments.
- Added docs/lora-support.md with model coverage, defaults, manifest syntax, tied-embedding handling, and validation limits. No model run or training experiment was started; implementation validation is limited to static checks.

### 2026-11-10 00:14 CST - proxy-DP full-cache paging issue corrected

After restarting with BF16 proxy reconstruction and chunked true-gradient arithmetic, cgroup memory rose to about 39 GiB while reading the 22 GiB full-gradient cache. Inspection showed 22.7 GiB resident file mapping from torch.from_file, not anonymous float32 tensors. Stopped the active run before another track started. Unweighted source ordinal 01 had completed and its per-track checkpoint was atomically written; its cache-derived metrics are retained for resume. Replaced whole-file mapping with four-start os.pread reads and POSIX_FADV_DONTNEED after each chunk, so only bounded chunk tensors are materialized and consumed file pages are advised for eviction. Used POSIX_FADV_DONTNEED on the already-read ordinal 01 files; no cache files were deleted. Current cgroup memory is around 13 GiB and no replay process is active.

The earlier 00:07 restart entry records the attempted session; it did not complete either objective batch. Current actual state is one unweighted source-track checkpoint (ordinal 01), with all other tracks and the window-weighted batch pending.


### 2026-11-10 00:46 CST - v0-proxy-dp-segmentation completed

Completed the unweighted and length-weighted exact DP replays on source ordinals 01, 05, 06, and 09. Each objective has 176 interval rows, eight trajectory-method summaries, four per-track checkpoints, and a .complete marker. Validation confirmed K=22 for all tracks and methods, lengths at least 16, exact contiguous coverage, and finite metrics. Equal-track means: unweighted DP versus uniform relative variance 0.5031/0.6762, direction dispersion 0.2995/0.4298, window-weighted direction 0.5844/0.4297, absolute variance 3.7787/9.4982. Window-weighted DP: 0.5717/0.6762, 0.3572/0.4298, 0.4923/0.4297, and 12.5689/9.4982. Results and limitations are recorded in docs/results/v0-proxy-dp-segmentation.md, EXP, RESULTS, and ORCHE. Peak observed cgroup use remained about 12.8 GiB after replacing whole-file mappings with four-start reads and POSIX_FADV_DONTNEED. No gradients were recomputed and no server shutdown or automatic inspection was issued.

### 2026-11-10 00:52 CST - final experiment configuration audit

A final artifact audit found configs/rage/v0-proxy-dp-segmentation.json still described the initial ten-track, unweighted-only plan. Updated it to the actual selected source ordinals, deleted ordinals, BF16 proxy reconstruction, both objectives, and output directories. Revalidated JSON syntax, source cache shapes for ordinals 01, 05, 06, and 09, absence of deleted full-gradient caches and markers for the shortest six, both completion markers, and 176 interval rows/eight summaries per objective. The exact DP transition applies length/N to V(j,i) for the window objective; both methods use K=22 and the reported minimum-length and contiguous-coverage checks pass. No replay process remains.

### 2026-11-10 - selection24 Time-300B directory cleanup

- Confirmed `selection24-time300b` was an empty output from the earlier failed single-BIN load; its `load/shards` tree contained no files. The valid 2,466,924-trajectory dataset and index were in `selection24-time300b-fixed`.
- Removed only the empty stale directory and renamed the valid dataset to `data/src/shard_dataset/selection24-time300b`. No dataset files were deleted. The `selection24-time300b-buildings` empty directory was not changed.
- Last modified: 2026-11-10.

### 2026-11-10 01:37 CST - all raw full-gradient caches deleted

At the user's direction, removed the four remaining raw BF16 full-gradient trajectory vectors (ordinals 01, 05, 06, 09; 93,481,107,456 bytes total, about 87.0 GiB) and their four cache-specific .complete markers from outputs/train/v0-crb-reconst-test/gradients/. The directory is now absent. Earlier, the six shortest trajectory caches had already been deleted. Searched the project and data disk: no BF16 full-gradient files or named raw full-gradient cache files remain. The ten head_gradients.npy files are output-head proxies, not full gradients, and were retained. Derived Gram matrices, summary artifacts, model weights, and experiment outputs were also retained. No replay process was active. Data-disk free space increased from 135 GiB to 222 GiB (60% used).


### 2026-11-10 02:03 CST - v1-rage-trev-base experiment preparation

- Read the experiment index, orchestration plan, operation log, source survey, LoRA support, Chronos-Base historical config, and training implementation. The RTX 4080 was idle, data disk had 222 GiB available, and the Chronos-T5-Base checkpoint directory exists.
- Confirmed the trainer samples each effective batch with replacement and supports low-level fixed-step updates, while production run_arm_training currently derives checkpoint boundaries from epochs. The requested fixed-step curve therefore needs explicit checkpoint-boundary support.
- Registered v1-rage-trev-base-rand followed by v1-rage-trev-base-full and recorded the five fixed candidate sources, BF16 LoRA settings, 576-point window protocol, requested update checkpoints, and 12-task evaluation suite in docs/EXP.md and docs/ORCHE.md.
- No training, dataset download, or deletion has started. Next: inspect exact source/evaluation pipeline requirements, implement fixed-step checkpoint orchestration, then perform the requested disposable end-to-end rehearsal before formal launch.

### 2026-11-10 - Rename Chronos-T5-Base Trev batches

- Renamed the active experiment batches to v1-rage-chronosT5-trev-base-rand and v1-rage-chronosT5-trev-base-full so the model family is explicit. Updated the current experiment index, detailed protocol, orchestration section, and random-batch config filename and identifier. Earlier log entry retains the original names as an accurate record of the initial registration.
- Training progress is defined by optimizer updates, not epochs: each step consumes one effective batch of 32 sampled 576-point windows. Checkpoints are fixed at 0, 100, 200, 500, 1,000, 2,000, 3,000, and 4,000 updates; the cosine schedule spans 4,000 updates.

### 2026-11-10 - Chronos-T5-Base Trev rehearsal and rename

- Canonical batch IDs are v1-rage-chronosT5-trev-base-rand and v1-rage-chronosT5-trev-base-full. Renamed the random config file and synchronized current docs and orchestration. Optimizer updates define training duration: batch 32 sampled windows per update, checkpoints at 0/100/200/500/1000/2000/3000/4000, and a 4000-update cosine schedule.
- The existing eligible source shards contain 475,152,633 points: wiki_daily_100k 274,100,000; m5 47,140,014; PEMS07 24,921,792; solar_10_minutes 7,200,720; UTSD-1G 121,790,107. Eligible series counts are 100,000 / 29,351 / 883 / 137 / 68,679 respectively. The measured total is 1.3% above the nominal 469M plan; preserve all five fixed configurations and record the exact count rather than truncating.
- BF16 Chronos-T5-Base LoRA forward/backward/update was exercised for one step; all model and adapter parameters were BF16. FEV rehearsal completed hospital_admissions_1D (SQL 0.5315187, MASE 0.6705529). TIME CPHL/15T short completed after adding python-dotenv 1.2.4 and handling non-finite historical context values by interpolation from finite observations (zero-fill only if a whole context is invalid); TIME metric computation preserves the original ground truth. TIME rehearsal metrics: MSE 0.2065, MAE 0.3161, RMSE 0.3579, MASE 1.3400, CRPS 0.2557.
- Rehearsal outputs and temporary one-task configs/data manifests were removed after recording the successful results. The official 12-task configs, shared source caches, model checkpoint, and evaluation data were retained. Formal training is not yet running.

### 2026-11-10 - Freeze measured Chronos Trev pool

- Scanned each configured bundle with the formal eligibility rule (at least 576 points, finite, non-constant). Exact pool: 199,030 series and 475,138,055 points. Per-source series/points: wiki_daily_100k 100,000 / 274,100,000; m5 29,351 / 47,140,014; PEMS07 883 / 24,921,792; solar_10_minutes 137 / 7,200,720; UTSD-1G 68,659 / 121,775,529. Corrected the preliminary UTSD count from the earlier metadata scan.
- Prepared deterministic 10% series subset: 19,903 series, 49,350,594 points, 37,906,369 legal windows. Materialized random and full pools under each batch output inputs directory. The full pool has 199,030 series, 475,138,055 points, and 360,695,805 legal windows. No training has started yet.

### 2026-11-10 02:43 CST - Start Chronos-T5-Base Trev sequence

- Started `scripts/temp/v1_rage_chronost5_trev.py` in tmux `v1-chronost5-trev`. The source manifests use the freshly scanned eligibility counts; random selection is 19,903/199,030 eligible series (seed 0), totaling 49,350,594 points. Full phase is queued in the same sequential runner.
- Verified the base model has 201,374,976 parameters, LoRA has 884,736 trainable parameters, and all parameters are BF16. The AdamW moment dtype guard passed through the first update boundary. Step 100 adapter checkpoint was written. At 02:43 CST GPU utilization was 99%, GPU memory 18.9/32 GiB, host memory available 456 GiB, and data disk had 220 GiB free. Console log: `logs/v1-rage-chronosT5-trev-base/experiment-console.log`.
- The logged ETA at step 100 was about 27 minutes for random-phase training only; evaluation follows all eight checkpoints, then full-pool training and its evaluations. No evaluation result is claimed yet.

### 2026-11-10 02:44 CST - Training heartbeat

- Random 10% phase remains active in tmux `v1-chronost5-trev`; steps 100 and 200 adapter checkpoints are present. At step 274 the ETA to step 4,000 was 25.6 minutes. GPU utilization was 91%, allocated memory 18.9 GiB / 32 GiB. No errors or dtype violations observed.

### 2026-11-10 02:45 CST - Random phase checkpoint

- The same tmux process 23453 remains active; step 500 LoRA adapter and optimizer resume state were persisted. At 02:45:59 CST the trainer was at step 500 with 98% GPU utilization and 18.9 GiB GPU memory. ETA to step 4,000 was about 24 minutes.

### 2026-11-10 02:49 CST - Random phase step 1,000

- Random 10% training reached optimizer step 1,000 and saved its adapter and resume state. The trainer process remains active in tmux `v1-chronost5-trev`; the latest ETA to step 4,000 was 20.5 minutes. GPU utilization was 89%, allocated memory 18.9 GiB. The BF16 AdamW moment guard continues to pass.


### 2026-11-10 12:37 CST - TimerXL and TimesFM Trev queue preparation

- Chronos random phase completed all eight fixed-step FEV/TIME evaluations and wrote .complete. Chronos full phase is training in tmux v1-chronost5-trev; step 500 was saved at 12:35 CST. The intermediate-checkpoint watcher removed evaluated random adapters, leaving only step 4000.
- Added TimerXL 84M and TimesFM 2.5 200M rand/full manifests with BF16 LoRA r=8/alpha=16, 4,000 optimizer updates, 32 effective batch / 8 microbatch, 576-point windows, and the same twelve FEV/TIME tasks. Both pairs reference the exact frozen Chronos input pools.
- Fixed foundation FEV quantile axis mapping after rehearsal showed a shape mismatch. Fixed FEV result metadata to tolerate deterministic adapters without Chronos sampling fields. TimerXL one-step BF16 LoRA training, merged export, FEV task 9 and TIME CPHL/15T short now pass. TimerXL rehearsal MASE=0.6650 (FEV); TIME MSE=0.1290, MAE=0.2469, MASE=1.0540.
- TimerXL environment initially imported local FEV code but lacked distribution metadata required by the result writer. Installed the project FEV checkout editable with --no-deps in both model environments and added this installation to scripts/temp/prepare_trev_other_models.sh. Both environments use datasets 4.8.5; verified TimerXL transformers 4.40.1/PEFT 0.10.0 and TimesFM transformers 5.3.0/PEFT 0.18.1.
- Started TimesFM 2.5 BF16 LoRA/FEV/TIME rehearsal in tmux v1-rehearse-timesfm25. Started automatic foundation queue in tmux v1-foundation-trev-queue; it observed Chronos rand .complete and is waiting for Chronos full .complete, then runs TimerXL rand/full followed by TimesFM rand/full.
- Runner/exporter, both model config pairs, queue shell syntax and Python syntax were checked. Formal TimerXL/TimesFM training has not started yet; no result is claimed for those batches.


### 2026-11-10 12:44 CST - Foundation rehearsals passed and EXP record recovered

- TimesFM 2.5 rehearsal passed: one BF16 LoRA update (finite loss), merged checkpoint export, FEV task 9 MASE=0.6565, and TIME CPHL/15T short MSE=0.1730, MAE=0.2900, MASE=1.2217. TimerXL passed the same path earlier: FEV task 9 MASE=0.6650; TIME MSE=0.1290, MAE=0.2469, MASE=1.0540.
- Fixed TimesFM FEV contexts to use the latest 512 observations, matching the fixed training/evaluation context and the model's 32-point patch divisibility requirement. Both FEV/TIME rehearsals now pass.
- Disposable rehearsal model checkpoints, temporary evaluation outputs/configs, and the one-task TIME data manifest/link were removed after metric capture. Source models and formal candidate/evaluation data remain.
- docs/EXP.md was accidentally truncated during a failed UTF-8 write. Restored its tracked historical contents from HEAD, then re-added the current Chronos and foundation batch records and corrected the v0-proxy completion status from docs/LOG.md and docs/RESULTS.md. Experiment data and result files were not affected.
- Automatic queue remains live in tmux v1-foundation-trev-queue. At 12:44 CST it had passed the Chronos rand dependency and was waiting for Chronos full. Chronos full remained active at step 1,689; no TimerXL or TimesFM formal training had started.


### 2026-11-10 12:46 CST - queue and cleanup audit

The foundation queue is live in tmux v1-foundation-trev-queue and is polling for Chronos full .complete. Chronos random has only adapter-step-4000 remaining after all step evaluations. Chronos full training reached step 2,000 and saved that adapter; it is still active. TimerXL and TimesFM rehearsal completion markers are present in their retained console logs, and temporary rehearsal artifacts are absent. Python compilation, shell syntax, four model JSON manifests, and git diff whitespace checks passed. Formal TimerXL/TimesFM training has not started yet by design; the queue starts them after Chronos rand/full completion.

### 2026-11-10 13:40 CST - Trev seeds 0-3 queue resumed

Last modified: 2026-11-10

Found no active tmux or training process. Chronos seed 0 rand/full both have .complete. TimerXL seed 0 rand has optimizer step 4000 and an incomplete step 0 FEV evaluation. Verified the short-context padding fix and JSON newline fix compile. Repaired the existing base-step-0000 manifest with a literal escaped newline. Ran the formal TimerXL step 0 FEV command with --resume; tasks 18 and 19 completed and the four-task results CSV was written. Created seed 1-3 rand/full configs for all three models, with matching pool and training sampling seeds. Started tmux v1-trev-seeds-0-3; its TimerXL seed 0 resume loaded the pool and saved optimizer state and began TIME step 0. Queue and progress are in logs/v1-rage-foundation-trev/seed-queue-console.log. Earlier queue state above is historical.

### 2026-11-10 13:44 CST - Seed 1 candidate pools prepared

Last modified: 2026-11-10

Ran the actual Chronos --prepare-only entry point for seed 1. Random pool: 19,903 series, 49,194,984 points. Full pool: 199,030 series, 475,138,055 points. Both input manifests parse and match sampling_seed=1; values, offsets, and regions arrays are present. Preparation tmux session exited after success. The serial seed 0-3 queue remains active and was evaluating TimerXL seed 0 step 2000 at last check.

### 2026-11-10 13:45 CST - TimerXL seed 0 random complete

Last modified: 2026-11-10

Verified outputs/train/v1-rage-timerXL-trev-84m-rand/.complete after all eight FEV/TIME checkpoint evaluations. The live queue then started TimerXL seed 0 full and saved step 100. No manual restart or data deletion was needed.

### 2026-11-10 13:52 CST - Seed 2-3 pools ready; TimerXL full training complete

Last modified: 2026-11-10

Ran the actual Chronos --prepare-only entry point sequentially for seeds 2 and 3. Seed 2 random has 19,903 series and 49,032,491 points; seed 3 random has 19,903 series and 45,892,809 points. Each full pool has 199,030 series and 475,138,055 points. All six seed 1-3 input manifests parse and match their seed; values, offsets, and regions arrays exist. The preparation session exited. TimerXL seed 0 full wrote its training manifest after 4,000 steps and entered FEV step 0 evaluation; evaluation completion is not yet claimed.

### 2026-11-10 13:58 CST - TimerXL seed 0 pair complete

Last modified: 2026-11-10

Verified TimerXL seed 0 rand and full .complete markers after all eight FEV and TIME checkpoint evaluations. The active queue automatically started TimesFM seed 0 random and loaded its candidate pool. No manual restart was needed.

### 2026-11-10 14:01 CST - TimesFM short-context rehearsal and fix

Last modified: 2026-11-10

A direct FEV task 18 rehearsal initially failed because a 59-point context was not divisible by the TimesFM patch length of 32. Updated src/adapters/foundation/eval/fev.py to left-pad TimesFM contexts to the next 32-point multiple after the existing 512-point cap. Re-ran the actual FEV evaluator with --resume: task 18 passed (MASE 8.8114). A separate task 19 run also passed (MASE 4.0807). These are base-model rehearsal metrics, not formal seed results. Deleted the temporary checkpoint wrapper, evaluation outputs, YAML config, and four rehearsal evaluator logs. The formal TimesFM seed 0 random training process remained active.

### 2026-11-10 - Trev metric clarification

Last modified: 2026-11-10

The user specified MASE as the only metric to inspect, compare, and report for all Chronos, TimerXL, and TimesFM Trev batches. Updated EXP, ORCHE, RESULTS, and the TimerXL seed 0 result document. This is a reporting decision; no training or evaluation parameters or running processes were changed.

### 2026-11-10 - Trev training-scope clarification

Last modified: 2026-11-10

The user confirmed that existing LoRA runs should be retained without rerunning. Clarified in EXP, ORCHE, RESULTS, and the TimerXL seed 0 result document that `full` means the full eligible data pool, while both rand and full train LoRA adapters with frozen base weights. No training configuration, evaluation configuration, output, or live process was changed.

### 2026-11-10 14:50 CST - v1 projected-gradient storage assessment

Last modified: 2026-11-10

Read the frozen v1 pool and evaluation declarations and measured the seed-0 full-pool offsets. The full pool has 199,030 series, 475,138,055 points, and 360,695,805 legal 576-point windows. Sampling legal starts at 0, 16, 32, ... independently within each series yields 22,669,472 projected gradients. At 4,096 dimensions, the training cache alone needs 172.954 GiB in BF16 or 345.909 GiB in FP32, before metadata and temporary files. The data disk had 226,501,459,968 bytes (210.946 GiB) free at 14:48 CST. The retained Chronos-T5-Base step-4000 adapter has 884,736 LoRA parameters, so a dense 884,736 by 4,096 projection matrix takes 13.5 GiB in FP32 or 6.75 GiB in BF16 if persisted. The 12 local evaluation Arrow files total fewer than 410,000 target points; their exact eligible history-gradient count depends on the frozen cutoff and window definition and has not been established. No gradient cache job was started and no existing data was removed. A full-pool BF16 cache is high storage risk, while FP32 and simultaneous BF16 LoRA plus head caches do not fit the current free space. Use one fixed projection operator per compatible parameter space across all training and evaluation windows; record its seed, algorithm, parameter order, and checkpoint identity. Stage output in bounded chunks without a second full-cache copy, and recheck free space before launch.

### 2026-11-10 - projected-cache scope clarification

Last modified: 2026-11-10

The user clarified that all three active models need projected-gradient preparation, the global precision is BF16, and only 4,096-dimensional projected vectors will be persisted. The full training pool contains 475,138,055 points (0.475138055 billion). The 12 local benchmark Arrow files contain 409,993 target points in total. Therefore the raw combined point-count upper bound is 475,548,048 (0.475548048 billion); the exact leakage-safe evaluation history count is lower and depends on task cutoffs. The training pool alone has 22,669,472 stride-16 legal-window anchors per model. At 8,192 bytes per BF16 projected vector, its projected cache requires 172.954 GiB per model or 518.863 GiB for three models, excluding evaluation vectors and metadata. This exceeds the data disk free space measured earlier (210.946 GiB). No raw LoRA-gradient persistence is planned or authorized by this clarification.

### 2026-11-10 15:15 CST - TimesFM TIME patch alignment recovered

Last modified: 2026-11-10

The seed queue exited at 15:03 after TimesFM seed 0 random completed 4,000 training updates and step 0 FEV. TIME step 0 failed on task 5 (ECDC_COVID/W): a 157-point context was not divisible by the model patch length 32. Updated src/adapters/foundation/eval/time.py to left-pad only TimesFM contexts to the next 32-point multiple. Python compilation passed. A single-task rehearsal was rejected by TIME manifest validation because the frozen manifest requires all eight tasks; its temporary YAML/output/log were removed. Ran the formal eight-task TIME evaluation with --resume. Tasks 1-4 were reused; tasks 5-8 completed and time_metadata.json records completion. No training data or checkpoint was deleted or rerun. The seed queue will resume from the saved optimizer and evaluation artifacts.

### 2026-11-10 15:16 CST - Trev queue restarted from saved outputs

Last modified: 2026-11-10

Started tmux v1-trev-seeds-0-3 with scripts/temp/queue_v1_rage_trev_seeds.sh, appending to the existing console log. Both TimerXL seed 0 phases were skipped via .complete. TimesFM seed 0 random resumed from its saved 4,000-step optimizer state; step 0 FEV/TIME were skipped as complete, and step 100 FEV began. The queue retains TimesFM full and seeds 1-3 in sequence. No training checkpoint was deleted or recomputed.

### 2026-11-10 - EXP Chinese revision and v1 registry audit

Last modified: 2026-11-10

Archived the previous 84,206-byte EXP text at docs/EXP.english-archive-20261110.md. Rebuilt docs/EXP.md in Chinese with the experiment and batch overview, all 24 v1 seed/model/pool batches, explicit training hyperparameters, evaluation decisions, current queue status, and links to preserved historical detail. Verified five seed-0 .complete markers; TimesFM seed-0 full was training at step 3791/4000 in the live queue log. No training process or configuration was changed.

### 2026-11-10 16:38 CST - Seed 0 complete and seed 1 started

Last modified: 2026-11-10

Verified .complete for both Chronos, TimerXL, and TimesFM seed 0 phases. For each TimesFM phase, all eight fixed checkpoints have four FEV task rows and completed TIME metadata with eight tasks. The resumed queue automatically started Chronos seed 1 random training; step 500 was saved, step 722 observed, and tmux v1-trev-cleanup-seed1 is active. GPU utilization was 98%; data disk had 211 GiB available. MASE-only TimesFM seed 0 results are recorded in docs/results/v1-rage-trev-timesfm-seed0.md and indexed in docs/RESULTS.md. No user data or checkpoints were removed.

### 2026-11-10 - EXP full-text restoration

Last modified: 2026-11-10

Restored the complete 84,206-byte EXP text from commit 12cadde directly into docs/EXP.md after the shortened Chinese rewrite removed historical protocol detail. Inserted the Chinese v1 hyperparameter and decision section into the same file, retained all 24 v1 batch rows, and refreshed TimesFM seed-0 and Chronos seed-1 statuses from .complete markers and the active queue log. No separate archive is required. Training processes and configurations were not changed.

### 2026-11-10 16:49 CST - Trev extension stopped at user request

Last modified: 2026-11-10

The user narrowed the Trev study to seed 0 only and requested cleanup and final results. Sent SIGINT to the active Chronos seed 1 random runner and stopped the seed 1 checkpoint watcher; the queue and both tmux sessions exited. Latest logged training progress was step 2445/4000, with no .complete or formal evaluation result. Removed six seed 1-3 Chronos input/output directories, including partial seed 1 optimizer and adapters, all 18 seed 1-3 configs, and the multi-seed queue script. Restored the checkpoint cleanup watcher to its seed 0 implementation. Seed 0 model checkpoints, evaluations, raw logs, and experiment history were retained. Data-disk free space increased from 211 GiB to 218 GiB. Final MASE-only results are in docs/results/v1-rage-trev-seed0-final.md.

## 2026-11-10 v0-crb-reconst-fine preparation

Last modified: 2026-11-10. Read current experiment, orchestration, result, and CRB protocol records. Confirmed prior v0-crb-reconst-test covered similar full-parameter gradient reconstruction with a different q grid and incomplete ten-track search; this new run targets all trainable LoRA coordinates on one fixed-seed random M5 trajectory. GPU was idle with 0 MiB used; data disk had 218 GiB available and cgroup memory use was about 12 GiB of 62 GiB. Added fixed config and resumable entry script, using the existing rage Conda environment and source bundle. Existing unrelated modified files were left intact. Next: start tmux run, inspect gradients and 16 matched result rows, and report actual outcomes.

## 2026-11-10 v0-crb-reconst-fine completion

Last modified: 2026-11-10. Ran the experiment in tmux v0-crb-reconst-fine using the rage Conda environment. Selected M5 manifest ordinal 771 (FOODS_1_079_CA_3), generated 1,068 full LoRA gradients (98,304 coordinates), and completed eight equal-budget Uniform/CRB comparisons. The output .complete and gradients.complete markers exist. Verified finite gradient cache shape, exact matched anchor counts and endpoints in all 16 rows, and exact recomputation of gradient rows 0,128,224,1067. A diagnostic invocation briefly overlapped the tmux writer and failed on a shared temporary progress filename; the primary writer continued, finished, and overlap spot checks were exact. No other experiment files or runs were changed. CRB mean L2 wins at q=2,4,8,128; Uniform wins at q=26,32,64; q=1 ties. Updated docs/EXP.md, docs/ORCHE.md, docs/RESULTS.md, and docs/results/v0-crb-reconst-fine.md. No automatic patrol was configured.

## 2026-11-10 v0-crb-reconst-fine twelve-track extension preparation

Last modified: 2026-11-10. User requested 12 total trajectories and authorized release of the current dense cache. Confirmed GPU idle, data disk 217 GiB free, and prior track 771 results and spot checks complete. Froze 11 additional M5 manifest ordinals using seed 20261110 without replacement. Extended the entry to accept a frozen ordinal and added a sequential, resumable queue that validates per-track metrics and gradient shape before releasing only dense LoRA gradient files. Existing prior experiment results, model checkpoint, and unrelated modified files remain untouched. Next: run queue, verify 12-track aggregation, update result report.

## 2026-11-10 v0-crb-reconst-fine twelve-track completion

Last modified: 2026-11-10. The sequential tmux queue finished at 17:38 CST after 17.0 minutes. Twelve of twelve fixed M5 FOODS_1 manifest trajectories produced 16 finite matched-budget rows each, for 192 total rows. Every trajectory was validated for exact K, unique in-range anchors, endpoints, and finite metrics. Each dense LoRA gradient cache was released after result validation, including the original ordinal 771; no lora_gradients.npy remains. twelve_tracks.complete, twelve_track_metrics.csv, and twelve_track_summary.json were verified. Data disk had about 218 GiB free and GPU was idle after completion. Equal-track mean L2 favors CRB only at q=2; Uniform at q=4 through 128. Updated experiment, orchestration, result index, and detailed result report. No automatic patrol was configured.

## 2026-11-10 v0-crb-reconst-fine cosine-only reanalysis

Last modified: 2026-11-10. User changed the decision metric to direction cosine and asked to disregard L2 for this comparison. Recomputed equal-track means and paired track win counts from the retained 192-row CSV; no model or gradient work occurred. Uniform mean cosine is higher at all seven nontrivial q values, with the q=2 comparison close and split 6/6 by trajectory. Updated the detailed result report and result index. Dense gradient caches remain released.

## 2026-11-10 v0-crb-cos-var preparation

Last modified: 2026-11-10. User requested direction dispersion of within-interval LoRA gradients projected to 4096d, using q={1,2,4,8,16,32,64,128}; clarified that blocks are between true gradient computation points. Protocol now excludes anchor endpoints and compares strict-interior projected gradients to the direction of their block mean; q=1 has no interior and is undefined. Reuses the fixed 12 M5 FOODS_1 tracks, warmed ChronosTiny checkpoint, and fresh LoRA initialization from the preceding study. Uses a fixed-seed ParameterHashProjector. The queue computes on demand without dense cache writes, with at most two concurrent track processes. Next: start queue and monitor GPU and cgroup memory; reduce concurrency if thresholds in ORCHE are crossed.

## 2026-11-10 v0-crb-cos-var first-four-track inspection

Last modified: 2026-11-10. Inspected completed ordinals 771, 835, 633, and 220 while the remaining queue continued. Equal-track mean direction dispersion was undefined at q=1, zero by construction at q=2 (one interior vector per interval), then 0.095114, 0.168161, 0.243842, 0.324056, 0.421194, and 0.531570 at q=4,8,16,32,64,128. Ordinal 771 is substantially lower than the other three at every q>=4. Queue active; keep monitoring memory and remaining tracks.

## 2026-11-10 v0-crb-cos-var protocol correction

Last modified: 2026-11-10. User clarified that the target is dispersion from Uniform reconstruction versus CRB reconstruction. Stopped the initial uniform-only diagnostic queue after ten tracks; its small projected-only outputs are preserved separately under outputs/train/v0-crb-cos-var-uniform-preview and excluded. The corrected protocol uses equal K=max(2,ceil(N/q)) anchors for both methods, CRB equal-stage allocation, and each interval endpoint-mean projected gradient as the reconstruction representative. Dispersion compares strict-interior true projected gradients with that representative direction. Replaced the single-track worker and queue aggregation to emit paired method rows. Next: run the corrected twelve-track comparison, with concurrency capped at two and memory monitoring.

## 2026-11-10 v0-crb-cos-var corrected first-four-track inspection

Last modified: 2026-11-10. The corrected Uniform versus matched-budget CRB runs completed ordinals 771, 835, 633, and 220. Equal-track mean of per-block direction dispersion (1-cosine) at q=2,4,8,16,32,64,128 was Uniform 0.164242/0.239985/0.325897/0.425628/0.522638/0.668560/0.772152 and CRB 0.161911/0.238646/0.316256/0.403336/0.516740/0.632380/0.752009. At q=1 both have no strict-interior points. CRB is lower on the equal-track mean at all tested q>1; the differences at q=2,4,32 are small. Queue continues with remaining tracks under two-process cap.

## 2026-11-10 v0-crb-cos-var corrected comparison completion

Last modified: 2026-11-10. The corrected two-process queue completed at 18:23 CST on all 12 fixed tracks. It generated 192 track-q-method summary rows and per-block Uniform/CRB metrics, with exact matched budgets and CRB stage allocations validated by each worker. Aggregate .complete exists. No full LoRA-gradient or full-trajectory projected cache was written; peak observed GPU allocation was about 0.9 GiB and cgroup memory about 15 GiB of 62 GiB. Equal-block equal-track mean favors CRB at q=2,8,16,32,64,128 and Uniform at q=4; window-weighted mean favors Uniform at every q>1. Preserved the earlier uniform-only diagnostic under outputs/train/v0-crb-cos-var-uniform-preview and excluded it. Updated EXP, ORCHE, RESULTS, and the detailed report.

## 2026-11-10 Time-300B data removal

Last modified: 2026-11-10. At 2026-11-10 18:52:21 CST, user requested deletion of the unused Time-300B raw data and converted shard dataset. The v1 Trev protocol and full-pool configuration omit Time-300B. Before deletion: raw directory 51 files, 63,511,169,049 bytes; converted directory 116 files, 122,167,109,471 bytes; total 167 files and 185,678,278,520 bytes (about 173 GiB). No matching Time-300B or v1 Trev process was running. The two directory paths will be retained with deletion notes. No other datasets are in scope.
Deletion completed at 2026-11-10 18:52:21 CST. Both original directories now contain only DELETION_NOTE.txt. Disk status after deletion: /dev/md0        550G  160G  391G  30% /root/autodl-tmp.


### 2026-11-10 - v1-rage-crb-vs-uniform-timerXL-84m preflight and storage audit

- User confirmed a timing-only pilot on the first task, hospital_admissions_1D; no formal selection or training may run during the pilot. Fixed TimerXL LoRA q_proj/v_proj (r=8, alpha=16, dropout=0), adapter seed 0, shared 4096d Count-Sketch projection seed 42, gradient micro-batch 8 for timing, and projected-gradient cosine scoring.
- The FEV Arrow file exists locally at the data-disk-backed project path. Its task record uses local train split, 8 univariate series, and no test split was read for references.
- Storage audit: /root/autodl-tmp has 218 GiB free of 550 GiB (61% used). No active experiment/download process or tmux session was found. Potentially regenerable caches currently present: pip HTTP cache 4.3 GiB and conda package cache 338 MiB. The previously observed 2.44 GiB .incomplete file was under the Time-300B raw tree and was removed with that separately authorized cleanup; a follow-up find found no remaining .incomplete files. Preserved the 51 GiB completed v0-gradient-proxy-test output because its head-gradient proxy caches are documented retained research artifacts; source data, shard data, environments, model weights, and completed training outputs were not changed. No files were deleted.
- Created pilot config and one-off entry script. Next: run the timing-only pilot in tmux, record actual throughput and estimated full-ref scoring time, then stop for scope review before any formal selection.


### 2026-11-10 18:57 CST - TimerXL selection feasibility pilot completed

- Ran the hospital_admissions_1D timing-only pilot in tmux v1-rage-crb-vs-uniform-pilot. The first attempt exposed that TimerXLGradientEngine expects the unwrapped model output; with PEFT it raised AttributeError before computing gradients. Replaced that pilot-only call path with NativeProbe over the TimerXL adapter's native training_loss, preserving LoRA-only gradients and the fixed Count-Sketch. The rerun completed, wrote feasibility_metrics.json and .complete, and the tmux session was closed.
- The FEV local train split has 9,246 eligible 576-point reference windows across eight univariate series. Measured 32 ref gradients at 18.64 points/s and 32 source gradients at 22.64 points/s using gradient batch 8. The 4096d score timing used a 32x512 tile and measured 630.35 million pairs/s.
- At N=360,695,805 and q=32, K=11,271,744. Across 12 tasks and two methods this is 270,521,856 source-gradient points, estimated at 138.3 single-GPU days at the measured source rate, excluding I/O, reference gradients, training, and evaluation. One complete hospital_admissions_1D ref-src sweep has 3,334,993,413,030 pairs and extrapolates to 1.47 hours before chunking overhead.
- Pilot scope only: no formal source selection, no selected training set, and no training updates. GPU is idle. The exact full protocol remains unstarted pending user review of the measured compute cost.
- Storage review recorded 218 GiB free before the pilot and 391 GiB free after the FEV load; no files were deleted by this work. The df readings changed from 218 GiB free during the audit to 391 GiB after the pilot. Existing docs record a separate authorized Time-300B raw/shard cleanup of about 173 GiB at 18:52 CST; both data paths now contain only deletion notes, which explains the increase. Recheck df before writing large gradient caches. The pip and conda caches remain untouched.


### 2026-11-10 - BF16 gradient optimization pilot prepared

- User requested trying acceleration options before revising the full-run estimate. Existing heavy TimerXL forward/backward was already BF16; the first throughput pilot used batch size 8 but the PEFT-compatible generic NativeProbe evaluated each window loss in a separate forward.
- Added a timing-only PEFT TimerXL probe with a single batched transformer forward and per-window shifted-patch losses, checked against scalar adapter losses at runtime. Configured batch sizes 8/16/32 and VJP chunks 4/8/16; added one `torch.compile(mode=reduce-overhead)` attempt at batch 32/VJP 16. Base and LoRA remain BF16; fixed Count-Sketch stays 4096d/seed 42 with FP32 projection accumulation.
- No formal selection, gradient cache, or training output will be written. Compile caches are directed to `/root/autodl-tmp/rage/cache/torchinductor` and `/root/autodl-tmp/rage/cache/triton`.
- Next: run one-task timing pilot, record speed and peak GPU memory for every configuration, then recompute the single-GPU estimate.


### 2026-11-10 19:39 CST - BF16 TimerXL gradient optimization timings completed

- Corrected the one-off PEFT probe to reproduce the checkpoint's masked per-head reduction and sequential head accumulation. The optimized per-window loss matched the scalar adapter loss exactly on the 8-window guard (maximum absolute difference 0.0).
- Swept eager batch sizes 8/16/32 and VJP chunks 4/8/16 on 96 source samples. Best was batch 32 / VJP 16 at 437.73 gradients/s (0.299 GiB allocated peak). The initial scalar-forward rate of 22.64/s is not the optimized rate.
- Tried `torch.compile(mode=reduce-overhead)` and `mode=default`; both failed during batched-VJP backward with `Cannot access data pointer of Tensor that doesn't have storage`. No compiled result is used.
- Repeated on 1,024 source samples with batch 32/64/128 and VJP 16/32/64. Best was batch 32 / VJP 32 at 624.38/s, 0.321 GiB peak; batch 32 / VJP 64 reached 620.35/s. Larger batches were slower. Wrote `optimization_metrics.json` and `.complete` under `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/optimization-stable/`.
- Updated source-gradient-only estimate: 270,521,856 points / 624.38 points/s = 5.01 days on one GPU. This excludes reference gradients, full ref-src scoring, I/O/cache writes, training, and evaluation. No formal selection or training ran. The tmux session exited and GPU is idle.


### 2026-11-10 19:46 CST - Sustained TimerXL gradient throughput measured

- Tested VJP chunk 128 on 1,024 sampled source windows; it did not improve the preferred batch 32 / VJP 32 setting.
- Repeated batch 32 / VJP 32 with 8,192 sampled source windows and lower progress-log frequency. Two runs measured 596.82 and 575.35 gradients/s; both used 0.321 GiB peak allocated GPU memory. The earlier 4,096-window single run measured 780.61/s but was not repeated and is excluded as an outlier for planning.
- Use the slower sustained rate, 575.35/s, for a conservative source-gradient estimate: 270,521,856 points / 575.35/s = 5.44 single-GPU days. This is roughly 25x faster than the initial 22.64/s scalar-forward baseline. Estimate excludes reference gradients, full score scans, I/O/cache writes, training, and evaluation.
- Results and `.complete` markers are in `optimization-vjp128/`, `optimization-steady-8192/`, and `optimization-steady-repeat/` under the one-task pilot output. The tmux session exited and the GPU is idle. Formal selection/training remain unstarted.


### 2026-11-10 20:10 CST - Same-GPU concurrent gradient throughput measured

- Verified same-GPU process parallelism instead of assuming the low per-process allocation prevented scaling. Each process used batch 32 / VJP 32 on 4,096 sampled source windows.
- Two processes measured 482.86 and 481.30 points/s each (964 aggregate; 1.68x the sustained single-process baseline). Four measured 1,087 aggregate (1.89x). Eight measured 1,159 aggregate (2.01x); seven workers were near 150 points/s each and one was 106.92/s. Eight added about 6.6% over four.
- Per-process PyTorch peak allocated memory was 0.321 GiB. At eight processes, `nvidia-smi` showed about 5.7 GiB total GPU memory and 100% utilization. Eight-model initialization took about 1.4 minutes with substantial CPU use; account for this in per-phase worker restarts.
- Tentative source-gradient-only projection at 1,159 aggregate points/s is 2.70 days for 270,521,856 points. This excludes worker startup, data sharding/cache synchronization, I/O, reference gradients, score computation, training, and evaluation. Parallel pilot outputs and `.complete` markers are under the `parallel-two/`, `parallel-four/`, and `parallel-eight/` output directories. All tmux sessions exited; GPU is idle. No formal selection or training ran.


### 2026-11-10 20:25 CST - v00 computational effort estimate assembled

- Established a deterministic random seed-0 sample of 48,000 trajectories from the current 199,030-series manifest: 114,772,172 source points, 87,172,172 legal 576-point windows, about 0.428 GiB float32 if copied. The sample itself has not been materialized.
- For q=32 and 12 task-specific selections x two methods, this implies K=2,724,131 anchors per method-task and 65,379,144 source-gradient points per model. TimerXL estimates: 31.6 h single-process, 15.7 h using the measured 8-worker aggregate rate (excluding refs/scoring/I/O/train/eval).
- TimesFM 2.5 timing-only BF16 LoRA pilot (batch 32/VJP 32) matched scalar per-window loss exactly and measured 23.436 gradients/s single-process, peak allocated 2.993 GiB. Two concurrent workers measured 13.835 and 13.992/s, 27.827/s aggregate, about 1.19x; GPU reached 100% utilization. Projected source-gradient time: 774.9 h single-process or 652.6 h with two workers for 24 method-task passes.
- Chronos-2 was not measured: no local weights/package and no project LoRA training adapter. The official 120M parameter count was used only for a low-confidence proportional placeholder versus TimesFM 231.3M: 402 h single-process, 338.6 h assuming the same two-worker speedup. Do not schedule from this estimate before integration and a timing pilot.
- Storage audit: `/root/autodl-tmp` has 391 GiB free of 550 GiB. 4096d float32 projected vectors are 16 KiB each; one method-task cache is 41.57 GiB and 24 retained caches are 997.6 GiB. Materializing every 10% selected window for 24 outputs would take about 449 GiB; int64 window indices are about 1.56 GiB. Stream/release caches and persist indices. No data were deleted.
- Added independent v00 experiment and orchestration entries to `docs/EXP.md` and `docs/ORCHE.md`. Pilot artifacts are under `outputs/train/v00-computational-effort-estimate/`; logs are under `logs/v00-computational-effort-estimate/`. All pilots exited and GPU is idle; no formal selection/training is running.


### 2026-11-10 20:36 CST - Chronos-T5-Base scope correction and four-process timing

The user clarified that the Chronos model is Chronos-T5-Base, alongside TimesFM 2.5. Replaced the prior Chronos-2 planning placeholder in the active v00 experiment and orchestration documentation. The earlier Chronos-2 placeholder remains only in this historical log for traceability and is superseded.

Measured the Chronos-T5-Base BF16 LoRA probe with the exact one-window path (batch 1/VJP 1) because batched BF16 loss values differed. Single process: 7.396 gradients/s, 0.831 GiB peak allocated, estimated 2,455 h for 65,379,144 source gradients. Two processes: 6.968 and 7.799/s, 14.766/s aggregate, estimated 1,229 h. Four processes: 4.483, 4.513, 4.516, and 4.540/s, 18.052/s aggregate, estimated 1,005 h. Four-worker nvidia-smi memory was about 16.2 GiB; GPU returned to idle after completion. The estimate covers only source gradients for 12 task-specific selections x two methods; it excludes reference gradients, scoring, cache writes, training, and evaluation.

The random 48k seed-0 pool remains unmaterialized. No experiment data were deleted; the timing probes remain under outputs/train/v00-computational-effort-estimate/ with logs under logs/v00-computational-effort-estimate/. Updated docs/EXP.md and docs/ORCHE.md. No formal selection or training was started.


### 2026-11-10 20:45 CST - v00 target size changed to 256k x 576 values

The user fixed the planned retained data scale at 256,000 x 576 = 147,456,000 scalar time-series values and asked for an approximate number of randomly selected trajectories. The full eligible pool has 199,030 trajectories and 475,138,055 values, averaging 2,387.27 values per trajectory. The target therefore corresponds to about 61,768 trajectories; use about 61.8k (roughly 62k) as the planning count. The final seed-0 sample should shuffle eligible trajectories and stop at the prefix whose cumulative values are closest to the target. No sample manifest or data pool was generated.

Using mean trajectory length, estimate about 111.94M legal 576-point windows, 3,498,113 anchors per method-task at q=32, and about 83.95M source-gradient computations per model across 12 tasks x two methods. Scaled timing estimates: TimerXL 40.5 h single / 20.1 h at the measured 8-process aggregate; TimesFM 995 h single / 838 h at 2 processes; Chronos-T5-Base 3,153 h single / 1,579 h at 2 processes / 1,292 h at 4 processes. These are source-gradient-only projections based on earlier short pilots. Updated projected storage and timing in EXP and ORCHE. The target sample and formal selection/training remain unstarted.


### 2026-11-10 20:48 CST - Clarified v00 window-count versus raw-point target

The prior quick estimate for 270k x 576 used 270k as the count of legal 576-point windows per task, yielding about 202.5k source-gradient computations across 12 tasks x two methods. The later fixed target, 256k x 576 scalar values, was interpreted as a raw-value budget for selecting complete eligible trajectories. Since the average selected trajectory is about 2,387 points, about 61.8k trajectories generate roughly 111.94M overlapping legal windows. At q=32 over 24 method-task passes this is about 83.95M gradient computations, roughly 415x the prior window-count workload. The short and long time estimates describe different workload definitions; the long estimates apply to retaining whole trajectories at the raw-value target. This distinction is recorded in EXP and ORCHE. No experiment was started.


### 2026-11-10 20:52 CST - Corrected task-shared gradient workload

The user clarified that source-gradient preparation is shared across tasks and should be counted once regardless of the number of tasks. Corrected active v00 estimates from 12 task repetitions to a single reusable gradient cache per method. Assuming Uniform and CRB require separate method caches, the 256k x 576 raw-value target needs about 6,996,226 gradients/model (2 x 3,498,113 anchors), not 83,954,712. Updated projections: TimerXL 3.38 h single / 1.68 h at measured 8-process aggregate; TimesFM 82.9 h single / 69.9 h at 2-process aggregate; Chronos-T5-Base 262.8 h single / 131.6 h at 2 processes / 107.7 h at 4 processes. These remain source-gradient-only estimates and assume the caches are reusable across all 12 tasks. The 512-trajectory example is about 928k legal windows, 28,997 anchors/method, and 57,994 total gradient computations for two method-specific caches: TimerXL about 1.7 min single / 50 sec at 8 processes; TimesFM about 41 min single / 35 min at 2 processes; Chronos-T5-Base about 2.18 h single / 65 min at 2 processes / 54 min at 4 processes. No data pool or formal selection was started.

## 2026-11-10 src-8192 source replacement

All prior v1-rage batches were marked deprecated due to unsuitable source selection. Historical result files were retained. The replacement pool selects exactly 8,192 trajectories balanced across available eligible configurations and records a fixed per-trajectory index in configs/data/src-8192.yaml. The old v1-rage training queue was not restarted.

2026-11-10 21:38 CST: Selected and converted src-8192 from existing downloaded source bundles. Verified 8,192 unique source/subset/series IDs, 19 configurations, 235,804,890 values, 8,193 offsets, matching trajectory lengths, and finite sampled values. Independent converted data are in data/src/shard_dataset/src-8192/ (about 1.8 GiB). Fixed index is configs/data/src-8192.yaml. No v1-rage training process was active.

2026-11-10 21:43-21:51 CST: Extended src-8192. Project Fetch downloaded 18 Chronos and six Monash configurations from Hugging Face with HF_HUB_DISABLE_XET=1 after an initial Xet 401 response. The first bounded Load attempts failed because BundleLoadWriter lacked the staging attribute used by SQLiteSeriesReservoir. Added staging to the project writer, checked and removed only empty failed bundle directories, then reran Load successfully for 20 configurations. Sixteen were eligible; four had no eligible length-576 series. Archived the original 19-configuration snapshot and YAML as version 1. Rebuilt version 2 with 8,192 trajectories across 35 configurations and 191,176,690 values. Verified all ordinals, unique source/subset/series identities, lengths, row metadata, and 100 exact source-versus-snapshot value samples.

2026-11-10 22:04 CST: Clarified that the TS-Lab code branch and upstream dataset main branches are separate. Updated the project Fetch adapters for Chronos, Monash, and UTSD to accept a full upstream commit ID as revision and record it in completion markers. Pinned single-config download probes passed for Chronos exchange_rate and Monash fred_md. Syntax checks and diff whitespace checks passed. The user will permanently upload the independent src-8192 snapshot, so no full pinned redownload or source-by-source verification was requested. The current 8,192-trajectory, 35-configuration snapshot remains authoritative; verified its value and offset counts.

2026-11-10 22:06 CST: Per user request, removed all remote project data other than data/src/shard_dataset/src-8192/. This removed 25 GiB of raw inputs and the other source bundles, including the archived prior src-8192 version. Before cleanup, removed stale load_dir and row path fields from the retained config/manifest; verified 8,192 trajectories, 35 configurations, 8,193 offsets, 191,176,690 values, and .complete. The data disk now has about 446 GiB free.

## 2026-11-10：计算量估算实验更名

将实验代号从 v00-computational-effort-estimate 改为 v1-computational-effort-estimate；同步更新 EXP、ORCHE、RESULTS 和独立结果报告的名称与链接。运行时产生的原始产物、日志及旧记录保留原路径，以维持证据可追溯性；本次未重跑计时或正式实验。

## 2026-11-11：旧梯度批次终止记录

`v0-grad-prepare-src-4096tr-q32-4096d` 因计算负担过大终止，未形成完整结果。按用户要求删除该批次全部正式结果、梯度准备缓存、日志和旧外存风险报告；保留这一条失败记录。独立的 src-4096 数据快照仍作为 src-96 的来源存在，不属于已删除的实验结果或梯度缓存。

## 2026-11-11：改为 src-96 q16 三模型梯度批次

按用户要求停止旧 src-4096 q32 四进程，旧正式产物随后按用户要求删除。根据 src-4096 的 35 配置固定分层规则选出 96 条完整轨迹，seed 0，生成 `configs/data/src-96.yaml` 与 `data/src/shard_dataset/src-96/`，逐条核对源值与偏移全部一致。q=16 时每方法 166,659 锚点，CRB 三预算 83,329/41,665/41,665；两方法合计上界 333,318，用户明确接受，不做长度替换。新增三模型统一入口与队列脚本；三模型各用一条短轨迹预演成功，验证 BF16 位编码、4096 维及预算 5/2/2，预演目录已清理。待正式启动。

2026-11-11 13:12 CST：用户确认保留 src-96 固定随机抽样及两方法合计 333,318 次梯度上界，不作长度调整。三模型短轨迹预演逐一通过后删除预演输出；在 tmux `v0-grad-src-96-q16` 启动正式顺序队列，首阶段 TimerXL 八进程已确认存活。后续 TimesFM 两进程、Chronos 四进程由同一队列自动接续。按 v1 计时纯梯度外推分别约 0.08、3.33、5.13 小时，合计 8.54 小时；旧 Chronos 正式流水线实际速率约 7.6 梯度/秒，其阶段若持续该速率则约 12.18 小时，三阶段合计至少约 15.6 小时。两种外推都未充分覆盖 CRB 分阶段调度、缓存写入与进程初始化，需用新批次实测修正。


## 2026-11-11：登记 v0-crb-mech-test

连接 ssh tsl 并检查 /root/TS-Lab。已阅读远端 AGENTS.md、docs/EXP.md、docs/ORCHE.md、docs/LOG.md、docs/RESULTS.md 和 CRB 相关脚本。GPU 当时由 TimerXL 八进程任务使用，利用率 100%；因此新实验规划为 CPU-only，不触碰其 GPU 作业。确认 Conda 环境 /root/autodl-tmp/rage/envs/rage 可用。新增入口 scripts/v0_crb_mech_test.py 与配置 configs/rage/v0-crb-mech-test.json，并登记实验、输出及调度方式。首次执行结果待记录。


## 2026-11-11：CRB 端点索引修正

首次启动在第一条轨迹的 CRB 端点断言处退出。原因：CRBResult.anchors 的键是合法起点数组的 0-based row index，而回调给出的 physical starts 是 1-based。修正脚本只对输出锚点读取 row index，不再减 1；回调仍按 physical start 转为轨迹下标。已重新启动同一 CPU-only 任务，继续验证全部预算。

2026-11-11 13:23 CST：再次确认新 src-96 队列仍运行，TimerXL 已完成 7/8 个 worker，最后一个 worker 正在处理长轨迹。清理旧 4096/q32 批次的输出、缓存和日志，删除旧外存风险报告；数据盘可用空间由 442 GiB 增至 443 GiB。同步更新 EXP、ORCHE、RESULTS，使旧批次仅保留失败记录；新批次状态改为正式运行中。

2026-11-11 13:23 CST：用户要求实验文档保持中文，并要求旧批次只留计算用时估计，不保留梯度计算结果。已将旧批次在 EXP 总表标为失败终止，清除旧结果报告与缓存引用；RESULTS 仅保留 v1 计算用时估计的链接和新批次运行状态。为减少后续尾部 GPU 空闲，修改新入口，使 TimesFM/Chronos worker 按每轨迹 q=16 锚点预算确定性均衡分配；TimerXL 当前运行不受修改影响。

2026-11-11 13:28 CST：TimerXL 阶段八个 worker 的 `.complete` 与模型 `.complete` 均存在，96 条 Uniform/CRB 清单一一对应，各方法预算合计 166,659，缓存分片确认为 4096 维 BF16 位编码，模型输出约 2.5 GiB。阶段实际运行约 15 分钟，高于 v1 短时纯梯度估算 4.8 分钟。TimesFM 两进程自动启动，GPU 占用约 8.1 GiB、利用率 100%。


## 2026-11-11 13:30 CST：v0-crb-mech-test 完成

修正 CRB anchor row index 映射后，在 tmux v0-crb-mech-test 使用 Conda rage 环境完成 CPU-only 回放；64/64 轨迹、q=4/8/16/32，输出 512 条 method-trajectory 指标并核实 .complete。实际用时约 7.6 分钟，GPU 原任务未受影响。结果：四个 q 的平均 L2 与方向余弦误差均为 Uniform 较低；q=32 时 64/64 条轨迹 Uniform 两项都较低。粗糙度差值 Spearman 已写入独立结果报告。更新 docs/EXP.md、docs/ORCHE.md、docs/RESULTS.md 和 docs/results/v0-crb-mech-test.md。完整指标与逐轨迹锚点位于 outputs/train/v0-crb-mech-test/。实验批次结束，无自动巡检设置。


## 2026-11-11：启动极平缓轨迹补充

用户提出再试一条极平缓轨迹。扩展入口支持配置正弦频率范围及可选轨迹范数归一化；新增独立配置 v0-crb-mech-test-ultra-smooth，单条 4096 维、4000 点，频率 0.005–0.05 周期/归一化时间，seed 20261112，归一化平均向量范数为 1；按原 CRB 预算和四个 q 同预算 Uniform 对比。独立输出路径，不覆盖原 64 条实验。结果待回填。


## 2026-11-11 13:52 CST：极平缓轨迹补充完成

单轨迹回放完成，运行约 7.3 秒。经验粗糙度分数 1.8373e-7，约为原 64 条实验最平缓轨迹的 1/246。q=4/8/16/32 各有 CRB 和 Uniform 结果，共 8 行并核实 .complete。L2 四个预算都由 Uniform 略低；方向余弦误差约 1e-8–1e-7，q=16、32 的 CRB 差异接近浮点误差尺度。已追加到 docs/results/v0-crb-mech-test.md，更新 EXP 和 ORCHE 状态。输出独立保存在 outputs/train/v0-crb-mech-test-ultra-smooth/。没有进行统计相关性推断。


## 2026-11-11：启动 mock 轨迹块内方差分析

按用户提出的分析方向，新增 scripts/v0_crb_mech_block_variance.py。沿用已有锚点，按相邻锚点形成互不重叠、完整覆盖轨迹的块；以块内向量均值为代表，计算 mean squared vector L2 deviation、归一化向量方差、相对块均值方向的 mean 1-cos，以及向量模长方差。将原 64 条轨迹和极平缓单轨迹分别分析，结果写入 outputs/train/v0-crb-mech-test/block_variance/，不重新生成或覆盖原始锚点。任务运行中。


## 2026-11-11 14:09 CST：mock 块内方差分析完成

脚本基于已保存锚点和确定性种子重建 mock vectors，未重跑模型或改变锚点。对 64 条原始轨迹及极平缓单轨迹，按相邻锚点构建不重叠的全覆盖块，计算块内向量均值周围的平均平方 L2 偏差、相对向量方差、相对块均值方向的平均 1-cos、模长方差。原轨迹结果 7168 块，极平缓结果 112 块，两个 .complete 均验证。64 轨迹等权结果显示 q=32 时 CRB 的相对方差和方向方差均在 64/64 轨迹低于 Uniform；绝对模长方差不是同一趋势。更新本结果报告、EXP 和 ORCHE；输出在 outputs/train/v0-crb-mech-test/block_variance/。


## 2026-11-11：启动 CES 曲率采样对比

用户要求在原 64 条 mock 轨迹比较 CES、Uniform、CRB。按未指定拆分比例的情形固定 50% 均匀 pilot、50% adaptive；q=4 受三点曲率估计所需条件约束，pilot 至少 3。Pilot 含端点；非均匀三点差分估计向量曲率；第二阶段按区间长度乘 (curvature^2+epsilon^2)^(1/5) 分配锚点，区间内等距放置。CRB/Uniform 使用原始匹配锚点；所有比较在原 4096 维计算。新增脚本和固定配置，开始 CPU-only 回放，结果待核实。


## 2026-11-11 14:49 CST：CES 与块内指标分析完成

CES 对比在 64 条轨迹、q=4/8/16/32 全部完成，输出 768 条 CRB/Uniform/CES 方法记录并验证锚点数、端点与 .complete。另对三种方法共 10,752 个块计算块内 L2 方差、相对方差、方向方差和模长方差，输出 .complete。全轨迹平均重建误差上，CES 每个 q 均优于 CRB，接近 Uniform 但 L2 仍略高；q=32 L2 在 64/64 轨迹低于 CRB。块内方面，CRB 在 q=8–32 的平均相对方差和方向方差最低，CES 接近 Uniform；q=4 时 CES 略低于两者。结果及局限写入 docs/results/v0-crb-mech-test.md；EXP、ORCHE 状态同步完成。CES 输出为 outputs/train/v0-crb-mech-test/ces/，三方法块统计为 outputs/train/v0-crb-mech-test/block_variance/ces-comparison/。


## 2026-11-11：启动 CES-Oracle

用户要求使用真实曲率验证 CES。对已知的 mock 正弦生成器解析计算 g''(t)，以其完整 4096 维范数作为 oracle 曲率；保留 CES 相同均匀 pilot 集合、50% pilot 预算（q=4 最少 3 个），自适应预算按各 pilot 区间真实曲率密度 (||g''||^2+epsilon^2)^(1/5) 的积分作最大余数分配，区间内使用真实曲率密度 CDF 分位点放置。入口 scripts/v0_crb_mech_ces_oracle.py，独立配置和输出；与前序 CES/Uniform/CRB 在同一全维 mock 轨迹上比较。运行中。


2026-11-11 14:59 CST：CES-Oracle 已完成 64/64 条轨迹，q={4,8,16,32}，共 1024 条四方法记录；解析真实曲率用于锚点分配，完整轨迹误差仍在 4096 维评估。开始四方法块内方差分析。


## 2026-11-11 15:03 CST：CES-Oracle 与四方法块分析完成

解析真实 g'' 的 CES-Oracle 完成 64 条、q=4/8/16/32，1024 条四方法指标记录。核实 oracle 轨迹生成器与原 make_track 的最大绝对差为 4.47e-8，锚点预算与两端点均有效。Oracle 平均 L2 四个 q 都低于 CRB，但没有优于 Uniform 或 CES-estimate；q=8 与 CES-estimate 的 L2 几乎相同，其他 q 略高。随后完成 CES-Oracle/CRB/Uniform/CES 四方法块分析，14,336 个块记录及 .complete 验证。真实曲率密度并未在当前 finite-budget mock 结果中带来一致提升；结果边界和指标写入 docs/results/v0-crb-mech-test.md，EXP 与 ORCHE 更新完成。


## 2026-11-11：启动 Uniform 多重建器比较

用户要求按方案 B 测试 Uniform 锚点配不同重建器。固定使用原 64 条轨迹与 q={4,8,16,32} 的 Uniform 锚点，增加自然三次样条、PCHIP、q-1 次 Chebyshev 全局多项式、RBF-GP（相邻间距倍数 1/2、alpha=1e-6），及固定 lambda={1e-8,1e-6,1e-4} 平滑样条敏感性组。SmoothingSpline q=4 按 SciPy 至少 5 点要求跳过。保留 Uniform/CRB/CES/CES-Oracle 线性重建基线；所有方法用完整 4096 维逐点误差评估。平滑强度固定为敏感性网格，不按测试轨迹最优值挑选。远端环境 SciPy 1.15.3；该版本 make_smoothing_spline 为单输出，使用固定 lambda 的单位基响应构造线性重建算子供全部维度共享。任务开始，结果待核验。


## 2026-11-11 15:33 CST：Uniform 多重建器比较完成

64/64 轨迹完成，共 2,880 条方法-轨迹记录；L2、RMSE、相对 L2、方向余弦误差均为有限值，.complete 已核实。比较包括 Uniform linear、Natural CubicSpline、PCHIP、Chebyshev q-1 次全局多项式、RBF-GP 两个固定长度尺度、SmoothingSpline 三个固定 lambda；CRB/CES/CES-Oracle linear 为参照。q=4 不运行 smoothing spline（最少需要 5 锚点）。结果：q=4/8/16 线性 Uniform 平均 L2 最低；q=32 Natural Cubic 与 lambda=1e-8 平滑样条略低，PCHIP 方向余弦误差最低。Chebyshev q=16/32 误差严重放大；固定 GP 长度尺度均未胜 Uniform linear。平滑样条 lambda 未经独立验证调优，结果按敏感性扫描报告。完整汇总写入 docs/results/v0-crb-mech-test.md；更新 EXP/ORCHE。输出在 outputs/train/v0-crb-mech-test/uniform-reconstructors/。


2026-11-11 16:07 CST：Uniform + Cancellation Split sweep 完成。64 条轨迹、q={4,8,16,32}、7 个 tau、两种切分规则；trajectory_scores.csv 与 interval_scores.csv 各 50,176 行，summary.json 和 .complete 均生成。首次预运行发现参考权重分母把 3999 个锚点跨度误写为 4000 个采样点；改为按 L-1 归一化后完成。结果显示 shape score 变化幅度小且方向不一致，详细结果与解释写入 docs/results/v0-crb-mech-test.md。输出 outputs/train/v0-crb-mech-test/cancellation-split/；入口 scripts/v0_crb_mech_cancellation_split.py，配置 configs/rage/v0-crb-mech-cancellation-split.json。


2026-11-11 16:15 CST：按用户明确优先级，新增块内代表性评估，比较 midpoint/minimum_norm 的虚拟子块与同预算未切分 Uniform 块。初版直接逐块扫描全部向量点，ETA 约 13 分钟；终止该计算并改为每轨迹向量和/单位方向和前缀统计，指标公式不变。与既有 block_metrics 在随机单点、首尾和一般区间的最大数值差约 5.5e-13。优化后 64/64 轨迹完成，用时约 1.4 分钟；3,584 条轨迹级、58,288 条块级记录和 .complete 均核实。tau=0.9 时 q=4/8 方差下降较明显，q=16 较弱，q=32 很弱；midpoint 与 minimum_norm 没有稳定胜负。数据及解释补入 docs/results/v0-crb-mech-test.md。


2026-11-11 16:22 CST：按用户澄清重开严格 Uniform+Linear 六方法实验。固定原 64 条轨迹和 Uniform 锚点；采用 kappa<=tau 触发 tau-mid、tau-rev；每轨迹/q/tau 随机抽取相同 K 区间，分别以 midpoint 和各区间 minimum-norm reversal 构成 tau-mid-base、tau-rev-base；另运行 all-mid、all-rev。所有真实锚点与 Linear 重建保持不变。固定 tau={0,0.2,0.4,0.6,0.8,0.9,1.0}，随机基线 seed offset=918273。入口与配置已写入，远端 tmux v0-crb-mech-cancellation-six-methods 已启动，块方差结果待核验。


2026-11-11 16:24 CST：六方法 Uniform+Linear 对照完成。64 条轨迹、q={4,8,16,32}、tau={0,0.2,0.4,0.6,0.8,0.9,1.0}；严格验证每个随机基线的 split K 与对应 tau 方法完全匹配，tau-mid-base/tau-rev-base 共用随机区间集合。7,680 条轨迹级、107,520 条区间记录及 .complete 均核验；每个分区按4000点精确覆盖。tau=0.9 时 q=8 相对 matched random 的等块权 L2/方向方差降幅约2.2–2.6%/4.2–4.3%；q=4有约0.5–1.3%等块权优势；q=16/32接近零。按点加权 q=8降幅更大，但 q=4改善不稳定。all-mid/all-rev 的方差明显降低，主要因为所有块都拆成两块。逐点 Uniform+Linear 重建保持不变。结果写入 docs/results/v0-crb-mech-test.md；输出 outputs/train/v0-crb-mech-test/uniform-linear-six-methods/。


2026-11-11 16:38 CST：用户指出上一批真实块均值对真实块值属于 real-real 方差估计，虽然分割只依赖锚点，但该 oracle 代表无法支持估计能力结论。明确将早先块方差批次标记为 real-real oracle 诊断。已启动 est-real 复评：复用六方法 anchor-only 分割文件；每块代表由原 Uniform+Linear 在块内各点重建值取均值得到，真实 mock 值只用于评估 L2、relative L2 与 direction error。未新增锚点，逐点重建不变。输出单独放入 outputs/train/v0-crb-mech-test/uniform-linear-six-methods/est-real/；tmux v0-crb-mech-cancellation-est-real 运行中。


2026-11-11 16:40 CST：est-real 六方法评估完成。每块估计代表严格取原 Uniform+Linear 在块内所有离散点的重建向量均值；真实轨迹只用于评估该代表的 true L2、relative L2 和 direction error。64 条轨迹、六方法、q={4,8,16,32}、全部 tau/all 组合共 7,680 条轨迹级和 130,912 条块级记录；.complete 已核验。tau=0.9 相对未分割 Uniform 的等块权/点权结果及与 matched-K random 对照已写入结果报告。按点权时 cancellation 选区在各 q 均优于随机基线；按等块权 q=4/8/16 的均值反而较差，q=32差异很小。前一批真实均值评估明确标记为 real-real oracle 诊断，不能代替本 est-real 结论。


2026-11-11：启动 Uniform Gradient Reconstruction 最优虚拟分块策略实验。固定原 64 条轨迹、4096D、Uniform anchors 与 Linear 重建；每个 q 的 K=0..q-1，midpoint split，不新增梯度。实现 Random-K（每 track/q/K 对应100个随机重复）、Cancellation-K、Endpoint-Difference-K、Predicted-Gain-K、L2-Oracle-K、Direction-Oracle-K。est-real 块代表是锚点线性重建在块内逐点均值，真值仅用于离线评价、Oracle 和区间收益相关分析。主指标 point-weighted，equal-block 为辅助；paired bootstrap 95% CI、效率、q/pooled Spearman、8张曲线图。运行目录 outputs/train/v0-crb-mech-test/virtual-split-selection/，入口和配置已登记。2026-11-11 17:07 CST 已在 tmux v0-crb-mech-virtual-split-selection 启动，使用远端 Conda rage 与 CPU 两线程；GPU 当时被其他进程占用，因此本批不使用 GPU。启动前修正 point-weighted K 曲线从总误差转为逐点均值的归一化，并重建完整唯一的轨迹级曲线记录；语法编译通过。


2026-11-11 17:16 CST：K 曲线第一遍完成后复核发现，等块权辅助曲线此前沿用了点加权 Oracle 区间排序，故不能作为等块权上限。第一遍完整产物保留在 outputs/train/v0-crb-mech-test/virtual-split-selection-pass1-pre-equal-oracle-fix/ 供审计，不纳入最终结论。已改为点权、等块权各自按对应单区间真实误差收益排序 Oracle，并把等块权 L2/方向实际收益写入区间明细；代码语法编译通过。17:16 CST 在同一 tmux 会话重跑，完成标记待新一轮结束后核验。


2026-11-11 17:21 CST：第二遍按指标和汇总权重分别排序 Oracle 后完成；tmux 进程正常退出，outputs/train/v0-crb-mech-test/virtual-split-selection/.complete 存在。核验 8 张图、curve_summary 1200 行、trajectory_curves 46,080 行、interval_prediction 3,584 行；Random 张量每个 q 均为 64×100×q×2，paired bootstrap=5,000。修正后点权 600 条汇总完全匹配第一遍结果；K=0 和 K=q−1 时所有方法误差分别相同；等块权 Oracle selection efficiency 均不超过 1。K=1 点权 Cancellation 相对 Random 的 L2/Direction 差值在四个 q 的配对95%CI均低于0。L2区间预测 pooled Spearman：Cancellation/Endpoint/Predicted=0.817/0.972/0.985；Direction=0.959/0.888/0.897。完整指标、误差/收益/效率/CI、八图索引和Q1-Q3结论已写入 docs/results/v0-crb-mech-test.md；docs/EXP.md、docs/ORCHE.md、docs/RESULTS.md 均已更新为完成状态。


2026-11-11 17:30 CST：按用户新增任务，为已完成 K-sweep 补充 MSE 方差分解。固定原始 Uniform anchors + Linear 重建、64 条轨迹、同一随机排列、所有方法/q/K；每块分别算 true block mean、估计块均值、representation SSE、estimation SSE，并与原 est-real total SSE 交叉核对。新增入口 scripts/v0_crb_mech_virtual_split_mse_decomposition.py，通过语法编译；输出子目录 outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/。point-weighted 与 trajectory-equal 两套汇总均保留，因轨迹长度一致预期相同。17:30 CST 已启动 tmux v0-crb-mech-mse-decomposition，CPU 两线程；等待 64 条轨迹/q 全部完成后核对恒等式并补文档。


2026-11-11 17:36 CST：MSE decomposition 子批次完成。检查 .complete 与 12 张图；summary 1080 行、trajectory 23,040 行、interval 3,584 行。逐轨迹 max |Total−Representation−Estimation|=8.99e-15；分块与既有误差函数交叉核对 max diff=5.46e-12，汇总曲线与原 est-real total L2 max diff=4.44e-16；point-weighted 与 trajectory-equal max difference=0。所有 q/K/6 种策略覆盖，Random 保留 100 次重复，paired bootstrap 95% CI 与 paired Random 差值加入汇总。实际块数作为图横轴，block_count 写入曲线 CSV。结果报告新增 K=1 全策略分量变化和 K=0/全切分比较；所有 q 中 K=1 的 Representation MSE 降低、Estimation MSE 上升，Total 净下降。


2026-11-11：启动批次 `64-large-var-k-error-composition`。按用户澄清，large-var mock 的约束为直接相邻点完整向量差比 `||g[t+1]-g[t]||_2 / ((||g[t]||_2+||g[t+1]||_2)/2)≈0.5`，并要求 lag-16 两点方向余弦接近 0；幅度用宽跨度平滑包络调制。生成器改为长度 8 的滚动随机方向和并乘平滑包络，写出逐轨迹诊断 CSV/summary。远程 tmux `v0-crb-mech-64-large-var` 串行排队：mock 生成、K-error sweep、MSE decomposition；q={4,8,16,32,64,128}，Random repeats=100、paired bootstrap=5000。计算全部在远程 Conda rage 环境、CPU 两线程；数据路径在项目 `data/` 软链接所指 AutoDL 数据盘，输出在本批 canonical 目录。用户要求启动后不巡检。


2026-11-11 18:30 CST：批次 `64-large-var-k-error-composition` 已完成；tmux 会话退出。64×4000×4096 float32 mock 生成完成，约束实测：相邻点完整向量差比轨迹均值 0.50005，lag-16 cosine 平均 0.0000375/abs mean 0.01245，梯度模长 CV 0.3225。K-error 完成 q={4,8,16,32,64,128}、K 全 sweep，12 张图、curve summary 5,040 行、track curve 193,536 行、interval 15,744 行；MSE decomposition 18 张图、summary 4,536 行，恒等残差 7.55e-15、total crosscheck 残差 9.10e-13；Random=100、paired bootstrap=5000。结果：全切分使 Direction 降低；L2 q≤64 上升、q=128 下降 0.00244。Cancellation 的 K=1 L2 相对 Random 无稳定优势；pooled score-vs-actual gain Spearman：Cancellation L2/Direction 0.304/0.175、Endpoint -0.055/0.010、Predicted -0.695/0.014。更新 EXP、RESULTS、ORCHE，并新增 `docs/results/64-large-var-k-error-composition.md`。产物位于 outputs/train/v0-crb-mech-test/64-large-var-k-error-composition/，数据位于 data/v0-crb-mech-test/64-large-var/。


2026-11-11：按用户要求将完整实验聚合表整理为 `docs/results/64-large-var-k-error-composition-full-tables.md`。包括 q/K 全 sweep 的 point-weighted 与 equal-block L2/Direction、matched paired differences、selection efficiency、Representation/Estimation/Total MSE 与 Spearman；每个表格单元保留轨迹 bootstrap 95% CI。原始 CSV 继续保留 Random 重复离散、逐轨迹和逐区间数据。主批次报告及 RESULTS 索引已链接完整表格文档。


2026-11-11：按用户询问复核 64-smooth 与 64-large-var 的 MSE 分量曲线及逐轨迹最优 K。共同 q={4,8,16,32} 下旧批次全切分 L2 总误差下降 0.5942/0.2738/0.1313/0.0677；新批次全切分则上升 0.1360/0.1302/0.1210/0.1036。新批次 q≤64 的 L2 每条轨迹最优均 K=0；q=128 Oracle-L2 平均曲线内部最优 K=81/127，64/64 轨迹内部最优；Endpoint/Predicted 61/64、Cancellation 46/64 轨迹内部最优。Direction 在两个批次都随最大 K 改善。补充解释、分量变化与切分上限定义至批次结果报告和 RESULTS 总览。

## 2026-11-11 21:01 CST：src-96 梯度批次进度

TimerXL 阶段 96/96 条轨迹完成，八个 worker 和模型 `.complete` 均存在；TimesFM 阶段 96/96 条完成，两个 worker 和模型 `.complete` 均存在，实际阶段约 4.4 小时。队列已自动转入 Chronos 四进程，四个 worker 存活，GPU 显存约 16.9 GiB，检查时利用率 99%。Chronos 已写 CRB 10/96 条（16,813/166,659 锚点），Uniform 13/96 条（48,386/166,659 锚点）；另有正在处理的窗口，四个缓存计时文件累计约 72,145 次梯度计算。三模型整批 `.complete` 尚不存在。正式输出约 5.6 GiB，数据盘可用 435 GiB。按 Chronos 当前约 6.6 梯度/秒的累计计算速率和两方法 333,318 次上界粗估，尚需约 11 小时纯梯度计算，实际完工还受阶段调度和缓存开销影响。

## 2026-11-11 21:03 CST：停止 CRB、改为仅准备 Uniform

用户决定不再计算 CRB，并删除本批次已生成的 CRB 部分；要求中文记录。检查时 TimerXL 和 TimesFM 的 CRB/Uniform 清单均为 96/96，Chronos 的 CRB 为 10/96、Uniform 为 13/96。已向 Chronos 四个 worker 发送终止信号，确认进程和原队列退出；暂停所有新写入。下一步在保留 Uniform 轨迹清单与已有梯度位数据的前提下，删除 CRB 清单、剔除缓存中仅由 CRB 使用的梯度，恢复 Chronos 剩余 Uniform 计算。旧的三模型完整完成标记将按新协议重新核验，不能沿用旧 CRB+Uniform 完成状态。

2026-11-11 21:06 CST：三模型缓存按 Uniform 清单逐行过滤并核验，TimerXL 保留 166,659 行、TimesFM 保留 166,659 行、Chronos 保留 48,386 行；CRB 清单目录与 CRB 专属梯度行已删除。修改入口为仅 Uniform，缓存身份增加 `method=uniform`；在 tmux `v0-grad-src-96-uniform` 启动 Chronos 四进程继续剩余轨迹。更新 EXP、ORCHE、RESULTS 的中文现行方案，旧 CRB 计算结果不再作为现行结果。

## 2026-11-11：src-96、q=4 输出头梯度估算

按用户要求只计算估计，不运行新梯度。读取固定 96 条轨迹的长度、现有输出头实现和历史计时，算得每模型 q=4 预算 666,575 个锚点。以 4096 维 BF16 投影为前提，三模型输出约 15.3–16.2 GiB；用时区间因缺乏三模型输出头专项计时而为低置信度。已写入 `docs/results/src-96-q4-output-head-estimate.md` 并登记 EXP、RESULTS。当前 Uniform 正式任务未修改。

2026-11-11：按用户要求更新 TEMP.md，逐项对照 docs/EXP.md、docs/RESULTS.md 与 docs/LOG.md，补充平滑 mock 的 CRB/CES/重建器、Cancellation 虚拟分块、K 扫描和误差分解，高变化 mock、CRB 重建扩展、v1 历史状态、src-8192 与 src-96 当前仅 Uniform 进度及 q=4 资源估算摘要；新增覆盖索引并将修改日期更新为 2026-11-11。未更改实验方案或运行任务。核验时 tmux v0-grad-src-96-uniform 仍存在；Chronos 运行状态以最新实际日志为准。

2026-11-11：补充 TEMP.md 中 v1 seed 0 的六个模型/数据池批次名称及 TimerXL、TimesFM 第 4000 步 FEV MASE 数值，并重申这些历史结果已废弃；seed 1–3 的状态以 EXP.md 逐批登记为准。文档差异空白检查通过。

## 2026-11-11：TimerXL 前四轨全窗口梯度插队

按用户要求暂停 Chronos Uniform 四进程，确认其进程与旧 tmux 退出，保留既有 Uniform 清单和 BF16 缓存。固定 src-96 前四条轨迹的合法窗口合计 484,907，预计纯 BF16 4096 维向量约 3.70 GiB。新增配置、一次性入口和八进程调度脚本。16 窗口预演成功：核对窗口 ID 1–16、BF16 位编码和 4096 维；已删除预演输出。正式批次准备启动，成功后自动恢复原 Chronos Uniform 队列。

2026-11-11 21:42 CST：在 tmux `v0-timerxl-first4-full` 启动八进程正式计算，验证八个进程存活、GPU 利用率 100%，已生成首批梯度分片。前四条轨迹全窗口总数 484,907，预演残留已清理；当前 Chronos Uniform 仍暂停，调度脚本会在插队批次成功后自动恢复。

## 2026-11-11 21:44 CST：TimerXL 前四轨全窗口批次 ETA

八个 worker 均存活，GPU 检查时利用率 100%。缓存计时记录累计 156,672/484,907 个窗口，约 32.3%；阶段计时的梯度计算约 1,143 窗口/秒，按纯计算剩余约 4.8 分钟。考虑缓存写入、收尾和短期波动，预计插队批次再需约 5–8 分钟；这是运行中 ETA，尚未完成。正式输出约 1.1 GiB。完成后调度脚本自动恢复 Chronos Uniform。


## 2026-11-11：TimerXL 四轨代表性与曲率分析

用户指定两个新批次：v0-timerXL-grad-4-lm-head、v0-timer-XL-grad-4-curvature。核验原始四轨全部484,907个窗口、八个worker和批次完成标记；使用既有 rage-timerxl 环境。新增 scripts/temp/v0_timerxl_grad4_analysis.py，在 tmux v0-timerxl-grad4-analysis 顺序执行完整LoRA二阶差分分析和每轨128个等距窗口的输出头梯度计算。代表性比较按用户澄清，比较同一窗口对在各自参数空间内的方向余弦，并排除轨迹内起点距离不足576的窗口对。两批次完成标记、指标JSON和抽样向量已核验。原始LoRA缓存未修改；Chronos Uniform任务仍由原调度会话运行。结果见 docs/results/v0-timerXL-grad-4-lm-head.md 与 docs/results/v0-timer-XL-grad-4-curvature.md。

2026-11-11 22:42 CST：按用户补充要求，增加三阶差 d3 和16段二阶差RMS，重新读取全四轨484,907个窗口。曲率相邻变化 d3 RMS/d2 RMS 为1.592、1.797、1.805、1.778；相对梯度 d2 RMS/g RMS 为0.540、0.944、1.535、0.845。16段最大/最小为1.38、6.46、1.29、1.33；短轨迹1有明显早高晚低分段，其余三轨段间量级较稳。更新正式结果JSON及曲率报告，完成标记已核验。

## 2026-11-11：TimerXL 四轨 LoRA 梯度变化速度

新增批次 v0-timer-XL-grad-4-change-velocity。已核验原始四轨逐窗口梯度完成标记；新增 scripts/temp/v0_timerxl_grad4_change_velocity.py，在 tmux v0-timerxl-grad4-change-velocity 中运行。固定k=1..512、三种逐窗口对指标，轨迹内所有有效窗口对均计入，不抽样。使用rage-timerxl环境与GPU矩阵乘法；已有Chronos Uniform四进程继续运行。结果待核验。

2026-11-11 22:56 CST：v0-timer-XL-grad-4-change-velocity 完成。四轨每个k=1..512均核验有效对数等于max(0,N-k)，输出2048行CSV和四张独立三面板PNG/SVG图；轨迹1图到k=152。原始指标与图目录完成标记均存在。使用rage-timerxl做数值计算、rage环境绘图；新增绘图入口 scripts/temp/plot_timerxl_grad4_change_velocity.py。结果及解释已写 docs/results/v0-timer-XL-grad-4-change-velocity.md、RESULTS.md，EXP批次状态改为完成。Chronos Uniform队列未修改。

## 2026-11-11：输出头与LoRA变化曲线比较

用户询问若按lm-head梯度计算三个随k变化指标是否与LoRA近似。原每轨128个等距头梯度不支持相邻滞后，故新增匹配连续窗口段补充批次。长轨0、2、3各两段768窗口，短轨1全部153窗口，合计4761；同窗口计算头梯度并提取已有LoRA，比较各自空间内的三条k=1..512曲线，另以全轨LoRA曲线检查段采样偏差。脚本 scripts/temp/v0_timerxl_grad4_head_velocity.py；tmux v0-timerxl-head-velocity 已启动，结果待核验。

2026-11-11 23:02 CST：输出头与LoRA变化曲线补充批次完成。4761个匹配窗口、7个连续窗口段，输出头与已有LoRA同段三指标曲线及全轨LoRA参考曲线已写入 lag_comparison.csv；四张独立叠加PNG/SVG图完成。长轨同段LoRA与全轨LoRA曲线MAE仅0.017–0.039；输出头系统性低估模长变化。更新 EXP、RESULTS 和独立报告 docs/results/v0-timerXL-grad-4-lm-head-change-velocity.md；任务未改变Chronos Uniform队列。

2026-11-11：用户询问方向余弦退化速度的跨轨排序。读取已有同窗口段逐k结果，计算k=1..16平均余弦、k=1至16下降量与首次跌破0.5的k；按水平/阈值排序一致，按绝对下降量不一致。已将定义、八行数值和解释追加至输出头与LoRA对照报告及RESULTS。未运行新梯度任务。

## 2026-11-11：每16窗口采样的变化速度排序

用户要求检验每16窗口采一个LoRA梯度，能否识别全量三指标变化速度的跨轨排序。新增 stride-16-rank 补充批次，使用已有四轨完整梯度缓存。默认相位0，同时做16个起点相位敏感性检查；仅比较可观测的k=16,32,...,512，四轨共同排序使用k=16..64的四个滞后，三长轨另用k=16..512。脚本 scripts/temp/v0_timerxl_grad4_stride16_rank.py 已在tmux v0-timerxl-stride16-rank 启动，结果待核验。

2026-11-11：用户撤回每16窗口抽一个梯度方案，改为每轨随机两个连续16窗口区间。收到撤回时旧stride-16-rank计算已完成并写出产物，但不作为现行结论；其产物和操作记录保留历史。已将EXP对应行标为撤回，新建two-random-16-blocks批次。新批只读取每轨两个区间的已有LoRA梯度，比较k=1..15曲线和四轨排序，不重算模型梯度。

2026-11-11 23:14 CST：two-random-16-blocks 批次完成。seed 0选择的四轨区间起点为0:[146595,195768]、1:[38,71]、2:[999,7502]、3:[3805,17320]；读取128个已有梯度、k=1..15逐区间计算。三项等k平均分数均未完整恢复四轨排序，都将0和3对调；pair-weighted补充检查结论相同。输出CSV、JSON、128梯度和四张独立PNG/SVG图，完成标记已核验。更新 EXP、RESULTS 和独立报告 docs/results/v0-timer-XL-grad-4-two-random-16-blocks.md。此前stride-16-rank产物保留历史但未用于结论。

## 2026-11-11 23:26 CST：启动64轨TimerXL梯度扩展

固定src-96前64轨共1,980,371个合法窗口；两类4096D BF16投影梯度原始位数据约30.22 GiB，数据盘当时可用434 GiB。只存投影向量，不存原始维度梯度。已有前四轨LoRA缓存复用，新算其余60轨LoRA与全部64轨输出头。暂停原Chronos Uniform四进程并确认退出，缓存保留。新增配置、准备脚本和队列脚本；两类各16窗口单进程预演成功，头梯度同此前样本方向余弦0.9999986；八worker头梯度并发预演完成。tmux v0-timerxl-grad64 已启动正式LoRA八worker阶段，后续顺序接头梯度和自动分析，成功后恢复Chronos Uniform。当前结果未完成。


## 2026-11-12 00:20 CST：TimerXL 64轨 CRB最终重放启动

用户指定实验 v0-crb-final-test-timerXL-64tr，对现有64轨4096D BF16 LoRA梯度缓存进行重放；新增梯度准备未执行。q为每轨精确锚点数2,4,6,8,12,16,24,32,64；同预算比较CRB与等距Uniform。按用户给定1:0:1配置CRB coverage:error:split，关闭error阶段，额外锚点在coverage和min-kappa split间均分，奇数余数给coverage。块含两个锚点及之间全部有效起点，公共端点计入相邻两块；代表为块内线性重建均值，指标按块起点数加权。新增一次性脚本 scripts/temp/v0_crb_final_test_timerxl_64tr.py，远端rage-timerxl环境tmux v0-crb-final-timerxl64运行，CPU重放避免占用忙碌GPU。输出目录outputs/train/v0-crb-final-test-timerXL-64tr；00:20启动，完成与数值核验待续。


## 2026-11-12 00:35 CST：CRB TimerXL 另外两个阶段预算臂入队

用户要求在当前CRB coverage:error:split=1:0:1重放后，继续计算1:1:0与1:0:0两个臂，并在给出ETA后停止主动轮询。新增脚本 scripts/temp/v0_crb_final_test_timerxl_64tr_variants.py：1:1:0在coverage与冻结的积分二次插值风险error阶段间分配额外锚点，关闭min-kappa split；1:0:0为coverage-only。均比较同预算Uniform，q、方向代表误差、块端点计数口径不变。新增串行队列 scripts/temp/queue_v0_crb_final_test_timerxl_64tr.sh，tmux v0-crb-final-queue已启动，等待当前批次outputs/train/v0-crb-final-test-timerXL-64tr/.complete后依次执行两臂。00:35最近观测基准臂处理至第29轨、已完成前28轨共1,089,156窗口。基于已观测吞吐估算当前臂剩约12分钟、新臂约26分钟/臂，从队列启动至全队列结束约1小时至1小时20分钟，属于粗略ETA。按用户要求此后不再主动轮询进度。

## 2026-11-12 00:35 CST：64轨梯度分析断点修复

64轨LoRA与输出头4096D BF16投影梯度均完成，分别有9个阶段/worker完成标记；输出头最后worker于00:29完成。自动分析已完成64轨LoRA部分，于切换输出头时退出；检查发现分析入口写死LoRA缓存目录后缀 local，而输出头缓存目录后缀是模型修订号。修正为读取各worker唯一 cache_manifest.json 所在目录定位缓存，验证16个LoRA缓存目录及8个输出头缓存目录均可定位。00:35在tmux v0-timerxl-grad64-analysis 重启分析，已有LoRA轨迹完成标记使其跳过重复计算。未删除或重算梯度。当前结果仍待输出头分析与汇总完成。

2026-11-12 00:37 CST：用户要求挂后台并给ETA，不继续主动轮询。64轨LoRA及输出头梯度全部完成，LoRA分析64/64，输出头分析检查时3/64且tmux v0-timerxl-grad64-analysis运行。新增 scripts/temp/finish_timerxl_grad64_then_resume.sh，在tmux v0-timerxl-grad64-finish 等待分析会话结束；若分析完成标记存在，写正式批次完成标记，再自动恢复Chronos Uniform；若分析失败，保留缓存且不误写完成标记，仍恢复Chronos避免GPU空闲。预计分析和文档汇总约再需5–10分钟，此为估计而非完成状态。无自动巡检。

2026-11-12：v0-timer-XL-grad-64-change-velocity 批次完成。前64轨共1,980,371合法窗口，LoRA与输出头4096D BF16投影梯度准备及全部64轨分析完成；头代理排序Spearman（方向/L2/模长）=0.912/0.916/0.557，均匀两段16窗口估计=0.909/0.880/0.609。所有worker、分析目录与整批完成标记已核验；报告 docs/results/v0-timer-XL-grad-64-change-velocity.md。Chronos Uniform由同一调度脚本自动恢复。

## 2026-11-12：两段16窗口输出头代理补充实验

按用户要求复用每轨两个均匀分布的16窗口区间，从已完成的64轨lm_heads缓存读取2048个4096D BF16投影梯度；未重算模型梯度。新增 scripts/temp/v0_timerxl_grad64_two_head_blocks.py，计算k=1..15三指标和64轨分数，分别对照全量LoRA、全量输出头、同段真实LoRA。输出960行逐k对照、64行轨迹分数与排序JSON，.complete核验。两段头代理对全量LoRA排序Spearman为0.868/0.829/0.294、MAE为0.172/0.158/0.167；模长前16名仅重合5/16。EXP、RESULTS和64轨报告已同步更新。


## 2026-11-12 01:23 CST：TimerXL CRB阶段结果与队列恢复

检查发现基准臂1:0:1于00:47:31完成，64/64轨、20,352块、1,152条轨迹汇总及.complete均存在。全局窗口起点加权方向误差：q=2时CRB/Uniform均0.869273；q=4 0.839748/0.849877；q=6 0.833680/0.836117；q=8 0.827456/0.837463；q=12 0.825175/0.842601；q=16 0.822729/0.825908；q=24 0.819705/0.816880；q=32 0.815820/0.816631；q=64 0.802208/0.801250。窗口起点加权指标CRB在q=4,6,8,12,16,32较低，Uniform在q=24,64较低，q=2相同；等轨迹均值方向有所不同，详细表见docs/results/v0-crb-final-test-timerXL-64tr.md。

追加臂初次在第1轨退出，日志显示ModuleNotFoundError: No module named src。已将变体脚本改为插入项目src目录并从rage.crb导入，确认CRB helper可导入，随后于01:23重新启动tmux v0-crb-final-queue。1:1:0已开始，1:0:0仍由队列在其后运行。初次失败未生成指标产物。


## 2026-11-12：TimerXL CRB实验更正q语义并清理旧产物

用户澄清q为平均锚点间距。此前错误使用q作为每轨锚点数量的全部CRB输出、对应日志、错误结果报告及summary条目已删除；源BF16梯度缓存保留。错误口径的追加队列与变体已停止。新口径按K=max(2,ceil(N/q))匹配CRB/Uniform预算，q={2,4,6,8,12,16,24,32,64}，三个CRB阶段预算臂依次1:0:1、1:1:0、1:0:0。输出按ratio分目录保存。此前向量指标误用float64，用户确认BF16足够；新跑使用BF16源值和float32归约，不使用float64向量运算。

主要计算开销来自缓存梯度重放的逐点余弦：每个arm约4,069.9万次窗口访问，4096维约1667亿个向量分量参与点积/范数；CRB锚点选择另需约252.7万次跨q的锚点分配及高维端点优先级计算。全64轨仅一次取梯度缓存，无模型反向传播。q=2单个预算约99万全局锚点。块级产物采用gzip流式写出，避免把数百万行同时放入内存。


## 2026-11-12 01:44 CST：按q间距口径重启64轨CRB队列

已清除错误q口径的输出目录、对应日志目录、错误结果报告和RESULTS摘要；旧的一次性错误口径脚本已移除，源BF16梯度缓存保留。旧追加队列终止。用户确认BF16足够；新脚本不做float64向量运算，缓存投影仍为BF16值，方向/区间指标使用float32向量归约。更正q定义为平均锚点间距，K=max(2,ceil(N/q))，q={2,4,6,8,12,16,24,32,64}。已启动tmux v0-crb-final-spacing-q，顺序运行1:0:1、1:1:0、1:0:0，并对Uniform匹配相同K。

资源估算：每臂跨64轨与9个q共约2,527,365个锚点预算，约5,053,570个闭区间块（CRB与Uniform合计）；逐点指标约40,699,104次窗口访问、4096维计算，块表gzip流式写出。旧口径实测单臂27.4分钟，但旧脚本包含float64和逐块Python循环；新口径的块数及CRB锚点选择更多，尽管指标计算已批处理，三臂总ETA粗估约2–3小时。实际时间受CPU吞吐与缓存读取影响。


## 2026-11-12 02:13 CST：spacing-q TimerXL CRB三臂全部完成

tmux队列v0-crb-final-spacing-q已顺序完成1:0:1、1:1:0、1:0:0。每臂64轨、9个q，三臂及顶层.complete均核验；每臂耗时约9.4–9.6分钟，全队列约29分钟。q=2,4,6,8,12,16,24,32,64均按K=max(2,ceil(N/q))锚点/轨计算。逐块CSV gzip保存，汇总行数和目录见docs/results/v0-crb-final-test-timerXL-64tr.md。窗口起点加权Uniform在1:0:1及1:1:0的全部q胜出；1:0:0只在q=64胜出。错误q口径产物已删除。

## 2026-11-12：补记 TimerXL 梯度时间变化实验链条

按用户要求，在 docs/EXP.md 的历史实验部分新增四轨代表性、曲率、k=1..512 变化速度、四轨随机两段16窗口、64轨扩展和两段输出头代理的贯通总结。记录了固定样本、投影与 BF16 口径、轨迹间/内统计、三种估计方法的数值误差与排序，以及适用限制。数值对照 docs/results/v0-timer-XL-grad-64-change-velocity.md；未启动新的计算任务，未更改实验参数或已有产物。

## 2026-11-12：启动 ChronosTiny 与 TimerXL 的64轨LoRA梯度相关性实验

用户要求固定 src-96 前64轨全部合法窗口的 ChronosTiny LoRA 4096D投影，并明确使用原始预训练权重而非100步预热检查点。已检查原始权重 cache/pretrained_models/amazon/chronos-t5-tiny/29d808298f1a62493e7b9a5e08529d0d930fa189、TimerXL既有缓存、数据盘余量约406 GiB。新增配置 configs/rage/v0-chronostiny-timerxl-lora-full-64.json 和准备入口 scripts/temp/prepare_chronostiny_lora_full64.py；16窗口预演成功，正式计算与分析待启动。已在EXP登记，未报告任何尚未生成的相关性数值。

## 2026-11-12 02:58 CST：ChronosTiny全量64轨任务已启动

已在 tmux v0-chronostiny-timerxl-full64 启动 scripts/temp/run_chronostiny_timerxl_full64.sh。8个worker各负责约247546个不重叠合法窗口；每个worker的首512窗口均成功并已输出进度日志，GPU利用率100%、显存约6.9 GiB。早期进度推算全量梯度阶段约9小时，受持续吞吐和其他负载影响会变化。已新增自动分析入口 scripts/temp/analyze_chronostiny_timerxl_full64.py 并通过语法检查；它在梯度阶段全部完成后自动运行。无相关性结果可报告。本次只做启动后一次状态核对，不设置轮询或自动巡检。

## 2026-11-12：按用户最新要求改为仅计算一条轨迹

用户将64轨范围缩减为先分析一条。已向 tmux v0-chronostiny-timerxl-full64 发送中断；核对原8 worker和tmux队列均已停止，部分缓存保留于 outputs/train/v0-chronostiny-timerxl-lora-full-64/，不作为完整轨迹结果。新增独立单轨配置 configs/rage/v0-chronostiny-timerxl-lora-track0.json，仅选src-96 ordinal 0；分析脚本和队列支持独立配置。已启动 tmux v0-chronostiny-timerxl-track0，单worker完整计算230,161个窗口，使用原始ChronosTiny权重。预计梯度阶段约3小时；分析将自动接续。EXP已区分停止的64轨部分缓存与当前单轨任务。

## 2026-11-12 03:04 CST：单轨任务扩为8个GPU worker

按用户要求将GPU资源留给本任务并尽量跑满。停止单worker进程后保留其已完成梯度缓存；配置改为8 workers，单轨230,161个窗口按连续全局范围均分，每worker约28,770个且互不重叠。已在 tmux v0-chronostiny-timerxl-track0 重启，8个进程均已加载原始ChronosTiny和LoRA并进入梯度阶段；worker0复用此前完成的缓存。分析仍由队列在全部worker完成后自动执行。后续不轮询等待。

## 2026-11-12 03:27 CST：单轨ChronosTiny吞吐核对

单轨任务进度约30%：worker约8,704/28,770，8个进程GPU利用率100%、显存约6.9 GiB。最近日志速率估算梯度余量约44分钟，早先约1小时ETA来自启动初期数据，稳定进度后修正。对照64轨TimerXL日志：TimerXL worker在稳定阶段约8,192窗口/58秒；ChronosTiny当前约4,096窗口/8.9分钟。TimerXL配置gradient_batch/vjp_batch均32并使用TimerXL专用批量梯度probe；ChronosTiny当前gradient_batch/vjp_batch为8，Chronos通用probe对每窗口分别构造loss再执行batched VJP，故小模型规模并未带来相同吞吐。全卡已被8个ChronosTiny进程占用，当前先保留已推进约30%的运行设置。


## 2026-11-12 04:01 CST：完成TimerXL 64轨逐轨PCA方向分析

按用户澄清后的目标，重点报告每轨PC1解释率与64轨PC1方向差异；全局PC1不作为结论依据。使用rage-timerxl环境和8个CPU线程流式读取已有BF16缓存，GPU保持给正在运行的ChronosTiny任务。核验1,980,371个窗口全部读取。逐轨PC1解释率中位数12.1%（P10–P90 6.6%–32.4%）；两两PC1夹角中位数87.3°，1,944/2,016对超过60°。前后半PC1余弦低于0.5的有18/64轨；按窗口数至少4096且半段余弦至少0.8筛选的34轨中，夹角中位数87.1°。结论仅作描述性比较，没有执行连续块bootstrap或正式p值检验。首次运行修正了缓存.lock匹配问题及前后半切分边界，最终输出完整标记已核验。详细报告docs/results/v0-timerxl-grad64-pca.md，产物outputs/train/v0-timer-XL-grad-64-change-velocity/analysis/pca/。


## 2026-11-12：补算64轨达到80%方差的主成分数

用户指出目标是判断逐轨方向的低秩性，不需要依赖全局PCA。先尝试CPU完整特征值谱累计，用户告知GPU空闲后停止未完成的CPU尝试，并在rage-timerxl环境改用CUDA。对64轨真实4096维BF16投影的中心化协方差逐轨求完整特征值谱；1,980,371个窗口全部核验。rank80等轨均值174、中位数137、P10–P90为15–380；窗口数至少4096的40轨均值/中位数248/235。结论为相对于4096维有谱集中，但典型轨迹仍需百级主成分达到80%，不属于十几个方向的极低秩；轨迹差异显著，短轨秩受样本数限制。逐轨谱与汇总位于outputs/train/v0-timer-XL-grad-64-change-velocity/analysis/pca-spectrum/，报告补充见docs/results/v0-timerxl-grad64-pca.md。该描述统计未作时间块bootstrap或正式显著性检验。

## 2026-11-12 12:10 CST：完成ChronosTiny与TimerXL ordinal 0比较

8个worker已完成单轨230,161个窗口；梯度和分析.complete均已核验。首次自动分析因仍含64轨固定长度检查而退出；修正为单轨TimerXL缓存读取后，遇到单轨Spearman不可定义导致JSON NaN拒写；改为将不可估计的轨迹级相关写为null后分析完成。结果：8,128个等距样本非临近窗口对余弦Spearman=0.0758、Pearson=0.0768、Top10%重合14.4%；全量相邻余弦Spearman=−0.1379。k=1 TimerXL/ChronosTiny方向余弦0.9314/0.9348、相对L2 0.4085/0.3574、相对模长差0.1799/0.0478。单轨不能支持跨轨相关性结论。结果已记入EXP.md与RESULTS.md；GPU已空闲，不自动继续后续轨迹。


## 2026-11-12：明确rank80比例的合法起点长度口径

用户明确要求逐轨计算 (v_i/L_i)，其中 (L_i) 是合法起点长度。核对本批cache_manifest为all_legal_windows，故逐轨完整梯度缓存行数就是精确的 (L_i)，共1,980,371个。分段文件k1_segments.csv只保留完整分段，最大末端值会漏掉尾部不足一段的合法起点，不能用于L。用精确缓存行数复核后，平均L=30,943.3，中位数12,144.5；等轨逐条v/L的均值4.07%、中位数1.63%、P10–P90为0.14%–11.39%；sum(v)/sum(L)=0.563%。逐轨比值已写入spectrum_comparison.json，详细口径见docs/results/v0-timerxl-grad64-pca.md。


## 2026-11-12 v0 结果汇总与归档

- 检查：远程 /root/TS-Lab；无 v0 实验计算进程。核对 EXP、RESULTS、TEMP、v0 专题报告、输出树、缓存目录和日志目录。
- 修订：新增 docs/results/v0-consolidated-summary-20261112.md；更新 docs/EXP.md、docs/RESULTS.md、根目录 TEMP.md；将 ChronosTiny 64 轨扩展状态由过期的运行中更正为用户调整后停止、无全轨结果；将 src-96 Chronos 梯度准备状态更正为未完成且当前无运行进程。
- 归档：v0/v00 输出中除 grad_cache/、gradients/ 及明确命名的梯度缓存外，所有生成文件及 v0/v00 日志移入 archive/v0-consolidated-20261112/artifacts/；原引用位置以符号链接保留。脚本、配置和结果报告副本放入归档快照目录。
- 验证：归档 README 记录移动文件数、体积和缓存保留规则；梯度缓存没有移动或删除。结果数值仅引用已生成报告和缓存状态，不对不完整批次补造结果。
- 后续：如恢复任何缓存驱动的 v0 分析，先按 EXP 中的批次范围、配置和完成标记确认输入。

## 2026-11-12 14:03 CST：将 TimerXL 64轨梯度缓存整理为自包含目录

核对 docs/EXP.md 与 64轨结果报告后，确认当前批次 LoRA 缓存 manifest 覆盖 ordinal 4–63；ordinal 0–3 的 484,907 个窗口仍保存在前四轨准备批次。将前四轨八个 worker 的 480 个 `windows-*.npz` 分片及 `cache_manifest.json`、`gradient_timing.json` 复制到 `outputs/train/v0-timer-XL-grad-64-change-velocity/formal/lora/reused-first4/worker-*/grad_cache/thuml-timer-base-84m--local/`。保留旧实验目录原件，不重新计算梯度。当前60轨缓存有1,464个分片，合并后 LoRA 共1,944个分片；配合已有64轨输出头缓存，覆盖1,980,371个合法窗口。新增 `formal/collection_manifest.json` 描述缓存范围和格式；将 formal/ 内归档引用型 `.complete` 符号链接替换为普通标记文件，核验 formal/ 下无符号链接。整理后 formal/ 约31 GiB；数据盘已用151 GiB、可用400 GiB。未改参数或实验结果。没有运行新的计算任务。

- 归档复核补充：发现 v0-gradient-proxy-test 的 head_gradients.npy 及其它抽样梯度数组属于后续可复用缓存。已将 14 个梯度缓存数组（50.30 GiB）从归档恢复到原 outputs/train 路径；归档 README 已记录。其余非缓存结果继续归档。

## 2026-11-12：为后续正式 v1 实验删除旧输出头梯度缓存

用户要求删除此前确认的超大输出头梯度缓存并保留说明。检查无相关运行进程后，仅删除 `outputs/train/v0-gradient-proxy-test/trajectory-00` 至 `trajectory-09/head_gradients.npy` 十个未投影 FP32 原始逐窗口梯度文件，共53,838,087,424字节（约50.14 GiB）。保留该实验全部派生表格、NPZ分析缓存、校验记录和完成标记；未删除 TimerXL 64轨4096D BF16输出头缓存。新增输出目录说明 `outputs/train/v0-gradient-proxy-test/OUTPUT_HEAD_GRADIENTS_REMOVED.md`，并更新对应专题结果、EXP和RESULTS记录。脚本仍可用于重新计算；若重跑依赖原始输出头梯度的分析，需要先重新生成这些数组。未运行实验或测试。

核验补充：十个目标文件均已不存在，派生报告所列 summary、Gram 矩阵与 Cross-Rank aggregate 仍可访问。数据盘空间从删除前已用151 GiB/可用400 GiB，变为已用100 GiB/可用451 GiB。

## 2026-11-12 spacing-q-dir 启动准备
最后修改日期：2026-11-12。读取 spacing-q 原实验代码和文档，复用64轨BF16 LoRA缓存、q及选点规则。新增 scripts/temp/v0_crb_final_test_timerxl_64tr_dir.py 和队列脚本；全局第1百分位模长阈值及置零数量写 direction_threshold.json，四比例均使用方向进行选点、插值和评分，增加块等权指标。输出目录独立于原 spacing-q。代码语法检查通过。用户明确禁止轮询，因此不设定巡检。

2026-11-12 14:57 CST：tmux v0-crb-final-spacing-q-dir 已启动，进程进入全局阈值扫描阶段，首次日志显示64轨、1,980,371窗口、q配置和比例1:0:0正确；输出目录已建立。仅做启动核验，后续不轮询。

2026-11-12 15:18 CST：按用户要求单次检查 spacing-q-dir 进度。比例1:0:0和1:0:1的aggregate.json及.complete已生成；1:1:0正在运行，日志显示第3/64轨完成；1:1:1尚待队列。进程存活，未见报错。未设置轮询。

## 2026-11-12 结果核验
最后修改日期：2026-11-12。按用户要求单次检查spacing-q-dir：四比例及顶层.complete、四份aggregate.json均存在，tmux会话已退出，无实验进程。核对direction_threshold.json：阈值0.020094235，置零19,804/1,980,371个方向。已将实际结果写入结果文档并更新EXP状态。未设置轮询。

## 2026-11-12 spacing-q-dir q=128,256 准备
最后修改日期：2026-11-12。用户要求追加q=128、256。复制既有方向实验脚本，仅修改q集合、独立输出目录及复用父目录阈值文件；创建串行队列。保持原K=max(2,ceil(N/q))定义。无轮询。

2026-11-12 16:48 CST：tmux v0-crb-spacing-q-dir-extra 已启动；首次日志确认q=(128,256)、64轨及1,980,371窗口，1:0:0进程存活，独立输出目录已建立。仅做启动检查，不轮询。

## 2026-11-12 v1-crb-redo / grad-prepare protocol and rehearsal

最后修改日期：2026-11-12。按用户固定决策，热身池为 src-8192，候选梯度池为 src-96；三模型仅训练/计算 LoRA，1000 优化步、有效批次 32，热身按轨迹等概率有放回抽取，再在该轨迹内等概率抽 576 点窗口。TimerXL 原生损失为 480+96，另两模型为 512+64。q=32，CRB 1:0:1，error deprecated，Uniform 同轨等预算；当前逐轨规则每模型每方法计划 83,330 锚点（coverage 41,665，split 41,665）。沿用 TimerXL 8、TimesFM 2、Chronos 4 worker 及 32/32、32/32、1/1 梯度批量/VJP 配方。

新增 scripts/v1_crb_redo_warmup.py、scripts/v1_crb_redo_gradients.py、scripts/temp/run_v1_crb_redo_grad_prepare.sh，扩展训练采样入口支持 RegionWindows 子类自定义 sample_indices。TimerXL 第一次单步预演因 PEFT 输出字段读取方式失败，修改 src/adapters/foundation/timerxl.py 的 native V6 loss，随后三模型各 1 步热身并保存检查点成功。三模型均在 src-96 ordinal 1 完成梯度全路径预演：CRB 与 Uniform 各 4 锚点、共享后 6 唯一缓存行，各 worker .complete 存在。预演数据不用于正式结果。确认没有旧准备进程后，删除旧 v0-grad-prepare-src-96tr-q16-4096d 的 3.7 GiB 无热身输出；保留 src-96/src-8192 快照、日志及其他实验产物。正式队列尚未启动。

2026-11-12 17:27 CST：在热身脚本增加保存后固定窗口损失与全部可训练 LoRA 张量梯度的重载一致性检查，并重新进行三模型独立单步预演，全部通过：TimerXL 固定损失 0.59882873、TimesFM 63.780872、Chronos 2.9834862，前后损失及梯度在 1e-4 容差内相符。使用这些新检查点再次完成三模型短轨梯度全路径预演，CRB/Uniform 各 4 锚点、共享后 6 缓存行，完成标记均存在。Chronos 正式梯度入口修正为文档已测最高吞吐的 batch/VJP 1/1；三模型正式进程数分别为 8/2/4。bash -n、py_compile 与 git diff --check 通过。下一步启动正式 tmux 队列并监控启动后最多 5 分钟。

2026-11-12 17:33 CST：正式 tmux 队列 v1-crb-redo-grad-prepare 于 17:27:45 启动。TimerXL 热身 1000/1000 优化更新完成，固定窗口检查点重载损失与梯度校验通过，warmup/timerxl-84m/.complete 已生成。其后八个 TimerXL 梯度 worker 自动启动；初始模型加载约 1 分钟内 GPU 利用率低，随后进入反传，17:31:33、17:32:19、17:33:01 三次检查利用率均为 100%，显存约 5794 MiB。17:33:01 八个 worker 均在运行，CRB 和 Uniform 分别形成 30 条轨迹清单，无 Traceback/ERROR/RuntimeError；整批及 TimerXL 梯度阶段尚未完成。启动后监控已超过 5 分钟，到此按用户要求停止主动持续巡检。队列保留运行，后续自动接续 TimesFM 与 Chronos；不设置 cron 巡检，不关机。

2026-11-12：按用户要求单次核验q=128、256追加批次。四比例和顶层.complete、四份aggregate.json均存在，追加批次tmux会话已退出。已整理两种加权结果并更新EXP与RESULTS；未设置轮询。

## 2026-11-12 v2 foundation expansion integration

最后修改日期：2026-11-12。用户批准接入 Chronos-2 Small、Chronos-2、Chronos-Bolt Tiny/Mini/Small 和 Moirai-2.0-R-Small，并确认所有新增模型使用专用 Conda 环境。按现有操作流新增模型注册、固定版本权重适配器、原生或固定窗口分位数损失、LoRA 目标模块、逐窗口投影梯度接口及真实权重冒烟脚本。`rage-chronos2` 已在数据盘创建并安装 `chronos-forecasting==2.3.2`；`rage-moirai2` 独立创建中，Torch 2.4.1 CUDA 轮包仍在下载。五个 Chronos-2/Bolt 模型的 CPU 真权重冒烟检查均通过：参数数目、64 步推理、损失、LoRA 反传和单窗口 128D 梯度均有效，完成标记位于 `outputs/train/v2-foundation-expansion/<model>/`。Moirai 真实权重检查和批量 VJP 等价性检查尚未完成。现有 `v1-crb-redo-grad-prepare` GPU 会话未改动；未设置自动巡检。
EOF'
2026-11-12 18:24 CST：五个 Chronos-2/Bolt 固定修订模型再次完成 CPU 真权重冒烟，并补充双窗口批量 VJP 与单窗口逐个反传的损失、4096D 正式探针路径及 LoRA AdamW 单步更新检查。五模型的双窗口 128D 梯度与逐个计算的最大绝对差分别小于 8e-6、6e-6、5e-6、2e-6、1e-6；全部有 `.complete`。Bolt-Tiny 的正式 `build_model` 使用 BF16 与 4096D 投影成功运行双窗口 batched-vjp。新增固定修订登记文件 `configs/pretrained_models/chronos2.yaml`、`moirai2.yaml`，更新 `requ/third_parties/model/integrations.lock`；YAML 解析、代码编译与 `git diff --check` 通过。Moirai 环境安装继续进行，未干预现有 GPU 批次，不设置轮询。
EOF'
2026-11-12 18:31 CST：`rage-moirai2` 安装结束，`pip check` 通过。首次真实权重冒烟发现 Moirai 官方 univariate 预测返回三维张量，本地适配器多索引了一个轴；修正 `src/adapters/foundation/moirai2.py` 后重试通过。将 Moirai 环境的 transformers 固定为 4.45.2，避免新版 5.20.2 与 Torch 2.4.1 的运行时禁用提示，并同步更新安装脚本。Moirai 11,387,208 参数、64 步预测、固定窗口分位数损失、73,728 可训练 LoRA 参数、反传、128D 双窗口 VJP、AdamW 单步与更新后预测均通过；双窗口梯度最大绝对差约 2.8e-7。正式 `build_model` FP32、4096D 双窗口 batched-vjp 路径亦通过。六个模型各自 `.complete` 和本工程接入总 `.complete` 已生成。两环境 `pip check`、代码编译、`git diff --check` 通过。尚未启动正式 GPU 训练或评估，现有 v1-crb-redo GPU 批次保持运行；按用户要求没有轮询。
EOF'
2026-11-12 18:32 CST：将六模型通用冒烟入口从临时目录移至正式 `scripts/smoke_foundation_models.py`。直接以文件路径启动时，项目内 `scripts/queue.py` 遮蔽 Python 标准库 `queue`，导致 Torch 导入失败；改用 `python -m scripts.smoke_foundation_models`，Bolt-Tiny 全路径复测通过。文档已记录正式调用方式。该失败仅影响冒烟脚本启动方式，已有模型结果未受影响。
EOF'
2026-11-12 18:33 CST：正式冒烟入口以 `python -m scripts.smoke_foundation_models` 再次核验 Bolt-Tiny 与 Moirai，均通过。根据六份实际 `smoke.json` 将工程验证参数数目和 LoRA 可训练参数整理到 `docs/RESULTS.md`；未写入预测质量指标。最终 Python 编译、`git diff --check`、六个模型及总完成标记均通过。本接入批次的代码和文档改动留在远端工作树，未提交；用户原有其他未提交改动未覆盖。
EOF'
## 2026-11-12 21:39 CST：v1-crb-redo / grad-prepare 单次进度检查

最后修改日期：2026-11-12。按用户要求通过 ssh tsl 单次检查 tmux、进程、GPU、完成标记和最新日志；未设置后续定时巡检。tmux v1-crb-redo-grad-prepare 存活。TimerXL 热身和梯度阶段均完成，CRB/Uniform 各 96 条轨迹、各 83,330 锚点；TimesFM 热身和梯度阶段均完成，同样各 96 条、各 83,330 锚点。Chronos 热身已完成；四个梯度 worker 存活，最新心跳均在 30 秒范围内，无 Traceback、ERROR 或 RuntimeError。截至 21:39:33，Chronos 两方法各完成 29/96 条轨迹、各 34,112/83,330 锚点；两方法清单逐轨对应。GPU 利用率 89%，显存 4,808 MiB；数据盘余量约 430 GiB。整批 .complete 尚不存在，Chronos 梯度阶段仍在运行。按轨迹数推算 ETA 不可靠，因为长短轨迹差异大，当前不承诺结束时间。未修改参数、停止进程或关机。

## 2026-11-12 22:03 CST：v1-crb-redo TimerXL 两基线排队

最后修改日期：2026-11-12 22:03 CST。用户要求 grad-prepare 完成后依次运行 timerXL-baseline-full 与 timerXL-baseline-rand01，各从预训练权重训练 4,000 步，使用固定 src-96，随后评估相同 4 FEV 加 8 TIME 任务。随机批次用 NumPy default_rng(0) 无放回固定选取 266,631/2,666,312 个全局合法窗口，训练仅从此集合采样。新增训练入口 `scripts/v1_crb_redo_timerxl_baselines.py` 与队列 `scripts/temp/run_v1_crb_redo_timerxl_baselines.sh`。当前 grad-prepare 的 Chronos worker 仍运行，整批标记尚未出现。两批尚无训练或评估结果；后续按实际输出更新 EXP 和 RESULTS。不设置自动巡检，不关机。

2026-11-12 22:04:40 CST：独立 tmux `v1-crb-redo-timerxl-baselines` 已启动并核验会话存活；`logs/v1-crb-redo/timerxl-baselines-queue.log` 记录开始等待 grad-prepare 完成标记。数据预检确认 96 轨、2,666,312 合法窗口、seed 0 固定抽取 266,631 个索引，固定子集取窗形状为 (2,576)。训练脚本编译、CLI 导入、队列 shell 语法和 git diff --check 均通过。当前未开始训练。

## 2026-11-12 23:25 CST：TimerXL 基线队列恢复

最后修改日期：2026-11-12 23:25 CST。单次检查确认 grad-prepare 整批 `.complete` 已于约 23:00 生成，原两个 tmux 会话均已退出。baseline-full 于 23:01 启动，23:08 到达 4,000/4,000 步；保存训练清单时 `adapter_dir.relative_to(ROOT)` 对数据盘真实路径抛出 ValueError，导致队列停止，未进入评估和 rand01。第 4,000 步 LoRA 文件与 169,978,507 字节恢复状态存在。已将训练清单和随机索引清单的路径字段改为项目内 `outputs/` 路径；代码编译与 `git diff --check` 通过。23:25:23 重启 `v1-crb-redo-timerxl-baselines` tmux，23:25:29 日志显示 full 批次模型加载后进入训练恢复。后续需核验训练清单、两基准评估和 rand01 完成标记。不设置自动巡检，不关机。

## 2026-11-12 23:30 CST：基线评估数据修复与队列续跑

最后修改日期：2026-11-12 23:30 CST。full 批次从第 4,000 步恢复状态成功完成训练清单与 161 MiB 合并检查点，首次 FEV 评估因 20 项任务声明中的第 0 项缺少本地 parquet/Arrow 而在进入实际任务前失败；TIME 数据也缺失。保留失败输出目录并重命名为 `fev-failed-20261112-2325`，未删除检查点。使用既有 `scripts/fetch_time_dataset.sh` 下载并验证既定 8 个 TIME 任务，数据 manifest 已生成。新增 `scripts/temp/prepare_v1_crb_redo_fev_data.py`，从 `autogluon/fev_datasets` 修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc` 仅下载所需 4 个 parquet，生成 `configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml` 与 `configs/eval/fev/v1-crb-redo-timerxl.yaml`；四文件存在且 FEV Benchmark 可解析 20 项声明和选中四项。修改项目 `src/evaluate/fev.py`：当指定 task_indices 时，仅要求选中任务本地数据存在，仍校验所有任务的指标声明；没有修改第三方代码。rand01 固定窗口索引已生成、训练在 tmux `v1-crb-redo-rand01-train` 运行。新 tmux `v1-crb-redo-baseline-eval` 已启动，等待 rand01 训练清单后顺序运行两批 12 项评估；未设置自动巡检，不关机。

## 2026-11-12 23:38 CST：TimerXL 两基线完成并整理结果

最后修改日期：2026-11-12 23:38 CST。rand01 在 23:34:10 到达 4,000/4,000 步并保存训练清单；评估队列于 23:34:14 开始 full 评估，随后完成 full 与 rand01 各 4 FEV + 8 TIME。两批训练及四个基准目录 `.complete` 均已核验，FEV CSV 各有 4 行，TIME 元数据各有 8 项和完成时间。队列于 23:35:46 输出 `all evaluations complete` 并退出。已按原始结果文件生成 `docs/results/v1-crb-redo-timerxl-baselines.md` 并更新 EXP、ORCHE、RESULTS；未编造或替换指标。服务器未关机，无自动巡检。
