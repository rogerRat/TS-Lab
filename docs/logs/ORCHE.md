# Task Orchestration

Last modified: 2026-11-11

## v0-active-value-rank

| Task | Resources | Dependency | Command | State |
|---|---|---|---|---|
| 1-expanded-cross-rank | RTX 4080, two CPU threads | Cached head gradients and 100-reference full scores | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_active_value_rank --batch 1` | Completed: 100 references, mean rho 0.7104, .complete |
| 2-active-correction | CPU, two BLAS threads | Batch 1 .complete | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_active_value_rank --batch 2` | Completed: 4 policies x 4 budgets x 3 fractions x 100 references, .complete |

Ran sequentially in tmux v0-active-value-rank; both batches completed and the session exited. Output: outputs/train/v0-active-value-rank/ on the data disk. Logs: logs/v0-active-value-rank/. Keep terminated CRB stopped.

## v0-directional-marginal-gain

| Task | Resources | Dependency | Command | State |
|---|---|---|---|---|
| Replay 2/4/8/16 uniform-anchor additions and compare dispersion to gain | CPU, two BLAS threads; no GPU model work | Completed 46-block v0-gradient-variance-decomposition and ten full/head Gram caches | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_directional_marginal_gain` | Completed: 184 block-budget rows, 64 association rows, 160 sensitivity rows, .complete |

The four fixed budgets were run in one CPU replay. Output: outputs/train/v0-directional-marginal-gain/ on the data disk. Logs: logs/v0-directional-marginal-gain/. No model calls or follow-on batch are pending. The terminated CRB replay remains stopped. Full result: docs/results/v0-directional-marginal-gain.md.

## v0-gradient-variance-decomposition

| Task | Resources | Dependency | Command | State |
|---|---|---|---|---|
| Decompose 46 full-gradient blocks and compare reconstruction correlations | CPU, two BLAS threads; no model GPU work | Completed v0-var-proxy-estimate full-variance output and ten full Gram caches | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_gradient_variance_decomposition` | Completed: 46 rows, 30 association rows, four figure files, .complete |

The run used the rage Conda environment and existing caches. It did not restart the terminated CRB queue. Outputs: outputs/train/v0-gradient-variance-decomposition/ on the data disk; logs: logs/v0-gradient-variance-decomposition/. Full result: docs/results/v0-gradient-variance-decomposition.md. No further batch is pending.

## v0-grad-traj-level-norm-gap

| Task | Resources | Dependency | Command | State |
|---|---|---|---|---|
| Compute complete and output-head trajectory mean-gradient norms | CPU, two BLAS threads; no GPU model work | Ten completed full/head Gram caches under outputs/train/v0-gradient-proxy-test/ | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_grad_traj_level_norm_gap` | Completed: 20 track-kind rows, spread summary, paired correlations, .complete |

The analysis reused saved Gram matrices; no new gradient calls or pending batches. Artifacts: outputs/train/v0-grad-traj-level-norm-gap/. Logs: logs/v0-grad-traj-level-norm-gap/. Results and limits: docs/results/v0-grad-traj-level-norm-gap.md. The user-terminated CRB replay remains stopped.

## v0-var-proxy-estimate

| Task | Resources | Dependency | Command | State |
|---|---|---|---|---|
| Validate and replay sparse proxy variance against four-anchor reconstruction | CPU, two BLAS threads; no GPU model work | Ten completed full/head Gram caches under outputs/train/v0-gradient-proxy-test/ | `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_var_proxy_estimate --config configs/rage/v0-var-proxy-estimate.json` | Completed: 46 full blocks, correlations, and .complete |

A follow-up true-variance comparison used all 256 cached full gradients per identical block. Command: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_true_var_reconstruction`. State: completed, 46 blocks; supplement `.complete` under outputs/train/v0-var-proxy-estimate/true_variance/.

A second follow-up used the arithmetic vector mean of the same four anchors as a constant block representative. Command: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_four_anchor_mean`. State: completed, 46 blocks; `.complete` under outputs/train/v0-var-proxy-estimate/four_anchor_mean/.

The replay used cached gradients, so no new model calls were needed. Results and limitations are recorded in docs/results/v0-var-proxy-estimate.md. The terminated CRB queue remains stopped. No follow-on batch is pending for this experiment.

## v0-grad-mech-l2-norm supplement

The completed grad-mech probe and ten BF16 caches are the only trajectory inputs. The completed gradient-proxy full-coordinate Gram matrices are reusable derived statistics. This CPU-only job has no dependency on the terminated CRB replay and must not restart it. GPU is idle. The command is `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/v0_grad_mech_l2_norm.py --config configs/rage/v0-grad-mech-l2-norm.json` in tmux `v0-grad-mech-l2-norm`, with two CPU threads. Outputs and markers are under outputs/train/v0-grad-mech-l2-norm/; logs are under logs/v0-grad-mech-l2-norm/. State: completed on 2026-11-08; ten trajectories and all 575 lags passed. The result artifacts were moved to archive/v0-grad-mech-test/outputs/train/v0-grad-mech-l2-norm/, and the original output path is a symbolic link. Results and conclusion are in docs/results/v0-grad-mech-l2-norm.md. The ten BF16 caches remain unarchived at outputs/train/v0-crb-reconst-test/gradients/. The tmux session exited after completion. No follow-on GPU work is implied by this supplement.

## Current resources and tasks

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| One-command environment installation | CPU, network, data disk; no GPU | None | tmux `rage-install`, running `bash _install.sh` from the project root | All six profiles passed installation verification |
| Static selection and integrated cycle checks | CPU | Main `rage` environment | Focused regression command; 119 passed, 1 skipped | Passed |
| Integrated preparation and training test | CPU, small RAM | Main `rage` environment installed | `pytest -q tests/test_minimal_cycle.py` | Passed in focused test run |
| Full regression suite | CPU, GPU available for optional cases | Main `rage` environment installed | `logs/tests/full-suite-20261107.log` | Passed: 784 passed, 3 skipped |

The install log is written below `/root/autodl-tmp/rage/logs/install/`. Project data, outputs, logs, and caches use project symlinks to `/root/autodl-tmp/rage/`. All six installation profiles have passed verification. The gradient probe and proxy-exploration batch are complete. No GPU experiment is active. The CRB replay was terminated by the user; see the final status below.

## v0-grad-mech-test readiness

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Fetch and verify registered ChronosTiny checkpoint | Network, data disk | Registry entry | tmux `v0-grad-model` | Complete: pinned checkpoint downloaded and registry validation passed |
| Prepare Chronos `m5` input and freeze first 1,000 eligible IDs | CPU, data disk | Pinned source revision | Completed | Complete: Chronos `m5` fetched (78 MB) and all 30,490 rows loaded to the data-disk bundle |
| 100-step warm-up and full-gradient GPU probe | RTX 4080, host RAM, data disk | Checkpoint and m5 bundle complete | tmux `v0-grad-run`; console log `logs/v0-grad-mech-test/experiment-console.log` | Probe complete: 1,000 / 1,000 trajectories and ten retained BF16 caches complete; CRB queue runs at most two CPU batches concurrently with two threads each |

The gradient probe completed all 1,000 trajectories and its .complete marker is present. The ten retained CRB BF16 cache files are complete and remain available through outputs/train/v0-crb-reconst-test/gradients/. The source experiment is archived under archive/v0-grad-mech-test/; no GPU task is active. The current CRB replay is CPU-only and capped at two concurrent batches with two threads per process.

## v0-crb-reconst-test 标准队列

| 任务 | 资源需求 | 依赖条件 | 命令或会话 | 状态 |
|---|---|---|---|---|
| 源梯度机制探索完成标记 | 无 | outputs/train/v0-grad-mech-test/.complete | 已完成 | 1,000 条轨迹和 10 份缓存均完成 |
| 逐轨迹 CRB 正式队列 | CPU replay; no GPU gradient recomputation | Each retained cache marker; source manifest and warm-up checkpoint | outputs/train/v0-crb-reconst-test/batches/trajectory-NN | 1/10 batches complete; batches 01 and 02 active; queue limit is two concurrent batches and two CPU threads per batch |
| CPU 参数搜索与重放 | 最多 4 个工作线程、BF16 内存映射文件 | 全量梯度完成 | tmux v0-crb-batch-01/02；队列按完成状态补充下一批 | 执行中；280/1,280 配置完成，18/80 个基准完成；批次 00 全部完成，批次 01 和 02 活跃。最新逐批配置计数见 docs/logs/ |

预算标准：q∈{1,4,16,64,128,256,512,576}，定义为 N / 梯度计算次数；每条轨迹 K=ceil(N/q)，首尾强制计算且 K 最低为 2。每条轨迹各自完成全部 8 个预算组，每组包含 baseline-uniform 及 16 个固定比例配置：分母 4 的 15 个单纯形格点加中心点 (1/3,1/3,1/3)，三参数均在 [0,1] 且和为 1。每条轨迹只执行一轮。最多同时运行 2 个轨迹批次，每批设置 2 个 CPU 线程；批次完成后队列补入下一条轨迹。

源探测 .complete 和全部 10 个 BF16 缓存均已存在；CRB 直接复用缓存，不加载 GPU 模型。队列检查每条轨迹批次的完成标记，最多启动 2 批，每批限制 2 个线程，异常退出后自动续跑。通过创建 outputs/train/v0-crb-reconst-test/STOP 请求安全停止；磁盘安全余量为 1 GiB。

## Scheduling

1. Installation and all-profile verification are complete.
2. Focused selector, asset preparation, and training/evaluation checks passed.
3. The full test suite passed (784 passed, 3 skipped); completion marker: `logs/tests/full-suite-20261107.complete`.
4. v0-grad-mech-test and all ten BF16 retained caches are complete. The CRB queue reuses those caches and runs at most two trajectory batches concurrently, with two CPU threads per batch.

Last modified: 2026-11-10

## 2026-11-07 gradient cache amendment

The original first-ten trajectory set has already passed through the probe without full-gradient persistence. The user authorized random trajectory selection. Ten unprocessed manifest ordinals are now fixed in the gradient and CRB configs; the probe is being resumed with atomic BF16 cache writes for those rows. The CRB queue will reuse the files after the predecessor completes. Estimated cache size is 215.5 GB with about 300 GiB disk remaining.

## Per-trajectory CRB execution

The probe completed and all ten retained BF16 caches are present. The formal queue runs ten single-trajectory batches in the fixed sampled order. Batch 00 is active; the queue starts each later batch as soon as its cache is available. Each batch writes to the standard CRB output tree and completes one initial round across all budget groups.

## 2026-11-07 container recovery and bounded replay

The container restarted after CPU and memory saturation. No CRB or probe process survived; all ten BF16 trajectory caches and the source completion marker remained intact. Incomplete outputs from the previous random-ratio search were discarded. The active design is deterministic: 15 denominator-4 simplex lattice points plus the centroid, repeated for all eight budget groups on each of ten tracks. The queue is capped at two concurrent batches and two CPU threads per batch.

## 2026-11-07 gradient experiment completion and archive

The v0-grad-mech-test probe completed all 1,000 trajectories and is archived at archive/v0-grad-mech-test/. Original paths remain available through symbolic links for CRB. The full results are recorded in docs/RESULTS.md. Earlier statements above describing the probe as running are superseded. The CRB queue remains a separate active experiment.

Last modified: 2026-11-07

The archive contains only the grad-mech probe results and visualization; data, checkpoint, manifest, scripts, and CRB gradient caches remain at their original paths.

Last modified: 2026-11-07


Last modified: 2026-11-08


## 2026-11-08 - gradient proxy preparation

Last modified: 2026-11-08

GPU proxy experiment prepared in tmux v0-gradient-proxy, rage environment, two CPU threads. Existing CPU CRB replay retained. Protocol and command: docs/gradient-proxy-protocol.md.


### 2026-11-08 - gradient proxy execution

Last modified: 2026-11-08

v0-gradient-proxy is active on the GPU with two CPU threads. Track 00 finished; track 01 is active. Tracks 02-09 execute sequentially in the same process. After all ten complete, scripts/summarize_gradient_proxies.py generates equal-trajectory summaries and docs/results/v0-gradient-proxy-test.md. Existing CRB uses its original two CPU workers.

## 2026-11-08 container resource pause

The proxy experiment completed 10/10 trajectories. Host RAM is abundant, but the container cgroup is limited to about 62 GiB and was charged about 60.5 GiB with high memory PSI. To avoid OOM and per user authorization, STOP is present for the CRB queue and SIGUSR1 was sent to the two active replay workers. No new CRB batch should launch. Batch workers 03 and 04 were SIGKILLed only after SIGUSR1 failed to stop their in-flight candidates for over 30 minutes. Their current candidates were not committed. Previously completed per-candidate results and the shared full-gradient caches remain. Batch-level and queue-level completion markers are absent because the ten-batch run is unfinished. Resume only after reducing memory pressure and removing STOP.


## 2026-11-08 final experiment status

Last modified: 2026-11-08

| Workstream | Resources | State | Artifacts / next action |
|---|---|---|---|
| v0-gradient-proxy-test proxy exploration batch | rage environment; GPU work complete | 10/10 trajectories complete; no active process | Results in docs/results/v0-gradient-proxy-test.md. A next batch awaits a fixed protocol. |
| v0-crb-reconst-test replay | CPU; no workers active | Terminated by user due broadly clear results and container resource constraints. Batches 00-02 complete, 03-04 partial, 05-09 not started. | Raw results and regenerated plots archived at archive/v0-crb-reconst-test/. Ten BF16 full-gradient caches remain intact. Queue tmux session stopped; STOP marker retained. No replay should restart unless requested. |

The regenerated CRB snapshot contains 429 committed configuration attempts, 326 unique metric files, and 28 uniform baselines. Earlier CRB scheduling paragraphs above describe historical plans and are superseded by this final status.


## 2026-11-08 direct proxy cost benchmark

Completed on the GPU using the confirmed rage environment and the same ten trajectories. Eight spaced windows per trajectory, batch size one, three repeats per method. Median times and speedups are recorded in docs/results/v0-gradient-proxy-cost.md. The CRB queue remains paused under STOP following the container memory incident.


## 2026-11-08 proxy batch extension: Cross-Rank

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Local-Risk summary | CPU and saved proxy metrics | Completed ten-track proxy extraction | docs/results/v0-gradient-proxy-test.md | Complete; positive descriptive block-dispersion association, not a budgeted selection result |
| Cross-Rank cross-track comparison | RTX 4080 cosine GEMM; about 1 GiB GPU score inputs; 2 CPU threads; no model or BP | Ten complete BF16 full-gradient caches and ten proxy feature caches | tmux v0-gradient-proxy-cross-rank; PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.cross_rank_gradient_proxies --config configs/rage/v0-gradient-proxy-cross-rank.json | Complete; 1,877/1,877 references and 45,048/45,048 pairs written; part .complete present |

Cross-Rank uses every reference window from tracks 00 and 07 and 24 deterministic disjoint source windows across tracks 01-06, 08, and 09. Expected pair count is 45,048. Writes only below outputs/train/v0-gradient-proxy-test/cross_rank and logs/v0-gradient-proxy-test. Completion marker is local to the part; existing full-gradient caches and the terminated CRB STOP marker remain untouched.

Last modified: 2026-11-08


## 2026-11-08 Cross-Rank completion

Cross-Rank completed in 1.5 minutes using the rage environment. All ten input manifests and full-gradient cache markers/sizes validated. All 1,877 reference queries and 45,048 reference-source pairs were written to outputs/train/v0-gradient-proxy-test/cross_rank/. The final cgroup charge remained below the configured 62 GiB limit; the task uses no model process and the GPU is now idle. No CRB queue was resumed.

Last modified: 2026-11-08


## 2026-11-08 Local-Risk sparse-sampling extension

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Sparse proxy variance analysis | CPU, 2 threads; saved per-track head/full Gram matrices; no GPU or gradient recomputation | v0-gradient-proxy-test ten-track .complete markers | tmux v0-gradient-proxy-local-risk; PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.analyze_sparse_proxy_risk --config configs/rage/v0-gradient-proxy-local-risk-sparse.json | Complete; 396 complete blocks, 20 replicates for each sparse count, exact 32-point checks passed; .complete present |

For each complete 32-start block, compare sampled output-head variance and direction dispersion at k={32,24,16,8,4} with the full-gradient statistics over all 32 points. Counts below 32 use 20 fixed-seed nested uniform samples without replacement; k=32 is exact and runs once. Summaries use per-track Pearson/Spearman correlations and equal-track averages, with the 10th-90th sampling-replicate range. Outputs and .complete stay below outputs/train/v0-gradient-proxy-test/local_risk_sparse/.

Last modified: 2026-11-08


## 2026-11-08 Local-Risk sparse-sampling completion

Completed with CPU-only analysis of saved Gram matrices in about 5 seconds. All ten track markers and Gram matrices were read; 396 complete blocks were analyzed, and all 396 k=32 proxy statistics matched the existing block cache. For k below 32, 20 fixed-seed nested samples were evaluated. Summary and replicate tables are under outputs/train/v0-gradient-proxy-test/local_risk_sparse/. No GPU, model, or gradient recomputation was used.

Last modified: 2026-11-08


## 2026-11-08 Local-Risk 64-point sampling extension

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Sparse proxy risk in 64-start blocks | CPU, 2 threads; saved full/head Gram matrices; no GPU or gradient recomputation | All ten proxy-track completion markers | tmux v0-gradient-proxy-local-risk-64; OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.analyze_sparse_proxy_risk_64 --config configs/rage/v0-gradient-proxy-local-risk-64.json | Complete; 194 complete blocks, 20 replicates below 64 points, all 194 64-point baselines generated after Gram shape, symmetry, and finite-value checks; .complete present |

Evaluate k={64,48,32,16,8,4} proxy points per 64-start block. Use 20 fixed-seed nested samples without replacement for k<64 and one exact all-64 baseline. Report full variance and direction-dispersion correlations across blocks, with equal-track weighting and sampling ranges. Output and .complete stay below outputs/train/v0-gradient-proxy-test/local_risk_sparse_64/.

Last modified: 2026-11-08


## 2026-11-08 Local-Risk 64-point completion

Completed with CPU-only analysis of saved full/head Gram matrices in about 5 seconds. All ten tracks were processed; 194 complete 64-start blocks were included, and all 194 full and proxy block targets were computed from every point in their respective 64-start blocks. The run wrote 19,594 sampled-block records and 4,040 per-track/per-replicate correlations. GPU is idle; no model or gradient recomputation was used.

Last modified: 2026-11-08


## 2026-11-08 Local-Risk long-block extension

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Sparse proxy risk at 128, 256, and 576 starts | CPU, 2 threads; saved full/head Gram matrices; no GPU or BP | Ten proxy-track completion markers | tmux v0-gradient-proxy-local-risk-long; OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.analyze_sparse_proxy_risk_long --config configs/rage/v0-gradient-proxy-local-risk-long.json | Complete; 158 total complete blocks across lengths, 20 replicates below full count; 19 pooled 576-start blocks are exploratory |

Keep proxy fractions {100%,75%,50%,25%,12.5%,6.25%}. This yields counts {128,96,64,32,16,8}, {256,192,128,64,32,16}, and {576,432,288,144,72,36}. Use 20 fixed-seed nested samples below full count. At 576 starts, report pooled correlations over only 19 blocks as exploratory; no per-track correlation is estimable with fewer than three blocks per track.

Last modified: 2026-11-08


## 2026-11-08 Local-Risk long-block completion

Completed with CPU-only analysis of saved full/head Gram matrices in about 14 seconds. The run processed 93 complete 128-start blocks, 46 complete 256-start blocks, and 19 complete 576-start blocks. All-point baselines were generated for every complete block. It wrote 15,958 sampled-block records, 13,332 replicate-correlation records, and 144 summary rows. The 576-start result is pooled and exploratory because no track has three complete blocks.

Last modified: 2026-11-08

## v0-vage Static replay completion

The ten-trajectory CPU replay is complete. It reused Gram matrices and BF16 cache markers without model or GPU work. Command: OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_vage_static --config configs/rage/v0-vage.json. Resources: two CPU threads; observed duration about 1.6 seconds. Outputs: outputs/train/v0-vage/. Prospective call accounting: VAGE 204 full-gradient and 204 proxy; Uniform 204 full-gradient and zero proxy. Uniform wins primary reconstruction metrics. No further VAGE batch is queued. The terminated CRB queue remains stopped.

Last modified: 2026-11-08

## v0-vage n_min=1 sensitivity completion

The separate CPU replay using configs/rage/v0-vage-nmin1.json completed in about 1.5 seconds. Outputs and .complete are under outputs/train/v0-vage-nmin1/. The proxy and gradient budgets match the n_min=2 run. Primary reconstruction metrics are worse: mean L2 3.8747 and mean cosine 0.3156. The original n_min=2 outputs remain intact. No follow-on task is queued.

Last modified: 2026-11-08

## v0-vage additional method comparison completion

The CPU replay of Quota, Fusion, Oracle, and the budget-constrained Residual diagnostic completed using scripts/v0_vage_methods.py. Outputs and .complete are under outputs/train/v0-vage-methods-nmin1/. No method beats Uniform on equal-track mean L2 or cosine. Residual is non-informative as an adaptive selector under Bg=Bp and a candidate pool restricted to observed proxies. No GPU work or CRB restart occurred.

Last modified: 2026-11-08

## v0-uni-reconst-scale completion

The CPU-only replay of eight Uniform spacing groups across ten trajectories completed in about 0.8 seconds. Command: OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_uni_reconst_scale --config configs/rage/v0-uni-reconst-scale.json. It used completed Gram matrices and BF16 cache validation, with no GPU or model calls. Outputs and .complete are in outputs/train/v0-uni-reconst-scale/. All 80 trajectory-scale rows passed the q=1 identity and q=64 baseline checks. No follow-up is queued; the terminated CRB queue remains stopped.

Last modified: 2026-11-08

## v0-uni-data-evaluation scheduled

Use the ten completed BF16 full-gradient caches, one RTX 4080, two CPU threads, and a single tmux session. Reference count is ten per track; all 100 are nonadjacent within tracks. Score all non-reference legal starts against all 100 references, then compute lag differences for 1-575. Command: OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_uni_data_evaluation --config configs/rage/v0-uni-data-evaluation.json. Output: outputs/train/v0-uni-data-evaluation/. State: prepared, not yet verified running.

Last modified: 2026-11-08

## v0-uni-data-evaluation completion

The ten-track CUDA scoring and lag analysis completed in tmux after one repaired first-run shape-unpacking error. It used 100 nonadjacent references, 12,736 non-reference query starts, and lag 1-575. Outputs and .complete are under outputs/train/v0-uni-data-evaluation/. The successful run took about 11.8 minutes. GPU is idle. No follow-up is queued, and the terminated CRB queue remains stopped. Full results: docs/results/v0-uni-data-evaluation.md.

Last modified: 2026-11-08

## Pair-mean reconstruction supplements completed

One CPU-only replay evaluated two reconstruction-only batches: v0-vage/pairmean and v0-uni-reconst-scale/pairmean. The existing VAGE anchor JSON files and Uniform budget rule were reused unchanged. Shared command: OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.evaluate_pairmean_reconstruction --config configs/rage/v0-pairmean-supplements.json. Each output subdirectory has trajectory_metrics.csv, summary.csv, and .complete. All work completed in about one second without model, proxy, or GPU calls. No follow-up is queued; the terminated CRB queue remains stopped.

Last modified: 2026-11-08

## 2026-11-09 v0-active-value-rank diagnostic extension

Last modified: 2026-11-09

| Task | Resources | Dependency | Command | State |
|---|---|---|---|---|
| 3-proxy-coverage | CPU, two BLAS threads; cached scores only | Batch 1 complete | PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_active_value_diagnostics --batch 3 | Completed; 100 figures, 7,900 coverage rows, .complete |
| 4-equal-budget-query | CPU, two BLAS threads; cached scores only | Batch 3 and batch 2 complete | PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_active_value_diagnostics --batch 4 | Completed; 13,300 recall rows, 30 random seeds, .complete |

Outputs are in outputs/train/v0-active-value-rank/batch-3 and batch-4 on the data disk. Logs are in logs/v0-active-value-rank. No follow-on computation is queued; this study has finished. The previously user-terminated CRB replay remains stopped.

The aggregate coverage plot and random-seed table were generated after the two batch replays with: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.temp.v0_active_value_summary. Both artifacts were verified present.


## v0-crb-var completion

The CPU replay completed for all ten trajectories using the rage environment and retained BF16 gradient caches. No GPU work or new gradient computation was used. Outputs and .complete are present in outputs/train/v0-crb-var/. The per-block CSV and per-trajectory JSON were summarized in docs/results/v0-crb-var.md. No follow-on task is queued.

## v0-proxy-dp-segmentation completed

| Task | Resources | Dependency | Output | State |
|---|---|---|---|---|
| Longest-four unweighted exact DP replay | rage Conda; RTX 4080 for BF16 proxy reconstruction and blocked Gram products; two CPU threads; four-start positional true-gradient reads | Selected cached proxy and BF16 full-gradient files | outputs/train/v0-proxy-dp-segmentation-longest4-unweighted/ | Complete; 176 interval rows, eight summaries, .complete |
| Longest-four window-weighted exact DP replay | Same | Unweighted pass and same selected caches | outputs/train/v0-proxy-dp-segmentation-longest4-window/ | Complete; 176 interval rows, eight summaries, .complete |
| Validate and report | CPU | Both batches complete | docs/results/v0-proxy-dp-segmentation.md; docs/RESULTS.md | Complete; K, minimum length, coverage, and finite-value checks passed |

Only source ordinals 01, 05, 06, and 09 were replayed. After both result sets were validated, all raw BF16 full-gradient vectors and their cache completion markers were deleted at the user's direction on 2026-11-10. Proxy-gradient caches and derived Gram matrices remain; no raw full-gradient cache remains. The initial whole-file mapping caused cgroup use near 39 GiB; the corrected chunked positional-read method stayed near 12.8 GiB under the 62 GiB cgroup limit. No follow-on task is queued, no automatic inspection is enabled, and the server remains on.

## v1-rage-chronosT5-trev-base-rand / v1-rage-chronosT5-trev-base-full

State at 2026-11-10 02:43 CST: random batch running in tmux `v1-chronost5-trev`; both source pools are materialized. RTX 4080 utilization was 99%, 18.9 GiB / 32 GiB GPU memory; host memory available 456 GiB; data disk 220 GiB free at launch. Experiment Python is `/root/autodl-tmp/rage/envs/rage/bin/python`; PEFT 0.18.1 and python-dotenv 1.2.4 are installed there. The Chronos-T5-Base checkpoint directory exists.

1. Freeze the source profile to Chronos `wiki_daily_100k`, Chronos `m5`, LOTSA `PEMS07`, Monash `solar_10_minutes`, and UTSD `UTSD-1G`. Check actual point counts and storage before fetching; all large data/output/cache artifacts stay under the data-disk-backed project paths.
2. Freeze the 12-task evaluation declaration: four native-univariate FEV tasks (`hospital_admissions_1D`, `M_DENSE_1D`, `us_consumption_1Y`, `world_co2_emissions`) and eight native-univariate TIME tasks (`CPHL/15T`, `CPHL/30T`, `CPHL/H`, `ECDC_COVID/D`, `ECDC_COVID/W`, `epf_electricity_price/H`, `Commodity_Import/M`, `Commodity_Production/M`). Pin upstream revisions, task horizons, metrics, and local data paths before starting training.
3. Fixed-update support is implemented at 0,100,200,500,1000,2000,3000,4000. Each update samples 32 valid 576-point windows with replacement, using 512 context + 64 target. Use BF16 only; verify base, adapter, loss, and AdamW moment dtypes at launch.
4. Rehearse fetch/load, seeded sampling, BF16 LoRA update, and one FEV plus one TIME task. The 1-step BF16 LoRA update and both evaluation paths completed. TIME context NaNs are linearly interpolated from finite observations, with all-invalid contexts zero-filled; the original missing ground truth remains available to metric masking. Remove only rehearsal artifacts after recording results.
5. Started `v1-rage-chronosT5-trev-base-rand` at 2026-11-10 02:42 CST with `PYTHONPATH=src:third_parties/TIME/src:. /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/temp/v1_rage_chronost5_trev.py` in tmux session `v1-chronost5-trev` with unbuffered console output and resumable optimizer state. Verified BF16 model and LoRA, optimizer checkpoints through step 1,000, and live GPU utilization. The script starts `v1-rage-chronosT5-trev-base-full` automatically after the random phase `.complete` and validation.
6. Log each stage and failure in `docs/logs/`; update batch states in `docs/exp/EXP.md`; summarize only verified results in `docs/RESULTS.md`. Never shut down the server.

Formal output roots: `outputs/train/v1-rage-chronosT5-trev-base-rand/` and `outputs/train/v1-rage-chronosT5-trev-base-full/`; logs: `logs/v1-rage-chronosT5-trev-base/`; manifests/configs: `configs/rage/v1-rage-chronosT5-trev-base-rand.json`, `configs/rage/v1-rage-chronosT5-trev-base-full.json`, `scripts/temp/v1_rage_chronost5_trev.py`, `configs/eval/fev/v1-rage-trev-run.yaml`, and `configs/eval/time/v1-rage-trev.yaml`.


## v1-rage-foundation-trev queue (2026-11-10)

| Order | Work | Environment and resources | Dependency | Command/session | State |
|---|---|---|---|---|---|
| 1 | Chronos-T5-Base rand then full; each has eight step checkpoints and FEV+TIME evaluations | rage Conda; RTX 4080 | Fixed source pool | tmux v1-chronost5-trev | Rand .complete; full training active, latest observed step 2,000 |
| 2 | TimerXL 84M rand then full | /root/autodl-tmp/rage/envs/rage-timerxl; RTX 4080; datasets 4.8.5; FEV editable metadata installed | Both Chronos .complete markers | tmux v1-foundation-trev-queue; scripts/temp/queue_v1_rage_foundation_trev.sh | Waiting; heartbeat confirms live |
| 3 | TimesFM 2.5 200M rand then full | /root/autodl-tmp/rage/envs/rage-timesfm; RTX 4080; datasets 4.8.5; FEV editable metadata installed | Both TimerXL .complete markers | Same queue session, after TimerXL | Waiting |

Both new model pairs use the exact Chronos inputs pools by reference, not copies. Effective batch is 32 with microbatch 8; all parameters, LoRA weights, and training are BF16. Each phase runs 4,000 updates at steps 0/100/200/500/1000/2000/3000/4000. The runner evaluates FEV then TIME at each step and removes that step's adapter only after both outputs are complete; adapter 4000 is retained. The environment installer is scripts/temp/prepare_trev_other_models.sh and now pins datasets==4.8.5 plus installs the local FEV package metadata in both model environments. Do not start the foundation runner manually while the queue is active.

TimerXL rehearsal passed a one-step BF16 LoRA update, merged-checkpoint export, FEV task 9, and TIME CPHL/15T short. TimesFM rehearsal passed in tmux v1-rehearse-timesfm25. Its one-step BF16 LoRA loss was finite; FEV task 9 MASE=0.6565; TIME CPHL/15T short MSE=0.1730, MAE=0.2900, MASE=1.2217. The queue remains dependency-gated on Chronos full completion.

## Seed 0-3 Trev queue (2026-11-10)

All queued rand/full phases for Chronos, TimerXL, and TimesFM use BF16 LoRA (r=8, alpha=16) with frozen base weights. `full` selects the full eligible data pool, not full-parameter training. Keep the existing runs and queue; no rerun is required for this clarification.

For all queued Chronos, TimerXL, and TimesFM Trev batches, use MASE alone for result inspection, comparisons, and summaries across FEV and TIME. Other evaluator metrics are raw outputs and do not enter decisions.

Last modified: 2026-11-10

Session: v1-trev-seeds-0-3. Command: bash scripts/temp/queue_v1_rage_trev_seeds.sh. Console: logs/v1-rage-foundation-trev/seed-queue-console.log. GPU phases execute serially. Seed 0 Chronos rand/full are complete; TimerXL rand resumes at evaluation step 0, then TimerXL full and TimesFM rand/full. For each seed 1, 2, 3, prepare Chronos rand/full input pools, start its checkpoint cleanup watcher, then run Chronos rand/full, TimerXL rand/full, TimesFM rand/full. Each phase is resumable from its checkpoint and evaluation outputs. The first checkpoint cleanup watcher is started by the queue after pool preparation. Do not launch a second queue concurrently.

Seed 1 Chronos random and full pools were prepared in tmux v1-trev-seed1-pool at 13:43 CST. Both input manifests validate with sampling_seed=1, and the temporary preparation session exited successfully. The main queue remains active on TimerXL seed 0.

At 13:45 CST, TimerXL seed 0 random completed all 4,000 updates and eight FEV/TIME checkpoint evaluations; .complete exists. The same queue automatically started TimerXL seed 0 full and saved step 100. Seed 1 Chronos pools are ready.

At 13:52 CST, the Chronos input pools for seeds 1, 2, and 3 are all prepared. All six manifests parse, match their respective sampling seeds, and have the expected 19,903 random or 199,030 full series. TimerXL seed 0 full finished training and entered step 0 FEV evaluation; its phase .complete remains pending.

At 13:58 CST, TimerXL seed 0 full completed all eight FEV/TIME evaluations and wrote .complete. The active queue automatically started TimesFM seed 0 random in the rage-timesfm environment. TimerXL seed 0 rand/full are both complete.

At 14:01 CST, TimesFM FEV short-context tasks 18 and 19 passed direct base-checkpoint rehearsals after patch-multiple left padding was fixed. Temporary rehearsal artifacts were removed. The active formal seed 0 random training process continues and will use the corrected adapter at evaluation.

At 15:15 CST, the original seed queue had exited after TimesFM seed 0 random reached 4,000 updates and its step 0 TIME evaluation failed on ECDC_COVID/W due to a 157-point context not divisible by the 32-point TimesFM patch. The TIME adapter now left-pads TimesFM contexts to the next 32-point multiple. A formal --resume evaluation reused the four complete TIME tasks and completed tasks 5-8; step 0 FEV was already complete. Resume the existing seed queue from saved training/evaluation outputs; do not rerun completed phases.

The seed 0-3 queue was restarted in tmux v1-trev-seeds-0-3 at 15:16 CST with tee -a preserving the previous console log. It skipped both completed TimerXL phases and TimesFM seed 0 step 0 evaluations; TimesFM seed 0 random is evaluating step 100 from saved step-4000 training state.

At 16:38 CST, all six seed 0 Chronos, TimerXL, and TimesFM rand/full phases have .complete markers and all eight FEV/TIME checkpoint evaluations each. The live queue moved to Chronos seed 1 random training; step 500 was saved and step 722 observed. Tmux v1-trev-cleanup-seed1 is live. GPU utilization was 98% and the data disk had 211 GiB free.

## Trev queue closed (2026-11-10)

Last modified: 2026-11-10

The user stopped the extension after seed 0. The multi-seed queue, seed 1 runner, and seed 1 checkpoint cleanup watcher have exited. All six seed 0 rand/full phases are complete; no Trev GPU task remains queued. Seed 1 random was interrupted at logged step 2445/4000 and its partial outputs were removed. All other seed 1-3 phases were cancelled before training and their configs/prepared pools were removed. Keep raw logs as historical operation records. Final MASE results: docs/results/v1-rage-trev-seed0-final.md.

## v0-crb-reconst-fine

Last modified: 2026-11-10. One GPU for dense LoRA gradient truth, then CPU replay; two CPU threads, rage Conda environment. Depends on the fixed M5 manifest, bundle, and warmed ChronosTiny checkpoint. Run in tmux v0-crb-reconst-fine with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_crb_reconst_fine`. Output is on the data disk via outputs/train/v0-crb-reconst-fine; log is logs/v0-crb-reconst-fine/run.log. State: completed 2026-11-10; 1,068 gradients, 16 metric rows, .complete verified. Gradient phase resumes from a saved row cursor; replay uses eight exact matched anchor counts. No automatic patrol requested.

## v0-crb-reconst-fine twelve-track extension

Last modified: 2026-11-10. Run 11 remaining fixed ordinals sequentially on one GPU, two CPU threads, in tmux v0-crb-reconst-fine-12. Command: PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_crb_reconst_fine_queue. The queue validates and releases the initial completed track cache, then runs, validates, and releases each new track cache. It stops on any nonzero child exit or failed validation and can be restarted without rerunning completed tracks. State: completed at 17:38 CST; 12/12 tracks validated, 192 metric rows, all dense caches released, twelve_tracks.complete verified. No automatic patrol requested.

## v0-crb-cos-var

Last modified: 2026-11-10. Compute one track per child process, with at most two simultaneous GPU processes and two CPU threads per process. The child projects each full trainable-LoRA gradient immediately using the shared 4096d ParameterHashProjector (seed 42), holds one full 4096d projected trajectory matrix in process memory for matched partition analysis, then releases it after each track. It writes per-block scalar metrics and no gradient cache to disk. Run in tmux v0-crb-cos-var with `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_crb_cos_var_queue`. Queue: 12 fixed ordinals from v0-crb-reconst-fine; maximum concurrency 2. Inspect nvidia-smi and cgroup memory while both children initialize and process gradients. Reduce to one active process if cgroup usage exceeds 75% or GPU memory exceeds 80%. Current state: completed at 18:23 CST; 12/12 tracks, 192 rows, aggregate .complete verified. Peak cgroup memory remained about 15 GiB of 62 GiB and GPU memory about 0.9 GiB. No automatic patrol requested.


## 2026-11-10 v1-rage-crb-vs-uniform-timerXL-84m feasibility pilot

| Task | Resources | Dependency | Command/session | State |
|---|---|---|---|---|
| Measure LoRA projected-gradient throughput and 4096d cosine scoring for hospital_admissions_1D only | rage-timerxl Conda; RTX 4080; 32 FEV refs and 32 full-pool src samples; shared Count-Sketch dimension 4096, seed 42 | Local FEV train Arrow and completed TimerXL checkpoint | PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:. PYTHONUNBUFFERED=1 /root/autodl-tmp/rage/envs/rage-timerxl/bin/python -u scripts/temp/v1_rage_crb_uniform_timerxl_pilot.py | Completed; timing-only output and .complete verified |

The pilot is a timing-only preflight. It must not publish a formal selection artifact or start training. Full source pool has 360,695,805 legal windows; at q=32 each method needs ceil(N/32) gradient anchors per task, so the two-method, twelve-task protocol represents about 270.5 million gradient points before reference gradients. Measured source gradients at 22.64 points/s and 630.35 million projected score pairs/s on a 32x512 timing tile. One full hospital_admissions_1D ref-src sweep extrapolates to 1.47 hours. At K=11,271,744, the two methods across 12 tasks require 270,521,856 source-gradient points, initially estimated at 138.3 days at the scalar-forward rate, before disk I/O, training, and evaluation; this estimate is superseded by the 5.01-day optimized source-gradient estimate below. Do not schedule formal batches until the user reviews the updated cost. The timing output and .complete are in outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/. GPU idle after completion.


## 2026-11-10 v1-rage-crb-vs-uniform-timerXL-84m BF16 gradient optimization pilot

| Task | Resources | Dependency | Command/session | State |
|---|---|---|---|---|
| Verify PEFT-compatible vectorized TimerXL LoRA loss; measure gradient batch 8/16/32, VJP 4/8/16, and optional compiled batch 32 | rage-timerxl Conda; RTX 4080; 96 sampled full-pool windows; BF16 model/LoRA, projection 4096 seed 42 | Initial hospital_admissions_1D feasibility pilot complete | PYTHONPATH=src:third_parties/fev/src:third_parties/TIME/src:. PYTHONUNBUFFERED=1 /root/autodl-tmp/rage/envs/rage-timerxl/bin/python -u scripts/temp/v1_rage_crb_uniform_timerxl_optimize.py | Completed; timing-only; results recorded below |

Do not write selection manifests or train. The measured vectorized per-window losses must match the TimerXL adapter scalar loss within 1e-4 absolute/relative tolerance. Compile cache is directed to the data-disk-backed cache path. Current data-disk free space: 391 GiB.


## v1-rage-crb-vs-uniform-timerXL-84m optimization timing update

Last modified: 2026-11-10

| Work | Resources | Command/session | State |
|---|---|---|---|
| One-task TimerXL BF16 LoRA gradient optimization timing on hospital_admissions_1D | rage-timerxl Conda; RTX 4080; 1,024 sampled full-pool windows; batch 32/64/128; VJP 16/32/64; fixed Count-Sketch 4096 seed 42 | `scripts/temp/v1_rage_crb_uniform_timerxl_optimize.py`; tmux `v1-rage-crb-uniform-opt-pilot` | Completed; timing only; session exited; GPU idle |

Use batch 32 / VJP 32 as the current best measured setting. Sustained 8,192-window runs measured 596.82 and 575.35 gradients/s (0.321 GiB peak); use 575.35/s for conservative planning. VJP 64/128 and batch 64/128 were slower. Batched losses exactly matched scalar adapter losses in the verification sample. `torch.compile` with both `reduce-overhead` and `default` failed during batched-VJP backward due to a tensor without storage; do not enable compile on this path. Stable timing metrics and `.complete`: `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/optimization-stable/`. Updated source-gradient-only projection: 5.44 days for 270,521,856 points at the slower sustained rate; this excludes refs, scoring, I/O/cache writes, training, and evaluation. Wait for user review before formal selection/training.


### v1-rage optimization timing: sustained rate selected for planning

Last modified: 2026-11-10. Added an 8,192-window repeat at batch 32 / VJP 32 to reduce short-run variability. The two sustained runs measured 596.82 and 575.35 gradients/s with 0.321 GiB peak allocated GPU memory. A VJP 128 follow-up did not improve throughput; batch 32 / VJP 32 remains preferred. Use the slower sustained rate, 575.35/s, for scheduling: 270,521,856 source-gradient points project to 5.44 days on one GPU, excluding reference gradients, scoring, data I/O/cache writes, training, and evaluation. Outputs with `.complete`: `optimization-steady-8192/` and `optimization-steady-repeat/` under `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/`. GPU is idle; no formal selection or training has started.


### Same-GPU gradient process parallelism pilot

Last modified: 2026-11-10. Tested 2/4/8 independent TimerXL BF16 LoRA processes on the same RTX 4080, batch 32 / VJP 32, 4,096 source windows per process. Aggregate rates were 964 / 1,087 / 1,159 gradients/s (1.68x / 1.89x / 2.01x the sustained single-process baseline). Per-process PyTorch peak allocation remained 0.321 GiB; eight processes used about 5.7 GiB total GPU memory and reached 100% utilization. Eight workers improved only 6.6% over four and took about 1.4 minutes to initialize due to CPU contention. For a long gradient phase, eight-worker sharding is a promising candidate, but first verify persistent workers, unique source-window partitioning, synchronization, and cache writing. Tentative source-gradient-only estimate: 2.70 days at the measured aggregate rate, excluding those overheads, ref gradients, score computation, training, and evaluation. The concurrent pilot sessions exited; GPU is idle. Artifacts: `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/parallel-two/`, `parallel-four/`, and `parallel-eight/`.


## v1-computational-effort-estimate: 256k x 576-value target

Last modified: 2026-11-10

| Model/work | Environment/resources | Measured or projected time | State |
|---|---|---|---|
| TimerXL 84M BF16 LoRA source gradients, about 61.8k trajectories / 147.456M values, q=32, 12 tasks x 2 methods | rage-timerxl; RTX 4080; batch 32/VJP 32 | 3.38 h single process; 1.68 h at measured 8-process aggregate rate, plus worker startup and I/O | TimerXL pilots complete; parallel path requires production partition/cache validation |
| TimesFM 2.5 BF16 LoRA source gradients, same workload | rage-timesfm; RTX 4080; batch 32/VJP 32 | 82.9 h single process; 69.9 h at measured 2-process aggregate rate | Timing-only pilot complete; 8-process rate not tested |
| Chronos-T5-Base BF16 LoRA source gradients, same workload | rage; RTX 4080; batch 1/VJP 1; exact per-window backward | 262.8 h single process; 131.6 h at 2-process aggregate rate; 107.7 h at 4-process aggregate rate | Timing-only pilots complete; four-process measurement uses 256 windows per process |

The target is 147,456,000 scalar values (256,000 x 576); current mean trajectory length implies about 61,800 randomly sampled trajectories and about 111.94M legal windows. Source-gradient caches are shared across all 12 tasks; assuming one cache per method, the model work uses two caches rather than 24 repeated task-method passes. The final seeded sample will stop at the trajectory prefix whose cumulative point count is closest to target. This remains a planning estimate: no target-sized pool or formal selection/training is queued. The earlier short estimate treated 270k as legal 576-point windows per task; the current target counts raw values across complete trajectories, which create about 111.94M overlapping windows and a much larger gradient workload. Storage estimate: 391 GiB currently free; one 4096d float32 gradient cache per method is about 53.38 GiB, two method caches about 106.76 GiB, and all selected windows materialized across 24 outputs about 576.5 GiB. Stream/release gradient caches and store window indices. Full methodology and measured details are in docs/exp/EXP.md#v1计算量与存储估算.

## src-8192 replacement

Last modified: 2026-11-10. Existing v1-rage batches and timing plans are deprecated due to unsuitable source selection. Retain historical outputs. Prepare and verify exactly 8,192 trajectories in data/src/shard_dataset/src-8192/ from the fixed configs/data/src-8192.yaml index. No old v1-rage queue should resume against its previous pool. Training and selection with src-8192 require a separately updated protocol.

## src-8192 expansion complete

Last modified: 2026-11-10. Chronos and Monash Fetch completed. The Load conversion queue finished after fixing the bounded-reservoir staging path in the project BundleLoadWriter; three Chronos configurations produced no trajectories and were excluded. Eligible coverage is 35 configurations. The new fixed 8,192-trajectory snapshot and index are complete at data/src/shard_dataset/src-8192/ and configs/data/src-8192.yaml. No training job is queued or running for this data-only request.

## Data cleanup complete

Last modified: 2026-11-10. Per user request, all project data except data/src/shard_dataset/src-8192/ were removed. This includes prior source bundles, raw downloads, temporary conversion bundles, and the prior src-8192 version. The retained YAML index references only the fixed snapshot and trajectory metadata.

## v1 并行计时结论用于后续排队

最后修改日期：2026-11-10。后续涉及相同 BF16 LoRA、4096 维投影的梯度准备，按 `docs/results/v1-computational-effort-estimate.md` 的实测最高并行吞吐量安排：Chronos-T5-Base 4 进程（18.052 梯度/秒）、TimesFM 2.5 2 进程（27.827 梯度/秒）、TimerXL 84M 8 进程（1159 梯度/秒）。同一 GPU 上先完成当前模型的队列再启动下一模型；每个模型内部按不重叠轨迹分片，每个进程使用独立缓存目录，全部子任务完成后写整批完成标记。三组计时来自旧候选池的短时测试，正式运行须检查显存、GPU 利用率、实际吞吐和任务成功率，异常时调整并记录。当前 src-96 新批次按 TimerXL 8 进程、TimesFM 2 进程、Chronos 4 进程排队。

## 2026-11-11 src-96 q16 仅 Uniform 梯度队列

最后修改日期：2026-11-11。用户决定停止本批次 CRB，原三模型 CRB+Uniform 队列于 21:03 CST 停止。已删除三模型 CRB 清单并过滤其共享 BF16 梯度缓存，仅保留 Uniform 清单所引用的行；TimerXL 和 TimesFM 的 Uniform 96/96 条及各 worker 完成标记已核验。Chronos 保留已有 Uniform 轨迹及缓存，四进程在 tmux `v0-grad-src-96-uniform` 继续处理剩余轨迹。命令：`bash scripts/temp/run_src_96_uniform_queue.sh`，使用 rage 环境；每 worker 独立缓存，按每轨迹 q=16 锚点预算均衡分配。所有 Chronos worker 成功后，脚本验证三模型完成标记并写整批 `.complete`。输出 `outputs/train/v0-grad-prepare-src-96tr-q16-4096d/formal/`；日志 `logs/v0-grad-prepare-src-96tr-q16-4096d/`。不得重新启动旧 CRB 队列。

## v0-crb-mech-test

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Mock smooth trajectories and CRB/Uniform reconstruction | CPU, two BLAS threads; no GPU | Config and script in place; existing CRB engine | tmux v0-crb-mech-test; PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/v0_crb_mech_test.py --config configs/rage/v0-crb-mech-test.json | Completed: 64 trajectories, 512 records, .complete |
| 64-large-var-k-error-composition: generate mocks, K-error sweep, MSE decomposition | CPU, two BLAS threads; no GPU | User-specified adjacent full-vector ratio 0.5 and lag-16 cosine 0; config and scripts | tmux v0-crb-mech-64-large-var; generator → K-error → MSE | Completed: all `.complete` markers; 12 + 18 figures; decomposition identity max residual 7.55e-15 |

Outputs and .complete: outputs/train/v0-crb-mech-test/; logs: logs/v0-crb-mech-test/. The active TimerXL eight-worker task owns the GPU; this experiment is CPU-only and does not alter it.

2026-11-11 13:23 CST：TimerXL 八进程中的七个已完成，最后一个处理较长轨迹。新入口对后续 TimesFM、Chronos 按每轨迹 q=16 预算作确定性贪心均衡分配，避免固定 ordinal 模分片造成尾部负载不均；输入轨迹、锚点预算与模型参数不变。旧批次结果及缓存已删除。

2026-11-11 13:28 CST：TimerXL 96/96 条完成、八个 worker 和模型完成标记均存在，实际阶段约 15 分钟。队列自动转入 TimesFM 两进程，GPU 检查时占用约 8.1 GiB、利用率 100%。Chronos 仍在后续队列。


## v0-crb-mech-test-ultra-smooth

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| One extremely smooth trajectory, all four matched budgets | CPU, two BLAS threads; no GPU | Original v0-crb-mech-test completed; dedicated config | tmux v0-crb-mech-test-ultra-smooth; PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:. /root/autodl-tmp/rage/envs/rage/bin/python -u scripts/v0_crb_mech_test.py --config configs/rage/v0-crb-mech-test-ultra-smooth.json | Completed: one trajectory, eight records, .complete |

Output: outputs/train/v0-crb-mech-test-ultra-smooth/; logs: logs/v0-crb-mech-test/.


## v0-crb-mech-test block-internal variance

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Compute block L2 and direction variance for original and ultra-smooth mocks | CPU, two BLAS threads; no GPU | Completed saved anchors and deterministic track configs | tmux v0-crb-mech-block-variance; run scripts/v0_crb_mech_block_variance.py for both configs | Completed: 64-track and ultra-smooth analyses, both .complete |

Outputs are separate under outputs/train/v0-crb-mech-test/block_variance/; logs are in logs/v0-crb-mech-test/.


## v0-crb-mech-test CES comparison

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| CES, Uniform, CRB comparison on the same 64 mock tracks | CPU, two BLAS threads; no GPU | Existing deterministic track config and matched CRB/Uniform anchors | tmux v0-crb-mech-ces; scripts/v0_crb_mech_ces.py --config configs/rage/v0-crb-mech-test-ces.json --baseline-metrics outputs/train/v0-crb-mech-test/trajectory_metrics.csv | Completed: 768 records, .complete; 64-track block analysis completed |

CES output: outputs/train/v0-crb-mech-test/ces/; then block analysis reuses all 3 methods' saved anchors. No GPU model work.


## v0-crb-mech-test CES-Oracle

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Oracle-curvature allocation on the same 64 tracks | CPU, two BLAS threads; no GPU | Known analytic mock generator; same pilot split and budgets as CES | tmux v0-crb-mech-ces-oracle; scripts/v0_crb_mech_ces_oracle.py --config configs/rage/v0-crb-mech-test-ces-oracle.json --baseline-metrics outputs/train/v0-crb-mech-test/ces/trajectory_metrics.csv | Completed: 1024 records, .complete |

Oracle output: outputs/train/v0-crb-mech-test/ces-oracle/; run four-method block analysis after anchor metrics complete.


## v0-crb-mech-test CES-Oracle block comparison

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Four-method block statistics using oracle anchors | CPU, two BLAS threads; no GPU | CES-Oracle anchors and all original anchors complete | tmux v0-crb-mech-ces-oracle-blocks; scripts/v0_crb_mech_block_variance.py --config configs/rage/v0-crb-mech-test.json --anchors outputs/train/v0-crb-mech-test/ces-oracle/trajectory_metrics.csv --output outputs/train/v0-crb-mech-test/block_variance/ces-oracle-comparison | Completed: 14,336 block records, .complete |


## v0-crb-mech-test uniform reconstruction sweep

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Reconstruct uniform anchors with cubic, shape-preserving, spectral, GP, and smoothing splines | CPU, two BLAS threads; no GPU | Same 64 deterministic tracks and saved Uniform/CRB/CES anchors | tmux v0-crb-mech-uniform-reconstructors; scripts/v0_crb_mech_uniform_reconstructors.py --config configs/rage/v0-crb-mech-uniform-reconstructors.json --baseline-metrics outputs/train/v0-crb-mech-test/ces-oracle/trajectory_metrics.csv | Completed: 2,880 rows, .complete |

Output: outputs/train/v0-crb-mech-test/uniform-reconstructors/. Smoothing-spline lambda values are fixed sensitivity settings; do not select by evaluation-trace error.


## v0-crb-mech-test Uniform + Cancellation Split

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Cancellation threshold sweep; midpoint and minimum-norm virtual splits | CPU, two BLAS threads; no GPU | Same deterministic 64-track generator; RAGE v7 shape feature and cross-fit opposite-fold Uniform q=32 reference | tmux v0-crb-mech-cancellation-split; scripts/v0_crb_mech_cancellation_split.py --config configs/rage/v0-crb-mech-cancellation-split.json | Completed: 50,176 scoring interval rows; variance follow-up completed with 3,584 track rows, 58,288 block rows, and .complete |

Output: outputs/train/v0-crb-mech-test/cancellation-split/ and variance/. Log group: logs/v0-crb-mech-test/. Also runs scripts/v0_crb_mech_cancellation_variance.py on the saved interval splits; compares equal-block and point-weighted variance to unsplit Uniform.


## v0-crb-mech-test Uniform Linear 六方法对照

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Six virtual partition variants on unchanged Uniform + Linear reconstruction | CPU, two BLAS threads; no GPU | Deterministic 64 mock tracks; cancellation split intervals and matched-K random baselines; all interval split controls | tmux v0-crb-mech-cancellation-six-methods; scripts/v0_crb_mech_cancellation_six_methods.py --config configs/rage/v0-crb-mech-cancellation-six-methods.json | Completed: 7,680 track rows, 107,520 interval rows, .complete verified |

Output: outputs/train/v0-crb-mech-test/uniform-linear-six-methods/. Evaluates block L2/direction variance against unsplit Uniform; pointwise Linear reconstruction is unchanged.


## v0-crb-mech-test Uniform Linear 六方法 est-real 复评

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Evaluate anchor-derived block representatives from Uniform linear reconstruction against true mock values | CPU, two BLAS threads; no GPU | Completed six-method interval selections; same 64 deterministic mock tracks | tmux v0-crb-mech-cancellation-est-real; scripts/v0_crb_mech_cancellation_estimated_rep.py --config configs/rage/v0-crb-mech-cancellation-six-methods.json --interval-selection outputs/train/v0-crb-mech-test/uniform-linear-six-methods/interval_selection.csv --output outputs/train/v0-crb-mech-test/uniform-linear-six-methods/est-real | Completed: 7,680 track rows, 130,912 block rows, .complete verified |

This corrects the earlier real-real oracle variance diagnostic. Estimated representatives use only Uniform anchor endpoints through the original Linear reconstruction; ground truth is used only for evaluation.


## v0-crb-mech-test virtual split selection K curves

| Task | Resources | Dependency | Command or session | State |
|---|---|---|---|---|
| Compare Random-K, Cancellation-K, endpoint-difference, predicted-gain, and metric/weighting-specific Oracle K curves | CPU, two BLAS threads; no GPU | Existing 64 deterministic mock tracks and fixed Uniform anchor positions | tmux v0-crb-mech-virtual-split-selection; scripts/v0_crb_mech_virtual_split_selection.py --config configs/rage/v0-crb-mech-virtual-split-selection.json | Complete; corrected Oracle run .complete verified |
| Decompose est-real MSE into Representation and Estimation components for every K-curve method | CPU, two BLAS threads; no GPU | Completed corrected virtual-split-selection K curves; same tracks and anchors | tmux v0-crb-mech-mse-decomposition; scripts/v0_crb_mech_virtual_split_mse_decomposition.py --config configs/rage/v0-crb-mech-virtual-split-selection.json | Complete; identity residual 8.99e-15; 12 plots verified |

q={4,8,16,32}; K=0..q-1; 100 Random repetitions; point-weighted est-real errors are primary; equal-block metrics auxiliary; 64-track paired bootstrap and interval-score Spearman.

## 2026-11-11 TimerXL 前四轨全窗口梯度插队

最后修改日期：2026-11-11。原 Chronos Uniform 四进程已停止，缓存保留。插队批次 `v0-timerxl-src96-first4-full-grad-4096d-bf16` 共 484,907 个窗口，使用八进程、rage-timerxl 环境。命令 `bash scripts/temp/run_timerxl_first4_then_resume.sh`，tmux `v0-timerxl-first4-full`。八个 worker 全部成功且各有 `.complete` 后写批次完成标记，随后自动执行 `scripts/temp/run_src_96_uniform_queue.sh` 恢复 Chronos Uniform。不得同时运行两个模型的 GPU 梯度任务。

2026-11-11 21:42 CST：插队批次已在 tmux `v0-timerxl-first4-full` 正式启动，八个 worker 存活；GPU 显存约 5.7 GiB，检查时利用率 100%，分片开始写入。全部完成后同一调度脚本自动恢复 Chronos Uniform。

2026-11-11 22:36 CST：TimerXL 前四轨逐窗口梯度批次已完成，八个 worker 和批次完成标记均存在。原调度会话自动恢复 Chronos Uniform 四进程，检查时四个进程存活。新增两批分析在独立 tmux v0-timerxl-grad4-analysis 中顺序完成：v0-timer-XL-grad-4-curvature（完整484,907窗口CPU分析）及 v0-timerXL-grad-4-lm-head（每轨128窗口的TimerXL输出头GPU梯度）。两批次输出和完成标记已核验；分析会话退出，不设置自动巡检。

2026-11-11：启动 v0-timer-XL-grad-4-change-velocity，使用已存储的四轨LoRA投影梯度，GPU上按256个起点分块计算所有k=1..512有效窗口对，不重新计算模型梯度。tmux v0-timerxl-grad4-change-velocity；入口 scripts/temp/v0_timerxl_grad4_change_velocity.py；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/。Chronos Uniform继续运行；本批完成后核验对数、写结果和完成标记，不设置巡检。

2026-11-11 22:56 CST：v0-timer-XL-grad-4-change-velocity 数值计算及分轨迹绘图完成，tmux计算会话退出，输出和图目录 .complete 已核验；无需后续排队或自动巡检。

2026-11-11 23:02 CST：v0-timerXL-grad-4-lm-head 的 change-velocity-comparison 补充批次完成，tmux v0-timerxl-head-velocity 退出；匹配段结果、图和完成标记已核验。无需后续排队或自动巡检。

2026-11-11 23:14 CST：按用户撤回每16窗口抽一个梯度方案，旧stride-16-rank产物不作为现行结论。新two-random-16-blocks从已有缓存读取128梯度并完成k=1..15比较和四张分轨图，输出及图目录完成标记已核验；无需后续调度或巡检。

2026-11-11 23:26 CST：64轨变化速度实验启动。tmux v0-timerxl-grad64 执行 scripts/temp/run_timerxl_grad64_then_resume.sh，顺序为其余60轨LoRA八worker、64轨lm_heads八worker、自动分析、恢复Chronos Uniform。数据盘434 GiB余量，投影梯度新增预计约26.5 GiB加元数据；不存原始维度梯度。前四轨LoRA复用原批次。当前Chronos四进程已暂停并保留缓存；不得并行启动第二组模型梯度任务。本批没有用户授权自动巡检，不设置cron。分析入口将由同批次脚本在梯度完成后运行；结果以完成标记和CSV/JSON为准。

2026-11-12 00:37 CST：梯度准备已完成，分析断点修复后在tmux v0-timerxl-grad64-analysis运行；另有tmux v0-timerxl-grad64-finish等待其结束并按完成标记决定是否写整批完成标记，随后恢复Chronos Uniform。用户要求后台挂任务和ETA，不做主动轮询。分析产物尚未全完成，结果不能提前引用。

2026-11-12：v0-timer-XL-grad-64-change-velocity 的64轨LoRA与输出头4096D投影梯度、自动分析及结果文档均完成，worker和分析完成标记已核验；调度脚本随后自动恢复Chronos Uniform。

## 2026-11-12 spacing-q-dir
最后修改日期：2026-11-12。CPU重放任务，两个BLAS线程，不占GPU；依赖已完成的64轨BF16 LoRA投影梯度缓存。tmux会话 v0-crb-final-spacing-q-dir 执行 bash scripts/temp/queue_v0_crb_final_test_timerxl_64tr_dir.sh。先全局扫描第1百分位阈值，再顺序执行1:0:0、1:0:1、1:1:0、1:1:1及各自匹配Uniform；输出路径为 outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q-dir/。按用户要求不轮询、不设置自动巡检。

## 2026-11-12 spacing-q-dir q=128,256
最后修改日期：2026-11-12。CPU批次；复用既有direction_threshold.json与64轨BF16梯度缓存。tmux v0-crb-spacing-q-dir-extra 顺序执行1:0:0、1:0:1、1:1:0、1:1:1，各比例含匹配Uniform。命令 bash scripts/temp/queue_v0_crb_final_test_timerxl_64tr_dir_q128_256.sh。无自动巡检。

## v1-crb-redo / grad-prepare

最后修改日期：2026-11-12。已固定 src-8192 三模型 LoRA 热身、src-96 q=32 候选梯度准备；CRB coverage:error:split=1:0:1，error 阶段 deprecated，Uniform 逐轨等预算。TimerXL 使用 480+96 原生 V6 损失，其余使用 512+64；总窗口长均为 576。三模型依次热身 1000 次优化更新，然后在相应热身检查点上准备梯度；每模型自身的热身和梯度路径完成后再切换下一模型。热身及梯度输出 outputs/train/v1-crb-redo/grad-prepare/，日志 logs/v1-crb-redo/。队列 scripts/temp/run_v1_crb_redo_grad_prepare.sh，tmux v1-crb-redo-grad-prepare。梯度阶段采用既有最高吞吐配方：TimerXL 8 worker、batch/VJP 32；TimesFM 2 worker、batch/VJP 32；Chronos-T5-Base 4 worker、batch/VJP 1；worker 之间按预估锚点数均衡分配轨迹，各自缓存，模型间不并发。每 worker 必须验证其 CRB/Uniform 清单与缓存后写 .complete；所有 worker 完成后写模型标记，三个模型完成后写批次标记。不得自动关机，不设置自动巡检。正式状态以日志和完成标记为准。

三模型均已通过独立预演：src-8192 上执行 1 次优化更新并保存 LoRA 检查点，在 src-96 的短轨迹 ordinal 1 上重载检查点，完成 CRB/Uniform 各 4 个锚点和共享缓存，各模型实际缓存 6 个唯一窗口。预演不作为正式结果。TimerXL 首次预演发现现有 native V6 loss 对 PEFT 模型读取错误输出字段，已修复为从基础模型的 transformer 和输出头读取；重试通过。正式运行前删除旧无热身 v0-grad-prepare-src-96tr-q16-4096d 输出 3.7 GiB，保留数据快照及其他实验产物。

## v1-crb-redo / TimerXL 基线队列

最后修改日期：2026-11-12 22:03 CST。依赖 `outputs/train/v1-crb-redo/grad-prepare/.complete` 和三模型热身、梯度标记；当前未完成。独立 tmux `v1-crb-redo-timerxl-baselines` 执行 `bash scripts/temp/run_v1_crb_redo_timerxl_baselines.sh`，每 60 秒等待完成标记，原会话消失而无标记则报错退出。依次运行 timerXL-baseline-full 与 timerXL-baseline-rand01；每批训练 4,000 步、评估 4 FEV 加 8 TIME 任务。均独立从预训练权重起训，使用 `rage-timerxl` Conda 环境及单 GPU；当前梯度准备不受影响。详细参数与路径见 EXP.md。此队列不是自动巡检；不设置 cron，不关机。以实际完成标记和结果文件判断状态。

2026-11-12 23:25 CST：grad-prepare 整批完成标记已生成。baseline-full 首次训练到 4,000/4,000 步后，保存清单因数据盘真实路径无法 `relative_to` 项目根目录而异常退出；第 4,000 步 LoRA 与恢复状态均保留。已将清单中的路径改为项目可访问的 `outputs/` 路径，重启原 tmux 队列并从保存状态继续。rand01 仍排在 full 完成之后；当前单 GPU 无其他实验进程。

2026-11-12 23:30 CST：FEV 初次评估因原 `default.yaml` 要求所有 20 项本地 Arrow 缓存、而当前 `data/bench/` 为空而失败；失败输出已保留为 `outputs/eval/v1-crb-redo/timerXL-baseline-full/fev-failed-20261112-2325/`。已补齐选定的四项 FEV parquet，锁定数据集修订；TIME 八项数据由 `scripts/fetch_time_dataset.sh` 按既定 revision 下载并验证。新 FEV 配置及任务文件只为选中四项指定本地数据路径，项目评估代码只要求选中任务的本地文件存在，全部 20 项指标定义仍校验。rand01 训练在 tmux `v1-crb-redo-rand01-train` 运行；独立 tmux `v1-crb-redo-baseline-eval` 执行 `scripts/temp/run_v1_crb_redo_baseline_evaluations.sh`，等待 rand01 训练清单，然后依次评估 full 和 rand01。若任一评估失败则停止，不写批次完成标记。

2026-11-12 23:38 CST：full 与 rand01 两批的训练、FEV 四项、TIME 八项均完成并核验完成标记。评估队列打印 `all evaluations complete` 后退出；当前不再需要后续基线队列。逐任务结果见 `docs/results/v1-crb-redo-timerxl-baselines.md`。
