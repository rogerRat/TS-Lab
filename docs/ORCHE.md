# Task Orchestration

Last modified: 2026-11-10

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
| CPU 参数搜索与重放 | 最多 4 个工作线程、BF16 内存映射文件 | 全量梯度完成 | tmux v0-crb-batch-01/02；队列按完成状态补充下一批 | 执行中；280/1,280 配置完成，18/80 个基准完成；批次 00 全部完成，批次 01 和 02 活跃。最新逐批配置计数见 docs/LOG.md |

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
6. Log each stage and failure in `docs/LOG.md`; update batch states in `docs/EXP.md`; summarize only verified results in `docs/RESULTS.md`. Never shut down the server.

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
