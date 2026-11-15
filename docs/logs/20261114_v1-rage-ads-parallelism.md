# v1-rage-ads 并行效率巡检与修复

- 检查时间：2026-11-14 11:46 CST
- 实验代号：`v1-rage-ads`
- 批次：`rolling-eight-task`、`rolling-parallelism-profile`
- 远端：`ssh tsl`，项目 `/root/TS-Lab`，Conda 环境 `rage-timerxl`

## 状态与证据

- 八任务四臂选择已完成，`outputs/selection/v1-rage-ads/rolling/.complete` 存在。原队列按任务与训练臂串行执行；各任务耗时 44.67–50.52 分钟，均值 47.75 分钟，日志中的总计约 6.4 小时只记为历史串行基线，不作为并行方案的实际耗时或耗时估计。
- 按相邻臂完成时间戳求得串行单臂耗时：rage 18.75–25.28 分钟（均值 22.67），ancUni 11.87–14.32 分钟（均值 12.80），Traj 11.35–14.93 分钟（均值 12.27），fullRand 约 0 秒。
- 旧选择日志没有 GPU 或主机内存的时间序列，平均占用未记录；单点 GPU 0%、显存 922 MiB 不作为平均值或阶段性结论。
- 正式审计在 11:09:57 因 `ModuleNotFoundError: timebench` 中断，审计根 `.complete` 不存在。此前 `np.divide` 的 `where` 广播问题已修正；`audit_metrics.py` 现显式加入项目、FEV、TIME import 路径。
- 旧 audit JSONL 有部分行，但没有完整任务标记与最终 manifest。正式并行审计将按任务重算并由文件锁安全 upsert，完成后单独 finalize。

## 本次修改

- 修改 `scripts/exp/v1-rage-ads/train_eval.py`：增加 ADS descriptor 紧凑抽样适配，避免训练反复构造 2,666,312 维概率向量并调用 O(N) dense weighted choice。
- 修改 `scripts/exp/v1-rage-ads/benchmark_sampler.py`：基于相同概率质量重建 dense 抽样基线，确保 dense 与 compact 比较同分布。
- 修改 `scripts/exp/v1-rage-ads/benchmark_gradient_parallelism.py`：吞吐分母改为同步释放后的 worker compute makespan，避免把模型初始化/热身时间混入纯计算吞吐。
- 修改 `scripts/exp/v1-rage-ads/audit_metrics.py`：为并行任务写入增加 `fcntl` 文件锁；新增按任务/标签分离输出与 finalize-only；任务抽样 seed 按固定任务索引生成，确保串行/并行调度不改变抽样。
- 新增 `scripts/exp/v1-rage-ads/parallel_profile.py`、`run_parallel_audit.py`、`run_parallel_train_eval.py`：分阶段并行度试跑、GPU/主机内存监控、心跳和中断时优雅停止子进程；正式 runner 支持完成标记续跑。
- 更新 `docs/exp/v1-rage-ads/exp.md`、`results.md`、`docs/EXP.md`、`docs/RESULTS.md`，记录串行计时口径、资源采样缺失、现有审计失败以及并行试跑协议。

## 已有性能试跑

- CPU 抽样微基准：FEV09/rage、2,666,312 个 source 窗口、400 轮×32 draws。dense weighted choice 1,328.9 draws/s；compact descriptor sampler 994,688.9 draws/s，约 748.5 倍。该指标只表示抽样操作吞吐，不外推为完整训练加速。
- 首轮 warmed 梯度短试跑：TimerXL step-1000 warmed probe；窗口批次/VJP 设置比较后，batch=32、VJP=16 单 worker 为 442.69 windows/s，较 batch=4/VJP=4 的 117.66 windows/s 高约 3.76 倍。
- 1/2/4/8 worker 首轮总吞吐分别为 442.69/707.49/991.31/1,100.35 windows/s。8 worker 首轮显存均值 4,190 MiB、峰值 7,772 MiB，GPU 利用率均值 42.3%、峰值 100%；仅 10 个约 1 秒样本，当前扩为每 worker 1,024 窗口复测。
- 性能输出：`outputs/selection/v1-rage-ads/parallelism/sampler-benchmark.json`、`gradient-parallelism-benchmark.json`；日志位于 `logs/v1-rage-ads/`。

## 下一步

- 完成 1,024 窗口/worker warmed 梯度复测。
- 依次运行 32-step 训练、TIME01 audit、完整 TIME01 rolling 评估的 1/2/4/8 worker 试跑，记录吞吐、GPU 利用率与显存、主机内存及进程 RSS 均值/峰值。
- 选择稳定并发配置后，通过 `run_parallel_audit.py --workers N` 续跑正式 4,000 抽样审计；然后通过 `run_parallel_train_eval.py --workers N` 完成八任务四臂训练与 rolling 评估。
- 训练和评估结果当前为空；不推测或填入预期指标。不得关机。
