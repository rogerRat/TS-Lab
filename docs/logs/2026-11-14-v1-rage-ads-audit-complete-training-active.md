# v1-rage-ads 正式审计完成与训练阶段记录

- 检查时间：2026-11-14 12:46 CST
- 批次：rolling-eight-task
- 状态：正式审计完成；12-worker训练/rolling评估运行中
- tmux：v1-rage-ads-rolling-optimized

## 正式审计

- 输出：outputs/selection/v1-rage-ads/rolling/audit/
- 配置：八任务、四臂、每臂4,000次有放回抽样，seed42，梯度window/VJP batch32/32，source q32全体83,330个梯度锚点。
- 运行时间：165.3秒；单任务墙钟103.7–162.4秒。
- 资源：GPU均值82.1%、显存均值/峰值4,683/15,218 MiB；进程树RSS均值/峰值27,985/37,988 MiB；主机内存均值/峰值179,772/187,941 MiB；worker CPU均值6.48核心当量、主机CPU均值16.1%。
- 跨任务 source q32 best-reference均值0.1722；fullRand/rage/ancUni/Traj抽样均值0.1712/0.2053/0.2072/0.2148；相对source基线提升-0.0010/+0.0331/+0.0350/+0.0426。
- Reference continuity best-history均值0.3831，平均p50为0.3665；history-test pairwise direction cosine均值0.0039。
- 32个score-matrix audit rows的raw exact-zero比例均为0；abs(score)<=1e-3平均比例1.13%；Top-rho mask零比例约79.96%。source P99 top-10轨迹份额均值83.8%，有效轨迹数均值9.29/96。
- Manifest：outputs/selection/v1-rage-ads/rolling/audit/parallel-run-manifest.json；指标JSONL：source-alignment.jsonl、arm-alignment.jsonl、reference-continuity.jsonl、score-matrix.jsonl、source-p99-clustering.jsonl；根 .complete 已生成。

## 训练与评估运行观察

- 正式runner于2026-11-14 12:39:39 CST开始训练/评估，配置12 workers、32个task-arm cells、每cell4,000 optimizer updates、rolling评估。
- 2026-11-14 12:44:59的tmux日志截面中，多项训练任务处于约2,100–2,200/4,000 updates；这是截面观测，不是全体任务汇总。
- 运行时继续记录worker心跳和阶段输出；单元完成、CPU/GPU/RSS采样和最终模型指标以runner manifest及每cell .complete为准。
- 暂无完整训练和rolling评测汇总指标。按本轮用户要求，使用12 workers的已有安全余量配置继续运行；仅当实际运行接近资源边界、发生CUDA/CPU OOM或任务失败时再降并发并从完成标记恢复。
