# main-crb-v0 resample 队列加速检查

- 检查时间：2026-11-16 21:11:38 CST
- 批次：`q16-src256-eight-task-resample`
- GPU快照：RTX 4080 利用率 99%，显存 7,756 / 32,760 MiB；八个 TSDS 任务的 `scripts.tools.train` 进程同时启动，各占用约 964 MiB 显存。当前已满载，无需继续增加并发以免争用计算资源。
- native 选择：TSDS 八任务概率完成；LP-Mirror 八任务概率完成，source proxy 缓存命中。准备阶段于 21:11:18 CST 结束，修复了 LP `total_equivalents` 校验口径后训练队列正常启动。
- 已完成的八组既有评估指标保存在 `docs/exp/main-crb-v0/results.md`；TSDS/LP-Mirror 的预测模型训练评估仍在进行，尚无新 MASE。
