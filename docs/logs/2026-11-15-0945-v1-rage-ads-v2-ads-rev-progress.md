# v1-rage-ads-v2 ads-rev 进度与匹配结果

- 检查时间：2026-11-15 09:45 CST
- 批次：ads-rev-reverse-complement
- 选择状态：8/8 完成；各任务 `outputs/selection/v1-rage-ads-v2/<task>/ads-rev/.complete` 已生成，汇总标记位于 `outputs/selection/v1-rage-ads-v2/ads-rev/.complete`。
- 训练/评估状态：8 个训练进程在 09:42:30 启动；逐任务训练约 1,100–1,200/4,000 步；训练与评估 `.complete` 暂为 0/8。尚无 MASE。
- 资源：GPU 利用率约 99%，显存约 7,756 MiB；容器内存约 19.9/66.6 GB；`memory.events` 的 oom/oom_kill 均为 0。训练 runner heartbeat 显示 8 active、0 failed。
- 矩阵口径：最终配对按 `cos>0.1` 统计；反向排除使用 `cos>=0.1`。八任务最终命中 98,904,745/1,761,504,746 个矩阵元素（5.615%）；q0 命中 98,792,454 对，q1=16 新增 112,291 对；source 锚点唯一命中 1,275,774/2,140,607（59.60%），reference 锚点唯一命中 8,700/8,700。逐任务值见 `docs/exp/v1-rage-ads-v2/results.md`。
- 中断与恢复：第一次选择运行完成全部 8 个任务产物，但生成臂级汇总文件时因未创建 `outputs/selection/v1-rage-ads-v2/ads-rev/` 父目录而报错；未启动训练。已修复脚本创建目录并跳过已有 `.complete` 任务。第二次启动复用了八项选择产物，未重算梯度，成功写入汇总标记并于 09:42:30 启动 8 个训练单元。
- 日志：`logs/v1-rage-ads-v2/ads-rev-pipeline.log`、`logs/v1-rage-ads-v2/ads-rev-selection_20261115034952.log`、`logs/v1-rage-ads-v2/ads-rev-selection_20261115094224.log` 及八个 `train-eval-<task>-ads-rev_20261115094232.log`。
