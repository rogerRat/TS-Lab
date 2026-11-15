# v1-rage-ads-v2 容器重启后恢复

- 检查时间：2026-11-14 19:23 CST
- 实验代号：v1-rage-ads-v2
- 批次：rolling-baseline-v2
- 状态：选择阶段恢复运行；训练与评估尚未开始。

## 容器状态

- 容器重启后没有遗留实验进程或 tmux 会话。cgroup 当前内存约 0.9 GB / 66.6 GB，GPU 利用率 0%，显存 0 MiB；OOM kill 计数仍为 0。
- 输出目录中的完成标记保留。完整选择任务为 3/8：FEV09、TIME01、TIME08；其余 TIME02、TIME03、TIME04、TIME06、TIME07 待续跑。TIME07 的部分 arm 文件保留，任务级未完成标记会触发恢复。
- 已完成任务的实际 source/reference 锚点预算分别为：FEV09 783,764 / 744，TIME01 830,581 / 458，TIME08 205,819 / 4。以上是选择阶段计数，不是评测指标。

## 恢复操作

- 以 `bash scripts/exp/v1-rage-ads-v2/recover_selection.sh --resume` 启动恢复，最多两个任务并发，跳过三个已完成任务；顺序处理其余五项。
- 所有选择任务完成后，恢复脚本写入 selection 根 `.complete`，并启动 `run_training_after_selection.sh`，以 12 workers 运行剩余训练/rolling 评估单元。
- 10 分钟状态采集会话 `v1_rage_ads_v2_monitor` 已重启；19:22 首次记录容器内存 919,666,688 bytes、GPU 0%。
- 本次恢复 tmux 会话：`v1_rage_ads_v2_recovery`；monitor 会话：`v1_rage_ads_v2_monitor`。
