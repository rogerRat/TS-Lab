# v1-rage-ads 最终远端状态复核

- 检查时间：2026-11-14 13:26:54 CST。
- 实验批次：`v1-rage-ads` / `rolling-eight-task`。
- 训练完成标记：32/32 单元，根 `.complete` 存在。
- 滚动评估完成标记：32/32 单元，根 `.complete` 存在。
- 下游汇总：`downstream-metrics.jsonl` 有 32 行，`.summary-complete` 存在；八任务×四臂结果与全局实验索引一致。
- 队列状态：更新为 `status=0`；汇总在 13:22:44 CST 完成。
- 资源状态：GPU 利用率 0%，显存占用 0/32,760 MiB；未关机。
- TIME08 故障原因、输入补齐修复、恢复运行、计时与资源记录限制见 `2026-11-14-v1-rage-ads-time08-recovery-complete.md`。
