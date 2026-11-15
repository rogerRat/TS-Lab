# v1-rage-ads-v2 运行器精简后的批次状态

- 检查时间：2026-11-15 02:03 CST
- 实验代号：v1-rage-ads-v2
- 批次：rolling-baseline-v2

## 批次状态

- 训练和 rolling 评估于 02:00:43 CST 自然结束；之后检查时没有剩余 tmux 会话或训练、审计、监控进程。
- selection 根完成标记存在；train 与 eval 根完成标记存在，task-arm 完成标记均为 40/40。
- 最终状态已同步到 docs/exp/v1-rage-ads-v2/results.md、docs/EXP.md 与 docs/RESULTS.md。
- 当前批次的运行器在本次修改前已载入旧代码，最终日志仍写 completed=31/32 和 cells=31；完整标记计数为 40/40，包含启动前已有的 9 个完成单元。修复后的源代码将按总数显示 40/40，并单独记录本轮新完成数。

## 维护修正

- 静态检查发现恢复脚本需要分别保存全部任务列表和待恢复列表；已改为 all_tasks 与 tasks 两组，保持恢复范围和行为不变。
- 本次没有启动、停止或重跑实验，也没有改动训练结果。
