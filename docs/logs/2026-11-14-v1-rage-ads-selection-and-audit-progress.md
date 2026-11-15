# v1-rage-ads 选择完成与审计续跑

- 状态检查时间：2026-11-14 11:01:34 CST。
- 选择阶段：重启后确认八任务四臂共 32 个 arm `.complete`、八个 task `.complete` 与选择根 `.complete` 均已生成。选择阶段日志 `logs/v1-rage-ads/rolling-eight-task_20261114042750.log` 在 2026-11-14 10:51:02 记录八任务完成，总耗时 6.4 小时。
- 审计阶段故障：`logs/v1-rage-ads/rolling-eight-task_20261114105106.log` 在 FEV 9 source 对齐扫描后报错。`p1_unit_rows()` 将长度为 N 的 `where` 条件用于 N×D 除法，NumPy 无法广播 `(N,)` 到 `(N,D)`。
- 修复：将掩码改为 `norms[:, None] > 0`；用 3×4 梯度样例运行 `p1_unit_rows()` 确认方向归一化与 P1 阈值过滤结果正确。队列脚本增加选择与审计完成标记检查：重启时跳过已完成选择；部分审计失败则重跑审计并沿用已落盘行。
- 续跑：2026-11-14 11:01:16 CST 重启同一 tmux session `v1-rage-ads-rolling`。队列跳过已完成选择，直接进入 `audit_metrics.py --task all --sample-count 4000 --seed 42`。截至 2026-11-14 11:01:34 CST，日志已完成 FEV 9 source q32 对齐扫描；审计根 `.complete` 尚未生成。
- 训练/评估：尚未开始，完成标记均为 0；待八任务审计结束后由队列自动执行。
- 实际指标：当前没有完整审计数值、训练或 rolling 评估结果。本记录只报告状态与已验证文件。
- 下一步：让审计自然完成，再由队列继续训练、rolling 评估和结果汇总；不轮询。
