# data-select-005-linear 中止记录

- 停止时间：2026-11-16 22:25 CST；原因：用户要求停止当前批次。
- 停止方式：向八个 task worker 独立进程组发送 SIGINT；子训练进程随组退出。随后确认目标进程均消失，tmux `main_crb_data_select_005_linear` 会话不存在，GPU 利用率 0%、显存 0 MiB。
- 最终状态：selection 完成标记 58；train 完成标记 42；eval 完成标记 64。完成评估的方法组为 ORI、REFT、rand、GraNd-static、CRB、UNI、CRB-traj、UNI-traj，各八任务。
- 部分工作：TSDS 八任务训练日志显示停止时位于约 3,000–3,300/4,000 步；未写入 train `.complete` 或最终 checkpoint。LP-Mirror 尚未开始训练评估。中止训练日志保留在 `logs/tools/train/`。
- 结果：最终八方法 MASE 表以及逐任务最低值见 `docs/exp/main-crb-v0/results.md`。所有已完成训练、评估、选择资产和指标均保留在 `outputs/` 对应批次目录。
- 批次状态文件：`outputs/eval/main-crb-v0/data-select-005-linear/.stopped`。未创建 `.complete`；不删除部分产物。
