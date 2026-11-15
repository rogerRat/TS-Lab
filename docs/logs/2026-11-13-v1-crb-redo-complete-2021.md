# v1-crb-redo CRB win 批次完成巡检

检查时间：2026-11-13 20:21 CST。tmux 队列日志记录 CRB win 五项任务全部完成，并写入 `outputs/train/v1-crb-redo/timerXL-ours-rage101-01-win-rr/.complete`。FEV 9、TIME 1–4 每项的选择、训练、评估 `.complete` 均存在；训练检查点已按协议清理。TIME 4 评估于 17:29 CST 完成，TIME CPHL/H 的 MASE 为 1.455521，TIME ECDC_COVID/D 的 MASE 为 3.167945。所有五项的 MASE、配置生成的全部辅助指标、选择耗时、BF16 矩阵峰值和选择文件大小已读取真实产物并写入 `docs/exp/v1-crb-redo/results.md`；同步更新实验设计及全局索引。

截至检查时 GPU 利用率 0%、显存 0 MiB。CRB win 队列会话已退出；其他指标整理会话仍存在，属于独立补充分析，不改变 CRB win 批次完成状态。未发现 CRB win 错误或未完成项。
