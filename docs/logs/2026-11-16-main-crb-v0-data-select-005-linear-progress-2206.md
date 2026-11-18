# data-select-005-linear 进度巡检

- 检查时间：2026-11-16 22:06:09 CST
- 批次：`main-crb-v0/data-select-005-linear`
- 状态：tmux `main_crb_data_select_005_linear` 仍存活；训练完成标记 26 个，评估完成标记 48 个。ORI、REFT、rand、GraNd-static、CRB、UNI 各八项评估完成；CRB-traj、UNI-traj、TSDS、LP-Mirror 尚未评估。
- 运行进度：剩余 task-conditioned 队列日志约为 2,000–2,100/4,000 步。
- GPU：99%，显存 7,756 / 32,760 MiB。
- MASE 当前最优：FEV09 CRB 0.723056；TIME1 UNI 0.717966；TIME2 rand 1.278146；TIME3 REFT 1.237555；TIME4 REFT 2.601949；TIME6 REFT 1.024618；TIME7 rand 1.588266；TIME8 ORI 1.107539。结果表和 MAE/MSE 已更新在 `docs/exp/main-crb-v0/results.md`。
- 当前无 GPU 空闲或训练失败信息；阶段结果仍不包含四个未完成方法。
