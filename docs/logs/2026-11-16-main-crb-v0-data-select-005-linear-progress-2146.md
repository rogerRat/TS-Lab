# data-select-005-linear 进度巡检

- 检查时间：2026-11-16 21:46:41 CST
- 批次：`main-crb-v0/data-select-005-linear`
- 状态：tmux `main_crb_data_select_005_linear` 存活。训练 `.complete` 10 个，评估 `.complete` 32 个；完成方法为 ORI、REFT、rand、GraNd-static 各八任务。CRB/UNI/轨迹变体/TSDS/LP-Mirror 尚无已完成训练评估标记；训练日志推进至约 200–400/4,000 步。
- GPU：利用率 99%，显存 7,756 MiB / 32,760 MiB。
- 阶段指标：已从八任务 FEV09 `summary.csv` 与 TIME `metrics.npz` 汇总 MASE、MAE、MSE，逐方法矩阵和阶段最优 MASE 已记录于 `docs/exp/main-crb-v0/results.md`。
- 当前阶段 MASE 最优：FEV09 ORI 0.731862；TIME1 ORI 0.720606；TIME2 rand 1.278146；TIME3 REFT 1.237555；TIME4 REFT 2.601949；TIME6 REFT 1.024618；TIME7 rand 1.588266；TIME8 ORI 1.107539。未完成方法未参与比较。
- 日志末尾显示八任务 worker 正并行训练；无异常退出信息。
