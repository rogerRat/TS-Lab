# main-crb-v0 其他指标汇总与 native 队列修复

- 检查时间：2026-11-16 21:03 CST
- 批次：`q16-src256-eight-task-resample`
- 从各任务已完成评估文件读取 MAE、MSE、RMSE、MAPE、sMAPE、ND、CRPS、SQL、WAPE、WQL，并将逐方法矩阵写入 `docs/exp/main-crb-v0/results.md`。FEV09 使用 `summary.csv`；TIME 使用 `metrics.npz` 数组均值。指标方向均按越低越好确定本批当前最佳方法。
- 实际结果：8 个方法组完成 64 项评估；FEV09 与七个 TIME 任务的 MASE 和其他可用指标均来自原始结果文件，不从其他指标推导。TIME 的 ND 与 CRPS 数组在已读文件中数值相同，按原值分别保留。
- TSDS：八任务 OT 轨迹概率已生成，输出位于 `outputs/selection/main-crb-v0/q16-src256-eight-task-resample/tsds/<TASK>/`；尚无预测模型评估结果。
- LP-Mirror：首次完成 source proxy 和 FEV09 全池打分后，构造 resampling payload 时因 native `total_equivalents` 校验口径不符而失败。已修正为完整轨迹等价数，并于 2026-11-16 21:03 CST 在 tmux `main_crb_resample_native_run` 重启原生概率准备和训练队列。LP-Mirror 尚无评估结果。
- 详细全矩阵见 `docs/exp/main-crb-v0/results.md`；任务结果文件仍在 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/`。
