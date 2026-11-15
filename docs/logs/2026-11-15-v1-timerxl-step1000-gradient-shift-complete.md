# TimerXL step-1000 梯度差异实验完成记录

- 完成时间：2026-11-15 15:14:33 CST。
- 批次：`paired-src96-step1000`；样本 6,144 个，96 条轨迹，每轨 64 个窗口。
- 状态：原始、step-1000 梯度计算及轨迹聚类 bootstrap 均完成；输出目录 `outputs/train/v1-timerxl-step1000-gradient-shift/` 的 `.complete` 存在。
- 结果：方向差均值 0.5653，95% CI [0.5256, 0.6054]；平均余弦 0.4347；投影相对 L2 差均值 7.1115，95% CI [4.9393, 10.2253]；范数比均值 7.3737，95% CI [5.2048, 10.5005]。
- 数值核验：原始组重复计算最大 sketch 绝对误差 7.45e-9；两组零范数梯度均为 0。
- 结论：该固定 checkpoint 上的梯度方向和幅值均与原始组显著不同；推广范围限于本批样本与投影。
- 证据：`summary.json`、`trajectory_metrics.csv`、`window_metrics.csv`、逐组梯度 sketch 和主运行日志。
