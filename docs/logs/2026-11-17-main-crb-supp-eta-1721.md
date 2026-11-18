# Chronos-2-small 队列 ETA 估算

- 检查时间：2026-11-17 17:21 CST。
- TimerXL：24/24 任务、384/384 评估点已完成。
- Chronos-2-small：run 从 14:14:04 开始；8/24 任务、128/384 评估点完成。当前第 9 个任务 `fev-ett-1h` 六臂训练约 3800–4000/4000 step，GPU 利用率 99%。前八个任务整体约 20–23 分钟/任务。
- 剩余工作量估算：剩余普通任务训练、选样和评估约 5 小时；新增 12 个大任务的 safe-reference 合计约 27.95M 合法窗口，按 q8 约 3.49M 梯度锚点。实测 reference 梯度吞吐约 185–230 anchors/s，单此部分约 4.2–5.2 小时。
- ETA：从本次检查起约 9–12 小时，预计 2026-11-18 02:30–05:30 CST。主要误差来源为 `azure2019_D/5T`（约 21.3M 合法窗口）和目标数据读取速度；该范围为估算，后续以任务日志刷新。
- 进度来源：`logs/main-crb-supp-chronos2-small/run_20261117141404.log`、各 task 原始 MASE 文件、`docs/exp/main-crb-supp-chronos2-small/results.md`。
