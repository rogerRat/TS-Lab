# 五任务跨模型 step-4000 结果总结

- 操作时间：2026-11-17 19:36 CST。
- 操作内容：汇总 `time-crypto-d`、`time-ne-china-wind-h`、`time-sg-pm25-h`、`fev-jena-weather-1h`、`fev-ett-1h` 五个任务在 TimerXL 84M 与 Chronos-2-small 两个模型上的 step-4000 结果，并整理为本批次实验设置细节。
- 新增文档：`docs/exp/main-crb-supp/main-crb-supp-five-task-two-model-step4000.md`。
- 数据核验：五个任务 × 六臂 × 两模型共 60 个 step-4000 评估目录均存在 `.complete` 与原始指标文件（TIME 为 `metrics.npz`，FEV 为 `summary.csv`）。
- 队列状态：TimerXL 24/24 任务完成（2026-11-17 14:07 CST）；Chronos-2-small 已完成 14/24 任务，截至 19:32 CST 正在运行 `time-open-electricity-nem-5t`，本批次五个任务均已完成。
- 关键结果（step 4000）：
  - Crypto/D：TimerXL 最优 E2 Uniform-Shape 7.1397；Chronos 最优 E2 Uniform-Shape 6.2169。
  - NE_China_Wind/H：TimerXL 最优 E3 CRB-Mean 0.5612；Chronos 最优 E0 Rand 0.5378。
  - SG_PM25/H：TimerXL 最优 E4 CRB-Shape 0.8834；Chronos 最优 E3 CRB-Mean 0.8944。
  - jena_weather_1H：TimerXL 最优 E4 CRB-Shape 0.5142；Chronos 最优 E0 Rand 0.4558。
  - ETT_1H：TimerXL 最优 E4 CRB-Shape 1.1594；Chronos 最优 E4 CRB-Shape 1.1575。
- 口径说明：只报告 step 4000；TIME 标量 MASE 为逐窗口、逐变量 MASE 的均值；不同任务 MASE 不跨数据集平均。
- 未改动任何训练或评估产物。
