# main-crb-v0 q16 四臂最终巡检

- 检查时间：2026-11-16 14:58 CST；批次 `q16-src256-time1-q8-four-arm`。REFT 于 14:56:51 完成 TIME1 rolling 评估；tmux `main_crb_q16_time1` 已退出；没有训练或评估相关进程。
- 已核验 ORI、CRB、UNI、REFT 各自 train/eval 完成标记（ORI 训练标记对应底座检查点），以及 train/eval 根目录 `.complete`。四个 `metrics.npz` 和 `time_metadata.json` 均已生成。
- 四臂指标（各指标文件逐元素平均，四位小数）：

| 方法 | MSE | MAE | RMSE | MAPE | sMAPE | MASE | ND | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ORI | 0.0640 | 0.1519 | 0.1746 | 0.2320 | 0.2001 | 0.7206 | 0.2187 | 0.2187 |
| CRB | 0.0637 | 0.1485 | 0.1710 | 0.2198 | 0.1957 | 0.7085 | 0.2084 | 0.2084 |
| UNI | 0.0633 | 0.1494 | 0.1718 | 0.2212 | 0.1976 | 0.7128 | 0.2101 | 0.2101 |
| REFT | 0.0771 | 0.1611 | 0.1866 | 0.2516 | 0.2089 | 0.7635 | 0.2394 | 0.2394 |

- 数据路径：`outputs/eval/main-crb-v0/q16-src256-time1-q8/<arm>/CPHL/15T/short/metrics.npz`。无错误或未解决事项。
