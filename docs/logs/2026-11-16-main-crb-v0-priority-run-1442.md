# main-crb-v0 q16 优先队列运行记录

- 检查/操作时间：2026-11-16 14:44 CST。实验批次：`q16-src256-time1-q8-four-arm`。
- 当前状态：方向资源已完整；ORI TIME1 rolling 评估完成；CRB 与 UNI 并行训练。14:43:25 两臂各 200/4,000 步，GPU 利用率 87%，显存 1,942 MiB。队列为 ORI → CRB/UNI 并行 → REFT。
- ORI 指标：MSE=0.0640、MAE=0.1519、RMSE=0.1746、MAPE=0.2320、sMAPE=0.2001、MASE=0.7206、ND=0.2187、CRPS=0.2187。指标路径：`outputs/eval/main-crb-v0/q16-src256-time1-q8/ori/CPHL/15T/short/metrics.npz`；`.complete` 已核验。
- 失败与修复：14:35 使用基础 Python 的入口快速退出，ORI 评估未完成。训练环境 `rage-timerxl` 中 TIME backend 最初报缺少 `timebench`；独立 `time` 环境虽有 TIME 依赖，但 Transformers 4.57.1 与 TimerXL 自定义模型使用的 `DynamicCache.seen_tokens` 不兼容。将 `third_parties/TIME/src` 加入 `PYTHONPATH`，在 `rage-timerxl`（Transformers 4.40.1）运行评估后，ORI 成功完成。无代码或第三方源码修改。
- 并行启动命令由 `scripts/temp/run_priority_main_crb_v0.py` 维护，训练与评估调用复用 `scripts/tools/train.py`、`scripts/tools/eval.py`。训练及评估输出保留在项目 `outputs/`，详细运行日志位于 `logs/main-crb-v0/` 和 `logs/tools/`。
- ETA：按当前训练速率估算，CRB/UNI 两臂约 6–8 分钟后完成训练及评估；随后 REFT 另需一次 4,000 步训练和评估，按相近吞吐粗估约再 6–8 分钟。ETA 会随实际吞吐变化。
- 尚无 CRB、UNI、REFT 指标；下一步待办为等待两并行训练臂完成并评估，然后运行 REFT、追加实际指标并同步更新索引。
