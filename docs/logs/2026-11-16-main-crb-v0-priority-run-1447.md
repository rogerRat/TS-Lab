# main-crb-v0 q16 优先队列进度更新

- 检查时间：2026-11-16 14:47 CST；批次 `q16-src256-time1-q8-four-arm`；tmux `main_crb_q16_time1` 仍在运行。
- ORI TIME1/CPHL/15T rolling 评估已完成，输出 `.complete` 存在；实测 MSE=0.0640、MAE=0.1519、RMSE=0.1746、MAPE=0.2320、sMAPE=0.2001、MASE=0.7206、ND=0.2187、CRPS=0.2187。
- CRB 与 UNI 正在并行训练；14:46:56 两臂各到 2,300/4,000 步。GPU 利用率 88%，显存 1,942 MiB。近段速率约 10 step/s，预计约 3 分钟后两臂训练和评估完成。
- REFT 排在 CRB/UNI 之后，按相近吞吐粗估还需约 7 分钟；整批剩余 ETA 约 10 分钟，实际随吞吐变化。
- 当前无错误，GPU 正在执行两条训练任务；无 CRB、UNI、REFT 实测指标。待训练臂评估完成后补录指标与结果文件路径。
