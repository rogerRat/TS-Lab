# Chronos-2-small 插队任务安排

- 操作时间：2026-11-17 18:21 CST。
- 用户要求：让 `time-sg-pm25-h` 插队先做。
- 当前在跑任务：`time-azure2019-u-5t` 的六臂训练；当时 Chronos 为 11/24 任务、176/384 评估点完成，GPU 利用率约 99%。
- 安排：保留当前 Azure_U 任务并等待其 task `.complete`；随后 watcher 对 Chronos 队列发送 SIGINT，在边界重启 queue。任务列表已把 `time-sg-pm25-h` 调到新增任务首位，run 模式将跳过有 task `.complete` 的任务，再执行 SG_PM25/H。剩余任务随后续跑。
- 实现：`scripts/exp/main-crb-supp-chronos2-small/priority_sg_after_current.sh` 在独立 tmux `chronos-sg-priority` 等待当前任务完成，再重启主队列；运行脚本用任务完成标记恢复，已有结果保留。
- 当前状态：等待 `time-azure2019-u-5t` 完成；新队列尚未重启。

- 2026-11-17 18:26 CST 复核：Azure_U 六臂训练已结束，正在逐步评估；任务级  尚未生成，watcher 正常等待，GPU 利用率 99%。
- 队列恢复修正：发现主队列入口原先不会跳过已完成的整任务，已在  增加基于 eval 任务  标记的跳过逻辑，重启后会保留并跳过已完成任务，再从 SG_PM25/H 开始。

- 2026-11-17 18:26 CST 复核：Azure_U 六臂训练已结束，正在逐步评估；任务级 .complete 标记尚未生成，watcher 正常等待，GPU 利用率 99%。
- 队列恢复修正：发现主队列入口原先不会跳过已完成的整任务，已在 run_queue.py 增加基于 eval 任务 .complete 标记的跳过逻辑，重启后会保留并跳过已完成任务，再从 SG_PM25/H 开始。

- 2026-11-17 18:35 CST：Azure_U 完成，24 点全部评估并生成任务标记；队列已按新顺序重启，跳过之前完成任务并启动 SG_PM25/H 六臂训练。各臂进度约 100/4000，GPU 利用率 99%，显存约 6.8 GiB。
- 启动时出现 Hugging Face 请求重试，使用 AutoDL 学术代理探测仍超时；Chronos 权重缓存可用，训练随后正常推进，当前无需中断。
- 2026-11-17 18:47 CST 巡检：time-sg-pm25-h 的 Chronos-2-small 六臂训练仍在进行，最新可见进度约 3600/4000 步；GPU 99%，评估点尚为 0/16，故当前没有可报告的 MASE 数值。
