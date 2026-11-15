# v1-rage-ads-v2 状态汇报与巡检进程检查

- 检查时间：2026-11-15 02:48 CST
- 实验代号：v1-rage-ads-v2
- 批次：rolling-baseline-v2
- 实验状态：选择 8/8、训练与 rolling 评估 40/40、概率支持审计 40/40 完成。
- 完成标记：selection 50 个、train 41 个、eval 41 个；根目录标记与 task-arm 标记已在结果文档确认。
- GPU 与资源：本次进程检查未发现实验运行器；最新实验巡检记录（02:03）显示训练和评估完成，GPU 0%，容器内存约 7.3 GB，OOM 计数为 0。
- 巡检进程：未发现匹配该实验的 monitor/poll/巡检进程；tmux 无运行会话。crontab 命令在该容器不可用。没有存活的实验巡检进程可停止，因此未发送终止信号。
- 结果：原始指标文件已生成，见 `outputs/eval/v1-rage-ads-v2/<task>/<arm>/`；采样支持汇总与逐任务明细见 `outputs/selection/v1-rage-ads-v2/audit/probability-support/`。模型指标分析见 `docs/exp/v1-rage-ads-v2/results.md`。
- 本次操作：只读核对实验文档、完成标记、进程、tmux 与定时任务可用性；未修改实验产物。
