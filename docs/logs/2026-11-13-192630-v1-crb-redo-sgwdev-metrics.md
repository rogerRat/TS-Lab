# v1-crb-redo sgwdev 指标补算

- 操作时间：2026-11-13 19:26 CST
- 批次：sgwdev（single-window-eval）；状态：三项指标均完成，8 项任务、16 个任务-方法组合。
- 命令：python -u scripts/exp/v1-crb-redo/metrics.py；复用 step-1000 TimerXL LoRA、既有参考锚点和 src-96 全窗口池。
- 输出：outputs/selection/v1-crb-redo/sgwdev/metrics/；主程序发布文档后写入 .complete。
- 文档：更新本实验 exp.md、results.md、全局 EXP.md、RESULTS.md；现有评测批次逻辑名追加 -sgwdev，原产物路径保留。
