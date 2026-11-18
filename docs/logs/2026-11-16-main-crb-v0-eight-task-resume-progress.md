# main-crb-v0 八任务队列恢复进度

- 检查时间：2026-11-16 16:57:15 CST。
- 批次：q16-src256-eight-task；tmux main_crb_q16_eight_tasks 存活，8 个任务选择子进程均在运行。
- 恢复情况：resource cache .complete 已生成，36,331 条 ref 梯度缓存复用，computed=0；方向/幅值分量、八项 ref block 方向和 REFT 全窗口资源均已重建。队列已成功进入 task selection 阶段，越过此前 write_configs 路径异常。
- 当前工作：八项任务分别执行 CRB 和 UNI 的 source 全窗口评分。日志样本约 0.49–0.62M / 6.38M 窗口，stage ETA 约 7.8–10.7 分钟；各任务评分尚未完成，未生成选择完成标记。
- GPU：RTX 4080 利用率 36%，显存 3,946 MiB；八个 worker CPU 约各 1.1 核。当前打分阶段包含 CPU 方向插值和窗口准备，GPU 有持续负载，资源阶段之后训练评估尚未开始。
- 结果：本批没有训练或评估指标。source 选择完成后，队列将进入 24 个新训练和 32 次评估。
- 下一步：保持后台队列；不清理部分产物，任务脚本按完成标记继续。
