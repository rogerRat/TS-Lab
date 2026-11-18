# main-crb-v0 并集训练臂运行进度

- 检查时间：2026-11-16 18:07 CST。
- 队列：tmux main_crb_union_train 存活，8 个 task worker 已启动；16 份 crb_reft/uni_reft 合并窗口数据均存在完成标记。
- 当前阶段：8 个任务均在 crb_reft 训练；最近训练日志约 600–700/4,000 steps。CRB+REFT 评估尚未开始，uni_reft 排在各 task 后续队列。
- GPU：RTX 4080 利用率 99%，显存约 7.6 GiB。
- 本批新 arm 的 eval .complete 尚无；新增指标尚未生成。
- 日志：logs/tools/train/ 下 main-crb-v0-eight-task-*-crb_reft_*.log；启动与参数记录见 2026-11-16-main-crb-v0-union-reference-start.md。
- 下一步由持久 tmux 队列继续完成 16 个新训练与评估；本次未启动轮询。
