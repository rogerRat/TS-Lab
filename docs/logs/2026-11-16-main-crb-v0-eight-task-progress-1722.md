# main-crb-v0 八任务批次进度巡检

- 检查时间：2026-11-16 17:22:45 CST。
- 实验批次：`q16-src256-eight-task`；远端项目 `/root/TS-Lab`；tmux `main_crb_q16_eight_tasks` 存活。
- 阶段状态：八项任务 CRB/UNI source cosine 选择均已完成；八项 ORI 评估均完成并有 `.complete`。当前八个 pipeline 均在 CRB 训练，最新日志约 1,200–1,500/4,000 steps/任务。
- GPU：RTX 4080 利用率 99%，显存约 7.8 GiB。8 个训练进程持续运行。
- 指标：本次未读取/汇总 ORI 数值；CRB/UNI/REFT 尚无新评估指标。
- 队列日志：`logs/main-crb-v0/eight-task-20261116.log`；训练日志位于 `logs/tools/train/`，文件前缀 `main-crb-v0-eight-task-`。
- 异常：本次未见训练报错；较早配置路径异常已修复，队列已越过选择并进入训练。
- 后续：队列继续执行 CRB 训练评估、UNI 训练评估、REFT 训练评估；本次只做单次巡检，不设置轮询。
