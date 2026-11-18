# 2026-11-16 main-crb-supp 正式队列启动

- 时间：2026-11-16 23:44:48（Asia/Shanghai）。
- 实验：main-crb-supp；顺序队列 12 个新任务，每任务 E1–E5 连续训练 4000 step 并评估 1000/2000/4000 checkpoint，E0 复用既有 4000 step Rand checkpoint 只评估。
- 环境：ssh tsl，项目 /root/TS-Lab，Conda rage-timerxl；tmux 会话 main-crb-supp；source q16/reference q8；seed 0；effective batch 32。
- 启动命令：`tmux new-session -d -s main-crb-supp "cd /root/TS-Lab && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/rage/envs/rage-timerxl/bin/python -u scripts/exp/main-crb-supp/run_queue.py --mode run"`。
- 启动确认：tmux 会话存在；source 共 6,382,020 合法窗口、每排布 399,044 个 q16 锚点；首任务 Crypto/D 的 4 条安全 reference 已开始准备。正式结果尚未产生。
- 日志：`logs/main-crb-supp/run_*.log`，各训练/评估入口继续使用全局日志器。
- 待观察：首个 4000 step 训练开始后的显卡利用率；只做启动期观察，不设置自动巡检轮询。未经用户要求不关机。
