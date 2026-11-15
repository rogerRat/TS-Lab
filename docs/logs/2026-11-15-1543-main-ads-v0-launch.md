# main-ads-v0 traj-replay 启动记录

- 检查时间：2026-11-15 15:43 CST。
- 实验代号：`main-ads-v0`。
- 批次：`traj-replay`。
- 任务状态：TimerXL LoRA 训练运行中，tmux 会话 `main_ads_v0_traj_replay` 存活。
- 启动时间：2026-11-15 15:42:54 CST。
- 执行环境：Conda `rage-timerxl`；单张 NVIDIA GeForce RTX 4080。
- 运行命令：`python -u scripts/exp/main-ads-v0/run_main_ads_v0.py --stage all`。
- 当前日志：`logs/main-ads-v0/traj-replay_20261115154254.log`。模型加载成功，LoRA 可训练参数 262,144；15:43:02 到达 step 20/2,000，最近一次 loss 为 0.76797134，记录的 ETA 约 4.3 分钟。
- 训练完成后，脚本生成 2,000 步新 LoRA 检查点、每 10 步 loss CSV 与 SVG 图，并自动计算 ordinal 0、1 的全部合法窗口梯度缓存。
- Replay 内容：待定。
