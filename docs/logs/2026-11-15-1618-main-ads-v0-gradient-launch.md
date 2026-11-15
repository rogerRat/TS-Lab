# main-ads-v0 梯度缓存启动记录

- 检查时间：2026-11-15 16:18 CST。
- 实验代号：`main-ads-v0`；批次：`traj-replay`。
- 训练状态：TimerXL LoRA 训练 2,000/2,000 步完成。step 10 loss=0.943739474；step 2,000 loss=0.730318844；step 1,950 loss=135.59654236。
- 训练检查点：`outputs/train/main-ads-v0/checkpoint/adapter/`。
- Loss 文件：`outputs/train/main-ads-v0/checkpoint/loss_every_10_steps.csv`（200 个采样点）；图：`outputs/train/main-ads-v0/checkpoint/loss_curve.svg`。
- 梯度任务：tmux 会话 `main_ads_v0_traj_replay` 已启动，2026-11-15 16:18:01 CST 进入模型加载阶段。
- 运行命令：`python -u scripts/exp/main-ads-v0/run_main_ads_v0.py --stage gradients`；Conda 环境 `rage-timerxl`。
- 梯度输出：`outputs/grad_cache/main-ads-v0/traj-replay/`；TimerXL LoRA 参数梯度，Count-Sketch 4096 维、seed 0；ordinal 0、1 各自计算全部合法窗口。
- 队列修正：训练已完成并保存检查点、loss CSV 和 SVG。首次队列在写训练 manifest 时遇到数据盘真实路径与项目软链接路径转换错误，补齐训练 manifest 与完成标记后，将梯度缓存目录改为用户指定路径并单独启动梯度阶段。
- Replay 内容：待定。
