# TimerXL step-1000 梯度差异实验启动记录

- 实际启动时间：2026-11-15 15:12:48 CST。
- 实验代号/批次：`v1-timerxl-step1000-gradient-shift` / `paired-src96-step1000`。
- 运行方式：tmux `timerxl_grad_shift_wait`；Conda `rage-timerxl`；入口 `scripts/exp/v1-timerxl-step1000-gradient-shift/compare_gradients.py`。
- 启动前核验：step-1000 checkpoint 的训练步数、seed 0、底座 revision、窗口协议和 LoRA 配置均通过；src-96 的 96 条轨迹最少有 150 个合法窗口，满足每轨无放回抽取 64 个窗口；共生成 6,144 个固定样本。
- 阶段状态：原始模型加载完成，原始梯度计算中；初始日志进度 256/6,144。
- 输出：`outputs/train/v1-timerxl-step1000-gradient-shift/`；运行日志：`logs/v1-timerxl-step1000-gradient-shift/paired-src96-step1000_20261115151248.log`。
- 实验仅在 GPU 空闲后启动；启动时利用率 11%、显存占用 922 MiB；无其他任务被中断。
