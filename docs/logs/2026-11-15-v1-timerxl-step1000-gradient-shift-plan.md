# TimerXL step-1000 梯度差异实验计划记录
- 检查时间：2026-11-15 CST。
- 实验：`v1-timerxl-step1000-gradient-shift`；状态：完成计划文档，未启动计算。
- 已发现并复用：TimerXL 84M step-1000 LoRA checkpoint，seed 0，底座 revision `70077a71acce1b4c00d98332fcaabc694255d8e5`；不需要重复训练。
- GPU：tmux `ads_vic_hist` 的 ads-v2 余弦直方图任务仍运行，两 worker。遵照用户要求，不打断、不并发启动。
- 用户确认：LoRA 参数梯度；src-96 每轨 64 个窗口、合计 6,144，seed 42；Count-Sketch 4,096 维、seed 42；轨迹聚类 bootstrap 10,000 次、95% 区间。
- 下一步：等待 GPU 作业自然结束；确认资源空闲后创建 ASCII-only 脚本，在 `rage-timerxl` Conda 环境的 tmux 会话运行并按实际产物登记结果。
