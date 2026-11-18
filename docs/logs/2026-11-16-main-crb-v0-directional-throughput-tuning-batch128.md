# main-crb-v0 梯度并行度二次调优

- 操作时间：2026-11-16 14:01 CST。
- 批次：`q16-src256-time1-q8-four-arm`；项目 `/root/TS-Lab`，Conda `rage-timerxl`。
- 用户要求：在约 50% GPU 利用率基础上继续提高并行度。
- 调优前单次观测：GPU 40% 利用率、710 MiB 显存使用，主进程 CPU 102%；用户侧报告约 50%。当时脚本处于 source UNI q16 锚点梯度准备阶段。
- 中断前资源：向 tmux `main_crb_q16_time1` 发送 Ctrl-C，梯度缓存正常关闭并发布 index；`window_index.npz` 有 505,872 条梯度、3,505 个 shard。
- 代码调整：将 `SelectionExecution.gradient_batch` 和 `gradient_vjp_batch` 提高到 128，并把 `TimerXLGradientEngine.max_gradient_batch` 实例上限调到 128。NativeProbe 已有 OOM 折半重试策略保留；此前 NumPy 向量化窗口 gather 也继续启用。
- 重启：2026-11-16 14:01:31 CST 创建 tmux `main_crb_q16_time1`，运行 `python -u scripts/exp/main-crb-v0/run_directional_time1.py --stage all`。缓存命中可跳过已完成梯度反向计算；CRB 位置选择从缓存梯度重放。
- 重启后未查询利用率或进度；调优后的实际 GPU 利用率尚未测量。
- 输出：`outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/raw_bf16/`；未删除或覆盖已缓存的梯度 shard。
