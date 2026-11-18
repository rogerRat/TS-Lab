# main-crb-v0 梯度并行度调优与续跑

- 操作时间：2026-11-16 13:54–13:57 CST。
- 批次：`q16-src256-time1-q8-four-arm`；远程 AutoDL，项目 `/root/TS-Lab`，Conda `rage-timerxl`。
- 用户要求：并行执行并提高 GPU 利用率。
- 调优前观察：梯度阶段 GPU 利用率 11%、显存 562 MiB；主 Python 进程 CPU 100%。该时点 source CRB q16 的 399,044 个锚点梯度已经完成，source UNI 梯度准备在进行。
- 中断与恢复：向 tmux `main_crb_q16_time1` 发送 Ctrl-C，入口 finally 正常关闭 `GradientCache`；检查到 `window_index.npz` 已发布，包含 436,582 条梯度、3,405 个 shard。13:57:18 CST 使用相同 tmux 名称重启入口；已缓存梯度会命中原索引，不重复反向计算。CRB anchor selection 会从缓存梯度重新遍历以恢复内存中的边界列表。
- 代码调整：`scripts/exp/main-crb-v0/run_directional_time1.py` 将 TimerXL 梯度 batch 从 8 调至 32、batched VJP 从默认 4 调至 32，并把逐窗口 Python gather 换为 NumPy 向量化索引。NativeProbe 现有 OOM 逻辑会自动折半 batch 后重试。
- 重启命令：`tmux new-session -d -s main_crb_q16_time1 "bash -lc 'cd /root/TS-Lab && source /root/miniconda3/etc/profile.d/conda.sh && conda activate rage-timerxl && python -u scripts/exp/main-crb-v0/run_directional_time1.py --stage all'"`。
- 启动时观察到 tmux session 创建成功。按项目规则不对恢复后的长任务进行轮询；调优后的实际 GPU 利用率尚未测量。
- 梯度缓存路径：`outputs/grad_cache/main-crb-v0/q16-src256-time1-q8/raw_bf16/`；输出目标其余路径不变。未删除或覆盖已有梯度 shard。
