# main-ads-v0 梯度并行调整

- 检查时间：2026-11-15 16:28 CST。
- 实验代号：`main-ads-v0`；批次：`traj-replay`。
- 调整前：单 worker、batch=1，速度约 50 窗口/秒；GPU 利用率约 11%，显存 944 MiB。
- 调整：中断单 worker，保留 ordinal 0 已落盘的前 16,000 行；启动 8 个独立 GPU worker，各自 batch=4，按不重叠窗口区间写入同一梯度缓存。完成 ordinal 0 后以单 worker 处理 153 个窗口的 ordinal 1。
- 输出：`outputs/grad_cache/main-ads-v0/traj-replay/trajectory-000/gradient_sketches.npy` 与 `trajectory-001/gradient_sketches.npy`。
- 核验：8 个 worker 进程启动后，GPU 利用率 99%，显存占用 7,356 MiB；容器内存占用 12.3 GiB / 62.0 GiB。
- tmux：`main_ads_v0_traj_replay`；当前运行入口：`scripts/exp/main-ads-v0/compute_grad_cache_parallel.py --workers 8 --batch-size 4 --resume-trajectory-0-from 16000`。
