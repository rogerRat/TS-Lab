# v1-rage-ads 梯度探针修复与队列重启

- 时间：首次启动 2026-11-14 04:20:47 CST；修复与重启于 2026-11-14 04:27:49。
- 实验代号 / 批次：`v1-rage-ads` / `rolling-eight-task`。
- 首次运行结果：`run_selection.py --task all` 在预检缓存后构建 TimerXL 对象时退出，状态码 1。日志 `logs/v1-rage-ads/rolling-eight-task_20261114042047.log` 显示 `AttributeError`：脚本把 `metrics.build_probe()` 返回的底层模型 engine 当作梯度 probe，访问了不存在的 `projection_dimension`。当时没有选择输出、梯度产物或 `.complete`。
- 修复：在 `scripts/exp/v1-crb-redo/metrics.py` 增加独立 `build_gradient_probe()`，保留旧 `build_probe()` 的模型 engine 语义；新选择与审计 runner 显式接收底层模型 engine 和暖身梯度 probe。采样对齐梯度经 `TimerXLGradientEngine` 计算；task-H 连续性另用共享同一暖身模型的 `HorizonProbe`。source q32 缓存仍只读复用，不重算、不另存。
- 运行时验证：在 rage-timerxl 环境实际构建梯度 probe，确认 480+96、投影维度 4096、seed 42、热身步数 1000；通过。新增/修改 Python 文件编译及 ASCII 检查通过，队列 shell 脚本 `bash -n` 通过。
- 重启：结束本实验失败后仅休眠的 tmux session，再以相同 session 名 `v1-rage-ads-rolling` 重启 `scripts/exp/v1-rage-ads/start_queue.sh`。`logs/v1-rage-ads/queue.status` 当前记录 `status=started`；入口日志为 `logs/v1-rage-ads/rolling-eight-task_20261114042750.log`，已输出配置并进入选择 runner。
- 指标：截至重启时尚无本批次对齐、连续性、矩阵、训练或评估结果；不得把第一次失败写作完成批次。
- 后续：让队列按选择、审计、训练评估、汇总顺序运行；除用户明确要求外不设自动轮询。
