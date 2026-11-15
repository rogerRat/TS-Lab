# v1-rage-ads-v2 批次启动记录

- 检查时间：2026-11-14 16:47 CST
- 实验代号：v1-rage-ads-v2
- 批次：rolling-baseline-v2
- 状态：八任务选择 worker 正在并行运行；训练评估阶段由独立 tmux 会话监听选择完成事件。

## 操作与执行

- 在远程 `/root/TS-Lab` 中创建 `src-256`：保留 src-96 全部轨迹，从 src-4096 扣除 src-96 后按配置余量比例、最大余数法、seed 0 无放回抽取 160 条。生成 256 条、35 配置、6,529,220 点；manifest 与数组长度字段核验完成，`.complete` 已写入。
- 新增 v2 选择入口 `scripts/exp/v1-rage-ads-v2/run_selection.py`，实现阈值匹配、连续命中锚点内包区间细算、q0=32/q1=8、事后同预算 uni、轨迹级 uni-traj、orig 与 reft 的 sampler manifest。
- 首次启动的 8 个任务都在首次梯度前因 0/1 起点映射错误退出；未产出结果。将内部零基窗口位置映射为 `extract_window` 的一基起点后重启。
- 2026-11-14 16:42:40 在 tmux `v1_rage_ads_v2_selection` 中启动八个独立任务。16:46 观测 GPU 利用率 100%、显存 7,820 MiB。各任务日志位于 `logs/v1-rage-ads-v2/selection-<task>.log`。
- 新增 v2 训练评估入口与并行 runner。tmux `v1_rage_ads_v2_train` 通过 queue 日志完成事件触发 `--workers 12` 的 40 个单元运行。

## 当前限制

- 选择阶段未完成，尚无实际锚点预算、训练 checkpoint 或指标。
- 训练与评估的最终完成状态以 runner manifest、各单元 `.complete` 和文档结果记录为准。
