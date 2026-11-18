# main-crb-v0 八任务批次完成巡检

- 检查时间：2026-11-16 17:51 CST；批次 `q16-src256-eight-task`。
- 队列于 17:44:59 CST 记录 `stage=batch-complete`，队列和训练/评估进程均已退出；RTX 4080 当前空闲。
- 8 项任务的 ORI/CRB/UNI/REFT 四臂评估全部完成。时间序列 28 份 `metrics.npz`，FEV09 4 份 `summary.csv`；train/eval 批次根 `.complete` 存在，32 个评估臂 `.complete` 均存在。
- 汇总 MSE/MASE 指标已追加至 `docs/exp/main-crb-v0/results.md`；批次状态与 `docs/EXP.md`、`docs/RESULTS.md` 已同步。
- 输出范围：outputs/eval/main-crb-v0/q16-src256-eight-task/<TASK>/<ARM>/；训练产物：outputs/train/main-crb-v0/q16-src256-eight-task/<TASK>/<ARM>/。
- 未发现失败任务或残留运行进程；本次为单次巡检，无自动轮询。
