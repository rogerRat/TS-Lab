# v1-rage-ads-v2 进度与资源巡检

- 检查时间：2026-11-14 17:45 CST
- 实验代号：v1-rage-ads-v2
- 批次：rolling-baseline-v2
- 状态：time-08 选择完成；time-01、time-03、time-04、time-06、time-07 仍在 q1 细算；fev-09、time-02 退出且未写完成标记。

## 进度与资源

- 当前选择完成数：1/8 任务；训练评估仍等待全部选择任务完成。
- GPU：17:44 实测 100%，显存约 4,889 MiB。
- 容器内存上限约 66.6 GB；五个 worker 合计接近 52 GB，进入稠密相似度矩阵阶段仍可能需要额外内存。
- 为控制峰值，已保留 time-04、time-06、time-07 三个 worker，暂停 time-01、time-03；操作后容器内存约 33.5 GB，GPU 利用率 99%。

## 修复与待办

- `similarity_hits` 已改为分块堆叠与点积，避免一次性复制完整 source 梯度矩阵；活动 worker 使用旧代码，恢复任务会使用分块版本。
- 恢复队列等待当前批次结束后，按每批最多 2 个 worker 重跑所有未完成任务。全部 8 项有完成标记后，写入 root `.complete` 并触发 12-worker 训练/评估。
- 10 分钟巡检器运行于 tmux `v1_rage_ads_v2_monitor`，每次将状态写入 `logs/v1-rage-ads-v2/progress-poll.jsonl` 并新增本目录下的巡检文档。
