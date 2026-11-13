# v1 计算量与存储估算结果

最后修改日期：2026-11-10

计时试跑完成；目标轨迹清单、正式筛选及训练尚未开始。下列时长仅为旧候选池目标规模的源梯度准备外推。当前数据池已改为 src-8192，不能直接沿用。

| 模型 | 单进程吞吐量（梯度/秒） | 已测最快并行吞吐量 | 两份方法缓存的最快估算 |
|---|---:|---:|---:|
| TimerXL 84M | 575.35 | 1159（8 进程） | 1.68 小时 |
| TimesFM 2.5 | 23.436 | 27.827（2 进程） | 69.9 小时 |
| Chronos-T5-Base | 7.396 | 18.052（4 进程） | 107.7 小时 |

旧候选池有 199030 条合格轨迹、475138055 个标量值及 360695805 个合法窗口。目标为 256000×576，即 147456000 个值；按平均长度估算约 61800 条轨迹、111939593 个窗口。每种方法约需 `ceil(111939593/32)=3498113` 个锚点。Uniform 与 CRB 分别建缓存，合计约 6996226 次源梯度计算；十二个任务共用缓存。每份 4096 维单精度投影缓存约 53.38 GiB，两份约 106.76 GiB。

计时采用 BF16 LoRA、固定 4096 维 Count-Sketch 投影及种子 42。原始产物：TimerXL 位于 `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/`；TimesFM 位于 `outputs/train/v00-computational-effort-estimate/timesfm-pilot/` 和 `timesfm-parallel-two/`；Chronos 位于同一原始计时目录下的 `chronos-t5-base-pilot/`、`chronos-t5-base-parallel-two/` 和 `chronos-t5-base-parallel-four/`。原始产物仍保留运行时的 v00 路径；这些试跑目录有完成标记；日志位于 `logs/v00-computational-effort-estimate/`。

短时试跑不含参考梯度、打分、缓存写入、训练和评估，也没有正式模型质量指标。
