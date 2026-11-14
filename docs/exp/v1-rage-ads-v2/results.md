# v1-rage-ads-v2 结果记录

## 批次：baseline-design-v2

- 状态：算法与源池协议已更新；尚未实现、未运行。
- Source pool：目标 256 条，必须包含 src-96 全部 96 条；剩余 160 条按 src-4096 中扣除 src-96 后的 4000 条轨迹配置比例分配，最大余数法、seed 0 无放回抽取。由于保留 src-96，最终配置比例不可能对所有低占比配置严格等于 src-4096 的 1/16；实际配比以生成后的 manifest 为准。
- ADS-VIC：阈值 `cosine > 0.1`；连续内包区间，不向外延拓；默认 `R=2`。
- q 是间隔：`q_0=32`、`q_1=8`，`R=3` 时 `q_2=2`。长度 `L`、间隔 `Q` 的布点规则 `{0,L-1} ∪ {kQ-1}` 去重；`L=13,Q=4` 时为 `{0,3,7,11,12}`。
- 同预算 `uni`：ADS-VIC 自然完成后，按每任务 source/reference 不同锚点位置并集计数 `B_src/B_ref`，在完整轨迹池均匀放置相同数量的锚点。每轨整数锚点数确定后，将等距浮点位置映射到整数窗口起点；有效间隔不要求为整数。缓存点计入逻辑预算并另报新反向传播数。
- 数据包、选择输出、训练指标和评估结果：暂无。v1 的已完成指标仅属于 v1，不代表 v2 结果。

## 批次：rolling-baseline-v2

- 状态：选择阶段运行中；后续训练和 rolling 评估已在 tmux 中排队，等待选择完成事件。
- Source pool：256 条、35 个配置、6,529,220 个点；包含原 src-96 全部轨迹。配置明细见 `configs/data/src-256.yaml` 与 `data/src/shard_dataset/src-256/manifest.json`。
- 选择配置：ADS-VIC `q0=32`、`q1=8`、严格 `cosine > 0.1`；source/reference 两侧按实际唯一锚点数配平 uni；traj 使用轨迹级均匀锚点。
- 并行状态：第二轮八个任务 worker 已重启。第一轮因发现稀疏锚点连续分组及 uni 梯度复用错误而主动中断；第一轮没有写出任何选择结果。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-correction.md`。当前尚无完整 selection manifest、训练结果或评估指标，故不记录推测值。
- 产物路径：`outputs/selection/v1-rage-ads-v2/`、`outputs/train/v1-rage-ads-v2/`、`outputs/eval/v1-rage-ads-v2/`；运行日志位于 `logs/v1-rage-ads-v2/`。

- 17:44 巡检：time-08 已完成选择；fev-09 与 time-02 未完成退出；五个任务仍在 q1 细算。容器内存限制 66.6 GB，发现五个 worker 当前合计约 52 GB；暂停 time-01 与 time-03，保留三个 worker，并增加最多两个 worker 的恢复队列。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-memory-capacity.md`。选择结果目前为 1/8，训练和评估尚未开始。

- 已完成的选择数据：TIME08 完成全部五个方法；source 锚点预算 205,819，reference 锚点预算 4，最终匹配矩阵 823,276 对。其余七项选择仍待恢复；这些数值仅属选择预算，不是模型评测指标。
- 18:08 巡检发现容器接近内存上限：cgroup 64.9/66.6 GB，`high` 事件累计 8,324,582，`oom_kill=0`。大型全量相似度临时矩阵造成单进程峰值偏高。已终止三个未完成旧 worker，改用分块相似度，并以最多 2 个 worker 分批恢复。18:10 当前内存约 7.7 GB，GPU 99%。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-memory-overflow-check.md`。

- 容器重启后确认 FEV09、TIME01、TIME08 选择产物完整并保留；source/reference 锚点预算依次为 783,764/744、830,581/458、205,819/4。TIME02、TIME03、TIME04、TIME06、TIME07 已按最多两个 worker 恢复；尚无训练或评测指标。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-container-restart.md`。


## 阶段巡检：2026-11-14 21:13 CST

| 已完成选择任务 | source 锚点预算 | reference 锚点预算 | 末轮配对规模 | 选择耗时（分钟） |
|---|---:|---:|---:|---:|
| FEV09 | 783,764 | 744 | 583,120,416 | 56.0 |
| TIME01 | 830,581 | 458 | 380,406,098 | 57.4 |
| TIME02 | 893,872 | 634 | 566,714,848 | 60.8 |
| TIME03 | 967,873 | 1,727 | 1,671,516,671 | 71.1 |
| TIME08 | 205,819 | 4 | 823,276 | 45.0 |

上述预算是选择阶段计算量，末轮配对规模是相似度矩阵规模，均不是预测误差。TIME08 只有 1 条 reference 轨迹、3 个粗锚点，最终 reference 预算为 4；其后续模型指标需要结合较少的 reference 证据解读。当前选择完成 5/8，TIME04/TIME06 运行中，TIME07 待运行；尚无 v2 训练或 rolling 评测指标。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-status-2113.md`。


## 阶段巡检：2026-11-14 21:57 CST

| 已完成选择任务 | source 锚点预算 | reference 锚点预算 | 末轮配对规模 | 选择耗时（分钟） |
|---|---:|---:|---:|---:|
| FEV09 | 783,764 | 744 | 583,120,416 | 56.0 |
| TIME01 | 830,581 | 458 | 380,406,098 | 57.4 |
| TIME02 | 893,872 | 634 | 566,714,848 | 60.8 |
| TIME03 | 967,873 | 1,727 | 1,671,516,671 | 71.1 |
| TIME04 | 716,799 | 503 | 360,549,897 | 47.3 |
| TIME08 | 205,819 | 4 | 823,276 | 45.0 |

截至 21:57，选择完成 6/8。TIME06 进程仍存活，CPU 约 157%，RSS 约 18.3 GB；从 21:13:49 起无阶段进度日志，约 43.9 分钟。该进程仍在 CPU 上计算，但具体内部阶段无法由当前日志确定；预计还需时间未知。TIME07 尚未开始。cgroup 内存约 19.2/66.6 GB，oom 与 oom_kill 计数均为 0；GPU 利用率 0%、显存占用 980 MiB。训练/评测完成标记均不存在。锚点预算和矩阵对数仅衡量选择阶段计算规模，不是预测效果指标。TIME08 的 reference 预算仅 4，后续模型结果需结合该任务可用 reference 证据很少这一条件解读。详见 `docs/logs/2026-11-14-2157-v1-rage-ads-v2-status.md`。


## 阶段巡检：2026-11-15 01:07 CST

- 选择完成 7/8：FEV09、TIME01-04、TIME06、TIME08 已完成；TIME07 仍在运行。训练和评估尚未启动，没有 v2 模型效果指标。
- TIME06 于 00:35:05 完成选择，耗时 14,449 秒（约 4.0 小时）；source/reference 锚点预算为 977,001/37,900，末轮配对规模为 37,028,337,900 对。reference 共 5 条轨迹。该规模和耗时说明该任务的相似度计算开销很大，不代表预测质量。
- TIME07 当前已完成 ADS-VIC 方法输出，正在 `uni` 的 source 侧梯度计算；最近日志为 fragment 225，source 累计新算 510,197 个锚点。当前任务进程仍存活。
- 01:07 检查资源：GPU 利用率约 54%、显存 980 MiB；容器内存约 21.9/66.6 GB，oom 与 oom_kill 均为 0。当前运行代码已在启动时载入，修改磁盘上的相似度函数不会热替换 TIME07 进程；要让本批次采用 GPU 相似度实现需重启并重算其未落盘的选择状态，因此本轮未改动选择代码。
- 详情见 `docs/logs/2026-11-15-0107-v1-rage-ads-v2-status.md`。



## 阶段巡检：2026-11-15 01:46 CST

- 选择阶段 8/8 完成；TIME07 于 01:10:20 完成，source/reference 锚点预算 663,852/222，耗时 35.1 分钟。
- 训练评估 13/40 个 task-arm 单元已有训练与评估 .complete 标记；恢复 runner 于 01:40:47 在 tmux v1_rage_ads_v2_train 中以 12 workers 重启，最近检查 completed=13/40、active=12、pending=15、failed=0。GPU 利用率约 100%、显存约 11.6 GiB；cgroup 内存约 20.8/66.6 GB，oom/oom_kill=0。现无 v2 模型汇总指标。
- 已完成 source probability support 审计 40 行（8 tasks × 5 arms）。src-256 共 6,382,020 个合法窗口/任务；ADS-VIC 零概率比例 30.3042%，uni 19.7117%，traj 17.5960%，orig 0%；reft 不适用（reference-only）。这些是训练采样支持范围，不是预测指标。审计产物：outputs/selection/v1-rage-ads-v2/audit/probability-support/summary.json。
- 训练恢复修复：FEV orig 使用输出目录内的只读预训练 checkpoint 软链接视图并写入 FEV provenance manifest；reference 窗口读取按每个采样行的 owner 取对应轨迹；并行调度器在单元报错后继续调度其余待处理单元。上一轮 3 个失败单元尚无完成标记，已由恢复队列重试；当前恢复批次失败数为 0。
- 相似度 CUDA 优化已写入选择脚本，保持 float32 阈值与最大分数语义，采用显存预算保护和 CPU fallback。CUDA/CPU 小规模一致性测试 1 项通过；未对本批已经完成的选择任务重跑。
- 详情见 docs/logs/2026-11-15-0146-v1-rage-ads-v2-status.md。
