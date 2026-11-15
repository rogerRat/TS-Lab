# v1-rage-ads-v2 结果记录

## 批次：baseline-design-v2

- 状态：算法与源池协议已更新；尚未实现、未运行。
- Source pool：目标 256 条，必须包含 src-96 全部 96 条；剩余 160 条按 src-4096 中扣除 src-96 后的 4000 条轨迹配置比例分配，最大余数法、seed 0 无放回抽取。由于保留 src-96，最终配置比例不可能对所有低占比配置严格等于 src-4096 的 1/16；实际配比以生成后的 manifest 为准。
- ADS-VIC：阈值 `cosine > 0.1`；连续内包区间，不向外延拓；默认 `R=2`。
- q 是间隔：`q_0=32`、`q_1=8`，`R=3` 时 `q_2=2`。长度 `L`、间隔 `Q` 的布点规则 `{0,L-1} ∪ {kQ-1}` 去重；`L=13,Q=4` 时为 `{0,3,7,11,12}`。
- 同预算 `uni`：ADS-VIC 自然完成后，按每任务 source/reference 不同锚点位置并集计数 `B_src/B_ref`，在完整轨迹池均匀放置相同数量的锚点。每轨整数锚点数确定后，将等距浮点位置映射到整数窗口起点；有效间隔不要求为整数。缓存点计入逻辑预算并另报新反向传播数。
- 数据包、选择输出、训练指标和评估结果：暂无。v1 的已完成指标仅属于 v1，不代表 v2 结果。

## 批次：ads-rev-reverse-complement

- 状态：选择 8/8、训练 8/8、评估 8/8 完成；各训练单元 4,000 步，评估于 2026-11-15 09:50 CST 全部完成。八任务 MASE 算术均值为 1.519652，逐任务结果见下方训练评估表。
- 唯一方法：`ads-rev`，从头计算 q0=32 粗锚点矩阵；凡参与任一 `cos>=0.1` 配对的 source/reference 粗锚点排除；其余粗锚点按连续段以 q1=16 均匀布一轮。这里 q1 是 q0 的一半间隔，即两倍密度；不接原 ADS-VIC 的 q1=8 轮次。
- 最终 source 窗口分数和概率继续使用现行规则：source 锚点对全部 reference 锚点取最大余弦，负分截零，按 h=4 有限邻域线性衰减、重叠取 max，归一化后有放回抽样。
- 不运行该臂对应的 `uni`。下表配对数按用户定义统计 `cos>0.1`；集合排除按 `cos>=0.1`。八任务最终矩阵共 98,904,745 个命中配对 / 1,761,504,746 个矩阵元素（5.615%）；source 唯一命中锚点 1,275,774/2,140,607（59.60%），reference 唯一命中锚点 8,700/8,700。q0 粗矩阵命中 98,792,454 对；反向 q1=16 后增加 112,291 对。严格 `>` 与 `>=` 计数相差 13 对，均为等于 0.1 的边界项。

| 任务 | 最终矩阵元素 | 最终 `cos>0.1` 配对 | 矩阵命中率 | 命中 source 锚点 | 命中 reference 锚点 |
|---|---:|---:|---:|---:|---:|
| FEV09 | 36,668,856 | 1,879,125 | 5.12% | 175,381 / 218,267 | 168 / 168 |
| TIME01 | 25,949,546 | 2,463,848 | 9.49% | 180,248 / 229,642 | 113 / 113 |
| TIME02 | 32,231,320 | 3,077,011 | 9.55% | 189,295 / 207,944 | 155 / 155 |
| TIME03 | 73,114,610 | 6,718,218 | 9.19% | 197,764 / 200,314 | 365 / 365 |
| TIME04 | 34,658,248 | 2,887,839 | 8.33% | 153,539 / 284,084 | 122 / 122 |
| TIME06 | 1,544,231,600 | 80,939,503 | 5.24% | 199,998 / 200,030 | 7,720 / 7,720 |
| TIME07 | 12,970,152 | 917,158 | 7.07% | 159,883 / 240,188 | 54 / 54 |
| TIME08 | 1,680,414 | 22,043 | 1.31% | 19,666 / 560,138 | 3 / 3 |

- 输出路径：`outputs/selection/v1-rage-ads-v2/<task>/ads-rev/selection.json` 记录配对数、命中锚点数和预算；训练评估入口为 `scripts/exp/v1-rage-ads-v2/run_parallel_train_eval.py --arms ads-rev`，模型产物分别位于 train/eval 的 `v1-rage-ads-v2/<task>/ads-rev/`。

## 批次：rolling-baseline-v2

- 状态：选择 8/8 完成；训练与 rolling 评估 40/40 单元完成；概率支持审计 40/40 行完成。
- Source pool：256 条、35 个配置、6,529,220 个点；包含原 src-96 全部轨迹。配置明细见 `configs/data/src-256.yaml` 与 `data/src/shard_dataset/src-256/manifest.json`。
- 选择配置：ADS-VIC `q0=32`、`q1=8`、严格 `cosine > 0.1`；source/reference 两侧按实际唯一锚点数配平 uni；traj 使用轨迹级均匀锚点。
- 并行状态：八个选择任务均已生成完整 manifest；训练/评估完成标记覆盖全部 40 个 task-arm 单元。过程中的恢复修复和审计见后续带时间戳记录。
- 产物路径：`outputs/selection/v1-rage-ads-v2/`、`outputs/train/v1-rage-ads-v2/`、`outputs/eval/v1-rage-ads-v2/`；运行日志位于 `logs/v1-rage-ads-v2/`。

- 17:44 巡检：time-08 已完成选择；fev-09 与 time-02 未完成退出；五个任务仍在 q1 细算。容器内存限制 66.6 GB，发现五个 worker 当前合计约 52 GB；暂停 time-01 与 time-03，保留三个 worker，并增加最多两个 worker 的恢复队列。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-memory-capacity.md`。选择结果目前为 1/8，训练和评估尚未开始。

- 已完成的选择数据：TIME08 完成全部五个方法；source 锚点预算 205,819，reference 锚点预算 4，最终匹配矩阵 823,276 对。其余七项选择仍待恢复；这些数值仅属选择预算，不是模型评测指标。
- 18:08 巡检发现容器接近内存上限：cgroup 64.9/66.6 GB，`high` 事件累计 8,324,582，`oom_kill=0`。大型全量相似度临时矩阵造成单进程峰值偏高。已终止三个未完成旧 worker，改用分块相似度，并以最多 2 个 worker 分批恢复。18:10 当前内存约 7.7 GB，GPU 99%。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-memory-overflow-check.md`。

- 容器重启后确认 FEV09、TIME01、TIME08 选择产物完整并保留；source/reference 锚点预算依次为 783,764/744、830,581/458、205,819/4。TIME02、TIME03、TIME04、TIME06、TIME07 已按最多两个 worker 恢复；尚无训练或评测指标。详见 `docs/logs/2026-11-14-v1-rage-ads-v2-container-restart.md`。

## 八任务总表与审计：rolling-baseline-v2 + ads-rev

rolling-baseline-v2 的 5 种基线方法已完成 40/40 个 task-arm 单元；新增 ads-rev 训练与评估 8/8 完成，合计对比 48 个评估单元。选择和概率支持审计按各自批次记录。MASE 越低越好；每行最低值加粗并标 `*`。FEV09 数值来自各臂 `summary.csv`；TIME 数值为官方 `metrics.npz` 的 MASE 数组沿 series/window/variate 维度取算术平均。结果保留 6 位小数。

| 任务 | ADS-VIC | 同预算均匀 `uni` | 轨迹均匀 `traj` | 原始模型 `orig` | Reference-only `reft` | `ads-rev` |
|---|---:|---:|---:|---:|---:|---:|
| FEV09 | 0.731554 | 0.732347 | **0.725683** * | 0.731862 | 0.728981 | 0.733576 |
| TIME01 | 0.719835 | **0.718823** * | 0.743106 | 0.720606 | 0.855704 | 0.720485 |
| TIME02 | 1.272645 | **1.269746** * | 1.295175 | 1.310588 | 1.304599 | 1.274494 |
| TIME03 | 1.259114 | 1.252078 | **1.233410** * | 1.275348 | 1.250132 | 1.250550 |
| TIME04 | 4.284066 | 4.219065 | 4.458427 | 3.806636 | **3.508317** * | 4.374898 |
| TIME06 | 1.092393 | 1.094186 | 1.091422 | 1.112716 | **1.032821** * | 1.095141 |
| TIME07 | 1.574087 | 1.583742 | **1.545458** * | 1.635249 | 2.150268 | 1.563999 |
| TIME08 | 1.167866 | **1.090529** * | 1.120567 | 1.107539 | 1.114911 | 1.144074 |
| 八任务算术均值 | 1.512695 | 1.495064 | 1.526656 | 1.462568 | 1.493217 | 1.519652 |
| 最低 MASE 任务数 | 0/8 | 3/8 | 3/8 | 0/8 | 2/8 | 0/8 |

FEV09 的结果文件为 `outputs/eval/v1-rage-ads-v2/fev-09/<arm>/summary.csv`。TIME 每臂每任务的 `metrics.npz` 和原始预测位于 `outputs/eval/v1-rage-ads-v2/<task>/<arm>/<dataset>/<term>/`。完整运行配置和数据版本见同级 `time_metadata.json`。

### 选择与概率支持审计

锚点预算和匹配对数取自各任务 `outputs/selection/v1-rage-ads-v2/<task>/manifest.json`；零概率比例是对每个任务全部 6,382,020 个合法 src-256 窗口精确计数，来自 `outputs/selection/v1-rage-ads-v2/audit/probability-support/probability-support.jsonl`。`reft` 从 reference 窗口采样，不适用 source 窗口支持口径；`orig` 对 source 全窗口均匀采样，零概率为 0%。

| 任务 | ADS source/ref 锚点预算 | ADS 配对数 | ADS-VIC 零概率 | `uni` 零概率 | `traj` 零概率 | `orig` 零概率 | `reft` |
|---|---:|---:|---:|---:|---:|---:|---|
| FEV09 | 783,764 / 744 | 583,120,416 | 27.68% | 14.06% | 4.06% | 0% | 不适用 |
| TIME01 | 830,581 / 458 | 380,406,098 | 23.39% | 8.92% | 0.18% | 0% | 不适用 |
| TIME02 | 893,872 / 634 | 566,714,848 | 18.03% | 1.98% | 0% | 0% | 不适用 |
| TIME03 | 967,873 / 1,727 | 1,671,516,671 | 11.20% | 0% | 2.02% | 0% | 不适用 |
| TIME04 | 716,799 / 503 | 360,549,897 | 33.25% | 21.40% | 6.63% | 0% | 不适用 |
| TIME06 | 977,001 / 37,900 | 37,028,337,900 | 9.72% | 0% | 36.61% | 0% | 不适用 |
| TIME07 | 663,852 / 222 | 147,375,144 | 38.11% | 27.21% | 18.35% | 0% | 不适用 |
| TIME08 | 205,819 / 4 | 823,276 | 81.05% | 84.11% | 72.92% | 0% | 不适用 |
| 合计/整体比例 | — | — | 30.30% | 19.71% | 17.60% | 0% | 不适用 |

`traj` 的零概率是整条 source 轨迹得分为零后，该轨迹所有合法窗口均不进入抽样支持；窗口比例按窗口数计权。TIME08 的 `uni` 与 `traj` 零概率窗口比例都较高，解读对应 MASE 时应保留这一支持范围信息。锚点预算、配对数和零概率比例描述选择计算量及训练采样支持，不是模型误差指标。

## 阶段完成：2026-11-15 02:00 CST

- 选择：8/8 任务完成。TIME07 于 01:10:20 完成，source/reference 锚点预算 663,852/222，耗时 35.1 分钟。
- 训练与 rolling 评估：40/40 task-arm 单元完成，outputs/train/v1-rage-ads-v2/.complete 与 outputs/eval/v1-rage-ads-v2/.complete 已生成；outputs/train/v1-rage-ads-v2/parallel-run-manifest.json 记录并行资源摘要。各任务、方法的原始预测和指标文件保存在对应 outputs/eval/v1-rage-ads-v2/<task>/<arm>/ 目录。
- 采样支持审计：40/40 行完成。零概率截断定义及逐任务计数见本文件的“选择与概率支持审计”表；这些数值描述样本支持范围，不代表预测质量。
- 完成日志：logs/v1-rage-ads-v2/pipeline.log；训练并行日志位于 logs/v1-rage-ads-v2/parallel-train-eval_20261115014047.log。

## ads-rev 训练评估结果（2026-11-15）

训练和评估均已完成 8/8，训练步数为每任务 4,000 步；进程于 09:50 CST 全部退出，GPU 随后空闲。MASE 按 FEV09 的 `summary.csv` 与 TIME 各任务 `metrics.npz` 的 MASE 数组算术平均计算，越低越好。

| 任务 | ads-rev MASE | 指标文件 |
|---|---:|---|
| FEV09 | 0.733576 | `outputs/eval/v1-rage-ads-v2/fev-09/ads-rev/summary.csv` |
| TIME01 | 0.720485 | `outputs/eval/v1-rage-ads-v2/time-01/ads-rev/CPHL/15T/short/metrics.npz` |
| TIME02 | 1.274494 | `outputs/eval/v1-rage-ads-v2/time-02/ads-rev/CPHL/30T/short/metrics.npz` |
| TIME03 | 1.250550 | `outputs/eval/v1-rage-ads-v2/time-03/ads-rev/CPHL/H/short/metrics.npz` |
| TIME04 | 4.374898 | `outputs/eval/v1-rage-ads-v2/time-04/ads-rev/ECDC_COVID/D/short/metrics.npz` |
| TIME06 | 1.095141 | `outputs/eval/v1-rage-ads-v2/time-06/ads-rev/epf_electricity_price/H/short/metrics.npz` |
| TIME07 | 1.563999 | `outputs/eval/v1-rage-ads-v2/time-07/ads-rev/Commodity_Import/M/short/metrics.npz` |
| TIME08 | 1.144074 | `outputs/eval/v1-rage-ads-v2/time-08/ads-rev/Commodity_Production/M/short/metrics.npz` |
| 八任务算术均值 | 1.519652 | 上列八项 MASE 的算术平均 |

选择矩阵匹配统计仍见上方 ads-rev 批次记录。八任务余弦频次直方图已按 q0=32 粗筛与 q1=16 精算阶段重算；q1 统计完整最终矩阵（包含粗筛锚点及新加密锚点）。图像：`outputs/selection/v1-rage-ads-v2/ads-rev/cosine-histograms.png`；交互图：`cosine-histograms.html`；100 桶计数及边界：`cosine-histograms.json`。两阶段每任务的频次总和均与各自矩阵元素数相等。重算的严格 `cos>0.1` 总数为 q0 98,792,453、q1 98,904,744；与原选择记录仅 FEV09 相差 1 对，来自阈值附近的重算浮点差异，原 selection.json 和训练结果未改动。

### ADS-VIC 已保存分数的阶段图

按用户要求先只看当时已有的 FEV09、TIME04 两项，并用已保存的 `probability.json` 绘制阶段分布：按 source 锚点属于 q0=32 粗网格或新增 q1=8 锚点分组，统计其对最终 reference 锚点集合的最大余弦。该图是每个 source 锚点的最大分数分布，不是逐对矩阵余弦频次。

SVG：`outputs/selection/v1-rage-ads-v2/ads-vic/anchor-max-cosine/current-fev-09-time-04/saved-anchor-max-histograms.svg`；交互 HTML 与 100 桶 JSON 位于同目录。完整逐对矩阵重算在 FEV09、TIME04 完成后按用户要求停止；其 worker 内存中的频次数组未写盘，因此未将其冒充为已保存的矩阵直方图。
