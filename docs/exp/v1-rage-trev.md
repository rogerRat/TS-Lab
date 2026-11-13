# v1-rage-trev

最后修改日期：2026-11-10

## v1：Trev 基础模型随机池与完整池训练

首次提出：2026-11-10。最后执行或修改：2026-11-10。目标：在固定数据池、训练预算和评估任务下，比较三种基础模型使用随机 10% 数据池与完整数据池训练 LoRA 后的 MASE。每个模型、池和随机种子都是独立批次。full 表示完整数据池，不表示全参数训练。基础模型权重冻结。


## 明确的训练超参数与设计决策

| 决策项 | 固定设置 |
|---|---|
| 候选数据 | Chronos wiki_daily_100k、m5；LOTSA PEMS07；Monash solar_10_minutes；UTSD UTSD-1G。完整池共 199,030 条序列、475,138,055 个点；各模型复用同一 seed 的清单。 |
| 随机池 | 按 seed 0、1、2、3 分别抽取完整候选池的 10%；seed 1–3 已生成的随机池清单各有 19,903 条序列。 |
| 窗口和抽样 | 上下文长度 512、预测长度 64、总窗口长度 576；每次更新有放回抽取有效窗口。 |
| 训练预算 | 每批次 4,000 次优化器更新；有效批量 32；TimerXL 和 TimesFM 的微批量为 8。 |
| 精度和可训练参数 | BF16；基础权重冻结；LoRA r=8、alpha=16、dropout=0。Chronos 目标模块为 q、v；TimerXL 和 TimesFM 为 q_proj、v_proj。 |
| 优化器 | AdamW；学习率 0.0001；权重衰减 0；betas=(0.9, 0.999)；eps=1e-8；最大梯度范数 1.0；余弦学习率调度；预热比例 0.03。 |
| 固定评估步骤 | 0、100、200、500、1000、2000、3000、4000。步骤 0 为训练前基线；每步先运行 FEV，再运行 TIME。两项评估完成后才能清理该步适配器；保留步骤 4000 适配器。 |
| 模型环境 | Chronos-T5-Base：rage；TimerXL 84M：rage-timerxl；TimesFM 2.5 200M：rage-timesfm。检查点路径以对应配置 JSON 为准。 |
| 评估任务 | 四个 FEV 原生单变量任务：hospital_admissions_1D、M_DENSE_1D、us_consumption_1Y、world_co2_emissions；八个 TIME 原生单变量任务：CPHL/15T、CPHL/30T、CPHL/H、ECDC_COVID/D、ECDC_COVID/W、epf_electricity_price/H、Commodity_Import/M、Commodity_Production/M。任务范围、预测长度和数据路径以对应评估 YAML 为准。 |
| 比较准则 | 仅使用 MASE 检查、比较和汇总；其余评估指标保留为原始输出，不用于决策。 |
| 完成条件 | 4,000 次更新、八个步骤的 FEV/TIME 评估，以及该批次输出目录的 .complete 均核实后标记完成。 |

配置文件位于 configs/rage/v1-rage-*.json；评估配置位于 configs/eval/fev/v1-rage-trev*.yaml 和 configs/eval/time/v1-rage-trev*.yaml。正式输出位于 outputs/train/，日志位于 logs/；通过 tmux 会话 v1-trev-seeds-0-3 运行 scripts/temp/queue_v1_rage_trev_seeds.sh。大文件实际存放在数据盘。


## v1 批次安排与现状

同一 seed 内按 Chronos 随机池、Chronos 完整池、TimerXL 随机池、TimerXL 完整池、TimesFM 随机池、TimesFM 完整池的顺序执行；种子按 0、1、2、3 顺序推进。seed 0 的 Chronos、TimerXL 两组及 TimesFM 随机池批次已有完成标记；TimesFM 完整池正在训练，seed 1–3 等待队列。seed 1–3 的 Chronos 输入池已备妥。上述未完成状态须以实际进程和产物持续更新。不得把排队或仅训练完成写成整个批次完成。



## v1 的已有更改记录

2026-11-10：明确 full 为完整候选池 LoRA 训练，而非全参数训练；决定仅以 MASE 汇总；TimesFM 的 FEV 与 TIME 输入在必要时向左补齐至 32 点 patch 的倍数；失败的 seed 0 随机池评估从已有优化器和评估产物恢复，不重跑已完成部分。当前仍使用原队列；详细故障过程见 docs/LOG.md，排队规则见 docs/ORCHE.md。


## Trev seed 范围收束（2026-11-10）

最后修改：2026-11-10

用户停止了多种子扩展，仅保留 seed 0。seed 0 的六个 Chronos、TimerXL 与 TimesFM rand/full LoRA 阶段全部完成；仅含 MASE 的结果见 docs/results/v1-rage-trev-seed0-final.md。Chronos seed 1 随机池训练在第 4000 步之前中断（日志中最新记录进度为第 2445 步），没有正式结果。其余 seed 1-3 阶段未训练。seed 1-3 的配置、已准备池、部分检查点与扩展队列均已删除。历史批次行保留在总览中以便追溯。



#### Time-300B 数据删除决策

最后修改：2026-11-10。2026-11-10 18:52:21 CST，用户要求删除未使用的 Time-300B 原始数据及转换后的分片。v1 Trev 候选池包含五个来源，不包含 Time-300B。删除前，`data/src/raw/time300b` 有 51 个文件、63,511,169,049 字节；`data/src/shard_dataset/selection24-time300b` 有 116 个文件、122,167,109,471 字节。保留两个目录路径，并附简要删除说明。


## src-8192 数据替换

最后修改：2026-11-10。此前所有 v1-rage 批次均已废弃，因为所选源数据不合适。其结果文件仅作为可能的参考保留，不构成正式答案。替换后的源集合为 src-8192：从 Chronos、LOTSA、Monash 与 UTSD 的 19 个合格配置中精确选取 8,192 条轨迹。固定的逐轨索引为 configs/data/src-8192.yaml；转换后的数值与偏移位于 data/src/shard_dataset/src-8192/ 下。尚未用该替换数据重启任何 v1-rage 训练批次。

19 个已选配置及各配置的计数记录在 configs/data/src-8192.yaml 中。选择使用 seed 0，要求合格序列长度至少 576、取值有限且非常数，并在各配置间均衡分配直至选出 8,192 条。固定 YAML 列出每个源、子集、bundle 行、序列 ID、长度与序号。独立的 float64 values.npy 与 offsets.npy 即使上游主分支变动也能保留已选数据。数据复用自已下载并转换的本地源 bundle；无需重新从网络下载。

#### src-8192 扩展至 35 个配置

最后修改：2026-11-10。使用项目的 Fetch 阶段额外下载了 Chronos 与 Monash 子集，然后用 Load 转换有界样本。新转换的 16 个配置通过固定合格规则（长度至少 576、有限、非常数），使覆盖从 19 个配置扩展到 35 个。选择 seed 仍为 0，轨迹数仍精确为 8,192。均衡分配每轮为每个配置选取一条合格轨迹，直至达到目标，耗尽的配置跳过。权威索引为 configs/data/src-8192.yaml 的第 2 版。它记录全部 8,192 条源行与序列 ID。独立的 float64 快照为 data/src/shard_dataset/src-8192/。在用户要求删除所有其他数据后，第 1 版已删除。未重启任何已废弃的 v1-rage 批次。

#### src-8192 保留决策

最后修改：2026-11-10。用户将永久上传独立的 src-8192 数据快照。该快照与 configs/data/src-8192.yaml 为权威来源。项目适配器现已支持上游 commit 固定抓取以供将来使用，但当前完整快照尚未重新下载并按这些 commit 核对。

#### 数据清理

最后修改：2026-11-10。按用户要求，删除除 data/src/shard_dataset/src-8192/ 之外的所有项目数据项。src-8192 YAML 不再引用那些已删除的源 bundle 路径。它保留稳定的序号、源/子集标签、序列 ID、长度、计数以及固定 values 与 offsets 的路径。此前的 src-8192 版本与 YAML 备份已删除。
