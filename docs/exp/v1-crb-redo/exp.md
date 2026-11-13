# v1-crb-redo

最后修改日期：2026-11-13

## v1-crb-redo

首次提出：2026-11-12；最后修改：2026-11-13，用户已确认并授权启动。

#### grad-prepare

首次提出与最后修改：2026-11-12。状态：2026-11-12 23:00 CST 前三模型热身、CRB 与 Uniform 梯度及整批完成标记均已生成。此前 21:39 的 Chronos 进度是历史检查值。当前批次只负责三模型热身检查点以及 src-96 候选池 CRB/Uniform 梯度素材；后续数据选择、正式训练、评估另立批次。

数据：热身使用 data/src/shard_dataset/src-8192/ 与 configs/data/src-8192.yaml 的 8192 条轨迹。每次先等概率有放回抽取一条轨迹，再在该轨迹全部合法 576 点窗口中等概率抽取一个窗口；不按配置或轨迹长度额外均衡。候选梯度只使用固定 data/src/shard_dataset/src-96/ 和 configs/data/src-96.yaml 的 96 条轨迹；该集合与 src-8192 有包含关系，允许热身样本与候选窗口重叠。两者均保持已有原始数值快照，不添加预处理。所有模型窗口总长 576、起点步长 1；Chronos-T5-Base 和 TimesFM 2.5 以 512 历史加 64 预测计算损失，TimerXL 84M 使用其已有原生 V6 480 历史加 96 预测损失。模型间的目标不同，方法内 CRB/Uniform 同模型目标一致。

三个模型分别从原始预训练权重及 seed 0 LoRA 初始化热身。基础权重冻结，仅训练 LoRA；r=8、alpha=16、dropout=0；Chronos 目标 q/v，TimerXL 与 TimesFM 目标 q_proj/v_proj。各训练 1000 次优化器更新，有效批次 32；Chronos 微批次 32，另两模型微批次 8；AdamW、学习率 1e-4、权重衰减 0、betas=(0.9,0.999)、eps=1e-8、梯度裁剪 1、cosine 学习率日程与 3% 预热。日程沿用原 4000 步正式训练时间轴，在第 1000 步截取检查点。保存三份 LoRA 检查点及基础权重身份、数据和采样协议、训练参数、seed、实际步数；重新加载后核对固定窗口损失和梯度。

全局称呼预算 q 的定义为平均每 q 个合法候选窗口允许计算一次梯度，本批 q=32。逐轨迹固定 CRB coverage:error:split=1:0:1，error 阶段标为 deprecated。coverage 包含端点，split 沿用现有 min_kappa 规则；Uniform 按相同轨迹、相同实际梯度点数等距选点并包含端点。逐轨取整，两方法必须逐轨预算匹配；若 CRB 未使用全部预算，不能标记该轨完成，需先记录原因及补足规则。对当前 src-96 按逐轨取整规则，每模型每方法计划 83,330 锚点，coverage 与 split 各 41,665；两方法名义预算合计每模型 166,660。每模型仅对可训练 LoRA 参数求逐窗口损失梯度；固定 4096 维 Count-Sketch、seed 42、FP32 投影累加、BF16 位编码保存。两方法同一窗口的梯度允许共享计算，分别记录名义方法预算与真实唯一反传次数。

旧 v0-grad-prepare-src-96tr-q16-4096d 的无热身缓存、锚点清单和完成标记按用户要求删除；不删除 src-96/src-8192 数据快照、历史文档或其他实验产物。正式运行依次完成三模型热身、重载校验、两方法同预算梯度准备与逐轨验证；仅全部完成后写新批次 .complete。使用现有 rage Conda 环境，长任务在 tmux 中运行；日志放 logs/，检查点及梯度放 outputs/，大文件由项目软链接落在数据盘。不设置自动巡检，不关机。
EOF'

#### timerXL-baseline-full

批次首次提出：2026-11-12 21:59 CST；批次最后修改：2026-11-12 23:38 CST。状态：4,000 步训练、LoRA、训练清单、合并权重及 12 项评估均完成；结果见 `docs/exp/v1-crb-redo/v1-crb-redo-timerxl-baselines.md`。首次 FEV 因数据缺失失败的记录保留于 LOG.md。目标：固定 src-96 全量窗口训练基线。

从 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m` 的 TimerXL 84M 原始预训练权重和 seed 0 独立初始化，不加载 grad-prepare 热身检查点。基础权重冻结，仅训练 BF16 LoRA：目标 `q_proj`、`v_proj`，r=8、alpha=16、dropout=0。数据为 `configs/data/src-96.yaml` 与 `data/src/shard_dataset/src-96/` 的 96 条完整轨迹；每轨窗口起点步长 1、窗口长 576，共 2,666,312 个合法窗口，无额外预处理。训练每次更新从全部合法窗口全局等概率有放回抽样 32 个窗口，seed 0。TimerXL 原生 V6 损失：480 历史点、96 预测点。

恰好 4,000 次优化器更新；有效批次 32、微批次 8、累积 4。AdamW：学习率 1e-4、weight decay 0、betas=(0.9,0.999)、eps=1e-8；梯度裁剪 1.0；cosine 学习率日程覆盖 4,000 步，3% 即 120 步预热，min_lr_ratio=0。模型、LoRA、窗口采样均使用 seed 0。每 500 步保存恢复状态；保存第 4,000 步 LoRA 并导出合并模型；仅最终步评估。

固定下游 12 任务：`configs/eval/fev/v1-crb-redo-timerxl.yaml` 的 FEV task indices 9、11、18、19；`configs/eval/time/v1-rage-trev-timerxl.yaml` 的 TIME short-term 任务 CPHL/15T、CPHL/30T、CPHL/H、ECDC_COVID/D、ECDC_COVID/W、epf_electricity_price/H、Commodity_Import/M、Commodity_Production/M。评估数据、预测长度、指标和推理参数沿用这两份既定配置：FEV 用 `v1-crb-redo-timerxl-tasks.yaml` 中沿用 `default.yaml` 原定义的四个固定 task index、期望任务数 4；FEV 推理 BF16、batch 32、single_window、seed 20260722。TIME 用显式八任务且各取 short term；推理 BF16、batch 32、context 512、quantile levels 0.1 到 0.9（步长 0.1）、single_window、seed 20260722；TIME 源代码修订 `b6e7ba9862acb9233c8c55b314c1777052c420ff`，数据集 `Real-TSF/TIME` 修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`，目录 `data/bench/time/v1-rage-trev/dataset`。FEV 的四个实际任务 parquet 固定来自 `autogluon/fev_datasets` 数据集修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`，项目目录 `data/bench/fev/datasets/v1-crb-redo/`；在 `v1-crb-redo-timerxl-tasks.yaml` 中仅将这四项的 dataset_path 改为本地 parquet，其余任务定义、horizon、指标、窗口数保持原配置。项目 FEV 入口仅对明确选中的任务要求本地数据，但仍检查全部 20 项的指标声明；不修改第三方包。各任务的具体 horizon、指标与数据定义以这两份锁定的评估配置及其引用的任务定义文件为准，不改动评估标准。入口 `scripts/v1_crb_redo_timerxl_baselines.py --batch timerXL-baseline-full`；队列 `scripts/temp/run_v1_crb_redo_timerxl_baselines.sh`。训练输出 `outputs/train/v1-crb-redo/timerXL-baseline-full/`；评估输出 `outputs/eval/v1-crb-redo/timerXL-baseline-full/`；日志 `logs/v1-crb-redo/`。训练及两个基准全部成功后才写批次 `.complete`。

#### timerXL-baseline-rand01

批次首次提出：2026-11-12 21:59 CST；批次最后修改：2026-11-12 23:38 CST。状态：固定随机窗口、4,000 步训练及 12 项评估均完成；结果见 `docs/exp/v1-crb-redo/v1-crb-redo-timerxl-baselines.md`。目标：与全量窗口基线对照。该批从同一原始预训练权重和 seed 0 独立初始化，不加载 full 或 grad-prepare 的训练权重。其模型、LoRA、数据原值、窗口定义、原生 V6 损失、优化器、学习率日程、批次大小、4,000 步、恢复间隔和 12 项评估均与 full 批次相同。

在 2,666,312 个全局合法窗口索引 `[0, 2666312)` 上，用 `numpy.random.default_rng(0).choice(N, size=N//10, replace=False)` 一次性等概率无放回抽取 266,631 个索引，升序保存到 `outputs/train/v1-crb-redo/timerXL-baseline-rand01/fixed_window_indices.npy`；重启读取这份索引，不重新抽样，并核验范围、数量与严格递增。之后每次更新仅从固定集合等概率有放回抽取 32 个窗口，训练采样 seed 0。较长轨迹因合法窗口更多而自然占更大比例。训练输出 `outputs/train/v1-crb-redo/timerXL-baseline-rand01/`；评估输出 `outputs/eval/v1-crb-redo/timerXL-baseline-rand01/`；日志 `logs/v1-crb-redo/`。训练及 12 项评估成功后才写批次 `.complete`。


#### 12tasks-ref-grad-prepare

批次首次提出：2026-11-13 01:23 CST；最后修改：2026-11-13 01:23 CST；实验启动时间：2026-11-13 01:42 CST；最后记录状态：8 项可行任务的安全历史与 CRB/Uniform 梯度均完成并核验，批次完成标记已生成；4 项短历史任务已放弃，后续不再使用。该批是下方匹配选择批次的强制前置依赖；后续各批只有在本批真实完成并验证后才可选样。与现有 grad-prepare 保持同一实验文档，不另立批次文档。

目标：从与正式评测相同版本的 12 个下游任务数据中提取安全历史，按 TimerXL 原生 V6 窗口损失准备 ref 的 CRB 1:0:1 锚点和投影梯度。安全历史定义为：对每条评测目标序列，取**所有真实评测答案起点之前**的前缀；不读入或保存任何评测答案取值到 ref 产物。对多通道数据每个目标通道分别作为单变量序列。禁止把完整原始序列直接当作参考轨迹，禁止用评测输出反推选样。

| 任务编号 | 基准与任务 | 安全历史边界 | 状态 |
| --- | --- | --- | --- |
| FEV 9 | hospital_admissions_1D，horizon 28，20 windows | FEV 最早正式评测窗口的 get_input_data() 中目标历史 | 已确认 |
| FEV 11 | M_DENSE_1D，horizon 28，10 windows | 同上 | 放弃：安全历史不足 576 点，无法支持筛选；后续不再使用 |
| FEV 18 | us_consumption_1Y，horizon 5，10 windows | 同上 | 放弃：安全历史不足 576 点，无法支持筛选；后续不再使用 |
| FEV 19 | world_co2_emissions，horizon 5，9 windows | 同上 | 放弃：安全历史不足 576 点，无法支持筛选；后续不再使用 |
| TIME 1 | CPHL/15T，short | 单窗口评测的第一个答案起点之前的前缀 | 已确认 |
| TIME 2 | CPHL/30T，short | 同上 | 已确认 |
| TIME 3 | CPHL/H，short | 同上 | 已确认 |
| TIME 4 | ECDC_COVID/D，short | 同上 | 已确认 |
| TIME 5 | ECDC_COVID/W，short | 同上 | 放弃：安全历史不足 576 点，无法支持筛选；后续不再使用 |
| TIME 6 | epf_electricity_price/H，short | 同上 | 已确认 |
| TIME 7 | Commodity_Import/M，short | 同上 | 已确认 |
| TIME 8 | Commodity_Production/M，short | 同上 | 已确认 |

FEV 锁定 `configs/eval/fev/v1-crb-redo-timerxl.yaml` 和 `configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml` 的 task indices 9、11、18、19，数据源为此前 baseline 评估锁定的本地 parquet 与修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`。FEV 应以 FEV 自身最早评测窗口的 `get_input_data()` 实际输出为准提取安全前缀，而非仅以 `horizon*num_windows` 猜测截断点；逐项记录 task、实体、目标列、原始长度、边界位置和安全长度。TIME 锁定 `configs/eval/time/v1-rage-trev-timerxl.yaml` 的上述 8 个 short 任务及本地数据修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`；现有评估入口在 `single_window` 模式将 test_length 设为该任务 short 的 prediction_length，故安全前缀固定为每条序列 `target[:-prediction_length]`，须与实际 TIME 评测输入/答案分割位置逐实体核对。TIME 多通道按评估使用的单变量拆分口径逐通道提取；FEV 使用任务声明中的目标列，不添加输入协变量。对安全前缀按连续有限区间分段，每段不少于 576 点才保留；不足长度的段逐项记录后剔除。每段合法窗口为 480 历史加 96 预测、步长 1，且窗口必须完全落在安全前缀内。若某任务没有合法 ref 窗口，停止并报告，不以其他任务或答案补足。

与已完成 grad-prepare 的模型和梯度协议一致：使用 `outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000/`，基础权重、LoRA、TimerXL 原生 V6 逐窗口损失和 4096 维 Count-Sketch 投影身份必须与 src 缓存 manifest 匹配；仅准备 ref，不再热身或训练模型。每条 ref 轨迹按 q=32 的逐轨取整预算执行 CRB coverage:error:split=1:0:1，coverage 含端点、split=min_kappa，error=0；每个被查询锚点记录 (任务、实体、通道、片段、物理窗口起点)、梯度及块归属。梯度以 FP32 累加投影、BF16 位编码保存。必须像 src 一样同时准备逐轨等预算 Uniform 锚点和梯度；后续 rage101 方法在 src/ref 双侧都只读取 CRB。任何后续 Uniform 方法必须在 src/ref 双侧都读取 Uniform，并使用一致的方向预处理、插值和块代表方法。禁止一侧 CRB、一侧 Uniform。各任务独立目录、manifest、计数及完成标记；8 个可行任务全部完成且通过答案隔离、窗口合法性、锚点预算、投影身份和缓存字段校验，并记录 4 项放弃原因后写总批次 `.complete`。固定输出 `outputs/train/v1-crb-redo/12tasks-ref-grad-prepare/`，日志 `logs/v1-crb-redo/`，使用 `rage-timerxl` Conda 环境和 tmux；具体 worker 并发按资源预检确定并记录，不提前假定 8 个 worker 适合全部 12 项。

已确认：FEV 以最早评测窗口输入历史为共同安全边界；TIME 按各 short 任务实际单窗口预测长度切掉答案区间；多目标/多通道按评估目标分别展开，重复实体以实体加通道组成唯一 ID，若仍重复则报错；非有限历史按连续有限段处理，不足 576 点记录后剔除；必须同时准备 Uniform 与 CRB。安全历史按定义与 src-96 不重合，逐项记录数据来源与边界；若发现来源或实体身份异常重叠，停止并核查。8 个可行任务各自独立一套 ref 缓存和后续选样。具体每项轨迹数、窗口数、梯度数以实际提取校验结果填写，不预估为已完成结果。

#### TopK 聚合方案与 RR 实验批次

首次提出：2026-11-13；最后修改：2026-11-13；新 RR 批次启动时间：2026-11-13 02:58 CST；最后记录状态：P1、窗口预算及 RR 单位粒度已确认，traj/Uniform/FEV 9 首组训练评估完成并清理检查点；traj/block 四批均已完成；Uniform win 正在选样，CRB win 在其后自动执行。

方法分类说明：max、linear、zeroclip、exp 会先把所有 ref 对每个 src 代表单位的意见聚合成一个全局分数，因此属于 TopK 排序方案。它们不能称为 RR：RR 需要保留每个 ref 的独立意见并按 ref 轮询。此前的四种聚合训练队列已停止，相关选样、训练、评估产物与运行记录已清除；四种聚合只作为 TopK 方法定义保留，不纳入本次 RR 执行。

本次只执行六个直接 RR 批次。统一数据选择比例 0.1，批次名统一包含 -01-。全局源池有 2,666,312 个合法窗口，预算固定为 floor(2,666,312×0.1)=266,631 个不同窗口。

| RR 批次 | src/ref 锚点 | src/ref 匹配粒度 | ref 逐轮顺序 | 队列顺序 |
| --- | --- | --- | --- | --- |
| timerXL-base-uni-01-traj-rr | 双侧 Uniform，逐轨 q=32 | traj：安全历史片段与源轨迹分别整轨平均后匹配，选源轨迹 | ref 轨迹按固定任务序及片段 ID 顺序 | 1 |
| timerXL-base-uni-01-block-rr | 双侧 Uniform，逐轨 q=32 | block：相邻 Uniform 锚点区间的方向代表匹配，选源块 | ref 块按固定任务、片段、块序 | 2 |
| timerXL-ours-rage101-01-traj-rr | 双侧 CRB coverage:error:split=1:0:1，q=32 | traj：安全历史片段与源轨迹分别整轨平均后匹配，选源轨迹 | ref 轨迹按固定任务序及片段 ID 顺序 | 3 |
| timerXL-ours-rage101-01-block-rr | 双侧 CRB coverage:error:split=1:0:1，q=32 | block：CRB 块方向代表匹配，选源块 | ref 块按固定任务、片段、块序 | 4 |
| timerXL-base-uni-01-win-rr | 双侧 Uniform，逐轨 q=32 | win：逐个合法窗口方向匹配，选源窗口 | ref 窗口按固定任务、片段、窗口起点顺序 | 5（自动续排） |
| timerXL-ours-rage101-01-win-rr | 双侧 CRB coverage:error:split=1:0:1，q=32 | win：逐个合法窗口方向匹配，选源窗口 | ref 窗口按固定任务、片段、窗口起点顺序 | 6（自动续排） |

方向及代表计算固定规则：全部方法均执行 P1 过滤。src/ref 两侧采用相同锚点方法和估计方式。对每个梯度求 FP64 L2 幅值和单位方向；分别在 src 全部锚点以及当前任务 ref 全部锚点上计算 P1 幅值阈值，幅值小于或等于阈值时方向严格置零；幅值保留审计，方向和幅值分别沿物理窗口起点线性插值，不跨轨迹。Uniform 与 CRB 均执行相同 P1 和方向重建。轨迹代表是轨迹全部合法窗口的插值方向等权平均后单位化。块代表是该块拥有的全部合法窗口插值方向等权平均后单位化。零代表与任何向量点积均记为 0。RR 只使用代表方向余弦，不使用幅值。

RR 选择定义：每个 ref 代表单位按表中顺序进入轮询。其一轮从尚未选择的 src 代表单位中，选取该 ref 与 src 代表余弦分数最高的一个单位，选中单位不再参与后续轮次。traj/block 阶段只在代表单位间打分，不计算窗口对窗口分数；win 阶段才逐窗口打分。总预算固定为 266,631 个合法窗口。每个被选 src 代表单位的合法窗口数计入预算；若最后一个单位会超过剩余预算，仅取该单位全局窗口起点升序排列的前 N 个窗口以恰好达到预算。分数同分按 src 单位序号、再按窗口起点升序。ref 无剩余 src 单位可选时跳过。选择结果保存实际窗口起点，并记录对应代表单位。选定集合之后，训练每步从集合内均匀有放回采样有效批次 32 个窗口。六批使用相同选择预算、同一 RR 算法与同分规则，锚点估计方式及代表单位粒度按表变化；traj/block 四批已完成；Uniform win 选择完成后自动启动 CRB win。

评估指标汇报规则：所有任务、所有方法都必须完整汇报评测配置实际产生的每一项指标；MASE 是主比较指标，但不得只报告 MASE 或省略辅助指标。FEV 逐任务报告 `fev_results.csv` 中的 MAE、MASE、MSE、SQL、WAPE、WQL；TIME 逐任务报告 `metrics.npz` 中的 MSE、MAE、RMSE、MAPE、sMAPE、MASE、ND、CRPS。严格沿用评测产物的定义和任务内汇总方式；跨任务宏平均仅报告预先指定的 MASE，其他不同尺度指标保留逐任务数值。若某配置未产生指标或产物值为 NaN，结果文档明确标记缺失并说明，不得静默省略。该规则适用于本实验全部已完成及待运行批次，也适用于后续实验的评测汇报。

预算与训练评估：每一方法×任务均独立构建 ref 排序、RR 选样、训练与对应单任务评估，共 6×8=48 组。训练沿用 TimerXL baseline：原始 TimerXL 84M 权重、seed 0、BF16 LoRA q_proj/v_proj、r=8、alpha=16、dropout=0、AdamW 学习率 1e-4、weight decay 0、betas=(0.9,0.999)、eps=1e-8、有效 batch 32、微批次 8、梯度裁剪 1、4000 步 cosine、3% 预热，每 500 步保存恢复状态。每组评估对应的一个 FEV/TIME 任务；完整性核验后删除本组 resume、LoRA adapter 和合并模型，保留 RR 选中窗口、训练清单、评估原始指标及完成标记。四个短安全历史任务继续标记放弃，后续不再使用。

TopK 方案记录（不属于 RR 队列、不在本轮执行）：

| TopK 方法 | 意见聚合 |
| --- | --- |
| timerXL-ours-rage101-01-block-max-topk | 每个 src 块的 ref 加权分数取最大 |
| timerXL-ours-rage101-01-block-linear-topk | 每个 src 块的 ref 加权分数线性求和 |
| timerXL-ours-rage101-01-block-zeroclip-topk | ref 加权前逐项 ReLU，再求和 |
| timerXL-ours-rage101-01-block-exp-topk | ref 加权前逐项 exp，指数温度 1，再求和 |

前置梯度素材沿用已完成的 src-96 TimerXL CRB/Uniform 缓存及 12tasks-ref-grad-prepare 的 8 项安全历史 CRB/Uniform 缓存。ref 不作为训练数据。四项被放弃任务不会参与任何选择、训练或评估。核心脚本为 scripts/exp/v1-crb-redo/rr_features.py、rr_select.py、rr_selection.py、rr_train_eval.py、prepare_rr_eval.py、queue_rr.sh；输出分别落在 outputs/selection、outputs/train、outputs/eval，日志放 logs/v1-crb-redo/。仅在脚本通过选择算法数值核验、选中数量和唯一性核验、窗口所属单位核验后启动训练队列。
