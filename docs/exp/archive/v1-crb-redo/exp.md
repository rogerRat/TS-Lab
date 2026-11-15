# v1-crb-redo

最后修改日期：2026-11-13 19:26 CST

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

批次首次提出：2026-11-12 21:59 CST；批次最后修改：2026-11-12 23:38 CST。状态：4,000 步训练、LoRA、训练清单、合并权重及 12 项评估均完成；结果见 `docs/exp/v1-crb-redo/results.md` 的 TimerXL 两基线小节。首次 FEV 因数据缺失失败的记录保留于 LOG.md。目标：固定 src-96 全量窗口训练基线。

从 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m` 的 TimerXL 84M 原始预训练权重和 seed 0 独立初始化，不加载 grad-prepare 热身检查点。基础权重冻结，仅训练 BF16 LoRA：目标 `q_proj`、`v_proj`，r=8、alpha=16、dropout=0。数据为 `configs/data/src-96.yaml` 与 `data/src/shard_dataset/src-96/` 的 96 条完整轨迹；每轨窗口起点步长 1、窗口长 576，共 2,666,312 个合法窗口，无额外预处理。训练每次更新从全部合法窗口全局等概率有放回抽样 32 个窗口，seed 0。TimerXL 原生 V6 损失：480 历史点、96 预测点。

恰好 4,000 次优化器更新；有效批次 32、微批次 8、累积 4。AdamW：学习率 1e-4、weight decay 0、betas=(0.9,0.999)、eps=1e-8；梯度裁剪 1.0；cosine 学习率日程覆盖 4,000 步，3% 即 120 步预热，min_lr_ratio=0。模型、LoRA、窗口采样均使用 seed 0。每 500 步保存恢复状态；保存第 4,000 步 LoRA 并导出合并模型；仅最终步评估。

固定下游 12 任务：`configs/eval/fev/v1-crb-redo-timerxl.yaml` 的 FEV task indices 9、11、18、19；`configs/eval/time/v1-rage-trev-timerxl.yaml` 的 TIME short-term 任务 CPHL/15T、CPHL/30T、CPHL/H、ECDC_COVID/D、ECDC_COVID/W、epf_electricity_price/H、Commodity_Import/M、Commodity_Production/M。评估数据、预测长度、指标和推理参数沿用这两份既定配置：FEV 用 `v1-crb-redo-timerxl-tasks.yaml` 中沿用 `default.yaml` 原定义的四个固定 task index、期望任务数 4；FEV 推理 BF16、batch 32、single_window、seed 20260722。TIME 用显式八任务且各取 short term；推理 BF16、batch 32、context 512、quantile levels 0.1 到 0.9（步长 0.1）、single_window、seed 20260722；TIME 源代码修订 `b6e7ba9862acb9233c8c55b314c1777052c420ff`，数据集 `Real-TSF/TIME` 修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`，目录 `data/bench/time/v1-rage-trev/dataset`。FEV 的四个实际任务 parquet 固定来自 `autogluon/fev_datasets` 数据集修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`，项目目录 `data/bench/fev/datasets/v1-crb-redo/`；在 `v1-crb-redo-timerxl-tasks.yaml` 中仅将这四项的 dataset_path 改为本地 parquet，其余任务定义、horizon、指标、窗口数保持原配置。项目 FEV 入口仅对明确选中的任务要求本地数据，但仍检查全部 20 项的指标声明；不修改第三方包。各任务的具体 horizon、指标与数据定义以这两份锁定的评估配置及其引用的任务定义文件为准，不改动评估标准。入口 `scripts/v1_crb_redo_timerxl_baselines.py --batch timerXL-baseline-full`；队列 `scripts/temp/run_v1_crb_redo_timerxl_baselines.sh`。训练输出 `outputs/train/v1-crb-redo/timerXL-baseline-full/`；评估输出 `outputs/eval/v1-crb-redo/timerXL-baseline-full/`；日志 `logs/v1-crb-redo/`。训练及两个基准全部成功后才写批次 `.complete`。

#### timerXL-baseline-rand01

批次首次提出：2026-11-12 21:59 CST；批次最后修改：2026-11-12 23:38 CST。状态：固定随机窗口、4,000 步训练及 12 项评估均完成；结果见 `docs/exp/v1-crb-redo/results.md` 的 TimerXL 两基线小节。目标：与全量窗口基线对照。该批从同一原始预训练权重和 seed 0 独立初始化，不加载 full 或 grad-prepare 的训练权重。其模型、LoRA、数据原值、窗口定义、原生 V6 损失、优化器、学习率日程、批次大小、4,000 步、恢复间隔和 12 项评估均与 full 批次相同。

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

首次提出：2026-11-13；最后修改：2026-11-13 20:21 CST；新 RR 批次启动时间：2026-11-13 02:58 CST；最后记录状态：traj/block 四批完成。旧版 FP32 Uniform win 进程于 12:29 CST 中断且未生成结果。修订后的 BF16 完整窗口矩阵流程已完成 Uniform win 选样、训练和评估 5 项。用户随后将 win 阶段范围限定为 CRB win 仅执行 Uniform 已完成的五项；两种方法的 TIME 6–8 均不执行：TIME 6 因 ref 窗口数增加导致约 1.26 TiB 矩阵超过可用空间；TIME 7、8 的矩阵估算约 7.83 GiB、0.30 GiB，按用户限定范围跳过。CRB win 五项队列于 2026-11-13 15:39 CST 在 tmux `v1-crb-redo-crb-win` 启动；截至 2026-11-13 20:21 CST，CRB win 限定的 FEV 9、TIME 1–4 五项均已完成选择、4,000 步训练和评估；五项及批次 `.complete` 标记已核验，训练检查点按协议删除。

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

RR 选择定义：每个 ref 代表单位按表中顺序进入轮询。其一轮从尚未选择的 src 代表单位中，选取该 ref 与 src 代表余弦分数最高的一个单位，选中单位不再参与后续轮次。traj/block 阶段只在代表单位间打分，不计算窗口对窗口分数；win 阶段才逐窗口打分。总预算固定为 266,631 个合法窗口。每个被选 src 代表单位的合法窗口数计入预算；若最后一个单位会超过剩余预算，仅取该单位全局窗口起点升序排列的前 N 个窗口以恰好达到预算。分数同分按 src 单位序号、再按窗口起点升序。ref 无剩余 src 单位可选时跳过。选择结果保存实际窗口起点，并记录对应代表单位。选定集合之后，训练每步从集合内均匀有放回采样有效批次 32 个窗口。六批使用相同选择预算、同一 RR 算法与同分规则，锚点估计方式及代表单位粒度按表变化；traj/block 四批已完成；用户于 2026-11-13 15:38 CST 将 win 阶段范围收敛为：保留已完成的 Uniform win 5 项；不执行 Uniform win TIME 6–8；CRB win 只执行与 Uniform win 已完成项相同的 FEV 9、TIME 1–4 共 5 项，不执行 CRB win TIME 6–8。TIME 6 因 ref 窗口数增加到 259,085，完整 BF16 矩阵预估 1,381,602,889,040 bytes（约 1.26 TiB），超过当时约 381 GiB 可用空间，故 Uniform/CRB win 均跳过。TIME 7、8 的矩阵分别约 7.83 GiB 和 0.30 GiB，容量本身可容纳；按用户确认的执行范围不运行这两项。Uniform 与 CRB 对同一任务具有相同 ref/src 窗口数和相同矩阵尺寸。CRB 前五项所需缓存均已核实齐全，将按队列脚本顺序选择、训练和评估。

新增分数精度与矩阵存储决策（用户确认：2026-11-13 12:26 CST）：此后所有匹配粒度的分数计算与分数持久化统一使用 BF16。参与评分的方向向量在评分时转为 BF16；分数以 BF16 计算输出并以 BF16 存储，不保留 FP32/FP64 分数矩阵或高精度副本。RR 排序和同分处理使用实际保存的 BF16 分数值。

win 级 RR 允许并要求为每个任务一次性物化完整评分矩阵，矩阵行按固定顺序对应 ref 合法窗口，列按全局起点升序对应 src 合法窗口；随后直接从该矩阵执行 RR，不再对每个 ref 重复扫描并重算全部 src 分数。矩阵存放在 `outputs/selection/v1-crb-redo/<batch>/<task>/`，属于可再生成的中间评分产物；矩阵以无头 row-major BF16 原始数据存储；矩阵形状、dtype、ref/src 次序和实际文件大小写入 manifest。评分特征原文件不变，仅在点积入口转换为 BF16。矩阵完整写入后直接用于 RR 排序；只有在选中索引完成预算与唯一性核验、manifest 写入成功后才删除矩阵，随后写最终 `.complete`。若选样中断但矩阵已完整生成，重启时复用该矩阵，不重复计算分数。FEV 9 的矩阵形状为 4,766×2,666,312，BF16 数据体积为 25,415,285,984 bytes（约 23.67 GiB）；TIME epf_electricity_price/H 的 259,085×2,666,312 矩阵需 1,381,602,889,040 bytes（约 1.26 TiB），超过本次检查时的数据盘剩余约 381 GiB。对同一任务，Uniform 与 CRB 的 ref/src 窗口数量相同，故所需矩阵字节数相同；已运行的 Uniform 在 TIME 6 即因该容量门槛停止，尚未运行的 CRB TIME 6 也无法在当前磁盘空间下按此方案执行。目录通过项目 `outputs/` 软链接落在数据盘。

旧版 FP32 分块 top-k 进程于 2026-11-13 12:29 CST 中断，未生成选择完成标记或启动训练。其替代实现已按 BF16 完整评分矩阵规则重新执行；旧版运行不作为实验结果。

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

## 评测协议批次与后续 TODO

本实验截至 2026-11-13 已有的两个基线运行及六个 RR 方法运行统一归入逻辑批次 `sgwdev`，全称 `single-window-eval`。实验文档和结果索引中的逻辑批次名按 `<既有运行名>-sgwdev` 书写；已有配置文件名、输出目录和原始结果路径保留不改，以便与已生成产物对应。

| 逻辑批次后缀 | 全称 | 覆盖范围 |
| --- | --- | --- |
| `-sgwdev` | single-window-eval | 当前已有 baseline full、baseline rand01、Uniform/CRB 的 traj、block、win 评测及本次新增统计指标 |
| `-rolling-eval` | rolling-eval | 后续独立新增，不与 sgwdev 混合汇总 |

### 补充探索性关联分析

以每个任务四个完整 RR arm 的 MASE 均值为筛选结果，以 full 和 rand01 中 MASE 更低者为较强基线，收益定义为 `(best baseline MASE - mean RR MASE) / best baseline MASE`。在八个任务上描述性计算收益与全源对齐均值、top10% 对齐均值、安全历史—测试均值及该配对余弦 P99 的 Pearson 和 Spearman 系数。此分析为单 seed 的事后探索，不作因果或显著性结论。具体数值见 results.md。

### src P99 高匹配单元的时间与轨迹聚集

每个 src 单元的分数定义为该单元方向与该任务所有参考单元方向之间的最大 cos（best-reference match）。在每任务、方法和粒度内按该分数降序取正好 top 1% 单元；这量度是“至少与某个参考方向匹配”，不是 RR 选择中每个参考单元独立排序的分数。完全复用已缓存 source_units.npz 和 reference_units.npz 方向，未重算梯度。traj 粒度每方法和任务只有 96 条源轨迹单元，top 1% 因此是 1 条，可报最匹配轨迹，但不能检查轨迹内时间聚集。block 粒度用每个已选高分 block 所在轨迹、同轨迹上相邻高分 block 对数及最长连续运行长度量化聚集。

时间聚集的参考分布是每条轨迹内保持 P99 block 个数不变，在该轨迹的 block 位置中均匀随机重选 500 次（seed=42）。报告随机相邻对数均值和上尾 p 值。此重排只保持各轨迹高分 block 数，不保持相邻块分数之间由特征构造带来的依赖。block 方向由相邻的缓存锚点方向插值构建，相邻 block 共用边界锚点；因此相邻超额可作为分数时间聚集的描述，但不解释为独立样本的显著性证据。

| Task | Method | Trajectories with P99 blocks | Effective trajectories | Top 10 trajectory share | Adjacent P99 block pairs / random mean | Longest run | Randomization p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | CRB | 28/96 | 7.3 | 88.4% | 441 / 198.9 | 12 | 0.002 |
| TIME 1 CPHL/15T | CRB | 36/96 | 5.8 | 82.8% | 210 / 52.2 | 6 | 0.002 |
| TIME 2 CPHL/30T | CRB | 27/96 | 5.3 | 91.0% | 213 / 36.2 | 7 | 0.002 |
| TIME 3 CPHL/H | CRB | 26/96 | 9.3 | 89.3% | 221 / 75.8 | 8 | 0.002 |
| TIME 4 ECDC_COVID/D | CRB | 23/96 | 7.0 | 95.0% | 178 / 33.5 | 6 | 0.002 |
| TIME 6 epf_electricity_price/H | CRB | 24/96 | 5.4 | 94.0% | 338 / 133.7 | 10 | 0.002 |
| TIME 7 Commodity_Import/M | CRB | 53/96 | 20.4 | 60.7% | 295 / 80.6 | 12 | 0.002 |
| TIME 8 Commodity_Production/M | CRB | 45/96 | 11.3 | 76.0% | 243 / 49.1 | 7 | 0.002 |
| FEV 9 hospital_admissions_1D | UNIFORM | 27/96 | 6.8 | 85.7% | 455 / 244.9 | 21 | 0.002 |
| TIME 1 CPHL/15T | UNIFORM | 29/96 | 5.2 | 85.0% | 275 / 72.1 | 8 | 0.002 |
| TIME 2 CPHL/30T | UNIFORM | 32/96 | 8.3 | 78.8% | 298 / 79.0 | 8 | 0.002 |
| TIME 3 CPHL/H | UNIFORM | 24/96 | 7.0 | 93.6% | 332 / 213.2 | 9 | 0.002 |
| TIME 4 ECDC_COVID/D | UNIFORM | 18/96 | 4.6 | 98.9% | 251 / 46.3 | 5 | 0.002 |
| TIME 6 epf_electricity_price/H | UNIFORM | 26/96 | 7.8 | 93.3% | 571 / 77.8 | 39 | 0.002 |
| TIME 7 Commodity_Import/M | UNIFORM | 50/96 | 20.6 | 61.9% | 342 / 100.3 | 13 | 0.002 |
| TIME 8 Commodity_Production/M | UNIFORM | 33/96 | 4.9 | 86.7% | 227 / 35.3 | 5 | 0.002 |

traj 粒度每个任务与方法的最高分源轨迹：

| Task | Method | Top trajectory | Best-reference cos |
| --- | --- | --- | ---: |
| FEV 9 hospital_admissions_1D | CRB | chronos/weatherbench_daily/T086145 | 0.5093 |
| TIME 1 CPHL/15T | CRB | chronos/wiki_daily_100k/T068491 | 0.5682 |
| TIME 2 CPHL/30T | CRB | chronos/monash_london_smart_meters/T000242 | 0.5377 |
| TIME 3 CPHL/H | CRB | chronos/monash_weather/T000866 | 0.4597 |
| TIME 4 ECDC_COVID/D | CRB | chronos/wind_farms_hourly/T000329 | 0.4380 |
| TIME 6 epf_electricity_price/H | CRB | chronos/monash_electricity_hourly/T000057 | 0.5901 |
| TIME 7 Commodity_Import/M | CRB | chronos/monash_fred_md/T000105 | 0.4136 |
| TIME 8 Commodity_Production/M | CRB | chronos/monash_temperature_rain/10917__t_mean | 0.1866 |
| FEV 9 hospital_admissions_1D | UNIFORM | chronos/weatherbench_daily/T086145 | 0.4200 |
| TIME 1 CPHL/15T | UNIFORM | chronos/wiki_daily_100k/T068491 | 0.5810 |
| TIME 2 CPHL/30T | UNIFORM | monash/australian_electricity_demand/T4 | 0.4395 |
| TIME 3 CPHL/H | UNIFORM | monash/weather/T1750 | 0.4450 |
| TIME 4 ECDC_COVID/D | UNIFORM | monash/oikolab_weather/T4 | 0.5123 |
| TIME 6 epf_electricity_price/H | UNIFORM | chronos/monash_electricity_hourly/T000057 | 0.6601 |
| TIME 7 Commodity_Import/M | UNIFORM | chronos/m5/FOODS_3_562_CA_2 | 0.3616 |
| TIME 8 Commodity_Production/M | UNIFORM | monash/weather/T1092 | 0.1806 |

TODO：若后续加入 rolling-eval，FEV 按全部声明 cutoff 评测；TIME 使用各任务完整 test_length 生成 rolling 窗口。必须在最早测试 cutoff 前重建 TIME 安全历史并重新准备 CRB/Uniform 梯度，不得复用 sgwdev 的安全历史缓存；结果按独立协议批次汇报，并注明同序列窗口之间存在时间依赖。

## 新增统计指标与补算方案（2026-11-13）

本次只对已具备安全历史梯度的八项任务统计：FEV 9（hospital_admissions_1D）及 TIME 1、2、3、4、6、7、8。FEV 11、18、19 与 TIME 5 因安全历史不足 576 点而已停止使用，不纳入这两项统计。指标均使用当前实验的 TimerXL 84M、step-1000 LoRA 检查点与 4096 维 Count-Sketch 梯度表示（projection seed=42）；不改变现有选择、训练或官方评测流程。

统一梯度配对协议（用户确认）：全源与 top10% 筛选后指标只配对已缓存的真实源锚点梯度和已缓存的同方法安全历史锚点梯度；安全历史—测试指标只配对安全历史缓存锚点与按 H 新算的真实测试样本梯度。三项都不用重建逐窗口梯度；全源和筛选后结果仅代表对应的缓存锚点子集估计。对已选梯度单位化后的全部交叉组合求余弦算术均值，以双方单位梯度均值的点积等价累加，不实体化余弦矩阵。

### 指标一：全源数据集的任务平均方向评分

全源逐窗口梯度不可得，本指标按“已计算真实锚点均值”估计：源侧直接读取 src-96 已有 CRB 或 Uniform 梯度缓存，参考侧读取对应任务已有安全历史 CRB 或 Uniform 梯度缓存；不重算源侧梯度，也不对未计算窗口做线性重建。两侧方法匹配，分别报告 CRB-CRB 与 Uniform-Uniform。

两侧分别按全部缓存锚点梯度范数的第 1 百分位执行 P1 过滤（小于或等于阈值的方向置零），其余梯度单位化。任务分数是该任务参考锚点与源锚点之间所有两两方向余弦的算术平均。通过双方单位方向均值的点积等价计算，不构造完整两两矩阵。该值是基于已计算锚点的估计值，不代表未计算的全部源窗口真实分数。
 
### 指标二：安全历史对真实测试集的任务内方向代表性

每个下游任务单独与其自身安全历史配对。安全历史侧直接使用 step-1000 检查点下已缓存的 CRB、Uniform 真实锚点梯度，不按任务 H 重算；测试侧使用锁定评测配置的 single-window 实际输入与真实标签，逐样本计算梯度。评测答案只用于这一独立诊断，不进入参考集合、选样或训练。

| 下游任务 | 预测长度 H |
| --- | ---: |
| FEV 9 hospital_admissions_1D | 28 |
| TIME 1 CPHL/15T | 12 |
| TIME 2 CPHL/30T | 12 |
| TIME 3 CPHL/H | 24 |
| TIME 4 ECDC_COVID/D | 30 |
| TIME 6 epf_electricity_price/H | 24 |
| TIME 7 Commodity_Import/M | 12 |
| TIME 8 Commodity_Production/M | 12 |

**指标二的损失口径（用户于 2026-11-13 确认）：**安全历史侧保持已缓存的原生 480+96 梯度；只有真实测试样本侧按表中 H 使用 480 点历史与 H 点真实目标计算梯度，TimerXL 仍取 96 点输出头并只对前 H 个预测计算 MSE，归一化统计仅由历史上下文计算。 TIME 真实标签中的 NaN 按评测器有效标签掩码计算 task-H MSE；H 内没有任何有效标签的样本不产生梯度并排除。 真实历史上下文不足 480 点的测试样本不填充，排除并计数。测试梯度使用与缓存一致的 step-1000 LoRA 检查点、4096 维 Count-Sketch 和 seed 42。此设置只定义本项诊断指标，不改变训练或官方 FEV/TIME 评测。

对投影梯度逐行 L2 单位化；零范数梯度与任意方向的余弦记为 0。对任务 t、参考方法 m，计算其安全历史锚点梯度与该任务全部测试样本梯度之间的余弦矩阵，并取全部元素算术平均：

history_test_alignment(t, m) = mean_{i,j} cosine(g_safe[t,m,i], g_test[t,j])。

该值是同任务安全历史与真实测试样本之间的平均两两方向余弦，作为安全历史代表下游测试梯度的描述性近似。安全历史侧沿用现有 q=32 锚点样本；测试侧不另做随机抽样，使用官方评测器实际采用的全部样本。

### 指标二的余弦分数分布

按任务和方法，对所有安全历史锚点—真实测试样本两两余弦报告 P50、P90、P95、P99、最大值，并报告 cos>0 与 cos>=0.1 的配对比例。另外按每个真实测试样本取安全历史锚点中的最大 cos，报告测试样本最佳匹配值的平均和最大值。数值来自已保存的梯度，没有重算梯度。

两两配对统计的分母为安全历史锚点数乘以真实测试样本数（本批次每任务 100–40,500 对）。同一锚点会参与多个配对，因此这些分位数是描述性分布统计，不视为独立样本置信区间。每项任务仅有 2–9 个真实测试样本，因此测试样本最佳匹配值的高分位数不稳定，报告其均值和最大值。此后新增相似度分析必须包含分布统计；大规模配对需在实验文档中声明是全量精确分位数还是按固定规则抽样估计，并记录数量、随机种子和实际算法。

| Task | Method | Pair count | P50 | P90 | P95 | P99 | Max | Positive | cos >= 0.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FEV 9 hospital_admissions_1D | CRB | 1152 | 0.010834 | 0.198855 | 0.257365 | 0.388612 | 0.569305 | 52.5% | 27.3% |
| FEV 9 hospital_admissions_1D | Uniform | 1152 | 0.009395 | 0.201206 | 0.254458 | 0.348050 | 0.619248 | 53.0% | 26.5% |
| TIME 1 CPHL/15T | CRB | 200 | 0.003821 | 0.142561 | 0.173722 | 0.231969 | 0.328059 | 53.0% | 14.5% |
| TIME 1 CPHL/15T | Uniform | 200 | 0.003930 | 0.117533 | 0.173413 | 0.208368 | 0.308987 | 50.5% | 12.5% |
| TIME 2 CPHL/30T | CRB | 264 | -0.004901 | 0.148476 | 0.220182 | 0.387366 | 0.456489 | 48.9% | 17.0% |
| TIME 2 CPHL/30T | Uniform | 264 | -0.009434 | 0.156222 | 0.216673 | 0.385690 | 0.456489 | 47.0% | 15.2% |
| TIME 3 CPHL/H | CRB | 1672 | 0.002196 | 0.156139 | 0.201210 | 0.288075 | 0.536552 | 50.8% | 18.8% |
| TIME 3 CPHL/H | Uniform | 1672 | -0.000749 | 0.157023 | 0.198629 | 0.292177 | 0.513484 | 49.8% | 18.9% |
| TIME 4 ECDC_COVID/D | CRB | 1188 | 0.018231 | 0.175268 | 0.226560 | 0.337262 | 0.509316 | 57.1% | 23.7% |
| TIME 4 ECDC_COVID/D | Uniform | 1188 | 0.027959 | 0.182204 | 0.244717 | 0.350152 | 0.482916 | 59.1% | 26.5% |
| TIME 6 epf_electricity_price/H | CRB | 40500 | 0.003769 | 0.087701 | 0.114698 | 0.177353 | 0.334714 | 52.5% | 7.4% |
| TIME 6 epf_electricity_price/H | Uniform | 40500 | 0.005470 | 0.089930 | 0.117976 | 0.181709 | 0.339312 | 53.6% | 7.8% |
| TIME 7 Commodity_Import/M | CRB | 250 | -0.011486 | 0.177083 | 0.234808 | 0.325868 | 0.398489 | 45.6% | 20.8% |
| TIME 7 Commodity_Import/M | Uniform | 250 | -0.009287 | 0.196155 | 0.247461 | 0.335832 | 0.378646 | 46.4% | 23.2% |
| TIME 8 Commodity_Production/M | CRB | 9 | -0.021656 | 0.072106 | 0.085432 | 0.096093 | 0.098758 | 33.3% | 0.0% |
| TIME 8 Commodity_Production/M | Uniform | 9 | -0.021656 | 0.072106 | 0.085432 | 0.096093 | 0.098758 | 33.3% | 0.0% |

### 指标三：top10% 筛选后源窗口与安全历史参考的方向接近度

对每个已完成选样批次，使用其实际 window_indices.npy 标记的 266,631 个源窗口集合。从现有 src-96 缓存中筛出全局窗口索引属于该选集的源锚点梯度，与同方法的安全历史锚点计算全部两两方向余弦的平均。源侧只读缓存，不重算源梯度。源侧与参考侧沿用指标一对全池锚点梯度范数计算的 P1 阈值，不根据选中子集重新确定。按批次报告选中源窗口数、全源缓存锚点数、选中锚点数和锚点覆盖率。因为未缓存窗口的源梯度不可知，该指标是基于选中集合内真实缓存锚点的估计，不等于全部 top10% 窗口的真值。

指标三的 MASE 主表汇总列为各任务 Uniform traj、Uniform block、CRB traj、CRB block 四个完整 RR 批次指标的等权平均；其他完成的 win RR 批次仍在指标详细表单列。

计算脚本：`scripts/exp/v1-crb-redo/metrics.py`。补算已完成，8 项任务各有 CRB 与 Uniform 两行结果；输出、逐项 CSV、梯度草图与清单位于 `outputs/selection/v1-crb-redo/sgwdev/metrics/`，成功标记为该目录 `.complete`。实际数值、样本计数及输入路径已登记于 results.md 与全局索引。

本实验后续每个新增评测协议批次完成时，自动计算并登记上述三项统计；计算使用该批次的评测协议、真实评测样本和对应安全历史参考集合，结果、样本计数与配置写入同批次输出和结果文档，无需另行触发。
