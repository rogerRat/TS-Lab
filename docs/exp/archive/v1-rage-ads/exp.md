# v1-rage-ads：src-96 自适应密度重采样与滚动评估

## 基础信息

| 字段 | 内容 |
|---|---|
| 实验代号 | v1-rage-ads |
| 目标 | 对 src-96 与八个下游任务安全历史计算 ADS 概率描述；按该分布有放回训练 TimerXL，并进行滚动评测 |
| 假设 | 真实锚点余弦匹配能在 source 窗口起点上提供任务相关的局部重采样概率；所有 source/reference 梯度均来自相同热身检查点和投影坐标系 |
| 文档创建时间 | 2026-11-14 |
| 最后修改时间 | 2026-11-14 |
| 实际实验启动时间 | 首次尝试 2026-11-14 04:20:47 CST；修复后重启 2026-11-14 04:27:49 |
| 当前状态 | 八任务选择、梯度审计、32 个训练/rolling 评估单元、下游汇总及最终零概率支持审计均完成；TIME08 短历史边界已修复，最终补充审计于 2026-11-14 14:12 CST 完成 |

## 模型、梯度与训练

### 梯度探针

| 项目 | 冻结设置 |
|---|---|
| 模型 | TimerXL 84M，`thuml/timer-base-84m`，本地权重 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m` |
| 梯度检查点 | 复用 `/root/TS-Lab/outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000`；src 与 reference 梯度均只能由经过 src-8192 热身 1,000 步的检查点计算 |
| 热身 | 从原始权重与 seed 0 LoRA 初始化；src-8192、576 点窗口、起点步长 1；q/v projection LoRA，r=8、alpha=16、dropout=0；AdamW 1e-4、weight decay 0、有效 batch 32、微批次 8、梯度裁剪 1、cosine 4000 步时间轴、3% warmup，在第 1000 步保存 |
| 损失窗口 | TimerXL 原生 V6，480 历史点 + 96 预测点；所有梯度锚点相同 |
| 投影 | 4,096 维 Count-Sketch，seed 42；模型权重及前向 BF16，梯度与投影累加 FP32，草图以 BF16 位编码保存 |
| source 缓存 | 只读复用 v1-crb-redo 的现有 source 梯度缓存，不复制或重新存储既有草图；逐字段检查热身检查点、窗口、轨迹、损失、参数及投影身份，不用哈希。source 粗锚点直接采用缓存 uniform 清单中的 q=32 窗口起点，避免通用端点网格的起点偏移造成重算；细阶段先查同一 source 缓存，只有缓存缺失的所需锚点才前向/反向计算，并仅留在本次 ADS 内存中 |
| reference 缓存 | 按本实验 rolling 安全历史重新提取并计算，不复用 single-window 安全历史梯度 |
| 运行环境 | 远程 Conda `rage-timerxl`，Python 路径 `/root/autodl-tmp/rage/envs/rage-timerxl/bin/python` |

### ADS 参数

| 参数 | 值 | 定义 |
|---|---:|---|
| q | 16 | source 与 reference 两侧每阶段目标锚点密度之和为 1/q |
| eta | 1.0 | reference/source 目标锚点密度比，必须显式配置 |
| source 间隔 | 32 | `q(1+eta)` 个合法窗口起点 |
| reference 间隔 | 32 | `q(1+eta)/eta` 个合法窗口起点 |
| 粗/细阶段 | 共用上述派生间隔 | 不再分别设置两个 q；粗细两阶段实际新增梯度分别计数 |
| rho | 0.2 | 每轮从完整粗锚点相似度矩阵中取全局前 20% 配对；活动 ADS 默认值 |
| 细算与重匹配轮数 | 1 | 粗匹配后细算一次并重匹配 |
| 衰减距离 h | 4 | 单侧线性衰减；距离 4 权重为 0 |
| 锚点分数聚合 | max | 对每个 source 锚点，沿所有 reference 锚点聚合余弦取最大值 |
| 重叠邻域 | max | 同一 source 起点被多个锚点覆盖时取最大单锚点贡献，不相加；重叠不会把权重抬高到锚点分数以上 |
| 重采样 | 有放回，seed 0 | 训练时直接依据每轨锚点近邻概率描述抽取，不生成固定窗口索引集 |

### 锚点分布与匹配偏置声明

reference 锚点预计并非均匀分布。全局 Top 20% 配对按锚点配对计数，因此局部包含更多真实粗锚点的片段可能获得更多匹配机会和更大话语权；本实验不将其重加权为每条轨迹或每个时间片段等权。这是有意设计：密集命中表示该局部有更多真实锚点证据，其他区域若没有粗筛命中或近邻证据，就不应自动获得同等权重。该密度效应属于匹配阶段；最终单点邻域重叠仍取 max，避免邻域叠加将点权重抬高。结果报告须给出各任务 reference 轨迹数、粗锚点数、命中数及区间分布，并将该偏置作为解释限制。

## 数据与评估

### Source 池

- 使用 `configs/data/src-96.yaml` 与 `data/src/shard_dataset/src-96/` 的 96 条原始完整轨迹；不额外清洗、归一化或截断。
- source 粗锚点以 `outputs/train/v1-crb-redo/grad-prepare/gradients/timerxl-84m/worker-*/uniform/*.json` 清单中的物理起点为准。只核对并读取现有缓存行；不重算、复制或另存这些 source 草图。现有 q=32 网格相对通用 ADS 端点网格有轻微取点差异，因此本批次显式锁定源缓存网格，reference 仍用滚动安全历史生成的 ADS 均匀粗网格。
- TimerXL 窗口由 480 历史 + 96 预测组成，候选窗口起点步长为 1；v1-crb-redo 记录的合法窗口总数为 2,666,312，执行时重新核对清单。
- 每个下游任务独立生成一份覆盖全 source 候选起点的概率描述。描述保存锚点起点/分数、合法起点范围、h、归一化质量与均匀回退标志，不保存训练抽样索引。

### Reference 及滚动安全边界

使用与 v1-crb-redo 相同锁定数据版本，但以滚动协议最早测试 cutoff 前的前缀重建任务内安全历史。每个任务的 reference 在所有滚动窗口中固定；任一测试目标值不得进入 reference。多变量数据按评测器实际目标通道分别展开；非有限值按连续有限段处理，不足 576 点的片段记录并排除。窗口必须完整落在安全历史内，起点步长为 1。

| 任务 | 基准 / 目标 | 预测长度 H | Reference 边界 |
|---|---|---:|---|
| FEV 9 | hospital_admissions_1D | 28 | FEV 全部声明 rolling cutoff 中最早答案起点之前的实际输入历史 |
| TIME 1 | CPHL/15T | 12 | 所有 rolling 窗口最早测试 cutoff 前缀 |
| TIME 2 | CPHL/30T | 12 | 同上 |
| TIME 3 | CPHL/H | 24 | 同上 |
| TIME 4 | ECDC_COVID/D | 30 | 同上 |
| TIME 6 | epf_electricity_price/H | 24 | 同上 |
| TIME 7 | Commodity_Import/M | 12 | 同上 |
| TIME 8 | Commodity_Production/M | 12 | 同上 |

FEV 任务定义沿用 `configs/eval/fev/v1-crb-redo-timerxl.yaml`、`configs/eval/fev/v1-crb-redo-timerxl-tasks.yaml` 与本地数据修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`。TIME 任务定义沿用 `configs/eval/time/v1-rage-trev-timerxl.yaml`、TIME 源代码修订 `b6e7ba9862acb9233c8c55b314c1777052c420ff` 与数据集修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`。

### 训练与滚动评估

每个任务独立训练一个 TimerXL 84M 模型，从原始预训练权重和 seed 0 初始化；梯度探针的热身 LoRA 仅用于 ADS 梯度计算，不作为学生训练初始化。训练沿用 v1-crb-redo：BF16 LoRA q_proj/v_proj，r=8、alpha=16、dropout=0；AdamW lr=1e-4、weight decay=0、betas=(0.9,0.999)、eps=1e-8；有效 batch 32、微批次 8、梯度裁剪 1；4000 次优化器更新，cosine 日程，3% warmup，min_lr_ratio=0；每 500 步保存恢复状态。训练每步从当前任务 ADS 分布有放回抽 32 个窗口，seed 0，总计每任务 128,000 次抽样。

评估全程使用滚动预测，不运行 single-window 模式。FEV 使用其配置声明的全部 cutoff；TIME 按完整 test_length 生成滚动窗口。沿用 v1-crb-redo 的任务指标、预测定义、batch 32、BF16 推理与其他任务配置；所有评测 seed 改为 0。记录每个 cutoff/窗口的原始指标和有效样本数，再报告任务汇总指标；不得把评测目标并入 reference 或 ADS 选择。

### 训练臂与抽样分布

| 训练臂 | 选择/候选池 | 每步训练抽样 |
|---|---|---|
| fullRand | src-96 全部 2,666,312 个合法 480+96 窗口起点 | 全池均匀、有放回 |
| rage | 主 ADS：q=16、eta=1、rho=0.2、细化轮数 R=1、h=4、max 锚点聚合与 max 邻域重叠 | 在完整 source 窗口总体上按 ADS 概率描述有放回抽样，不保存固定窗口筛选集 |
| ancUni | ADS 锚点均匀控制：q=16、eta=1、rho=0.2、R=0；不计算细锚点 | 在完整 source 窗口总体上按控制臂概率描述有放回抽样 |
| Traj | 对每条轨迹的全部粗锚点方向取均值并归一化，按全局 rho=0.2 配对匹配选出 source 轨迹 | 按轨迹概率抽轨迹，再在所抽轨迹的全部合法窗口中均匀抽样；有放回 |

四臂都按各自 sampler 有放回生成训练窗口；不把 ADS 概率压成固定的硬筛选窗口清单。每个任务的 `sampling.json` 和概率描述是训练、审计共同读取的采样依据。四臂使用相同学生训练预算、TimerXL 配置与滚动评测流程；后续新增对比作为新训练臂追加到本实验，不覆盖现有记录。

## 选择与梯度审计指标

审计在训练前按八任务、四训练臂分别执行，结果只描述采样分布和梯度代表性，不改变训练样本规则。

| 审计项 | 计算口径 | 样本与协议 | 输出 |
|---|---|---|---|
| Source 池对齐分布 | 每个 source 梯度对 rolling-safe reference 锚点取最大余弦，报告均值、标准差、最小/最大值和 p01/p05/p10/p25/p50/p75/p90/p95/p99；另按 v1-crb-redo 的方向口径，分别以 source/ref 梯度范数 P1 阈值置零后报告全配对平均 cos 及每个 source 单元的参考平均 cos 分布 | 使用现有 q=32 uniform source cache 的全部唯一锚点，按窗口等权；source/ref 均用 480+96 窗口与热身 step-1000 探针 | `outputs/selection/v1-rage-ads/rolling/audit/source-alignment.jsonl` |
| 各臂采样对齐分布 | 对每任务每臂按实际 sampler 抽取窗口；报告每窗口 best-reference cos 及分位数、与 source q=32 基线之差；另按 P1 阈值置零后的 v1-crb-redo 配对方向口径报告平均 cos、全配对分布和逐窗口参考平均 cos 分位数 | 每任务每臂默认 4,000 次、有放回；seed=42。重复抽中的窗口按出现次数计入统计，唯一窗口梯度可去重计算。fullRand 全池均匀；rage 和 ancUni 按各自 ADS 概率描述抽样；Traj 先按轨迹分布抽轨迹，再在轨迹内均匀抽窗 | `outputs/selection/v1-rage-ads/rolling/audit/arm-alignment.jsonl` 与每任务抽样 ID 文件 |
| Reference 连续性 | 每个 rolling 测试起点计算相对安全历史 task-H 梯度的 best-reference cos 分布；并复刻历史审计的全体历史锚点—测试起点配对平均、P50/P90/P95/P99、最大值和正值比例 | 使用全部 rolling 起点与真实预测长度 H；reference/test 都是 480+H 窗口、step-1000 模型和相同投影。两两分位数最多抽 2,000,000 对，报告抽样数和 seed；不反馈选择或训练 | `outputs/selection/v1-rage-ads/rolling/audit/reference-continuity.jsonl` |
| ADS 分数矩阵分布 | 分别记录粗匹配轮和细化后最终矩阵的元素数、精确零比例、`abs(score)<=1e-8` 与 `abs(score)<=1e-3` 比例、正/负比例、均值/标准差/min/max、p01/p05/p10/p25/p50/p75/p90/p95/p99 | raw cosine matrix 的稀疏度与符号比例按全矩阵精确计数；分位数用固定 seed=42、有放回均匀抽取最多 2,000,000 个矩阵元素估计；精确零、近零和符号比例按矩阵全量计数。另报 Top-`rho` 匹配掩码稀疏度，实际匹配数必须与 Top-`rho` 预算核对 | `outputs/selection/v1-rage-ads/rolling/audit/score-matrix.jsonl` |
| Source P99 高匹配聚集 | 对 q=32 source 锚点按 best-reference cos 取 top 1%，报告命中轨迹数、有效轨迹数、top-10 轨迹占比、同轨迹相邻命中数和最长连续段；Traj 轨迹级 top 1% 只有 1 条时只报告最佳轨迹 | 与 v1-crb-redo 相同，按每轨迹固定高分锚点个数做 500 次均匀重排，seed=42；上尾比例仅作描述性聚集诊断，不解释为独立样本显著性 | `outputs/selection/v1-rage-ads/rolling/audit/source-p99-clustering.jsonl` |
| 最终采样零概率支持 | 按最终四臂采样规则逐个计数 2,666,312 个 source 合法窗口起点中 p=0 的比例；ADS描述复用与训练相同的max线性邻域支持，Traj按未选轨迹窗口数计数，fullRand均匀全池 | 精确计数，不抽样；八任务四臂全覆盖 | `outputs/selection/v1-rage-ads/rolling/audit/probability-support/probability-support.jsonl` 与 `summary.json` |

source q=32 网格是现有梯度缓存覆盖的确定性窗口样本，不等同于对 2,666,312 个 source 窗口逐一计算的全量对齐分布；因此报告中称为“source q=32 缓存锚点基线”。4,000 次臂内抽样才是本次各训练分布的窗口级对齐估计。两者的总体量与抽样机制不同，解释时同时展示分布和抽样口径。

Reference 连续性采用任务实际 H 的单独诊断梯度；ADS 选点的 source/reference 余弦矩阵仍使用缓存协议的 480+96 梯度。两类结果用途不同，不能混作同一统计量。任何测试窗口或测试标签只用于训练后的连续性审计，不进入 reference、ADS 矩阵、采样概率或训练数据。

所有新增梯度仅可由 `/root/TS-Lab/outputs/train/v1-crb-redo/grad-prepare/warmup/timerxl-84m/adapter-step-1000` 热身检查点计算。Source q=32 缓存复用须继续进行字段身份核对，不复制已有草图；审计输出写到 `outputs/`，控制台和文件日志遵守项目规范。

## 并行效率试跑与计时口径

### 已完成选择的串行基线与计时定义

选择阶段在 2026-11-14 04:27:49 启动，旧实现按八个任务顺序执行，任务内四臂也顺序处理。历史队列总墙钟约 6.4 小时仅表示八个任务的串行累计时间；本实验计时按单任务串行报告，不能把 6.4 小时当成单任务耗时或并行运行耗时。任务之间独立，具备任务级并行条件。

| 口径 | 串行墙钟耗时 | 平均 GPU 显存 | 平均主机内存 | 说明 |
|---|---:|---:|---:|---|
| 单任务，四臂合计 | 44.67–50.52 分钟，均值 47.75 分钟 | 历史运行未采样 | 历史运行未采样 | 从八个任务逐任务完成时间戳计算 |
| rage | 18.75–25.28 分钟，均值 22.67 分钟 | 历史运行未采样 | 历史运行未采样 | 相邻训练臂完成时间戳差 |
| ancUni | 11.87–14.32 分钟，均值 12.80 分钟 | 历史运行未采样 | 历史运行未采样 | 同上 |
| Traj | 11.35–14.93 分钟，均值 12.27 分钟 | 历史运行未采样 | 历史运行未采样 | 同上 |
| fullRand | 约 0 秒 | 历史运行未采样 | 历史运行未采样 | 复用全 source 池，不运行 ADS 梯度选择 |

旧容器日志没有显存和主机内存周期采样。仅有一次 GPU 0%、显存 922 MiB 的瞬时记录，不能推算历史平均值。下表的单任务串行资源值来自本次受控 profile，且按阶段、任务与采样协议分别标记；不回填为旧选择运行的历史资源值。

### 本次性能优化与不变的实验工作量

- 梯度计算改用测得较快的 32-window / VJP-32 批次。每个窗口仍使用同一 step-1000 warmed TimerXL 探针、同一 480+96 窗口和同一 4,096 维投影；只改变向量化批次，不减少窗口数或梯度。
- 审计把全部 83,330 个 q=32 source 锚点的归一化向量保存为共享只读 memmap（outputs/selection/v1-rage-ads/parallelism/source-q32-unit-cache/，约 1.36 GiB）。缓存覆盖完整 q=32 锚点总体；审计仍计算全部 source 锚点与各任务 reference 的对齐，未抽减总体。
- 审计的重复派生量通过共享缓存复用；打分采用 GPU 分块稠密矩阵计算；并行任务用锁保护 JSONL 更新并各自写完成清单。抽样仍为每任务每臂 4,000 次、seed=42，重复窗口只复用唯一窗口梯度，重复次数仍按抽样出现频率统计。
- 训练 sampler 使用紧凑概率描述直接抽样，保持完整 2,666,312 个合法 source 起点及原有抽样分布，避免构造数百万元素的 dense 概率数组。抽样器微基准为 994,688.9 draws/s，dense rng.choice(p=...) 为 1,328.9 draws/s；该比值仅代表抽样操作，不代表端到端训练加速。
- 训练仍为 32 个任务/训练臂单元、每单元 4,000 次更新、每步有效 batch 32、128,000 次窗口抽样；评估仍完整使用 rolling 配置。下列 32/128-step 试跑仅用于并发容量测量，不替代正式训练预算。
- 并行调度器每秒记录 GPU、主机内存、worker RSS 与 CPU；worker 独立进程组支持中断和回收。正式队列按审计 8 workers、训练/评估 12 workers 启动，实际任务未减少。

### 梯度准备并行试跑

所有 profile 窗口梯度均由 step-1000 warmed TimerXL 检查点计算。每个并发设置每 worker 1,024 个窗口；记录窗口吞吐和约 1 秒采样的 GPU、主机资源。

| window batch / VJP batch | 单 worker windows/s |
|---|---:|
| 4 / 4 | 117.5 |
| 8 / 4 | 149.5 |
| 16 / 4 | 165.3 |
| 32 / 4 | 170.9 |
| 32 / 8 | 293.5 |
| 32 / 16 | 431.0 |
| 32 / 32 | 608.1 |

32/32 比 32/16 的单 worker 吞吐高 41.1%，GPU 平均利用率为 49.3%，显存均值/峰值为 821/980 MiB；单 worker 平均约 1.01 CPU 核心当量，主机 CPU 总利用率均值 12.6%。

| worker 数 | 总吞吐 windows/s | GPU 均值利用率 | 显存均值/峰值 MiB | 主机内存均值/峰值 MiB | worker CPU 核心当量 |
|---:|---:|---:|---:|---:|---:|
| 1 | 608.1 | 49.3% | 822 / 980 | 110,992 / 110,995 | 1.01 |
| 2 | 931.8 | 74.8% | 1,535 / 1,958 | 112,068 / 112,075 | 1.98 |
| 4 | 1,081.2 | 82.0% | 3,163 / 3,912 | 114,150 / 114,165 | 4.00 |
| 8 | 1,112.8 | 77.0% | 6,116 / 7,820 | 118,268 / 118,287 | 7.98 |

8 workers 比 4 workers 仅快 2.9%，而显存均值接近翻倍；该试跑用于确认可行并发边界，正式配置按吞吐和资源余量综合选择。源/参考新梯度的正式选择脚本现默认 32/32。输出：outputs/selection/v1-rage-ads/parallelism/gradient-parallelism-benchmark-vjp32.json；此前 32/16 扫描保留在 gradient-parallelism-benchmark.json。

### 正式抽样规模的审计并发试跑

TIME01 代表任务，每臂按正式规则抽样 4,000 个窗口；四臂梯度、source 对齐、连续性及 score-matrix 审计均包含在一次 task job 内。CPU threads/worker=2。

| workers | task/s | GPU 均值 | 显存均值/峰值 MiB | 主机内存均值/峰值 MiB | worker RSS 均值/峰值 MiB | worker CPU 核心当量 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0131 | 18.0% | 602 / 2,262 | 113,147 / 114,643 | 4,306 / 5,452 | 1.02 |
| 2 | 0.0227 | 40.5% | 1,195 / 3,682 | 116,203 / 118,282 | 8,509 / 10,809 | 2.02 |
| 4 | 0.0349 | 57.6% | 2,532 / 8,784 | 121,384 / 125,310 | 16,898 / 21,245 | 4.01 |
| 8 | 0.0497 | 77.1% | 5,036 / 16,524 | 109,332 / 138,437 | 31,637 / 43,091 | 7.44 |

8 workers 在本次 1/2/4/8 扫描中吞吐最高，正式审计据此使用 8 workers。8-worker 试跑期间主机 CPU 总利用率均值约 16.0%，不是主机 CPU 饱和；源单位向量缓存及批量梯度把瓶颈从重复 CPU 读取/归一化转向 GPU 批计算。主机内存是系统级读数，会受同机其他进程影响；worker RSS 是该试跑进程树的 RSS 合计。

此前 64 抽样 profile 也完整保留：无源单位向量缓存的 8-worker 为 0.0926 task/s；加入完整源单位缓存和 GPU 分块后为 0.3625 task/s（同为 64 抽样、32/16 梯度批次）。此微型 profile 用于验证缓存/并行调度变化，正式并发决策以 4,000 抽样 profile 为准。文件分别为 outputs/selection/v1-rage-ads/parallelism/audit-parallel-profile.json 与 audit-parallel-profile-unitcache.json；正式规模试跑为 audit-parallel-profile-sample4000.json。

正式八任务审计于 2026-11-14 12:39:36 完成，实际墙钟 165.3 秒，8 workers，sample_count=4,000、seed=42、梯度 window/VJP batch=32/32。实测 GPU 利用率均值 82.1%，显存均值/峰值 4,683/15,218 MiB；进程树 RSS 均值/峰值 27,985/37,988 MiB；主机内存均值/峰值 179,772/187,941 MiB；worker CPU 均值 6.48 核心当量，主机 CPU 利用率均值 16.1%。完整结果见 outputs/selection/v1-rage-ads/rolling/audit/parallel-run-manifest.json；源对齐、四臂抽样对齐、reference continuity、score matrix 与 P99 聚集分别在该目录的 source-alignment.jsonl、arm-alignment.jsonl、reference-continuity.jsonl、score-matrix.jsonl、source-p99-clustering.jsonl。

### 训练与滚动评估并发试跑

训练 profile 使用实际 TimerXL LoRA、AdamW、数据 sampler 和训练日程；32-step profile用于粗略筛选并发度，随后增加到每 worker 128 次更新确认趋势。cells/s 是任务/训练臂 cell 吞吐，不是更新/秒。

| 训练试跑 | workers | cell/s | GPU 均值 | 显存均值/峰值 MiB | 主机内存峰值 MiB | worker RSS 峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 32 updates/worker | 8 | 0.5964 | 29.0% | 3,950 / 7,756 | 76,823 | 17,305 |
| 32 updates/worker | 12 | 0.7123 | 35.2% | 5,985 / 11,635 | 89,422 | 25,437 |
| 32 updates/worker | 16 | 0.6343 | 34.9% | 7,720 / 15,512 | 115,687 | 34,103 |
| 32 updates/worker | 24 | 0.5997 | 32.8% | 12,264 / 23,265 | 139,629 | 49,685 |
| 128 updates/worker | 8 | 0.3135 | 59.7% | 5,565 / 7,756 | 123,661 | 15,629 |
| 128 updates/worker | 12 | 0.4036 | 66.4% | 8,577 / 11,635 | 131,503 | 24,587 |
| 128 updates/worker | 16 | 0.3395 | 58.2% | 11,610 / 15,512 | 139,321 | 33,404 |

128-update 稳态试跑中 12 workers 吞吐较高且显存峰值约为设备总量的 35%，为正式任务留有余量。16/24-worker 测试未带来稳定收益且资源更高。正式训练及 cell 内 rolling 评估选择 12 workers。32-step 与 128-step 结果分别在 outputs/train/v1-rage-ads/parallelism/train-parallel-profile-extended.json 和 train-parallel-profile-steady128.json；1/2/4/8 基准保留在 train-parallel-profile.json。

评估并发试跑完整运行 TIME01 rolling 配置，但使用本地预训练 TimerXL 权重，只用于测评估阶段吞吐，不是正式模型指标；正式评估仍用各训练 cell 合并后的检查点。

| workers | TIME01 rolling jobs/s | GPU 均值 | 显存均值/峰值 MiB | 主机内存峰值 MiB | worker RSS 峰值 MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1834 | 0.4% | 199 / 528 | 111,321 | 1,444 |
| 2 | 0.3662 | 0.4% | 398 / 1,054 | 112,394 | 2,885 |
| 4 | 0.6027 | 2.3% | 712 / 2,104 | 114,549 | 4,807 |
| 8 | 1.0312 | 16.9% | 1,513 / 4,184 | 118,545 | 11,222 |
| 16 | 1.3571 | 11.1% | 2,374 / 8,394 | 123,222 | 18,794 |
| 24 | 1.1780 | 6.1% | 2,909 / 10,666 | 128,967 | 25,487 |
| 32 | 1.0748 | 7.2% | 4,445 / 14,703 | 139,949 | 40,335 |

TIME01 独立评估试跑的吞吐在 16 workers 最高；24/32 workers 增加 RSS 和显存且变慢。整体 train/eval runner 使用 12 workers，以训练稳态配置为准并留资源余量。profile 输出保留在 outputs/eval/v1-rage-ads/parallelism/eval-parallel-profile-rolling.json 与 eval-parallel-profile-extended.json。

### 单任务串行与并行耗时的解释

历史选择单任务串行均值为 47.75 分钟，旧队列八任务串行累计约 6.4 小时。旧选择执行未采样平均 GPU 显存或主机内存，不对这两个历史均值作补值。本次审计 4,000 抽样的 TIME01 单任务 profile 为约 1.27 分钟，GPU 显存均值 602 MiB、主机内存均值 113,147 MiB、worker RSS 均值 4,306 MiB；这是审计单任务 profile，不可替代选择运行的资源值。正式阶段按审计 8 workers、训练/评估 12 workers 并行，真实耗时以正式 runner manifest 为准，不把 6.4 小时写成当前队列实际耗时。

### 正式训练与滚动评估执行记录

- 2026-11-14 12:39:36 CST 首次以 12 workers 启动 32 个训练/评估单元。首轮运行 27.7 分钟后已有 28/32 单元成功；剩余四个均为 TIME08，各自训练都达到 4,000 步，失败发生在 TIME rolling 评估阶段，不涉及训练预算减少。
- 故障原因是 TIME08 `Commodity_Production/M` 的早期 rolling 输入历史短于 TimerXL 模型的最小 96 点输入。修复在 TimerXL 预测入口对短历史使用最早观测值进行左侧 edge padding，补到模型的 96 点最小 patch；所有原有 rolling 窗口和测试目标均保留，没有跳过或减少评估样本。另将训练/评估入口改为对已有未完成评估目录使用评估器的 `--resume`，支持从元数据写入后的中断点恢复。
- fullRand 单元于 13:21:56–13:22:02 CST 单独恢复成功；其余 rage、ancUni、Traj 三单元以 12-worker runner 并行恢复，13:22:13–13:22:25 CST 全部成功。最终训练与评估单元均为 32/32 完成，根完成标记写入，13:22:44 CST 生成 32 行指标与 18 项描述性关联汇总。
- 首轮 runner 在失败退出前没有写出其内存中的资源采样 manifest。成功恢复生成的 `outputs/train/v1-rage-ads/rolling/parallel-run-manifest.json` 只覆盖本次恢复的三个单元，包含 10 个短期采样点；不能视为完整 32 单元训练阶段的资源均值。该恢复阶段 GPU 利用率均值 9.6%、显存峰值 1,561 MiB、主机内存均值/峰值 167,866/170,333 MiB。训练并发配置参考 128-step 实测 profile：12 workers 吞吐最高且 GPU 显存峰值 11,635 MiB；完整正式首轮的 GPU/主机均值未保留，故不从恢复或 profile 数据回填。
- 首轮 27.7 分钟加四单元恢复约 18 秒是本批次可确认的计算运行时间。13:07:17 首轮退出后至 13:21:56 修复诊断区间属于等待/排障，不计作任务计算耗时。八任务历史串行累计约 6.4 小时不是本批次实际耗时。

## 决策确认与暂定项

| 决策 | 当前值 | 确认状态 |
|---|---|---|
| 实验范围 | src-96，FEV 9 与 TIME 1/2/3/4/6/7/8 | 用户已确认 |
| 评测协议 | 全部使用 rolling 预测；reference 按最早 rolling cutoff 的安全历史重建 | 用户已确认 |
| 训练臂 | fullRand、rage、ancUni、Traj；后续对比待补充 | 当前四臂已确认；后续待定 |
| ADS 参数 | q=16、eta=1、rho=0.2、细化轮数 R=1、h=4、max 聚合、max 重叠邻域 | 按用户“按照你的建议来”冻结 |
| RAGE/ancUni 对照含义 | 两臂共享 q=16、eta=1 与粗锚点；rage 为 R=1，ancUni 为 R=0 | 作为是否细化的消融，不按总梯度计算量匹配 |
| 梯度身份 | TimerXL 84M，step-1000 warmed LoRA，480+96，Count-Sketch 4096，seed=42 | 按上一批已验证协议冻结 |
| 学生训练 | seed=0，4000 更新，每步有效 batch 32，合计每任务 128,000 次窗口抽样 | 按 v1-crb-redo 方案冻结 |
| 对齐抽样 | 每任务每臂默认 4,000 次，有放回，seed=42；支持通过 CLI 调整次数 | 4,000 默认由用户提出；seed=42 为本次审计默认 |
| 分位数 | p01/p05/p10/p25/p50/p75/p90/p95/p99 | 本次审计默认 |
| 分数矩阵近零阈值 | abs(score)<=1e-8 与 abs(score)<=1e-3 | 本次审计默认 |
| Pairwise 分位数样本 | 最多 2,000,000 对；超出时均匀有放回抽样，seed=42 | 复用 v1-crb-redo 的估计口径 |
| Source P99 聚集审计 | best-reference top 1%，每轨迹命中数固定的 500 次随机重排，seed=42 | 复用 v1-crb-redo 的审计口径 |
| 最终概率零值比例 | 以最终 sampler 对所有 source 窗口起点的概率是否精确为零定义；同时汇总 fullRand/rage/ancUni/Traj，明确区分候选窗口支持比例与抽样样本中的命中率 | 对压缩 ADS descriptor 按训练 sampler 构造完全相同的正支持；Traj按选中轨迹及其轨迹概率；全 source 共2,666,312起点，精确计数 | 新增审计，覆盖八任务四臂 |
| 分数矩阵分位抽样 | 最多 2,000,000 个元素，有放回，seed=42 | 本次审计默认 |

审计默认值仅影响描述统计，不改变 ADS 选择、训练抽样或评测预测。若后续修改，更新本表、脚本参数与结果 manifest，并保留已完成批次的历史结果。

## 执行与产物

| 项目 | 路径 / 状态 |
|---|---|
| 算法规范 | `docs/ads.md` |
| 本实验正式配置 | `configs/eval/fev/v1-rage-ads-fev-09.yaml` 与 `configs/eval/time/v1-rage-ads-time-*.yaml`；均显式设置 rolling |
| 本实验专用 runner | `scripts/exp/v1-rage-ads/run_selection.py`、`audit_metrics.py`、`train_eval.py`、`run_parallel_audit.py`、`run_parallel_train_eval.py`、`parallel_profile.py` |
| 概率描述 | `outputs/selection/v1-rage-ads/rolling/` |
| 热身 / 梯度素材 | `outputs/train/v1-rage-ads/rolling/`；每任务审计用 ADS q=16 reference 粗梯度副本为 `gradients/<task>/reference_coarse.npz` |
| 训练结果 | `outputs/train/v1-rage-ads/rolling/` |
| 滚动评测 | `outputs/eval/v1-rage-ads/rolling/` |
| 对齐、连续性、P99 聚集与分数矩阵审计 | `outputs/selection/v1-rage-ads/rolling/audit/`；入口 `scripts/exp/v1-rage-ads/audit_metrics.py` |
| 下游指标与对齐关联 | `outputs/selection/v1-rage-ads/rolling/audit/downstream-metrics.jsonl`、`alignment-outcome-correlations.json`；入口 `scripts/exp/v1-rage-ads/summarize_results.py` |
| 任务级审计与 MASE 拼接总表 | `docs/exp/v1-rage-ads/results.md` 的“八任务审计与 MASE 总表”；现含最终 p=0 窗口比例及平均名次，MASE 每行最小值加粗并标 `*` |
| 最终概率支持审计 | `scripts/exp/v1-rage-ads/audit_probability_support.py`；输出位于 `outputs/selection/v1-rage-ads/rolling/audit/probability-support/` |
| 日志 | `logs/v1-rage-ads/` |
| 数据盘 | 项目 `outputs/`、`data/` 软链接实际位置；大型文件须落在 `/root/autodl-tmp/` |

审计命令（选择阶段产物齐备后）：conda activate rage-timerxl && python -u scripts/exp/v1-rage-ads/audit_metrics.py --task all --sample-count 4000 --seed 42。成功后在审计输出目录写 .complete。

选择脚本先逐字段核对 source 缓存和 step-1000 热身身份，再为滚动安全 reference 生成 q=16 粗梯度；R=1 与 R=0 两臂共用相同粗锚点，R=0 的 ancUni 仅保留粗层，R=1 的 rage 追加一轮细化；新增细锚点也由同一 step-1000 热身模型计算。该设置对比细化开关，不等价于相同总梯度预算。审计按实际 `sampling.json` 抽样 4,000 个窗口，默认 seed=42。队列依次运行选择、四类审计、八任务四臂训练及 rolling 评估，最后复刻 v1-crb-redo 的任务级下游收益与对齐量描述性相关分析（8 个任务、单 seed，不作因果或显著性推断）。

正式执行步骤：验证 src-96 与任务数据版本；提取每任务最早 rolling cutoff 前的 reference；逐项验证答案隔离和 576 点窗口合法性；复用暖身 source q=32 梯度锚点并为滚动 reference 计算暖身 q=16 梯度；生成 `rage`、`ancUni` 与 `Traj` 的概率 sampler；先完成 source/arm 对齐、reference 连续性和分数矩阵审计；再训练八任务四臂并执行 rolling 评估；核对计数、输出和指标后写批次 `.complete`。脚本入口为 `scripts/exp/v1-rage-ads/queue_rolling.sh`。

## 批次管理

| 批次 | 内容 | 状态 | 输出 |
|---|---|---|---|
| ads-api-q-single | 单一 q/eta 算法接口、CLI/YAML 参数、紧凑概率描述与重采样函数验证 | 完成：25 项 ADS 定向测试通过；outputs/selection/v1-rage-ads/ads-api-q-single/.complete 已生成 | 仅实现验证，无模型指标 |
| rolling-eight-task | 安全历史重建、ADS 概率选择、四类梯度审计、按概率训练、八任务 rolling 评估及任务级关联分析 | 完成：八任务四臂 32/32 训练与评估单元通过；13:22:44 CST 汇总完成 | 32 行正式指标和 18 项描述性关联见 results.md 与 downstream-metrics.jsonl；TIME08 短历史问题及恢复过程已记录 |

