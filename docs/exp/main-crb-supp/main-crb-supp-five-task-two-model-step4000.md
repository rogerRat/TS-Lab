# main-crb-supp 五任务跨模型 step-4000 结果总结

- 文档用途：为论文实验设置与结果段落提供可直接引用的批次细节与数值。
- 覆盖任务：`time-crypto-d`、`time-ne-china-wind-h`、`time-sg-pm25-h`、`fev-jena-weather-1h`、`fev-ett-1h`。
- 覆盖模型：TimerXL 84M（实验 `main-crb-supp`）与 Chronos-2-small（实验 `main-crb-supp-chronos2-small`）。
- 结果口径：只报告 step = 4000 的最终 checkpoint。step 1000/2000 的评估点已存在但不在本文档内。
- 数据读取时间：2026-11-17 20:34 CST（按当前原始产物复核；五个任务 step-4000 数值与 19:32 版本完全一致）。
- 完成状态：五个任务 × 六臂 × 两模型，共 60 个 step-4000 评估目录，全部存在 `.complete` 与原始指标文件。
- 队列状态：TimerXL 24/24 任务已完成（2026-11-17 14:07 CST 生成 `.complete`）；Chronos-2-small 正按 24 任务队列续跑，截至本次复核已完成 17/24 任务（正在运行 `time-current-velocity-10t`），本批次五个任务均已完成且数值未变。

---

## 1. 实验概述

- 实验目的：在固定 TimerXL 84M source pool、reference 构造、梯度预算和训练条件下，比较源窗口的**锚点排布（Uniform vs. CRB）**与**梯度表示（Mean vs. Shape）**的独立与交互作用，并与 Target-only（只在目标安全历史训练）和 Rand（全源池均匀随机采样）对照。
- 迁移设定：E1–E4 使用**同一套 TimerXL LoRA 梯度探针**进行源选择；被训练、被评估的模型分别为 TimerXL 84M 与 Chronos-2-small。因此 Chronos-2-small 的结果解释为“TimerXL 选择信号迁移到 Chronos-2-small”，不是 Chronos 原生梯度选择结果。
- 对照严格性：同一任务内六个臂共享同一 source pool、reference、seed、训练步数与优化器配置；唯一变化是采样方法（E1–E4）、目标域数据（E5）或均匀随机源窗口（E0）。

## 2. 模型

| 角色 | 模型 | 参数量 | 权重来源 | 备注 |
| --- | --- | ---: | --- | --- |
| 被训练/被评估模型 A | TimerXL 84M | 84M | 本地预训练权重 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m`（thuml-timer-base-84m） | 确定性点预测，评估时按分位层级平铺 |
| 被训练/被评估模型 B | Chronos-2-small | 27,934,624 | Hugging Face `autogluon/chronos-2-small`，固定 revision `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` | 原生概率分位数输出 |
| 选择探针（两模型共用） | TimerXL 84M LoRA 梯度 | 同 TimerXL | 同上 | Count-Sketch 投影到 4096 维，seed 0 |

- 两模型均从预训练权重起训，仅训练 LoRA adapter；TimerXL LoRA 作用于 `q_proj`、`v_proj`，Chronos-2-small 作用于 `self_attention.q`、`self_attention.v`。
- TimerXL 为确定性模型，评估时 9 个分位层级由点预测平铺得到；Chronos-2-small 为概率模型，使用原生分位数并在 9 个层级上插值。

## 3. 任务与数据

| 任务代号 | Benchmark | 数据集 | 频率 | 评估预测长度 | 测试长度 | 测试窗口数 | 实体/通道数 | 安全历史合法窗口数 | 参考 q8 锚点数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `time-crypto-d` | TIME | Crypto/D | 日 | 30 | 273 | 9 | 4 | 7,976 | 1,000 |
| `time-ne-china-wind-h` | TIME | NE_China_Wind/H | 小时 | 24 | 720 | 30 | 4 | 29,876 | 3,736 |
| `time-sg-pm25-h` | TIME | SG_PM25/H | 小时 | 24 | 2,208 | 92 | 5 | 44,096 | 5,549 |
| `fev-jena-weather-1h` | FEV | jena_weather_1H（index 10） | 小时 | 24 | FEV 20 评估窗口 | 20 | 21 | 150,213 | 18,795 |
| `fev-ett-1h` | FEV | ETT_1H（index 6） | 小时 | 168 | FEV 20 评估窗口 | 20 | 14（ETTh1/ETTh2 × 7 通道） | 188,790 | 23,604 |

- TIME 数据版本 revision：`83e3d0b3be28d11c7182bffcc1892d19b36c4da1`；TIME evaluator commit：`b6e7ba9862acb9233c8c55b314c1777052c420ff`。
- FEV 数据版本 revision：`d3e7e720f4158eab03ebcb021f1cce779cd92acc`。
- 任务均为 “short” 预测项（TIME）；时间频率与预测长度直接取自 benchmark 定义。
- 训练与评估窗口：`context = 480`、`prediction = 96`、`stride = 1`，即训练窗口长度 576。
- 评估时 context 最多截取最后 512 个点；预测长度使用任务自身的 horizon（TIME 为 24/30，FEV 为 24/168）。因此 FEV 的 ETT_1H 评估 horizon（168）长于训练预测长度（96），这是 benchmark 规定，不是训练配置。

## 4. Source pool、Reference 与 Probe

- Source pool：固定 `src-256`，256 条轨迹，合法窗口总数 6,382,020；窗口规格 context 480 / prediction 96 / stride 1。
- Source 锚点：q = 16，全池共 **399,044** 次真实锚点梯度评估；由相邻锚点形成的采样单元（block）共 **398,788** 个。Uniform 与 CRB 使用完全相同的锚点/block 预算。
- CRB 排布：在最长连续区间内用中点二分放置 cover 锚点，split 使用 min-kappa；预算比例 `cover:error:split = 1:0:1`，不启用 error 阶段。
- Uniform 排布：在相同每轨迹预算下均匀放置锚点。
- Probe：TimerXL LoRA 梯度经 Count-Sketch 投影到 4096 维（projection seed 0）；source 梯度以 BF16 存储，使用时按“FP16 方向 + FP32 模长”重构；reference 使用同一 probe 与同一投影。
- Reference q = 8：每段至少 2 个锚点，其余按 `ceil(合法窗口数/8)` 均匀放置。多段属于同一实体时合并该实体锚点，不以窗口数加权。
- Reference Mean：每个原始实体对 q8 原始投影梯度取均值，再对实体向量做 L2 归一化，最后各实体等权平均。
- Reference Shape：每个实体对相邻 q8 锚点计算 Shape 并取均值，L2 归一化后各实体等权平均。
- Source Mean：block 两端原始投影梯度取均值并 L2 归一化。
- Source Shape：主分支 V7 公式 `[m,(g_R-g_L)/sqrt(12)] / sqrt(||m||^2 + ||g_R-g_L||^2/12)`，零能量置零。
- 打分：source block 向量与 reference 等权均值向量做内积（Linear aggregation）；按 block 得分排序，使用固定 V5 排名映射 `F(x)=(1-exp(-x/T))/(1-exp(-1/T))`，其中 `T=1/(2ln4)=0.36067376`。
- 采样：调用 rank_mass 时所有对象权重为 1，无 source 长度先验、无 reference 长度先验；选中 block 后，在其原生相邻锚点区间归属的合法窗口内均匀抽取，不按 block 长度加权。相邻 block 共享端点归右侧 block，最后一个 block 包含末端，所有合法窗口恰好属于一个 block。
- Reference 安全边界：只使用测试评估开始之前的历史；测试答案不进入 reference 或训练。TIME 按 rolling test tail 截断，FEV 使用第一个评估窗口的 input。

## 5. 六个实验臂

| 臂 | 排布 | 表示 | 采样对象 | 训练数据 | 说明 |
| --- | --- | --- | --- | --- | --- |
| E1 Uniform-Mean | Uniform q16 | Mean | block | 源 block 重采样 | 均匀锚点 + Mean 表示 |
| E2 Uniform-Shape | Uniform q16 | Shape | block | 源 block 重采样 | 均匀锚点 + Shape 表示 |
| E3 CRB-Mean | CRB q16 | Mean | block | 源 block 重采样 | CRB 锚点 + Mean 表示 |
| E4 CRB-Shape | CRB q16 | Shape | block | 源 block 重采样 | CRB 锚点 + Shape 表示 |
| E5 Target-only | 不适用 | 不适用 | target safe window | 安全 reference 的全部合法窗口 | 只训练目标安全历史 |
| E0 Rand | 不适用 | 不适用 | source legal window | src-256 全部合法窗口 | 概率为全源池 6,382,020 合法窗口的均匀分布（ESS = 6,382,020） |

- E1–E4 的每 block 概率质量用于其归属合法窗口内的均匀抽样；四个选择臂的 source 梯度预算完全相同。
- E0 不是 10% 子集，而是 full source pool 上的均匀随机采样。

## 6. 训练配置

| 项目 | 两模型共用设置 |
| --- | --- |
| 优化步数 | 4000 optimizer steps |
| 评估快照 | step 1000 / 2000 / 4000（本文只用 step 4000） |
| Effective batch | 32 |
| Micro batch | 8 |
| 优化器 | AdamW |
| 学习率 | 1e-4 |
| Weight decay | 0 |
| Betas | (0.9, 0.999) |
| Eps | 1e-8 |
| Max grad norm | 1.0 |
| LR schedule | warmup 3%，cosine，min_lr_ratio = 0 |
| LoRA rank / alpha / dropout | r = 8，alpha = 16，dropout = 0 |
| Seed | 0 |
| 设备/精度 | CUDA，BF16 |
| 训练窗口 | context 480 / prediction 96 / stride 1 |
| 训练数据 | E1–E4：源 block 重采样；E5：目标安全历史全部窗口；E0：全源池均匀随机 |

- 运行环境：TimerXL 使用 Conda `rage-timerxl`（`/root/autodl-tmp/rage/envs/rage-timerxl/bin/python`）；Chronos-2-small 使用 Conda `rage-chronos2`（`/root/autodl-tmp/rage/envs/rage-chronos2/bin/python`）。

## 7. 评估协议

- 适配器：TIME 使用 `FoundationTIMEEvaluationAdapter`，FEV 使用 `FoundationFEVEvaluationAdapter`；两者均为 rolling 评估、batch size 32、seed 0、BF16。
- TIME：context 截取最后 512 点，分位层级 `0.1,0.2,...,0.9`；按 benchmark 的 `prediction_length` 对测试尾段滚动切窗。
- FEV：context 截取最后 512 点，对 20 个评估窗口 × 各 target 通道输出预测；预测长度取任务 horizon。
- 主指标：**MASE**，越低越好。
- TIME 的标量 MASE = 对 `metrics.npz` 中形状为 `(1, 测试窗口数, 变量数)` 的 MASE 数组取全局均值（例如 Crypto/D 为 `(1,9,4)`，NE_China_Wind/H 为 `(1,30,4)`，SG_PM25/H 为 `(1,92,5)`）。
- FEV 的标量 MASE 由 FEV evaluator 写入 `summary.csv`，对 20 个评估窗口 × 各 target 通道聚合；同时保留 `SQL`（任务默认指标）、`WQL`、`WAPE`、`MAE`、`MSE`。
- 不同任务的 MASE 量纲不同，**不跨数据集平均 MASE**；跨任务比较只使用名次或胜负计数。

## 8. 结果：step 4000 主指标 MASE

### 8.1 两模型逐任务 MASE 对照

| 任务 | 实验臂 | TimerXL 84M | Chronos-2-small |
| --- | --- | ---: | ---: |
| `time-crypto-d` (Crypto/D) | E1 Uniform-Mean | 7.5048 | 6.4575 |
| | E2 Uniform-Shape | **7.1397** | **6.2169** |
| | E3 CRB-Mean | 7.4299 | 6.3285 |
| | E4 CRB-Shape | 7.1962 | 6.3532 |
| | E5 Target-only | 12.3148 | 9.0780 |
| | E0 Rand | 7.5576 | 6.4226 |
| `time-ne-china-wind-h` (NE_China_Wind/H) | E1 Uniform-Mean | 0.5660 | 0.5428 |
| | E2 Uniform-Shape | 0.5624 | 0.5400 |
| | E3 CRB-Mean | **0.5612** | 0.5394 |
| | E4 CRB-Shape | 0.5631 | 0.5385 |
| | E5 Target-only | 0.6581 | 0.5791 |
| | E0 Rand | 0.5666 | **0.5378** |
| `time-sg-pm25-h` (SG_PM25/H) | E1 Uniform-Mean | 0.8843 | 0.8993 |
| | E2 Uniform-Shape | 0.8847 | 0.9033 |
| | E3 CRB-Mean | 0.8853 | **0.8944** |
| | E4 CRB-Shape | **0.8834** | 0.8968 |
| | E5 Target-only | 0.9090 | 0.9168 |
| | E0 Rand | 0.9099 | 0.8997 |
| `fev-jena-weather-1h` (jena_weather_1H) | E1 Uniform-Mean | 0.5207 | 0.4613 |
| | E2 Uniform-Shape | 0.5188 | 0.4631 |
| | E3 CRB-Mean | 0.5152 | 0.4597 |
| | E4 CRB-Shape | **0.5142** | 0.4571 |
| | E5 Target-only | 0.5394 | 0.4598 |
| | E0 Rand | 0.5330 | **0.4558** |
| `fev-ett-1h` (ETT_1H) | E1 Uniform-Mean | 1.1936 | 1.2026 |
| | E2 Uniform-Shape | 1.1658 | 1.1801 |
| | E3 CRB-Mean | 1.1749 | 1.1803 |
| | E4 CRB-Shape | **1.1594** | **1.1575** |
| | E5 Target-only | 1.1731 | 1.1616 |
| | E0 Rand | 1.1701 | 1.1849 |

（粗体为每个任务 × 模型内的最优臂。）

### 8.2 逐任务最优臂

| 任务 | TimerXL 最优臂 | TimerXL MASE | Chronos 最优臂 | Chronos MASE | 跨模型更优 |
| --- | --- | ---: | --- | ---: | --- |
| Crypto/D | E2 Uniform-Shape | 7.1397 | E2 Uniform-Shape | 6.2169 | Chronos-2-small |
| NE_China_Wind/H | E3 CRB-Mean | 0.5612 | E0 Rand | 0.5378 | Chronos-2-small |
| SG_PM25/H | E4 CRB-Shape | 0.8834 | E3 CRB-Mean | 0.8944 | TimerXL 84M |
| jena_weather_1H | E4 CRB-Shape | 0.5142 | E0 Rand | 0.4558 | Chronos-2-small |
| ETT_1H | E4 CRB-Shape | 1.1594 | E4 CRB-Shape | 1.1575 | Chronos-2-small（差距 0.0019） |

### 8.3 关键胜负计数（step 4000）

- **E1–E4（选择臂）对比 E5（Target-only）**：TimerXL 在 5/5 个任务上有选择臂优于 E5；Chronos-2-small 在 5/5 个任务上有选择臂优于 E5。
- **E1–E4 对比 E0（Rand）**：TimerXL 在 5/5 个任务上存在选择臂优于 E0（但 ETT_1H 上 E1/E3 劣于 E0）；Chronos-2-small 在 3/5 个任务上存在选择臂优于 E0（Crypto/D、SG_PM25/H、ETT_1H），在 NE_China_Wind/H 与 jena_weather_1H 上 E0 为全局最优。
- **Shape 对比 Mean（同排布配对）**：TimerXL 的 E2 优于 E1 于 4/5 任务（仅 SG_PM25/H 的 Mean 略优），E4 优于 E3 于 3/5（Crypto/D、jena、ETT_1H，均改善明显）；Chronos-2-small 的 E2 vs E1、E4 vs E3 呈任务相关，Shape 在 ETT_1H 与 jena 的 CRB 配对中更优，Mean 在 SG_PM25/H 更优。
- **CRB 对比 Uniform（同表示配对）**：TimerXL 上 E3 为 NE_China_Wind/H 最优、E4 为 SG_PM25/H / jena / ETT_1H 最优；Chronos 上 E3 为 SG_PM25/H 最优、E4 为 ETT_1H 最优。CRB 并非在所有任务上稳定优于 Uniform。

## 9. 辅助指标（step 4000）

### 9.1 TIME（CRPS / MAE）

| 任务 | 模型 | 臂 | MASE | CRPS | MAE |
| --- | --- | --- | ---: | ---: | ---: |
| Crypto/D | TimerXL | E1 | 7.5048 | 0.1166 | 1566.63 |
| Crypto/D | TimerXL | E2 | 7.1397 | 0.1078 | 1527.89 |
| Crypto/D | TimerXL | E3 | 7.4299 | 0.1150 | 1561.53 |
| Crypto/D | TimerXL | E4 | 7.1962 | 0.1093 | 1537.63 |
| Crypto/D | TimerXL | E5 | 12.3148 | 0.1726 | 3360.15 |
| Crypto/D | TimerXL | E0 | 7.5576 | 0.1145 | 1693.23 |
| Crypto/D | Chronos | E1 | 6.4575 | 0.0765 | 1453.12 |
| Crypto/D | Chronos | E2 | 6.2169 | 0.0742 | 1398.76 |
| Crypto/D | Chronos | E3 | 6.3285 | 0.0751 | 1446.48 |
| Crypto/D | Chronos | E4 | 6.3532 | 0.0750 | 1428.98 |
| Crypto/D | Chronos | E5 | 9.0780 | 0.1137 | 2061.12 |
| Crypto/D | Chronos | E0 | 6.4226 | 0.0754 | 1477.78 |
| NE_China_Wind/H | TimerXL | E1 | 0.5660 | 0.4839 | 0.24 |
| NE_China_Wind/H | TimerXL | E2 | 0.5624 | 0.4854 | 0.24 |
| NE_China_Wind/H | TimerXL | E3 | 0.5612 | 0.4843 | 0.23 |
| NE_China_Wind/H | TimerXL | E4 | 0.5631 | 0.4823 | 0.24 |
| NE_China_Wind/H | TimerXL | E5 | 0.6581 | 0.5407 | 0.27 |
| NE_China_Wind/H | TimerXL | E0 | 0.5666 | 0.4932 | 0.24 |
| NE_China_Wind/H | Chronos | E1 | 0.5428 | 0.3646 | 0.23 |
| NE_China_Wind/H | Chronos | E2 | 0.5400 | 0.3626 | 0.23 |
| NE_China_Wind/H | Chronos | E3 | 0.5394 | 0.3649 | 0.23 |
| NE_China_Wind/H | Chronos | E4 | 0.5385 | 0.3625 | 0.23 |
| NE_China_Wind/H | Chronos | E5 | 0.5791 | 0.3768 | 0.24 |
| NE_China_Wind/H | Chronos | E0 | 0.5378 | 0.3703 | 0.23 |
| SG_PM25/H | TimerXL | E1 | 0.8843 | 0.3985 | 4.67 |
| SG_PM25/H | TimerXL | E2 | 0.8847 | 0.3958 | 4.67 |
| SG_PM25/H | TimerXL | E3 | 0.8853 | 0.3995 | 4.68 |
| SG_PM25/H | TimerXL | E4 | 0.8834 | 0.3962 | 4.67 |
| SG_PM25/H | TimerXL | E5 | 0.9090 | 0.4088 | 4.80 |
| SG_PM25/H | TimerXL | E0 | 0.9099 | 0.4182 | 4.80 |
| SG_PM25/H | Chronos | E1 | 0.8993 | 0.3074 | 4.75 |
| SG_PM25/H | Chronos | E2 | 0.9033 | 0.3076 | 4.77 |
| SG_PM25/H | Chronos | E3 | 0.8944 | 0.3072 | 4.72 |
| SG_PM25/H | Chronos | E4 | 0.8968 | 0.3069 | 4.73 |
| SG_PM25/H | Chronos | E5 | 0.9168 | 0.3147 | 4.84 |
| SG_PM25/H | Chronos | E0 | 0.8997 | 0.3120 | 4.75 |

### 9.2 FEV（SQL / WQL / WAPE / MAE / MSE）

| 任务 | 模型 | 臂 | MASE | SQL | WQL | WAPE | MAE | MSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jena_weather_1H | TimerXL | E1 | 0.5207 | 0.5207 | 0.7594 | 0.7594 | 8.9046 | 1135.6576 |
| jena_weather_1H | TimerXL | E2 | 0.5188 | 0.5188 | 0.7153 | 0.7153 | 8.8403 | 1144.0579 |
| jena_weather_1H | TimerXL | E3 | 0.5152 | 0.5152 | 0.7626 | 0.7626 | 8.8916 | 1139.4626 |
| jena_weather_1H | TimerXL | E4 | 0.5142 | 0.5142 | 0.7188 | 0.7188 | 8.8539 | 1151.5370 |
| jena_weather_1H | TimerXL | E5 | 0.5394 | 0.5394 | 0.8683 | 0.8683 | 8.9073 | 1088.8574 |
| jena_weather_1H | TimerXL | E0 | 0.5330 | 0.5330 | 0.8913 | 0.8913 | 9.2419 | 1126.3035 |
| jena_weather_1H | Chronos | E1 | 0.4613 | 0.3757 | 0.2340 | 0.2631 | 7.6092 | 1183.7001 |
| jena_weather_1H | Chronos | E2 | 0.4631 | 0.3759 | 0.2369 | 0.2628 | 7.6619 | 1197.2046 |
| jena_weather_1H | Chronos | E3 | 0.4597 | 0.3750 | 0.2364 | 0.2616 | 7.6239 | 1182.8721 |
| jena_weather_1H | Chronos | E4 | 0.4571 | 0.3727 | 0.2326 | 0.2619 | 7.5889 | 1179.6709 |
| jena_weather_1H | Chronos | E5 | 0.4598 | 0.3730 | 0.2176 | 0.2689 | 7.6544 | 1177.3436 |
| jena_weather_1H | Chronos | E0 | 0.4558 | 0.3702 | 0.2894 | 0.2826 | 7.6851 | 1209.2459 |
| ETT_1H | TimerXL | E1 | 1.1936 | 1.1936 | 0.2587 | 0.2587 | 2.3390 | 15.0838 |
| ETT_1H | TimerXL | E2 | 1.1658 | 1.1658 | 0.2532 | 0.2532 | 2.2795 | 14.5824 |
| ETT_1H | TimerXL | E3 | 1.1749 | 1.1749 | 0.2566 | 0.2566 | 2.2952 | 14.7142 |
| ETT_1H | TimerXL | E4 | 1.1594 | 1.1594 | 0.2528 | 0.2528 | 2.2584 | 14.3514 |
| ETT_1H | TimerXL | E5 | 1.1731 | 1.1731 | 0.2539 | 0.2539 | 2.2985 | 14.9404 |
| ETT_1H | TimerXL | E0 | 1.1701 | 1.1701 | 0.2576 | 0.2576 | 2.2661 | 14.5626 |
| ETT_1H | Chronos | E1 | 1.2026 | 0.9536 | 0.2036 | 0.2587 | 2.3765 | 16.5760 |
| ETT_1H | Chronos | E2 | 1.1801 | 0.9387 | 0.2017 | 0.2553 | 2.3306 | 16.1418 |
| ETT_1H | Chronos | E3 | 1.1803 | 0.9422 | 0.2021 | 0.2552 | 2.3269 | 16.0179 |
| ETT_1H | Chronos | E4 | 1.1575 | 0.9219 | 0.1988 | 0.2512 | 2.2852 | 15.6548 |
| ETT_1H | Chronos | E5 | 1.1616 | 0.9113 | 0.2028 | 0.2583 | 2.2917 | 15.3300 |
| ETT_1H | Chronos | E0 | 1.1849 | 0.9301 | 0.2008 | 0.2574 | 2.3485 | 16.4579 |

## 10. 分析与结论（基于 step 4000）

1. **源选择总体有效。** 在 5/5 个任务上，TimerXL 与 Chronos-2-small 都至少有一个 E1–E4 选择臂优于 E5 Target-only；E5 在两个 TIME 任务上显著最差，在 FEV 上因目标域数据已较充足而接近最优。
2. **表示（Shape/Mean）的作用随模型与任务变化，但 FEV 上 Shape 更稳定占优。**
   - TimerXL：E2 在 4/5 任务优于 E1；E4 在 Crypto/D、jena、ETT_1H 优于 E3，其中 ETT_1H 降幅最大（1.1749 → 1.1594）。
   - Chronos-2-small：E4 在 ETT_1H 与 jena 的 CRB 配对中最优；E2 在 Crypto/D、ETT_1H 优于 E1。
3. **锚点排布（CRB/Uniform）没有一致优势。** TimerXL 的 CRB 在 NE_China_Wind/H（E3）、SG_PM25/H、jena、ETT_1H（E4）最优；Chronos 的 CRB 在 SG_PM25/H（E3）与 ETT_1H（E4）最优，但 Uniform 在 Crypto/D 最优。应表述为“CRB 与 Uniform 存在任务相关差异”，而非 CRB 普遍更优。
4. **Rand 基线在 Chronos-2-small 上异常强。** Chronos-2-small 的 E0 在 NE_China_Wind/H（0.5378）与 jena_weather_1H（0.4558）为全局最优；在 Crypto/D、SG_PM25/H、ETT_1H 上选择臂仍优于 E0。TimerXL 的 E0 在 Crypto/D、SG_PM25/H 明显较差，在 ETT_1H 上略优于部分选择臂（E1/E3）。
5. **模型对比（同一 TimerXL 选择信号、同一训练配方）。** Chronos-2-small 在 Crypto/D、NE_China_Wind/H、jena_weather_1H、ETT_1H 上取得更低 MASE；TimerXL 84M 仅在 SG_PM25/H 上更优。该比较同时受模型容量、预训练分布和原生输出形式（确定性 vs. 概率）影响，不能单独归因于选择方法。
6. **MASE 绝对值跨任务不可直接比较。** Crypto/D 的 MASE 量级约 6–12，而 NE_China_Wind/H 约 0.54–0.66；原因是各数据集目标尺度与季节性归一化不同。论文中应按任务分行报告，或使用名次/相对改进。

## 11. 可复现入口与产物路径

- 队列脚本：
  - TimerXL：`scripts/exp/main-crb-supp/run_queue.py`（正式 `--mode run`，24 任务扩展 `--mode extend`）。
  - Chronos-2-small：`scripts/exp/main-crb-supp-chronos2-small/run_queue.py`（`--mode run`）。
- 训练配置：`configs/experiments/main-crb-supp/run/<任务>/<方法>-{train,eval,pipeline}.yaml`；Chronos 同名目录在 `configs/experiments/main-crb-supp-chronos2-small/run/<任务>/`。
- 评估配置：`configs/eval/time/main-crb-supp-*.yaml`、`configs/eval/fev/main-crb-supp-*.yaml`；Chronos 对应文件名含 `main-crb-supp-chronos2-small`。
- 安全 reference：`data/ref/shard_dataset/main-crb-supp/run/<任务>/` 与 `.../main-crb-supp-chronos2-small/run/<任务>/`。
- 选择资产：`outputs/selection/main-crb-supp/run/<任务>/<方法>/resampling_payload.npz`（Chronos 同名目录在 `outputs/selection/main-crb-supp-chronos2-small/run/...`）。
- 梯度缓存：`outputs/grad_cache/main-crb-supp/run/<任务>/`。
- 训练权重：`outputs/train/<实验>/run/<任务>/<方法>/checkpoint`。
- step-4000 评估目录（本文所有数值来源）：
  - TIME：`outputs/eval/<实验>/run/<任务>/<方法>/step-4000/<数据集短名>/short/metrics.npz`
  - FEV：`outputs/eval/<实验>/run/<任务>/<方法>/step-4000/summary.csv`
- 汇总文档：`docs/exp/main-crb-supp/results.md`、`docs/exp/main-crb-supp-chronos2-small/results.md`；全局索引 `docs/EXP.md`、`docs/RESULTS.md`。
- 大文件实际位置：`/root/autodl-tmp/rage/...`（项目目录通过软链接访问）。

## 12. 引用注意事项与局限

- 两模型共享 TimerXL 探针，因此不能把 Chronos-2-small 的结果表述为“Chronos 自身梯度选择”；正确表述是“TimerXL 选择信号的跨模型迁移”。
- E0 Rand 在 Chronos-2-small 上重新训练 4000 步，未复用 TimerXL 权重；E5 在两模型上也各自重新训练。
- 训练预测长度固定为 96，而 TIME/ FEV 评估使用 benchmark 自身的预测长度（Crypto 30、NE/SG/jena 24、ETT 168）。评估协议的 context 上限为 512 点。
- 仅使用单一 seed = 0；当前结果不含随机方差估计，论文中如需显著性应补多 seed。
- 本文只覆盖五个任务；同一批次其余 19 个任务的结果见 `results.md`，不应从这五个任务外推到全部 24 个任务。

---

## 附录 A. 可直接改写的英文实验设置段落（草稿）

> We evaluate a source-window selection method for adapting time-series foundation models under a fixed training budget. The source pool contains 256 univariate trajectories with 6,382,020 legal windows (context length 480, prediction length 96, stride 1). Source anchors are placed with q = 16, yielding 399,044 anchor gradient evaluations and 398,788 contiguous blocks; all compared sampling arms use exactly this budget. Anchor placement is either Uniform or CRB (curvature-aware, cover:error:split = 1:0:1, no error stage, longest-interval midpoint bisection with min-kappa split). Each source block is represented by either a Mean or a Shape descriptor. Gradients are computed with a TimerXL-84M LoRA probe (rank 8, alpha 16, dropout 0, applied to q_proj and v_proj), projected to 4096 dimensions by Count-Sketch (seed 0), and scored by the inner product with an equal-weight entity-level reference built from q = 8 target-history anchors. Block scores are converted to sampling probabilities with the fixed V5 mapping F(x) = (1 - exp(-x/T))/(1 - exp(-1/T)), T = 1/(2 ln 4); selected blocks resample uniformly among their owned legal windows, with no length prior. We compare six arms: Uniform-Mean, Uniform-Shape, CRB-Mean, CRB-Shape, Target-only (training on all safe target-history windows), and Rand (uniform sampling over the full source pool). All models are trained for 4,000 optimizer steps with AdamW, effective batch size 32, micro batch 8, learning rate 1e-4, weight decay 0, betas (0.9, 0.999), eps 1e-8, gradient clipping 1.0, 3% warmup, cosine decay, and LoRA rank 8 / alpha 16 / dropout 0 (seed 0, BF16). The two adapted models are TimerXL 84M and Chronos-2-small (27.9M parameters, revision ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a); the selection probe is always TimerXL 84M, so Chronos-2-small results measure cross-model transfer of the TimerXL selection signal. Evaluation is rolling: TIME uses the benchmark's short-term horizon (Crypto/D 30, NE_China_Wind/H 24, SG_PM25/H 24) with context up to 512 points and nine quantile levels; FEV uses 20 evaluation windows with horizons 24 (jena_weather_1H) and 168 (ETT_1H). We report MASE as the primary metric, together with CRPS/MAE for TIME and SQL/WQL/WAPE/MAE/MSE for FEV.

## 附录 B. 五个任务的 step-4000 最优值速查

| 任务 | TimerXL 84M 最优 MASE | Chronos-2-small 最优 MASE |
| --- | ---: | ---: |
| `time-crypto-d` (Crypto/D) | 7.1397（E2 Uniform-Shape） | 6.2169（E2 Uniform-Shape） |
| `time-ne-china-wind-h` (NE_China_Wind/H) | 0.5612（E3 CRB-Mean） | 0.5378（E0 Rand） |
| `time-sg-pm25-h` (SG_PM25/H) | 0.8834（E4 CRB-Shape） | 0.8944（E3 CRB-Mean） |
| `fev-jena-weather-1h` (jena_weather_1H) | 0.5142（E4 CRB-Shape） | 0.4558（E0 Rand） |
| `fev-ett-1h` (ETT_1H) | 1.1594（E4 CRB-Shape） | 1.1575（E4 CRB-Shape） |
