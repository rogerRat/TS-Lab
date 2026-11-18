# main-crb-supp-chronos2-small：Chronos-2-small 跨任务补充实验

- 创建时间：2026-11-17；最后修改：2026-11-17。
- 状态：TimerXL 原 12 任务队列已完成；Chronos-2-small 曾启动首任务验证和训练准备，随后按用户要求于 2026-11-17 02:20 优雅暂停，等待 TimerXL 同一实验补齐 24 任务。
- 目标：在现有 12 个补充任务基础上增加 12 个任务，对同一套 RAGE/CRB 采样流程使用 Chronos-2-small 完成训练与评估。
- 模型：autogluon/chronos-2-small，固定 revision ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a，27,934,624 参数；权重许可 Apache-2.0。服务器本地缓存已存在。
- 环境：AutoDL tsl，项目 /root/TS-Lab；Conda rage-chronos2；Python /root/autodl-tmp/rage/envs/rage-chronos2/bin/python；训练和评估使用 CUDA。
- 依赖队列原 12 任务批次完成时间：2026-11-17 02:10:30；Chronos 首次运行于 02:10:53 开始，02:20 优雅暂停；补齐 TimerXL 24 任务后再恢复。
- 最后记录状态：TimerXL-24 完成标记已生成，Chronos-2-small 已恢复正式队列；最近检查为 8/24 任务完成，`fev-ett-1h` 训练推进至约 3800–4000/4000 step；粗估剩余 9–12 小时。

## 已确认和沿用的设置

| 项目 | 设置 |
| --- | --- |
| 数据任务 | 24 个：现有 12 个 + 新增 12 个 |
| Context / horizon / stride | 480 / 96 / 1 |
| 训练 | 每方法 4000 optimizer steps；step 1000、2000、4000 保存和评估 |
| Batch | effective batch 32，micro batch 8 |
| 优化器 | AdamW；learning rate 1e-4；weight decay 0；betas=(0.9,0.999)；eps=1e-8；max grad norm=1 |
| 学习率计划 | warmup 3%；cosine；min_lr_ratio=0 |
| LoRA | rank=8，alpha=16，dropout=0；Chronos-2 模块 self_attention.q、self_attention.v |
| 随机种子 | 0 |
| 采样资产 | src-256，共 6,382,020 个合法窗口；q16 source 为 399,044 次锚点梯度评估、398,788 个 block；reference q8 |
| 评估 | TIME 与 FEV 使用 rolling evaluation；主指标 MASE；TIME 原始 metrics.npz、FEV summary.csv 保留 |
| 对照 | E0 Rand 在 Chronos-2-small 上重新训练 4000 steps，不能复用 TimerXL 权重 |
| 运行顺序 | 每任务先准备 reference 和 E1–E4 选择结果，再并行训练 E1–E5 与 E0；之后按方法评估并进入下一任务 |

以上训练步数、优化器参数、采样预算和数据处理沿用现有 main-crb-supp 配方，以便比较模型迁移表现。它们没有针对 Chronos-2-small 单独调参；Chronos-2-small 的 probe 是否也使用该模型仍待确认。

## 方法

| 臂 | 采样方法 | 表示 | 训练数据 | 训练与评估 |
| --- | --- | --- | --- | --- |
| E1 Uniform-Mean | Uniform q16 | Mean | 源 block 重采样 | 4000 steps；1k/2k/4k |
| E2 Uniform-Shape | Uniform q16 | Shape | 源 block 重采样 | 4000 steps；1k/2k/4k |
| E3 CRB-Mean | CRB q16 | Mean | 源 block 重采样 | 4000 steps；1k/2k/4k |
| E4 CRB-Shape | CRB q16 | Shape | 源 block 重采样 | 4000 steps；1k/2k/4k |
| E5 Target-only | 不适用 | 不适用 | 安全 reference 全部合法窗口 | 4000 steps；1k/2k/4k |
| E0 Rand | 固定随机 source 重采样 | 不适用 | src-256 随机窗口 | Chronos-2-small 训练 4000 steps；评估 4k |

CRB 使用最长区间中点二分、split=min-kappa、cover:error:split=1:0:1，不设 error 阶段。按 block 得分排序，使用固定 V5 排名映射 F(x)=(1-exp(-x/T))/(1-exp(-1/T))，T=1/(2ln4)；source 长度先验和 reference 长度先验均关闭。E1–E4 每个 block 的概率质量用于其归属合法窗口内的均匀抽样。Target-only 使用安全历史的全部合法窗口。

Reference 构造与现有实验保持一致：TIME 按 rolling test tail 截断安全历史；FEV 使用第一个评估窗口的 input；先按原始实体拆成单变量序列，只保留有限值连续段且长度至少 576。测试答案不进入 reference 或训练。

## 探针模型决策

Chronos-2-small 的本地 checkpoint、LoRA 模块、训练 adapter 和原生分位数输出已接入现有通用训练/评估入口。当前队列脚本暂沿用 TimerXL q16 source 梯度及 TimerXL q8 reference probe，并使用 Chronos-2-small 作为被训练和评估的模型。这样可以复用已完成的 source 梯度资产，将本批次解释为“TimerXL 选择信号迁移到 Chronos-2-small”的实验。

是否将 probe 也切换为 Chronos-2-small 已向用户确认。若改为 Chronos-2-small probe，需要重新计算 399,044 个 source anchor 梯度及全部 reference q8 梯度；队列启动前按用户答复定稿。不能把不同 probe 的得分混为同一方法。

## 任务与安全 reference

TIME revision 83e3d0b3be28d11c7182bffcc1892d19b36c4da1；TIME evaluator commit b6e7ba9862acb9233c8c55b314c1777052c420ff。FEV revision d3e7e720f4158eab03ebcb021f1cce779cd92acc。

| 顺序 | 任务 | Benchmark | 安全 reference 合法窗口数 |
| ---: | --- | --- | ---: |
| 1 | Crypto/D | TIME | 7976 |
| 2 | Supply_Chain_Customer/D | TIME | 38412 |
| 3 | Supply_Chain_Location/D | TIME | 54417 |
| 4 | NE_China_Wind/H | TIME | 29876 |
| 5 | Coastal_T_S/20T | TIME | 19629 |
| 6 | Australia_Solar/H | TIME | 31969 |
| 7 | Finland_Traffic/15T | TIME | 31585 |
| 8 | jena_weather_1H（index 10） | FEV | 150213 |
| 9 | ETT_1H（index 6） | FEV | 188790 |
| 10 | MetroPT-3/5T | TIME | 11082 |
| 11 | current_velocity/5T | TIME | 129546 |
| 12 | azure2019_U/5T | TIME | 127077 |
| 13 | Water_Quality_Darwin/15T | TIME | 555006 |
| 14 | SG_PM25/H | TIME | 44096 |
| 15 | OpenElectricity_NEM/5T | TIME | 388810 |
| 16 | EWELD_Load/15T | TIME | 186250 |
| 17 | Port_Activity/D | TIME | 235026 |
| 18 | current_velocity/10T | TIME | 1041390 |
| 19 | Coastal_T_S/5T | TIME | 3440268 |
| 20 | epf_electricity_price/H | TIME | 246605 |
| 21 | current_velocity/15T | TIME | 194646 |
| 22 | Coastal_T_S/15T | TIME | 282828 |
| 23 | azure2019_D/5T | TIME | 21326859 |
| 24 | hospital_admissions_1D（index 9） | FEV | 4766 |

新增任务通过安全历史检查后纳入。初筛未通过的候选未加入队列；本轮发现包括 SG_Carpark/15T、Global_Influenza/W、US_Term_Structure/B、Job_Claims/W、Housing_Inventory/M、Vehicle_Supply/M、Auto_Production_SF/M、JOLTS/M、US_Labor/M、Vehicle_Sales/M、Online_Retail_2_UCI/D、Smart_Manufacturing/H、Global_Price/Q、WUI_Global/Q、ECDC_COVID/W、Port_Activity/W、FEV M_DENSE_1D、us_consumption_1Y 和 world_co2_emissions。

## 执行入口与路径

- 队列脚本：scripts/exp/main-crb-supp-chronos2-small/run_queue.py。
- 配置：configs/experiments/main-crb-supp-chronos2-small/run/<任务>/。
- TIME/FEV evaluation configs：configs/eval/time/、configs/eval/fev/，文件名含 main-crb-supp-chronos2-small。
- 安全 reference：data/ref/shard_dataset/main-crb-supp-chronos2-small/run/<任务>/。
- 选择资产：outputs/selection/main-crb-supp-chronos2-small/run/<任务>/。
- q8/reference gradient cache：outputs/grad_cache/main-crb-supp-chronos2-small/run/<任务>/。
- 训练：outputs/train/main-crb-supp-chronos2-small/run/<任务>/<方法>/。
- 评估：outputs/eval/main-crb-supp-chronos2-small/run/<任务>/<方法>/step-<步数>/。
- 日志：logs/main-crb-supp-chronos2-small/；文档操作日志写入 docs/logs/。
- 每个 arm 训练目录、评估节点及队列完成后均写 .complete。
- 安全历史预检日志：/root/autodl-tmp/rage/logs/main-crb-supp-chronos2-small/preflight_*.log。

正式命令由 tmux 依赖任务启动：Chronos 依赖队列改为等待 outputs/eval/main-crb-supp-24/run/.complete；该标记仅在 TimerXL 新增 12 个任务全部完成后生成。届时 Chronos 自动重新 prepare、validate 并 run，复用已有的安全完成标记并续跑。前置 TimerXL 队列退出但没有 .complete 时停止，不启动 Chronos-2 训练。验证只用首任务短跑并清理验证输出。

## 输出与分析

每任务 16 个评估点，24 个任务共 384 个点。仅在完成标记和原始指标文件同时存在时填写指标。主指标为 MASE；不同任务按各自测试集定义计算，不跨任务直接平均。失败、中断、未完成批次保留实际状态和已生成结果。

## 探针方案定稿

- 当前默认方案：沿用 TimerXL q16 source 梯度与 TimerXL q8 reference 梯度作为选择 probe；Chronos-2-small 是全部 24 个任务的被训练和评估模型。
- 解释范围：本实验衡量既有 TimerXL CRB 选择信号迁移到 Chronos-2-small 后的训练表现；不将结果表述为 Chronos-2-small 原生梯度选择结果。
- 原因：保持本轮 source block 排布、参考信号与现有 TimerXL 结果一致，并复用已完成的 source q16 梯度资产。
- 未确认项：用户尚未明确要求重算 Chronos-2-small probe 梯度；若后续要求，需重新计算 399,044 个 q16 source anchors 及全部任务 q8 reference gradients，并单独记录为另一 probe 方案。
- 队列：Chronos 于 2026-11-17 02:20 收到 SIGINT 并按队列处理器优雅停止；已有产物保留，等待 TimerXL 24 任务批次完成标记后自动恢复。按 TimerXL 实测 12 任务耗时 2.3 小时，并考虑新增任务的安全 reference 梯度计算量，粗估 Chronos 正式批次约需 8–12 小时，预计 2026-11-17 10:30–14:30 CST 完成；此为容量估算，最大任务可能延长用时。
