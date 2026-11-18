# main-crb-supp：受控补充实验

- 创建时间：2026-11-16；最后修改：2026-11-17。
- 状态：并行正式队列运行中；首次启动时间 2026-11-16 23:44:48，完成并行验证后于 2026-11-16 23:52:58 重启；最后记录时间 2026-11-16 23:55（Asia/Shanghai）。旧 E1 未完成训练现场已归档。首任务五臂训练的 12 次 GPU 采样均为 99%。
- 目标：在相同 TimerXL 84M、source pool、reference、梯度评估预算和训练条件下，检验锚点排布（均匀/CRB）与梯度表示（Mean/Shape）的独立及交互作用，并与 Target-only、Rand 比较。
- 假设：CRB 与 Shape 可能改善迁移预测；所有结论以实际评估结果为准。本实验单独成表，不与主实验混排。

## 固定设置

- 模型：TimerXL 84M，预训练权重 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m`；训练从同一预训练权重启动，LoRA 仅作用于 q_proj/v_proj，r=8、alpha=16、dropout=0。
- Conda 环境：`rage-timerxl`；Python：`/root/autodl-tmp/rage/envs/rage-timerxl/bin/python`；seed=0；计算设备 CUDA。
- 窗口：context=480，prediction=96，stride=1。训练 4000 optimizer steps，分别在 1000、2000、4000 step 的权重上评估，effective batch=32，micro batch=8，AdamW，学习率 1e-4，weight decay=0，betas=(0.9,0.999)，eps=1e-8，max grad norm=1，warmup=3%，cosine，min_lr_ratio=0。
- Source：固定 src-256，共 256 条轨迹、6,382,020 个合法窗口。复用已完成的 q16 梯度资产，每种排布 399,044 次真实锚点梯度评估、398,788 个 block。CRB cover 使用最长区间中点二分，split 使用 min-kappa，cover:error:split=1:0:1，无 error 阶段；Uniform 在相同每轨迹预算下均匀排布。
- Probe：复用 main-ads-v0 的 TimerXL LoRA probe；Count-Sketch 投影维度 4096、seed=0；source 的 BF16 投影梯度以 FP16 方向与 FP32 模长重构，reference 使用相同 probe 和投影。
- Reference：每任务先从预测评估第一个测试窗口之前截取安全历史。TIME rolling 使用该任务完整 test_length 截断，按 item_id 只取首个 input；多变量拆成单变量序列。FEV 使用第一个 FEV 窗口的 input。只保留有限值连续段且长度至少 576。测试答案不进入 reference 或训练。Reference q8：每段至少 2 个、其余按 ceil(合法窗口数/8) 均匀锚点。
- Reference Mean：每个原始实体汇聚 q8 原始投影梯度并取均值，再对实体向量 L2 归一化，最后实体等权平均。Reference Shape：每实体对相邻 q8 锚点计算 Shape、取均值、L2 归一化，最后实体等权平均。多个有限连续段属于同一实体时合并该实体锚点；不以实体/段合法窗口数加权。
- Source Mean：block 两端原始投影梯度均值并 L2 归一化。Shape：按主分支 V7 的 `[m,(g_R-g_L)/sqrt(12)]/sqrt(||m||^2+||g_R-g_L||^2/12)`，零能量置零。
- 得分：source block 向量与 reference 等权均值向量的内积（Linear aggregation）。按 block 得分排序，使用固定 V5 映射 `F(x)=(1-exp(-x/T))/(1-exp(-1/T))`，`T=1/(2ln4)`，相同得分等分区间质量。调用 rank_mass 时对象权重全为 1；没有 source 合法窗口数先验。选中 block 后，在其原生相邻锚点区间归属的合法窗口中均匀抽取；不按 block 长度加权。相邻 block 的共享端点由右侧 block 拥有，最后一个 block 包含末端；所有合法窗口恰好属于一个 block。
- E5：直接以安全 target reference 历史的全部合法窗口训练，相同训练 steps；E0：复用相同 src-256、seed 0、4000 steps、相同优化配置的既有 Rand checkpoint，只在各任务上新评估。
- 指标：TIME 以 rolling 模式、FEV 以现有 rolling 任务配置评估；主指标为 MASE，TIME 原始 metrics.npz、FEV 原始 summary.csv 保留辅助指标。不同任务各自独立，不用 reference 的测试区间评估。

## 方法矩阵

| 臂 | 排布 | 表示 | 采样对象 | source 梯度预算 | 训练 |
| --- | --- | --- | --- | ---: | --- |
| E1 Uniform-Mean | Uniform q16 | Mean | block | 399044 | 4000 steps；评估 1k/2k/4k |
| E2 Uniform-Shape | Uniform q16 | Shape | block | 399044 | 4000 steps；评估 1k/2k/4k |
| E3 CRB-Mean | CRB q16 | Mean | block | 399044 | 4000 steps；评估 1k/2k/4k |
| E4 CRB-Shape | CRB q16 | Shape | block | 399044 | 4000 steps；评估 1k/2k/4k |
| E5 Target-only | 不适用 | 不适用 | target safe windows | 不适用 | 4000 steps；评估 1k/2k/4k |
| E0 Rand | 不适用 | 不适用 | source legal window | 不适用 | 复用同配方 4k checkpoint；仅评估 4k |

## 队列与批次

先从 11 个本地可用、未做过且初筛成本较低的 TIME 候选中以 NumPy default_rng(0) 无放回抽取 10 个，再加入两个新下载的固定版本 FEV 任务并打乱顺序。逐任务连续有限段审计发现 SG_Weather/D、Oil_Price/B 均无长度至少 576 的可用段；用 default_rng(1) 从通过审计的备用候选中无放回抽取 current_velocity/5T 与 azure2019_U/5T，保留其队列位置。FEV 数据版本 d3e7e720f4158eab03ebcb021f1cce779cd92acc；TIME 数据版本 83e3d0b3be28d11c7182bffcc1892d19b36c4da1；TIME 代码版本 b6e7ba9862acb9233c8c55b314c1777052c420ff。原八任务不在队列中。各批次实时状态和实际指标见 results.md、docs/EXP.md 与 docs/RESULTS.md。

| 顺序 | 批次/任务 | 基准 | 数据集 | 安全 reference 合法窗口数（预检） |
| ---: | --- | --- | --- | ---: |
| 1 | time-crypto-d | TIME | Crypto/D | 7976 |
| 2 | time-supply-chain-customer-d | TIME | Supply_Chain_Customer/D | 38412 |
| 3 | time-supply-chain-location-d | TIME | Supply_Chain_Location/D | 54417 |
| 4 | time-ne-china-wind-h | TIME | NE_China_Wind/H | 29876 |
| 5 | time-coastal-t-s-20t | TIME | Coastal_T_S/20T | 19629 |
| 6 | time-australia-solar-h | TIME | Australia_Solar/H | 31969 |
| 7 | time-finland-traffic-15t | TIME | Finland_Traffic/15T | 31585 |
| 8 | fev-jena-weather-1h | FEV | jena_weather_1H（index 10） | 150213 |
| 9 | fev-ett-1h | FEV | ETT_1H（index 6） | 188790 |
| 10 | time-metropt-3-5t | TIME | MetroPT-3/5T | 11082 |
| 11 | time-current-velocity-5t | TIME | current_velocity/5T | 129546 |
| 12 | time-azure2019-u-5t | TIME | azure2019_U/5T | 127077 |

每任务严格依次执行：reference 输入与 q8 梯度准备 → E1–E4 block 概率生成 → E1–E5 在同一任务内五个进程并行，各训练一次至 4000 step，保存 1000/2000 中途快照和 4000 最终权重，再分别评估三个快照；E0 仅评估既有 4000 step checkpoint → 写入结果 → 下一个任务。source 梯度资产在最前核验一次。每任务产生 16 个评估点（E1–E5 各 3 个、E0 1 个），12 任务共 192 个评估点；结果按完成顺序写入 `results.md` 与全局索引，不预填指标。失败时保留现场并停止队列。首次串行版本验证已在独立目录完成；并行方案已复验通过：先对首任务六臂分别做 1 step 训练/评估，再对 E1 连续训练 3 step 并分别评估 step 1/2/3 的快照；每轮验证结束均清理全部验证产物。并行方案已通过五臂并发与三 checkpoint 评估验证，验证产物已清理，正式队列已重启。正式任务用 tmux，启动后检查 GPU 利用率。当前前 5/12 个任务完成；time-australia-solar-h 为第 6 个任务，E1–E5 均已完成 step-1000 快照，最新约 1200/4000 step，尚无正式评估指标。

## 入口与路径

- 脚本：`scripts/exp/main-crb-supp/run_queue.py`；中途快照由通用训练入口 `scripts/tools/train.py` 的可选 evaluation_steps 参数保存，不改变其他训练调用。
- 验证：`python scripts/exp/main-crb-supp/run_queue.py --mode validate`。
- 正式：`python -u scripts/exp/main-crb-supp/run_queue.py --mode run`。
- 配置：`configs/experiments/main-crb-supp/run/<任务>/<方法>-{train,eval,pipeline}.yaml`；评估配置 `configs/eval/{time,fev}/main-crb-supp-*.yaml`。
- 数据：`data/ref/shard_dataset/main-crb-supp/run/<任务>/`；FEV 源文件 `data/bench/fev/datasets/main-crb-supp/`；TIME 使用项目 data 目录下软链接。
- 缓存：`outputs/grad_cache/main-crb-supp/run/<任务>/`；选择 `outputs/selection/main-crb-supp/run/<任务>/`；训练 `outputs/train/main-crb-supp/run/<任务>/<方法>/`；评估 `outputs/eval/main-crb-supp/run/<任务>/<方法>/`；日志 `logs/main-crb-supp/`。大文件实际位于 /root/autodl-tmp/rage 数据盘。
- 完成标记：每任务、每 arm 的 step-1000/step-2000/step-4000 评估目录内 `.complete`；总队列结束后 `outputs/eval/main-crb-supp/run/.complete`。

## 决策依据

用户已确认 TimerXL 84M、src-256、Conda rage-timerxl、seed 0、4000 steps、effective batch 32、source q16/reference q8，以及主矩阵 block 对象、trajectory 留待后续探索；已确认无长度先验、q8 reference 等权均值、最长二分 cover 与 1:0:1 预算、V5 固定排名映射。本轮用户授权至少 12 个新 FEV/TIME 任务的顺序自动队列，要求先验证并清理验证产物，再正式启动并检查显卡利用率。
