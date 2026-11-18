# 实验总表

最后修改日期：2026-11-16

本文件逐行记录全部实验与批次的工作、状态和实际结果，未开始、部分完成、终止的批次同样列出。
每个实验的完整参数、方法、数据构建方式和评估口径见 `docs/exp/<实验代号>.md`；
结果汇总见 `docs/RESULTS.md`，逐实验报告见 `docs/results/<实验代号>.md`。

批次结果中引用的 `docs/results/...` 路径对应当前 `results/` 目录下的同名文件。

## 实验记录文件索引

同一个实验的多个种子/补充批次共用一份记录文件，文件内按章节区分。

| 实验代号 | 记录文件 | 批次数 |
|---|---|---:|
| `v0-crb-mech-test` | [v0-crb-mech-test.md](exp/archive/v0-crb-mech-test/exp.md) | 15 |
| `v0-crb-reconst-test` | [v0-crb-reconst-test.md](exp/archive/v0-crb-reconst-test/exp.md) | 14 |
| `v0-uni-reconst-scale` | [v0-uni-reconst-scale.md](exp/archive/v0-uni-reconst-scale/exp.md) | 9 |
| `v0-directional-marginal-gain` | [v0-directional-marginal-gain.md](exp/archive/v0-directional-marginal-gain/exp.md) | 6 |
| `v0-gradient-proxy-test` | [v0-gradient-proxy-test.md](exp/archive/v0-gradient-proxy-test/exp.md) | 6 |
| `v0-grad-mech-test` | [v0-grad-mech-test.md](exp/archive/v0-grad-mech-test/exp.md) | 5 |
| `v0-gradient-variance-decomposition` | [v0-gradient-variance-decomposition.md](exp/archive/v0-gradient-variance-decomposition/exp.md) | 5 |
| `v0-var-proxy-estimate` | [v0-var-proxy-estimate.md](exp/archive/v0-var-proxy-estimate/exp.md) | 5 |
| `rage-smoke-static-cycle` | [rage-smoke-static-cycle.md](exp/archive/rage-smoke-static-cycle/exp.md) | 4 |
| `v0-active-value-rank` | [v0-active-value-rank.md](exp/archive/v0-active-value-rank/exp.md) | 4 |
| `v0-timer-XL-grad-64-change-velocity` | [v0-timer-XL-grad-64-change-velocity.md](exp/archive/v0-timer-XL-grad-64-change-velocity/exp.md) | 4 |
| `v0-uni-data-evaluation` | [v0-uni-data-evaluation.md](exp/archive/v0-uni-data-evaluation/exp.md) | 4 |
| `v0-vage` | [v0-vage.md](exp/archive/v0-vage/exp.md) | 4 |
| `v0-crb-final-test-timerXL-64tr` | [v0-crb-final-test-timerXL-64tr.md](exp/archive/v0-crb-final-test-timerXL-64tr/exp.md) | 3 |
| `v0-grad-mech-l2-norm` | [v0-grad-mech-l2-norm.md](exp/archive/v0-grad-mech-l2-norm/exp.md) | 3 |
| `v0-grad-traj-level-norm-gap` | [v0-grad-traj-level-norm-gap.md](exp/archive/v0-grad-traj-level-norm-gap/exp.md) | 3 |
| `v0-timer-XL-grad-4-change-velocity` | [v0-timer-XL-grad-4-change-velocity.md](exp/archive/v0-timer-XL-grad-4-change-velocity/exp.md) | 3 |
| `v1-crb-redo` | [v1-crb-redo.md](exp/v1-crb-redo/exp.md) | 15 |
| `v0-proxy-dp-segmentation` | [v0-proxy-dp-segmentation.md](exp/archive/v0-proxy-dp-segmentation/exp.md) | 2 |
| `v0-timerXL-grad-4-lm-head` | [v0-timerXL-grad-4-lm-head.md](exp/archive/v0-timerXL-grad-4-lm-head/exp.md) | 2 |
| `v0-chronostiny-timerxl-lora-full-64` | [v0-chronostiny-timerxl-lora-full-64.md](exp/archive/v0-chronostiny-timerxl-lora-full-64/exp.md) | 1 |
| `v0-chronostiny-timerxl-lora-track0` | [v0-chronostiny-timerxl-lora-track0.md](exp/archive/v0-chronostiny-timerxl-lora-track0/exp.md) | 1 |
| `v0-crb-var` | [v0-crb-var.md](exp/archive/v0-crb-var/exp.md) | 1 |
| `v0-grad-prepare-src-4096tr-q32-4096d` | [v0-grad-prepare-src-96tr-q16-4096d.md](exp/archive/v0-grad-prepare-src-96tr-q16-4096d/exp.md) | 1 |
| `v0-grad-prepare-src-96tr-q16-4096d` | [v0-grad-prepare-src-96tr-q16-4096d.md](exp/archive/v0-grad-prepare-src-96tr-q16-4096d/exp.md) | 1 |
| `v0-gradient-proxy-cost` | [v0-gradient-proxy-cost.md](exp/archive/v0-gradient-proxy-cost/exp.md) | 1 |
| `v0-timer-XL-grad-4-curvature` | [v0-timer-XL-grad-4-curvature.md](exp/archive/v0-timer-XL-grad-4-curvature/exp.md) | 1 |
| `v0-timerxl-src96-first4-full-grad-4096d-bf16` | [v0-timerxl-src96-first4-full-grad-4096d-bf16.md](exp/archive/v0-timerxl-src96-first4-full-grad-4096d-bf16/exp.md) | 1 |
| `v1-computational-effort-estimate` | [v1-computational-effort-estimate.md](exp/archive/v1-computational-effort-estimate/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-full` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-full-seed1` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-full-seed2` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-full-seed3` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-rand` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-rand-seed1` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-rand-seed2` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-chronosT5-trev-base-rand-seed3` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-crb-vs-uniform-timerXL-84m` | [v1-rage-crb-vs-uniform-timerXL-84m.md](exp/archive/v1-rage-crb-vs-uniform-timerXL-84m/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-full` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-full-seed1` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-full-seed2` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-full-seed3` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-rand` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-rand-seed1` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-rand-seed2` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timerXL-trev-84m-rand-seed3` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-full` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-full-seed1` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-full-seed2` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-full-seed3` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-rand` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-rand-seed1` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-rand-seed2` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-timesFM-trev-2.5-rand-seed3` | [v1-rage-trev.md](exp/archive/v1-rage-trev/exp.md) | 1 |
| `v1-rage-ads` | [v1-rage-ads.md](exp/v1-rage-ads/exp.md) | 3 |
| `v1-rage-ads-v2` | [v1-rage-ads-v2.md](exp/v1-rage-ads-v2/exp.md) | 2 |
| `v1-timerxl-step1000-gradient-shift` | [v1-timerxl-step1000-gradient-shift.md](exp/v1-timerxl-step1000-gradient-shift/exp.md) | 1 |

## 批次总表

状态以完成标记、日志和实际产物核验为准。

| 实验代号 | 批次 | 工作 | 状态 | 实际结果 |
|---|---|---|---|---|
| v1-rage-ads | ads-api-q-single | ADS 单一 q/eta 接口、max 邻域概率描述与抽样函数验证 | 完成：25 项 ADS 定向测试通过，完成标记已生成 | 无模型指标；[实验记录](exp/v1-rage-ads/exp.md) / [结果](exp/v1-rage-ads/results.md) |
| v1-rage-ads | rolling-parallelism-profile | 梯度、审计、训练、rolling评估并发试跑及资源测量 | 完成：正式规模审计、32/128-step训练及评估并发扫描完成 | 测得正式审计选8 workers、训练/评估选12 workers；未产出训练效果指标；profile JSON及详细资源见[结果](exp/v1-rage-ads/results.md) |
| v1-rage-ads | rolling-eight-task | 八任务 rolling 安全历史、ADS 选择、梯度与零概率支持审计、训练评估及任务级汇总 | 完成：32/32训练评估单元和零概率支持审计完成 | rage平均名次1.625；p=0窗口比例rage/ancUni/Traj=59.10%/78.96%/53.82%；八任务平均MASE=1.4788/1.4637/1.4744/1.4905；[实验记录](exp/v1-rage-ads/exp.md) / [结果](exp/v1-rage-ads/results.md) |
| v1-rage-ads-v2 | baseline-design-v2 | 必含src-96的src-256源池及ADS-VIC/基线方法定义 | 设计冻结 | 仅方法定义，无指标；[实验记录](exp/v1-rage-ads-v2/exp.md) / [结果](exp/v1-rage-ads-v2/results.md) |
| v1-rage-ads-v2 | ads-rev-reverse-complement | 八任务 ads-rev：q0=32，排除 cos>=0.1 锚点，在未命中区间按 q1=16 均匀加密；无 uni | 选择8/8、训练8/8、评估8/8均完成（2026-11-15 09:50 CST） | 选择配对98,904,745/1,761,504,746（5.615%）；source锚点命中59.60%、reference锚点100%；MASE：FEV09 0.733576；TIME01 0.720485；TIME02 1.274494；TIME03 1.250550；TIME04 4.374898；TIME06 1.095141；TIME07 1.563999；TIME08 1.144074；均值1.519652；MASE 指标见 outputs/eval/v1-rage-ads-v2/<task>/ads-rev/；ads-rev q0/q1 直方图见 outputs/selection/v1-rage-ads-v2/ads-rev/cosine-histograms.png；ADS-VIC 当前任务最大余弦阶段图见 outputs/selection/v1-rage-ads-v2/ads-vic/anchor-max-cosine/current-fev-09-time-04/saved-anchor-max-histograms.svg；[实验记录](exp/v1-rage-ads-v2/exp.md) / [结果](exp/v1-rage-ads-v2/results.md) |
| v1-rage-ads-v2 | rolling-baseline-v2 | src-256 上 ADS-VIC、uni、traj、orig、reft 八任务选择及训练评估 | 已完成：选择8/8、训练评估40/40、概率支持审计40/40 | 各方法零概率支持比例见实验结果文档；模型原始指标位于 outputs/eval/v1-rage-ads-v2/；[实验记录](exp/v1-rage-ads-v2/exp.md) / [结果](exp/v1-rage-ads-v2/results.md) |
| v1-crb-redo | grad-prepare | src-8192热身三模型1000步；src-96按q=32准备CRB 1:0:1和同预算Uniform梯度 | 完成：三模型热身与梯度标记、整批完成标记均已生成 | TimerXL、TimesFM、Chronos 各方法均完成；详见梯度输出 |
| v1-crb-redo | 12tasks-ref-grad-prepare | 对 4 FEV 与 8 TIME 目标任务的安全历史准备 TimerXL CRB 1:0:1 与等预算 Uniform 参考梯度；后续匹配选择批次的前置依赖 | 完成：8 项安全历史及两方法梯度均核验；4 项因无法支持筛选而放弃，后续不再使用 | 8 项 CRB/Uniform 各合计 9079 个名义锚点，逐项见实验文档 |
| v1-crb-redo | timerXL-base-uni-01-traj-rr | 双侧 Uniform 轨迹代表，RR 无放回选择全局前 10% 源窗口；8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.601606；选样 55分42秒；索引文件每任务 2.03 MiB，详见结果报告 |
| v1-crb-redo | timerXL-base-uni-01-block-rr | 双侧 Uniform 块代表，RR 无放回选择全局前 10% 源窗口；8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.527675；选样 53分30秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-ours-rage101-01-traj-rr | 双侧 CRB 轨迹代表，RR 无放回选择全局前 10% 源窗口；8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.590119；选样 56分12秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-ours-rage101-01-block-rr | 双侧 CRB 块代表，RR 无放回选择全局前 10% 源窗口；8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.516385；选样 59分30秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-base-uni-01-win-rr | 双侧 Uniform 窗口 RR；BF16 完整评分矩阵；8 项原计划中的 win 评测 | 部分完成：FEV 9、TIME 1–4 共 5 项选样、训练评估完成；TIME 6 因 259,085 个 ref 窗口对应约 1.26 TiB 矩阵、超过当时约 381 GiB 可用空间而跳过；TIME 7–8 按用户限定范围跳过 | 五项 MASE 为 0.670380、1.208172、1.025688、1.549088、3.120710；TIME 7/8 矩阵估算 7.83/0.30 GiB，按范围跳过；详见实验结果文档 |
| v1-crb-redo | timerXL-ours-rage101-01-win-rr | 双侧 CRB 窗口 RR；BF16 完整评分矩阵；仅 FEV 9、TIME 1–4 | 完成：FEV 9、TIME 1–4 五项均完成选样、4,000 步训练和评估，批次标记已生成 | 五项 MASE：0.675108、1.213646、1.134415、1.455521、3.167945；TIME 6 因约 1.26 TiB 矩阵超出空间跳过，TIME 7–8 按用户范围跳过；详见实验结果文档 |
| v1-crb-redo | sgwdev-metrics | 8 任务的全源方向 cos、安全历史—测试梯度对齐和 top10% 筛选后方向接近度 | 完成：16/16 任务-方法组合指标已记录，批次 .complete 已生成 | 全源方向 cos 范围 -0.00766274–0.00379170；安全历史—真实测试方向 cos 范围 -0.04071205–0.02130944；top10% 筛选后 cos 范围 -0.00254030–0.04166267；详细数值见 results.md。
| v1-crb-redo | timerXL-ours-rage101-01-block-max-topk | ref 评分 max 聚合的 TopK 方案 | 本轮不执行 | 无 |
| v1-crb-redo | timerXL-ours-rage101-01-block-linear-topk | ref 评分 linear 聚合的 TopK 方案 | 本轮不执行 | 无 |
| v1-crb-redo | timerXL-ours-rage101-01-block-zeroclip-topk | ref 评分 zeroclip 聚合的 TopK 方案 | 本轮不执行 | 无 |
| v1-crb-redo | timerXL-ours-rage101-01-block-exp-topk | ref 评分 exp 聚合的 TopK 方案 | 本轮不执行 | 无 |
| v1-crb-redo | timerXL-baseline-full-sgwdev | src-96全量合法窗口；预训练权重起训4,000步；12任务评估 | 完成：4,000步训练及FEV 4 + TIME 8评估，批次标记已生成 | 12项中10项MASE低于rand01；详见docs/exp/v1-crb-redo/results.md |
| v1-crb-redo | timerXL-baseline-rand01-sgwdev | src-96固定10%随机窗口；预训练权重起训4,000步；12任务评估 | 完成：4,000步训练及FEV 4 + TIME 8评估，批次标记已生成 | 12项中2项MASE低于full；详见docs/exp/v1-crb-redo/results.md |
| v0-timerxl-src96-first4-full-grad-4096d-bf16 | 前四条完整轨迹 | 逐窗口计算 TimerXL LoRA 梯度并投影 4096 维、BF16 保存 | 完成 | 484,907 个合法窗口；八个 worker 与批次完成标记已核验 |
| v0-chronostiny-timerxl-lora-track0 | ordinal 0全量梯度与相关性 | 原始ChronosTiny LoRA 4096D投影，完整230,161窗口；同轨TimerXL关系比较 | 完成 | 非临近窗口对余弦Spearman 0.076；相邻窗口余弦Spearman −0.138；只代表单轨 |
| v0-chronostiny-timerxl-lora-full-64 | 64轨全窗口梯度与相关性 | 使用原始ChronosTiny权重计算前64轨全部合法窗口的LoRA 4096D BF16投影；与同窗口TimerXL比较梯度关系 | 用户调整后停止，保留部分缓存 | 8个worker曾启动；未完成全轨，结果不用于分析 |
| v0-timer-XL-grad-64-change-velocity | 梯度准备 | src-96前64轨逐窗口准备TimerXL LoRA与lm_heads的4096D BF16投影梯度；复用前四轨LoRA | 完成 | 64轨两类4096D BF16投影梯度已核验；前四轨LoRA复用；共1,980,371窗口 |
| v0-crb-final-test-timerXL-64tr | CRB-1:0:1 | 按平均间距q作64轨BF16 LoRA梯度同预算CRB/Uniform方向代表误差比较；K=max(2,ceil(N/q)) | 完成 | 64轨、9个q、5,053,578块/臂，结果见RESULTS |
| v0-crb-final-test-timerXL-64tr | CRB-1:1:0 | 同预算coverage:error CRB/Uniform，q为平均锚点间距 | 完成 | 64轨×9个q；Uniform在全部q的窗口加权误差更低 |
| v0-crb-final-test-timerXL-64tr | CRB-1:0:0 | 同预算coverage-only CRB/Uniform，q为平均锚点间距 | 完成 | 64轨×9个q；CRB仅q=64的窗口加权误差更低 |
| v0-timer-XL-grad-64-change-velocity | 自动分析 | 轨迹内与轨迹间速度差异、头梯度代理、两段均匀16窗口代理 | 完成 | 轨迹内/间差异、头代理与两段16窗口估计均完成；详见独立报告 |
| v0-timer-XL-grad-64-change-velocity | per-track-pca | 64轨真实4096维LoRA投影的逐轨PC1解释率与方向差异 | 完成 | PC1解释率中位数12.1%；PC1两两夹角中位数87.3°；rank80均值174、中位数137；v/L均值4.07%、中位数1.63%，L为每轨完整合法起点缓存行数；稳定性和限制见docs/exp/archive/v0-timer-XL-grad-64-change-velocity/v0-timerxl-grad64-pca.md |
| v0-timer-XL-grad-64-change-velocity | two-head-16-blocks | 每轨两段均匀16窗口的lm-head梯度代理，评估LoRA数值与排序恢复 | 完成 | 排序rho 0.868/0.829/0.294，前16重合11/11/5 |
| v0-timerXL-grad-4-lm-head | 四轨同窗口输出头梯度及代表性 | 每轨均匀取128个窗口，比较非临近窗口对的输出头与LoRA方向相似性 | 完成 | 三条可评估轨迹 Spearman 0.249–0.303；详见独立报告 |
| v0-timerXL-grad-4-lm-head | change-velocity-comparison | 在匹配连续窗口段上比较输出头与LoRA三项滞后曲线 | 完成 | 4761个头梯度窗口；模长变化系统性低估，见独立报告 |
| v0-timer-XL-grad-4-curvature | 四轨完整梯度曲率 | 对全部LoRA梯度作二阶时间差分，分析方向与幅度稳定性 | 完成 | 四轨中心化二阶差RMS/原RMS约1；详见独立报告 |
| v0-timer-XL-grad-4-change-velocity | 四轨LoRA梯度随时间间隔变化 | 对k=1..512逐点计算方向余弦、相对L2差和相对模长差，并分轨迹绘图 | 完成 | 2048行滞后指标、四张三面板图；详见独立报告 |
| v0-timer-XL-grad-4-change-velocity | stride-16-rank | 每16窗口采一个梯度的旧方案 | 用户撤回；不作为现行结论 | 已生成的旧产物保留作历史，不用于当前分析 |
| v0-timer-XL-grad-4-change-velocity | two-random-16-blocks | 每轨随机两个连续16窗口区间，估计k=1..15的变化速度与排序 | 完成 | 三项均把轨迹0和3顺序排反；详见独立报告 |
| v0-crb-mech-test | 64-smooth-anchor-reconstruction | 原始 64 条平滑 mock；CRB/Uniform 线性重建，q={4,8,16,32} | 完成 | 512 条方法级记录；Uniform 平均误差更低 |
| v0-crb-mech-test | 64-smooth-uniform-reconstructors | 同一数据和 Uniform anchors；比较多种重建器 | 完成 | 2880 条方法-轨迹记录；细节见结果报告 |
| v0-crb-mech-test | 64-smooth-ces-pilot | 均匀 pilot + 曲率自适应 CES | 完成 | 768 条重建记录；CES 误差接近 Uniform 并低于 CRB |
| v0-crb-mech-test | 64-smooth-ces-oracle | 使用生成器解析曲率的 CES-Oracle | 完成 | 1024 条四方法指标记录；Oracle 未优于 Uniform 或估计版 CES |
| v0-crb-mech-test | 64-smooth-block-internal-variance | 原 64 条轨迹，块内方差诊断 | 完成 | 向量、相对向量、方向、模长方差；详见结果报告 |
| v0-crb-mech-test | 64-smooth-ces-block-variance | CES、Uniform、CRB 的块内方差比较 | 完成 | 详见 results/v0-crb-mech-test.md |
| v0-crb-mech-test | 64-smooth-ces-oracle-block-variance | CES-Oracle 块内方差比较 | 完成 | 详见 results/v0-crb-mech-test.md |
| v0-crb-mech-test | ultra-smooth-single-reconstruction | 单条极平缓轨迹；q={4,8,16,32} | 完成 | Uniform 的 L2 较低；单轨迹只作补充 |
| v0-crb-mech-test | ultra-smooth-block-variance | 极平缓轨迹块内方差诊断 | 完成 | 详见 results/v0-crb-mech-test.md |
| v0-crb-mech-test | 64-smooth-cancellation-split-diagnostic | cancellation 阈值、切分位置与 real-real oracle 诊断 | 完成 | real-real 只作为 oracle 描述，不作估计能力结论 |
| v0-crb-mech-test | 64-smooth-cancellation-six-methods-real-real | 六种分块方式和 matched-K 随机切分；real-real 诊断 | 完成 | 7,680 条轨迹级、107,520 条区间级记录 |
| v0-crb-mech-test | 64-smooth-cancellation-six-methods-est-real | 同一切分，线性重建块均值作代表，真实值评估 | 完成 | 7,680 条轨迹级、130,912 条块级记录 |
| v0-crb-mech-test | 64-smooth-virtual-split-selection-k-error | Random/Cancellation/Endpoint/Predicted/Oracle；K=0..q-1 | 完成 | 64 条轨迹、8 张主图、100 次 Random 重复、5000 次 paired bootstrap |
| v0-crb-mech-test | 64-smooth-virtual-split-mse-decomposition | Representation/Estimation MSE 分解和恒等式验证 | 完成 | 最大恒等残差 8.99e-15；12 张 MSE 图 |
| v0-crb-mech-test | 64-large-var-k-error-composition | 新建 64 条 4096D×4000 mock；相邻点完整向量差比 0.50005，lag-16 余弦均值 0.0000375；q={4,8,16,32,64,128} | 完成 | 12 张 K-error 图、18 张 MSE 分解图；MSE 恒等式残差 7.55e-15；详见批次报告 |
| v0-grad-prepare-src-96tr-q16-4096d | src-96；三模型；仅 Uniform | 96 条轨迹、35 配置，q=16、4096 维 BF16 LoRA 梯度 | TimerXL、TimesFM 的 Uniform 已完成；Chronos 未完成，当前无运行进程，部分缓存保留 | 每模型 Uniform 预算 166,659 锚点；CRB 产物已删除 |
| v0-grad-prepare-src-4096tr-q32-4096d | 旧批次 | src-4096、q=32 的梯度准备 | 失败并终止：计算负担过大 | 结果及梯度缓存已删除；仅保留本失败记录 |
| v1-computational-effort-estimate | 9 个计时批次；目标清单及正式实验 | 三种模型的计算与存储估算 | 计时完成；正式实验未开始 | 旧候选池估算见[独立报告](exp/archive/v1-computational-effort-estimate/results.md) |
| v1-rage-crb-vs-uniform-timerXL-84m | single-task optimization timing | optimize TimerXL LoRA gradient computation and measure one task | Deprecated | single worker 575-597 points/s; 8-worker same-GPU pilot 1,159 points/s aggregate (~2.0x); source gradients tentatively ~2.70 days with 8 workers; source selection invalid; retained for reference only |
| v0-crb-var | 1-ten-trajectory-replay | 按原定批次执行 v0-crb-var 的实验 | 完成 | 10/10、0.6410、0.6802、0.4051、0.4402；详情见历史原文与结果文档 |
| v0-proxy-dp-segmentation | longest4-unweighted | 按原定批次执行 v0-proxy-dp-segmentation 的实验 | 完成 | 4/4、0.5031、0.6762、0.2995、0.4298；详情见历史原文与结果文档 |
| v0-proxy-dp-segmentation | longest4-window-weighted | 按原定批次执行 v0-proxy-dp-segmentation 的实验 | 完成 | 4/4、0.5717、0.6762、0.3572、0.4298；详情见历史原文与结果文档 |
| v1-rage-chronosT5-trev-base-rand | 1-random-10pct | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | 已核实 .complete；详细指标见评估产物; source selection invalid; retained for reference only |
| v1-rage-chronosT5-trev-base-full | 1-full-pool | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | 已核实 .complete；详细指标见评估产物; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-rand | 1-random-10pct | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | 已核实 .complete；详细指标见评估产物; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-full | 1-full-pool | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | 已核实 .complete；详细指标见评估产物; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-rand | 1-random-10pct | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | 已核实 .complete；详细指标见评估产物; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-full | 1-full-pool | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | 4,000 updates; all eight FEV/TIME checkpoints; .complete verified; source selection invalid; retained for reference only |
| v0-active-value-rank | 1-expanded-cross-rank | 按原定批次执行 v0-active-value-rank 的实验 | 完成 | 100、714、766、0.7104；详情见历史原文与结果文档 |
| v0-active-value-rank | 2-active-correction | 按原定批次执行 v0-active-value-rank 的实验 | 完成 | 10%、64,、0.04192、0.05148；详情见历史原文与结果文档 |
| v0-active-value-rank | 3-proxy-coverage | 按原定批次执行 v0-active-value-rank 的实验 | 完成 | 2、3、0.4023/0、6480/0、7950、65、5；详情见历史原文与结果文档 |
| v0-active-value-rank | 4-equal-budget-query | 按原定批次执行 v0-active-value-rank 的实验 | 完成 | 64,、0.5064、0.4776、0.4295、0.5415；详情见历史原文与结果文档 |
| v0-directional-marginal-gain | 1 | 按原定批次执行 v0-directional-marginal-gain 的实验 | 完成 | 46；详情见历史原文与结果文档 |
| v0-directional-marginal-gain | n=2 | 按原定批次执行 v0-directional-marginal-gain 的实验 | 完成 | 46、0.1072；详情见历史原文与结果文档 |
| v0-directional-marginal-gain | n=4 | 按原定批次执行 v0-directional-marginal-gain 的实验 | 完成 | 46、0.0407；详情见历史原文与结果文档 |
| v0-directional-marginal-gain | n=8 | 按原定批次执行 v0-directional-marginal-gain 的实验 | 完成 | 46、0.0144；详情见历史原文与结果文档 |
| v0-directional-marginal-gain | n=16 | 按原定批次执行 v0-directional-marginal-gain 的实验 | 完成 | 46、0.0075；详情见历史原文与结果文档 |
| v0-directional-marginal-gain | 6 | 按原定批次执行 v0-directional-marginal-gain 的实验 | 完成 | 64、160；详情见历史原文与结果文档 |
| v0-gradient-variance-decomposition | 1 | 按原定批次执行 v0-gradient-variance-decomposition 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-gradient-variance-decomposition | 2 | 按原定批次执行 v0-gradient-variance-decomposition 的实验 | 完成 | 46；详情见历史原文与结果文档 |
| v0-gradient-variance-decomposition | 3 | 按原定批次执行 v0-gradient-variance-decomposition 的实验 | 完成 | 30、252；详情见历史原文与结果文档 |
| v0-gradient-variance-decomposition | 4 | 按原定批次执行 v0-gradient-variance-decomposition 的实验 | 完成 | 0.431；详情见历史原文与结果文档 |
| v0-gradient-variance-decomposition | 5 | 按原定批次执行 v0-gradient-variance-decomposition 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-grad-traj-level-norm-gap | 1 | 按原定批次执行 v0-grad-traj-level-norm-gap 的实验 | 完成 | 12,836；详情见历史原文与结果文档 |
| v0-grad-traj-level-norm-gap | 2 | 按原定批次执行 v0-grad-traj-level-norm-gap 的实验 | 完成 | 1.9345、2.4960；详情见历史原文与结果文档 |
| v0-grad-traj-level-norm-gap | 3 | 按原定批次执行 v0-grad-traj-level-norm-gap 的实验 | 完成 | 0.8788；详情见历史原文与结果文档 |
| v0-var-proxy-estimate | 1 | 按原定批次执行 v0-var-proxy-estimate 的实验 | 完成 | 46、256；详情见历史原文与结果文档 |
| v0-var-proxy-estimate | 2 | 按原定批次执行 v0-var-proxy-estimate 的实验 | 完成 | 46；详情见历史原文与结果文档 |
| v0-var-proxy-estimate | 3 | 按原定批次执行 v0-var-proxy-estimate 的实验 | 完成 | 2、0.8538、0.3977、0.4970；详情见历史原文与结果文档 |
| v0-var-proxy-estimate | 4 | 按原定批次执行 v0-var-proxy-estimate 的实验 | 完成 | 2、0.9463、0.3240、0.6381；详情见历史原文与结果文档 |
| v0-var-proxy-estimate | 5 | 按原定批次执行 v0-var-proxy-estimate 的实验 | 完成 | 2、0.9873、0.2670、0.8078；详情见历史原文与结果文档 |
| v0-uni-data-evaluation | 1 | 按原定批次执行 v0-uni-data-evaluation 的实验 | 完成 | 100；详情见历史原文与结果文档 |
| v0-uni-data-evaluation | 2 | 按原定批次执行 v0-uni-data-evaluation 的实验 | 完成 | 12,736、1,273,600；详情见历史原文与结果文档 |
| v0-uni-data-evaluation | 3 | 按原定批次执行 v0-uni-data-evaluation 的实验 | 完成 | 11,500、1、0.4856/0、5372；详情见历史原文与结果文档 |
| v0-uni-data-evaluation | 4 | 按原定批次执行 v0-uni-data-evaluation 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=1 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 12,836、2、0.0000、1.0000；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=2 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 6,420、2、0.9650、0.8952；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=4 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 3,213、2、1.6969、0.7941；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=8 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 1,609、2、2.2654、0.6956；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=16 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 808、2、2.7382、0.6002；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=32 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 406、2、3.1250、0.5063；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=64 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 204、2、3.4824、0.3971；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | s=128 | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 103、2、3.8809、0.2563；详情见历史原文与结果文档 |
| v0-uni-reconst-scale | pair-mean supplement | 按原定批次执行 v0-uni-reconst-scale 的实验 | 完成 | 64、2、3.5445、3.4824、0.3498、0.3971；详情见历史原文与结果文档 |
| v0-vage | Static n_min=2 | 按原定批次执行 v0-vage 的实验 | 完成 | 2、3.5437、3.4824、0.3651、0.3971；详情见历史原文与结果文档 |
| v0-vage | Static n_min=1 | 按原定批次执行 v0-vage 的实验 | 完成 | 2、3.8747、0.3156；详情见历史原文与结果文档 |
| v0-vage | n_min=1 method comparison | 按原定批次执行 v0-vage 的实验 | 完成 | 2、3.5031、3.4824；详情见历史原文与结果文档 |
| v0-vage | pair-mean supplement | 按原定批次执行 v0-vage 的实验 | 完成 | 2、3.5445、2、2、3.5880；详情见历史原文与结果文档 |
| v0-gradient-proxy-test | Local-Risk core | 按原定批次执行 v0-gradient-proxy-test 的实验 | 完成 | 0.9476；详情见历史原文与结果文档 |
| v0-gradient-proxy-test | Local-Risk sparse 32 | 按原定批次执行 v0-gradient-proxy-test 的实验 | 完成 | 0.867、4；详情见历史原文与结果文档 |
| v0-gradient-proxy-test | Local-Risk sparse 64 | 按原定批次执行 v0-gradient-proxy-test 的实验 | 完成 | 0.837、4；详情见历史原文与结果文档 |
| v0-gradient-proxy-test | Local-Risk long blocks | 按原定批次执行 v0-gradient-proxy-test 的实验 | 完成 | 0.809、8/128、0.867、16/256、576；详情见历史原文与结果文档 |
| v0-gradient-proxy-test | Local-Risk target spread | 按原定批次执行 v0-gradient-proxy-test 的实验 | 完成 | 90、10、7.44、5.25、4.42、3.88、32,、64,；详情见历史原文与结果文档 |
| v0-gradient-proxy-test | Cross-Rank | 按原定批次执行 v0-gradient-proxy-test 的实验 | 完成 | 0.640、5、0.487；详情见历史原文与结果文档 |
| v0-gradient-proxy-cost | Direct cost benchmark | 按原定批次执行 v0-gradient-proxy-cost 的实验 | 完成 | 42.80、14.21；详情见历史原文与结果文档 |
| v0-grad-mech-l2-norm | 1 | 按原定批次执行 v0-grad-mech-l2-norm 的实验 | 完成 | 30；详情见历史原文与结果文档 |
| v0-grad-mech-l2-norm | 2 | 按原定批次执行 v0-grad-mech-l2-norm 的实验 | 完成 | 5,750、1、0.6372；详情见历史原文与结果文档 |
| v0-grad-mech-l2-norm | 3 | 按原定批次执行 v0-grad-mech-l2-norm 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | 1 | 按原定批次执行 v0-crb-reconst-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | 2 | 按原定批次执行 v0-crb-reconst-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | 3 | 按原定批次执行 v0-crb-reconst-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 00 | 按原定批次执行 v0-crb-reconst-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 01 | 按原定批次执行 v0-crb-reconst-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 02 | 按原定批次执行 v0-crb-reconst-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 03 | 按原定批次执行 v0-crb-reconst-test 的实验 | 部分完成后终止 | 1、4、13/16；详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 04 | 按原定批次执行 v0-crb-reconst-test 的实验 | 部分完成后终止 | 1；详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 05 | 按原定批次执行 v0-crb-reconst-test 的实验 | 未开始；用户已终止 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 06 | 按原定批次执行 v0-crb-reconst-test 的实验 | 未开始；用户已终止 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 07 | 按原定批次执行 v0-crb-reconst-test 的实验 | 未开始；用户已终止 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 08 | 按原定批次执行 v0-crb-reconst-test 的实验 | 未开始；用户已终止 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | trajectory 09 | 按原定批次执行 v0-crb-reconst-test 的实验 | 未开始；用户已终止 | 详情见历史原文与结果文档 |
| v0-crb-reconst-test | extreme-track extension | 按原定批次执行 v0-crb-reconst-test 的实验 | 未开始；用户已终止 | 详情见历史原文与结果文档 |
| v0-grad-mech-test | 1 | 按原定批次执行 v0-grad-mech-test 的实验 | 完成 | 8,394,496；详情见历史原文与结果文档 |
| v0-grad-mech-test | 2 | 按原定批次执行 v0-grad-mech-test 的实验 | 完成 | 30,490、1,000；详情见历史原文与结果文档 |
| v0-grad-mech-test | 3 | 按原定批次执行 v0-grad-mech-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| v0-grad-mech-test | 4 | 按原定批次执行 v0-grad-mech-test 的实验 | 完成 | 1,000/1、000、16；详情见历史原文与结果文档 |
| v0-grad-mech-test | 5 | 按原定批次执行 v0-grad-mech-test 的实验 | 完成 | 详情见历史原文与结果文档 |
| rage-smoke-static-cycle | 1 | 按原定批次执行 rage-smoke-static-cycle 的实验 | 完成 | 详情见历史原文与结果文档 |
| rage-smoke-static-cycle | 2 | 按原定批次执行 rage-smoke-static-cycle 的实验 | 完成 | 详情见历史原文与结果文档 |
| rage-smoke-static-cycle | 3 | 按原定批次执行 rage-smoke-static-cycle 的实验 | 完成 | 784、3；详情见历史原文与结果文档 |
| rage-smoke-static-cycle | 4 | 按原定批次执行 rage-smoke-static-cycle 的实验 | 完成 | 详情见历史原文与结果文档 |
| v1-rage-chronosT5-trev-base-rand-seed1 | 1-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Stopped before step 4000; latest logged step 2445; partial outputs removed; source selection invalid; retained for reference only |
| v1-rage-chronosT5-trev-base-full-seed1 | 1-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-rand-seed1 | 1-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-full-seed1 | 1-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-rand-seed1 | 1-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-full-seed1 | 1-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-chronosT5-trev-base-rand-seed2 | 2-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-chronosT5-trev-base-full-seed2 | 2-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-rand-seed2 | 2-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-full-seed2 | 2-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-rand-seed2 | 2-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-full-seed2 | 2-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-chronosT5-trev-base-rand-seed3 | 3-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-chronosT5-trev-base-full-seed3 | 3-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-rand-seed3 | 3-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timerXL-trev-84m-full-seed3 | 3-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-rand-seed3 | 3-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |
| v1-rage-timesFM-trev-2.5-full-seed3 | 3-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Deprecated | Not trained; prepared inputs and configs removed; source selection invalid; retained for reference only |

| v1-timerxl-step1000-gradient-shift | paired-src96-step1000 | 原始与 step-1000 TimerXL 的配对 LoRA 参数梯度比较 | 完成：6,144 窗口、96 轨迹；梯度计算、bootstrap 与完成标记核验通过 | 方向差 0.5653 [0.5256, 0.6054]；平均余弦 0.4347；详见 [实验记录](exp/v1-timerxl-step1000-gradient-shift/exp.md) / [结果](exp/v1-timerxl-step1000-gradient-shift/results.md) |
| `main-ads-v0` | [main-ads-v0/exp.md](exp/main-ads-v0/exp.md) | 1 |
| `main-crb-v0` | [main-crb-v0/exp.md](exp/main-crb-v0/exp.md) | 6 |
| `main-reconst-v0` | [main-reconst-v0/exp.md](exp/main-reconst-v0/exp.md) | 1 |
| main-ads-v0 | traj-replay | src-256 TimerXL LoRA 训练 2,000 步；每 10 步记录 loss；ordinal 0、1 全窗口梯度投影缓存 | traj-replay 训练与两条轨迹梯度缓存已完成；继承八个下游任务，仅启用 TIME1 | 无训练指标或梯度文件；Replay 内容待定；[实验记录](exp/main-ads-v0/exp.md) / [结果](exp/main-ads-v0/results.md) |
| main-ads-v0 | ads-histgram | ordinal 1 为 ref、ordinal 0 为 src；七种 q 下 uni、全矩阵最高25%配对 ADS-VIC、预算匹配 uni | 完成；旧 cos>0.1 结果作废；BF16计算/uint8 256桶结果及 .complete 已核验 | 100/256桶统计、预算匹配矩阵和图见 outputs/selection/main-ads-v0/ads-histgram-bf16-u8/；q级结果见 [实验文档](exp/main-ads-v0/exp.md) / [实验结果](exp/main-ads-v0/results.md)；top-25%时序矩阵及边际激活图21张已生成；另有 uni 粗锚点 top-1/16 激活图7张及正值 top-1/32 激活图7张 |
| main-ads-v0 | ads-budgeted-vein | 均布粗采样后，以 ref 归一化平均方向给 src 评分，在固定 src 预算内选区精算 | ref 归一化平均方向评分定义已确认；预算参数待定；未启动 | 固定 src 预算等参数待定；无结果；见 [实验文档](exp/main-ads-v0/exp.md) |
| main-crb-v0 | crb-histgram | ordinal 1 为 ref、ordinal 0 为 src；q=8/16/32/64 四档 cover+抵消排序 split 的 CRB 与逐侧预算匹配 uni | 完成；复用 main-ads-v0 训练与梯度缓存（只读）；BF16 余弦矩阵、精确分数统计、100/256桶直方图与选择审计 `.complete` 已核验 | q32 src 锚点7195/ref8；负值占比 CRB/uni=0.519/0.585，>0.05占比0.181/0.161，均值-0.0022/-0.0116；详见 [实验记录](exp/main-crb-v0/exp.md) / [结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-grad-budget-time1-ref-q8 | 旧 ordinal 0 单轨迹 q16 预算草案 | 已被完整 src-256 池范围取代；旧核算仅保留在历史操作日志 | 无结果文件；更正见 [实验记录](exp/main-crb-v0/exp.md) / [结果](exp/main-crb-v0/results.md) |
| main-reconst-v0 | bisect-vs-uniform | ordinal 0/1 两条完整轨迹；q=4/8/16/32/64/128 下最长区间二分与预算匹配均匀锚点的线性插值回放重建 | 完成；纯回放无梯度计算，只读复用 main-ads-v0 缓存；24 个布局与 `.complete` 已核验 | 长轨迹 q4/8/16/32/64/128 平均余弦 二分=0.9655/0.8912/0.7603/0.5969/0.4498/0.3007，均匀=0.9744/0.8984/0.7624/0.6015/0.4689/0.3063；均匀在全部六档更高；短轨迹 q>=32 两法退化为相同锚点；详见 [实验记录](exp/main-reconst-v0/exp.md) / [结果](exp/main-reconst-v0/results.md) |
| main-crb-v0 | q16-src256-time1-q8-four-arm | 完整 src-256 q16 CRB/UNI、TIME1 ref q8，方向插值筛选及 CRB/REFT/ORI/UNI 对比 | 完成；ORI、CRB、UNI、REFT 均已评估，四臂完成标记核验 | MSE/MAE/RMSE：ORI 0.0640/0.1519/0.1746，CRB 0.0637/0.1485/0.1710，UNI 0.0633/0.1494/0.1718，REFT 0.0771/0.1611/0.1866；详见 outputs/eval/main-crb-v0/q16-src256-time1-q8/{ori,crb,uni,reft}/ 及实验结果 |
| main-crb-v0 | q16-src256-eight-task-FEV09 | hospital_admissions_1D：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/FEV09/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME1 | CPHL/15T：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME1/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME2 | CPHL/30T：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME2/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME3 | CPHL/H：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME3/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME4 | ECDC_COVID/D：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME4/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME6 | epf_electricity_price/H：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME6/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME7 | Commodity_Import/M：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME7/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-TIME8 | Commodity_Production/M：新 ref 梯度、新 source 选择、CRB/UNI/REFT 重训和四臂评估 | 完成：2026-11-16 17:44:59 CST；ORI/CRB/UNI/REFT 四臂评估均完成 | 四臂 MSE/MASE 指标及结果文件路径见 [实验结果](exp/main-crb-v0/results.md)；outputs/eval/main-crb-v0/q16-src256-eight-task/TIME8/；[结果](exp/main-crb-v0/results.md) |
| main-crb-v0 | q16-src256-eight-task-union-FEV09 | hospital_admissions_1D：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=0.7266（CRB+REFT）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/FEV09/
| main-crb-v0 | q16-src256-eight-task-union-TIME1 | CPHL/15T：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=0.7090（CRB）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME1/
| main-crb-v0 | q16-src256-eight-task-union-TIME2 | CPHL/30T：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=1.2323（UNI+REFT）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME2/
| main-crb-v0 | q16-src256-eight-task-union-TIME3 | CPHL/H：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=1.1749（REFT）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME3/
| main-crb-v0 | q16-src256-eight-task-union-TIME4 | ECDC_COVID/D：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=2.4663（REFT）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME4/
| main-crb-v0 | q16-src256-eight-task-union-TIME6 | epf_electricity_price/H：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=1.0154（REFT）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME6/
| main-crb-v0 | q16-src256-eight-task-union-TIME7 | Commodity_Import/M：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=1.6025（UNI+REFT）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME7/
| main-crb-v0 | q16-src256-eight-task-union-TIME8 | Commodity_Production/M：选中 source 窗口分别与对应任务全部 ref 窗口合并，训练 CRB+REFT 和 UNI+REFT | 完成：2026-11-16 18:22:58 CST；CRB+REFT、UNI+REFT 均已训练评估 | 最低 MASE=1.1075（ORI）；六臂矩阵见 [实验结果](exp/main-crb-v0/results.md)，指标目录 outputs/eval/main-crb-v0/q16-src256-eight-task/TIME8/
| main-crb-v0 | q16-src256-eight-task-baselines-rand | src-256 全部合法窗口中以 PCG64(seed=0) 无放回选 638,202 个；共享训练一次并评估八任务 | 方案已记录；未启动 | 无指标或结果文件；规划输出 outputs/{selection,train,eval}/main-crb-v0/q16-src256-eight-task-baselines/；[实验文档](exp/main-crb-v0/exp.md) |
| main-crb-v0 | q16-src256-eight-task-baselines-tsds | 复用 TSDS 冻结编码器特征及 IVF/KDE/OT 原生概率；q16/q8 锚点适配，top25 轨迹概率截断；八任务分别训练评估 | 方案已记录；未启动 | 无指标或结果文件；规划输出 outputs/{selection,train,eval}/main-crb-v0/q16-src256-eight-task-baselines/；[实验文档](exp/main-crb-v0/exp.md) |
| main-crb-v0 | q16-src256-eight-task-baselines-lp-mirror | 复用线性 proxy/ref adaptation/loss-drop value mapping，取 top 638,202 source 窗口；八任务分别训练评估 | 方案已记录；未启动 | 无指标或结果文件；全池打分及 proxy 输出待实现；[实验文档](exp/main-crb-v0/exp.md) |
| main-crb-v0 | q16-src256-eight-task-baselines-grand-static | UNI q16 锚点梯度范数形成区间分数，取 top 10% 区间；共享训练一次并评估八任务 | 方案已记录；未启动 | 无指标或结果文件；区间展开窗口数待生成后记录；[实验文档](exp/main-crb-v0/exp.md) |
| main-crb-v0 | q16-src256-eight-task-baselines-crb-traj | CRB q16 锚点生成轨迹表示，以 ref 最大 cosine 选择 top25 轨迹；每次抽样在轨迹内均匀取窗；八任务分别训练评估 | 方案已记录；未启动 | 无指标或结果文件；[实验文档](exp/main-crb-v0/exp.md) |
| main-crb-v0 | q16-src256-eight-task-baselines-uni-traj | UNI q16 锚点生成轨迹表示，以 ref 最大 cosine 选择 top25 轨迹；每次抽样在轨迹内均匀取窗；八任务分别训练评估 | 方案已记录；未启动 | 无指标或结果文件；[实验文档](exp/main-crb-v0/exp.md) |


### main-crb-v0：q16-src256-eight-task-resample

以下为本次实际执行批次。早期 `q16-src256-eight-task-baselines` 六行是未启动的硬截断草案，已被本批全池概率重采样方案取代；大写 `Rand` 固定 10% 子集不启动，小写 `rand` 为完整 src-256 均匀训练。

| 实验代号 | 批次 | 工作 | 状态 | 实际结果 |
|---|---|---|---|---|
| main-crb-v0 | q16-src256-eight-task-resample-ori | 八任务 TimerXL 底座评估 | 无需训练；八任务评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/ori/` |
| main-crb-v0 | q16-src256-eight-task-resample-reft | 八任务全安全 ref 窗口训练与评估 | 八任务训练与评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/reft/` |
| main-crb-v0 | q16-src256-eight-task-resample-rand | 完整 src-256 均匀重采样概率，训练一次并评估八任务 | 共享训练 1/1、八任务评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/rand/` |
| main-crb-v0 | q16-src256-eight-task-resample-grand-static | UNI q16 范数区间全池概率，训练一次并评估八任务 | 共享训练 1/1、八任务评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/grand-static/` |
| main-crb-v0 | q16-src256-eight-task-resample-crb | ref `linearSum` 的 CRB 全池窗口概率 | 八任务训练与评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/crb/` |
| main-crb-v0 | q16-src256-eight-task-resample-uni | ref `linearSum` 的 UNI 全池窗口概率 | 八任务训练与评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/uni/` |
| main-crb-v0 | q16-src256-eight-task-resample-crb-traj | CRB q16 轨迹表示概率，轨迹内均匀抽窗 | 八任务训练与评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/crb-traj/` |
| main-crb-v0 | q16-src256-eight-task-resample-uni-traj | UNI q16 轨迹表示概率，轨迹内均匀抽窗 | 八任务训练与评估完成 | MASE 矩阵与原始指标文件见 `docs/exp/main-crb-v0/results.md`；评估路径 `outputs/eval/main-crb-v0/q16-src256-eight-task-resample/<TASK>/uni-traj/` |
| main-crb-v0 | q16-src256-eight-task-resample-tsds | TSDS 原生 OT 轨迹概率，训练评估八任务 | native 特征/概率准备已重启；尚无评估 | 无 MASE；TSDS/LP-Mirror 原生准备 tmux 队列启动于 2026-11-16 20:47 CST |
| main-crb-v0 | q16-src256-eight-task-resample-lp-mirror | LP-Mirror 全池 value 概率，训练评估八任务 | native 特征/概率准备已重启；尚无评估 | 无 MASE；TSDS/LP-Mirror 原生准备 tmux 队列启动于 2026-11-16 20:47 CST |

| main-crb-v0 | data-select-005-linear | ORI/CRB/UNI/REFT/rand/GraNd-static/TSDS/LP-Mirror/CRB-traj/UNI-traj；八任务固定选择 top 5% 窗口训练 | 已中止（用户要求）；42/58 训练、64/80 评估、58/58 选择完成；ORI/REFT/rand/GraNd-static/CRB/UNI/CRB-traj/UNI-traj 已评估；TSDS 部分训练、LP-Mirror 未启动；详见实验结果文档 | 尚无结果；58 次训练、80 次评估计划；详见 [实验文档](exp/main-crb-v0/exp.md) |

<!-- main-crb-supp begin -->
| 实验 | 任务 | 方法 | 工作内容 | 状态 | 实际结果 | 详情文档 |
| --- | --- | --- | --- | --- | --- | --- |
| main-crb-supp | time-crypto-d | e1-uniform-mean | Crypto/D step 1000 | 已完成 | MASE=7.4192848 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e1-uniform-mean | Crypto/D step 2000 | 已完成 | MASE=7.4154385 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e1-uniform-mean | Crypto/D step 4000 | 已完成 | MASE=7.5048464 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e2-uniform-shape | Crypto/D step 1000 | 已完成 | MASE=7.2275756 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e2-uniform-shape | Crypto/D step 2000 | 已完成 | MASE=7.1159316 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e2-uniform-shape | Crypto/D step 4000 | 已完成 | MASE=7.1397204 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e3-crb-mean | Crypto/D step 1000 | 已完成 | MASE=7.3321109 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e3-crb-mean | Crypto/D step 2000 | 已完成 | MASE=7.4007632 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e3-crb-mean | Crypto/D step 4000 | 已完成 | MASE=7.429943 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e4-crb-shape | Crypto/D step 1000 | 已完成 | MASE=7.1789961 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e4-crb-shape | Crypto/D step 2000 | 已完成 | MASE=7.172536 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e4-crb-shape | Crypto/D step 4000 | 已完成 | MASE=7.1962432 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e5-target-only | Crypto/D step 1000 | 已完成 | MASE=11.054436 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e5-target-only | Crypto/D step 2000 | 已完成 | MASE=12.157008 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e5-target-only | Crypto/D step 4000 | 已完成 | MASE=12.314751 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-crypto-d | e0-rand | Crypto/D step 4000 | 已完成 | MASE=7.5576342 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e1-uniform-mean | Supply_Chain_Customer/D step 1000 | 已完成 | MASE=1.0716929 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e1-uniform-mean | Supply_Chain_Customer/D step 2000 | 已完成 | MASE=1.0715374 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e1-uniform-mean | Supply_Chain_Customer/D step 4000 | 已完成 | MASE=1.0719844 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e2-uniform-shape | Supply_Chain_Customer/D step 1000 | 已完成 | MASE=1.0753001 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e2-uniform-shape | Supply_Chain_Customer/D step 2000 | 已完成 | MASE=1.0758565 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e2-uniform-shape | Supply_Chain_Customer/D step 4000 | 已完成 | MASE=1.0746984 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e3-crb-mean | Supply_Chain_Customer/D step 1000 | 已完成 | MASE=1.0813615 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e3-crb-mean | Supply_Chain_Customer/D step 2000 | 已完成 | MASE=1.0854521 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e3-crb-mean | Supply_Chain_Customer/D step 4000 | 已完成 | MASE=1.0879403 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e4-crb-shape | Supply_Chain_Customer/D step 1000 | 已完成 | MASE=1.0850257 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e4-crb-shape | Supply_Chain_Customer/D step 2000 | 已完成 | MASE=1.0801942 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e4-crb-shape | Supply_Chain_Customer/D step 4000 | 已完成 | MASE=1.0822905 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e5-target-only | Supply_Chain_Customer/D step 1000 | 已完成 | MASE=0.73185718 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e5-target-only | Supply_Chain_Customer/D step 2000 | 已完成 | MASE=0.69837674 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e5-target-only | Supply_Chain_Customer/D step 4000 | 已完成 | MASE=0.69473948 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-customer-d | e0-rand | Supply_Chain_Customer/D step 4000 | 已完成 | MASE=1.0762807 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e1-uniform-mean | Supply_Chain_Location/D step 1000 | 已完成 | MASE=1.0607197 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e1-uniform-mean | Supply_Chain_Location/D step 2000 | 已完成 | MASE=1.0611659 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e1-uniform-mean | Supply_Chain_Location/D step 4000 | 已完成 | MASE=1.0577799 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e2-uniform-shape | Supply_Chain_Location/D step 1000 | 已完成 | MASE=1.0710086 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e2-uniform-shape | Supply_Chain_Location/D step 2000 | 已完成 | MASE=1.058407 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e2-uniform-shape | Supply_Chain_Location/D step 4000 | 已完成 | MASE=1.0617993 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e3-crb-mean | Supply_Chain_Location/D step 1000 | 已完成 | MASE=1.0696135 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e3-crb-mean | Supply_Chain_Location/D step 2000 | 已完成 | MASE=1.0753476 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e3-crb-mean | Supply_Chain_Location/D step 4000 | 已完成 | MASE=1.07366 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e4-crb-shape | Supply_Chain_Location/D step 1000 | 已完成 | MASE=1.0685196 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e4-crb-shape | Supply_Chain_Location/D step 2000 | 已完成 | MASE=1.0740856 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e4-crb-shape | Supply_Chain_Location/D step 4000 | 已完成 | MASE=1.0719948 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e5-target-only | Supply_Chain_Location/D step 1000 | 已完成 | MASE=0.73730998 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e5-target-only | Supply_Chain_Location/D step 2000 | 已完成 | MASE=0.71789209 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e5-target-only | Supply_Chain_Location/D step 4000 | 已完成 | MASE=0.71726027 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-supply-chain-location-d | e0-rand | Supply_Chain_Location/D step 4000 | 已完成 | MASE=1.0569077 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e1-uniform-mean | NE_China_Wind/H step 1000 | 已完成 | MASE=0.57289688 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e1-uniform-mean | NE_China_Wind/H step 2000 | 已完成 | MASE=0.56929818 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e1-uniform-mean | NE_China_Wind/H step 4000 | 已完成 | MASE=0.56595949 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e2-uniform-shape | NE_China_Wind/H step 1000 | 已完成 | MASE=0.57129801 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e2-uniform-shape | NE_China_Wind/H step 2000 | 已完成 | MASE=0.56484912 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e2-uniform-shape | NE_China_Wind/H step 4000 | 已完成 | MASE=0.56237633 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e3-crb-mean | NE_China_Wind/H step 1000 | 已完成 | MASE=0.56754678 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e3-crb-mean | NE_China_Wind/H step 2000 | 已完成 | MASE=0.55974638 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e3-crb-mean | NE_China_Wind/H step 4000 | 已完成 | MASE=0.56122152 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e4-crb-shape | NE_China_Wind/H step 1000 | 已完成 | MASE=0.57171805 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e4-crb-shape | NE_China_Wind/H step 2000 | 已完成 | MASE=0.56031013 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e4-crb-shape | NE_China_Wind/H step 4000 | 已完成 | MASE=0.56305958 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e5-target-only | NE_China_Wind/H step 1000 | 已完成 | MASE=0.64233483 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e5-target-only | NE_China_Wind/H step 2000 | 已完成 | MASE=0.65342085 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e5-target-only | NE_China_Wind/H step 4000 | 已完成 | MASE=0.65811015 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-ne-china-wind-h | e0-rand | NE_China_Wind/H step 4000 | 已完成 | MASE=0.56655073 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e1-uniform-mean | Coastal_T_S/20T step 1000 | 已完成 | MASE=0.96713499 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e1-uniform-mean | Coastal_T_S/20T step 2000 | 已完成 | MASE=0.96247784 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e1-uniform-mean | Coastal_T_S/20T step 4000 | 已完成 | MASE=0.96368114 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e2-uniform-shape | Coastal_T_S/20T step 1000 | 已完成 | MASE=0.94811293 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e2-uniform-shape | Coastal_T_S/20T step 2000 | 已完成 | MASE=0.9515056 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e2-uniform-shape | Coastal_T_S/20T step 4000 | 已完成 | MASE=0.9520363 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e3-crb-mean | Coastal_T_S/20T step 1000 | 已完成 | MASE=0.96135956 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e3-crb-mean | Coastal_T_S/20T step 2000 | 已完成 | MASE=0.95515966 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e3-crb-mean | Coastal_T_S/20T step 4000 | 已完成 | MASE=0.95460704 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e4-crb-shape | Coastal_T_S/20T step 1000 | 已完成 | MASE=0.95552344 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e4-crb-shape | Coastal_T_S/20T step 2000 | 已完成 | MASE=0.9476279 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e4-crb-shape | Coastal_T_S/20T step 4000 | 已完成 | MASE=0.94901016 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e5-target-only | Coastal_T_S/20T step 1000 | 已完成 | MASE=0.87125877 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e5-target-only | Coastal_T_S/20T step 2000 | 已完成 | MASE=0.86312245 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e5-target-only | Coastal_T_S/20T step 4000 | 已完成 | MASE=0.85960794 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-20t | e0-rand | Coastal_T_S/20T step 4000 | 已完成 | MASE=0.96048981 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e1-uniform-mean | Australia_Solar/H step 1000 | 已完成 | MASE=0.68100703 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e1-uniform-mean | Australia_Solar/H step 2000 | 已完成 | MASE=0.68200138 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e1-uniform-mean | Australia_Solar/H step 4000 | 已完成 | MASE=0.68293672 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e2-uniform-shape | Australia_Solar/H step 1000 | 已完成 | MASE=0.66571156 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e2-uniform-shape | Australia_Solar/H step 2000 | 已完成 | MASE=0.66819924 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e2-uniform-shape | Australia_Solar/H step 4000 | 已完成 | MASE=0.66698649 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e3-crb-mean | Australia_Solar/H step 1000 | 已完成 | MASE=0.67884567 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e3-crb-mean | Australia_Solar/H step 2000 | 已完成 | MASE=0.67694043 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e3-crb-mean | Australia_Solar/H step 4000 | 已完成 | MASE=0.67747172 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e4-crb-shape | Australia_Solar/H step 1000 | 已完成 | MASE=0.65957539 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e4-crb-shape | Australia_Solar/H step 2000 | 已完成 | MASE=0.66203433 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e4-crb-shape | Australia_Solar/H step 4000 | 已完成 | MASE=0.66132167 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e5-target-only | Australia_Solar/H step 1000 | 已完成 | MASE=0.59746867 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e5-target-only | Australia_Solar/H step 2000 | 已完成 | MASE=0.59222444 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e5-target-only | Australia_Solar/H step 4000 | 已完成 | MASE=0.59419195 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-australia-solar-h | e0-rand | Australia_Solar/H step 4000 | 已完成 | MASE=0.66979378 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e1-uniform-mean | Finland_Traffic/15T step 1000 | 已完成 | MASE=0.68752304 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e1-uniform-mean | Finland_Traffic/15T step 2000 | 已完成 | MASE=0.68221762 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e1-uniform-mean | Finland_Traffic/15T step 4000 | 已完成 | MASE=0.67441473 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e2-uniform-shape | Finland_Traffic/15T step 1000 | 已完成 | MASE=0.69336519 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e2-uniform-shape | Finland_Traffic/15T step 2000 | 已完成 | MASE=0.68841843 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e2-uniform-shape | Finland_Traffic/15T step 4000 | 已完成 | MASE=0.68806845 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e3-crb-mean | Finland_Traffic/15T step 1000 | 已完成 | MASE=0.69142619 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e3-crb-mean | Finland_Traffic/15T step 2000 | 已完成 | MASE=0.68132879 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e3-crb-mean | Finland_Traffic/15T step 4000 | 已完成 | MASE=0.67830249 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e4-crb-shape | Finland_Traffic/15T step 1000 | 已完成 | MASE=0.69019823 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e4-crb-shape | Finland_Traffic/15T step 2000 | 已完成 | MASE=0.68337335 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e4-crb-shape | Finland_Traffic/15T step 4000 | 已完成 | MASE=0.68230408 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e5-target-only | Finland_Traffic/15T step 1000 | 已完成 | MASE=0.49015699 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e5-target-only | Finland_Traffic/15T step 2000 | 已完成 | MASE=0.46746813 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e5-target-only | Finland_Traffic/15T step 4000 | 已完成 | MASE=0.46593138 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-finland-traffic-15t | e0-rand | Finland_Traffic/15T step 4000 | 已完成 | MASE=0.68989616 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e1-uniform-mean | 10 step 1000 | 已完成 | MASE=0.52213836 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e1-uniform-mean | 10 step 2000 | 已完成 | MASE=0.52031516 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e1-uniform-mean | 10 step 4000 | 已完成 | MASE=0.52069804 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e2-uniform-shape | 10 step 1000 | 已完成 | MASE=0.51893111 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e2-uniform-shape | 10 step 2000 | 已完成 | MASE=0.51891339 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e2-uniform-shape | 10 step 4000 | 已完成 | MASE=0.51878768 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e3-crb-mean | 10 step 1000 | 已完成 | MASE=0.5186137 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e3-crb-mean | 10 step 2000 | 已完成 | MASE=0.5155919 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e3-crb-mean | 10 step 4000 | 已完成 | MASE=0.51520432 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e4-crb-shape | 10 step 1000 | 已完成 | MASE=0.51780706 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e4-crb-shape | 10 step 2000 | 已完成 | MASE=0.51664102 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e4-crb-shape | 10 step 4000 | 已完成 | MASE=0.51416732 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e5-target-only | 10 step 1000 | 已完成 | MASE=0.52020863 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e5-target-only | 10 step 2000 | 已完成 | MASE=0.53611716 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e5-target-only | 10 step 4000 | 已完成 | MASE=0.53942242 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-jena-weather-1h | e0-rand | 10 step 4000 | 已完成 | MASE=0.53299378 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e1-uniform-mean | 6 step 1000 | 已完成 | MASE=1.1941609 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e1-uniform-mean | 6 step 2000 | 已完成 | MASE=1.190632 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e1-uniform-mean | 6 step 4000 | 已完成 | MASE=1.1935967 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e2-uniform-shape | 6 step 1000 | 已完成 | MASE=1.1698991 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e2-uniform-shape | 6 step 2000 | 已完成 | MASE=1.1698741 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e2-uniform-shape | 6 step 4000 | 已完成 | MASE=1.1658128 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e3-crb-mean | 6 step 1000 | 已完成 | MASE=1.1714572 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e3-crb-mean | 6 step 2000 | 已完成 | MASE=1.1742403 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e3-crb-mean | 6 step 4000 | 已完成 | MASE=1.1748618 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e4-crb-shape | 6 step 1000 | 已完成 | MASE=1.1589232 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e4-crb-shape | 6 step 2000 | 已完成 | MASE=1.1617327 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e4-crb-shape | 6 step 4000 | 已完成 | MASE=1.1593857 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e5-target-only | 6 step 1000 | 已完成 | MASE=1.1721509 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e5-target-only | 6 step 2000 | 已完成 | MASE=1.1688485 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e5-target-only | 6 step 4000 | 已完成 | MASE=1.173124 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-ett-1h | e0-rand | 6 step 4000 | 已完成 | MASE=1.1701318 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e1-uniform-mean | MetroPT-3/5T step 1000 | 已完成 | MASE=0.69504795 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e1-uniform-mean | MetroPT-3/5T step 2000 | 已完成 | MASE=0.69529678 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e1-uniform-mean | MetroPT-3/5T step 4000 | 已完成 | MASE=0.69469144 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e2-uniform-shape | MetroPT-3/5T step 1000 | 已完成 | MASE=0.70201767 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e2-uniform-shape | MetroPT-3/5T step 2000 | 已完成 | MASE=0.69916254 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e2-uniform-shape | MetroPT-3/5T step 4000 | 已完成 | MASE=0.70006681 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e3-crb-mean | MetroPT-3/5T step 1000 | 已完成 | MASE=0.69487918 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e3-crb-mean | MetroPT-3/5T step 2000 | 已完成 | MASE=0.69651304 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e3-crb-mean | MetroPT-3/5T step 4000 | 已完成 | MASE=0.69622639 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e4-crb-shape | MetroPT-3/5T step 1000 | 已完成 | MASE=0.70254759 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e4-crb-shape | MetroPT-3/5T step 2000 | 已完成 | MASE=0.7026569 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e4-crb-shape | MetroPT-3/5T step 4000 | 已完成 | MASE=0.70218962 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e5-target-only | MetroPT-3/5T step 1000 | 已完成 | MASE=0.69694598 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e5-target-only | MetroPT-3/5T step 2000 | 已完成 | MASE=0.69948907 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e5-target-only | MetroPT-3/5T step 4000 | 已完成 | MASE=0.70106016 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-metropt-3-5t | e0-rand | MetroPT-3/5T step 4000 | 已完成 | MASE=0.69524684 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e1-uniform-mean | current_velocity/5T step 1000 | 已完成 | MASE=0.59813109 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e1-uniform-mean | current_velocity/5T step 2000 | 已完成 | MASE=0.59498473 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e1-uniform-mean | current_velocity/5T step 4000 | 已完成 | MASE=0.596304 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e2-uniform-shape | current_velocity/5T step 1000 | 已完成 | MASE=0.60059213 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e2-uniform-shape | current_velocity/5T step 2000 | 已完成 | MASE=0.60371871 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e2-uniform-shape | current_velocity/5T step 4000 | 已完成 | MASE=0.60369881 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e3-crb-mean | current_velocity/5T step 1000 | 已完成 | MASE=0.59680531 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e3-crb-mean | current_velocity/5T step 2000 | 已完成 | MASE=0.60009711 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e3-crb-mean | current_velocity/5T step 4000 | 已完成 | MASE=0.60318128 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e4-crb-shape | current_velocity/5T step 1000 | 已完成 | MASE=0.60072655 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e4-crb-shape | current_velocity/5T step 2000 | 已完成 | MASE=0.60510322 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e4-crb-shape | current_velocity/5T step 4000 | 已完成 | MASE=0.60117035 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e5-target-only | current_velocity/5T step 1000 | 已完成 | MASE=0.61761652 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e5-target-only | current_velocity/5T step 2000 | 已完成 | MASE=0.64364384 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e5-target-only | current_velocity/5T step 4000 | 已完成 | MASE=0.63862766 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-5t | e0-rand | current_velocity/5T step 4000 | 已完成 | MASE=0.59170645 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e1-uniform-mean | azure2019_U/5T step 1000 | 已完成 | MASE=0.87870582 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e1-uniform-mean | azure2019_U/5T step 2000 | 已完成 | MASE=0.87713649 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e1-uniform-mean | azure2019_U/5T step 4000 | 已完成 | MASE=0.87747281 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e2-uniform-shape | azure2019_U/5T step 1000 | 已完成 | MASE=0.87580101 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e2-uniform-shape | azure2019_U/5T step 2000 | 已完成 | MASE=0.87624909 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e2-uniform-shape | azure2019_U/5T step 4000 | 已完成 | MASE=0.87828522 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e3-crb-mean | azure2019_U/5T step 1000 | 已完成 | MASE=0.87802323 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e3-crb-mean | azure2019_U/5T step 2000 | 已完成 | MASE=0.87367568 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e3-crb-mean | azure2019_U/5T step 4000 | 已完成 | MASE=0.87588757 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e4-crb-shape | azure2019_U/5T step 1000 | 已完成 | MASE=0.88047138 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e4-crb-shape | azure2019_U/5T step 2000 | 已完成 | MASE=0.87905904 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e4-crb-shape | azure2019_U/5T step 4000 | 已完成 | MASE=0.87812717 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e5-target-only | azure2019_U/5T step 1000 | 已完成 | MASE=0.85593466 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e5-target-only | azure2019_U/5T step 2000 | 已完成 | MASE=0.85177463 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e5-target-only | azure2019_U/5T step 4000 | 已完成 | MASE=0.85138624 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-u-5t | e0-rand | azure2019_U/5T step 4000 | 已完成 | MASE=0.90189597 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e1-uniform-mean | Water_Quality_Darwin/15T step 1000 | 已完成 | MASE=0.92489696 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e1-uniform-mean | Water_Quality_Darwin/15T step 2000 | 已完成 | MASE=0.93078329 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e1-uniform-mean | Water_Quality_Darwin/15T step 4000 | 已完成 | MASE=0.92429212 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e2-uniform-shape | Water_Quality_Darwin/15T step 1000 | 已完成 | MASE=0.92504267 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e2-uniform-shape | Water_Quality_Darwin/15T step 2000 | 已完成 | MASE=0.92750913 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e2-uniform-shape | Water_Quality_Darwin/15T step 4000 | 已完成 | MASE=0.92512797 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e3-crb-mean | Water_Quality_Darwin/15T step 1000 | 已完成 | MASE=0.93519862 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e3-crb-mean | Water_Quality_Darwin/15T step 2000 | 已完成 | MASE=0.93373667 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e3-crb-mean | Water_Quality_Darwin/15T step 4000 | 已完成 | MASE=0.93349772 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e4-crb-shape | Water_Quality_Darwin/15T step 1000 | 已完成 | MASE=0.92718172 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e4-crb-shape | Water_Quality_Darwin/15T step 2000 | 已完成 | MASE=0.92802705 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e4-crb-shape | Water_Quality_Darwin/15T step 4000 | 已完成 | MASE=0.92644363 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e5-target-only | Water_Quality_Darwin/15T step 1000 | 已完成 | MASE=0.85127939 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e5-target-only | Water_Quality_Darwin/15T step 2000 | 已完成 | MASE=0.84632171 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e5-target-only | Water_Quality_Darwin/15T step 4000 | 已完成 | MASE=0.84451361 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-water-quality-darwin-15t | e0-rand | Water_Quality_Darwin/15T step 4000 | 已完成 | MASE=0.95859565 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e1-uniform-mean | SG_PM25/H step 1000 | 已完成 | MASE=0.88614518 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e1-uniform-mean | SG_PM25/H step 2000 | 已完成 | MASE=0.88489656 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e1-uniform-mean | SG_PM25/H step 4000 | 已完成 | MASE=0.884251 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e2-uniform-shape | SG_PM25/H step 1000 | 已完成 | MASE=0.8849632 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e2-uniform-shape | SG_PM25/H step 2000 | 已完成 | MASE=0.88525775 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e2-uniform-shape | SG_PM25/H step 4000 | 已完成 | MASE=0.88474776 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e3-crb-mean | SG_PM25/H step 1000 | 已完成 | MASE=0.88768067 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e3-crb-mean | SG_PM25/H step 2000 | 已完成 | MASE=0.88589504 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e3-crb-mean | SG_PM25/H step 4000 | 已完成 | MASE=0.88534152 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e4-crb-shape | SG_PM25/H step 1000 | 已完成 | MASE=0.88361962 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e4-crb-shape | SG_PM25/H step 2000 | 已完成 | MASE=0.88386794 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e4-crb-shape | SG_PM25/H step 4000 | 已完成 | MASE=0.88343784 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e5-target-only | SG_PM25/H step 1000 | 已完成 | MASE=0.9096941 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e5-target-only | SG_PM25/H step 2000 | 已完成 | MASE=0.90746226 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e5-target-only | SG_PM25/H step 4000 | 已完成 | MASE=0.90897083 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-sg-pm25-h | e0-rand | SG_PM25/H step 4000 | 已完成 | MASE=0.90985603 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e1-uniform-mean | OpenElectricity_NEM/5T step 1000 | 已完成 | MASE=0.56719551 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e1-uniform-mean | OpenElectricity_NEM/5T step 2000 | 已完成 | MASE=0.56612143 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e1-uniform-mean | OpenElectricity_NEM/5T step 4000 | 已完成 | MASE=0.56722102 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e2-uniform-shape | OpenElectricity_NEM/5T step 1000 | 已完成 | MASE=0.56731191 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e2-uniform-shape | OpenElectricity_NEM/5T step 2000 | 已完成 | MASE=0.56198171 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e2-uniform-shape | OpenElectricity_NEM/5T step 4000 | 已完成 | MASE=0.56065142 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e3-crb-mean | OpenElectricity_NEM/5T step 1000 | 已完成 | MASE=0.56149759 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e3-crb-mean | OpenElectricity_NEM/5T step 2000 | 已完成 | MASE=0.55432312 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e3-crb-mean | OpenElectricity_NEM/5T step 4000 | 已完成 | MASE=0.55633549 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e4-crb-shape | OpenElectricity_NEM/5T step 1000 | 已完成 | MASE=0.55679604 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e4-crb-shape | OpenElectricity_NEM/5T step 2000 | 已完成 | MASE=0.55711773 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e4-crb-shape | OpenElectricity_NEM/5T step 4000 | 已完成 | MASE=0.5567145 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e5-target-only | OpenElectricity_NEM/5T step 1000 | 已完成 | MASE=0.49897416 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e5-target-only | OpenElectricity_NEM/5T step 2000 | 已完成 | MASE=0.48759892 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e5-target-only | OpenElectricity_NEM/5T step 4000 | 已完成 | MASE=0.48683319 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-open-electricity-nem-5t | e0-rand | OpenElectricity_NEM/5T step 4000 | 已完成 | MASE=0.55463701 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e1-uniform-mean | EWELD_Load/15T step 1000 | 已完成 | MASE=0.72977287 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e1-uniform-mean | EWELD_Load/15T step 2000 | 已完成 | MASE=0.73054975 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e1-uniform-mean | EWELD_Load/15T step 4000 | 已完成 | MASE=0.72426089 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e2-uniform-shape | EWELD_Load/15T step 1000 | 已完成 | MASE=0.72483601 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e2-uniform-shape | EWELD_Load/15T step 2000 | 已完成 | MASE=0.72260005 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e2-uniform-shape | EWELD_Load/15T step 4000 | 已完成 | MASE=0.72177606 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e3-crb-mean | EWELD_Load/15T step 1000 | 已完成 | MASE=0.73115807 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e3-crb-mean | EWELD_Load/15T step 2000 | 已完成 | MASE=0.72843701 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e3-crb-mean | EWELD_Load/15T step 4000 | 已完成 | MASE=0.72543024 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e4-crb-shape | EWELD_Load/15T step 1000 | 已完成 | MASE=0.73416546 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e4-crb-shape | EWELD_Load/15T step 2000 | 已完成 | MASE=0.72595549 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e4-crb-shape | EWELD_Load/15T step 4000 | 已完成 | MASE=0.72605098 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e5-target-only | EWELD_Load/15T step 1000 | 已完成 | MASE=0.70170407 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e5-target-only | EWELD_Load/15T step 2000 | 已完成 | MASE=0.70456038 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e5-target-only | EWELD_Load/15T step 4000 | 已完成 | MASE=0.70180225 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-eweld-load-15t | e0-rand | EWELD_Load/15T step 4000 | 已完成 | MASE=0.74862779 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e1-uniform-mean | Port_Activity/D step 1000 | 已完成 | MASE=0.71342388 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e1-uniform-mean | Port_Activity/D step 2000 | 已完成 | MASE=0.71462153 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e1-uniform-mean | Port_Activity/D step 4000 | 已完成 | MASE=0.71397261 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e2-uniform-shape | Port_Activity/D step 1000 | 已完成 | MASE=0.7151438 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e2-uniform-shape | Port_Activity/D step 2000 | 已完成 | MASE=0.71393591 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e2-uniform-shape | Port_Activity/D step 4000 | 已完成 | MASE=0.71496848 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e3-crb-mean | Port_Activity/D step 1000 | 已完成 | MASE=0.71950736 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e3-crb-mean | Port_Activity/D step 2000 | 已完成 | MASE=0.71977612 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e3-crb-mean | Port_Activity/D step 4000 | 已完成 | MASE=0.72072606 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e4-crb-shape | Port_Activity/D step 1000 | 已完成 | MASE=0.71694229 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e4-crb-shape | Port_Activity/D step 2000 | 已完成 | MASE=0.71656849 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e4-crb-shape | Port_Activity/D step 4000 | 已完成 | MASE=0.71639494 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e5-target-only | Port_Activity/D step 1000 | 已完成 | MASE=0.72066191 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e5-target-only | Port_Activity/D step 2000 | 已完成 | MASE=0.72079916 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e5-target-only | Port_Activity/D step 4000 | 已完成 | MASE=0.71907231 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-port-activity-d | e0-rand | Port_Activity/D step 4000 | 已完成 | MASE=0.73058484 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e1-uniform-mean | current_velocity/10T step 1000 | 已完成 | MASE=0.93782195 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e1-uniform-mean | current_velocity/10T step 2000 | 已完成 | MASE=0.93890088 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e1-uniform-mean | current_velocity/10T step 4000 | 已完成 | MASE=0.9380725 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e2-uniform-shape | current_velocity/10T step 1000 | 已完成 | MASE=0.93914641 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e2-uniform-shape | current_velocity/10T step 2000 | 已完成 | MASE=0.93525125 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e2-uniform-shape | current_velocity/10T step 4000 | 已完成 | MASE=0.9360845 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e3-crb-mean | current_velocity/10T step 1000 | 已完成 | MASE=0.92469269 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e3-crb-mean | current_velocity/10T step 2000 | 已完成 | MASE=0.92353827 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e3-crb-mean | current_velocity/10T step 4000 | 已完成 | MASE=0.92315395 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e4-crb-shape | current_velocity/10T step 1000 | 已完成 | MASE=0.92891771 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e4-crb-shape | current_velocity/10T step 2000 | 已完成 | MASE=0.92736949 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e4-crb-shape | current_velocity/10T step 4000 | 已完成 | MASE=0.92717958 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e5-target-only | current_velocity/10T step 1000 | 已完成 | MASE=0.70118122 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e5-target-only | current_velocity/10T step 2000 | 已完成 | MASE=0.69253817 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e5-target-only | current_velocity/10T step 4000 | 已完成 | MASE=0.69009819 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-10t | e0-rand | current_velocity/10T step 4000 | 已完成 | MASE=0.95654371 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e1-uniform-mean | Coastal_T_S/5T step 1000 | 已完成 | MASE=1.1736758 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e1-uniform-mean | Coastal_T_S/5T step 2000 | 已完成 | MASE=1.1765825 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e1-uniform-mean | Coastal_T_S/5T step 4000 | 已完成 | MASE=1.1752976 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e2-uniform-shape | Coastal_T_S/5T step 1000 | 已完成 | MASE=1.1667169 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e2-uniform-shape | Coastal_T_S/5T step 2000 | 已完成 | MASE=1.1694527 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e2-uniform-shape | Coastal_T_S/5T step 4000 | 已完成 | MASE=1.1705558 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e3-crb-mean | Coastal_T_S/5T step 1000 | 已完成 | MASE=1.1678166 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e3-crb-mean | Coastal_T_S/5T step 2000 | 已完成 | MASE=1.1699351 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e3-crb-mean | Coastal_T_S/5T step 4000 | 已完成 | MASE=1.169647 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e4-crb-shape | Coastal_T_S/5T step 1000 | 已完成 | MASE=1.1646619 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e4-crb-shape | Coastal_T_S/5T step 2000 | 已完成 | MASE=1.1655651 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e4-crb-shape | Coastal_T_S/5T step 4000 | 已完成 | MASE=1.1661922 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e5-target-only | Coastal_T_S/5T step 1000 | 已完成 | MASE=1.1480398 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e5-target-only | Coastal_T_S/5T step 2000 | 已完成 | MASE=1.1481779 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e5-target-only | Coastal_T_S/5T step 4000 | 已完成 | MASE=1.1459682 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-5t | e0-rand | Coastal_T_S/5T step 4000 | 已完成 | MASE=1.1694402 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e1-uniform-mean | epf_electricity_price/H step 1000 | 已完成 | MASE=1.0925619 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e1-uniform-mean | epf_electricity_price/H step 2000 | 已完成 | MASE=1.0919749 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e1-uniform-mean | epf_electricity_price/H step 4000 | 已完成 | MASE=1.0909038 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e2-uniform-shape | epf_electricity_price/H step 1000 | 已完成 | MASE=1.0871538 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e2-uniform-shape | epf_electricity_price/H step 2000 | 已完成 | MASE=1.0855195 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e2-uniform-shape | epf_electricity_price/H step 4000 | 已完成 | MASE=1.0846278 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e3-crb-mean | epf_electricity_price/H step 1000 | 已完成 | MASE=1.0943886 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e3-crb-mean | epf_electricity_price/H step 2000 | 已完成 | MASE=1.0901754 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e3-crb-mean | epf_electricity_price/H step 4000 | 已完成 | MASE=1.0884258 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e4-crb-shape | epf_electricity_price/H step 1000 | 已完成 | MASE=1.0894547 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e4-crb-shape | epf_electricity_price/H step 2000 | 已完成 | MASE=1.0875287 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e4-crb-shape | epf_electricity_price/H step 4000 | 已完成 | MASE=1.0864759 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e5-target-only | epf_electricity_price/H step 1000 | 已完成 | MASE=1.0354177 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e5-target-only | epf_electricity_price/H step 2000 | 已完成 | MASE=1.037641 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e5-target-only | epf_electricity_price/H step 4000 | 已完成 | MASE=1.0324299 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-epf-electricity-price-h | e0-rand | epf_electricity_price/H step 4000 | 已完成 | MASE=1.0928438 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e1-uniform-mean | current_velocity/15T step 1000 | 已完成 | MASE=1.2277084 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e1-uniform-mean | current_velocity/15T step 2000 | 已完成 | MASE=1.2246501 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e1-uniform-mean | current_velocity/15T step 4000 | 已完成 | MASE=1.223902 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e2-uniform-shape | current_velocity/15T step 1000 | 已完成 | MASE=1.2207585 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e2-uniform-shape | current_velocity/15T step 2000 | 已完成 | MASE=1.2226013 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e2-uniform-shape | current_velocity/15T step 4000 | 已完成 | MASE=1.2229497 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e3-crb-mean | current_velocity/15T step 1000 | 已完成 | MASE=1.222171 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e3-crb-mean | current_velocity/15T step 2000 | 已完成 | MASE=1.2204368 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e3-crb-mean | current_velocity/15T step 4000 | 已完成 | MASE=1.2201366 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e4-crb-shape | current_velocity/15T step 1000 | 已完成 | MASE=1.2231324 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e4-crb-shape | current_velocity/15T step 2000 | 已完成 | MASE=1.2204818 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e4-crb-shape | current_velocity/15T step 4000 | 已完成 | MASE=1.2201563 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e5-target-only | current_velocity/15T step 1000 | 已完成 | MASE=1.1995337 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e5-target-only | current_velocity/15T step 2000 | 已完成 | MASE=1.1995156 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e5-target-only | current_velocity/15T step 4000 | 已完成 | MASE=1.2000679 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-current-velocity-15t | e0-rand | current_velocity/15T step 4000 | 已完成 | MASE=1.2261465 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e1-uniform-mean | Coastal_T_S/15T step 1000 | 已完成 | MASE=1.4485231 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e1-uniform-mean | Coastal_T_S/15T step 2000 | 已完成 | MASE=1.4496912 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e1-uniform-mean | Coastal_T_S/15T step 4000 | 已完成 | MASE=1.4483123 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e2-uniform-shape | Coastal_T_S/15T step 1000 | 已完成 | MASE=1.4474655 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e2-uniform-shape | Coastal_T_S/15T step 2000 | 已完成 | MASE=1.4482338 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e2-uniform-shape | Coastal_T_S/15T step 4000 | 已完成 | MASE=1.44839 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e3-crb-mean | Coastal_T_S/15T step 1000 | 已完成 | MASE=1.4504322 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e3-crb-mean | Coastal_T_S/15T step 2000 | 已完成 | MASE=1.4512973 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e3-crb-mean | Coastal_T_S/15T step 4000 | 已完成 | MASE=1.450983 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e4-crb-shape | Coastal_T_S/15T step 1000 | 已完成 | MASE=1.4473832 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e4-crb-shape | Coastal_T_S/15T step 2000 | 已完成 | MASE=1.4489008 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e4-crb-shape | Coastal_T_S/15T step 4000 | 已完成 | MASE=1.4496001 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e5-target-only | Coastal_T_S/15T step 1000 | 已完成 | MASE=1.4371879 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e5-target-only | Coastal_T_S/15T step 2000 | 已完成 | MASE=1.4299904 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e5-target-only | Coastal_T_S/15T step 4000 | 已完成 | MASE=1.4298869 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-coastal-t-s-15t | e0-rand | Coastal_T_S/15T step 4000 | 已完成 | MASE=1.4440931 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e1-uniform-mean | azure2019_D/5T step 1000 | 已完成 | MASE=0.81845409 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e1-uniform-mean | azure2019_D/5T step 2000 | 已完成 | MASE=0.82205897 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e1-uniform-mean | azure2019_D/5T step 4000 | 已完成 | MASE=0.82070378 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e2-uniform-shape | azure2019_D/5T step 1000 | 已完成 | MASE=0.81635627 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e2-uniform-shape | azure2019_D/5T step 2000 | 已完成 | MASE=0.81536517 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e2-uniform-shape | azure2019_D/5T step 4000 | 已完成 | MASE=0.81632677 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e3-crb-mean | azure2019_D/5T step 1000 | 已完成 | MASE=0.81708687 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e3-crb-mean | azure2019_D/5T step 2000 | 已完成 | MASE=0.81569188 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e3-crb-mean | azure2019_D/5T step 4000 | 已完成 | MASE=0.81502107 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e4-crb-shape | azure2019_D/5T step 1000 | 已完成 | MASE=0.81323257 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e4-crb-shape | azure2019_D/5T step 2000 | 已完成 | MASE=0.81422809 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e4-crb-shape | azure2019_D/5T step 4000 | 已完成 | MASE=0.8140264 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e5-target-only | azure2019_D/5T step 1000 | 已完成 | MASE=0.81992245 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e5-target-only | azure2019_D/5T step 2000 | 已完成 | MASE=0.81067674 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e5-target-only | azure2019_D/5T step 4000 | 已完成 | MASE=0.81311508 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | time-azure2019-d-5t | e0-rand | azure2019_D/5T step 4000 | 已完成 | MASE=0.84251707 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e1-uniform-mean | 9 step 1000 | 已完成 | MASE=0.72828076 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e1-uniform-mean | 9 step 2000 | 已完成 | MASE=0.72830032 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e1-uniform-mean | 9 step 4000 | 已完成 | MASE=0.72840308 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e2-uniform-shape | 9 step 1000 | 已完成 | MASE=0.7283076 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e2-uniform-shape | 9 step 2000 | 已完成 | MASE=0.73001843 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e2-uniform-shape | 9 step 4000 | 已完成 | MASE=0.72998624 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e3-crb-mean | 9 step 1000 | 已完成 | MASE=0.72886791 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e3-crb-mean | 9 step 2000 | 已完成 | MASE=0.72943119 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e3-crb-mean | 9 step 4000 | 已完成 | MASE=0.72926193 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e4-crb-shape | 9 step 1000 | 已完成 | MASE=0.73029309 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e4-crb-shape | 9 step 2000 | 已完成 | MASE=0.73072175 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e4-crb-shape | 9 step 4000 | 已完成 | MASE=0.73064058 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e5-target-only | 9 step 1000 | 已完成 | MASE=0.72985873 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e5-target-only | 9 step 2000 | 已完成 | MASE=0.72964279 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e5-target-only | 9 step 4000 | 已完成 | MASE=0.72970011 | docs/exp/main-crb-supp/results.md |
| main-crb-supp | fev-hospital-admissions-1d | e0-rand | 9 step 4000 | 已完成 | MASE=0.73436862 | docs/exp/main-crb-supp/results.md |<!-- main-crb-supp end -->

<!-- main-crb-supp-chronos2-small begin -->
| 实验 | 任务 | 方法 | 工作内容 | 状态 | 实际结果 | 详情文档 |
| --- | --- | --- | --- | --- | --- | --- |
| main-crb-supp-chronos2-small | time-crypto-d | e1-uniform-mean | Crypto/D step 1000 | 已完成 | 6.4247198 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e1-uniform-mean | Crypto/D step 2000 | 已完成 | 6.3547506 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e1-uniform-mean | Crypto/D step 4000 | 已完成 | 6.4575414 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e2-uniform-shape | Crypto/D step 1000 | 已完成 | 6.3521239 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e2-uniform-shape | Crypto/D step 2000 | 已完成 | 6.27129 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e2-uniform-shape | Crypto/D step 4000 | 已完成 | 6.2168976 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e3-crb-mean | Crypto/D step 1000 | 已完成 | 6.3327146 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e3-crb-mean | Crypto/D step 2000 | 已完成 | 6.3092514 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e3-crb-mean | Crypto/D step 4000 | 已完成 | 6.3285007 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e4-crb-shape | Crypto/D step 1000 | 已完成 | 6.3515964 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e4-crb-shape | Crypto/D step 2000 | 已完成 | 6.348512 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e4-crb-shape | Crypto/D step 4000 | 已完成 | 6.3532367 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e5-target-only | Crypto/D step 1000 | 已完成 | 8.4366043 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e5-target-only | Crypto/D step 2000 | 已完成 | 9.1679464 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e5-target-only | Crypto/D step 4000 | 已完成 | 9.077976 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-crypto-d | e0-rand | Crypto/D step 4000 | 已完成 | 6.4225716 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e1-uniform-mean | Supply_Chain_Customer/D step 1000 | 已完成 | 0.59567145 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e1-uniform-mean | Supply_Chain_Customer/D step 2000 | 已完成 | 0.59553868 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e1-uniform-mean | Supply_Chain_Customer/D step 4000 | 已完成 | 0.59389875 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e2-uniform-shape | Supply_Chain_Customer/D step 1000 | 已完成 | 0.59509568 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e2-uniform-shape | Supply_Chain_Customer/D step 2000 | 已完成 | 0.5954757 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e2-uniform-shape | Supply_Chain_Customer/D step 4000 | 已完成 | 0.59469807 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e3-crb-mean | Supply_Chain_Customer/D step 1000 | 已完成 | 0.60096194 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e3-crb-mean | Supply_Chain_Customer/D step 2000 | 已完成 | 0.59956536 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e3-crb-mean | Supply_Chain_Customer/D step 4000 | 已完成 | 0.59943984 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e4-crb-shape | Supply_Chain_Customer/D step 1000 | 已完成 | 0.5929224 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e4-crb-shape | Supply_Chain_Customer/D step 2000 | 已完成 | 0.59486266 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e4-crb-shape | Supply_Chain_Customer/D step 4000 | 已完成 | 0.59544241 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e5-target-only | Supply_Chain_Customer/D step 1000 | 已完成 | 0.55843852 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e5-target-only | Supply_Chain_Customer/D step 2000 | 已完成 | 0.55808482 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e5-target-only | Supply_Chain_Customer/D step 4000 | 已完成 | 0.55759119 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-customer-d | e0-rand | Supply_Chain_Customer/D step 4000 | 已完成 | 0.60027546 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e1-uniform-mean | Supply_Chain_Location/D step 1000 | 已完成 | 0.63731117 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e1-uniform-mean | Supply_Chain_Location/D step 2000 | 已完成 | 0.63455555 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e1-uniform-mean | Supply_Chain_Location/D step 4000 | 已完成 | 0.63494699 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e2-uniform-shape | Supply_Chain_Location/D step 1000 | 已完成 | 0.63774681 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e2-uniform-shape | Supply_Chain_Location/D step 2000 | 已完成 | 0.63828352 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e2-uniform-shape | Supply_Chain_Location/D step 4000 | 已完成 | 0.63794585 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e3-crb-mean | Supply_Chain_Location/D step 1000 | 已完成 | 0.63916236 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e3-crb-mean | Supply_Chain_Location/D step 2000 | 已完成 | 0.63960221 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e3-crb-mean | Supply_Chain_Location/D step 4000 | 已完成 | 0.64124491 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e4-crb-shape | Supply_Chain_Location/D step 1000 | 已完成 | 0.63995001 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e4-crb-shape | Supply_Chain_Location/D step 2000 | 已完成 | 0.64151144 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e4-crb-shape | Supply_Chain_Location/D step 4000 | 已完成 | 0.64137265 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e5-target-only | Supply_Chain_Location/D step 1000 | 已完成 | 0.60569417 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e5-target-only | Supply_Chain_Location/D step 2000 | 已完成 | 0.60618063 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e5-target-only | Supply_Chain_Location/D step 4000 | 已完成 | 0.60574608 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-supply-chain-location-d | e0-rand | Supply_Chain_Location/D step 4000 | 已完成 | 0.64587767 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e1-uniform-mean | NE_China_Wind/H step 1000 | 已完成 | 0.54959781 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e1-uniform-mean | NE_China_Wind/H step 2000 | 已完成 | 0.54236627 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e1-uniform-mean | NE_China_Wind/H step 4000 | 已完成 | 0.54279854 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e2-uniform-shape | NE_China_Wind/H step 1000 | 已完成 | 0.54474442 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e2-uniform-shape | NE_China_Wind/H step 2000 | 已完成 | 0.54341208 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e2-uniform-shape | NE_China_Wind/H step 4000 | 已完成 | 0.54004663 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e3-crb-mean | NE_China_Wind/H step 1000 | 已完成 | 0.54016656 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e3-crb-mean | NE_China_Wind/H step 2000 | 已完成 | 0.53743632 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e3-crb-mean | NE_China_Wind/H step 4000 | 已完成 | 0.53936355 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e4-crb-shape | NE_China_Wind/H step 1000 | 已完成 | 0.53621029 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e4-crb-shape | NE_China_Wind/H step 2000 | 已完成 | 0.53805679 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e4-crb-shape | NE_China_Wind/H step 4000 | 已完成 | 0.53846172 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e5-target-only | NE_China_Wind/H step 1000 | 已完成 | 0.54654106 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e5-target-only | NE_China_Wind/H step 2000 | 已完成 | 0.56638424 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e5-target-only | NE_China_Wind/H step 4000 | 已完成 | 0.57908021 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-ne-china-wind-h | e0-rand | NE_China_Wind/H step 4000 | 已完成 | 0.53777035 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e1-uniform-mean | Coastal_T_S/20T step 1000 | 已完成 | 0.49321151 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e1-uniform-mean | Coastal_T_S/20T step 2000 | 已完成 | 0.50402238 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e1-uniform-mean | Coastal_T_S/20T step 4000 | 已完成 | 0.50027053 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e2-uniform-shape | Coastal_T_S/20T step 1000 | 已完成 | 0.509063 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e2-uniform-shape | Coastal_T_S/20T step 2000 | 已完成 | 0.50488781 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e2-uniform-shape | Coastal_T_S/20T step 4000 | 已完成 | 0.5042597 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e3-crb-mean | Coastal_T_S/20T step 1000 | 已完成 | 0.48817858 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e3-crb-mean | Coastal_T_S/20T step 2000 | 已完成 | 0.49385368 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e3-crb-mean | Coastal_T_S/20T step 4000 | 已完成 | 0.49638108 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e4-crb-shape | Coastal_T_S/20T step 1000 | 已完成 | 0.50998544 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e4-crb-shape | Coastal_T_S/20T step 2000 | 已完成 | 0.50190911 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e4-crb-shape | Coastal_T_S/20T step 4000 | 已完成 | 0.50409993 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e5-target-only | Coastal_T_S/20T step 1000 | 已完成 | 0.46577123 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e5-target-only | Coastal_T_S/20T step 2000 | 已完成 | 0.47388875 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e5-target-only | Coastal_T_S/20T step 4000 | 已完成 | 0.47026378 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-20t | e0-rand | Coastal_T_S/20T step 4000 | 已完成 | 0.51150699 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e1-uniform-mean | Australia_Solar/H step 1000 | 已完成 | 0.61909923 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e1-uniform-mean | Australia_Solar/H step 2000 | 已完成 | 0.6130007 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e1-uniform-mean | Australia_Solar/H step 4000 | 已完成 | 0.61419723 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e2-uniform-shape | Australia_Solar/H step 1000 | 已完成 | 0.64828435 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e2-uniform-shape | Australia_Solar/H step 2000 | 已完成 | 0.64470949 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e2-uniform-shape | Australia_Solar/H step 4000 | 已完成 | 0.64765744 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e3-crb-mean | Australia_Solar/H step 1000 | 已完成 | 0.61362447 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e3-crb-mean | Australia_Solar/H step 2000 | 已完成 | 0.60854241 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e3-crb-mean | Australia_Solar/H step 4000 | 已完成 | 0.60890189 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e4-crb-shape | Australia_Solar/H step 1000 | 已完成 | 0.6241359 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e4-crb-shape | Australia_Solar/H step 2000 | 已完成 | 0.62942803 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e4-crb-shape | Australia_Solar/H step 4000 | 已完成 | 0.63055845 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e5-target-only | Australia_Solar/H step 1000 | 已完成 | 0.56324721 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e5-target-only | Australia_Solar/H step 2000 | 已完成 | 0.56792377 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e5-target-only | Australia_Solar/H step 4000 | 已完成 | 0.56432571 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-australia-solar-h | e0-rand | Australia_Solar/H step 4000 | 已完成 | 0.62924401 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e1-uniform-mean | Finland_Traffic/15T step 1000 | 已完成 | 0.58984123 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e1-uniform-mean | Finland_Traffic/15T step 2000 | 已完成 | 0.58539089 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e1-uniform-mean | Finland_Traffic/15T step 4000 | 已完成 | 0.58273222 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e2-uniform-shape | Finland_Traffic/15T step 1000 | 已完成 | 0.5817024 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e2-uniform-shape | Finland_Traffic/15T step 2000 | 已完成 | 0.58084026 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e2-uniform-shape | Finland_Traffic/15T step 4000 | 已完成 | 0.57977309 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e3-crb-mean | Finland_Traffic/15T step 1000 | 已完成 | 0.57502486 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e3-crb-mean | Finland_Traffic/15T step 2000 | 已完成 | 0.57630811 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e3-crb-mean | Finland_Traffic/15T step 4000 | 已完成 | 0.577112 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e4-crb-shape | Finland_Traffic/15T step 1000 | 已完成 | 0.57033482 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e4-crb-shape | Finland_Traffic/15T step 2000 | 已完成 | 0.57002112 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e4-crb-shape | Finland_Traffic/15T step 4000 | 已完成 | 0.57406178 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e5-target-only | Finland_Traffic/15T step 1000 | 已完成 | 0.42152716 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e5-target-only | Finland_Traffic/15T step 2000 | 已完成 | 0.42100585 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e5-target-only | Finland_Traffic/15T step 4000 | 已完成 | 0.41867829 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-finland-traffic-15t | e0-rand | Finland_Traffic/15T step 4000 | 已完成 | 0.62328744 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e1-uniform-mean | 10 step 1000 | 已完成 | 0.46355797 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e1-uniform-mean | 10 step 2000 | 已完成 | 0.46270436 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e1-uniform-mean | 10 step 4000 | 已完成 | 0.4612691 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e2-uniform-shape | 10 step 1000 | 已完成 | 0.46537595 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e2-uniform-shape | 10 step 2000 | 已完成 | 0.46431737 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e2-uniform-shape | 10 step 4000 | 已完成 | 0.4630668 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e3-crb-mean | 10 step 1000 | 已完成 | 0.46129283 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e3-crb-mean | 10 step 2000 | 已完成 | 0.46088394 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e3-crb-mean | 10 step 4000 | 已完成 | 0.45974854 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e4-crb-shape | 10 step 1000 | 已完成 | 0.46103015 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e4-crb-shape | 10 step 2000 | 已完成 | 0.45782019 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e4-crb-shape | 10 step 4000 | 已完成 | 0.45705045 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e5-target-only | 10 step 1000 | 已完成 | 0.4572786 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e5-target-only | 10 step 2000 | 已完成 | 0.45878745 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e5-target-only | 10 step 4000 | 已完成 | 0.4598131 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-jena-weather-1h | e0-rand | 10 step 4000 | 已完成 | 0.45575576 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e1-uniform-mean | 6 step 1000 | 已完成 | 1.2006858 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e1-uniform-mean | 6 step 2000 | 已完成 | 1.200551 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e1-uniform-mean | 6 step 4000 | 已完成 | 1.2025657 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e2-uniform-shape | 6 step 1000 | 已完成 | 1.178164 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e2-uniform-shape | 6 step 2000 | 已完成 | 1.1799629 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e2-uniform-shape | 6 step 4000 | 已完成 | 1.1800971 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e3-crb-mean | 6 step 1000 | 已完成 | 1.1796145 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e3-crb-mean | 6 step 2000 | 已完成 | 1.1776384 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e3-crb-mean | 6 step 4000 | 已完成 | 1.1803268 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e4-crb-shape | 6 step 1000 | 已完成 | 1.1604211 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e4-crb-shape | 6 step 2000 | 已完成 | 1.1565015 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e4-crb-shape | 6 step 4000 | 已完成 | 1.1574633 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e5-target-only | 6 step 1000 | 已完成 | 1.1484972 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e5-target-only | 6 step 2000 | 已完成 | 1.1584205 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e5-target-only | 6 step 4000 | 已完成 | 1.1616075 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-ett-1h | e0-rand | 6 step 4000 | 已完成 | 1.1849221 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e1-uniform-mean | MetroPT-3/5T step 1000 | 已完成 | 0.63682605 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e1-uniform-mean | MetroPT-3/5T step 2000 | 已完成 | 0.64095032 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e1-uniform-mean | MetroPT-3/5T step 4000 | 已完成 | 0.64178627 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e2-uniform-shape | MetroPT-3/5T step 1000 | 已完成 | 0.63923465 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e2-uniform-shape | MetroPT-3/5T step 2000 | 已完成 | 0.64585119 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e2-uniform-shape | MetroPT-3/5T step 4000 | 已完成 | 0.64321192 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e3-crb-mean | MetroPT-3/5T step 1000 | 已完成 | 0.63779466 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e3-crb-mean | MetroPT-3/5T step 2000 | 已完成 | 0.64019412 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e3-crb-mean | MetroPT-3/5T step 4000 | 已完成 | 0.64124686 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e4-crb-shape | MetroPT-3/5T step 1000 | 已完成 | 0.63828503 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e4-crb-shape | MetroPT-3/5T step 2000 | 已完成 | 0.63928121 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e4-crb-shape | MetroPT-3/5T step 4000 | 已完成 | 0.64061675 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e5-target-only | MetroPT-3/5T step 1000 | 已完成 | 0.62654827 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e5-target-only | MetroPT-3/5T step 2000 | 已完成 | 0.62525623 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e5-target-only | MetroPT-3/5T step 4000 | 已完成 | 0.62838809 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-metropt-3-5t | e0-rand | MetroPT-3/5T step 4000 | 已完成 | 0.63957261 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e1-uniform-mean | current_velocity/5T step 1000 | 已完成 | 0.39186633 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e1-uniform-mean | current_velocity/5T step 2000 | 已完成 | 0.39450499 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e1-uniform-mean | current_velocity/5T step 4000 | 已完成 | 0.39634216 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e2-uniform-shape | current_velocity/5T step 1000 | 已完成 | 0.39681954 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e2-uniform-shape | current_velocity/5T step 2000 | 已完成 | 0.4022894 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e2-uniform-shape | current_velocity/5T step 4000 | 已完成 | 0.39937639 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e3-crb-mean | current_velocity/5T step 1000 | 已完成 | 0.39898824 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e3-crb-mean | current_velocity/5T step 2000 | 已完成 | 0.40041434 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e3-crb-mean | current_velocity/5T step 4000 | 已完成 | 0.40075778 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e4-crb-shape | current_velocity/5T step 1000 | 已完成 | 0.39657161 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e4-crb-shape | current_velocity/5T step 2000 | 已完成 | 0.40117838 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e4-crb-shape | current_velocity/5T step 4000 | 已完成 | 0.39852845 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e5-target-only | current_velocity/5T step 1000 | 已完成 | 0.37485843 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e5-target-only | current_velocity/5T step 2000 | 已完成 | 0.37586008 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e5-target-only | current_velocity/5T step 4000 | 已完成 | 0.37545648 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-5t | e0-rand | current_velocity/5T step 4000 | 已完成 | 0.4037078 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e1-uniform-mean | azure2019_U/5T step 1000 | 已完成 | 0.70050727 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e1-uniform-mean | azure2019_U/5T step 2000 | 已完成 | 0.70100389 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e1-uniform-mean | azure2019_U/5T step 4000 | 已完成 | 0.69779581 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e2-uniform-shape | azure2019_U/5T step 1000 | 已完成 | 0.71983281 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e2-uniform-shape | azure2019_U/5T step 2000 | 已完成 | 0.70981301 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e2-uniform-shape | azure2019_U/5T step 4000 | 已完成 | 0.7091312 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e3-crb-mean | azure2019_U/5T step 1000 | 已完成 | 0.70026006 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e3-crb-mean | azure2019_U/5T step 2000 | 已完成 | 0.6973181 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e3-crb-mean | azure2019_U/5T step 4000 | 已完成 | 0.69805987 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e4-crb-shape | azure2019_U/5T step 1000 | 已完成 | 0.70062404 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e4-crb-shape | azure2019_U/5T step 2000 | 已完成 | 0.70070167 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e4-crb-shape | azure2019_U/5T step 4000 | 已完成 | 0.69627658 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e5-target-only | azure2019_U/5T step 1000 | 已完成 | 0.68207516 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e5-target-only | azure2019_U/5T step 2000 | 已完成 | 0.68834698 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e5-target-only | azure2019_U/5T step 4000 | 已完成 | 0.68825149 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-u-5t | e0-rand | azure2019_U/5T step 4000 | 已完成 | 0.69220533 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e1-uniform-mean | SG_PM25/H step 1000 | 已完成 | 0.89885847 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e1-uniform-mean | SG_PM25/H step 2000 | 已完成 | 0.89878751 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e1-uniform-mean | SG_PM25/H step 4000 | 已完成 | 0.89928866 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e2-uniform-shape | SG_PM25/H step 1000 | 已完成 | 0.90528587 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e2-uniform-shape | SG_PM25/H step 2000 | 已完成 | 0.90563874 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e2-uniform-shape | SG_PM25/H step 4000 | 已完成 | 0.90332064 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e3-crb-mean | SG_PM25/H step 1000 | 已完成 | 0.89719761 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e3-crb-mean | SG_PM25/H step 2000 | 已完成 | 0.89451224 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e3-crb-mean | SG_PM25/H step 4000 | 已完成 | 0.89436181 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e4-crb-shape | SG_PM25/H step 1000 | 已完成 | 0.89551636 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e4-crb-shape | SG_PM25/H step 2000 | 已完成 | 0.89697693 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e4-crb-shape | SG_PM25/H step 4000 | 已完成 | 0.89680527 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e5-target-only | SG_PM25/H step 1000 | 已完成 | 0.91049609 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e5-target-only | SG_PM25/H step 2000 | 已完成 | 0.91471422 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e5-target-only | SG_PM25/H step 4000 | 已完成 | 0.91680868 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-sg-pm25-h | e0-rand | SG_PM25/H step 4000 | 已完成 | 0.89971601 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e1-uniform-mean | Water_Quality_Darwin/15T step 1000 | 已完成 | 0.63784421 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e1-uniform-mean | Water_Quality_Darwin/15T step 2000 | 已完成 | 0.64460581 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e1-uniform-mean | Water_Quality_Darwin/15T step 4000 | 已完成 | 0.64188386 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e2-uniform-shape | Water_Quality_Darwin/15T step 1000 | 已完成 | 0.63669811 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e2-uniform-shape | Water_Quality_Darwin/15T step 2000 | 已完成 | 0.64732504 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e2-uniform-shape | Water_Quality_Darwin/15T step 4000 | 已完成 | 0.63880471 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e3-crb-mean | Water_Quality_Darwin/15T step 1000 | 已完成 | 0.64311549 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e3-crb-mean | Water_Quality_Darwin/15T step 2000 | 已完成 | 0.63643032 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e3-crb-mean | Water_Quality_Darwin/15T step 4000 | 已完成 | 0.6371905 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e4-crb-shape | Water_Quality_Darwin/15T step 1000 | 已完成 | 0.63387308 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e4-crb-shape | Water_Quality_Darwin/15T step 2000 | 已完成 | 0.63675231 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e4-crb-shape | Water_Quality_Darwin/15T step 4000 | 已完成 | 0.63402641 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e5-target-only | Water_Quality_Darwin/15T step 1000 | 已完成 | 0.57156388 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e5-target-only | Water_Quality_Darwin/15T step 2000 | 已完成 | 0.56834441 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e5-target-only | Water_Quality_Darwin/15T step 4000 | 已完成 | 0.56772962 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-water-quality-darwin-15t | e0-rand | Water_Quality_Darwin/15T step 4000 | 已完成 | 0.66121009 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e1-uniform-mean | OpenElectricity_NEM/5T step 1000 | 已完成 | 0.47204254 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e1-uniform-mean | OpenElectricity_NEM/5T step 2000 | 已完成 | 0.47206154 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e1-uniform-mean | OpenElectricity_NEM/5T step 4000 | 已完成 | 0.47089214 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e2-uniform-shape | OpenElectricity_NEM/5T step 1000 | 已完成 | 0.46484081 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e2-uniform-shape | OpenElectricity_NEM/5T step 2000 | 已完成 | 0.46560717 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e2-uniform-shape | OpenElectricity_NEM/5T step 4000 | 已完成 | 0.46490813 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e3-crb-mean | OpenElectricity_NEM/5T step 1000 | 已完成 | 0.46240489 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e3-crb-mean | OpenElectricity_NEM/5T step 2000 | 已完成 | 0.46298739 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e3-crb-mean | OpenElectricity_NEM/5T step 4000 | 已完成 | 0.46350187 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e4-crb-shape | OpenElectricity_NEM/5T step 1000 | 已完成 | 0.46463228 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e4-crb-shape | OpenElectricity_NEM/5T step 2000 | 已完成 | 0.46193916 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e4-crb-shape | OpenElectricity_NEM/5T step 4000 | 已完成 | 0.46138651 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e5-target-only | OpenElectricity_NEM/5T step 1000 | 已完成 | 0.42667034 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e5-target-only | OpenElectricity_NEM/5T step 2000 | 已完成 | 0.42185615 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e5-target-only | OpenElectricity_NEM/5T step 4000 | 已完成 | 0.42047993 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-open-electricity-nem-5t | e0-rand | OpenElectricity_NEM/5T step 4000 | 已完成 | 0.46808756 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e1-uniform-mean | EWELD_Load/15T step 1000 | 已完成 | 0.605694 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e1-uniform-mean | EWELD_Load/15T step 2000 | 已完成 | 0.60732579 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e1-uniform-mean | EWELD_Load/15T step 4000 | 已完成 | 0.60768664 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e2-uniform-shape | EWELD_Load/15T step 1000 | 已完成 | 0.60912613 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e2-uniform-shape | EWELD_Load/15T step 2000 | 已完成 | 0.60869414 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e2-uniform-shape | EWELD_Load/15T step 4000 | 已完成 | 0.60885765 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e3-crb-mean | EWELD_Load/15T step 1000 | 已完成 | 0.60543893 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e3-crb-mean | EWELD_Load/15T step 2000 | 已完成 | 0.60781801 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e3-crb-mean | EWELD_Load/15T step 4000 | 已完成 | 0.6061019 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e4-crb-shape | EWELD_Load/15T step 1000 | 已完成 | 0.60734878 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e4-crb-shape | EWELD_Load/15T step 2000 | 已完成 | 0.61357402 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e4-crb-shape | EWELD_Load/15T step 4000 | 已完成 | 0.61126595 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e5-target-only | EWELD_Load/15T step 1000 | 已完成 | 0.58131058 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e5-target-only | EWELD_Load/15T step 2000 | 已完成 | 0.57773039 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e5-target-only | EWELD_Load/15T step 4000 | 已完成 | 0.57352242 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-eweld-load-15t | e0-rand | EWELD_Load/15T step 4000 | 已完成 | 0.59962494 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e1-uniform-mean | Port_Activity/D step 1000 | 已完成 | 0.66012632 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e1-uniform-mean | Port_Activity/D step 2000 | 已完成 | 0.66043247 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e1-uniform-mean | Port_Activity/D step 4000 | 已完成 | 0.66028991 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e2-uniform-shape | Port_Activity/D step 1000 | 已完成 | 0.65985714 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e2-uniform-shape | Port_Activity/D step 2000 | 已完成 | 0.65983256 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e2-uniform-shape | Port_Activity/D step 4000 | 已完成 | 0.65990575 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e3-crb-mean | Port_Activity/D step 1000 | 已完成 | 0.66085756 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e3-crb-mean | Port_Activity/D step 2000 | 已完成 | 0.66075101 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e3-crb-mean | Port_Activity/D step 4000 | 已完成 | 0.66085586 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e4-crb-shape | Port_Activity/D step 1000 | 已完成 | 0.66042588 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e4-crb-shape | Port_Activity/D step 2000 | 已完成 | 0.66059253 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e4-crb-shape | Port_Activity/D step 4000 | 已完成 | 0.66051956 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e5-target-only | Port_Activity/D step 1000 | 已完成 | 0.6582136 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e5-target-only | Port_Activity/D step 2000 | 已完成 | 0.65807231 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e5-target-only | Port_Activity/D step 4000 | 已完成 | 0.65816154 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-port-activity-d | e0-rand | Port_Activity/D step 4000 | 已完成 | 0.6616978 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e1-uniform-mean | current_velocity/10T step 1000 | 已完成 | 0.58211397 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e1-uniform-mean | current_velocity/10T step 2000 | 已完成 | 0.58237804 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e1-uniform-mean | current_velocity/10T step 4000 | 已完成 | 0.57942059 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e2-uniform-shape | current_velocity/10T step 1000 | 已完成 | 0.58610066 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e2-uniform-shape | current_velocity/10T step 2000 | 已完成 | 0.58447266 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e2-uniform-shape | current_velocity/10T step 4000 | 已完成 | 0.58520393 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e3-crb-mean | current_velocity/10T step 1000 | 已完成 | 0.57571661 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e3-crb-mean | current_velocity/10T step 2000 | 已完成 | 0.57840473 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e3-crb-mean | current_velocity/10T step 4000 | 已完成 | 0.58082226 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e4-crb-shape | current_velocity/10T step 1000 | 已完成 | 0.58064337 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e4-crb-shape | current_velocity/10T step 2000 | 已完成 | 0.58097661 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e4-crb-shape | current_velocity/10T step 4000 | 已完成 | 0.5811232 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e5-target-only | current_velocity/10T step 1000 | 已完成 | 0.52797498 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e5-target-only | current_velocity/10T step 2000 | 已完成 | 0.5258639 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e5-target-only | current_velocity/10T step 4000 | 已完成 | 0.52547508 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-10t | e0-rand | current_velocity/10T step 4000 | 已完成 | 0.59431773 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e1-uniform-mean | Coastal_T_S/5T step 1000 | 已完成 | 0.79917322 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e1-uniform-mean | Coastal_T_S/5T step 2000 | 已完成 | 0.79946305 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e1-uniform-mean | Coastal_T_S/5T step 4000 | 已完成 | 0.80070534 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e2-uniform-shape | Coastal_T_S/5T step 1000 | 已完成 | 0.80054994 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e2-uniform-shape | Coastal_T_S/5T step 2000 | 已完成 | 0.80100444 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e2-uniform-shape | Coastal_T_S/5T step 4000 | 已完成 | 0.80162311 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e3-crb-mean | Coastal_T_S/5T step 1000 | 已完成 | 0.80131602 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e3-crb-mean | Coastal_T_S/5T step 2000 | 已完成 | 0.80011778 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e3-crb-mean | Coastal_T_S/5T step 4000 | 已完成 | 0.79989182 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e4-crb-shape | Coastal_T_S/5T step 1000 | 已完成 | 0.79485654 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e4-crb-shape | Coastal_T_S/5T step 2000 | 已完成 | 0.79736156 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e4-crb-shape | Coastal_T_S/5T step 4000 | 已完成 | 0.79768797 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e5-target-only | Coastal_T_S/5T step 1000 | 已完成 | 0.78476747 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e5-target-only | Coastal_T_S/5T step 2000 | 已完成 | 0.78298905 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e5-target-only | Coastal_T_S/5T step 4000 | 已完成 | 0.78259057 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-5t | e0-rand | Coastal_T_S/5T step 4000 | 已完成 | 0.80357517 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e1-uniform-mean | epf_electricity_price/H step 1000 | 已完成 | 0.99030755 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e1-uniform-mean | epf_electricity_price/H step 2000 | 已完成 | 0.99667382 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e1-uniform-mean | epf_electricity_price/H step 4000 | 已完成 | 0.99706204 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e2-uniform-shape | epf_electricity_price/H step 1000 | 已完成 | 0.9914875 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e2-uniform-shape | epf_electricity_price/H step 2000 | 已完成 | 0.98994395 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e2-uniform-shape | epf_electricity_price/H step 4000 | 已完成 | 0.9909273 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e3-crb-mean | epf_electricity_price/H step 1000 | 已完成 | 1.0034269 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e3-crb-mean | epf_electricity_price/H step 2000 | 已完成 | 1.0044857 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e3-crb-mean | epf_electricity_price/H step 4000 | 已完成 | 1.0038417 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e4-crb-shape | epf_electricity_price/H step 1000 | 已完成 | 0.98437546 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e4-crb-shape | epf_electricity_price/H step 2000 | 已完成 | 0.98645284 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e4-crb-shape | epf_electricity_price/H step 4000 | 已完成 | 0.98773536 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e5-target-only | epf_electricity_price/H step 1000 | 已完成 | 0.95131603 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e5-target-only | epf_electricity_price/H step 2000 | 已完成 | 0.95166485 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e5-target-only | epf_electricity_price/H step 4000 | 已完成 | 0.94881043 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-epf-electricity-price-h | e0-rand | epf_electricity_price/H step 4000 | 已完成 | 0.98843537 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e1-uniform-mean | current_velocity/15T step 1000 | 已完成 | 0.53692924 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e1-uniform-mean | current_velocity/15T step 2000 | 已完成 | 0.53782091 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e1-uniform-mean | current_velocity/15T step 4000 | 已完成 | 0.53589205 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e2-uniform-shape | current_velocity/15T step 1000 | 已完成 | 0.52858007 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e2-uniform-shape | current_velocity/15T step 2000 | 已完成 | 0.53200074 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e2-uniform-shape | current_velocity/15T step 4000 | 已完成 | 0.53127522 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e3-crb-mean | current_velocity/15T step 1000 | 已完成 | 0.53467854 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e3-crb-mean | current_velocity/15T step 2000 | 已完成 | 0.52935421 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e3-crb-mean | current_velocity/15T step 4000 | 已完成 | 0.52852517 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e4-crb-shape | current_velocity/15T step 1000 | 已完成 | 0.53106671 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e4-crb-shape | current_velocity/15T step 2000 | 已完成 | 0.53114518 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e4-crb-shape | current_velocity/15T step 4000 | 已完成 | 0.52982704 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e5-target-only | current_velocity/15T step 1000 | 已完成 | 0.51284469 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e5-target-only | current_velocity/15T step 2000 | 已完成 | 0.51103282 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e5-target-only | current_velocity/15T step 4000 | 已完成 | 0.51164517 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-current-velocity-15t | e0-rand | current_velocity/15T step 4000 | 已完成 | 0.53860805 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e1-uniform-mean | Coastal_T_S/15T step 1000 | 已完成 | 1.1363127 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e1-uniform-mean | Coastal_T_S/15T step 2000 | 已完成 | 1.1346062 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e1-uniform-mean | Coastal_T_S/15T step 4000 | 已完成 | 1.1352501 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e2-uniform-shape | Coastal_T_S/15T step 1000 | 已完成 | 1.133941 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e2-uniform-shape | Coastal_T_S/15T step 2000 | 已完成 | 1.1347056 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e2-uniform-shape | Coastal_T_S/15T step 4000 | 已完成 | 1.135137 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e3-crb-mean | Coastal_T_S/15T step 1000 | 已完成 | 1.1371288 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e3-crb-mean | Coastal_T_S/15T step 2000 | 已完成 | 1.1359713 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e3-crb-mean | Coastal_T_S/15T step 4000 | 已完成 | 1.1370498 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e4-crb-shape | Coastal_T_S/15T step 1000 | 已完成 | 1.1333853 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e4-crb-shape | Coastal_T_S/15T step 2000 | 已完成 | 1.1348856 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e4-crb-shape | Coastal_T_S/15T step 4000 | 已完成 | 1.1356063 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e5-target-only | Coastal_T_S/15T step 1000 | 已完成 | 1.1344577 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e5-target-only | Coastal_T_S/15T step 2000 | 已完成 | 1.1343915 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e5-target-only | Coastal_T_S/15T step 4000 | 已完成 | 1.1339115 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-coastal-t-s-15t | e0-rand | Coastal_T_S/15T step 4000 | 已完成 | 1.1347863 | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e1-uniform-mean | azure2019_D/5T step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e1-uniform-mean | azure2019_D/5T step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e1-uniform-mean | azure2019_D/5T step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e2-uniform-shape | azure2019_D/5T step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e2-uniform-shape | azure2019_D/5T step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e2-uniform-shape | azure2019_D/5T step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e3-crb-mean | azure2019_D/5T step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e3-crb-mean | azure2019_D/5T step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e3-crb-mean | azure2019_D/5T step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e4-crb-shape | azure2019_D/5T step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e4-crb-shape | azure2019_D/5T step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e4-crb-shape | azure2019_D/5T step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e5-target-only | azure2019_D/5T step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e5-target-only | azure2019_D/5T step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e5-target-only | azure2019_D/5T step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | time-azure2019-d-5t | e0-rand | azure2019_D/5T step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e1-uniform-mean | 9 step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e1-uniform-mean | 9 step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e1-uniform-mean | 9 step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e2-uniform-shape | 9 step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e2-uniform-shape | 9 step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e2-uniform-shape | 9 step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e3-crb-mean | 9 step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e3-crb-mean | 9 step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e3-crb-mean | 9 step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e4-crb-shape | 9 step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e4-crb-shape | 9 step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e4-crb-shape | 9 step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e5-target-only | 9 step 1000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e5-target-only | 9 step 2000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e5-target-only | 9 step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
| main-crb-supp-chronos2-small | fev-hospital-admissions-1d | e0-rand | 9 step 4000 | 未开始 | - | docs/exp/main-crb-supp-chronos2-small/results.md |
<!-- main-crb-supp-chronos2-small end -->
