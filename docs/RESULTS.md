# 实验结果总览

最后修改日期：2026-11-13

本文件逐行记录全部实验与批次的工作、状态和实际结果，绝不编造或替换指标；
数值以实际生成的产物、完成标记和日志为依据。逐实验完整报告见 `docs/results/<实验代号>.md`。

## 结果报告索引

| 实验代号 | 报告 |
|---|---|
| `64-large-var-k-error-composition` | [results/64-large-var-k-error-composition.md](exp/v0-crb-mech-test/64-large-var-k-error-composition.md) |
| `64-large-var-k-error-composition-full-tables` | [results/64-large-var-k-error-composition-full-tables.md](exp/v0-crb-mech-test/64-large-var-k-error-composition-full-tables.md) |
| `src-8192-source-coverage` | [results/src-8192-source-coverage.md](exp/src-8192-source-coverage/results.md) |
| `src-96-q4-output-head-estimate` | [results/src-96-q4-output-head-estimate.md](exp/src-96-q4-output-head-estimate/results.md) |
| `v0-active-value-rank` | [results/v0-active-value-rank.md](exp/v0-active-value-rank/results.md) |
| `v0-active-value-rank-diagnostics` | [results/v0-active-value-rank-diagnostics.md](exp/v0-active-value-rank/v0-active-value-rank-diagnostics.md) |
| `v0-chronostiny-timerxl-20261112` | [results/v0-chronostiny-timerxl-20261112.md](exp/v0-chronostiny-timerxl-20261112/results.md) |
| `v0-consolidated-summary-20261112` | [results/v0-consolidated-summary-20261112.md](exp/v0-consolidated-summary-20261112/results.md) |
| `v0-crb-cos-var` | [results/v0-crb-cos-var.md](exp/v0-crb-cos-var/results.md) |
| `v0-crb-final-test-timerXL-64tr` | [results/v0-crb-final-test-timerXL-64tr.md](exp/v0-crb-final-test-timerXL-64tr/results.md) |
| `v0-crb-final-test-timerXL-64tr-spacing-q-dir` | [results/v0-crb-final-test-timerXL-64tr-spacing-q-dir.md](exp/v0-crb-final-test-timerXL-64tr/v0-crb-final-test-timerXL-64tr-spacing-q-dir.md) |
| `v0-crb-mech-test` | [results/v0-crb-mech-test.md](exp/v0-crb-mech-test/results.md) |
| `v0-crb-reconst-fine` | [results/v0-crb-reconst-fine.md](exp/v0-crb-reconst-fine/results.md) |
| `v0-crb-reconst-test` | [results/v0-crb-reconst-test.md](exp/v0-crb-reconst-test/results.md) |
| `v0-crb-var` | [results/v0-crb-var.md](exp/v0-crb-var/results.md) |
| `v0-directional-marginal-gain` | [results/v0-directional-marginal-gain.md](exp/v0-directional-marginal-gain/results.md) |
| `v0-grad-mech-l2-norm` | [results/v0-grad-mech-l2-norm.md](exp/v0-grad-mech-l2-norm/results.md) |
| `v0-grad-mech-test` | [results/v0-grad-mech-test.md](exp/v0-grad-mech-test/results.md) |
| `v0-grad-prepare-src-96tr-q16-4096d` | [results/v0-grad-prepare-src-96tr-q16-4096d.md](exp/v0-grad-prepare-src-96tr-q16-4096d/results.md) |
| `v0-grad-traj-level-norm-gap` | [results/v0-grad-traj-level-norm-gap.md](exp/v0-grad-traj-level-norm-gap/results.md) |
| `v0-gradient-proxy-cost` | [results/v0-gradient-proxy-cost.md](exp/v0-gradient-proxy-cost/results.md) |
| `v0-gradient-proxy-test` | [results/v0-gradient-proxy-test.md](exp/v0-gradient-proxy-test/results.md) |
| `v0-gradient-variance-decomposition` | [results/v0-gradient-variance-decomposition.md](exp/v0-gradient-variance-decomposition/results.md) |
| `v0-proxy-dp-segmentation` | [results/v0-proxy-dp-segmentation.md](exp/v0-proxy-dp-segmentation/results.md) |
| `v0-timer-XL-grad-4-change-velocity` | [results/v0-timer-XL-grad-4-change-velocity.md](exp/v0-timer-XL-grad-4-change-velocity/results.md) |
| `v0-timer-XL-grad-4-curvature` | [results/v0-timer-XL-grad-4-curvature.md](exp/v0-timer-XL-grad-4-curvature/results.md) |
| `v0-timer-XL-grad-4-two-random-16-blocks` | [results/v0-timer-XL-grad-4-two-random-16-blocks.md](exp/v0-timer-XL-grad-4-change-velocity/v0-timer-XL-grad-4-two-random-16-blocks.md) |
| `v0-timer-XL-grad-64-change-velocity` | [results/v0-timer-XL-grad-64-change-velocity.md](exp/v0-timer-XL-grad-64-change-velocity/results.md) |
| `v0-timerXL-grad-4-lm-head` | [results/v0-timerXL-grad-4-lm-head.md](exp/v0-timerXL-grad-4-lm-head/results.md) |
| `v0-timerXL-grad-4-lm-head-change-velocity` | [results/v0-timerXL-grad-4-lm-head-change-velocity.md](exp/v0-timerXL-grad-4-lm-head/v0-timerXL-grad-4-lm-head-change-velocity.md) |
| `v0-timerxl-grad64-pca` | [results/v0-timerxl-grad64-pca.md](exp/v0-timer-XL-grad-64-change-velocity/v0-timerxl-grad64-pca.md) |
| `v0-uni-data-evaluation` | [results/v0-uni-data-evaluation.md](exp/v0-uni-data-evaluation/results.md) |
| `v0-uni-reconst-scale` | [results/v0-uni-reconst-scale.md](exp/v0-uni-reconst-scale/results.md) |
| `v0-vage` | [results/v0-vage.md](exp/v0-vage/results.md) |
| `v0-var-proxy-estimate` | [results/v0-var-proxy-estimate.md](exp/v0-var-proxy-estimate/results.md) |
| `v1-computational-effort-estimate` | [results/v1-computational-effort-estimate.md](exp/v1-computational-effort-estimate/results.md) |
| `v1-crb-redo` | [results/v1-crb-redo.md](exp/v1-crb-redo/results.md) |
| `v1-rage-trev-seed0-final` | [results/v1-rage-trev-seed0-final.md](exp/v1-rage-trev/v1-rage-trev-seed0-final.md) |
| `v1-rage-trev-timerxl-seed0` | [results/v1-rage-trev-timerxl-seed0.md](exp/v1-rage-trev/v1-rage-trev-timerxl-seed0.md) |
| `v1-rage-trev-timesfm-seed0` | [results/v1-rage-trev-timesfm-seed0.md](exp/v1-rage-trev/v1-rage-trev-timesfm-seed0.md) |
| `v2-foundation-expansion` | [results/v2-foundation-expansion.md](exp/v2-foundation-expansion/results.md) |

## 批次结果总表

| 实验代号 | 批次 | 工作 | 状态 | 实际结果 |
|---|---|---|---|---|
| v2-foundation-expansion | Chronos-2、Bolt、Moirai 2.0 六模型接入 | 训练、推理、LoRA、逐窗口梯度选择接口与验证 | 工程接入完成：六模型 CPU 真权重、LoRA 更新、单窗梯度和双窗批量 VJP 已通过；GPU 正式实验未启动 | 六模型冒烟产物已生成；尚无正式训练或评估结果，现有 GPU 批次保持运行 |
| v1-crb-redo | 12tasks-ref-grad-prepare | 12 项安全历史资格检查，8 项准备 CRB/Uniform 梯度 | 完成；4 项放弃 | 8 项每方法各 9079 个名义锚点，4 项因无法支持筛选不再使用 |
| v1-crb-redo | grad-prepare | src-8192热身三模型1000步；src-96按q=32准备CRB 1:0:1和同预算Uniform梯度 | 完成：三模型热身与梯度标记、整批完成标记均已生成 | TimerXL、TimesFM、Chronos 各方法均完成；详见梯度输出 |
| v1-crb-redo | timerXL-baseline-full | src-96全量合法窗口；预训练权重起训4,000步；12任务评估 | 完成：4,000步训练及FEV 4 + TIME 8评估，批次标记已生成 | 12项中10项MASE低于rand01；详见docs/exp/v1-crb-redo/results.md |
| v1-crb-redo | timerXL-baseline-rand01 | src-96固定10%随机窗口；预训练权重起训4,000步；12任务评估 | 完成：4,000步训练及FEV 4 + TIME 8评估，批次标记已生成 | 12项中2项MASE低于full；详见docs/exp/v1-crb-redo/results.md |
| v1-crb-redo | timerXL-base-uni-01-traj-rr | 双侧 Uniform 轨迹 RR，8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.601606；选样 55分42秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-base-uni-01-block-rr | 双侧 Uniform 块 RR，8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.527675；选样 53分30秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-ours-rage101-01-traj-rr | 双侧 CRB 轨迹 RR，8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.590119；选样 56分12秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-ours-rage101-01-block-rr | 双侧 CRB 块 RR，8 项单任务训练评估 | 完成：8/8 组训练评估 | MASE 宏平均 1.516385；选样 59分30秒；索引文件每任务 2.03 MiB |
| v1-crb-redo | timerXL-base-uni-01-win-rr | 双侧 Uniform 窗口 RR，BF16 完整评分矩阵，8 项单任务训练评估 | 部分完成：4/8 项选样，3/8 项训练评估完成；截至 13:57 CST，TIME CPHL/H 训练中 | 已完成 FEV 9、TIME 1、2 的 MASE 为 0.670380、1.208172、1.025688；全指标见逐实验结果 |
| v1-crb-redo | timerXL-ours-rage101-01-win-rr | 双侧 CRB 窗口 RR，BF16 完整评分矩阵，8 项单任务训练评估 | 等待 Uniform win 队列完成 | 尚未开始，暂无结果 |
| v0-timerxl-src96-first4-full-grad-4096d-bf16 | 前四条完整轨迹 | 逐窗口计算 TimerXL LoRA 梯度并投影 4096 维、BF16 保存 | 完成 | 484,907 个合法窗口；八个 worker 与批次完成标记已核验 |
| v0-chronostiny-timerxl-lora-track0 | ordinal 0全量梯度与相关性 | 原始ChronosTiny LoRA 4096D投影，完整230,161窗口；同轨TimerXL关系比较 | 完成 | 非临近窗口对余弦Spearman 0.076；相邻窗口余弦Spearman −0.138；只代表单轨 |
| v0-chronostiny-timerxl-lora-full-64 | 64轨全窗口梯度与相关性 | 使用原始ChronosTiny权重计算前64轨全部合法窗口的LoRA 4096D BF16投影；与同窗口TimerXL比较梯度关系 | 用户调整后停止，保留部分缓存 | 8个worker曾启动；未完成全轨，结果不用于分析 |
| v0-timer-XL-grad-64-change-velocity | 梯度准备 | src-96前64轨逐窗口准备TimerXL LoRA与lm_heads的4096D BF16投影梯度；复用前四轨LoRA | 完成 | 64轨两类4096D BF16投影梯度已核验；前四轨LoRA复用；共1,980,371窗口 |
| v0-crb-final-test-timerXL-64tr | CRB-1:0:1 | 按平均间距q作64轨BF16 LoRA梯度同预算CRB/Uniform方向代表误差比较；K=max(2,ceil(N/q)) | 完成 | 64轨、9个q、5,053,578块/臂，结果见RESULTS |
| v0-crb-final-test-timerXL-64tr | CRB-1:1:0 | 同预算coverage:error CRB/Uniform，q为平均锚点间距 | 完成 | 64轨×9个q；Uniform在全部q的窗口加权误差更低 |
| v0-crb-final-test-timerXL-64tr | CRB-1:0:0 | 同预算coverage-only CRB/Uniform，q为平均锚点间距 | 完成 | 64轨×9个q；CRB仅q=64的窗口加权误差更低 |
| v0-timer-XL-grad-64-change-velocity | 自动分析 | 轨迹内与轨迹间速度差异、头梯度代理、两段均匀16窗口代理 | 完成 | 轨迹内/间差异、头代理与两段16窗口估计均完成；详见独立报告 |
| v0-timer-XL-grad-64-change-velocity | per-track-pca | 64轨真实4096维LoRA投影的逐轨PC1解释率与方向差异 | 完成 | PC1解释率中位数12.1%；PC1两两夹角中位数87.3°；rank80均值174、中位数137；v/L均值4.07%、中位数1.63%，L为每轨完整合法起点缓存行数；稳定性和限制见docs/exp/v0-timer-XL-grad-64-change-velocity/v0-timerxl-grad64-pca.md |
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
| v1-computational-effort-estimate | 9 个计时批次；目标清单及正式实验 | 三种模型的计算与存储估算 | 计时完成；正式实验未开始 | 旧候选池估算见[独立报告](exp/v1-computational-effort-estimate/results.md) |
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
