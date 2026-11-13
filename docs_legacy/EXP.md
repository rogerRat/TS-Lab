# 实验

最后修改日期：2026-11-12 23:38 CST

## 实验与批次总表

本表逐行保留所有已登记批次，并核对 v1 的 seed 1–3 批次。数值摘自原始登记；完整参数、方法和结果记录保存在 [英文历史原文](EXP.english-archive-20261110.md)，各实验的正式结果见 [RESULTS.md](RESULTS.md)。队列状态以本次核实的完成标记和日志为准。

| 实验代号 | 批次 | 工作 | 状态 | 实际结果 |
|---|---|---|---|---|
| v2-foundation-expansion | Chronos-2、Bolt、Moirai 2.0 六模型接入 | 训练、推理、LoRA、逐窗口梯度选择接口与验证 | 工程接入完成：六模型 CPU 真权重、LoRA 更新、单窗梯度和双窗批量 VJP 已通过；GPU 正式实验未启动 | 六模型冒烟产物已生成；尚无正式训练或评估结果，现有 GPU 批次保持运行 |
| v1-crb-redo | grad-prepare | src-8192热身三模型1000步；src-96按q=32准备CRB 1:0:1和同预算Uniform梯度 | 完成：三模型热身与梯度标记、整批完成标记均已生成 | TimerXL、TimesFM、Chronos 各方法均完成；详见梯度输出 |
| v1-crb-redo | timerXL-baseline-full | src-96全量合法窗口；预训练权重起训4,000步；12任务评估 | 完成：4,000步训练及FEV 4 + TIME 8评估，批次标记已生成 | 12项中10项MASE低于rand01；详见docs/results/v1-crb-redo-timerxl-baselines.md |
| v1-crb-redo | timerXL-baseline-rand01 | src-96固定10%随机窗口；预训练权重起训4,000步；12任务评估 | 完成：4,000步训练及FEV 4 + TIME 8评估，批次标记已生成 | 12项中2项MASE低于full；详见docs/results/v1-crb-redo-timerxl-baselines.md |
| v0-timerxl-src96-first4-full-grad-4096d-bf16 | 前四条完整轨迹 | 逐窗口计算 TimerXL LoRA 梯度并投影 4096 维、BF16 保存 | 完成 | 484,907 个合法窗口；八个 worker 与批次完成标记已核验 |
| v0-chronostiny-timerxl-lora-track0 | ordinal 0全量梯度与相关性 | 原始ChronosTiny LoRA 4096D投影，完整230,161窗口；同轨TimerXL关系比较 | 完成 | 非临近窗口对余弦Spearman 0.076；相邻窗口余弦Spearman −0.138；只代表单轨 |
| v0-chronostiny-timerxl-lora-full-64 | 64轨全窗口梯度与相关性 | 使用原始ChronosTiny权重计算前64轨全部合法窗口的LoRA 4096D BF16投影；与同窗口TimerXL比较梯度关系 | 用户调整后停止，保留部分缓存 | 8个worker曾启动；未完成全轨，结果不用于分析 |
| v0-timer-XL-grad-64-change-velocity | 梯度准备 | src-96前64轨逐窗口准备TimerXL LoRA与lm_heads的4096D BF16投影梯度；复用前四轨LoRA | 完成 | 64轨两类4096D BF16投影梯度已核验；前四轨LoRA复用；共1,980,371窗口 |
| v0-crb-final-test-timerXL-64tr | CRB-1:0:1 | 按平均间距q作64轨BF16 LoRA梯度同预算CRB/Uniform方向代表误差比较；K=max(2,ceil(N/q)) | 完成 | 64轨、9个q、5,053,578块/臂，结果见RESULTS |
| v0-crb-final-test-timerXL-64tr | CRB-1:1:0 | 同预算coverage:error CRB/Uniform，q为平均锚点间距 | 完成 | 64轨×9个q；Uniform在全部q的窗口加权误差更低 |
| v0-crb-final-test-timerXL-64tr | CRB-1:0:0 | 同预算coverage-only CRB/Uniform，q为平均锚点间距 | 完成 | 64轨×9个q；CRB仅q=64的窗口加权误差更低 |
| v0-timer-XL-grad-64-change-velocity | 自动分析 | 轨迹内与轨迹间速度差异、头梯度代理、两段均匀16窗口代理 | 完成 | 轨迹内/间差异、头代理与两段16窗口估计均完成；详见独立报告 |
| v0-timer-XL-grad-64-change-velocity | per-track-pca | 64轨真实4096维LoRA投影的逐轨PC1解释率与方向差异 | 完成 | PC1解释率中位数12.1%；PC1两两夹角中位数87.3°；rank80均值174、中位数137；v/L均值4.07%、中位数1.63%，L为每轨完整合法起点缓存行数；稳定性和限制见docs/results/v0-timerxl-grad64-pca.md |
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
| v1-computational-effort-estimate | 9 个计时批次；目标清单及正式实验 | 三种模型的计算与存储估算 | 计时完成；正式实验未开始 | 旧候选池估算见[独立报告](results/v1-computational-effort-estimate.md) |
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

All v1-rage rows above are deprecated; historical results are retained as possible references only. The new source pool is src-8192.

## 现行正在进行的实验

### v1：Trev 基础模型随机池与完整池训练

首次提出：2026-11-10。最后执行或修改：2026-11-10。目标：在固定数据池、训练预算和评估任务下，比较三种基础模型使用随机 10% 数据池与完整数据池训练 LoRA 后的 MASE。每个模型、池和随机种子都是独立批次。full 表示完整数据池，不表示全参数训练。基础模型权重冻结。

### 明确的训练超参数与设计决策

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

### v1 批次安排与现状

同一 seed 内按 Chronos 随机池、Chronos 完整池、TimerXL 随机池、TimerXL 完整池、TimesFM 随机池、TimesFM 完整池的顺序执行；种子按 0、1、2、3 顺序推进。seed 0 的 Chronos、TimerXL 两组及 TimesFM 随机池批次已有完成标记；TimesFM 完整池正在训练，seed 1–3 等待队列。seed 1–3 的 Chronos 输入池已备妥。上述未完成状态须以实际进程和产物持续更新。不得把排队或仅训练完成写成整个批次完成。


### v0-crb-final-test-timerXL-64tr

首次提出：2026-11-12。用户澄清q表示平均每q个有效窗口起点计算一次锚点。固定q={2,4,6,8,12,16,24,32,64}；每轨N个有效起点时锚点预算K=max(2,ceil(N/q))，Uniform与CRB严格匹配K并包含首尾锚点。CRB三个串行臂为coverage:error:split=1:0:1、1:1:0、1:0:0；按整数最大余数法分配K-2个额外锚点，余数相同优先coverage。复用冻结的CRB coverage、积分二次插值风险error、递归min-kappa split算法；Uniform为K个含端点的等距起点。

相邻锚点之间的闭区间定义为一个块，共享锚点计入左右两个块。对端点梯度作线性重建，以块内线性估计向量的均值作代表；每个有效窗口起点计算1-cos(真实梯度,代表)，按块内起点数加权。输入是已有64轨4096D BF16 LoRA投影梯度，共1,980,371个有效起点，不重新计算模型梯度。向量计算使用BF16量化的源值，不使用FP64。正确输出位于outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q/，三个臂及顶层完成标记均已核验。结果见docs/results/v0-crb-final-test-timerXL-64tr.md。

此前将q误读为每轨锚点总数生成的旧产物已删除，不作为结果。正确spacing-q队列于2026-11-12 01:43在tmux v0-crb-final-spacing-q启动，02:12三个臂完成。

### v2-foundation-expansion：轻量基础模型接入

首次提出和最后修改：2026-11-12。目标：按用户批准的范围，将 Chronos-2-Small（27,934,624 参数）、Chronos-2（119,477,664）、Chronos-Bolt-Tiny（8,652,672）、Mini（21,236,096）、Small（47,718,016）和 Moirai-2.0-R-Small（11,387,208）接入现有预测、训练、LoRA 与基于梯度的数据选择流程。参数数目与模型修订来自官方 Hugging Face 模型记录；六模型的实际参数、推理、损失、LoRA 反传及逐窗口梯度已逐一核验。Moirai 权重为 CC-BY-NC-4.0，仅用于非商业研究。

| 批次 | 权重 ID 与固定修订 | 接入方法 | 状态 |
|---|---|---|---|
| Chronos-2-Small | autogluon/chronos-2-small @ ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a | 官方模型前向、原生分位数损失、注意力 LoRA、独立序列梯度 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Chronos-2 | amazon/chronos-2 @ 29ec3766d36d6f73f0696f85560a422f50e8498c | 同上 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Bolt-Tiny | amazon/chronos-bolt-tiny @ a0e552de83495b5c28c14c71c374f3e33280b340 | 官方模型前向、原生分位数损失、注意力 LoRA、逐窗口梯度 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Bolt-Mini | amazon/chronos-bolt-mini @ 251268337516a88e253628c43e1d26ec577b376b | 同上 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Bolt-Small | amazon/chronos-bolt-small @ 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 同上 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |
| Moirai-2.0-R-Small | Salesforce/moirai-2.0-R-small @ 30f43ff08c8494f4943ae1521e9d4e94a0fbb389 | 官方模块与窗口转换、直接预测 token 的分位数损失、注意力 LoRA、逐窗口梯度 | CPU 真权重冒烟通过：推理、损失、LoRA、单窗梯度、双窗 VJP 等价 |

现阶段仅开发接口，不启动正式实验。窗口、优化器参数、数据池、训练步数、评估配置仍由后续具体实验配置决定，不沿用其他模型的数值。代码路径：src/adapters/foundation/chronos_family.py、moirai2.py、spec.py、factory.py；src/gradient/new_foundation.py；src/rage/experiments/students.py、full_pool_models.py、lora.py；验证入口 scripts/smoke_foundation_models.py（以 `python -m scripts.smoke_foundation_models` 运行）。已核对 Chronos 官方损失与逐窗口损失、六模型预测输出、LoRA 可训练参数、梯度投影和标准输出路径；真实权重烟雾测试均完成。用户确认所有新增模型使用专用 Conda 环境；Chronos-2/Bolt 统一使用新建 rage-chronos2，Moirai 使用新建 rage-moirai2。后者已安装并通过依赖检查，Torch 2.4.1、uni2ts 2.0.0、transformers 4.45.2、PEFT 0.13.2。真实权重冒烟使用 512 点历史、64 点预测、LoRA r=8、alpha=16、dropout=0、128D 梯度投影；双窗口 VJP 与逐个反传核对。Chronos-2 与 Bolt 的训练使用各自官方模型前向返回的分位数损失；逐窗口梯度以同一归一化分位数公式一次批量前向配合 VJP 计算，避免逐窗重复加载。Moirai 使用官方预测窗口转换和模块前向，以最后一个历史 token 的直接预测 token 计算有观测掩码的缩放空间 pinball 损失；该固定窗口训练损失是项目实现，并非官方发布的微调配方，当前限制预测长度不超过 `num_predict_token * patch_size`。所有六模型使用 512 历史点、64 预测点做工程冒烟，128D 双窗口 VJP 与逐窗梯度等价，且 LoRA AdamW 更新及更新后预测通过；Bolt-Tiny 和 Moirai 另验证正式模型构建器的 4096D 双窗口探针。模型注册见 configs/pretrained_models/chronos2.yaml、moirai2.yaml 和 requ/third_parties/model/integrations.lock。该验证不产生正式预测质量指标。当前 GPU 上的 v1-crb-redo 批次不受此次接入影响。

## 历史实验和更改记录

以下保留每项实验的目标、关键方法、状态与结果入口。2026-11-10 中文整理前的完整逐批次方案与原文参数均保留在 [英文历史原文](EXP.english-archive-20261110.md)，没有删除既有实验决定。

### v0-active-value-rank

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。固定 100 个参考窗口，比较输出头代理排序与预算内完整梯度查询的候选选择；包含扩展跨轨排序、主动残差校正、代理覆盖诊断及等预算查询。结果和局限见 docs/results/ 与 RESULTS.md。

### v0-directional-marginal-gain

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。在 46 个 256 起点区块，以 n=2、4、8、16 个锚点加入最大间隙中点，检验方向离散度能否预测重建误差的真实边际下降。全部批次完成；详情见 docs/results/v0-directional-marginal-gain.md。

### v0-gradient-variance-decomposition

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。复用 46 个区块及全量梯度 Gram 缓存，分解幅值和方向贡献，控制轨迹与均值范数后分析重建误差。全部批次完成；详情见 docs/results/v0-gradient-variance-decomposition.md。

### v0-grad-traj-level-norm-gap

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。比较十条轨迹的完整梯度与输出头梯度均值范数及排序。全部批次完成；详情见对应结果文档。

### v0-var-proxy-estimate

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。用每个 256 起点区块的八个输出头代理估计梯度方差，比较四锚点插值重建误差及均值替代方案。全部批次完成；详情见 docs/results/v0-var-proxy-estimate.md。

### v0-uni-data-evaluation

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。从十条轨迹选取 100 个不相邻参考窗口，对其他窗口计算完整梯度余弦与点积，并分析间隔 1–575 的分数变化。全部批次完成；详情见 docs/results/v0-uni-data-evaluation.md。

### v0-uni-reconst-scale

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。在十条轨迹上对均匀锚点间隔 1、2、4、8、16、32、64、128 评估线性重建，并补充相邻锚点均值重建。全部完成；详情见 docs/results/v0-uni-reconst-scale.md。

### v0-vage

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。比较代理引导的区块锚点配置与全轨迹均匀配置，包含最小锚点数、配额、融合、预言机及残差诊断。所有已登记批次完成；主指标未超过均匀基线。详情见 docs/results/v0-vage.md。

### v0-gradient-proxy-test

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。评估 32–576 起点区块内稀疏输出头代理对真实梯度方差和方向离散度的排序能力，并评估跨轨候选排序及目标离散程度。各批次完成；详情见 docs/results/v0-gradient-proxy-test.md。2026-11-12 为给之后正式 v1 实验预留数据盘空间，已删除十条轨迹的未投影 FP32 输出头梯度数组（约50.14 GiB）；派生分析结果保留，删除范围见 outputs/train/v0-gradient-proxy-test/OUTPUT_HEAD_GRADIENTS_REMOVED.md。

### v0-gradient-proxy-cost

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。直接测量完整反向传播与代理计算耗时；完整反向传播中位数 42.80 毫秒，输出头代理每窗口 14.21 毫秒。实验完成。

### v0-grad-mech-l2-norm

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。基于十份 BF16 梯度缓存，测量轨迹内间隔 1–575 的相对 L2 差异。完成并归档。

### v0-crb-reconst-test

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。固定十条源轨迹，执行 CRB 重建回放。轨迹 00–02 完成；03–04 部分完成后终止；05–09 和极端轨迹扩展未开始，均因用户终止。已产生的结果和故障记录保留；具体配置与批次见历史原文。

### v0-grad-mech-test

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。固定 ChronosTiny 检查点与 M5 的前 1,000 条合格轨迹，完成预热、完整梯度探测、十份 BF16 缓存保留与结果汇总。

### rage-smoke-static-cycle

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。安装并检查六个 Conda 环境，验证数据准备、选择、训练和评估集成循环；全部批次完成。

### v0-crb-var

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。在十条轨迹上比较 CRB 自适应区间与等预算均匀区间；全部预算验证通过。详情见 docs/results/v0-crb-var.md。

### v0-proxy-dp-segmentation

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。在最长四条轨迹上比较精确 BF16 代理方差动态规划与同数量均匀分段，执行未加权和窗口加权两批次。两批次完成。原始 BF16 完整梯度向量缓存已按用户指示删除，代理缓存与 Gram 矩阵仍保留。

### v1 的已有更改记录

2026-11-10：明确 full 为完整候选池 LoRA 训练，而非全参数训练；决定仅以 MASE 汇总；TimesFM 的 FEV 与 TIME 输入在必要时向左补齐至 32 点 patch 的倍数；失败的 seed 0 随机池评估从已有优化器和评估产物恢复，不重跑已完成部分。当前仍使用原队列；详细故障过程见 docs/LOG.md，排队规则见 docs/ORCHE.md。

## Trev seed scope closed (2026-11-10)

Last modified: 2026-11-10

The user stopped the multi-seed extension and retained seed 0 only. All six seed 0 Chronos, TimerXL, and TimesFM rand/full LoRA phases are complete; MASE-only results are in docs/results/v1-rage-trev-seed0-final.md. Chronos seed 1 random training was interrupted before step 4000 (latest logged progress step 2445) and has no formal result. The other seed 1-3 phases were not trained. Seed 1-3 configs, prepared pools, partial checkpoints, and the extension queue were removed. Historical batch rows remain in the overview for traceability.



### Time-300B data retention decision

Last modified: 2026-11-10. At 2026-11-10 18:52:21 CST, the user requested removal of the unused Time-300B raw data and converted shards. The v1 Trev candidate pool has five sources and does not include Time-300B. Before deletion, `data/src/raw/time300b` held 51 files and 63,511,169,049 bytes; `data/src/shard_dataset/selection24-time300b` held 116 files and 122,167,109,471 bytes. Keep both directory paths with brief deletion notes.

## v1-rage-crb-vs-uniform-timerXL-84m

First proposed: 2026-11-10. Last modified: 2026-11-10. Current state: one-task optimization timing complete; formal selection and training have not started and await user review of the updated single-GPU estimate.

Fixed protocol: use the v1 TimerXL 84M BF16 LoRA training and evaluation recipe. Select 10% of all legal 576-point windows, floor(0.1*N). The 12 existing tasks perform independent selection sequentially; each task uses only its own eligible references. On each turn, choose a random eligible ref for that task, then add its highest-cosine-scoring src that is not already selected; stop at the data budget. Higher cosine is better.

Gradients use trainable TimerXL LoRA parameters only, r=8, alpha=16, dropout=0, targeting q_proj/v_proj, with a frozen base model. A single fixed Count-Sketch projection is shared across tasks and methods, dimension 4096, seed 42. LoRA initialization seed is 0. Uniform and CRB use equal gradient-point budgets, q=32, K=ceil(N/32), including endpoints; both use linear reconstruction between adjacent anchors. CRB anchor allocation and placement follow the fixed CRB protocol.

The current source pool has N=360,695,805 and the formal data budget is floor(0.1*N)=36,069,580 distinct legal windows. Each method uses K=ceil(N/32)=11,271,744 gradient anchors. After selection, each task independently trains one TimerXL LoRA batch using the v1 recipe: 4,000 updates, batch 32, microbatch 8, BF16, checkpoints at 0/100/200/500/1000/2000/3000/4000, and evaluation on all 12 FEV/TIME tasks. Training and evaluation start only after formal data selection completes.

### One-task feasibility timing: hospital_admissions_1D

The first timing task is FEV hospital_admissions_1D (task index 9). Eligible references come from the local FEV train split and are finite, nonconstant univariate windows with at least 576 points. Timing pilots produce no formal selection manifest or anchors and run no selection or training. Initial config, entry point, and output are `configs/rage/v1-rage-crb-vs-uniform-timerXL-84m-pilot.json`, `scripts/temp/v1_rage_crb_uniform_timerxl_pilot.py`, and `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/`. There were 9,246 eligible refs. The initial source-gradient rate was 22.64 points/s and its 138.3-day source-gradient estimate is superseded by the optimization timing below. Initial ref-src matrix timing was 630.35 million pairs/s; one exhaustive hospital_admissions_1D score sweep was estimated at 1.47 hours before chunking overhead.

### BF16 gradient optimization timing

Last modified: 2026-11-10. The PEFT-compatible batched TimerXL per-window loss was corrected to reproduce the checkpoint loss-mask reduction and sequential per-head accumulation. Its maximum absolute difference from scalar adapter losses was 0.0 on the verification sample.

On 96 source windows, eager batch 32 / VJP 16 reached 437.73 gradients/s (0.299 GiB peak allocated GPU memory). Both `torch.compile(mode=reduce-overhead)` and `torch.compile(mode=default)` compiled the forward path but failed during batched-VJP backward with `Cannot access data pointer of Tensor that doesn't have storage`; compiled execution is not used.

A follow-up tested 1,024 source windows with batch 32/64/128 and VJP chunks 16/32/64; its best was batch 32 / VJP 32 at 624.38 gradients/s. An additional VJP 128 sweep did not improve throughput. Two sustained 8,192-window runs at batch 32 / VJP 32 measured 596.82 and 575.35 gradients/s, both with 0.321 GiB peak allocated memory. Use the slower sustained rate for planning: 270,521,856 source gradient points across two methods and twelve tasks require 5.44 single-GPU days. This excludes reference gradients, complete score scans, source I/O/cache writes, training, and evaluation. Artifacts are in `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/optimization/`, `.../optimization-stable/`, `.../optimization-vjp128/`, `.../optimization-steady-8192/`, and `.../optimization-steady-repeat/`; each contains `.complete`. Recommended timing setting: BF16, one batched forward, batch 32, VJP 32, FP32 projection accumulation. This remains a feasibility estimate only; formal selection and training have not started.


### Same-GPU process parallelism timing

Last modified: 2026-11-10. The gradient probe uses about 0.321 GiB peak PyTorch-allocated GPU memory per process. Concurrent process timing on the same RTX 4080 used independent TimerXL BF16 LoRA processes, each processing 4,096 sampled source windows at batch 32 / VJP 32. Two processes reached 964 points/s aggregate (about 1.68x the sustained single-process rate); four reached 1,087/s (1.89x); eight reached 1,159/s (2.01x). Each process retained about 0.321 GiB peak allocated memory. During the eight-process run, `nvidia-smi` showed about 5.7 GiB total process memory and 100% GPU utilization. Eight processes add only about 6.6% over four, showing diminishing returns despite remaining VRAM headroom. Their model initialization was also CPU-heavy and took about 1.4 minutes in the eight-process pilot.

If this rate holds in the real distributed source-window pipeline, the 270,521,856 source-gradient points project to about 2.70 days at 1,159 points/s, before worker startup, synchronization, source I/O, reference gradients, score scans, training, and evaluation. Treat this as an initial parallel upper estimate, not a formal runtime commitment; validate persistence, unique work partitioning, and end-to-end cache writes before using it for the final schedule. Parallel timing outputs with `.complete` are under `outputs/train/v1-rage-crb-vs-uniform-timerXL-84m/pilot/hospital_admissions_1D/parallel-two/`, `parallel-four/`, and `parallel-eight/`.


## v1：计算量与存储估算

首次提出：2026-11-10。最后修改：2026-11-10。计时试跑已完成；目标清单、正式筛选和训练未开始。详情见[独立结果报告](results/v1-computational-effort-estimate.md)。旧数据池估算不适用于当前 src-8192 数据池。

| 计时批次 | 状态 | 吞吐量（梯度/秒） |
|---|---|---|
| TimerXL 单进程、2 进程、4 进程、8 进程 | 完成 | 575.35、964、1087、1159 |
| TimesFM 单进程、2 进程 | 完成 | 23.436、27.827 |
| Chronos 单进程、2 进程、4 进程 | 完成 | 7.396、14.766、18.052 |
| 目标轨迹清单 | 未开始 | 无实际轨迹数或窗口数 |
| 正式筛选和训练 | 未开始 | 无正式指标 |

历史目标规模为 147456000 个标量值，按旧候选池均值估算约 61800 条轨迹、111939593 个合法窗口。每种方法约 3498113 个梯度锚点；Uniform 与 CRB 各有一份缓存，十二个任务共用。详细口径与局限见独立报告。

## src-8192 data replacement

Last modified: 2026-11-10. All previous v1-rage batches are deprecated because the selected source data were unsuitable. Their result files remain available only as possible references and are not formal answers. The replacement source collection is src-8192: exactly 8,192 trajectories selected from 19 eligible configurations across Chronos, LOTSA, Monash, and UTSD. The fixed per-trajectory index is configs/data/src-8192.yaml; converted values and offsets are under data/src/shard_dataset/src-8192/. No v1-rage training batch has been restarted with this replacement data.

The 19 selected configurations and per-configuration counts are recorded in configs/data/src-8192.yaml. Selection used seed 0, eligible series length at least 576, finite nonconstant values, and balanced allocation across configurations until 8,192 were selected. The fixed YAML lists every source, subset, bundle row, series ID, length, and ordinal. The independent float64 values.npy and offsets.npy preserve the selected data even if an upstream main branch changes. Data were reused from the already downloaded and converted local source bundles; no fresh network download was needed.

### src-8192 expansion to 35 configurations

Last modified: 2026-11-10. Downloaded additional Chronos and Monash subsets using the project Fetch stage, then converted bounded samples with Load. Sixteen newly converted configurations passed the fixed eligibility rule (length at least 576, finite, nonconstant), expanding coverage from 19 to 35 configurations. The selection seed remains 0 and the trajectory count remains exactly 8,192. Balanced allocation selects one eligible trajectory per configuration per round until the target is reached, with exhausted configurations skipped. The authoritative index is configs/data/src-8192.yaml, version 2. It records all 8,192 source rows and series IDs. The independent float64 snapshot is data/src/shard_dataset/src-8192/. Version 1 was removed after the user requested deletion of all other data. No deprecated v1-rage batch was restarted.

### src-8192 preservation decision

Last modified: 2026-11-10. The user will permanently upload the independent src-8192 data snapshot. That snapshot and configs/data/src-8192.yaml are authoritative. Upstream commit-pinned fetching is now supported by the project adapters for future use, but the full current snapshot has not been re-downloaded and verified against those commits.

### Data cleanup

Last modified: 2026-11-10. Per user request, removed every project data item except data/src/shard_dataset/src-8192/. The src-8192 YAML no longer refers to source bundle paths that were removed. It retains stable ordinals, source/subset labels, series IDs, lengths, counts, and the paths to the fixed values and offsets. The previous src-8192 version and YAML backup were removed.

## v0-grad-prepare-src-96tr-q16-4096d

首次提出：2026-11-11。最后修改：2026-11-11。状态：仅 Uniform 梯度准备；TimerXL 和 TimesFM 已完成，Chronos 四进程运行中。用户于 2026-11-11 21:03 CST 决定停止 CRB 并删除本批次已产生的 CRB 清单及仅由 CRB 使用的梯度缓存。原三模型 CRB+Uniform 队列已停止，后续仅计算 Uniform；旧的 CRB+Uniform 批次完成标记已按新协议重新核验。

输入为固定 `src-96`：96 条完整轨迹，覆盖 src-4096 的 35 个配置，每配置至少 1 条，其余按配置规模分配，配置内 seed 0 无放回抽样；不按轨迹长度替换。索引 `configs/data/src-96.yaml`，独立数据快照 `data/src/shard_dataset/src-96/`；共有 2,721,512 个值和 2,666,312 个合法窗口，逐条源值与偏移已验证。

三模型为 TimerXL 84M、TimesFM 2.5 和 Chronos-T5-Base。窗口上下文 512、预测 64、步长 1。基础参数冻结；LoRA r=8、alpha=16、dropout=0，Chronos 目标 q/v，另两模型目标 q_proj/v_proj，初始化 seed 0。固定 4096 维 Count-Sketch 投影，seed 42，FP32 累加后以 BF16 位编码保存。Uniform 在每条轨迹的合法起点等距取锚点，q=16；逐轨迹取整后每模型共 166,659 个锚点。三模型共用相同 96 条轨迹和相同的 Uniform 锚点身份，各自保存独立梯度缓存。

入口 `scripts/prepare_src_96_gradients.py`；当前队列 `scripts/temp/run_src_96_uniform_queue.sh`，tmux `v0-grad-src-96-uniform`。TimerXL 已完成八进程、TimesFM 已完成两进程；两模型的旧共享缓存已过滤为仅 Uniform 行，CRB 清单和 CRB 专属梯度已删除。Chronos 已完成的 Uniform 行同样保留并过滤，剩余轨迹由四进程继续准备。各模型与整批仅在 Uniform 缓存和清单全部验证后写 `.complete`。旧 CRB+Uniform 计划和约 333,318 次/模型的总计算上界已由用户决策终止，不再作为现行执行方案。

## v0-crb-mech-test

首次提出与开始：2026-11-11。最后修改：2026-11-11。状态：完成原始对比、补充分析、CES 与 CES-Oracle 对比。

目标：在 64 条计算机生成的平滑向量轨迹上，按 CRB 固定三阶段预算逻辑选锚点，与等预算均匀锚点比较逐点向量重建误差，并分析轨迹粗糙度和 CRB 相对 Uniform 的误差差值。

| 批次 | 轨迹和预算 | 方法、指标与完成标准 | 状态 |
|---|---|---|---|
| 64-smooth-uniform-reconstructors | 同一 64 条轨迹、原 Uniform 锚点和 q={4,8,16,32} | 比较线性、自然三次样条、PCHIP、q−1 次 Chebyshev、RBF-GP（两个固定长度尺度）与平滑样条（固定 lambda 敏感性网格）；全 4096 维评估并与 CRB/CES 对照 | 完成 | 2880 条方法-轨迹记录；q≤16 线性 Uniform 平均 L2 最低，q=32 自然三次和低 lambda 平滑样条略低；PCHIP q=32 方向误差最低；见结果报告 |
| 64-smooth-ces-oracle | 同一 64 条轨迹；复用 CES 均匀 pilot 和 q 预算 | 用 mock 生成器解析 g'' 得到真实曲率；剩余锚点按真实曲率密度积分分配并按局部曲率 CDF 放置；与估计曲率 CES、Uniform、CRB 对比 | 完成 | 1024 条四方法指标记录；CES-Oracle 平均误差未优于 Uniform 或估计版 CES，但四个 q 的 L2 均低于 CRB；详见结果报告 |
| 64-smooth-ces-pilot | 64 条既有轨迹；q={4,8,16,32}；CES、Uniform、CRB 等预算 | 均匀 pilot 占预算 50%，至少 3 个 pilot（端点包含）；按非均匀三点二阶差分估曲率；剩余预算按区间长度乘曲率的 2/5 次方分配，区间内均匀布点；报告全 4096 维逐点误差及块内向量/方向方差 | 完成 | 768 条方法级重建记录；CES 平均误差接近 Uniform 且低于 CRB；q=32 的 L2 低于 CRB 64/64 轨迹；见结果报告 |
| 64-smooth-block-internal-variance | 原 64 条及极平缓单轨迹，复用各方法已有锚点块 | 块内向量 L2 方差、相对向量方差、方向方差与模长方差；每点只计入一个块，均值作块内代表 | 完成 | 原 64 条上 q=32 时 CRB 的相对向量方差、方向方差均在 64/64 轨迹较低；绝对模长方差不一致；详情见结果报告 |
| ultra-smooth-single-reconstruction | 1 条、长度 4000、维数 4096；frequency=0.005–0.05 周期/归一化时间；整条轨迹归一化到平均向量范数 1 | 复用 q={4,8,16,32} 的 CRB/Uniform 锚点重建和指标；单条结果只作极平缓曲线补充，不做相关性统计 | 完成 | 粗糙度分数 1.8373e-7；L2 四个 q 均 Uniform 较低；方向余弦误差约 1e-8–1e-7 |
| 64-smooth-anchor-reconstruction | 64 条，长度 4000，维数 4096；随机种子 20261111；平滑频率参数几何分布于 0.05–2.0 | q={4,8,16,32}；每个 q 严格使用 q 个锚点；CRB 覆盖阶段计入两个端点，剩余预算按覆盖/误差/分裂 2:1:1 分配；CRB 与 Uniform 均在全长上逐维线性插值；报告 mean vector L2、逐坐标 RMSE、mean relative L2、mean direction cosine error；并计算粗糙度与各方法误差及 CRB-Uniform 差值的 Spearman 相关 | 完成 | 64 条；512 条 method-trajectory 指标记录；q=4/8/16/32 的平均 L2 及余弦误差均为 Uniform 较低；见结果报告 |

方法细节：每条轨迹由独立随机的 12 维正弦系数通过 4096 维随机基底映射得到；频率从 0.25 到 18 周期/归一化时间，粗糙度参数单调调制相位。粗糙度排序指标为二阶差分 RMS 除以平均向量范数。轨迹每个合法点视为一个 CRB 位置，通过 TimeSeriesFragment 和现有 CRBEngine 执行真实的覆盖、误差探测与分裂调度。Uniform 端点包含在内。q=4 时端点占用两项覆盖预算，故剩余两项按离散化后的覆盖/误差/分裂配额处理；脚本必须核实最终配额与总预算。方向余弦误差定义为逐点平均 1-cos(x_t, xhat_t)，零范数项按 0 cosine 处理。

Uniform 重建器 sweep 入口与配置：scripts/v0_crb_mech_uniform_reconstructors.py、configs/rage/v0-crb-mech-uniform-reconstructors.json。平滑样条 q=4 不满足 SciPy 最少 5 个锚点条件；lambda 网格为敏感性分析，不从测试误差挑选。GP 使用固定 RBF 长度尺度和 alpha。

64-smooth-ces-oracle 入口与配置：scripts/v0_crb_mech_ces_oracle.py、configs/rage/v0-crb-mech-test-ces-oracle.json。解析真实曲率仅因本 mock 生成器已知；真实梯度场通常不可用。CES 入口与配置：scripts/v0_crb_mech_ces.py、configs/rage/v0-crb-mech-test-ces.json；块内分析入口：scripts/v0_crb_mech_block_variance.py。CES 第一阶段固定为预算一半的等距 pilot，若少于 3 个则取 3 个；q=4 时因此 pilot 用 3 个，剩 1 个自适应点。第二阶段区间权重为区间长度乘 (曲率²+epsilon²)^(1/5)，即曲率的 2/5 次方密度的区间积分近似；采用 largest-remainder 整数分配，区间内等距布置。所有方法用完整 4096 维重建误差评价。

入口与配置：scripts/v0_crb_mech_test.py、configs/rage/v0-crb-mech-test.json。输出写入 outputs/train/v0-crb-mech-test/，运行日志写入 logs/v0-crb-mech-test/。运行环境：AutoDL Conda rage；CPU 两线程，不占用 GPU；在 tmux 执行；成功后写 .complete。


## v0-crb-mech-test Uniform + Cancellation Split（real-real oracle 诊断）

最后修改日期：2026-11-11。状态：完成。

目标：检验只依赖 Uniform 真实锚点端点的 cancellation 判据，能否通过增加虚拟分块边界改善 RAGE shape representativeness。64 条轨迹、4096 维、4000 点、种子 20261111 和频率/粗糙度生成参数沿用主批次；q={4,8,16,32}。

每个相邻 Uniform 锚点区间计算 kappa=||g_a+g_b||/(||g_a||+||g_b||)，零分母时定义为 1。若 kappa<tau，则仅增加一个虚拟切分点；其梯度值由端点线性插值得到，不增加真实梯度计算。阈值 tau={0,0.2,0.4,0.6,0.8,0.9,1.0}，tau=0 为无分割基线。对比 midpoint 与 minimum_norm 两种位置。Minimum-norm 使用 clamp(-g_a dot (g_b-g_a)/||g_b-g_a||^2,0,1)，并把分割索引限制为父区间两侧至少各有 max(2,ceil(0.1*width)) 点；宽度不足时取不超过 floor(width/2) 的可行长度。

每个子块用项目 RAGE v7 shape_rows 特征 [midpoint,slope/sqrt(12)]/sqrt(energy)，shape score 使用已有 linearSum 加权余弦。为避免待评轨迹泄漏到参考库，按轨迹序号分两折，参考集使用另一半轨迹的 Uniform q=32 区间；交换两折后评估。参考权重按区间覆盖跨度归一化，source 子块分数按其覆盖跨度加权。此指标表示相对固定 q=32 Uniform 参考形状分布的相似度，不等同于逐点重建 L2。

应用户补充要求，主要终点评估改为块内代表性。对原始 mock 梯度点按每个 Uniform 区间或虚拟子区间分块，以块内均值为描述性代表，沿用 scripts/v0_crb_mech_block_variance.py 的 population vector L2 variance、relative vector variance、direction variance、norm variance。每条轨迹逐 q/tau/rule 与未分割 Uniform 块配对；分别报告等块权均值和按块覆盖点数加权均值。块内方差与方向方差均按原分析定义计算，块均值使用区间内所有真实 mock 点，只用于代表性分析，不表示运行时免费获得块内真实梯度。新增入口 scripts/v0_crb_mech_cancellation_variance.py，输出 3,584 条轨迹级与 58,288 条子块级记录；前缀和实现与原 block_metrics 数值差在 5.5e-13 以内。

入口 scripts/v0_crb_mech_cancellation_split.py；配置 configs/rage/v0-crb-mech-cancellation-split.json；结果 outputs/train/v0-crb-mech-test/64-smooth-cancellation-split-diagnostic/；日志 logs/v0-crb-mech-test/。输出含逐轨迹汇总、逐区间 cancellation/切分/子块分数以及 .complete。计算采用 AutoDL Conda rage、CPU 两线程，在 tmux 会话 v0-crb-mech-cancellation-split 执行。


## v0-crb-mech-test Uniform Linear 六方法对照（real-real oracle 诊断）

最后修改日期：2026-11-11。状态：完成。

目标：严格固定原 Uniform 的真实锚点与分段线性重建，只改变块分析用的虚拟分割边界。各方法的逐点线性重建及其误差因此与原 Uniform+Linear 相同；新终点只比较块内 vector L2 variance、relative vector variance、direction variance、norm variance。

同一 64 条轨迹、4096 维、长度 4000、seed=20261111；q={4,8,16,32}；阈值 tau={0,0.2,0.4,0.6,0.8,0.9,1.0}。Cancellation 使用 kappa=||g_a+g_b||/(||g_a||+||g_b||)，零分母取 1；当 kappa<=tau 时触发。Minimum-norm 反转点按 clamp(-g_a dot (g_b-g_a)/||g_b-g_a||^2,0,1) 计算，切分点按原预注册规则限制父区间两侧长度至少 10%、至少 2 点，且不超过区间可行长度。

| 方法 | 选区间 | 分割位置 |
|---|---|---|
| tau-mid | kappa<=tau | midpoint |
| tau-rev | kappa<=tau | minimum-norm reversal |
| tau-mid-base | 每轨迹/q/tau 随机抽取与 tau-mid 相同 K 个区间 | midpoint |
| tau-rev-base | 与 tau-mid-base 共用同一随机 K 区间 | 每个随机入选区间自己的 minimum-norm reversal |
| all-mid | 所有 Uniform 区间 | midpoint |
| all-rev | 所有 Uniform 区间 | 每个区间自己的 minimum-norm reversal |

随机基线不放回抽样；seed 为 20261111+918273，并按轨迹/q/tau 派生可复现随机数。这样对比 tau 方案与 random-base 时，切分数和位置规则匹配，唯一变化是选择哪些父区间。All 方法不依赖 tau，所有相邻 Uniform 锚点区间均切一次。

分块后以块内全部原始 mock 点的向量均值作为描述性代表，按项目原 block variance 定义计算指标。逐轨迹与原始未切分 Uniform 块配对；分别提供等块权与按覆盖点数加权汇总。块均值属于离线分析量，不是可额外获得的真实梯度。入口 scripts/v0_crb_mech_cancellation_six_methods.py；配置 configs/rage/v0-crb-mech-cancellation-six-methods.json；输出 outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/；日志 logs/v0-crb-mech-test/。远端 Conda rage，CPU 两线程，tmux v0-crb-mech-cancellation-six-methods。结果：outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/ 中 .complete 已核实；7,680 条轨迹级记录与 107,520 条父区间记录完整。


## v0-crb-mech-test Uniform Linear 六方法 est-real 复评

最后修改日期：2026-11-11。状态：完成。

纠正前一批次的指标口径：上一批以块内真实均值代表真实值，命名为 real-real 方差估计，属于 oracle 诊断。当前 est-real 批次重用同一份仅由 Uniform 锚点生成的六方法区间分割；每个子块的代表值是原 Uniform+Linear 在该块每个采样点的估计值的算术均值。块内真实值不参与代表值构造或分割选择，只在最终评估中用于计算真实 L2 误差、相对 L2 误差和方向余弦误差。此批不增加梯度、不改变 Uniform anchor set、不改变逐点 Linear 重建。

入口 scripts/v0_crb_mech_cancellation_estimated_rep.py；配置 configs/rage/v0-crb-mech-cancellation-six-methods.json；区间分割来自 outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/interval_selection.csv；输出 outputs/train/v0-crb-mech-test/64-smooth-cancellation-six-methods/est-real/；日志 logs/v0-crb-mech-test/；tmux v0-crb-mech-cancellation-est-real。完成 64 条轨迹、q={4,8,16,32}、六方法；7,680 条轨迹级与 130,912 条块级记录，.complete 已核验。按点数加权 tau=0.9 时 cancellation 选区的 L2/方向误差均低于匹配随机基线；等块权结果方向不同，详见报告。


## v0-crb-mech-test Uniform Gradient Reconstruction 最优虚拟分块策略

最后修改日期：2026-11-11。状态：完成。

目标：固定 64 条既有 4096D 轨迹、Uniform 锚点及原 Linear 重建，比较不同 anchor-only 区间排名在相同 K 下对真实 est-real 块代表误差的预测/降低能力。q={4,8,16,32}，每个 q 的原始区间数 m=q-1；K=0,...,m。真实锚点位置、重建器和真实梯度计算预算不变；虚拟切分一律取离散 Midpoint，不计算新梯度。

代表定义：每个块的代表向量是原始 Uniform piecewise-linear 重建在该块所有离散时刻的向量均值。真实轨迹只用于计算代表相对真实点的 absolute vector L2 error、relative L2 error、direction error；并用于独立 Oracle 与 interval-gain 诊断。主指标按覆盖点数加权，等块权为辅助。原区间采样点采用不重叠归属：非末区间为 [a,b)，末区间为 [a,b]。

| 方法 | 每轨迹/q 决策 |
|---|---|
| Random-K | 从 m 个区间无放回随机取 K；100 个固定随机重复 |
| Cancellation-K | 计算端点 kappa，按 kappa 升序取前 K；零分母按 1 |
| Endpoint-Difference-K | 按 ||g_b-g_a||^2 降序取前 K |
| Predicted-Gain-K | 按离散精确的线性代表方差下降系数乘 ||g_b-g_a||^2 降序；该系数趋近 n_I/16 |
| L2-Oracle-K | 点加权曲线按真实 est-real L2 单区间 midpoint split 总收益降序；等块权曲线按父块与两个子块的块均值误差下降排序，均固定取 K |
| Direction-Oracle-K | 点加权曲线按真实方向误差总收益降序；等块权曲线按父块与两个子块的块均值方向误差下降排序，均固定取 K |

Oracle 对每个指标和汇总权重分别排序，作为对应曲线的上限。真实单区间收益即使为负，也保持 K 并继续按排序选择。预测分数与实际单区间收益按 q 及跨 q pooled 计算 Spearman。Selection Efficiency 为相对 K=0 的总体误差下降之比，分母接近零时记为未定义。64 条轨迹为统计单位；非随机方法曲线、方法间差值使用 paired bootstrap 95% CI；Random 每轨迹/q/K 使用 100 次随机抽样，报告重复均值、标准差、区间及对轨迹 paired bootstrap。所有 CI 和随机种子/抽样口径进入结果产物。

产物包括 8 张主指标 K–Error 曲线图（4 q × L2/Direction）、逐 q 表格、效率表、Spearman 表、逐轨迹曲线和随机重复数据。 已补充分解子批次：E_total=E_repr+E_estim，按点加权与轨迹等权两种方式汇总（轨迹等长，数值相同），覆盖所有方法/q/K；逐轨迹最大身份残差 8.99e-15，12 张 MSE 曲线见 docs/results/v0-crb-mech-test.md。入口 scripts/v0_crb_mech_virtual_split_selection.py；配置 configs/rage/v0-crb-mech-virtual-split-selection.json；输出 outputs/train/v0-crb-mech-test/64-smooth-virtual-split-selection-k-error/；日志 logs/v0-crb-mech-test/。远端 Conda rage、CPU 两线程、tmux v0-crb-mech-virtual-split-selection。


### 64-large-var-k-error-composition

批次完成（2026-11-11）。64 条 4096D×4000 mock；相邻点完整向量差与两点平均模长之比目标约 0.5，实测轨迹均值 0.50005；lag-16 方向余弦目标为 0，实测均值 0.0000375。模长乘宽幅平滑包络（CV=0.3225）。固定 Uniform anchors + Linear，q={4,8,16,32,64,128}，midpoint 虚拟分块 K=0..q−1；Random-K 100 次、paired bootstrap 5000 次。所有方法、point-weighted 与 equal-block、6 个 q 的 K 曲线和 Representation/Estimation/Total MSE 已完成。结论：Direction 随切分数下降；L2 在 q≤64 的全切分下上升，q=128 才略降。Cancellation K=1 对 L2 没有稳定胜过 Random；端点差异与 Predicted-Gain 在本批也未给出一致优势。完整参数、CI、区间相关、选择效率及图表索引见 `docs/results/64-large-var-k-error-composition.md`。输出 `.complete` 已核验。

## src-96、q=4 输出头梯度资源测算

首次提出：2026-11-11。最后修改：2026-11-11。仅估算，未运行。沿用固定 96 条轨迹，按逐轨迹 q=4 预算合计每模型 666,575 个锚点。模型为 TimerXL 84M、TimesFM 2.5、Chronos-T5-Base；输出头梯度投影 4096 维并以 BF16 保存的容量约每模型 5.1–5.4 GiB、三模型 15.3–16.2 GiB。用时低置信度区间：TimerXL 0.5–2 小时、TimesFM 9–30 小时、Chronos 8–30 小时。详细依据和未投影容量风险见 `docs/results/src-96-q4-output-head-estimate.md`。本估算不改变当前运行中的 Uniform 批次。

## v0-timerxl-src96-first4-full-grad-4096d-bf16

首次提出：2026-11-11。最后修改：2026-11-11。用户要求插队，对当前固定 `src-96` 的前四条轨迹（ordinal 0、1、2、3）的每一个合法窗口计算 TimerXL 84M LoRA 梯度，固定 4096 维 Count-Sketch 投影并以 BF16 位编码保存。窗口上下文 512、预测 64、步长 1；四条轨迹分别有 230,161、153、24,384、230,209 个合法窗口，合计 484,907。基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标 q_proj/v_proj，初始化 seed 0；投影 seed 42，批量 32、VJP 子批量 32。使用 rage-timerxl 环境、八个 GPU 进程，每进程处理不重叠的连续全局窗口区间。纯 BF16 向量数据约 3.70 GiB，另外包含索引与分片元数据。

配置 `configs/rage/v0-timerxl-src96-first4-full-grad.json`；入口 `scripts/temp/prepare_timerxl_first4_full.py`；调度 `scripts/temp/run_timerxl_first4_then_resume.sh`；正式输出 `outputs/train/v0-timerxl-src96-first4-full-grad-4096d-bf16/formal/`。已用 16 个窗口预演，核实索引、4096 维与 BF16 分片并删除预演输出。暂停原 Chronos Uniform 四进程，保留其缓存；插队批次全部完成后由同一 tmux 会话自动恢复。正式任务于 2026-11-11 21:41 CST 启动，八个 worker 均存活，GPU 已达到 100% 利用率；尚未生成整批完成标记。

## v0-timerXL-grad-4-lm-head

首次提出：2026-11-11。最后修改：2026-11-11。批次完成。使用已完成的四条完整 TimerXL LoRA 梯度轨迹，每轨等距抽取128个同窗口，计算 TimerXL 输出头 lm_heads 的4096D投影梯度（seed 42），并匹配已有4096D BF16 LoRA投影。输出头使用冻结骨干与原生 shifted-patch MSE 的解析梯度。两个参数空间不同，只比较各自空间内窗口对的余弦相似度，不做跨空间逐坐标余弦。轨迹内仅纳入起点相隔至少576的窗口对；另报告跨轨迹窗口对。指标：窗口对余弦的 Spearman/Pearson、输出头最接近10%窗口对的 LoRA 余弦均值提升、LoRA最接近10%的召回。轨迹1只有153个窗口，不存在轨迹内非临近窗口对。代码 scripts/temp/v0_timerxl_grad4_analysis.py；正式产物 outputs/train/v0-timerXL-grad-4-lm-head/；结论 docs/results/v0-timerXL-grad-4-lm-head.md。

## v0-timer-XL-grad-4-curvature

首次提出：2026-11-11。最后修改：2026-11-11。批次完成。读取四条完整 TimerXL LoRA 4096D BF16 投影梯度，按各轨迹物理窗口起点严格排序。对每轨计算一阶差 g[t+1]-g[t] 与二阶差 g[t+2]-2g[t+1]+g[t]，不跨轨迹边界。指标：二阶差 RMS、均值模长/RMS、中心化 RMS/RMS、模长 CV、相邻二阶差余弦均值及16个等宽时间段的二阶差均值模长。恒定曲率要求二阶差向量围绕同一个值的小幅波动，中心化 RMS/RMS 应接近0；另以二阶差 RMS 与一阶差 RMS 比较平滑性。代码 scripts/temp/v0_timerxl_grad4_analysis.py；正式产物 outputs/train/v0-timer-XL-grad-4-curvature/；结论 docs/results/v0-timer-XL-grad-4-curvature.md。该结论只针对4096D投影和BF16表示。

原 v0-timerxl-src96-first4-full-grad-4096d-bf16 批次已于2026-11-11完成，八个worker与整批完成标记核验；调度脚本自动恢复 Chronos Uniform。上文21:41的运行中状态属于历史启动记录。

曲率批次补充（2026-11-11）：按用户要求增加相邻曲率变化 d3[t]=d2[t+1]-d2[t]，报告 d3 RMS/d2 RMS 与 d2 RMS/梯度 RMS；16个时间段另计算 d2 RMS 以检测分段特征，并比较跨轨绝对与归一化曲率。完整定义和结果见 docs/results/v0-timer-XL-grad-4-curvature.md。

## v0-timer-XL-grad-4-change-velocity

首次提出：2026-11-11。最后修改：2026-11-11。使用已完成的四条 TimerXL LoRA 4096D BF16投影梯度轨迹，长度分别230161、153、24384、230209。对每个滞后 k=1..512 和每条轨迹的每个有效窗口对 (t,t+k) 逐一计算三个指标：(1) 方向余弦 cos(g[t],g[t+k])，先以各向量模长归一化；(2) 相对L2差 ||g[t]-g[t+k]|| / ((||g[t]||+||g[t+k]||)/2)；(3) 相对模长差 abs(||g[t]||-||g[t+k]||) / ((||g[t]||+||g[t+k]||)/2)。每个k报告所有有效对的均值、总体标准差和对数 N-k。轨迹1仅153个窗口，k>=153无有效对，记录为零对且指标未定义。直接使用全部有效窗口对，不抽样；不跨轨迹边界。禁止将投影空间结论视为未投影LoRA参数空间结论。代码 scripts/temp/v0_timerxl_grad4_change_velocity.py；输入 outputs/train/v0-timerxl-src96-first4-full-grad-4096d-bf16/formal/；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/。远端 Conda rage-timerxl，CUDA矩阵计算，tmux v0-timerxl-grad4-change-velocity。完成后核验全滞后对数并分析曲线形态。

v0-timer-XL-grad-4-change-velocity 已完成（2026-11-11）：四轨全部有效窗口对、k=1..512，每轨各一张三面板PNG/SVG图。轨迹1仅k<=152有效。原始结果与分析见 docs/results/v0-timer-XL-grad-4-change-velocity.md；输出目录和图目录完成标记均已核验。

### v0-timerXL-grad-4-lm-head：change-velocity-comparison 补充批次

首次提出：2026-11-11。最后修改：2026-11-11。检验输出头梯度计算的方向余弦、相对L2差、相对模长差曲线是否接近LoRA。原每轨128个等距样本不能构成k=1..512的连续窗口对，故在长轨迹0、2、3各选两个相距很远、长度768的连续窗口段（起点分别约为可用范围的25%、75%），短轨迹1使用完整153窗口，总共4761个窗口。在相同窗口上计算输出头lm_heads的4096D投影梯度，使用相同模型、窗口、投影维度和seed；匹配既有LoRA BF16投影梯度。各段内所有k=1..512有效窗口对按对数加权合并；短轨迹仅k<=152。三个指标定义沿用 v0-timer-XL-grad-4-change-velocity，逐k报告输出头曲线、同段LoRA曲线及全轨LoRA参考曲线。比较同段两曲线的数值误差与形态，并检查抽样段LoRA曲线相对全轨是否代表性。代码 scripts/temp/v0_timerxl_grad4_head_velocity.py；输出 outputs/train/v0-timerXL-grad-4-lm-head/change-velocity/；远端rage-timerxl环境、tmux v0-timerxl-head-velocity。此补充批次不代表全轨所有窗口的输出头曲线。

v0-timerXL-grad-4-lm-head 的 change-velocity-comparison 补充批次于2026-11-11完成。4761个同窗口输出头梯度、逐k对照表和四张分轨叠加图均已生成并核验；输出头不能整体数值近似LoRA三指标曲线。详见 docs/results/v0-timerXL-grad-4-lm-head-change-velocity.md。

### v0-timer-XL-grad-4-change-velocity：stride-16-rank 补充批次

首次提出：2026-11-11。最后修改：2026-11-11。四轨各从第1个合法窗口开始，每隔16窗口取一个已有4096D BF16投影LoRA梯度。仅对k=16,32,...,512计算同一采样相位上所有有效窗口对的方向余弦、相对L2差、相对模长差，和全量曲线的相同k逐点对照；k<16不可直接识别。以四轨共同k={16,32,48,64}的等k平均值比较三指标排序：方向余弦低者变化快，相对L2差及相对模长差高者变化快。另比较三条长轨k=16..512的排序。为了评估起点敏感性，重复采样相位0..15；正式默认相位0即第1、17、33...个窗口。短轨153点在k较大时有效采样对很少，仅共同前四个k用于四轨排序。读取现有完整缓存，不重新计算模型梯度。代码 scripts/temp/v0_timerxl_grad4_stride16_rank.py；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/stride-16-rank/；远端rage-timerxl环境，tmux v0-timerxl-stride16-rank。

### v0-timer-XL-grad-4-change-velocity：two-random-16-blocks 补充批次

首次提出：2026-11-11。最后修改：2026-11-11。用户撤回每16窗口采一个梯度方案，改为每条轨迹随机选择两个互不重叠、各含16个连续合法窗口的区间。使用固定随机种子0以便复现，仅从四轨现有完整LoRA 4096D BF16投影梯度缓存读取所选128个窗口的梯度，不新增模型计算。每个区间内计算k=1..15的方向余弦、相对L2差及相对模长差，跨区间不配对；每轨每k按两区间合计2*(16-k)对取均值。和全轨全量k=1..15曲线比较逐k绝对差、曲线形态；按15个k等权平均构造各轨变化速度分数，方向余弦越低表示方向变化越快，相对L2/模长差越高表示相应变化越快。检验四轨全量排序能否由两个区间恢复。k=15每轨仅2对，必须注明不确定性。该一次抽样只能判断此次随机区间的表现，不能估计随机抽样成功率。代码 scripts/temp/v0_timerxl_grad4_two_blocks.py；输出 outputs/train/v0-timer-XL-grad-4-change-velocity/two-random-16-blocks/。

two-random-16-blocks 补充批次已完成（2026-11-11）。seed 0每轨两个互不重叠的16窗口区间，共128梯度；三项k=1..15排序都未完整恢复全轨，均将轨迹0和3对调。结果与限制见 docs/results/v0-timer-XL-grad-4-two-random-16-blocks.md；旧stride-16-rank方案已撤回，不作为现行结论。

## v0-timer-XL-grad-64-change-velocity

首次提出：2026-11-11。最后修改：2026-11-11。用户要求扩展至固定src-96前64条轨迹，分别回答：（1）真实LoRA梯度变化速度与轨迹身份的关系、同一轨迹内的时间变化；（2）TimerXL lm_heads 输出头投影梯度对真实LoRA变化速度数值和跨轨排序的代理效果；（3）每轨在约25%和75%时间位置各取一个连续16窗口区间，使用其中的真实LoRA梯度估计全轨变化速度数值与排序的效果。固定上下文512、预测64、步长1，轨迹合法窗口数合计1,980,371，四轨旧LoRA缓存484,907行复用，其余1,495,464行新算。模型TimerXL 84M，基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标q_proj/v_proj、初始化seed 0；两类梯度均Count-Sketch 4096D seed42，BF16位编码存储，不保存原始参数维度梯度。项目使用远端rage-timerxl环境。

数据盘可用434 GiB；两类全部投影梯度纯位数据合计约30.22 GiB，扣除已存在的前四轨LoRA后新增约26.52 GiB，加索引与分片元数据远低于当前余量。输入configs/data/src-96.yaml；配置configs/rage/v0-timerxl-grad-64-change-velocity.json；准备入口scripts/temp/prepare_timerxl_grad64.py；正式输出outputs/train/v0-timer-XL-grad-64-change-velocity/formal/。准备阶段先八worker计算其余60轨LoRA，再八worker计算64轨输出头；各worker不重叠，按全局窗口顺序分片。16窗口两类梯度预演和八worker输出头并发预演完成。原Chronos Uniform四进程暂停且缓存保留；本批成功后自动恢复。调度scripts/temp/run_timerxl_grad64_then_resume.sh、tmux v0-timerxl-grad64。不得自动关机。

分析口径：全部梯度逐轨计算k=1..512的方向余弦、相对L2差、相对模长差，定义同v0-timer-XL-grad-4-change-velocity；未满足N>k的轨迹滞后标为未定义。全64轨统一跨轨排序只使用公共k=1..15，对每个k的均值再等权平均；方向余弦低、相对L2/模长差高分别表示变化快。轨迹内时间变化用连续非重叠64窗口分段的k=1局部速度，报告段间分布和轨迹间/轨迹内方差分解。输出头代理与真实LoRA在各自参数空间计算同口径，比较64轨数值误差、Spearman/Kendall排序、top16重合。两段16窗口在每轨可用区间的25%、75%处确定，不随机；各段内k=1..15有效窗口对的均值估计同轨全量k=1..15曲线，再比较数值、排序与top16。此两段代理无法估计k>=16。完整结果必须依据生成的CSV/JSON核验后写入RESULTS，不预设结论。

#### 64轨梯度缓存自包含整理（2026-11-12）

完成后将前四轨（ordinal 0–3）的 484,907 个 TimerXL LoRA 投影梯度窗口，从 v0-timerxl-src96-first4-full-grad-4096d-bf16 的独立缓存复制到本实验 formal/lora/reused-first4/worker-*/grad_cache/。其余60轨（ordinal 4–63，1,495,464窗口）保留在 formal/lora/worker-*/grad_cache/；64轨输出头缓存保留在 formal/head/。因此本实验的 formal/ 目录现在包含64轨 LoRA 与64轨输出头缓存，共1,980,371个合法窗口，不依赖旧实验路径。

前四轨原件保留在原实验目录；复制后的缓存保留原 cache_manifest.json 和分片文件。formal/collection_manifest.json 记录合并布局。formal/ 下归档引用型符号链接已替换为普通完成标记；核验到 LoRA 共1,944个窗口分片（前四轨480个、后60轨1,464个），formal/ 内无符号链接。此操作仅整理缓存，没有重新计算梯度，也没有改变实验结果或参数。

### v0-timer-XL-grad-64-change-velocity：two-head-16-blocks 补充批次

首次提出与最后修改：2026-11-12。用户要求沿用前批每轨约25%、75%位置的两个连续16窗口区间，改以同窗口lm_heads输出头4096D BF16投影梯度估计k=1..15的方向余弦、相对L2差、相对模长差，并检验64轨全量LoRA的数值和排序恢复。共选2048个已有头梯度，不重新计算模型梯度。每轨每k合并2×(16−k)个区间内窗口对，对k=1..15等权平均构造轨迹分数；与全量LoRA、全量头梯度和同段LoRA三种已有口径比较MAE、平均偏差、Spearman、Kendall、前16名重合。k>=16不可由两段16窗口估计。入口 scripts/temp/v0_timerxl_grad64_two_head_blocks.py；输出 outputs/train/v0-timer-XL-grad-64-change-velocity/analysis/two-head-blocks/；正式结果见 docs/results/v0-timer-XL-grad-64-change-velocity.md。

### TimerXL 梯度时间变化实验链条与综合结论（2026-11-12）

本段汇总上述四轨试验、64轨扩展及输出头补充批次；逐批方法和参数保留在前文，原始结果见 docs/results/v0-timerXL-grad-4-lm-head.md、docs/results/v0-timer-XL-grad-4-curvature.md、docs/results/v0-timer-XL-grad-4-change-velocity.md、docs/results/v0-timer-XL-grad-4-two-random-16-blocks.md 和 docs/results/v0-timer-XL-grad-64-change-velocity.md。所有结论仅针对固定 src-96 样本、TimerXL 84M、4096D Count-Sketch 投影和 BF16 表示。

实验起点是判断输出头梯度能否代表 LoRA 梯度的窗口关系，以及 LoRA 梯度随时间变化是否足够平滑。两类梯度处于不同参数空间，因此代表性试验比较各自空间内窗口对的余弦相似度关系，不直接计算跨空间向量余弦。四轨各等距取128个窗口，轨迹内非临近窗口对的 Spearman 在可评估的三轨上为0.249–0.303；短轨仅153个窗口，没有满足间隔至少576的轨迹内窗口对。这个结果表明输出头只能部分反映 LoRA 的窗口关系，不能据此当作数值等价的梯度。

随后对四轨的全量 LoRA 梯度逐窗口求一、二、三阶时间差，并按16个时间段检查二阶差幅度。二阶差 RMS/原梯度 RMS 依轨约为0.54、0.94、1.53、0.85，三阶差 RMS/二阶差 RMS 为1.59–1.81；二阶差不呈轨迹级近似常定，也没有足够平滑的变化。再对全部有效窗口对计算 k=1..512 的方向余弦、相对 L2 差、相对模长差，并逐轨绘图。四轨试验表明曲线有差异；每轨随机抽两个连续16窗口区间（seed 0）估计 k=1..15 时，三项排序均把轨迹0与3对调，因此不能凭四轨单次随机抽样断定全轨排序可靠。此前每16窗口采一点的 stride-16 方案已由用户撤回，不纳入现行结论。

为检验轨迹差异和估计可靠性，实验扩展到固定 src-96 前64轨的1,980,371个合法窗口；复用前四轨484,907个 LoRA 梯度，新算其余60轨 LoRA 与全部64轨输出头梯度。两类梯度只保存4096D BF16投影，不保存原始维度。64轨的跨轨统一分数是各轨 k=1..15 逐k均值的等权平均；方向余弦越低、相对 L2/模长差越高，表示对应变化越快。轨迹内时间异质性另以非重叠64窗口段的 k=1 局部速度检查，不能把它与 k=1..15 的跨轨分数混为同一个统计量。

| 指标 | 轨迹间方差份额（64窗口段k=1） | 段内CV中位数 | 前后半轨排序Spearman |
|---|---:|---:|---:|
| 方向变化 1−cos | 0.798 | 0.213 | 0.944 |
| 相对L2差 | 0.805 | 0.109 | 0.927 |
| 相对模长差 | 0.487 | 0.289 | 0.836 |

由此可见方向与 L2 变化速度有强烈而相对稳定的轨迹级差异；同轨时间波动仍存在，模长变化的轨迹间区分度较弱、轨迹内波动更明显。方差份额是描述性统计，不是显著性检验，也不意味着任一单窗口梯度可预测。

下表以全轨 LoRA 的 k=1..15 分数为目标，报告64轨平均绝对误差 MAE、平均偏差（估计减真值）、跨轨排序 Spearman 和前16名重合。两段16窗口方案在每轨约25%和75%位置取连续区间，段内按每个 k 的有效窗口对合并，再对 k 等权平均；不能估计 k>=16。方向项的误差以余弦分数计算，与 1−cos 的绝对误差相同，但偏差符号按方向余弦报告。

| 估计方法 | 指标 | MAE | 平均偏差 | 排序Spearman | 前16重合 |
|---|---|---:|---:|---:|---:|
| 全量输出头 | 方向余弦 | 0.158 | −0.112 | 0.912 | 11/16 |
| 全量输出头 | 相对L2差 | 0.135 | +0.053 | 0.916 | 13/16 |
| 全量输出头 | 相对模长差 | 0.160 | −0.160 | 0.557 | 9/16 |
| 两段16窗口真实LoRA | 方向余弦 | 0.058 | −0.002 | 0.909 | 14/16 |
| 两段16窗口真实LoRA | 相对L2差 | 0.071 | −0.002 | 0.880 | 14/16 |
| 两段16窗口真实LoRA | 相对模长差 | 0.068 | −0.006 | 0.609 | 10/16 |
| 两段16窗口输出头 | 方向余弦 | 0.172 | −0.115 | 0.868 | 11/16 |
| 两段16窗口输出头 | 相对L2差 | 0.158 | +0.057 | 0.829 | 11/16 |
| 两段16窗口输出头 | 相对模长差 | 0.167 | −0.161 | 0.294 | 5/16 |

全量输出头对方向与 L2 的跨轨排序有用，但数值存在系统偏差，模长指标较差。同样只取两个16窗口区间时，真实 LoRA 的数值恢复明显优于输出头，方向和 L2 排序仍较好；模长排序仅中等。两段输出头对全量输出头自身的 MAE 为0.056/0.063/0.039，排序 Spearman 为0.938/0.901/0.690（方向/L2/模长），说明稀疏区间能部分恢复输出头自己的速度；它映射到 LoRA 时的系统差异仍在。上述两段位置为一次固定设计，不代表随机抽样成功率。整体结论是梯度变化速度并非完全随机，但时间曲率不够平滑，输出头尤其不能直接替代 LoRA 的模长变化速度。

## v0-chronostiny-timerxl-lora-full-64

首次提出与最后修改：2026-11-12。状态：用户调整后停止，部分梯度缓存保留；无全轨分析结果。目标是在此前固定 src-96 前64轨的所有1,980,371个合法窗口上，计算原始预训练 ChronosTiny 的 LoRA 梯度，并与已完成的同窗口 TimerXL LoRA 梯度检验变化结构是否相关。用户明确要求全窗口计算，先前讨论的每轨128点方案不用于梯度准备。

ChronosTiny 使用原始预训练权重 cache/pretrained_models/amazon/chronos-t5-tiny/29d808298f1a62493e7b9a5e08529d0d930fa189，不使用 v0-grad-mech-test 的100步预热检查点。上下文512、预测64、步长1；基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标q/v、初始化seed 0。全量梯度用Count-Sketch投影到4096D，seed 42，以BF16位编码保存；不保存98304维原始LoRA梯度。TimerXL对照是 v0-timer-XL-grad-64-change-velocity 已完成的同轨同窗口4096D BF16 LoRA缓存。两模型原始参数空间不同，虽然投影维度和seed相同，投影坐标不具有逐维对齐含义；不得将两个4096D向量直接作跨模型余弦来声称模型梯度相关。分析应比较各模型内部的同窗口对余弦关系，以及全轨逐k变化指标与轨迹级排序；轨迹内与轨迹间分别报告。全轨批次完成前不得填写相关性结果。

配置 configs/rage/v0-chronostiny-timerxl-lora-full-64.json；准备入口 scripts/temp/prepare_chronostiny_lora_full64.py；输出 outputs/train/v0-chronostiny-timerxl-lora-full-64/，日志 logs/v0-chronostiny-timerxl-lora-full-64/。运行环境 rage。16窗口预演成功：原始权重可加载、LoRA可构建，投影与BF16缓存写入正常。正式全量计算已于2026-11-12 02:56 CST 在 tmux v0-chronostiny-timerxl-full64 启动，8个worker各处理不重叠的约247546个窗口。512窗口单worker预演用时23.4秒；8 worker同时运行首512窗口的早期ETA约9小时，仅作初步估计。队列 scripts/temp/run_chronostiny_timerxl_full64.sh 将在全部worker及正式完成标记核实后自动运行 scripts/temp/analyze_chronostiny_timerxl_full64.py。分析计算各轨全量k=1..15方向余弦、相对L2差和相对模长差的轨迹级排序，逐窗口相邻方向余弦的同轨时间相关，以及每轨最多128个等距窗口构成的非临近窗口对余弦相关。128点只用于分析窗口对，不改变全窗口梯度准备范围。


## v0-chronostiny-timerxl-lora-track0

首次提出与最后修改：2026-11-12。状态：完成（2026-11-12）。按用户最新要求，仅计算固定 src-96 的第0条轨迹，完整覆盖其230,161个合法窗口。使用原始预训练 ChronosTiny 权重 cache/pretrained_models/amazon/chronos-t5-tiny/29d808298f1a62493e7b9a5e08529d0d930fa189；基础模型冻结，LoRA r=8、alpha=16、dropout=0、目标q/v、初始化seed 0。上下文512、预测64、步长1。LoRA梯度投影到4096D Count-Sketch（seed42），BF16位编码保存，不保存原始参数维度。完成后与TimerXL同轨全部窗口梯度比较；两模型参数空间不同，采用同窗口配对关系和轨迹内时间变化指标，不把投影坐标视作跨模型对齐。配置 configs/rage/v0-chronostiny-timerxl-lora-track0.json；准备入口 scripts/temp/prepare_chronostiny_lora_full64.py；分析入口 scripts/temp/analyze_chronostiny_timerxl_full64.py；tmux v0-chronostiny-timerxl-track0。用户要求GPU资源尽量跑满后，已改为8个worker并行，每worker负责约28,770个不重叠窗口；worker0复用之前单worker已完成的窗口缓存。启动后各worker首批512窗口均已进入进度阶段。单worker512窗口预演约23秒；8 worker早期并行阶段测得首批处理约1.2秒，仍需以持续吞吐更新ETA。结果：230,161/230,161个窗口梯度及分析目录.complete均已核验。每轨128个等距窗口构成8,128个非临近窗口对（窗口起点间隔至少576），两模型内部窗口对余弦的Spearman=0.0758、Pearson=0.0768；Top 10%窗口对重合率14.4%。全量相邻窗口对余弦Spearman为−0.1379。轨迹级排序因只有一条轨迹而不可估计。k=1时TimerXL/ChronosTiny平均方向余弦0.9314/0.9348，相对L2差0.4085/0.3574，相对模长差0.1799/0.0478。两模型投影坐标没有跨模型对齐含义；数值仅用于各自内部窗口关系与时间曲线比较。单轨结果不足以判断跨轨普遍相关性。正式输出 outputs/train/v0-chronostiny-timerxl-lora-track0/；分析表为 analysis/pair_geometry.csv、lag_metrics.csv、summary.json。


## v0 汇总与归档核对（2026-11-12）

本次按实际结果整理全部 v0 专题，汇总入口为 [v0 结果总览](results/v0-consolidated-summary-20261112.md)，批次级结果仍以本文件前文和各专题报告为准。ordinal 0 ChronosTiny 批次完成；非临近窗口对余弦 Spearman=0.0758，相邻窗口对余弦 Spearman=-0.1379。64 轨 ChronosTiny 扩展已按用户调整停止，不把未完成缓存当作结果。当前进程检查未发现 v0 实验计算进程；已有批次按完成标记、日志和专题报告分别标为完成、停止或部分完成。

2026-11-12 已将所有非缓存 v0/v00 输出文件与日志归入 archive/v0-consolidated-20261112/。为维持报告中记录的输出路径可访问，原位置保留指向归档文件的符号链接。grad_cache/、gradients/ 等可供后续分析的梯度缓存原位保留，包括未完成批次的部分缓存。v0 配置、脚本及报告副本存于归档 source-and-report-snapshot/；项目源代码、配置与本文件继续留在原处。归档文件数量、字节数和缓存保留规则见归档 README.md。

最后修改日期：2026-11-12

### v0-crb-final-test-timerXL-64tr-spacing-q-dir

首次提出：2026-11-12；最后修改日期：2026-11-12。输入沿用 spacing-q 的64轨、1,980,371个有效窗口、4096D BF16 LoRA投影梯度缓存，不重新计算梯度。全局一次性以 float32 计算全部梯度模长，取第1百分位阈值；模长小于或等于阈值的真实方向置零，记录阈值、置零数量和比例。其余梯度除以各自模长，后续 CRB 选点、Uniform 对照、端点插值和离散度计算全部使用方向向量。零真实方向或零重建代表的余弦定义为0，因此对应误差为1。

q={2,4,6,8,12,16,24,32,64}，每轨 K=max(2,ceil(N/q))，包含首尾锚点；Uniform 按相同 K 等距取点。CRB coverage:error:split 依次为 1:0:0、1:0:1、1:1:0、1:1:1，按原最大余数法分配 K-2 个额外锚点。闭区间块包含两个端点，相邻块共享锚点。用两个端点的单位方向线性插值，块代表为插值向量的均值；逐窗口计算 1-cos(真实方向,块代表)。同时报告按块长度加权的误差与各块等权的误差，保留逐轨、逐块和总体结果。计算沿用 BF16 源值及 float32 归约。

| 批次 | 工作 | 状态 | 产物 |
|---|---|---|---|
| 阈值与四比例队列 | 全局第1百分位阈值；顺序运行四种 CRB 比例及匹配 Uniform | 全部完成，四比例及顶层完成标记已核验 | outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q-dir/ |

入口 scripts/temp/v0_crb_final_test_timerxl_64tr_dir.py；队列 scripts/temp/queue_v0_crb_final_test_timerxl_64tr_dir.sh；环境 rage-timerxl。各比例以及顶层 .complete 生成后才视为完成。禁止自动巡检。

### spacing-q-dir q=128,256 追加批次
首次提出：2026-11-12；最后修改日期：2026-11-12。沿用原方向阈值、64轨缓存、四种CRB比例、匹配Uniform、方向线性重建与两种加权误差，仅将q改为128和256。沿用原预算公式 K=max(2,ceil(N/q))，每轨含首尾锚点；实际相邻锚点平均间距为(N-1)/(K-1)，CRB局部间距可不等。新产物独立写入 outputs/train/v0-crb-final-test-timerXL-64tr/spacing-q-dir/extra-q-128-256/，每比例及顶层完成标记为完成条件。状态：已完成，四比例及顶层完成标记均已核验；结果见 docs/results/v0-crb-final-test-timerXL-64tr-spacing-q-dir.md。

## v1-crb-redo

首次提出：2026-11-12；最后修改：2026-11-12 23:38 CST。

### grad-prepare

首次提出与最后修改：2026-11-12。状态：2026-11-12 23:00 CST 前三模型热身、CRB 与 Uniform 梯度及整批完成标记均已生成。此前 21:39 的 Chronos 进度是历史检查值。当前批次只负责三模型热身检查点以及 src-96 候选池 CRB/Uniform 梯度素材；后续数据选择、正式训练、评估另立批次。

数据：热身使用 data/src/shard_dataset/src-8192/ 与 configs/data/src-8192.yaml 的 8192 条轨迹。每次先等概率有放回抽取一条轨迹，再在该轨迹全部合法 576 点窗口中等概率抽取一个窗口；不按配置或轨迹长度额外均衡。候选梯度只使用固定 data/src/shard_dataset/src-96/ 和 configs/data/src-96.yaml 的 96 条轨迹；该集合与 src-8192 有包含关系，允许热身样本与候选窗口重叠。两者均保持已有原始数值快照，不添加预处理。所有模型窗口总长 576、起点步长 1；Chronos-T5-Base 和 TimesFM 2.5 以 512 历史加 64 预测计算损失，TimerXL 84M 使用其已有原生 V6 480 历史加 96 预测损失。模型间的目标不同，方法内 CRB/Uniform 同模型目标一致。

三个模型分别从原始预训练权重及 seed 0 LoRA 初始化热身。基础权重冻结，仅训练 LoRA；r=8、alpha=16、dropout=0；Chronos 目标 q/v，TimerXL 与 TimesFM 目标 q_proj/v_proj。各训练 1000 次优化器更新，有效批次 32；Chronos 微批次 32，另两模型微批次 8；AdamW、学习率 1e-4、权重衰减 0、betas=(0.9,0.999)、eps=1e-8、梯度裁剪 1、cosine 学习率日程与 3% 预热。日程沿用原 4000 步正式训练时间轴，在第 1000 步截取检查点。保存三份 LoRA 检查点及基础权重身份、数据和采样协议、训练参数、seed、实际步数；重新加载后核对固定窗口损失和梯度。

全局称呼预算 q 的定义为平均每 q 个合法候选窗口允许计算一次梯度，本批 q=32。逐轨迹固定 CRB coverage:error:split=1:0:1，error 阶段标为 deprecated。coverage 包含端点，split 沿用现有 min_kappa 规则；Uniform 按相同轨迹、相同实际梯度点数等距选点并包含端点。逐轨取整，两方法必须逐轨预算匹配；若 CRB 未使用全部预算，不能标记该轨完成，需先记录原因及补足规则。对当前 src-96 按逐轨取整规则，每模型每方法计划 83,330 锚点，coverage 与 split 各 41,665；两方法名义预算合计每模型 166,660。每模型仅对可训练 LoRA 参数求逐窗口损失梯度；固定 4096 维 Count-Sketch、seed 42、FP32 投影累加、BF16 位编码保存。两方法同一窗口的梯度允许共享计算，分别记录名义方法预算与真实唯一反传次数。

旧 v0-grad-prepare-src-96tr-q16-4096d 的无热身缓存、锚点清单和完成标记按用户要求删除；不删除 src-96/src-8192 数据快照、历史文档或其他实验产物。正式运行依次完成三模型热身、重载校验、两方法同预算梯度准备与逐轨验证；仅全部完成后写新批次 .complete。使用现有 rage Conda 环境，长任务在 tmux 中运行；日志放 logs/，检查点及梯度放 outputs/，大文件由项目软链接落在数据盘。不设置自动巡检，不关机。
EOF'

### timerXL-baseline-full

批次首次提出：2026-11-12 21:59 CST；批次最后修改：2026-11-12 23:38 CST。状态：4,000 步训练、LoRA、训练清单、合并权重及 12 项评估均完成；结果见 `docs/results/v1-crb-redo-timerxl-baselines.md`。首次 FEV 因数据缺失失败的记录保留于 LOG.md。目标：固定 src-96 全量窗口训练基线。

从 `/root/autodl-tmp/pretrained_models/timer-base-84m-84m` 的 TimerXL 84M 原始预训练权重和 seed 0 独立初始化，不加载 grad-prepare 热身检查点。基础权重冻结，仅训练 BF16 LoRA：目标 `q_proj`、`v_proj`，r=8、alpha=16、dropout=0。数据为 `configs/data/src-96.yaml` 与 `data/src/shard_dataset/src-96/` 的 96 条完整轨迹；每轨窗口起点步长 1、窗口长 576，共 2,666,312 个合法窗口，无额外预处理。训练每次更新从全部合法窗口全局等概率有放回抽样 32 个窗口，seed 0。TimerXL 原生 V6 损失：480 历史点、96 预测点。

恰好 4,000 次优化器更新；有效批次 32、微批次 8、累积 4。AdamW：学习率 1e-4、weight decay 0、betas=(0.9,0.999)、eps=1e-8；梯度裁剪 1.0；cosine 学习率日程覆盖 4,000 步，3% 即 120 步预热，min_lr_ratio=0。模型、LoRA、窗口采样均使用 seed 0。每 500 步保存恢复状态；保存第 4,000 步 LoRA 并导出合并模型；仅最终步评估。

固定下游 12 任务：`configs/eval/fev/v1-crb-redo-timerxl.yaml` 的 FEV task indices 9、11、18、19；`configs/eval/time/v1-rage-trev-timerxl.yaml` 的 TIME short-term 任务 CPHL/15T、CPHL/30T、CPHL/H、ECDC_COVID/D、ECDC_COVID/W、epf_electricity_price/H、Commodity_Import/M、Commodity_Production/M。评估数据、预测长度、指标和推理参数沿用这两份既定配置：FEV 用 `v1-crb-redo-timerxl-tasks.yaml` 中沿用 `default.yaml` 原定义的四个固定 task index、期望任务数 4；FEV 推理 BF16、batch 32、single_window、seed 20260722。TIME 用显式八任务且各取 short term；推理 BF16、batch 32、context 512、quantile levels 0.1 到 0.9（步长 0.1）、single_window、seed 20260722；TIME 源代码修订 `b6e7ba9862acb9233c8c55b314c1777052c420ff`，数据集 `Real-TSF/TIME` 修订 `83e3d0b3be28d11c7182bffcc1892d19b36c4da1`，目录 `data/bench/time/v1-rage-trev/dataset`。FEV 的四个实际任务 parquet 固定来自 `autogluon/fev_datasets` 数据集修订 `d3e7e720f4158eab03ebcb021f1cce779cd92acc`，项目目录 `data/bench/fev/datasets/v1-crb-redo/`；在 `v1-crb-redo-timerxl-tasks.yaml` 中仅将这四项的 dataset_path 改为本地 parquet，其余任务定义、horizon、指标、窗口数保持原配置。项目 FEV 入口仅对明确选中的任务要求本地数据，但仍检查全部 20 项的指标声明；不修改第三方包。各任务的具体 horizon、指标与数据定义以这两份锁定的评估配置及其引用的任务定义文件为准，不改动评估标准。入口 `scripts/v1_crb_redo_timerxl_baselines.py --batch timerXL-baseline-full`；队列 `scripts/temp/run_v1_crb_redo_timerxl_baselines.sh`。训练输出 `outputs/train/v1-crb-redo/timerXL-baseline-full/`；评估输出 `outputs/eval/v1-crb-redo/timerXL-baseline-full/`；日志 `logs/v1-crb-redo/`。训练及两个基准全部成功后才写批次 `.complete`。

### timerXL-baseline-rand01

批次首次提出：2026-11-12 21:59 CST；批次最后修改：2026-11-12 23:38 CST。状态：固定随机窗口、4,000 步训练及 12 项评估均完成；结果见 `docs/results/v1-crb-redo-timerxl-baselines.md`。目标：与全量窗口基线对照。该批从同一原始预训练权重和 seed 0 独立初始化，不加载 full 或 grad-prepare 的训练权重。其模型、LoRA、数据原值、窗口定义、原生 V6 损失、优化器、学习率日程、批次大小、4,000 步、恢复间隔和 12 项评估均与 full 批次相同。

在 2,666,312 个全局合法窗口索引 `[0, 2666312)` 上，用 `numpy.random.default_rng(0).choice(N, size=N//10, replace=False)` 一次性等概率无放回抽取 266,631 个索引，升序保存到 `outputs/train/v1-crb-redo/timerXL-baseline-rand01/fixed_window_indices.npy`；重启读取这份索引，不重新抽样，并核验范围、数量与严格递增。之后每次更新仅从固定集合等概率有放回抽取 32 个窗口，训练采样 seed 0。较长轨迹因合法窗口更多而自然占更大比例。训练输出 `outputs/train/v1-crb-redo/timerXL-baseline-rand01/`；评估输出 `outputs/eval/v1-crb-redo/timerXL-baseline-rand01/`；日志 `logs/v1-crb-redo/`。训练及 12 项评估成功后才写批次 `.complete`。
