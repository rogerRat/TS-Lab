# 实验

最后修改日期：2026-11-10

## 实验与批次总表

本表逐行保留所有已登记批次，并核对 v1 的 seed 1–3 批次。数值摘自原始登记；完整参数、方法和结果记录保存在 [英文历史原文](EXP.english-archive-20261110.md)，各实验的正式结果见 [RESULTS.md](RESULTS.md)。队列状态以本次核实的完成标记和日志为准。

| 实验代号 | 批次 | 工作 | 状态 | 实际结果 |
|---|---|---|---|---|
| v0-crb-var | 1-ten-trajectory-replay | 按原定批次执行 v0-crb-var 的实验 | 完成 | 10/10、0.6410、0.6802、0.4051、0.4402；详情见历史原文与结果文档 |
| v0-proxy-dp-segmentation | longest4-unweighted | 按原定批次执行 v0-proxy-dp-segmentation 的实验 | 完成 | 4/4、0.5031、0.6762、0.2995、0.4298；详情见历史原文与结果文档 |
| v0-proxy-dp-segmentation | longest4-window-weighted | 按原定批次执行 v0-proxy-dp-segmentation 的实验 | 完成 | 4/4、0.5717、0.6762、0.3572、0.4298；详情见历史原文与结果文档 |
| v1-rage-chronosT5-trev-base-rand | 1-random-10pct | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | 完成 | 已核实 .complete；详细指标见评估产物 |
| v1-rage-chronosT5-trev-base-full | 1-full-pool | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | 完成 | 已核实 .complete；详细指标见评估产物 |
| v1-rage-timerXL-trev-84m-rand | 1-random-10pct | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | 完成 | 已核实 .complete；详细指标见评估产物 |
| v1-rage-timerXL-trev-84m-full | 1-full-pool | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | 完成 | 已核实 .complete；详细指标见评估产物 |
| v1-rage-timesFM-trev-2.5-rand | 1-random-10pct | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | 完成 | 已核实 .complete；详细指标见评估产物 |
| v1-rage-timesFM-trev-2.5-full | 1-full-pool | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Completed | 4,000 updates; all eight FEV/TIME checkpoints; .complete verified |
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
| v1-rage-chronosT5-trev-base-rand-seed1 | 1-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Terminated | Stopped before step 4000; latest logged step 2445; partial outputs removed |
| v1-rage-chronosT5-trev-base-full-seed1 | 1-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timerXL-trev-84m-rand-seed1 | 1-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timerXL-trev-84m-full-seed1 | 1-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timesFM-trev-2.5-rand-seed1 | 1-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timesFM-trev-2.5-full-seed1 | 1-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-chronosT5-trev-base-rand-seed2 | 2-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-chronosT5-trev-base-full-seed2 | 2-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timerXL-trev-84m-rand-seed2 | 2-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timerXL-trev-84m-full-seed2 | 2-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timesFM-trev-2.5-rand-seed2 | 2-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timesFM-trev-2.5-full-seed2 | 2-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-chronosT5-trev-base-rand-seed3 | 3-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-chronosT5-trev-base-full-seed3 | 3-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timerXL-trev-84m-rand-seed3 | 3-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timerXL-trev-84m-full-seed3 | 3-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timesFM-trev-2.5-rand-seed3 | 3-rand | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；随机 10% 数据池 | Cancelled | Not trained; prepared inputs and configs removed |
| v1-rage-timesFM-trev-2.5-full-seed3 | 3-full | 训练 BF16 LoRA 4,000 步，并在八个步骤完成 FEV/TIME 评估；完整数据池 | Cancelled | Not trained; prepared inputs and configs removed |

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

首次提出与最后执行时间、逐批次参数及原始修改记录见 [英文历史原文](EXP.english-archive-20261110.md)。评估 32–576 起点区块内稀疏输出头代理对真实梯度方差和方向离散度的排序能力，并评估跨轨候选排序及目标离散程度。各批次完成；详情见 docs/results/v0-gradient-proxy-test.md。

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
