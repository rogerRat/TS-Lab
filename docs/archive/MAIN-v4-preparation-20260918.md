# MAIN-v4 — 实验协议与编排准备

- 状态：prepared，尚无V4训练/评估结果。
- 执行配置：configs/rags/main_v4.json。

## 2026-09-18 V4范围与编排：两模型、自然数seed逐轮串行

本节为最新实验范围与编排决策，覆盖此前“三模型、只做seed0”的新意见聚合矩阵规模。聚合公式不变：RAGS Linear/ZeroClip/Max先按双层alpha加权，Exp先exp(c/1.00)再按alpha加权加和，外层Softmax .05；LESS-Main保留原始余弦Max。13方法、九任务不变，不指定任何RAGS主方法。

V4暂时只做chronos-base和timesfm-2.5。TimeMoE-200M因当前微调困难退出V4，不运行它的准备、训练或评估；历史结果、材料和实现保留。替代模型尚未确定，本轮不自行选择或加入新模型。此前正在运行的V3批次沿其原冻结配置处理，本次不停止或修改历史批次。

编排制度：seed固定自然数0、1、2、3、4……，从0开始逐轮串行；必须先完成同一seed下两个模型×九任务×全部训练方法的训练和评估，再放行seed+1。并行放在当前seed的一轮内部，两模型并行，不能让较快模型提前进入下个seed。失败、缺项、身份不符或非有限MASE都会阻止跨轮；续跑从最早未完成的轮次继续，已完成项按原有续跑逻辑跳过。父套件使用单例锁，避免两个编排进程分别推进不同seed。

没有指定总seed数，不擅自把01234限定为五轮。默认持续自然数序列；--max-seeds只提供可选的运行次数上限，不是当前科学协议的一部分。各轮位于独立rounds/seedK目录，配置seeds只包含[K]，训练checkpoint、结果和seed身份独立。

第一轮为234格=18 orig基线+216训练/评估格（两个模型×九任务×12训练方法）。orig与训练seed无关，在第一轮训练前测量，后续只引用首次测量结果。每增加一个seed，新增216训练/评估格；S个seed的唯一测量总数为18+216×S。每轮的完整展示仍包含18个基线引用。原351格规模是旧三模型seed0设计，不能再作为V4规模。

训练seed控制PyTorch随机数、CUDA随机数与训练时源窗口抽样。源池pool_seed=0、4096维投影seed42、固定参考、梯度材料及静态选择概率保持不变；后续seed复用已完成seed0的静态概率选集，并采用自己的训练/抽样seed，不是重复引用训练结果。已有字段校验梯度缓存按轮链接复用，不重新为每个seed计算梯度，不新增TimeMoE梯度任务；现有缓存的单写者锁仍有效。

远程配置：configs/rags/main_v4.json。入口：python -m rags.experiments.v4_seed_rounds --config configs/rags/main_v4.json --run <数据盘V4父目录>。--dry-run仅打印计划，不加载模型、不启动任务。正式入口须遵守父目录execution_control.json运行开关；本次修改与验证不启动V4实验。父state.json记录当前seed、完成seed及下一seed；各轮结果沿用原cell JSON/selection NPZ格式，无hash，协议与字段明确校验、JSON原子发布。

验证状态：77项相关回归测试通过（3条既有Swig弃用警告）。验证了跨模型轮次屏障、自然数seed配置、失败/缺项阻止下一轮、续跑、静态材料与orig复用、实际seed汇总及旧V3兼容。编排dry-run为两模型、首轮234格、后续每seed新增216格、jobs_launched=0。本次未启动V4实验，未提交/Push。
