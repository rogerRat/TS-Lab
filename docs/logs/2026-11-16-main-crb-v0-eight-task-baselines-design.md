# 2026-11-16 main-crb-v0 八任务基线方案记录

- 操作时间：2026-11-16 18:14:18 CST
- 实验代号：main-crb-v0
- 批次：q16-src256-eight-task-baselines
- 操作范围：仅在远程 TS-Lab 项目补充实验设计与全局索引，没有修改代码、配置、训练数据或已有结果。
- 任务状态：六种基线均未启动；本次没有启动选择、训练、评估，也没有查询既有队列。既有 union-reference 状态沿用此前文档记录，不在本次核验。

## 记录内容

- Rand：固定 PCG64(seed=0)，在 6,382,020 个 source 窗口中无放回选 638,202 个，单模型评估八个任务。
- TSDS：记录冻结编码器轨迹特征、FAISS IVF/KDE/OT 原生概率参数、q16/q8 锚点适配、top-25 概率截断和轨迹内随机抽窗规则。
- LP-Mirror：记录线性 proxy 训练、ref 适配、loss drop 打分、全 source 窗口扫描和 topK 预算。
- GraNd-static：复用 UNI q16 锚点投影梯度范数，按区间取 top 10%，单模型评估八个任务。
- CRB-traj 与 UNI-traj：记录各自 q16 source 锚点、q8 ref 轨迹表示、top-25 支持集、等概率轨迹抽样及轨迹内随机抽窗。
- 六种基线合计规划 34 个预测模型训练和 48 个方法-任务评估单元；当前没有任何实际指标或结果文件。

## 修改文件

- docs/exp/main-crb-v0/exp.md
- docs/EXP.md
- docs/RESULTS.md
- docs/logs/2026-11-16-main-crb-v0-eight-task-baselines-design.md

## 下一步

- 若后续实现入口，先创建并审阅配置/脚本，再由用户另行决定启动实验；本次不启动实验。
