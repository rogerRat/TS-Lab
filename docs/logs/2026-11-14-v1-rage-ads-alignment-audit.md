# v1-rage-ads 滚动审计方案与实现记录

- 检查时间：2026-11-14 03:39 CST。
- 实验代号：v1-rage-ads。
- 批次：rolling-eight-task。
- 状态：八任务 rolling 安全历史输入已生成；选择 runner、审计计算、训练与评估尚未完成。
- 实验臂：fullRand、rage、ancUni、Traj；用户确认当前先执行这四臂，后续新增对比追加到本实验，不覆盖历史批次。

## 本次操作

- 在 docs/exp/v1-rage-ads/exp.md 记录四训练臂及其候选池/重采样口径。
- 新增 source 对齐、每臂 4,000 次默认抽样对齐、rolling reference 连续性和 ADS 分数矩阵审计定义。
- Source 基线定义为 v1-crb-redo q=32 uniform 缓存锚点；每条窗口分数为对任务 rolling-safe reference 锚点的最大余弦。
- 每任务每臂按真实训练采样机制有放回抽样；固定默认 seed=42，重复抽样按出现次数计入统计，梯度按唯一窗口去重计算。
- Reference 连续性定义为任务真实预测长度 H 下，每个 rolling 测试起点与安全历史 q=32 梯度锚点集合的最大余弦。仅作事后诊断，不反馈 reference、ADS 或训练样本。
- ADS score matrix 报告每轮/最终矩阵的全量精确零、近零和正负比例、均值/标准差/范围及估计分位数；另单独报告 Top-rho 匹配掩码稀疏度。
- 新增 scripts/exp/v1-rage-ads/audit_metrics.py；脚本会校验 source q=32 缓存的 warm-up、窗口、梯度协议、LoRA 和投影字段，并要求选择阶段提供每任务 reference_coarse.npz、四臂 sampling.json 和 score_matrix_summaries。
- ADS Top-K 匹配改为仅保留匹配端点掩码与配对计数，避免物化大规模 Python 配对列表；相似度矩阵按 GPU 分块计算，并生成稀疏度/分位数摘要。
- ADS rounds 支持 0，用于 ancUni 只做均匀粗锚点匹配、不执行细锚点阶段。
- max 重叠邻域规则保持为取最大贡献，不相加。

## 计算与结果

- 本次没有启动模型梯度计算、训练、评估或审计脚本。
- 当前没有新增指标数值；不推测 source 对齐提升或 reference 连续性。
- 既有 ads-api-q-single 批次的 25 项测试对应其先前算法接口版本。本次新增大矩阵摘要、Top-K 命中掩码和 rounds=0 尚未运行测试。
- 本次未生成新的 .complete 文件。

## 下一步

- 实现专用八任务选择 runner，输出 ADS q=16 reference 粗梯度、四臂 sampling manifests 与选择诊断。
- 完成 warm-up 身份字段核对后运行 audit_metrics.py；再继续训练与 rolling 评估。
- 新结果写入 outputs/selection/v1-rage-ads/rolling/audit/，更新本实验 results.md 和全局索引。
