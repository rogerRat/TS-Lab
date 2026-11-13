# v0 实验结果总览

最后修改日期：2026-11-12

## 总体结论

v0 的主要证据覆盖完整梯度几何、低成本梯度代理、均匀与自适应锚点重建、重建误差分解、reference-source 排序和 TimerXL 梯度时间变化。结论均限于对应报告中的模型、数据子集、投影维度和缓存精度。

- 完整梯度随窗口滞后增加而方向相似性下降；在百万级 M5 窗口对统计中，短滞后并不意味着完整向量 L2 差很小。
- 输出头梯度比完整反向传播便宜约 2.93 倍，但代理指标没有普遍复现完整 LoRA 梯度的变化曲线或跨轨排序。
- 在 M5 LoRA 重建中，CRB 在不同轨迹、预算和指标上的结果混合。12 条选定 FOODS_1 轨迹的逐窗口方向余弦在所有非平凡预算均偏向 Uniform；不能从块内离散度改善推出点对点插值误差改善。
- 方差驱动的 VAGE/DP 变体未在所有目标指标上支配 Uniform。平滑 mock 中的 CES 和虚拟切分收益依赖变化结构与加权口径；高变化 mock 上没有稳定的 cancellation 优势。
- reference-source 代理排序有一定关联，但遗漏的高价值窗口可落在代理候选池较深处；固定参考集合上的查询回放不构成 held-out 或下游训练收益证据。
- TimerXL 梯度时间变化实验已完成四轨和 64 轨分析。64 轨 LoRA 每轨 PC1 解释率中位数为 12.1%，PC1 方向两两夹角中位数为 87.3%；两段 16 窗口输出头代理的排序恢复程度因指标而异。
- ChronosTiny 与 TimerXL 的 ordinal 0 全轨分析完成；非临近窗口对余弦相关 Spearman 为 0.0758，单轨不能支持跨轨普遍相关。ChronosTiny 64 轨扩展按用户调整停止，部分缓存仅供后续使用，不纳入分析结论。

## 专题报告索引

| 专题 | 报告 |
|---|---|
| 梯度几何、轨迹尺度与方差分解 | [机制 L2](v0-grad-mech-l2-norm.md)、[轨迹尺度](v0-grad-traj-level-norm-gap.md)、[方差分解](v0-gradient-variance-decomposition.md) |
| 代理成本、风险与排序 | [代理成本](v0-gradient-proxy-cost.md)、[代理评估](v0-gradient-proxy-test.md)、[价值排序](v0-active-value-rank.md)、[排序诊断](v0-active-value-rank-diagnostics.md) |
| 均匀、VAGE、CRB 与方向增益 | [均匀尺度](v0-uni-reconst-scale.md)、[VAGE](v0-vage.md)、[CRB 方差](v0-crb-var.md)、[CRB 重建](v0-crb-reconst-test.md)、[CRB 精细重建](v0-crb-reconst-fine.md)、[CRB 方向离散](v0-crb-cos-var.md)、[方向边际收益](v0-directional-marginal-gain.md)、[均匀数据评分](v0-uni-data-evaluation.md)、[代理 DP](v0-proxy-dp-segmentation.md) |
| 平滑与高变化 mock 机制 | [CRB/CES/虚拟切分](v0-crb-mech-test.md) |
| TimerXL 4 轨/64 轨变化与曲率 | [四轨变化](v0-timer-XL-grad-4-change-velocity.md)、[曲率](v0-timer-XL-grad-4-curvature.md)、[随机短块](v0-timer-XL-grad-4-two-random-16-blocks.md)、[输出头几何](v0-timerXL-grad-4-lm-head.md)、[输出头变化曲线](v0-timerXL-grad-4-lm-head-change-velocity.md)、[64 轨分析](v0-timer-XL-grad-64-change-velocity.md)、[64 轨 PCA](v0-timerxl-grad64-pca.md)、[64 轨 CRB 最终比较](v0-crb-final-test-timerXL-64tr.md) |
| ChronosTiny 跨模型补充 | [专题报告](v0-chronostiny-timerxl-20261112.md)：ordinal 0 完成结果与 64 轨扩展停止状态 |

## 归档

非缓存 v0/v00 输出与日志已整理到 archive/v0-consolidated-20261112/。可能后续使用的梯度缓存保留在原项目路径，原输出引用路径通过符号链接继续可访问。归档索引见归档目录 README.md。
