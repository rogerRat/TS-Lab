# 历史归档

这里保留旧设计、实验数值、决策与开发交付证据。文内“当前”“权威”“必须执行”均指当时版本，
不授权恢复已退役算法、运行旧脚本或覆盖当前配置。当前入口见 [文档索引](../README.md)。

| 范围 | 内容 |
| --- | --- |
| [MAIN-v1](MAIN-v1.md)、[MAIN-v2](MAIN-v2.md) | V1/V2完整主实验文档、结果与决策；后续仍可参考 |
| [Progress](Progress.md)、[decisions](decisions.md) | 历史开发进度和版本决策 |
| [实验总控记录](exp_rec.md) | 0913以前实验接管、运行与结果证据 |
| [旧FEV实验记录](exp/exp_rec.md) | 早期Random/OT/KDE逐检查点数值，完整保留 |
| [关机整理记录](shutdown_cleanup_0914.md) | 历史产物位置及恢复线索 |
| `dev/` | 旧OT/KDE研究设计、三学生固定预算交付、旧优化测量等 |
| `sop/` | 已退役训练与AnaSeg工作流 |

路径整理保留旧目录层级，便于寻找同阶段文档；正文中的旧源码/配置路径是历史证据。
引用已删除文件的说明已重定向到其当前替代页；更早就不存在的文件和外部工作目录不据此重建。

## 本次移除的重复或退役说明

| 原文档 | 处理依据/当前替代 |
| --- | --- |
| `cache_layout_0914.md` | 内容并入梯度资源页 |
| `dev/GradientModule.md` | 旧Tiny入口已由梯度资源CLI替代 |
| `dev/GradientGeometryDiagnostic.md` | 依赖已退役Feature Bundle诊断链 |
| `dev/TsDatasets.md` | 保留次日扩充版，移至reference/DataSourcesSurvey.md |
| `todo.md` | 已声明为历史的Catalog/特征网格待办；决策与进度归档保留 |
| `HyperParameter-Computation.md` | 已退役AnaSeg/OT计算量推算 |
| `sop/SOP-1-data.md` | 当前Fetch/Load/Dedup入口集中到DataPreparation.md |
| `sop/SOP-1.1-reference.md` | 当前安全参考历史与入口集中到DataPreparation.md及探测协议 |
| `sop/SOP-3-pca-population-visualization.md` | 依赖已退役Catalog/PCA生产链 |

完整原文可从清理前快照或Git历史恢复，恢复位置见[文档入口](../README.md)。

2026-09-18：V3/V4 主表及 V5 逐次方法裁定归档至本目录和 `method-decisions/`；现行唯一方法文档是 `../method.md`，旧分数/温度/P10/随机并列方案不再有效。
