# FEV 静态特征匹配：当前结果、边界与解释

> 历史归档：保留当时的设计、操作或实验证据，不作为当前执行指令。当前入口见 `docs/README.md`。

> 本文档总结 FEV 20 个任务、Rand/OT/KDE、1K--10K 步实验的离线诊断结果。它是当前结果的解释性记录，不把 task-level 相关性当作因果证明，也不让 benchmark future 参与训练或数据选择。

## 1. 当前方法实际匹配什么

对每个 reference 局部历史窗口 `X`，TS3 提取 16 维连续描述符 `Z = h(X)`：趋势、局部动态、频谱、分布形态等。OT/KDE 根据 `Z` 对 source 加权，其目标是让加权 source 的静态 descriptor 分布接近 reference：

```text
p_weighted-source(Z) ~= p_reference(Z)
```

这里的“边缘匹配”主要不是指逐个一维特征分别匹配；OT/KDE 使用完整的 16 维 descriptor 向量。它指的是匹配时只保留历史描述符 `Z`，而没有约束未来答案 `Y`：

```text
当前方法匹配：       p(Z)
预测真正需要：       p(Z, Y) 或 p(Y | Z)
时间演化还需要：     p(Z_(t+1) | Z_t)
```

因此，传输后 descriptor gap 下降只说明静态历史特征更接近，不能单独说明未来预测会改善。当前诊断中的 post-transfer gap 是加权 source/reference **特征均值差**；它比完整的 `p(Z)` 匹配检查还弱，不能作为 forecast utility 的证据。

### 高异质 reference 中为什么整体匹配仍会错

若 reference 包含两个 regime（例如低频平稳 `A` 与高频波动 `B`），整体静态分布可写为：

```text
p_reference(Z) = 0.5 p_A(Z) + 0.5 p_B(Z)
```

全局 OT/KDE 可以得到同样为 50/50 的 source 混合，因此 `p(Z)` 看起来匹配成功；但它仍不知道：

- `A/B` 在 reference 中出现的时间顺序，以及预测边界当前处在哪个 regime；
- source 中同一 descriptor regime 后面的 future 是否遵循相同规律；
- 同一 descriptor 区域是否混有不同机制（噪声、周期、促销尖峰、结构突变等）；
- 不同 regime 对模型参数给出的训练梯度是否彼此冲突。

所以“整体静态分布相同”不等于“每个内部 regime、其转移规律和未来效用都相同”。这就是高离散度任务中 global static reweighting 的主要风险。

## 2. 已完成的离线分析与证据强度

结果文件位于：

- `outputs/analysis/fev-anaseg-checkpoint-curves-1k-10k/diagnostics.csv`
- `outputs/analysis/fev-anaseg-checkpoint-curves-1k-10k/experiment/improvements.csv`
- `outputs/analysis/fev-anaseg-checkpoint-curves-1k-10k/experiment/diagnostic_correlations.csv`

`improvement = Rand error - method error`；正值表示 OT/KDE 优于 Rand。以下相关性是 Spearman task-level 相关性。

| 分析维度 | 当前测量 | 与 OT/KDE 相对 Rand 提升的关系 | 结论强度 |
| --- | --- | --- | --- |
| Reference 特征均值/分位数 | 每个 descriptor 的 mean/p10/p90 | 没有发现比离散度更稳定的总体信号 | 弱 |
| **Reference 特征离散度** | 跨局部 segment 的 descriptor std | 多个 descriptor 在 OT、KDE 与 MASE/WQL 上均呈负向关系 | **强，当前最重要** |
| Reference 局部时间变化 | 相邻 segment 的 16D 距离 mean/p90 | 方向大多为负，但仅 12 个 task 可算且大多不显著 | 中等偏弱，趋势性证据 |
| 历史--答案差异 | 安全历史与答案的 descriptor gap | 不是单调负向 predictor；OT MASE 甚至为正相关 | 弱，不能作为 gate |
| Source--reference 传输前差距 | nearest distance、mean feature gap | 没有稳定的总体关系；尾部覆盖可能比均值更重要 | 弱到中等 |
| 传输后特征 gap | OT/KDE 加权后 feature mean gap reduction | 20/20 task 都下降，但好坏 task 都存在 | 明确证明“匹配发生”，不能证明“预测受益” |
| 被舍弃 reference 质量 | discarded reference mass | OT/KDE 均为 0 | 排除“舍弃 reference 导致退化”的解释 |
| 数值尺度 | reference range/IQR | 大尺度与 KDE WQL 退化相关（10K rho=-0.574, p=0.008） | 中等，KDE 风险信号 |

## 3. 最强发现：reference 内部特征离散度

“异质性”当前定义为 **同一 task 的不同 reference 局部 segment 的连续 descriptor 值是否分散**，不是窗口类别数量，也不是 cluster entropy。对 descriptor `j`：

```text
heterogeneity_j = std(Z_1,j, Z_2,j, ..., Z_M,j)
```

最稳定的三个信号是：

- `ref_volatility_clustering_std`：窗口间波动聚集机制是否一致；
- `ref_spectral_centroid_std`：窗口间主频结构是否一致；
- `ref_asymmetry_std`：窗口间涨跌偏斜形态是否一致。

10K 时，三个 std 百分位排名的平均构成一个临时 heterogeneity index。它与 improvement 的关系为：

| 方法 | MASE improvement | WQL improvement |
| --- | ---: | ---: |
| OT | rho=-0.616, p=0.004 | rho=-0.662, p=0.001 |
| KDE | rho=-0.526, p=0.017 | rho=-0.508, p=0.022 |

leave-one-task-out 后四个关系仍全部为负，说明 10K 结果不由单个极端 task 驱动。跨 1K--10K，四种“method × metric”组合中有 8--9/10 个 checkpoint 呈负向；信号在训练后期更明显。

直观分组也支持该结论：频谱中心 std 最低三分位中，OT 的 MASE 中位提升为 `+0.129`（6/7 task 正提升）；最高三分位为 `-0.114`（1/7 task 正提升）。KDE 对应为 `+0.079`（5/7）与 `-0.061`（2/7）。

**当前可支持的表述：**

> 在当前 20 个 FEV task、16 维静态 descriptor、全局 OT/KDE 和当前训练预算下，reference 内部局部特征离散度越高，OT/KDE 相比 Rand 的收益越弱，负迁移风险越高。

这是一条风险关联，不是“离散度导致失败”的因果结论，也不应泛化为所有 OT/KDE 都不适用于异质数据。

## 4. 两条关键假设质疑

### H1：安全历史的静态特征未必代表未来

当前 selection 使用安全历史；真正评估的是其后的 future。即使 `Z_history` 相似，future 转移也可能不同：

```text
p_source(Y | Z_history) != p_reference(Y | Z_history)
```

高离散度 reference 会放大问题：一个全局历史混合物不一定代表预测边界附近的当前 regime。相邻局部变化强度的负向趋势支持这一解释，但 endpoint change 和 history--answer gap 目前没有给出稳定、单调的证据。因此 H1 很合理，尚未被当前统计完全确认；应使用安全历史内部 pseudo-cutoff 的 history-to-pseudo-future 一致性进一步验证。

### H2：descriptor 相同，不代表模型认为样本相同

人工 descriptor 是压缩映射。可能存在 `X_a != X_b`，但：

```text
h(X_a) ~= h(X_b)
```

于是 OT/KDE 会赋予相近权重；但 Chronos 内部 representation、loss 或训练梯度仍可不同：

```text
g_theta(X_a) != g_theta(X_b)
gradient_theta L(X_a, Y_a) != gradient_theta L(X_b, Y_b)
```

因此 descriptor similarity 不等于 training utility similarity。该假设解释了为什么 post-transfer feature gap 在 20/20 task 上下降，却没有稳定带来 MASE/WQL 改善。

## 5. Rand 多样性与尚未验证的机制

在实验设计中，Rand 与 OT/KDE 的关键差别是 source exposure probability。高 reference 离散度下，OT/KDE 的权重可能集中在少数 source mode，或者给多个冲突 mode 增加曝光；Rand 的广覆盖可能更稳健。

但当前没有记录 source weight entropy、effective sample size (ESS)、top-k mass、实际 unique segment 数或重复曝光率。因此“Rand 见过更多样本所以更好”是合理机制假设，而不是已证实结论。

下一步应检验：

```text
reference heterogeneity
  -> source weight concentration / gradient conflict
  -> lower effective diversity or harmful exposure
  -> negative transfer
```

其中应记录 `ESS = (sum_i w_i)^2 / sum_i w_i^2`、weight entropy、top-1%/5% mass 和 unique source/window exposure，并加入 matched-concentration Random 作为控制。

## 6. DSIR 与 LESS 的参考

DSIR 在语言模型 continued pretraining 中使用 hashed n-gram importance resampling，能在其特定的、近似静态领域适配设定中改善平均表现。其成功依赖于 n-gram 统计与 token-prediction 训练目标较接近、target 样本可代表未来数据，以及完整 pipeline 中的质量过滤与重采样多样性。

LESS 说明这个假设不能泛化：在 targeted instruction tuning 中，DSIR 的 n-gram matching 在其对比中不稳定，甚至可低于 Random；LESS 改用与目标损失相关的梯度相似度。LESS 也不声称无条件保证：warmup、模型状态、样本交互、代理 loss 与最终 metric 的错位都会限制稳定提升。

这对 TS3 的启示是：当前结果并不否定数据选择，而是表明 **静态 descriptor 分布接近不是 forecast utility 的充分条件**。更安全的方向是 high-heterogeneity gate 回退 Rand，以及 cluster/regime-conditioned、local-window 或 model-aware 的匹配。

## 7. 结论

当前最有可能、与 OT/KDE 相对 Rand 表现最相关的解释是：

> **reference 内部局部特征离散度高，使全局静态 `p(Z)` 匹配难以代表预测时刻的 regime 或 `p(Y | Z)`；因此纯 descriptor 重加权更容易发生负迁移。**

这一结论解释的是当前方法的适用边界，不是对 OT/KDE、数据选择或模型感知匹配的一般否定。后续最有价值的验证是：记录 source exposure diversity/ESS，并用安全历史 pseudo-future 或模型损失特征测试“静态相似”是否能转化为“预测效用相似”。
