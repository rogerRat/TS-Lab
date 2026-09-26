# V5 / V5-uni / TimerXL 结果分析

以下分析使用已完成的 Chronos-Base 与 TimesFM 2.5 全矩阵：18 个模型×任务场景、15 个方法臂，其中 10 个属于 RAGS 方法族（含两个 uni 臂）。MASE 越低越好，正的相对改善表示更好。TimerXL 仍在运行，其已落盘结果保留在后面的第三张表中，不参与下面的最终方法选择统计。

## 1. 从所有 RAGS 方法族选一个谁是最好的方法？

**按排名选：`rags-block-linearSum`。** 按每个模型×任务的 MASE 排名，在全部 15 个方法中计算平均名次，`rags-block-linearSum` 的平均名次为 **5.06/15**，是 RAGS 方法族中最好；在只比较 RAGS 的 10 个方法时，平均名次为 **3.78/10**。`rags-block-expSum` 非常接近，全部方法平均名次为 5.11。

**按平均提升选：`rags-block-expSum`。** 它相对 rand 的 18 个模型×任务平均改善为 **+1.67%**，14/18 个场景的均值优于 rand；逐 seed 配对胜为 **38/54**。因此，如果需要论文中指定一个“平均效果最强”的 RAGS 主方法，应选 `rags-block-expSum`；如果强调整体名次稳定性，则选 `rags-block-linearSum`。

## 2. 得出的最优 RAGS 方法多大程度上比 Rand 好？

| 选择标准 | 方法 | 相对 rand 平均改善 | 均值优于 rand | seed 配对胜 |
|---|---|---:|---:|---:|
| 平均排名最佳 | rags-block-linearSum | +0.92% | 13/18（72.2%） | 40/54（74.1%） |
| 平均提升最佳 | rags-block-expSum | +1.67% | 14/18（77.8%） | 38/54（70.4%） |

结论：两个候选方法都比 rand 有**小幅但非压倒性的优势**。`expSum` 的平均提升更大，`linearSum` 的逐 seed 胜率更高；不能把优势描述成在所有任务上稳定成立。

## 3. 得出的最好 RAGS 方法多大程度上比其他方法好？

下表逐个模型×任务比较候选方法与对应基线；“平均改善”是 `(基线 MASE − RAGS MASE) / 基线 MASE` 的平均值。

| RAGS 候选 | 基线 | RAGS 胜出场景 | 平均改善 | 判断 |
|---|---|---:|---:|---|
| block-linearSum | rand | 13/18 | +0.92% | 小幅占优 |
| block-linearSum | orig | 15/18 | +6.80% | 多数明显占优 |
| block-linearSum | LESS | 9/18 | +0.15% | 基本持平 |
| block-linearSum | TSDS | 17/18 | -0.04% | 胜出场景多，但总体几乎持平 |
| block-linearSum | LP-Mirror | 13/18 | +4.99% | 明显占优 |
| block-expSum | rand | 14/18 | +1.67% | 小幅占优，候选中最好 |
| block-expSum | orig | 14/18 | +7.19% | 多数明显占优 |
| block-expSum | LESS | 14/18 | +1.02% | 小幅占优 |
| block-expSum | TSDS | 14/18 | +0.93% | 小幅占优 |
| block-expSum | LP-Mirror | 13/18 | +5.68% | 明显占优 |

结论：`block-expSum` 是更适合作为单一主方法的选择，因为它相对 rand、LESS、TSDS、LP-Mirror 都有正的平均改善；`block-linearSum` 的主要优点是整体排名和逐 seed 胜率更稳，但相对 TSDS 的平均差异接近零。两者都不是对所有任务和所有基线都绝对占优。

## 4. uni 版本的 RAGS 方法是否大多数弱于非 uni？

**是。** 在 54 个逐 seed 配对中：

- `rags-block-expSum-uni` 相对 `rags-block-expSum`：18 次更好、36 次更差，平均 ΔMASE = **+0.0311**。
- `rags-block-linearSum-uni` 相对 `rags-block-linearSum`：13 次更好、41 次更差，平均 ΔMASE = **+0.0369**。

两种 uni 版本都是多数退化，且平均 ΔMASE 为正。因此当前证据支持：**轨迹内部的块级差异化权重是有效信息，uni 不应作为默认增强版本。**

## 5. block 级方法是否大多数时候强于 traj 级别？

在原始 v5 中按相同聚合方式配对比较 block 与 traj（4 种聚合 × 18 个模型×任务 = 72 对；uni 不参与，因为没有对应的 traj-uni 臂）：

- block 胜出 **43/72（59.7%）**，平均相对 traj 改善 **+0.47%**，中位数 **+0.19%**。
- `linearSum`：block 胜 14/18，平均 +0.56%。
- `expSum`：block 胜 12/18，平均 +1.09%。
- `zeroClipSum`：block 胜 9/18，平均 +0.82%。
- `maxSum`：block 胜 8/18，平均 **-0.57%**，traj 反而略占优。

结论：**在 RAGS 内部，block 级方法总体是轻微多数占优，但优势不强且依赖聚合方式；linearSum 和 expSum 的 block 级形式更有支持，maxSum 不支持 block 普遍优于 traj。**

## 6. TimerXL 和另外两个模型的结论是否大体一致？

**目前不能说大体一致；只有 block 优于 traj 这一条方向一致。** 为避免 seed2 尚未完成造成偏差，下面先只使用 TimerXL 已完整落盘的 seed0/seed1（9 个任务、15 个臂）比较；TimerXL 当前总落盘为 298/387 个去重 cell，seed2 仍是部分结果。

- **最优 RAGS 方法不同。** TimerXL 当前按平均排名和平均相对 rand 提升都指向 `rags-block-zeroClipSum`，平均排名 5.67/15、相对 rand +0.52%；Chronos/TimesFM 合并结果则分别指向 `block-linearSum`（排名）和 `block-expSum`（平均提升）。
- **RAGS 相对 rand 的总体方向不同。** TimerXL 的 10 个 RAGS 臂只有 23/90 个模型×任务×方法均值格优于 rand，平均相对 rand 为 **-0.96%**，逐 seed 胜为 **49/180**；而 Chronos/TimesFM 的 10 臂合计为 112/180 格、平均 +0.16%。因此不能把前两个模型的 RAGS 优势直接外推到 TimerXL。
- **uni 退化结论尚未复现。** TimerXL seed0/1 中，`expSum-uni` 相对 expSum 为 10 胜、8 负；`linearSum-uni` 为 9 胜、9 负。与 Chronos/TimesFM 中 expSum 18 胜/36 负、linearSum 13 胜/41 负的多数退化不同，TimerXL 当前看不出 uni 大多数更弱。
- **block vs traj 方向一致。** TimerXL 中 block 在 72 个同聚合配对中胜 51 次（70.8%），平均相对 traj 改善 +1.16%；前两个模型为 43/72（59.7%）、平均 +0.47%。

因此当前最稳妥的跨模型表述是：**block 级 RAGS 相对 traj 级 RAGS 的方向具有一定模型一致性；但“RAGS 优于 rand”以及“uni 大多数退化”具有明显模型依赖，TimerXL seed2 完成前不能下最终三模型统一结论。**

以下保留三模型、9 任务、15 臂的全量明细表；TimerXL 行只使用当前已落盘的正式 cell。

## chronos-base

| 汇总 | 内容 |
|---|---|
| 数值来源 | v5 `MAIN.md` + v5-uni `UNI.md` |
| 表内行数 | 135（9 任务 × 15 臂） |

| 任务 | 方法 / 臂 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| time_nem_demand_5t | rags-block-zeroClipSum | 3 | 1.323450 | 0.066163 | +23.39% | +4.26% | 2/3 | v5 |
| time_nem_demand_5t | rags-block-linearSum | 3 | 1.327354 | 0.115937 | +23.16% | +3.98% | 2/3 | v5 |
| time_nem_demand_5t | rags-traj-expSum | 3 | 1.357953 | 0.089838 | +21.39% | +1.76% | 3/3 | v5 |
| time_nem_demand_5t | rags-traj-linearSum | 3 | 1.379673 | 0.066397 | +20.14% | +0.19% | 2/3 | v5 |
| time_nem_demand_5t | rand | 3 | 1.382337 | 0.103409 | +19.98% | +0.00% | — | v5 |
| time_nem_demand_5t | tsds | 3 | 1.382384 | 0.084658 | +19.98% | -0.00% | 1/3 | v5 |
| time_nem_demand_5t | rags-block-expSum-uni | 3 | 1.385323 | 0.091416 | +19.81% | -0.22% | 1/3 | v5-uni |
| time_nem_demand_5t | rags-traj-maxSum | 3 | 1.418465 | 0.054267 | +17.89% | -2.61% | 1/3 | v5 |
| time_nem_demand_5t | rags-block-expSum | 3 | 1.436664 | 0.045787 | +16.84% | -3.93% | 1/3 | v5 |
| time_nem_demand_5t | less-main | 3 | 1.441543 | 0.037287 | +16.55% | -4.28% | 1/3 | v5 |
| time_nem_demand_5t | rags-traj-zeroClipSum | 3 | 1.456554 | 0.066228 | +15.68% | -5.37% | 0/3 | v5 |
| time_nem_demand_5t | rags-block-linearSum-uni | 3 | 1.469781 | 0.105081 | +14.92% | -6.33% | 0/3 | v5-uni |
| time_nem_demand_5t | rags-block-maxSum | 3 | 1.497766 | 0.090333 | +13.30% | -8.35% | 1/3 | v5 |
| time_nem_demand_5t | orig | 1 | 1.727507 | — | +0.00% | -24.97% | — | v5 |
| time_nem_demand_5t | lp-mirror | 3 | 2.172320 | 0.181392 | -25.75% | -57.15% | 0/3 | v5 |
| time_azure_i_cpu_5t | rags-block-zeroClipSum | 3 | 1.182011 | 0.008908 | +11.24% | +3.92% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-traj-zeroClipSum | 3 | 1.191853 | 0.029867 | +10.50% | +3.12% | 3/3 | v5 |
| time_azure_i_cpu_5t | rags-block-expSum-uni | 3 | 1.202954 | 0.035912 | +9.66% | +2.22% | 3/3 | v5-uni |
| time_azure_i_cpu_5t | rags-block-linearSum | 3 | 1.205612 | 0.015714 | +9.46% | +2.00% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-traj-linearSum | 3 | 1.211428 | 0.012685 | +9.03% | +1.53% | 2/3 | v5 |
| time_azure_i_cpu_5t | tsds | 3 | 1.212413 | 0.042393 | +8.95% | +1.45% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-traj-expSum | 3 | 1.213672 | 0.022960 | +8.86% | +1.35% | 2/3 | v5 |
| time_azure_i_cpu_5t | lp-mirror | 3 | 1.216375 | 0.028541 | +8.66% | +1.13% | 2/3 | v5 |
| time_azure_i_cpu_5t | less-main | 3 | 1.219202 | 0.009377 | +8.44% | +0.90% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-block-maxSum | 3 | 1.220678 | 0.021710 | +8.33% | +0.78% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-traj-maxSum | 3 | 1.224140 | 0.016007 | +8.07% | +0.50% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-block-expSum | 3 | 1.227666 | 0.049358 | +7.81% | +0.21% | 2/3 | v5 |
| time_azure_i_cpu_5t | rand | 3 | 1.230251 | 0.043551 | +7.61% | +0.00% | — | v5 |
| time_azure_i_cpu_5t | rags-block-linearSum-uni | 3 | 1.231544 | 0.032688 | +7.52% | -0.11% | 2/3 | v5-uni |
| time_azure_i_cpu_5t | orig | 1 | 1.331634 | — | +0.00% | -8.24% | — | v5 |
| time_darwin_water_15t | rags-block-maxSum | 3 | 1.656349 | 0.004625 | +0.46% | +0.38% | 2/3 | v5 |
| time_darwin_water_15t | rags-block-expSum | 3 | 1.657247 | 0.005153 | +0.41% | +0.32% | 3/3 | v5 |
| time_darwin_water_15t | rags-block-zeroClipSum | 3 | 1.659809 | 0.003826 | +0.26% | +0.17% | 2/3 | v5 |
| time_darwin_water_15t | less-main | 3 | 1.660169 | 0.003378 | +0.24% | +0.15% | 2/3 | v5 |
| time_darwin_water_15t | rags-traj-zeroClipSum | 3 | 1.660336 | 0.008302 | +0.23% | +0.14% | 2/3 | v5 |
| time_darwin_water_15t | rags-block-linearSum | 3 | 1.660981 | 0.002592 | +0.19% | +0.10% | 3/3 | v5 |
| time_darwin_water_15t | tsds | 3 | 1.661259 | 0.004549 | +0.17% | +0.08% | 1/3 | v5 |
| time_darwin_water_15t | rags-block-expSum-uni | 3 | 1.662165 | 0.002837 | +0.12% | +0.03% | 1/3 | v5-uni |
| time_darwin_water_15t | rand | 3 | 1.662618 | 0.002768 | +0.09% | +0.00% | — | v5 |
| time_darwin_water_15t | orig | 1 | 1.664086 | — | +0.00% | -0.09% | — | v5 |
| time_darwin_water_15t | rags-block-linearSum-uni | 3 | 1.664324 | 0.001707 | -0.01% | -0.10% | 2/3 | v5-uni |
| time_darwin_water_15t | rags-traj-expSum | 3 | 1.665946 | 0.005450 | -0.11% | -0.20% | 0/3 | v5 |
| time_darwin_water_15t | rags-traj-maxSum | 3 | 1.668158 | 0.003966 | -0.24% | -0.33% | 0/3 | v5 |
| time_darwin_water_15t | rags-traj-linearSum | 3 | 1.669625 | 0.009725 | -0.33% | -0.42% | 1/3 | v5 |
| time_darwin_water_15t | lp-mirror | 3 | 1.688070 | 0.077659 | -1.44% | -1.53% | 2/3 | v5 |
| time_finland_15t | tsds | 3 | 1.021053 | 0.043874 | +50.33% | +17.28% | 3/3 | v5 |
| time_finland_15t | rags-block-expSum | 3 | 1.104193 | 0.109595 | +46.28% | +10.54% | 3/3 | v5 |
| time_finland_15t | less-main | 3 | 1.137035 | 0.128787 | +44.69% | +7.88% | 2/3 | v5 |
| time_finland_15t | rags-traj-linearSum | 3 | 1.141705 | 0.257522 | +44.46% | +7.51% | 2/3 | v5 |
| time_finland_15t | rags-traj-maxSum | 3 | 1.141705 | 0.257522 | +44.46% | +7.51% | 2/3 | v5 |
| time_finland_15t | rags-traj-expSum | 3 | 1.141705 | 0.257522 | +44.46% | +7.51% | 2/3 | v5 |
| time_finland_15t | rags-block-maxSum | 3 | 1.152989 | 0.271234 | +43.91% | +6.59% | 2/3 | v5 |
| time_finland_15t | rags-traj-zeroClipSum | 3 | 1.221960 | 0.181537 | +40.55% | +1.00% | 2/3 | v5 |
| time_finland_15t | rags-block-zeroClipSum | 3 | 1.233872 | 0.200841 | +39.97% | +0.04% | 2/3 | v5 |
| time_finland_15t | rand | 3 | 1.234354 | 0.062098 | +39.95% | +0.00% | — | v5 |
| time_finland_15t | rags-block-linearSum | 3 | 1.335114 | 0.198919 | +35.05% | -8.16% | 1/3 | v5 |
| time_finland_15t | rags-block-expSum-uni | 3 | 1.491474 | 0.074510 | +27.44% | -20.83% | 0/3 | v5-uni |
| time_finland_15t | rags-block-linearSum-uni | 3 | 1.518281 | 0.083580 | +26.14% | -23.00% | 0/3 | v5-uni |
| time_finland_15t | lp-mirror | 3 | 1.587298 | 0.121479 | +22.78% | -28.59% | 0/3 | v5 |
| time_finland_15t | orig | 1 | 2.055572 | — | +0.00% | -66.53% | — | v5 |
| time_eweld_15t | lp-mirror | 3 | 1.175806 | 0.035272 | +21.40% | +9.68% | 3/3 | v5 |
| time_eweld_15t | rags-block-expSum | 3 | 1.203214 | 0.059509 | +19.57% | +7.57% | 2/3 | v5 |
| time_eweld_15t | rags-traj-maxSum | 3 | 1.228749 | 0.072498 | +17.86% | +5.61% | 2/3 | v5 |
| time_eweld_15t | less-main | 3 | 1.235223 | 0.042140 | +17.43% | +5.11% | 2/3 | v5 |
| time_eweld_15t | rags-block-linearSum | 3 | 1.241541 | 0.088262 | +17.01% | +4.63% | 2/3 | v5 |
| time_eweld_15t | rags-block-zeroClipSum | 3 | 1.242002 | 0.072458 | +16.98% | +4.59% | 2/3 | v5 |
| time_eweld_15t | rags-traj-expSum | 3 | 1.256428 | 0.052334 | +16.01% | +3.48% | 2/3 | v5 |
| time_eweld_15t | rags-block-expSum-uni | 3 | 1.261863 | 0.057220 | +15.65% | +3.06% | 2/3 | v5-uni |
| time_eweld_15t | tsds | 3 | 1.273446 | 0.040967 | +14.88% | +2.17% | 2/3 | v5 |
| time_eweld_15t | rand | 3 | 1.301758 | 0.119424 | +12.98% | +0.00% | — | v5 |
| time_eweld_15t | rags-traj-zeroClipSum | 3 | 1.302969 | 0.064181 | +12.90% | -0.09% | 2/3 | v5 |
| time_eweld_15t | rags-traj-linearSum | 3 | 1.317095 | 0.018932 | +11.96% | -1.18% | 2/3 | v5 |
| time_eweld_15t | rags-block-linearSum-uni | 3 | 1.347244 | 0.095935 | +9.94% | -3.49% | 1/3 | v5-uni |
| time_eweld_15t | rags-block-maxSum | 3 | 1.373724 | 0.203239 | +8.17% | -5.53% | 1/3 | v5 |
| time_eweld_15t | orig | 1 | 1.495984 | — | +0.00% | -14.92% | — | v5 |
| time_cphl_h | rags-block-linearSum | 3 | 1.508442 | 0.001773 | +1.97% | +3.65% | 3/3 | v5 |
| time_cphl_h | rags-traj-zeroClipSum | 3 | 1.513568 | 0.040629 | +1.64% | +3.32% | 3/3 | v5 |
| time_cphl_h | rags-block-maxSum | 3 | 1.514595 | 0.030408 | +1.57% | +3.26% | 3/3 | v5 |
| time_cphl_h | rags-block-expSum-uni | 3 | 1.517492 | 0.017938 | +1.38% | +3.07% | 3/3 | v5-uni |
| time_cphl_h | orig | 1 | 1.538772 | — | +0.00% | +1.71% | — | v5 |
| time_cphl_h | rags-block-expSum | 3 | 1.539277 | 0.039354 | -0.03% | +1.68% | 2/3 | v5 |
| time_cphl_h | rags-traj-linearSum | 3 | 1.539250 | 0.038053 | -0.03% | +1.68% | 2/3 | v5 |
| time_cphl_h | tsds | 3 | 1.543695 | 0.019645 | -0.32% | +1.40% | 3/3 | v5 |
| time_cphl_h | rags-traj-expSum | 3 | 1.545664 | 0.026843 | -0.45% | +1.27% | 2/3 | v5 |
| time_cphl_h | rags-block-zeroClipSum | 3 | 1.546878 | 0.033455 | -0.53% | +1.19% | 3/3 | v5 |
| time_cphl_h | rags-traj-maxSum | 3 | 1.555795 | 0.011564 | -1.11% | +0.62% | 1/3 | v5 |
| time_cphl_h | lp-mirror | 3 | 1.561549 | 0.058348 | -1.48% | +0.26% | 2/3 | v5 |
| time_cphl_h | less-main | 3 | 1.562286 | 0.015816 | -1.53% | +0.21% | 2/3 | v5 |
| time_cphl_h | rand | 3 | 1.565579 | 0.017958 | -1.74% | +0.00% | — | v5 |
| time_cphl_h | rags-block-linearSum-uni | 3 | 1.570874 | 0.010302 | -2.09% | -0.34% | 1/3 | v5-uni |
| time_nem_price_5t | rags-block-expSum | 3 | 1.032367 | 0.003353 | +2.81% | +1.75% | 3/3 | v5 |
| time_nem_price_5t | less-main | 3 | 1.040161 | 0.002707 | +2.07% | +1.01% | 3/3 | v5 |
| time_nem_price_5t | rags-block-linearSum | 3 | 1.040698 | 0.001987 | +2.02% | +0.96% | 2/3 | v5 |
| time_nem_price_5t | rags-traj-expSum | 3 | 1.042299 | 0.005468 | +1.87% | +0.81% | 2/3 | v5 |
| time_nem_price_5t | tsds | 3 | 1.043533 | 0.010296 | +1.76% | +0.69% | 2/3 | v5 |
| time_nem_price_5t | rags-block-maxSum | 3 | 1.045123 | 0.005538 | +1.61% | +0.54% | 2/3 | v5 |
| time_nem_price_5t | rags-traj-zeroClipSum | 3 | 1.046404 | 0.006250 | +1.49% | +0.42% | 2/3 | v5 |
| time_nem_price_5t | rags-block-zeroClipSum | 3 | 1.046861 | 0.009176 | +1.44% | +0.37% | 1/3 | v5 |
| time_nem_price_5t | rags-block-linearSum-uni | 3 | 1.047600 | 0.002511 | +1.37% | +0.30% | 1/3 | v5-uni |
| time_nem_price_5t | rags-traj-maxSum | 3 | 1.049906 | 0.004722 | +1.16% | +0.08% | 1/3 | v5 |
| time_nem_price_5t | rand | 3 | 1.050788 | 0.014902 | +1.07% | +0.00% | — | v5 |
| time_nem_price_5t | rags-block-expSum-uni | 3 | 1.052422 | 0.009424 | +0.92% | -0.16% | 1/3 | v5-uni |
| time_nem_price_5t | rags-traj-linearSum | 3 | 1.052576 | 0.006229 | +0.91% | -0.17% | 1/3 | v5 |
| time_nem_price_5t | lp-mirror | 3 | 1.054419 | 0.006381 | +0.73% | -0.35% | 1/3 | v5 |
| time_nem_price_5t | orig | 1 | 1.062202 | — | +0.00% | -1.09% | — | v5 |
| time_solar_h | rand | 3 | 1.204102 | 0.060388 | +1.57% | +0.00% | — | v5 |
| time_solar_h | rags-block-maxSum | 3 | 1.207896 | 0.245254 | +1.26% | -0.32% | 2/3 | v5 |
| time_solar_h | rags-block-expSum | 3 | 1.217178 | 0.016173 | +0.50% | -1.09% | 2/3 | v5 |
| time_solar_h | rags-block-linearSum | 3 | 1.221579 | 0.101530 | +0.15% | -1.45% | 1/3 | v5 |
| time_solar_h | orig | 1 | 1.223355 | — | +0.00% | -1.60% | — | v5 |
| time_solar_h | lp-mirror | 3 | 1.228822 | 0.158462 | -0.45% | -2.05% | 1/3 | v5 |
| time_solar_h | less-main | 3 | 1.244282 | 0.056064 | -1.71% | -3.34% | 1/3 | v5 |
| time_solar_h | rags-block-linearSum-uni | 3 | 1.251029 | 0.048191 | -2.26% | -3.90% | 1/3 | v5-uni |
| time_solar_h | rags-block-expSum-uni | 3 | 1.263914 | 0.090305 | -3.32% | -4.97% | 0/3 | v5-uni |
| time_solar_h | rags-block-zeroClipSum | 3 | 1.270936 | 0.142540 | -3.89% | -5.55% | 1/3 | v5 |
| time_solar_h | tsds | 3 | 1.292194 | 0.080047 | -5.63% | -7.32% | 1/3 | v5 |
| time_solar_h | rags-traj-zeroClipSum | 3 | 1.308656 | 0.041455 | -6.97% | -8.68% | 0/3 | v5 |
| time_solar_h | rags-traj-linearSum | 3 | 1.319936 | 0.272292 | -7.89% | -9.62% | 1/3 | v5 |
| time_solar_h | rags-traj-maxSum | 3 | 1.319936 | 0.272292 | -7.89% | -9.62% | 1/3 | v5 |
| time_solar_h | rags-traj-expSum | 3 | 1.319936 | 0.272292 | -7.89% | -9.62% | 1/3 | v5 |
| time_port_activity_d | orig | 1 | 0.882133 | — | +0.00% | +1.31% | — | v5 |
| time_port_activity_d | rags-traj-maxSum | 3 | 0.883752 | 0.009543 | -0.18% | +1.13% | 2/3 | v5 |
| time_port_activity_d | lp-mirror | 3 | 0.883800 | 0.002488 | -0.19% | +1.12% | 2/3 | v5 |
| time_port_activity_d | rags-block-zeroClipSum | 3 | 0.886293 | 0.005916 | -0.47% | +0.84% | 2/3 | v5 |
| time_port_activity_d | rags-block-expSum | 3 | 0.888557 | 0.006190 | -0.73% | +0.59% | 2/3 | v5 |
| time_port_activity_d | less-main | 3 | 0.892152 | 0.011043 | -1.14% | +0.19% | 2/3 | v5 |
| time_port_activity_d | rags-traj-zeroClipSum | 3 | 0.893739 | 0.019669 | -1.32% | +0.01% | 1/3 | v5 |
| time_port_activity_d | rand | 3 | 0.893838 | 0.015555 | -1.33% | +0.00% | — | v5 |
| time_port_activity_d | rags-block-linearSum-uni | 3 | 0.894828 | 0.002636 | -1.44% | -0.11% | 2/3 | v5-uni |
| time_port_activity_d | rags-block-expSum-uni | 3 | 0.895378 | 0.007164 | -1.50% | -0.17% | 2/3 | v5-uni |
| time_port_activity_d | rags-traj-expSum | 3 | 0.896194 | 0.015326 | -1.59% | -0.26% | 1/3 | v5 |
| time_port_activity_d | rags-block-maxSum | 3 | 0.897754 | 0.018922 | -1.77% | -0.44% | 2/3 | v5 |
| time_port_activity_d | rags-block-linearSum | 3 | 0.899795 | 0.004386 | -2.00% | -0.67% | 2/3 | v5 |
| time_port_activity_d | rags-traj-linearSum | 3 | 0.901212 | 0.024135 | -2.16% | -0.82% | 1/3 | v5 |
| time_port_activity_d | tsds | 3 | 0.917647 | 0.005653 | -4.03% | -2.66% | 0/3 | v5 |

## timesfm-2.5

| 汇总 | 内容 |
|---|---|
| 数值来源 | v5 `MAIN.md` + v5-uni `UNI.md` |
| 表内行数 | 135（9 任务 × 15 臂） |

| 任务 | 方法 / 臂 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| time_nem_demand_5t | rags-block-expSum | 3 | 0.685112 | 0.011305 | +22.32% | +7.57% | 3/3 | v5 |
| time_nem_demand_5t | rags-block-linearSum | 3 | 0.694575 | 0.007059 | +21.24% | +6.30% | 3/3 | v5 |
| time_nem_demand_5t | rags-traj-maxSum | 3 | 0.710861 | 0.006832 | +19.40% | +4.10% | 3/3 | v5 |
| time_nem_demand_5t | rags-traj-linearSum | 3 | 0.712236 | 0.010111 | +19.24% | +3.91% | 3/3 | v5 |
| time_nem_demand_5t | rags-traj-zeroClipSum | 3 | 0.717496 | 0.012561 | +18.64% | +3.20% | 3/3 | v5 |
| time_nem_demand_5t | tsds | 3 | 0.721019 | 0.001882 | +18.24% | +2.73% | 3/3 | v5 |
| time_nem_demand_5t | less-main | 3 | 0.721196 | 0.004319 | +18.22% | +2.70% | 3/3 | v5 |
| time_nem_demand_5t | rags-traj-expSum | 3 | 0.722459 | 0.012771 | +18.08% | +2.53% | 3/3 | v5 |
| time_nem_demand_5t | rags-block-zeroClipSum | 3 | 0.724503 | 0.002566 | +17.85% | +2.26% | 3/3 | v5 |
| time_nem_demand_5t | rags-block-maxSum | 3 | 0.726064 | 0.020073 | +17.67% | +2.05% | 2/3 | v5 |
| time_nem_demand_5t | rand | 3 | 0.741243 | 0.016305 | +15.95% | +0.00% | — | v5 |
| time_nem_demand_5t | rags-block-linearSum-uni | 3 | 0.742662 | 0.018560 | +15.79% | -0.19% | 2/3 | v5-uni |
| time_nem_demand_5t | rags-block-expSum-uni | 3 | 0.754772 | 0.017082 | +14.42% | -1.83% | 0/3 | v5-uni |
| time_nem_demand_5t | lp-mirror | 3 | 0.877323 | 0.000192 | +0.52% | -18.36% | 0/3 | v5 |
| time_nem_demand_5t | orig | 1 | 0.881911 | — | +0.00% | -18.98% | — | v5 |
| time_azure_i_cpu_5t | lp-mirror | 3 | 0.762043 | 0.001374 | +0.23% | +0.33% | 2/3 | v5 |
| time_azure_i_cpu_5t | less-main | 3 | 0.763533 | 0.003292 | +0.04% | +0.14% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-block-maxSum | 3 | 0.763522 | 0.005380 | +0.04% | +0.14% | 2/3 | v5 |
| time_azure_i_cpu_5t | orig | 1 | 0.763820 | — | +0.00% | +0.10% | — | v5 |
| time_azure_i_cpu_5t | rags-traj-maxSum | 3 | 0.763798 | 0.002214 | +0.00% | +0.10% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-block-zeroClipSum | 3 | 0.763992 | 0.001641 | -0.02% | +0.08% | 2/3 | v5 |
| time_azure_i_cpu_5t | rags-traj-zeroClipSum | 3 | 0.764064 | 0.003582 | -0.03% | +0.07% | 1/3 | v5 |
| time_azure_i_cpu_5t | rand | 3 | 0.764581 | 0.002946 | -0.10% | +0.00% | — | v5 |
| time_azure_i_cpu_5t | rags-traj-linearSum | 3 | 0.766217 | 0.001643 | -0.31% | -0.21% | 0/3 | v5 |
| time_azure_i_cpu_5t | rags-block-linearSum | 3 | 0.766367 | 0.001361 | -0.33% | -0.23% | 2/3 | v5 |
| time_azure_i_cpu_5t | tsds | 3 | 0.766826 | 0.001165 | -0.39% | -0.29% | 1/3 | v5 |
| time_azure_i_cpu_5t | rags-block-linearSum-uni | 3 | 0.767024 | 0.002348 | -0.42% | -0.32% | 1/3 | v5-uni |
| time_azure_i_cpu_5t | rags-traj-expSum | 3 | 0.767493 | 0.001745 | -0.48% | -0.38% | 0/3 | v5 |
| time_azure_i_cpu_5t | rags-block-expSum-uni | 3 | 0.767560 | 0.001597 | -0.49% | -0.39% | 1/3 | v5-uni |
| time_azure_i_cpu_5t | rags-block-expSum | 3 | 0.768138 | 0.003192 | -0.57% | -0.47% | 0/3 | v5 |
| time_darwin_water_15t | rags-block-linearSum | 3 | 1.001466 | 0.004124 | +6.38% | +2.99% | 3/3 | v5 |
| time_darwin_water_15t | rags-block-expSum | 3 | 1.004411 | 0.004390 | +6.10% | +2.71% | 3/3 | v5 |
| time_darwin_water_15t | rags-block-zeroClipSum | 3 | 1.008161 | 0.003795 | +5.75% | +2.35% | 3/3 | v5 |
| time_darwin_water_15t | rags-traj-zeroClipSum | 3 | 1.023760 | 0.005129 | +4.30% | +0.83% | 3/3 | v5 |
| time_darwin_water_15t | rags-traj-maxSum | 3 | 1.023887 | 0.003982 | +4.28% | +0.82% | 3/3 | v5 |
| time_darwin_water_15t | rags-block-maxSum | 3 | 1.026001 | 0.001023 | +4.09% | +0.62% | 3/3 | v5 |
| time_darwin_water_15t | rags-traj-linearSum | 3 | 1.027278 | 0.003369 | +3.97% | +0.49% | 3/3 | v5 |
| time_darwin_water_15t | less-main | 3 | 1.027568 | 0.001079 | +3.94% | +0.47% | 2/3 | v5 |
| time_darwin_water_15t | rags-traj-expSum | 3 | 1.029867 | 0.004354 | +3.72% | +0.24% | 2/3 | v5 |
| time_darwin_water_15t | rags-block-expSum-uni | 3 | 1.032242 | 0.006200 | +3.50% | +0.01% | 2/3 | v5-uni |
| time_darwin_water_15t | rand | 3 | 1.032380 | 0.006643 | +3.49% | +0.00% | — | v5 |
| time_darwin_water_15t | rags-block-linearSum-uni | 3 | 1.034264 | 0.006609 | +3.31% | -0.18% | 2/3 | v5-uni |
| time_darwin_water_15t | tsds | 3 | 1.039987 | 0.002520 | +2.78% | -0.74% | 0/3 | v5 |
| time_darwin_water_15t | lp-mirror | 3 | 1.069666 | 0.000241 | +0.00% | -3.61% | 0/3 | v5 |
| time_darwin_water_15t | orig | 1 | 1.069706 | — | +0.00% | -3.62% | — | v5 |
| time_finland_15t | rags-block-linearSum | 3 | 0.609725 | 0.002420 | +1.95% | +1.02% | 3/3 | v5 |
| time_finland_15t | rags-traj-zeroClipSum | 3 | 0.610231 | 0.002732 | +1.87% | +0.93% | 3/3 | v5 |
| time_finland_15t | rags-traj-linearSum | 3 | 0.610333 | 0.000888 | +1.86% | +0.92% | 3/3 | v5 |
| time_finland_15t | rags-traj-maxSum | 3 | 0.610333 | 0.000888 | +1.86% | +0.92% | 3/3 | v5 |
| time_finland_15t | rags-traj-expSum | 3 | 0.610333 | 0.000888 | +1.86% | +0.92% | 3/3 | v5 |
| time_finland_15t | rags-block-expSum | 3 | 0.611438 | 0.001785 | +1.68% | +0.74% | 2/3 | v5 |
| time_finland_15t | rags-block-zeroClipSum | 3 | 0.612653 | 0.001008 | +1.48% | +0.54% | 3/3 | v5 |
| time_finland_15t | rags-block-maxSum | 3 | 0.612945 | 0.004107 | +1.44% | +0.49% | 2/3 | v5 |
| time_finland_15t | rags-block-linearSum-uni | 3 | 0.613722 | 0.001613 | +1.31% | +0.37% | 3/3 | v5-uni |
| time_finland_15t | tsds | 3 | 0.614979 | 0.000692 | +1.11% | +0.16% | 2/3 | v5 |
| time_finland_15t | rand | 3 | 0.615987 | 0.003555 | +0.95% | +0.00% | — | v5 |
| time_finland_15t | rags-block-expSum-uni | 3 | 0.616517 | 0.001906 | +0.86% | -0.09% | 2/3 | v5-uni |
| time_finland_15t | less-main | 3 | 0.616636 | 0.002182 | +0.84% | -0.11% | 2/3 | v5 |
| time_finland_15t | orig | 1 | 0.621873 | — | +0.00% | -0.96% | — | v5 |
| time_finland_15t | lp-mirror | 3 | 0.625578 | 0.001012 | -0.60% | -1.56% | 0/3 | v5 |
| time_eweld_15t | less-main | 3 | 0.721271 | 0.001098 | +0.96% | +0.24% | 3/3 | v5 |
| time_eweld_15t | rags-block-linearSum | 3 | 0.721899 | 0.000427 | +0.87% | +0.15% | 2/3 | v5 |
| time_eweld_15t | rags-block-expSum-uni | 3 | 0.722539 | 0.002087 | +0.78% | +0.06% | 2/3 | v5-uni |
| time_eweld_15t | rags-block-expSum | 3 | 0.722798 | 0.000506 | +0.75% | +0.03% | 1/3 | v5 |
| time_eweld_15t | rags-traj-zeroClipSum | 3 | 0.722861 | 0.001092 | +0.74% | +0.02% | 2/3 | v5 |
| time_eweld_15t | rags-traj-expSum | 3 | 0.722889 | 0.000901 | +0.74% | +0.01% | 1/3 | v5 |
| time_eweld_15t | rand | 3 | 0.722988 | 0.001693 | +0.72% | +0.00% | — | v5 |
| time_eweld_15t | rags-traj-linearSum | 3 | 0.723484 | 0.001942 | +0.65% | -0.07% | 1/3 | v5 |
| time_eweld_15t | rags-block-zeroClipSum | 3 | 0.723570 | 0.000187 | +0.64% | -0.08% | 1/3 | v5 |
| time_eweld_15t | rags-block-maxSum | 3 | 0.723782 | 0.000801 | +0.61% | -0.11% | 1/3 | v5 |
| time_eweld_15t | rags-block-linearSum-uni | 3 | 0.723925 | 0.000711 | +0.59% | -0.13% | 1/3 | v5-uni |
| time_eweld_15t | tsds | 3 | 0.724220 | 0.002607 | +0.55% | -0.17% | 2/3 | v5 |
| time_eweld_15t | rags-traj-maxSum | 3 | 0.724653 | 0.002027 | +0.49% | -0.23% | 0/3 | v5 |
| time_eweld_15t | orig | 1 | 0.728248 | — | +0.00% | -0.73% | — | v5 |
| time_eweld_15t | lp-mirror | 3 | 0.729592 | 0.002234 | -0.18% | -0.91% | 0/3 | v5 |
| time_cphl_h | rags-block-maxSum | 3 | 1.329569 | 0.003257 | +2.68% | +1.29% | 3/3 | v5 |
| time_cphl_h | rags-block-expSum | 3 | 1.333185 | 0.002204 | +2.42% | +1.02% | 3/3 | v5 |
| time_cphl_h | rags-block-zeroClipSum | 3 | 1.333501 | 0.005359 | +2.39% | +1.00% | 3/3 | v5 |
| time_cphl_h | rags-block-linearSum | 3 | 1.335261 | 0.003390 | +2.27% | +0.86% | 3/3 | v5 |
| time_cphl_h | rags-block-linearSum-uni | 3 | 1.339061 | 0.002019 | +1.99% | +0.58% | 3/3 | v5-uni |
| time_cphl_h | rags-block-expSum-uni | 3 | 1.340682 | 0.006156 | +1.87% | +0.46% | 3/3 | v5-uni |
| time_cphl_h | less-main | 3 | 1.345672 | 0.007804 | +1.50% | +0.09% | 1/3 | v5 |
| time_cphl_h | rags-traj-maxSum | 3 | 1.346804 | 0.002875 | +1.42% | +0.01% | 2/3 | v5 |
| time_cphl_h | rand | 3 | 1.346911 | 0.001671 | +1.41% | +0.00% | — | v5 |
| time_cphl_h | lp-mirror | 3 | 1.346897 | 0.003127 | +1.41% | +0.00% | 2/3 | v5 |
| time_cphl_h | rags-traj-zeroClipSum | 3 | 1.348653 | 0.004292 | +1.29% | -0.13% | 2/3 | v5 |
| time_cphl_h | rags-traj-linearSum | 3 | 1.349335 | 0.001386 | +1.24% | -0.18% | 1/3 | v5 |
| time_cphl_h | rags-traj-expSum | 3 | 1.351715 | 0.001343 | +1.06% | -0.36% | 0/3 | v5 |
| time_cphl_h | tsds | 3 | 1.364836 | 0.005639 | +0.10% | -1.33% | 0/3 | v5 |
| time_cphl_h | orig | 1 | 1.366210 | — | +0.00% | -1.43% | — | v5 |
| time_nem_price_5t | rags-traj-expSum | 3 | 0.710460 | 0.003316 | +5.17% | +1.95% | 3/3 | v5 |
| time_nem_price_5t | rags-traj-linearSum | 3 | 0.710718 | 0.002332 | +5.14% | +1.92% | 3/3 | v5 |
| time_nem_price_5t | rags-traj-zeroClipSum | 3 | 0.712162 | 0.003544 | +4.94% | +1.72% | 3/3 | v5 |
| time_nem_price_5t | rags-block-expSum-uni | 3 | 0.715827 | 0.003709 | +4.46% | +1.21% | 2/3 | v5-uni |
| time_nem_price_5t | rags-block-expSum | 3 | 0.716131 | 0.000700 | +4.41% | +1.17% | 3/3 | v5 |
| time_nem_price_5t | less-main | 3 | 0.716408 | 0.000599 | +4.38% | +1.13% | 3/3 | v5 |
| time_nem_price_5t | rags-block-linearSum | 3 | 0.716635 | 0.001839 | +4.35% | +1.10% | 3/3 | v5 |
| time_nem_price_5t | rags-traj-maxSum | 3 | 0.716659 | 0.004233 | +4.34% | +1.10% | 2/3 | v5 |
| time_nem_price_5t | rags-block-zeroClipSum | 3 | 0.716862 | 0.003424 | +4.32% | +1.07% | 3/3 | v5 |
| time_nem_price_5t | rags-block-linearSum-uni | 3 | 0.720863 | 0.006682 | +3.78% | +0.52% | 2/3 | v5-uni |
| time_nem_price_5t | tsds | 3 | 0.720954 | 0.002474 | +3.77% | +0.51% | 2/3 | v5 |
| time_nem_price_5t | rags-block-maxSum | 3 | 0.722611 | 0.003252 | +3.55% | +0.28% | 2/3 | v5 |
| time_nem_price_5t | rand | 3 | 0.724624 | 0.006311 | +3.28% | +0.00% | — | v5 |
| time_nem_price_5t | orig | 1 | 0.749205 | — | +0.00% | -3.39% | — | v5 |
| time_nem_price_5t | lp-mirror | 3 | 0.755532 | 0.000901 | -0.84% | -4.27% | 0/3 | v5 |
| time_solar_h | rags-block-expSum | 3 | 1.114610 | 0.005932 | +1.30% | +1.15% | 3/3 | v5 |
| time_solar_h | lp-mirror | 3 | 1.117018 | 0.011846 | +1.09% | +0.93% | 2/3 | v5 |
| time_solar_h | rags-block-linearSum | 3 | 1.118433 | 0.009260 | +0.96% | +0.81% | 3/3 | v5 |
| time_solar_h | rags-traj-zeroClipSum | 3 | 1.118650 | 0.011620 | +0.94% | +0.79% | 2/3 | v5 |
| time_solar_h | rags-block-zeroClipSum | 3 | 1.120632 | 0.010944 | +0.77% | +0.61% | 2/3 | v5 |
| time_solar_h | less-main | 3 | 1.121734 | 0.009768 | +0.67% | +0.51% | 1/3 | v5 |
| time_solar_h | rand | 3 | 1.127534 | 0.007553 | +0.16% | +0.00% | — | v5 |
| time_solar_h | rags-traj-linearSum | 3 | 1.127492 | 0.014421 | +0.16% | +0.00% | 1/3 | v5 |
| time_solar_h | rags-traj-maxSum | 3 | 1.127492 | 0.014421 | +0.16% | +0.00% | 1/3 | v5 |
| time_solar_h | rags-traj-expSum | 3 | 1.127492 | 0.014421 | +0.16% | +0.00% | 1/3 | v5 |
| time_solar_h | orig | 1 | 1.129299 | — | +0.00% | -0.16% | — | v5 |
| time_solar_h | rags-block-maxSum | 3 | 1.131380 | 0.014981 | -0.18% | -0.34% | 1/3 | v5 |
| time_solar_h | tsds | 3 | 1.132088 | 0.001585 | -0.25% | -0.40% | 0/3 | v5 |
| time_solar_h | rags-block-linearSum-uni | 3 | 1.140119 | 0.010161 | -0.96% | -1.12% | 0/3 | v5-uni |
| time_solar_h | rags-block-expSum-uni | 3 | 1.145249 | 0.015188 | -1.41% | -1.57% | 0/3 | v5-uni |
| time_port_activity_d | orig | 1 | 0.716205 | — | +0.00% | +0.90% | — | v5 |
| time_port_activity_d | rand | 3 | 0.722692 | 0.000984 | -0.91% | +0.00% | — | v5 |
| time_port_activity_d | lp-mirror | 3 | 0.722733 | 0.000367 | -0.91% | -0.01% | 2/3 | v5 |
| time_port_activity_d | rags-block-linearSum-uni | 3 | 0.726186 | 0.000940 | -1.39% | -0.48% | 0/3 | v5-uni |
| time_port_activity_d | rags-block-expSum-uni | 3 | 0.726686 | 0.000881 | -1.46% | -0.55% | 0/3 | v5-uni |
| time_port_activity_d | rags-traj-maxSum | 3 | 0.727924 | 0.000993 | -1.64% | -0.72% | 0/3 | v5 |
| time_port_activity_d | rags-traj-zeroClipSum | 3 | 0.728760 | 0.001995 | -1.75% | -0.84% | 0/3 | v5 |
| time_port_activity_d | less-main | 3 | 0.728935 | 0.000851 | -1.78% | -0.86% | 0/3 | v5 |
| time_port_activity_d | rags-traj-linearSum | 3 | 0.728903 | 0.000890 | -1.77% | -0.86% | 0/3 | v5 |
| time_port_activity_d | rags-traj-expSum | 3 | 0.729109 | 0.001311 | -1.80% | -0.89% | 0/3 | v5 |
| time_port_activity_d | rags-block-maxSum | 3 | 0.730437 | 0.001239 | -1.99% | -1.07% | 0/3 | v5 |
| time_port_activity_d | rags-block-linearSum | 3 | 0.733143 | 0.001494 | -2.36% | -1.45% | 0/3 | v5 |
| time_port_activity_d | rags-block-expSum | 3 | 0.733374 | 0.000704 | -2.40% | -1.48% | 0/3 | v5 |
| time_port_activity_d | rags-block-zeroClipSum | 3 | 0.734667 | 0.000179 | -2.58% | -1.66% | 0/3 | v5 |
| time_port_activity_d | tsds | 3 | 0.736037 | 0.000884 | -2.77% | -1.85% | 0/3 | v5 |

## timerxl-84m

| 汇总 | 内容 |
|---|---|
| 正式矩阵 | `v5-timerxl`；9 任务 × 15 臂 × 3 seeds = 387 cells |
| 当前落盘 | 298 个去重 cell：seed0=135、seed1=126、seed2=37；未完成行明确标记 |
| 标定配方 | lr=1e-4，epochs=4，schedule_epochs=6，batch=32，cosine，weight_decay=0.01 |

| 任务 | 方法 / 臂 | 已有 seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| time_nem_demand_5t | orig | 1 | 0.959664 | — | +0.00% | +18.78% | — | v5-timerxl 已落盘 seed0 |
| time_nem_demand_5t | lp-mirror | 3 | 0.975348 | 0.036958 | -1.63% | +17.02% | 3/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rand | 3 | 1.175448 | 0.006779 | -22.49% | +0.00% | — | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-block-expSum-uni | 3 | 1.178519 | 0.007999 | -22.81% | -0.26% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-block-maxSum | 3 | 1.189152 | 0.019545 | -23.91% | -1.17% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-block-expSum | 3 | 1.197744 | 0.021420 | -24.81% | -1.90% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-block-zeroClipSum | 3 | 1.203262 | 0.018820 | -25.38% | -2.37% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-traj-expSum | 3 | 1.203907 | 0.024605 | -25.45% | -2.42% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-block-linearSum-uni | 3 | 1.206122 | 0.022992 | -25.68% | -2.61% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-traj-maxSum | 3 | 1.207490 | 0.008308 | -25.82% | -2.73% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-traj-linearSum | 3 | 1.214124 | 0.010139 | -26.52% | -3.29% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-traj-zeroClipSum | 3 | 1.214474 | 0.021378 | -26.55% | -3.32% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | less-main | 3 | 1.215959 | 0.009343 | -26.71% | -3.45% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | rags-block-linearSum | 3 | 1.216941 | 0.026495 | -26.81% | -3.53% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_nem_demand_5t | tsds | 3 | 1.289582 | 0.007171 | -34.38% | -9.71% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | orig | 1 | 1.257500 | — | +0.00% | +0.83% | — | v5-timerxl 已落盘 seed0 |
| time_azure_i_cpu_5t | lp-mirror | 3 | 1.266484 | 0.009429 | -0.71% | +0.49% | 2/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | tsds | 3 | 1.270006 | 0.019692 | -0.99% | +0.21% | 2/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rand | 3 | 1.272671 | 0.006299 | -1.21% | +0.00% | — | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-block-maxSum | 3 | 1.277439 | 0.009720 | -1.59% | -0.37% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-block-zeroClipSum | 3 | 1.279051 | 0.011558 | -1.71% | -0.50% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-block-linearSum | 3 | 1.281370 | 0.005083 | -1.90% | -0.68% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-block-linearSum-uni | 3 | 1.281928 | 0.012496 | -1.94% | -0.73% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-block-expSum-uni | 3 | 1.287636 | 0.006802 | -2.40% | -1.18% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-traj-expSum | 3 | 1.288710 | 0.003957 | -2.48% | -1.26% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | less-main | 3 | 1.291364 | 0.009061 | -2.69% | -1.47% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-traj-zeroClipSum | 3 | 1.291802 | 0.009020 | -2.73% | -1.50% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-block-expSum | 3 | 1.292172 | 0.018313 | -2.76% | -1.53% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-traj-linearSum | 3 | 1.294971 | 0.008032 | -2.98% | -1.75% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_azure_i_cpu_5t | rags-traj-maxSum | 3 | 1.295995 | 0.009851 | -3.06% | -1.83% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | orig | 1 | 1.258166 | — | +0.00% | +20.50% | — | v5-timerxl 已落盘 seed0 |
| time_darwin_water_15t | lp-mirror | 2 | 1.279860 | 0.010924 | -1.72% | +19.81% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_darwin_water_15t | rags-block-maxSum | 3 | 1.594848 | 0.001166 | -26.76% | +0.34% | 2/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rand | 3 | 1.600357 | 0.015390 | -27.20% | +0.00% | — | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rags-block-linearSum-uni | 2 | 1.600699 | 0.005585 | -27.22% | -0.29% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_darwin_water_15t | rags-block-expSum | 3 | 1.610717 | 0.010459 | -28.02% | -0.65% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rags-block-linearSum | 3 | 1.612562 | 0.011818 | -28.17% | -0.76% | 1/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rags-block-expSum-uni | 2 | 1.610386 | 0.025166 | -27.99% | -0.90% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_darwin_water_15t | rags-block-zeroClipSum | 3 | 1.621250 | 0.010213 | -28.86% | -1.31% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | tsds | 2 | 1.666510 | 0.008088 | -32.46% | -4.41% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_darwin_water_15t | rags-traj-expSum | 2 | 1.675414 | 0.014463 | -33.16% | -4.97% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_darwin_water_15t | less-main | 3 | 1.681506 | 0.007164 | -33.65% | -5.07% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rags-traj-linearSum | 3 | 1.682931 | 0.020243 | -33.76% | -5.16% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rags-traj-maxSum | 3 | 1.684490 | 0.036479 | -33.88% | -5.26% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_darwin_water_15t | rags-traj-zeroClipSum | 3 | 1.706568 | 0.008549 | -35.64% | -6.64% | 0/3 | v5-timerxl 已落盘 seed0,1,2 |
| time_finland_15t | rags-block-linearSum | 2 | 0.901060 | 0.034189 | +11.21% | +3.70% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-block-zeroClipSum | 2 | 0.905484 | 0.003789 | +10.77% | +3.22% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-block-linearSum-uni | 2 | 0.922346 | 0.080442 | +9.11% | +1.42% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rand | 2 | 0.935654 | 0.011952 | +7.80% | +0.00% | — | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-block-expSum-uni | 2 | 0.935865 | 0.020039 | +7.78% | -0.02% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-traj-zeroClipSum | 2 | 0.972551 | 0.029185 | +4.16% | -3.94% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-traj-linearSum | 2 | 0.977401 | 0.078165 | +3.69% | -4.46% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-traj-maxSum | 2 | 0.977401 | 0.078165 | +3.69% | -4.46% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-traj-expSum | 2 | 0.977401 | 0.078165 | +3.69% | -4.46% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-block-expSum | 2 | 0.991358 | 0.005640 | +2.31% | -5.95% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | rags-block-maxSum | 2 | 1.009676 | 0.087548 | +0.50% | -7.91% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | orig | 1 | 1.014798 | — | +0.00% | -9.45% | — | v5-timerxl 已落盘 seed0 |
| time_finland_15t | less-main | 2 | 1.053492 | 0.084602 | -3.81% | -12.59% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | lp-mirror | 2 | 1.276071 | 0.150972 | -25.75% | -36.38% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_finland_15t | tsds | 2 | 1.880737 | 0.001997 | -85.33% | -101.01% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | orig | 1 | 0.947177 | — | +0.00% | +11.48% | — | v5-timerxl 已落盘 seed0 |
| time_eweld_15t | rags-traj-maxSum | 2 | 1.019866 | 0.003961 | -7.67% | +5.36% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-block-zeroClipSum | 2 | 1.024557 | 0.018236 | -8.17% | +4.92% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | lp-mirror | 2 | 1.028007 | 0.033873 | -8.53% | +4.60% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | less-main | 2 | 1.038197 | 0.043629 | -9.61% | +3.66% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-block-expSum | 2 | 1.038478 | 0.035064 | -9.64% | +3.63% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-block-maxSum | 2 | 1.046327 | 0.003493 | -10.47% | +2.90% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-traj-linearSum | 2 | 1.060886 | 0.032522 | -12.01% | +1.55% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-traj-zeroClipSum | 2 | 1.067704 | 0.012136 | -12.72% | +0.92% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-traj-expSum | 2 | 1.075677 | 0.012644 | -13.57% | +0.18% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rand | 2 | 1.077587 | 0.010663 | -13.77% | +0.00% | — | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-block-expSum-uni | 2 | 1.086220 | 0.015611 | -14.68% | -0.80% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-block-linearSum | 2 | 1.113110 | 0.042855 | -17.52% | -3.30% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | rags-block-linearSum-uni | 2 | 1.135061 | 0.014906 | -19.84% | -5.33% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_eweld_15t | tsds | 2 | 1.144135 | 0.061745 | -20.79% | -6.18% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-block-linearSum | 2 | 1.485682 | 0.014587 | +0.60% | +1.05% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-block-zeroClipSum | 2 | 1.487753 | 0.008287 | +0.46% | +0.91% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-block-expSum | 2 | 1.492417 | 0.004887 | +0.15% | +0.60% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | orig | 1 | 1.494688 | — | +0.00% | +0.16% | — | v5-timerxl 已落盘 seed0 |
| time_cphl_h | rand | 2 | 1.501439 | 0.006133 | -0.45% | +0.00% | — | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-block-maxSum | 2 | 1.505156 | 0.003264 | -0.70% | -0.25% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | lp-mirror | 2 | 1.506511 | 0.017021 | -0.79% | -0.34% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-traj-expSum | 2 | 1.518454 | 0.001379 | -1.59% | -1.13% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-traj-linearSum | 2 | 1.520192 | 0.015106 | -1.71% | -1.25% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-traj-zeroClipSum | 2 | 1.520234 | 0.017026 | -1.71% | -1.25% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-traj-maxSum | 2 | 1.520560 | 0.007456 | -1.73% | -1.27% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-block-linearSum-uni | 2 | 1.521320 | 0.004233 | -1.78% | -1.32% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | rags-block-expSum-uni | 2 | 1.521853 | 0.022264 | -1.82% | -1.36% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | less-main | 2 | 1.523629 | 0.031704 | -1.94% | -1.48% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_cphl_h | tsds | 2 | 1.562217 | 0.016956 | -4.52% | -4.05% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | orig | 1 | 0.772254 | — | +0.00% | +13.74% | — | v5-timerxl 已落盘 seed0 |
| time_nem_price_5t | lp-mirror | 2 | 0.785706 | 0.001433 | -1.74% | +12.28% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rand | 2 | 0.895692 | 0.000536 | -15.98% | +0.00% | — | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-traj-maxSum | 2 | 0.896627 | 0.011721 | -16.11% | -0.10% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-block-expSum-uni | 2 | 0.897738 | 0.001470 | -16.25% | -0.23% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-block-linearSum | 2 | 0.898796 | 0.004758 | -16.39% | -0.35% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-block-linearSum-uni | 2 | 0.899411 | 0.014708 | -16.47% | -0.42% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | less-main | 2 | 0.899629 | 0.002637 | -16.49% | -0.44% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-traj-linearSum | 2 | 0.900648 | 0.003536 | -16.63% | -0.55% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-block-maxSum | 2 | 0.901367 | 0.000831 | -16.72% | -0.63% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-traj-zeroClipSum | 2 | 0.901482 | 0.000419 | -16.73% | -0.65% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-block-zeroClipSum | 2 | 0.901783 | 0.001960 | -16.77% | -0.68% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-block-expSum | 2 | 0.901924 | 0.006605 | -16.79% | -0.70% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | rags-traj-expSum | 2 | 0.902693 | 0.000194 | -16.89% | -0.78% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_nem_price_5t | tsds | 2 | 0.907940 | 0.004089 | -17.57% | -1.37% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-block-linearSum-uni | 2 | 1.127012 | 0.003952 | +8.04% | +2.36% | 2/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-block-zeroClipSum | 2 | 1.143534 | 0.007622 | +6.69% | +0.93% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-block-expSum-uni | 2 | 1.145378 | 0.018340 | +6.54% | +0.77% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-block-linearSum | 2 | 1.149304 | 0.001919 | +6.22% | +0.43% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | tsds | 2 | 1.152391 | 0.032720 | +5.97% | +0.16% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-block-maxSum | 2 | 1.152848 | 0.023949 | +5.93% | +0.12% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rand | 2 | 1.154251 | 0.020804 | +5.81% | +0.00% | — | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-block-expSum | 2 | 1.156867 | 0.009285 | +5.60% | -0.23% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-traj-linearSum | 2 | 1.160468 | 0.020190 | +5.31% | -0.54% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-traj-maxSum | 2 | 1.160468 | 0.020190 | +5.31% | -0.54% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-traj-expSum | 2 | 1.160468 | 0.020190 | +5.31% | -0.54% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | lp-mirror | 2 | 1.160836 | 0.014279 | +5.28% | -0.57% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | rags-traj-zeroClipSum | 2 | 1.162497 | 0.020734 | +5.14% | -0.71% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | less-main | 2 | 1.163343 | 0.005155 | +5.07% | -0.79% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_solar_h | orig | 1 | 1.225503 | — | +0.00% | -4.84% | — | v5-timerxl 已落盘 seed0 |
| time_port_activity_d | rags-block-maxSum | 2 | 0.787526 | 0.006474 | +0.54% | +0.53% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-traj-maxSum | 2 | 0.789506 | 0.004050 | +0.29% | +0.28% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-block-expSum-uni | 2 | 0.789704 | 0.001462 | +0.27% | +0.26% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | orig | 1 | 0.791821 | — | +0.00% | +0.13% | — | v5-timerxl 已落盘 seed0 |
| time_port_activity_d | rags-block-linearSum | 2 | 0.791179 | 0.000700 | +0.08% | +0.07% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | less-main | 2 | 0.791334 | 0.006892 | +0.06% | +0.05% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rand | 2 | 0.791740 | 0.001587 | +0.01% | +0.00% | — | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-block-zeroClipSum | 2 | 0.791915 | 0.001782 | -0.01% | -0.02% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-block-linearSum-uni | 2 | 0.793482 | 0.002857 | -0.21% | -0.22% | 1/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-traj-linearSum | 2 | 0.793748 | 0.003364 | -0.24% | -0.25% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | lp-mirror | 2 | 0.794896 | 0.002558 | -0.39% | -0.40% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-traj-zeroClipSum | 2 | 0.795113 | 0.003586 | -0.42% | -0.43% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-traj-expSum | 2 | 0.795469 | 0.000126 | -0.46% | -0.47% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | rags-block-expSum | 2 | 0.796574 | 0.001122 | -0.60% | -0.61% | 0/2 | v5-timerxl 已落盘 seed0,1 |
| time_port_activity_d | tsds | 2 | 0.803700 | 0.005020 | -1.50% | -1.51% | 0/2 | v5-timerxl 已落盘 seed0,1 |

## 口径与追溯

- v5 的 13 臂来自 `MAIN.md`；v5-uni 的两个 uni 臂来自 `UNI.md`；同名 v5 基线不重复计入。
- `orig` 通常只计 seed0；训练臂和 rand 通常为 3 seeds；TimerXL 表只汇总目前实际落盘的 seeds。
- TimerXL 的重复 `orig` 文件已按 `(任务, 臂, seed)` 去重；缺少 rand 配对的未完成 cell 不计算相对改善。
- 需要最终论文数字时，以 TimerXL 正式 run 完成后的 cell 汇总为准，并回填本文件第三张表。
