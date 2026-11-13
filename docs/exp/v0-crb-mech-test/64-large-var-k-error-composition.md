# 64-large-var-k-error-composition

最后修改日期：2026-11-11

完整聚合结果表（全部 q/K、点加权与等块权、方法 CI、选择效率、MSE 分解和 Spearman）见[完整表格文档](64-large-var-k-error-composition-full-tables.md)。

## 目的与设置

本批固定 64 条 mock 轨迹、4096 维、每条 4000 点，固定 Uniform anchors 与原始 Linear 重建，仅改变虚拟分块区间。预算 q={4,8,16,32,64,128}，K=0..q−1；Random-K 每条轨迹重复 100 次，paired bootstrap 5000 次。比较 Random-K、Cancellation-K、Endpoint-Difference-K、Predicted-Gain-K、Oracle-L2-K、Oracle-Direction-K。非 Oracle 选区只使用锚点端点；完整轨迹仅用于离线评估及 Oracle。MSE 分解中的块代表是 Uniform+Linear 重建在块内的均值。

Mock 生成目标按用户最终澄清实现：直接相邻点的完整向量差比

`||g[t+1]-g[t]||_2 / ((||g[t]||_2+||g[t+1]||_2)/2) ≈ 0.5`，

且 lag-16 两点方向余弦接近 0。轨迹由长度 8 的滚动随机方向和乘以平滑宽幅模长包络构成。实测跨轨迹平均相邻点差比为 0.50005（轨迹均值的 5–95 百分位：0.49972–0.50041）；lag-16 cosine 均值为 0.0000375、绝对值均值 0.01245、标准差 0.01558。梯度模长 CV 平均 0.3225，点模长 5–95 百分位均值约 0.484–1.516。目标达成。

## 主结果：K=0 对比全部原区间切分

以下为按点加权、64 条轨迹等权平均的 est-real L2 与方向误差。K=q−1 时所有原始区间都切分，因此策略结果相同。

| q | L2 K=0 | L2 K=q−1 | L2 变化 | Direction K=0 | Direction K=q−1 | Direction 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.65422 | 1.79026 | +0.13604 | 0.99557 | 0.99408 | −0.00148 |
| 8 | 1.64175 | 1.77191 | +0.13015 | 0.99011 | 0.98667 | −0.00343 |
| 16 | 1.62284 | 1.74380 | +0.12097 | 0.97951 | 0.97237 | −0.00714 |
| 32 | 1.58835 | 1.69191 | +0.10357 | 0.95684 | 0.94188 | −0.01496 |
| 64 | 1.51896 | 1.58748 | +0.06853 | 0.91176 | 0.88144 | −0.03032 |
| 128 | 1.37707 | 1.37463 | −0.00244 | 0.82054 | 0.75950 | −0.06104 |

增加虚拟分块明显降低 Direction error，但 q≤64 时总 L2 反而上升；q=128 的 L2 改善很小（约 0.18%）。这说明该 mock 上不能把“更多块”直接视为更好的 L2 估计。

## MSE 分解

对每块使用真实均值 μ 和估计均值 μ-hat：Representation MSE 是块内点相对 μ 的误差，Estimation MSE 是 μ-hat 相对 μ 的误差。验证 `Total = Representation + Estimation` 的最大绝对残差为 7.55e−15；对既有块级 total 交叉核验的最大差为 9.10e−13。轨迹等权与点权汇总一致（每条轨迹长度相同）。

K=q−1 相对 K=0 的分量变化（所有策略相同）：

| q | Representation Δ | Estimation Δ | Total Δ |
|---:|---:|---:|---:|
| 4 | −0.00660 | +0.14264 | +0.13604 |
| 8 | −0.01527 | +0.14543 | +0.13015 |
| 16 | −0.03215 | +0.15311 | +0.12097 |
| 32 | −0.06441 | +0.16798 | +0.10357 |
| 64 | −0.12202 | +0.19055 | +0.06853 |
| 128 | −0.21061 | +0.20817 | −0.00244 |

分块确实降低 Representation MSE，但也使块均值估计误差上升；q≤64 时后者占主导。q=128 时二者近乎抵消，Total 略降。

## 区间选择机制

K=1 时 Cancellation 相对 Random 的点加权 paired L2 差值（method−random，负数较好）：

| q | 差值 | Paired 95% CI |
|---:|---:|---:|
| 4 | +0.002784 | [−0.000362, +0.005944] |
| 8 | +0.002947 | [+0.000475, +0.005453] |
| 16 | −0.000538 | [−0.000917, −0.000157] |
| 32 | +0.000512 | [+0.000067, +0.000944] |
| 64 | +0.000323 | [+0.000189, +0.000453] |
| 128 | −0.000014 | [−0.000030, +0.000002] |

Cancellation 在 K=1 的 L2 上没有稳定优势；除 q=16 的小幅改善外，多数 q 持平或更差。Direction 上 Cancellation 的差值在 q=32/64/128 为 −1.61e−5/−2.41e−5/−2.43e−5，paired 95% CI 均低于 0；低 q 的 CI 覆盖 0。Endpoint-Difference 与 Predicted-Gain 在 q≤64 的 K=1 L2 上总体比 Random 更差；q=128 才有很小的相对改善。

pooled 区间分数与实际单区间收益的 Spearman：Cancellation 对 L2/Direction 为 0.304/0.175；Endpoint-Difference 为 −0.055/0.010；Predicted-Gain 为 −0.695/0.014（15,744 个区间）。对本批 est-real 总误差，端点差异和解析预测收益并未成为有效的区间选择信号。Predicted-Gain 更像在预测线性片段的表示误差下降，却没有覆盖分块后块估计项的增加。

Selection efficiency 在 K≤64 时常出现负 Oracle gain，因此比值不应按通常“效率百分比”解释；selection_efficiency.csv 保留原始分子分母及 near-zero 标记。q=128、K=1 的 Oracle-L2 gain 为正约 2.20e−4，而 Random/Cancellation/Endpoint/Predicted 的效率约为 0.090/0.153/0.325/0.221；可部署策略与 Oracle 仍有明显差距。各 q、K 的完整误差、收益、CI、等块权诊断、效率和相关性都在 CSV 中。

## 产物与复现

- 配置：`configs/rage/v0-crb-mech-64-large-var-k-error-composition.json`
- 生成器：`scripts/v0_crb_mech_generate_large_var.py`
- K 曲线：`scripts/v0_crb_mech_virtual_split_selection.py`
- MSE 分解：`scripts/v0_crb_mech_virtual_split_mse_decomposition.py`
- 数据：`data/v0-crb-mech-test/64-large-var/gradient_tracks.npy`
- 输出：`outputs/train/v0-crb-mech-test/64-large-var-k-error-composition/`
- K-error 12 张图（6 个 q × L2/Direction）；MSE 18 张图（6 个 q × Representation/Estimation/Total）。K 曲线汇总 5,040 行，轨迹曲线 193,536 行，区间 15,744 行；MSE 汇总 4,536 行，轨迹级 96,768 行，区间 15,744 行。两个阶段的 `.complete` 均存在。

## 与 64-smooth 的跨批次解释

在共同预算 q={4,8,16,32} 的全切分端点，64-smooth 的点权 Total L2 MSE 分别下降 0.5942、0.2738、0.1313、0.0677；对应 Representation MSE 下降量远大于 Estimation MSE 上升量。以 q=4 为例，ΔRepresentation=−0.8016、ΔEstimation=+0.2074，净变化 −0.5942。新批次方向则相反：q=4 时 ΔRepresentation=−0.0066、ΔEstimation=+0.1426、Total +0.1360；q=32 时分别为 −0.0644、+0.1680、Total +0.1036。相对基线，旧批次全切分 L2 下降约 25%–49%，新批次 q≤64 上升约 6.5%–8.2%。这说明两种 mock 对分块代表的响应差异很大，不只是重建误差数值不同。

需准确描述分块范围：本轮 K 只到 q−1，即每个原 Uniform 区间最多加一个 midpoint，形成约 2(q−1) 个块；这不是把 4000 个时间点切成单点块，也没有计算新梯度。Direction 指真实点方向与该块重建代表方向之间的平均 1−cosine，MSE 分解只适用于 L2。

针对“是否存在内部最优 K”：当 q≤64 时，新批次 64 条轨迹在所有被评估的 L2 分块策略下，逐轨迹最优都为 K=0；此处估计误差增长从开始就压过表示误差下降，所以最优是“不切分”，不是内部切分。q=128 时则出现内部最优：平均曲线上 Oracle-L2 最优 K=81/127，L2 从 1.377069 降到 1.373246（约 0.28%）；Endpoint-Difference 与 Predicted-Gain 的最优 K 均为 117，误差约 1.374619；Cancellation 与 Random 的均值曲线最优在 K=127，误差约 1.374631。逐轨迹上，q=128 的 Oracle-L2 在 64/64 条轨迹的最佳 K 都是内部点；Endpoint-Difference/Predicted-Gain 有 61/64 条、Cancellation 有 46/64 条内部最优，余下主要在最大 K。q=128 的 Direction 则所有策略和逐轨迹均以最大 K 最优。

因此本实验复现了“表示收益与估计代价竞争时会形成折中”的机制，但不是所有预算下都出现内部最优：早期估计代价占优会把最佳点推到 K=0；如果随着切分增加，表示收益后来超过代价，则会出现内部最优；若表示收益始终占优，则最佳点在最大测试 K。q=128 的 oracle 给出了清楚的内部最优，可部署排序也大多出现内部最优但与 oracle 有差距。两批 mock 的生成过程不止改变一个因素，因此这是对机制的支持，不是单独证明“变化幅度”是唯一原因。

## 已有结果摘要

最新完成批次 `64-large-var-k-error-composition`：mock 达成相邻点完整向量差比约 0.5、lag-16 cosine 约 0；模长 CV=0.323。全切分使 Direction error 在所有 q 都下降，但 L2 在 q≤64 上升、q=128 仅下降 0.18%。Cancellation K=1 对 L2 没有稳定优于 Random；Endpoint-Difference/Predicted-Gain 也未显示一致优势。12 张 K-error 图、18 张 MSE 分解图及 paired 95% CI、相关性与 Oracle 差距见[批次报告](64-large-var-k-error-composition.md)。


跨批次结果显示：64-smooth 在共同 q≤32 下全切分均使 L2 总误差下降；64-large-var 在 q≤64 的 L2 最佳点为 K=0，而 q=128 的 Oracle-L2 曲线出现内部最优 K=81/127。此处 K=q−1 只给每个 Uniform 区间再加一个 midpoint，并非切成逐点块。详细机制解释与逐轨迹最优 K 统计见 [64-large-var 批次报告](64-large-var-k-error-composition.md)。
