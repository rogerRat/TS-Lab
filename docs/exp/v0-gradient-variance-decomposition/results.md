# v0-gradient-variance-decomposition：梯度方差分解

最后修改日期：2026-11-08

## 问题与固定输入

block 级向量方差主要反映梯度幅度还是方向变化，以及控制梯度尺度和轨迹身份能否消除此前向量方差与线性重建余弦之间的正相关？精确复用来自 v0-var-proxy-estimate 的同样 46 个完整、不重叠的 256 起点 block，跨越同样的十条保留轨迹。复用完整全梯度 Gram 矩阵、八点输出 head proxy 方差，以及局部位置 0、85、170、255 处的四锚点线性重建。没有进行任何模型、梯度或 proxy 调用。

对每个 block，令 `r_t=||g_t||`、`u_t=g_t/r_t`，并使用对全部 256 个起点的总体平均。定义平均梯度范数 `E r_t`、幅度方差 `Var(r_t)=E r_t^2-(E r_t)^2`、完整方向离散度 `D_h=1-||E u_t||^2`、总向量方差 `V_g=E r_t^2-||E g_t||^2` 以及相对向量方差 `V_g/E r_t^2`。这里的 `D_h` 由完整全梯度方向计算，而非输出 head proxy。精确的方向方差分量为 `V_g-Var(r_t)=(E r_t)^2-||E g_t||^2`；当幅度与方向协变时，它一般不等于 `(E r_t)^2 D_h`。

从同样的 Gram block 重算平均 L2、绝对范数差、方向余弦和 `1-cosine`。同时记录全部 256 个位置和 252 个非锚点位置。原始 Pearson 和 Spearman 相关性合并 46 个 block。等 track Spearman 对九个已定义的 track 内相关性求平均；track 00 只有两个 block。受控 Pearson 将两个变量对平均梯度范数和十个 track 固定效应做残差化。受控 Spearman 先对两个变量和平均梯度范数做全局排序，然后在排序后的平均范数和 track 固定效应上残差化这些秩，再做相关。这些是描述性的偏相关，不含独立窗口 p 值。留一 track 范围通过在省略每条 track 后重新计算受控秩关联得到。

入口：`scripts/v0_gradient_variance_decomposition.py`。环境：`rage` Conda。命令：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src /root/autodl-tmp/rage/envs/rage/bin/python -u -m scripts.v0_gradient_variance_decomposition`。输出：`outputs/train/v0-gradient-variance-decomposition/`。日志：`logs/v0-gradient-variance-decomposition/`。

## 方差构成与尺度

| 跨 46 个 block 的关联 | 合并 Pearson | 合并 Spearman | 受控秩关联 |
|---|---:|---:|---:|
| `V_g` 对平均梯度范数 | 0.945 | 0.970 | 未定义：平均范数是控制变量 |
| `V_g` 对幅度方差 | 0.690 | 0.569 | 0.663 |
| `V_g` 对方向离散度 `D_h` | -0.123 | -0.222 | 0.578 |
| `V_g` 对相对向量方差 | -0.282 | -0.289 | 0.603 |

平均梯度范数在 block 间的范围为 1.337-6.467，并与绝对向量方差强烈共变。幅度方差范围为 0.331-13.157。精确的幅度方差占比 `Var(r_t)/V_g` 的中位数为 0.176、均值为 0.202；仅有 46 个 block 中的一个超过 0.5。因此幅度方差在数值上并未主导 block 内的向量方差分解。剩余分量与方向有关但带有幅度加权。`V_g` 对 `D_h` 的合并关联为负，而受控秩关联为正，表明 block 尺度与 track 差异会掩盖方向关系。此处的"尺度驱动"描述的是绝对 `V_g` 的跨 block 变化；它并不意味着 block 内方差主要是幅度方差。

## 重建关联

以下数值使用每个 block 的 252 个非锚点位置。包含锚点的相关性在显示精度上与前者相同，因为每个 block 中都出现同样的四个精确锚点：非锚点平均误差等于全位置平均误差乘以 `256/252`，而非锚点平均余弦是全位置平均余弦的仿射变换。排除锚点会改变指标水平，但不改变 Pearson、Spearman 或这些残差化关联。

| 预测子对重建目标 | 合并 Pearson | 合并 Spearman | 等 track Spearman | 受控 Pearson | 受控秩关联 |
|---|---:|---:|---:|---:|---:|
| `V_g` 对幅度误差 | 0.702 | 0.638 | 0.622 | 0.420 | 0.286 |
| 幅度方差对幅度误差 | 0.707 | 0.541 | 0.567 | 0.560 | 0.373 |
| `V_g` 对方向误差 | -0.122 | -0.324 | -0.444 | 0.109 | 0.431 |
| `D_h` 对方向误差 | 0.776 | 0.701 | 0.689 | 0.842 | 0.713 |
| 相对向量方差对方向误差 | 0.546 | 0.639 | 0.544 | 0.585 | 0.605 |
| 八点 proxy 方差对方向误差 | -0.177 | -0.398 | -0.344 | -0.001 | 0.202 |

方向误差为 `E[1-cos(g_t, ghat_t)]`，因此其相关性符号与平均余弦者相反。原本 `V_g` 对余弦的合并 Spearman 正相关 0.324，在控制平均梯度范数和 track 后变为受控秩关联 -0.431（等价地，对方向误差为 +0.431）。八 proxy 方差对余弦的关联从合并 +0.398 变为受控 -0.202。方向离散度比绝对向量方差更直接地与方向重建误差相关：以正误差目标计，合并 Spearman 为 0.701 对 -0.324，受控秩关联为 0.713 对 0.431。相对向量方差在尺度归一化后也与方向误差呈正对齐。

留一 track 的受控秩范围：`V_g` 对余弦为 -0.524 到 -0.329，proxy 方差对余弦为 -0.349 到 -0.057，`D_h` 对方向误差为 0.617 到 0.756。它们的符号在全部十次省略中保持不变，尽管 proxy 关联较弱。这是一个固定效应的描述性敏感性检查，而非因果结论或总体层面的推断。

非锚点平均幅度误差为 1.100，含锚点为 1.083；平均方向误差为 0.648，含锚点为 0.638。排除后每个 block 使用 256 个点中的 252 个。因此，合并的方差-余弦正相关并非由四个精确锚点位置造成。其方向在控制梯度尺度和 track 身份后发生改变。

## 图表与产物

`figures/scatter_all.png` 和 `figures/scatter_nonanchor.png` 展示按 track 着色的 46 个 block。每个图都包含全方差对余弦、proxy 方差对余弦、全方向离散度对方向误差的原始与受控秩面板。同时也提供 SVG 版本。`block_metrics.csv` 包含全部 46 行派生 block 记录，`correlations.csv` 给出所有要求检查的合并、等 track 和受控关联，`input_manifest.json` 记录定义和源路径。`.complete` 标记存在。所选十条轨迹和相互重叠的原始窗口限制了泛化性；回归只有 46 个 block 观测值和十个 track 分组。
EOF'

## v0-gradient-variance-decomposition：梯度方差分解

于 2026-11-08 完成，使用来自 v0-var-proxy-estimate 的 46 个固定 256 起点 block、全部缓存全梯度 Gram 矩阵，以及同样的四锚点线性重建。该表报告在调整平均梯度范数和 track 固定效应后的合并 Spearman 与受控秩关联。重建误差使用每个 block 的 252 个非锚点位置。

| 预测子与目标 | 合并 Spearman | 受控秩关联 |
|---|---:|---:|
| 向量方差对平均梯度范数 | 0.970 | 平均范数是控制变量 |
| 向量方差对幅度方差 | 0.569 | 0.663 |
| 向量方差对方向离散度 | -0.222 | 0.578 |
| 向量方差对幅度重建误差 | 0.638 | 0.286 |
| 方向离散度对方向重建误差 | 0.701 | 0.713 |
| 相对向量方差对方向重建误差 | 0.639 | 0.605 |
| 向量方差对方向重建误差 | -0.324 | 0.431 |

总向量方差中幅度方差占比的中位数为 0.176；46 个 block 中只有一个的占比超过 0.5。这区分了与梯度尺度的强跨 block 关联和 block 内方差分解。原本向量方差对重建余弦的合并 Spearman 正相关 +0.324，在受控秩关联下变为 -0.431，而受控 Pearson 仅为 -0.109。八 proxy 方差对余弦从合并 Spearman +0.398 变为受控秩关联 -0.202，受控 Pearson 接近零（+0.001）。这些是固定十条 track 样本的描述性结果。

排除四个精确锚点使平均幅度误差从 1.083 升至 1.100，平均方向误差从 0.638 升至 0.648。相关性保持不变，因为每个 block 排除的是同样的四个零误差位置。完整定义、Pearson 和逐 track 结果、留一 track 检查以及局限性：docs/exp/v0-gradient-variance-decomposition/results.md。原始与受控散点图为 `outputs/train/v0-gradient-variance-decomposition/figures/scatter_all.png` 和 `scatter_nonanchor.png`（另有 SVG）。block 数据、相关性表和 `.complete` 位于 `outputs/train/v0-gradient-variance-decomposition/`。
