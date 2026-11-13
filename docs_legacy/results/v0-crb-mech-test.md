# v0-crb-mech-test

最后修改日期：2026-11-11

## 实验状态与产物

64 条轨迹均完成；每条轨迹在 q=4、8、16、32 下分别执行 CRB 与等预算 Uniform 锚点重建，共 512 条方法级指标记录，且 outputs/train/v0-crb-mech-test/.complete 存在。原始文件：outputs/train/v0-crb-mech-test/trajectory_metrics.csv、summary.json。运行耗时约 7.6 分钟；日志位于 logs/v0-crb-mech-test/。

## 汇总结果

误差均对 64 条轨迹等权平均。方向余弦误差为逐点 1-cos(x, xhat)。CRB 优胜数分别统计 L2 和余弦误差低于 Uniform 的轨迹数。Spearman 数值为粗糙度分数与 CRB-Uniform 误差差值的相关系数；正值表示粗糙度较高时 CRB 相对更差。

| q | L2 CRB | L2 Uniform | CRB L2 优胜 | 余弦误差 CRB | 余弦误差 Uniform | CRB 余弦优胜 | rho(粗糙度, L2差) | rho(粗糙度, 余弦差) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.227648 | 1.165281 | 23/64 | 0.176501 | 0.151011 | 15/64 | 0.037 | 0.052 |
| 8 | 0.782398 | 0.716502 | 8/64 | 0.064237 | 0.057248 | 16/64 | -0.259 | -0.174 |
| 16 | 0.472314 | 0.423123 | 2/64 | 0.024420 | 0.020920 | 8/64 | 0.059 | 0.043 |
| 32 | 0.245018 | 0.195215 | 0/64 | 0.007385 | 0.004469 | 0/64 | 0.210 | 0.466 |

逐坐标 RMSE 和相对 L2 的完整汇总见 outputs/train/v0-crb-mech-test/summary.json；逐轨迹指标和锚点索引见 CSV。

## 结果解读

在本组 mock 曲线上，Uniform 的平均 L2 与方向余弦误差在四种预算下均低于 CRB。q=32 时 Uniform 在 64/64 条轨迹上两项误差都更低；q=16 时 L2 为 62/64、余弦误差为 56/64。CRB-Uniform 的 L2 差随粗糙度的 Spearman 相关在 q=4/8/16/32 分别为 0.037、-0.259、0.059、0.210；余弦误差差值对应为 0.052、-0.174、0.043、0.466。样本显示较高预算下 CRB 相对劣势与粗糙度更一致；这是固定合成机制下的描述性相关，不是因果结论。

## 方法和限制

轨迹长度 4000、维数 4096；64 个不同随机种子。每条轨迹为 12 维随机正弦系数映射到随机 4096 维基底，参数在 0.05–2.0 几何递增；粗糙度分数为二阶差分 RMS 除以平均向量范数。CRB 使用项目现有 CRBEngine；端点计入覆盖预算，剩余锚点按 2:1:1 的覆盖/误差/分裂比例离散分配。q=4 的离散配额为 3/1/0，q=8 为 4/2/2，q=16 为 6/4/4，q=32 为 16/8/8。Uniform 等间距包含两端点。两种方法都对全轨迹逐维做相邻锚点线性插值。

随机相位、频率分量和基底会造成经验粗糙度分数波动，因此参数顺序与经验粗糙度排序并不完全相同；关联分析使用经验粗糙度分数。当前只测试一种合成曲线族和单个总体 seed，结论不应外推到真实梯度或其他函数族。


## 极平缓单轨迹补充

使用独立配置 configs/rage/v0-crb-mech-test-ultra-smooth.json，频率范围降至 0.005–0.05 周期/归一化时间，粗糙度参数 0.0001，并将轨迹归一化至平均向量范数 1。4000 点、4096 维；q=4/8/16/32；CRB 与 Uniform 各用相同数量锚点。经验粗糙度分数为 1.8373e-7，约为原实验最平缓轨迹的 1/246。产物位于 outputs/train/v0-crb-mech-test-ultra-smooth/，8 条方法记录及 .complete 均已核实。

| q | L2 CRB | L2 Uniform | 余弦误差 CRB | 余弦误差 Uniform |
|---:|---:|---:|---:|---:|
| 4 | 3.967e-04 | 2.822e-04 | 9.686e-08 | 4.412e-08 |
| 8 | 6.941e-05 | 5.184e-05 | 2.465e-08 | 2.253e-08 |
| 16 | 1.363e-05 | 1.129e-05 | 2.296e-08 | 2.374e-08 |
| 32 | 2.947e-06 | 2.645e-06 | 2.480e-08 | 2.484e-08 |

该轨迹上两种方法的误差都随预算增加快速下降，q=32 的平均向量 L2 已约为 3e-6。Uniform 的 L2 在四个 q 下都更低。方向余弦误差约为 1e-8 至 1e-7；CRB 在 q=16、32 的数值略低，但差值仅约 7.7e-10 和 4.5e-11，接近浮点误差尺度，不视为有意义的优势。单条轨迹不能用于相关性分析或一般性结论。


## 块内代表性补充：L2 方差与方向方差

对每种方法保留的锚点，按相邻锚点划分互不重叠的块，每个点恰好进入一个块。以块内向量均值作为块内代表，计算 population mean squared vector L2 deviation（vector_l2_variance）、该值除以块内平均平方范数的 relative_vector_variance、块内向量相对块均值方向的平均 1-cos（direction_variance），并额外计算向量模长的标量方差（norm_variance）。先对每条轨迹的块取等权均值，再对 64 条轨迹等权平均。该分析衡量块内同质性；块内均值使用了块内全部向量，是描述性 oracle 代表，不是 CRB 的锚点可用重建器。

| q | Vector L2 variance CRB / Uniform | Relative vector variance CRB / Uniform | Direction variance CRB / Uniform | Norm variance CRB / Uniform |
|---:|---:|---:|---:|---:|
| 4 | 1.5140 / 1.5552 | 0.25883 / 0.26509 | 0.14327 / 0.14505 | 0.05673 / 0.05642 |
| 8 | 0.6109 / 0.6429 | 0.10651 / 0.11438 | 0.05426 / 0.05854 | 0.01817 / 0.01633 |
| 16 | 0.2559 / 0.2702 | 0.04449 / 0.04844 | 0.02218 / 0.02421 | 0.00479 / 0.00481 |
| 32 | 0.09504 / 0.09884 | 0.01598 / 0.01775 | 0.00791 / 0.00879 | 0.00147 / 0.00138 |

64 条轨迹中，CRB 的相对向量方差低于 Uniform 的轨迹数在 q=4/8/16/32 分别为 35/64、54/64、58/64、64/64；方向方差对应为 35/64、53/64、58/64、64/64。模长方差的胜负并不一致，CRB 较低轨迹数为 37/64、30/64、33/64、22/64。因此 CRB 的块划分在较高预算下更好地分组相近方向/向量；模长异质性没有同样稳定的趋势。

这与之前逐点线性插值结果并不矛盾：插值误差评价锚点之间的线性重建；这里评价每个分块内部围绕块内均值的离散程度。CRB 可产生更同质的块，但若其锚点间线性插值曲线不贴合真实轨迹，整体插值误差仍可能高于均匀锚点。

极平缓单轨迹中，四个 q 下 Uniform 的 vector L2 variance、relative vector variance、direction variance 和 norm variance 均略低。该单条结果仅描述此轨迹。逐块结果与汇总分别在 outputs/train/v0-crb-mech-test/block_variance/64/ 和 outputs/train/v0-crb-mech-test/block_variance/ultra-smooth/，二者均有 .complete；入口为 scripts/v0_crb_mech_block_variance.py。

## CES：Curvature-Equidistributed Sampling

CES uses half of the q budget for uniformly spaced pilot anchors, including endpoints; q=4 is constrained to three pilot anchors so a second difference can be formed, leaving one adaptive anchor. Nonuniform three-point differences estimate vector curvature at interior pilots. The remaining anchors are distributed over pilot intervals in proportion to interval length times (curvature^2 + epsilon^2)^(1/5), using largest-remainder integer allocation; points within each interval are evenly spaced. Thus the adaptive density follows curvature^(2/5), while interval length accounts for the integrated mass. CES anchors are evaluated by full 4096-dimensional piecewise-linear reconstruction. The original Uniform and CRB anchor sets and outputs are reused.

| q | Mean L2: Uniform / CRB / CES | CES vs Uniform (lower tracks) | CES vs CRB (lower tracks) | Mean cosine error: Uniform / CRB / CES |
|---:|---:|---:|---:|---:|
| 4 | 1.1653 / 1.2276 / 1.2258 | 17/64 | 33/64 | 0.15101 / 0.17650 / 0.17478 |
| 8 | 0.7165 / 0.7824 / 0.7243 | 25/64 | 54/64 | 0.05725 / 0.06424 / 0.05942 |
| 16 | 0.4231 / 0.4723 / 0.4268 | 25/64 | 62/64 | 0.02092 / 0.02442 / 0.02130 |
| 32 | 0.1952 / 0.2450 / 0.1977 | 19/64 | 64/64 | 0.00447 / 0.00739 / 0.00468 |

CES is much closer to Uniform than CRB on the average full-trajectory reconstruction metrics. CES average L2 remains slightly higher than Uniform at every q (by about 0.9–5.2%), but is lower than CRB at every q; it beats CRB in per-track L2 on 33/64, 54/64, 62/64, and 64/64 tracks as q increases. Direction-error averages show the same ordering; CES beats CRB on 38/64, 45/64, 54/64, and 64/64 tracks. CES does not dominate Uniform, especially at q=4, and no significance claim is made.

Block-internal statistics use the block-mean representative protocol described above. Equal-trajectory means for relative vector variance (CRB / Uniform / CES) were:
- q=4: 0.25883 / 0.26509 / 0.25758
- q=8: 0.10651 / 0.11438 / 0.11335
- q=16: 0.04449 / 0.04844 / 0.04839
- q=32: 0.01598 / 0.01775 / 0.01784

CRB retains the lowest relative vector and direction dispersion at q=8–32; CES is close to Uniform for those budgets. At q=4, CES is marginally below both baselines on these block metrics. This differs from the full-trace interpolation comparison because each statistic has a different target: a block's internal homogeneity versus pointwise reconstruction from anchor interpolation. Block means use all points inside each block and are descriptive oracle representatives, not a CES reconstruction available from the anchors alone.

Artifacts: outputs/train/v0-crb-mech-test/ces/ contains anchor_metrics.csv (all three methods), summary.json, and ces_anchor_details.jsonl. Block metrics are in outputs/train/v0-crb-mech-test/block_variance/ces-comparison/. Both output directories have .complete. The reproducible entry points are scripts/v0_crb_mech_ces.py and scripts/v0_crb_mech_block_variance.py; fixed config is configs/rage/v0-crb-mech-test-ces.json.

## CES-Oracle：使用真实曲率

CES-Oracle 使用 mock 生成器已知解析式，直接计算完整 4096 维 g''(t) 的范数作为真实曲率；均匀 pilot 集合、预算拆分、曲率的 2/5 次方密度与总 q 均和 CES 相同。第二阶段按 pilot 区间的真实曲率密度积分分配锚点，并用区间内真实密度的逆 CDF 分位点放置。这样测试的是这个曲率密度方案在理想曲率信息下的表现，不是所有可能锚点布置中的全局 oracle 最优解。

| q | L2：Uniform / CRB / CES / CES-Oracle | CES-Oracle L2 优于 CRB 的轨迹数 | 方向误差：Uniform / CRB / CES / CES-Oracle | CES-Oracle 方向误差优于 CRB 的轨迹数 |
|---:|---:|---:|---:|---:|
| 4 | 1.1653 / 1.2276 / 1.2258 / 1.2466 | 21/64 | 0.15101 / 0.17650 / 0.17478 / 0.18395 | 24/64 |
| 8 | 0.7165 / 0.7824 / 0.7243 / 0.7239 | 52/64 | 0.05725 / 0.06424 / 0.05942 / 0.05944 | 43/64 |
| 16 | 0.4231 / 0.4723 / 0.4268 / 0.4271 | 61/64 | 0.02092 / 0.02442 / 0.02130 / 0.02151 | 51/64 |
| 32 | 0.1952 / 0.2450 / 0.1977 / 0.1984 | 64/64 | 0.00447 / 0.00739 / 0.00468 / 0.00475 | 64/64 |

全轨迹 L2 上，CES-Oracle 仍未超过 Uniform 的平均误差。相对 CRB，CES-Oracle 平均 L2 在四个 q 下均较低，q=32 时 64/64 条轨迹都低于 CRB。和 pilot 曲率估计 CES 相比，Oracle 没有带来一致改善：q=8 的 L2 仅略低（0.723908 对 0.724336），q=4、16、32 均略高；方向余弦误差平均值在四种预算下均略高于估计版 CES。差别整体很小。

块内相对向量方差与方向方差也没有显示真实曲率 oracle 的普遍优势：q=4 时 Oracle 的等权平均低于其余三种方法；q=8–32 时 CRB 的平均值最低，Oracle 与 CES/Uniform 接近或略高。L2 插值、块内同质性仍是不同目标。

解释边界：这是一个已知函数族和单组 64 条轨迹上的曲率 oracle 方案比较，不代表真实梯度轨迹上可获得精确 g''，也不构成全局最优锚点的下界/上界。有限 q、固定 pilot 锚点以及曲率密度近似都会影响结果。CES-Oracle 锚点数、端点与四方法结果均已验证；运行输出位于 outputs/train/v0-crb-mech-test/ces-oracle/，块指标位于 outputs/train/v0-crb-mech-test/block_variance/ces-oracle-comparison/。入口 scripts/v0_crb_mech_ces_oracle.py，配置 configs/rage/v0-crb-mech-test-ces-oracle.json。

## Uniform 锚点的不同重建器

固定原有均匀锚点位置和 q，在同一 64 条轨迹上替换重建器；CRB、CES 与 CES-Oracle 线性重建作为参照。指标为 64 条轨迹等权平均。Natural cubic 使用自然边界条件；PCHIP 逐坐标执行；Chebyshev 使用 q−1 次全局多项式拟合等距锚点；GP 是固定 RBF 核的多输出后验均值，长度尺度为 1 或 2 个平均锚点间距、alpha=1e-6；平滑样条 lambda 在 {1e-8,1e-6,1e-4} 上做敏感性扫描，不据评估轨迹选择超参数，q=4 因 SciPy 最少 5 个锚点限制而跳过。

| q | 重建器 | 平均向量 L2 | 逐坐标 RMSE | 相对 L2 | 方向余弦误差 |
|---:|---|---:|---:|---:|---:|
| 4 | uniform_linear | 1.165281 | 0.019436 | 0.476778 | 0.151011 |
| 4 | crb_linear | 1.227648 | 0.020804 | 0.501030 | 0.176501 |
| 4 | ces_linear | 1.225762 | 0.020779 | 0.498712 | 0.174776 |
| 4 | ces_oracle_linear | 1.246588 | 0.021172 | 0.508515 | 0.183946 |
| 4 | uniform_natural_cubic | 1.199269 | 0.020092 | 0.491931 | 0.155896 |
| 4 | uniform_pchip | 1.212170 | 0.020264 | 0.497356 | 0.160052 |
| 4 | uniform_chebyshev | 1.288363 | 0.021716 | 0.528964 | 0.171587 |
| 4 | uniform_gp_rbf_lenx1.0 | 1.208312 | 0.020294 | 0.495783 | 0.155790 |
| 4 | uniform_gp_rbf_lenx2.0 | 1.252520 | 0.021075 | 0.514093 | 0.165256 |
| 8 | uniform_linear | 0.716502 | 0.012178 | 0.293786 | 0.057248 |
| 8 | crb_linear | 0.782398 | 0.013651 | 0.311126 | 0.064237 |
| 8 | ces_linear | 0.724336 | 0.012387 | 0.296949 | 0.059415 |
| 8 | ces_oracle_linear | 0.723908 | 0.012379 | 0.297155 | 0.059442 |
| 8 | uniform_natural_cubic | 0.736708 | 0.012603 | 0.302318 | 0.060159 |
| 8 | uniform_pchip | 0.731285 | 0.012464 | 0.300169 | 0.059489 |
| 8 | uniform_chebyshev | 1.142317 | 0.021820 | 0.469678 | 0.118641 |
| 8 | uniform_gp_rbf_lenx1.0 | 0.760500 | 0.013053 | 0.312045 | 0.063415 |
| 8 | uniform_gp_rbf_lenx2.0 | 0.863845 | 0.015090 | 0.354652 | 0.079600 |
| 8 | uniform_smoothing_lam_1e-08 | 0.736701 | 0.012603 | 0.302315 | 0.060158 |
| 8 | uniform_smoothing_lam_1e-06 | 0.736045 | 0.012591 | 0.302046 | 0.060077 |
| 8 | uniform_smoothing_lam_1e-04 | 0.761939 | 0.012596 | 0.312527 | 0.061025 |
| 16 | uniform_linear | 0.423123 | 0.007443 | 0.173206 | 0.020920 |
| 16 | crb_linear | 0.472314 | 0.008580 | 0.186397 | 0.024420 |
| 16 | ces_linear | 0.426829 | 0.007520 | 0.174572 | 0.021304 |
| 16 | ces_oracle_linear | 0.427119 | 0.007540 | 0.174890 | 0.021506 |
| 16 | uniform_natural_cubic | 0.428861 | 0.007645 | 0.175436 | 0.021900 |
| 16 | uniform_pchip | 0.426218 | 0.007535 | 0.174557 | 0.021453 |
| 16 | uniform_chebyshev | 8.909869 | 0.388073 | 3.582112 | 0.229292 |
| 16 | uniform_gp_rbf_lenx1.0 | 0.448267 | 0.008015 | 0.183241 | 0.023703 |
| 16 | uniform_gp_rbf_lenx2.0 | 0.507535 | 0.009308 | 0.207219 | 0.030446 |
| 16 | uniform_smoothing_lam_1e-08 | 0.428817 | 0.007644 | 0.175419 | 0.021896 |
| 16 | uniform_smoothing_lam_1e-06 | 0.427753 | 0.007585 | 0.175014 | 0.021621 |
| 16 | uniform_smoothing_lam_1e-04 | 0.588126 | 0.009442 | 0.240823 | 0.033333 |
| 32 | uniform_linear | 0.195215 | 0.003474 | 0.079764 | 0.004469 |
| 32 | crb_linear | 0.245018 | 0.004842 | 0.095003 | 0.007385 |
| 32 | ces_linear | 0.197708 | 0.003547 | 0.080806 | 0.004676 |
| 32 | ces_oracle_linear | 0.198365 | 0.003563 | 0.081168 | 0.004746 |
| 32 | uniform_natural_cubic | 0.182307 | 0.003390 | 0.074461 | 0.004401 |
| 32 | uniform_pchip | 0.187545 | 0.003375 | 0.076624 | 0.004282 |
| 32 | uniform_chebyshev | 39932.918971 | 2902.098677 | 16060.234156 | 0.375810 |
| 32 | uniform_gp_rbf_lenx1.0 | 0.191838 | 0.003641 | 0.078456 | 0.005112 |
| 32 | uniform_gp_rbf_lenx2.0 | 0.226679 | 0.004349 | 0.092626 | 0.007055 |
| 32 | uniform_smoothing_lam_1e-08 | 0.182239 | 0.003386 | 0.074432 | 0.004389 |
| 32 | uniform_smoothing_lam_1e-06 | 0.221370 | 0.003694 | 0.090489 | 0.005119 |
| 32 | uniform_smoothing_lam_1e-04 | 0.498234 | 0.007938 | 0.203945 | 0.023220 |

在 q=4、8、16 下，线性 Uniform 的平均 L2 最低。q=32 下自然三次和 lambda=1e-8 平滑样条的 L2 略低于线性 Uniform，PCHIP 的方向余弦误差最低；低 lambda 平滑样条与自然三次很接近。由于 lambda 是敏感性扫描而未用独立验证集选定，该 q=32 结果是本组对比中的观察，不是独立调参后确认的提升。Chebyshev 在 q=16、32 出现严重误差放大，反映等距节点下高阶全局多项式对当前含多频分量轨迹不稳。两个固定 GP 长度尺度均未优于线性 Uniform；长度尺度选择明显影响结果。

所有方法的逐轨迹记录及各指标均值在 outputs/train/v0-crb-mech-test/uniform-reconstructors/trajectory_metrics.csv 和 summary.json，.complete 已核实。方法使用 SciPy 1.15.3；SmoothingSpline 的多输出算子由固定 lambda 下单位响应组装，再统一作用于 4096 个坐标。


## Uniform + Cancellation Split：real-real oracle 方差诊断（不代表估计能力）

用户要求将 midpoint 与 minimum-norm 两种 cancellation 虚拟分割都落地，并重点比较块级 L2 方差和方向方差。该方法保留原 Uniform 真实锚点，并沿用同一分段线性重建，因此逐点线性重建及其 L2/方向误差与 Uniform+Linear 完全相同。新增分析只改变分块边界，不增加真实梯度。

### 指标与比较口径

对每块内全部原始 mock 梯度，以块内向量均值作为描述性代表，计算 population vector L2 variance、relative vector variance、direction variance 和 norm variance，定义沿用 scripts/v0_crb_mech_block_variance.py。报告等块权重均值，也报告按块所覆盖轨迹点数加权均值；按每条轨迹分别与未分割 Uniform 区间配对，再对 64 条轨迹等权汇总。块均值使用了块内完整轨迹值，只是块代表性的分析 oracle，不能解释为实际在线可免费获得的梯度均值。

试验 64 条原 mock 轨迹、q={4,8,16,32}、tau={0,0.2,0.4,0.6,0.8,0.9,1.0}。若 cancellation kappa<tau，每个父区间增加一个虚拟边界。两种切分规则分别为 midpoint 和按线性插值向量的最小模长位置切分；minimum-norm 位置按父区间两侧至少 10% 长度、至少 2 点约束裁剪。tau=0 是无切分的等价 Uniform 基线。

### tau=0.9 的方差变化

以下均为相对未切分 Uniform 的下降百分比；数字越大表示块内方差越低。每格按“vector L2 variance / direction variance”排列。

| q | 平均拆分数/轨迹 | Midpoint 等块权 | Minimum-norm 等块权 | Midpoint 按点加权 | Minimum-norm 按点加权 | L2/方向均降低的轨迹 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.45 | 47.48% / 48.90% | 47.59% / 49.83% | 45.82% / 49.58% | 40.98% / 47.41% | 63/64 |
| 8 | 2.27 | 27.63% / 31.58% | 27.71% / 32.17% | 21.76% / 30.87% | 19.40% / 28.59% | 56/64 |
| 16 | 0.66 | 5.22% / 8.94% | 5.11% / 8.93% | 3.45% / 9.07% | 3.08% / 8.41% | 21/64 |
| 32 | 0.09 | 0.52% / 1.23% | 0.50% / 1.23% | 0.30% / 1.23% | 0.26% / 1.09% | 5/64 |

在 tau=0.8 时，q=4 平均切分 1.36 个区间/轨迹，L2/方向方差下降约 32.8%/35.7%（midpoint）和 32.8%/36.6%（minimum-norm）；q=8 平均切分 0.33 个区间，下降约 4.7%/7.8% 与 4.7%/7.9%；q=16 平均切分 0.03 个区间，下降不到 1%；q=32 没有切分，因而没有变化。tau=0.6 以下几乎不触发切分。tau=1 会切分每个区间，分块数正好翻倍，方差下降很大；这主要反映所有块被强制细分，不能当作 cancellation 选择的实际收益。

### 结论和边界

Cancellation 分割确实能降低块内代表性方差，收益集中在 q=4/8 且阈值较高、触发足够多切分时；q=16 收益较弱，q=32 的 cancellation 通常接近 1，tau≤0.9 几乎不触发。因此该 64 条 mock 结果支持“块划得更贴合变化时，块内方向与向量更同质”这一块级结论，但不能推导为重建误差改善，因虚拟点不增加线性重建自由度。

Minimum-norm 相较 midpoint 只有很小的差异：等块权在 q=4/8 略优，按点加权时 midpoint 的下降反而较大；q=16/32 两者接近。没有证据支持 minimum-norm 在当前轨迹族上稳定占优。由于拆分增加了块数，等块权指标会受块粒度影响；按覆盖点加权的结果通常更保守，故两种汇总均保留。

此前的 cross-fit RAGE linearSum shape score 幅度约为 1e-5 量级，随 tau 的变化小且没有一致上升；它仅作辅助信号，不取代上述块方差主指标。

### 产物

入口 scripts/v0_crb_mech_cancellation_split.py 与 scripts/v0_crb_mech_cancellation_variance.py；配置 configs/rage/v0-crb-mech-cancellation-split.json。阈值与 shape-score 明细在 outputs/train/v0-crb-mech-test/cancellation-split/；逐轨迹方差、逐块方差及 summary 在其 variance/ 子目录。两个输出均有 .complete。日志在 logs/v0-crb-mech-test/。


## Uniform + Linear：六方法 real-real oracle 诊断与匹配随机分割

按用户澄清，此批只评估原 Uniform 锚点和原分段线性重建。所有虚拟边界只用于定义块，不新增真实锚点、不改变任何重建值。因此逐点重建 L2 和方向误差与 Uniform+Linear 完全相同；本批主要比较块内代表性方差。

六个方法如下：tau-mid=对所有 kappa<=tau 区间中点切分；tau-rev=同一触发区间按最小模长位置切分；tau-mid-base=随机抽取同一个 K 数目的父区间并中点切分；tau-rev-base=与 tau-mid-base 共用随机区间，分别在每区间的 minimum-norm reversal 点切分；all-mid/all-rev=不看 kappa，对所有 Uniform 区间分别中点/反转点切分。K 按每条轨迹、q、tau 分别精确匹配。随机区间无放回，seed 为 20261111+918273，并按轨迹/q/tau派生；每个轨迹组合使用一个固定随机样本。trigger 使用 kappa<=tau。tau 网格仍为 {0,0.2,0.4,0.6,0.8,0.9,1.0}。

### tau 选择的区间是否优于匹配随机区间

表内给出 cancellation 选区方法相对同位置规则的随机 K 对照，方差下降百分比；正数表示 cancellation 选区的平均方差更低。每格顺序是“vector L2 variance / direction variance”；括号中为按点数加权，未括号为等块权。Track win 为等块权下 cancellation 方法低于随机基线的轨迹数，按 L2/方向列出。

| q | tau | 平均 K | tau-mid 相对 tau-mid-base | tau-mid track win | tau-rev 相对 tau-rev-base | tau-rev track win |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.8 | 1.36 | 2.08% / 2.72%（3.66% / 10.46%） | 21/22 | 1.93% / 3.20%（4.85% / 11.31%） | 24/24 |
| 4 | 0.9 | 2.45 | 1.31% / 0.92%（2.95% / 6.56%） | 13/10 | 0.61% / 0.54%（3.23% / 6.47%） | 10/12 |
| 8 | 0.8 | 0.33 | 0.40% / 1.26%（0.63% / 4.47%） | 10/10 | 0.47% / 1.49%（0.87% / 4.73%） | 10/12 |
| 8 | 0.9 | 2.27 | 2.57% / 4.28%（4.16% / 13.40%） | 42/43 | 2.21% / 4.20%（4.79% / 12.85%） | 39/43 |
| 16 | 0.8 | 0.03 | 0.07% / 0.29%（0.10% / 0.62%） | 2/2 | 0.05% / 0.34%（0.11% / 0.67%） | 2/2 |
| 16 | 0.9 | 0.66 | 0.33% / 2.08%（0.59% / 5.04%） | 13/20 | 0.39% / 2.16%（0.86% / 5.07%） | 15/20 |
| 32 | 0.8 | 0.00 | 0.00% / 0.00%（0.00% / 0.00%） | 0/0 | 0.00% / 0.00%（0.00% / 0.00%） | 0/0 |
| 32 | 0.9 | 0.09 | 0.08% / 0.58%（0.10% / 1.02%） | 4/5 | 0.12% / 0.64%（0.15% / 0.97%） | 5/5 |

在更低阈值 tau=0.6 时，q=4 平均 K=0.17、q=8 K=0.02，q=16/32 为 0；tau=0.4 时所有 q 的平均 K 均为 0。原因是这组 mock 的相邻 Uniform 端点 cancellation 值通常偏高，严格低 tau 几乎不触发。q=8、tau=0.9 是主要的正向信号：midpoint 和 reversal 选区都优于各自匹配随机 K 基线，等块权方向方差改善约 4.2%，L2 方差约 2.2–2.6%，多数轨迹方向方差更低。q=4 的平均优势较小且等块权逐轨迹胜出数低，说明均值收益不稳定；q=16/32 也接近零。

minimum-norm reversal 未显示稳定优于 midpoint；两者对照各自的 matched random 后差异很小。随机基线每个轨迹/q/tau 使用一次可复现的随机无放回抽样，64 条轨迹形成配对重复；没有在单条轨迹内对多个随机种子做 Monte Carlo 平均，故接近零的差异不应作强结论。

### 无条件全区间切分

all-mid 与 all-rev 都把每个 Uniform 父区间拆为两块，块数翻倍。相对未切分 Uniform，等块权的 L2/方向方差下降约：q=4 midpoint 51.5%/52.7%、reversal 52.3%/53.9%；q=8 midpoint 54.3%/54.9%、reversal 55.0%/56.2%；q=16 midpoint 61.4%/61.7%、reversal 59.8%/60.8%；q=32 midpoint 70.8%/70.9%、reversal 65.7%/66.5%。按点加权时 midpoint 结果保持一致，reversal 降幅较小（约 44.5%–51.6% L2、48.0%–53.7%方向）。这主要是每块长度减半后的粒度效应，不证明 cancellation 区间选择有效；选择是否有效应看前面的 matched-K random 对照。

### 结论

以本批 mock 轨迹为准，cancellation 选区对 Uniform+Linear 的块级代表性可能有小幅改善，证据主要来自 q=8、tau=0.9；其他预算/阈值大多较弱或几乎不触发。方向与 L2 改善均不改变逐点重建误差。两种切分位置接近，无法宣称 reversal 点稳定优于 midpoint。较低 tau 在这组轨迹上没有足够触发数，因此如果坚持近似完全抵消才切分，需要更长区间/更低 q 的场景或专门含反向段的轨迹验证。

### 产物

入口 scripts/v0_crb_mech_cancellation_six_methods.py；配置 configs/rage/v0-crb-mech-cancellation-six-methods.json。输出 outputs/train/v0-crb-mech-test/uniform-linear-six-methods/，含 trajectory_metrics.csv、interval_selection.csv、summary.json 与 .complete；日志 logs/v0-crb-mech-test/cancellation-six-methods.log_*。


## Uniform + Linear：六方法 est-real 评估

### 指标纠正

前一批 real-real 方差诊断中，分割仅使用端点锚点信息，但块内代表直接取真实轨迹值均值，再测真实值围绕该均值的方差。这是 oracle 方差；块越小，代表越接近真实值，因而不能回答有限锚点下如何估计代表值。前面的 real-real 结论保留作 oracle 诊断，不作为方法有效性证据。

当前 est-real 批次沿用六方法之前保存的分割选择：tau-mid、tau-rev、tau-mid-base、tau-rev-base、all-rev、all-mid。区间选择仅依赖锚点 κ、端点线性段最小模长位置、随机数和等切分数 K。每块估计代表值由原始 Uniform+Linear 在块内每个离散采样点的估计值取算术均值；真实 mock 轨迹只在评估时计算各点相对该估计代表的 vector L2 error、relative L2 error 和 direction cosine error。锚点和逐点重建始终不变。

入口 scripts/v0_crb_mech_cancellation_estimated_rep.py；配置 configs/rage/v0-crb-mech-cancellation-six-methods.json；输入分割 outputs/train/v0-crb-mech-test/uniform-linear-six-methods/interval_selection.csv；结果写入 outputs/train/v0-crb-mech-test/uniform-linear-six-methods/est-real/，状态待完成后补入数字。


### est-real 六方法结果

以下为 tau=0.9 的真实值相对 est block representative 的绝对 vector L2 error / direction error，相对未分割 Uniform 的下降百分比。每格括号前是等块权，括号内是按覆盖点数加权。tau-mid/rev 的 matched base 保持完全相同的 K 和分割位置规则，只随机化所选父区间。

| q | tau-mid | tau-rev | tau-mid-base | tau-rev-base | all-rev | all-mid |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.77% / 23.82% (22.84% / 25.83%) | 26.65% / 26.07% (20.56% / 25.63%) | 23.19% / 25.04% (21.79% / 23.85%) | 27.01% / 27.59% (19.51% / 23.51%) | 30.04% / 30.20% (22.38% / 26.70%) | 25.20% / 27.01% (25.20% / 27.01%) |
| 8 | 15.22% / 13.90% (11.68% / 17.08%) | 16.76% / 16.04% (10.36% / 15.97%) | 13.32% / 15.77% (9.48% / 10.74%) | 16.37% / 18.42% (8.01% / 9.81%) | 34.38% / 34.53% (22.94% / 25.72%) | 27.88% / 28.94% (27.89% / 28.94%) |
| 16 | 2.90% / 2.64% (1.86% / 5.05%) | 3.18% / 3.55% (1.69% / 4.79%) | 2.61% / 3.60% (1.45% / 2.17%) | 3.11% / 4.40% (1.15% / 1.84%) | 40.19% / 40.36% (24.76% / 26.76%) | 32.33% / 32.83% (32.33% / 32.83%) |
| 32 | 0.36% / 0.55% (0.21% / 0.87%) | 0.39% / 0.73% (0.18% / 0.75%) | 0.28% / 0.49% (0.14% / 0.15%) | 0.28% / 0.49% (0.08% / 0.08%) | 51.58% / 52.24% (34.88% / 36.81%) | 49.03% / 49.64% (49.03% / 49.63%) |

结果依赖汇总权重。对绝对 true L2 error，等块权下选区方法相对随机基线的均值只有小幅优势/劣势，方向误差在 q=4/8/16 多数更偏向随机基线；按覆盖点数加权时，tau-mid/rev 四个 q 的 L2 与方向误差均低于匹配随机基线。tau=0.9 的按点加权相对随机误差下降（绝对 true L2 / direction）为：q=4 midpoint 1.35%/2.59%、reversal 1.31%/2.77%；q=8 midpoint 2.43%/7.11%、reversal 2.55%/6.83%；q=16 midpoint 0.42%/2.95%、reversal 0.55%/3.00%；q=32 midpoint 0.07%/0.72%、reversal 0.10%/0.67%。q=8 信号最明显。具体逐轨迹配对胜负数保存在 trajectory_metrics.csv。等块权与点权会给出不同判断，因此结论须注明所用权重。

All-mid/all-rev 相对未分割块误差有更大降幅，但将每个 Uniform 区间都拆开；它们说明缩短代表区间能降低 est-real 块代表误差，不证明 cancellation 选区优于随机区间。该组指标不是逐点 Uniform 重建误差，逐点重建曲线在所有方法中完全相同。

产物：scripts/v0_crb_mech_cancellation_estimated_rep.py；输出目录 outputs/train/v0-crb-mech-test/uniform-linear-six-methods/est-real/，含 trajectory_metrics.csv、block_metrics.csv、summary.json 和 .complete。总计 7,680 条轨迹级、130,912 条块级记录。



## Uniform Gradient Reconstruction：虚拟分块选区策略 K 曲线

最后修改日期：2026-11-11。状态：完成。

### 设计与计算口径

复用前批相同的 64 条确定性 mock 轨迹（4096 维、4000 点），从已有基线指标读入 Uniform 锚点；固定原 Linear 重建与真实锚点预算 q={4,8,16,32}。每个原区间最多 midpoint split 一次，K=0,...,q-1。每个块的代表是原 Uniform 线性重建在该块离散点上的向量均值。真实轨迹只用于计算 est-real 误差及 Oracle/区间收益；Random、Cancellation、Endpoint-Difference、Predicted-Gain 的选区只用随机数或锚点端点信息。没有计算新的真实梯度。

L2 主指标为逐点平方欧氏误差（块代表与真实向量之差的平方范数），方向误差为逐点 1-cos；主汇总按覆盖点数加权。等块权曲线作为辅助，分别以等块权收益构造其 L2/Direction Oracle；点权 Oracle 则按对应点权真实区间收益排序。两种 Oracle 都固定 K，若单区间收益为负仍按收益排序。方法间置信区间以 64 条轨迹为配对单位，5,000 次 paired bootstrap；Random 每条轨迹/q 使用 100 个无放回随机排列，每个 K 取其前缀。

### K=1 点权结果

表中误差均值的区间为轨迹 bootstrap 95% CI；ΔRandom 为方法误差减 Random-K 误差，负值表示方法更好，其 95% CI 为轨迹配对 bootstrap。η 是相对 K=0 的误差下降除以匹配指标 Oracle 的下降。

#### L2

| q | 方法 | est-real 误差 [95% CI] | 相对 K=0 下降 | ΔRandom [paired 95% CI] | η |
|---:|---|---:|---:|---:|---:|
| 4 | Random-K | 2.160504 [2.053282, 2.272425] | 8.356% | 0.000000 [0.000000, 0.000000] | 0.746 |
| 4 | Cancellation-K | 2.112498 [2.008765, 2.221731] | 10.392% | -0.048005 [-0.059508, -0.036237] | 0.928 |
| 4 | Endpoint-Difference-K | 2.100378 [1.995824, 2.209828] | 10.906% | -0.060126 [-0.070618, -0.049538] | 0.974 |
| 4 | Predicted-Gain-K | 2.100424 [1.995881, 2.209965] | 10.904% | -0.060079 [-0.070616, -0.049444] | 0.973 |
| 4 | L2-Oracle-K | 2.093422 [1.989723, 2.201936] | 11.201% | -0.067081 [-0.075857, -0.058568] | 1.000 |
| 8 | Random-K | 0.943027 [0.891087, 0.994875] | 3.971% | 0.000000 [0.000000, 0.000000] | 0.656 |
| 8 | Cancellation-K | 0.930901 [0.879145, 0.982750] | 5.206% | -0.012126 [-0.014950, -0.009229] | 0.860 |
| 8 | Endpoint-Difference-K | 0.925161 [0.874039, 0.976503] | 5.791% | -0.017867 [-0.020089, -0.015714] | 0.957 |
| 8 | Predicted-Gain-K | 0.925161 [0.874039, 0.976503] | 5.791% | -0.017867 [-0.020089, -0.015714] | 0.957 |
| 8 | L2-Oracle-K | 0.922592 [0.871090, 0.973493] | 6.052% | -0.020436 [-0.022223, -0.018714] | 1.000 |
| 16 | Random-K | 0.397578 [0.372966, 0.422855] | 2.140% | 0.000000 [0.000000, 0.000000] | 0.634 |
| 16 | Cancellation-K | 0.394733 [0.370165, 0.419931] | 2.841% | -0.002845 [-0.003385, -0.002277] | 0.842 |
| 16 | Endpoint-Difference-K | 0.393069 [0.368561, 0.418179] | 3.250% | -0.004509 [-0.004967, -0.004034] | 0.963 |
| 16 | Predicted-Gain-K | 0.393069 [0.368561, 0.418179] | 3.250% | -0.004509 [-0.004967, -0.004034] | 0.963 |
| 16 | L2-Oracle-K | 0.392560 [0.368046, 0.417598] | 3.376% | -0.005019 [-0.005446, -0.004600] | 1.000 |
| 32 | Random-K | 0.135936 [0.126120, 0.146017] | 1.586% | 0.000000 [0.000000, 0.000000] | 0.633 |
| 32 | Cancellation-K | 0.134987 [0.125262, 0.144999] | 2.273% | -0.000949 [-0.001080, -0.000819] | 0.907 |
| 32 | Endpoint-Difference-K | 0.134677 [0.124954, 0.144683] | 2.497% | -0.001259 [-0.001368, -0.001150] | 0.996 |
| 32 | Predicted-Gain-K | 0.134676 [0.124948, 0.144682] | 2.498% | -0.001260 [-0.001369, -0.001153] | 0.997 |
| 32 | L2-Oracle-K | 0.134665 [0.124939, 0.144676] | 2.506% | -0.001271 [-0.001378, -0.001164] | 1.000 |

#### Direction

| q | 方法 | est-real 误差 [95% CI] | 相对 K=0 下降 | ΔRandom [paired 95% CI] | η |
|---:|---|---:|---:|---:|---:|
| 4 | Random-K | 0.207812 [0.191120, 0.224844] | 8.991% | 0.000000 [0.000000, 0.000000] | 0.609 |
| 4 | Cancellation-K | 0.195379 [0.180155, 0.210702] | 14.436% | -0.012433 [-0.015546, -0.009747] | 0.978 |
| 4 | Endpoint-Difference-K | 0.196489 [0.181292, 0.211923] | 13.950% | -0.011323 [-0.014514, -0.008509] | 0.945 |
| 4 | Predicted-Gain-K | 0.196497 [0.181292, 0.211939] | 13.946% | -0.011315 [-0.014509, -0.008492] | 0.944 |
| 4 | Direction-Oracle-K | 0.194625 [0.179421, 0.209999] | 14.766% | -0.013187 [-0.016160, -0.010666] | 1.000 |
| 8 | Random-K | 0.087669 [0.079501, 0.096511] | 4.132% | 0.000000 [0.000000, 0.000000] | 0.469 |
| 8 | Cancellation-K | 0.083554 [0.076075, 0.091670] | 8.632% | -0.004115 [-0.005136, -0.003206] | 0.980 |
| 8 | Endpoint-Difference-K | 0.085077 [0.077240, 0.093603] | 6.967% | -0.002592 [-0.003479, -0.001816] | 0.791 |
| 8 | Predicted-Gain-K | 0.085077 [0.077240, 0.093603] | 6.967% | -0.002592 [-0.003479, -0.001816] | 0.791 |
| 8 | Direction-Oracle-K | 0.083396 [0.075875, 0.091517] | 8.804% | -0.004273 [-0.005267, -0.003391] | 1.000 |
| 16 | Random-K | 0.035782 [0.032487, 0.039350] | 2.171% | 0.000000 [0.000000, 0.000000] | 0.413 |
| 16 | Cancellation-K | 0.034675 [0.031498, 0.038092] | 5.197% | -0.001107 [-0.001354, -0.000887] | 0.990 |
| 16 | Endpoint-Difference-K | 0.035110 [0.031892, 0.038614] | 4.009% | -0.000672 [-0.000904, -0.000470] | 0.763 |
| 16 | Predicted-Gain-K | 0.035110 [0.031892, 0.038614] | 4.009% | -0.000672 [-0.000904, -0.000470] | 0.763 |
| 16 | Direction-Oracle-K | 0.034655 [0.031483, 0.038073] | 5.252% | -0.001127 [-0.001371, -0.000909] | 1.000 |
| 32 | Random-K | 0.012025 [0.010877, 0.013271] | 1.614% | 0.000000 [0.000000, 0.000000] | 0.393 |
| 32 | Cancellation-K | 0.011723 [0.010609, 0.012928] | 4.088% | -0.000302 [-0.000351, -0.000256] | 0.996 |
| 32 | Endpoint-Difference-K | 0.011859 [0.010724, 0.013094] | 2.972% | -0.000166 [-0.000201, -0.000133] | 0.724 |
| 32 | Predicted-Gain-K | 0.011860 [0.010725, 0.013094] | 2.967% | -0.000165 [-0.000200, -0.000132] | 0.723 |
| 32 | Direction-Oracle-K | 0.011721 [0.010607, 0.012928] | 4.104% | -0.000304 [-0.000354, -0.000258] | 1.000 |

### K=q-1 端点与切分数收敛

当所有 q-1 个原区间都各切一次后，所有选区顺序必然得到同一分区，因此五条方法曲线与对应 Oracle 重合。误差及相对 K=0 的下降如下：

| q | K | L2 est-real MSE | L2 下降 | Direction est-real 误差 | Direction 下降 |
|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 1.763318 [1.673645, 1.855823] | 25.204% | 0.166677 [0.152944, 0.180339] | 27.006% |
| 8 | 7 | 0.708189 [0.665925, 0.750616] | 27.885% | 0.064984 [0.058746, 0.071860] | 28.939% |
| 16 | 15 | 0.274940 [0.255846, 0.294677] | 32.326% | 0.024569 [0.022156, 0.027221] | 32.827% |
| 32 | 31 | 0.070410 [0.064434, 0.076604] | 49.025% | 0.006157 [0.005514, 0.006844] | 49.629% |

### 等块权 K=1：相对 Random-K 的误差差值

数值为 (方法误差−Random误差)/Random误差 × 100%；负数表示更好。该诊断的配对 CI 与完整曲线见 curve_summary.csv。

| q | 指标 | Cancellation | Endpoint-Difference | Predicted-Gain | 指标 Oracle |
|---:|---|---:|---:|---:|---:|
| 4 | L2 | -2.15% | -2.61% | -2.64% | -4.59% |
| 4 | Direction | +2.19% | +0.66% | +0.68% | -5.33% |
| 8 | L2 | -1.27% | -2.63% | -2.63% | -3.93% |
| 8 | Direction | +1.81% | -1.73% | -1.73% | -4.02% |
| 16 | L2 | -0.95% | -1.55% | -1.55% | -2.41% |
| 16 | Direction | +0.52% | -1.32% | -1.32% | -2.49% |
| 32 | L2 | -0.96% | -1.30% | -1.30% | -1.48% |
| 32 | Direction | -1.44% | -1.50% | -1.50% | -2.04% |

### Selection Efficiency：K=1 点权

| q | L2 Cancellation | L2 Endpoint | L2 Predicted | Direction Cancellation | Direction Endpoint | Direction Predicted |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.928 | 0.974 | 0.973 | 0.978 | 0.945 | 0.944 |
| 8 | 0.860 | 0.957 | 0.957 | 0.980 | 0.791 | 0.791 |
| 16 | 0.842 | 0.963 | 0.963 | 0.990 | 0.763 | 0.763 |
| 32 | 0.907 | 0.996 | 0.997 | 0.996 | 0.724 | 0.723 |

K=0 时 Oracle 收益为零，η 不定义；分母接近零的条目在 CSV 中标记。所有 K 与两类加权的完整效率均在 selection_efficiency.csv。

### 区间预测能力：Spearman ρ

得分分别为 cancellation score −κ、端点差平方范数、精确离散 Predicted-Gain；收益为单区间真实 est-real 收益。跨 q pooled 与 q 分层结果如下。

| 范围 | 预测分数 | 真实 L2 收益 | 真实 Direction 收益 | 区间数 |
|---|---|---:|---:|---:|
| q4 | cancellation_score | 0.646 | 0.921 | 192 |
| q4 | endpoint_difference_score | 0.873 | 0.689 | 192 |
| q4 | predicted_gain_exact | 0.873 | 0.689 | 192 |
| q8 | cancellation_score | 0.606 | 0.944 | 448 |
| q8 | endpoint_difference_score | 0.895 | 0.657 | 448 |
| q8 | predicted_gain_exact | 0.895 | 0.657 | 448 |
| q16 | cancellation_score | 0.595 | 0.950 | 960 |
| q16 | endpoint_difference_score | 0.917 | 0.663 | 960 |
| q16 | predicted_gain_exact | 0.917 | 0.663 | 960 |
| q32 | cancellation_score | 0.621 | 0.989 | 1984 |
| q32 | endpoint_difference_score | 0.979 | 0.669 | 1984 |
| q32 | predicted_gain_exact | 0.979 | 0.669 | 1984 |
| pooled | cancellation_score | 0.817 | 0.959 | 3584 |
| pooled | endpoint_difference_score | 0.972 | 0.888 | 3584 |
| pooled | predicted_gain_exact | 0.985 | 0.897 | 3584 |

### Random 不确定性、图与数据文件

Random 汇总的随机重复标准差与 2.5–97.5 百分位数保存在 curve_summary.csv；其误差均值的轨迹 bootstrap CI 也单独列出。64 条轨迹的逐方法逐 K 记录提供所有 paired CI。8 张主图：

| q | L2 曲线 | Direction 曲线 |
|---:|---|---|
| 4 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q4_l2_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q4_direction_k_error.png) |
| 8 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q8_l2_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q8_direction_k_error.png) |
| 16 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q16_l2_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q16_direction_k_error.png) |
| 32 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q32_l2_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/plots/q32_direction_k_error.png) |

完整交付：curve_summary.csv（两类权重下每 q/K/方法的均值、CI、相对 K0 降幅、paired Random 差值与区间、η、Random 重复分布）；selection_efficiency.csv；interval_prediction.csv（包括真实点权与等块权区间收益）；spearman_correlations.csv；trajectory_curves.csv；random_curves.npz；summary.json；.complete。

### 结果与问题回答

**Q1：Cancellation 是否优于 Random？** 在最省切分的 K=1 上，Cancellation 的点权 L2 与方向误差在四个 q 均低于 Random；相对 Random 的 paired 95% CI 全为负。它能用端点抵消识别有价值的区间，方向收益排序尤其明显（pooled Spearman ρ=0.959）。但它对 L2 收益排序较弱（ρ=0.817），收益有但未充分抓到最佳 L2 区间。

**Q2：Endpoint-Difference / Predicted-Gain 是否更有效？** 点权 L2 上两者在四个 q 都优于 Cancellation；它们 pooled 对真实 L2 收益的 Spearman ρ 分别为 0.972 与 0.985。q=8/16 二者排序相同；总体预测收益分数为更好的 L2 选区规则。方向指标上 Cancellation 更强：pooled ρ=0.959，而 Endpoint/Predicted 分别为 0.888/0.897；K=1 时 Cancellation 在 q=8/16/32 优于 endpoint/predicted。

**Q3：部署方法离 Oracle 有多远？** K=1 点权 L2 最佳部署策略的 η 为 q=4/8/16/32 下 0.974/0.957/0.963/0.997；Direction 最佳部署策略 Cancellation 的 η 为 0.978/0.980/0.990/0.996。即相对各自 Oracle 的可实现收益差距约 0.3–4.3%（L2）与 0.4–2.2%（Direction）。K 增至 q−1 时，所有方法切完整个区间集，曲线重合、η=1。

按点权，K=1 的平均误差下降范围为 L2：Random 1.6–8.4%、Cancellation 2.3–10.4%、Endpoint/Predicted 2.5–10.9%；Direction：Random 1.6–9.0%、Cancellation 4.1–14.4%。q=4 时单次切分的绝对误差收益最大；高 q 下绝对误差更小，但简单端点规则仍能接近 Oracle。

### 结论边界

结论针对这 64 条确定性合成轨迹和块代表误差。虚拟切分不改变原始锚点，也不改变 Uniform+Linear 的逐点重建，因此不构成逐点重建误差的改善。跨 q Spearman 将相同轨迹在不同 q 下的区间合并，仅作为总体排序诊断；q 分层值同时保留。Random Monte Carlo 的重复区间与 64 轨迹 bootstrap CI 表示不同来源的不确定性，不能互相替代。

产物目录：outputs/train/v0-crb-mech-test/virtual-split-selection/；入口 scripts/v0_crb_mech_virtual_split_selection.py；配置 configs/rage/v0-crb-mech-virtual-split-selection.json；日志位于 logs/v0-crb-mech-test/。第一遍曾误用点权 Oracle 排序生成等块权曲线，已完整保留在 virtual-split-selection-pass1-pre-equal-oracle-fix/ 作审计，不纳入此处结论。



## est-real MSE 分解补充：Representation 与 Estimation

最后修改日期：2026-11-11。状态：完成。

### 定义与验证

对每条轨迹、每种 q、每个分块方法和 K，对块 I 令真实均值为 μ_I，原 Uniform+Linear 重建在该块内的均值为 μ̂_I。计算：

- Representation MSE：块内真实梯度围绕真实块均值的平方误差。

- Estimation MSE：真实块均值与估计块均值的距离，按块大小加权。

因为 Σ_{t∈I}(g(t)−μ_I)=0，逐块恒有 Σ||g−μ̂_I||² = Σ||g−μ_I||² + |I|·||μ_I−μ̂_I||²。因此全局 Total MSE 精确等于 Representation MSE + Estimation MSE。μ_I 只用于离线评价/分解，没有用于任何非 Oracle 策略的选区。每块代表 μ̂_I 始终来自原 Uniform 线性重建。

按点加权值是 64 条×4000 点上汇总后的 MSE；轨迹等权值是先对每条轨迹按 4000 点归一化、再对 64 条轨迹取平均。由于每条轨迹长度相同，两者数值相同，输出中仍分别保留并核验。

最大逐轨迹身份残差 |Total−Representation−Estimation| = 8.99e-15；分块逐项与原误差计算函数交叉核对的最大差为 5.46e-12，汇总曲线与原 K-sweep 点权 L2 最大差为 4.44e-16。逐轨迹加和、原指标交叉核对均通过。

### 随块数变化的分解

下表为 Random-K 的基线 K=0 与全切分 K=q−1。K=0 和全切分端点上，所选区间次序不影响分区，故所有策略具有相同分解。单位为每点 MSE。

| q | K=0 块数 | K=0 Representation | K=0 Estimation | K=0 Total | K=q−1 块数 | 全切 Representation | 全切 Estimation | 全切 Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 3 | 1.55522 | 0.80227 | 2.35749 | 6 | 0.75360 | 1.00971 | 1.76332 |
| 8 | 7 | 0.64293 | 0.33910 | 0.98203 | 14 | 0.29359 | 0.41460 | 0.70819 |
| 16 | 15 | 0.27024 | 0.13603 | 0.40627 | 30 | 0.10423 | 0.17071 | 0.27494 |
| 32 | 31 | 0.09884 | 0.03929 | 0.13813 | 62 | 0.02888 | 0.04153 | 0.07041 |

全切分后 Representation MSE 相对基线降低约 51.5%、54.3%、61.4%、70.8%（q=4/8/16/32）；Estimation MSE 分别上升约 25.9%、22.3%、25.5%、5.7%。Representation 的下降超过 Estimation 的上升，因此 Total MSE 仍下降 25.2%、27.9%、32.3%、49.0%。

K=1 时全部策略的分量变化如下；正百分比代表相对 K=0 下降，负值代表上升。

| q | 方法 | Representation 下降 | Estimation 下降 | Total 下降 |
|---:|---|---:|---:|---:|
| 4 | Random-K | +17.14% | -8.67% | +8.36% |
| 4 | Cancellation-K | +20.29% | -8.80% | +10.39% |
| 4 | Endpoint-Difference-K | +20.89% | -8.44% | +10.91% |
| 4 | Predicted-Gain-K | +20.92% | -8.51% | +10.90% |
| 4 | Oracle-L2-K | +20.92% | -7.64% | +11.20% |
| 4 | Oracle-Direction-K | +20.32% | -7.93% | +10.71% |
| 8 | Random-K | +7.74% | -3.18% | +3.97% |
| 8 | Cancellation-K | +9.63% | -3.19% | +5.21% |
| 8 | Endpoint-Difference-K | +10.75% | -3.61% | +5.79% |
| 8 | Predicted-Gain-K | +10.75% | -3.61% | +5.79% |
| 8 | Oracle-L2-K | +10.87% | -3.09% | +6.05% |
| 8 | Oracle-Direction-K | +9.79% | -2.76% | +5.45% |
| 16 | Random-K | +4.06% | -1.68% | +2.14% |
| 16 | Cancellation-K | +5.26% | -1.96% | +2.84% |
| 16 | Endpoint-Difference-K | +5.83% | -1.88% | +3.25% |
| 16 | Predicted-Gain-K | +5.83% | -1.88% | +3.25% |
| 16 | Oracle-L2-K | +5.94% | -1.71% | +3.38% |
| 16 | Oracle-Direction-K | +5.31% | -1.80% | +2.93% |
| 32 | Random-K | +2.29% | -0.19% | +1.59% |
| 32 | Cancellation-K | +3.28% | -0.26% | +2.27% |
| 32 | Endpoint-Difference-K | +3.60% | -0.28% | +2.50% |
| 32 | Predicted-Gain-K | +3.60% | -0.28% | +2.50% |
| 32 | Oracle-L2-K | +3.62% | -0.31% | +2.51% |
| 32 | Oracle-Direction-K | +3.32% | -0.28% | +2.30% |

在本组轨迹中，所有方法 K=1 的 Representation MSE 都降低，Estimation MSE 都上升；切分缩小了真实轨迹的块内离散度，但更多、更短的块让线性重建块均值与真实块均值的偏差略增。Total MSE 的净下降由 Representation 改善超过 Estimation 恶化所致。完整 K 曲线显示该分解并非只在 K=1 成立，见逐方法逐轨迹数据。

### 三类 MSE 曲线

以下 12 张图分别展示 q={4,8,16,32} 的 Total、Representation、Estimation MSE；横轴是实际块数 q−1+K，阴影为 64 条轨迹的 bootstrap 95% CI。

| q | Total MSE | Representation MSE | Estimation MSE |
|---:|---|---|---|
| 4 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q4_mse_total_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q4_mse_representation_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q4_mse_estimation_k_error.png) |
| 8 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q8_mse_total_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q8_mse_representation_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q8_mse_estimation_k_error.png) |
| 16 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q16_mse_total_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q16_mse_representation_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q16_mse_estimation_k_error.png) |
| 32 | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q32_mse_total_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q32_mse_representation_k_error.png) | [PNG](../../outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/plots/q32_mse_estimation_k_error.png) |

### 汇总与产物

覆盖 64 条轨迹、q={4,8,16,32}、Random-K/Cancellation-K/Endpoint-Difference-K/Predicted-Gain-K/L2-Oracle-K/Direction-Oracle-K、所有 K=0,...,q−1。Random 每轨迹/q 重复 100 次；每条策略曲线在 64 条轨迹上作 bootstrap，报告 MSE、相对 K=0 变化、paired Random 差值与 95% CI。

输出目录：outputs/train/v0-crb-mech-test/virtual-split-selection/mse_decomposition/。其中 mse_decomposition_summary.csv 含两种汇总口径、完整 K 曲线、误差变化与置信区间；mse_decomposition_trajectory.csv 含逐轨迹分量与加和残差；mse_decomposition_intervals.csv 含每个原区间父/子块分解和真实收益；mse_decomposition_random_curves.npz 保留 100 次随机曲线；summary.json 与 .complete 保存配置、校验与完成状态。入口 scripts/v0_crb_mech_virtual_split_mse_decomposition.py。
