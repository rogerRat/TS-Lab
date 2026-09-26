# MAIN-v5 — 主实验

- 更新时间：2026-09-20T00:55:33+08:00
- 配置：`configs/rags/main.json`；运行：`/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918`
- LESS/RAGS 统一对象 Rank 温度 T=1/(2ln4)=0.360673760222；LP-Mirror 原生 T=.05，TSDS 原生分布不变。
- 九任务、2模型；矩阵 rand / orig / less-main / rags-block-linearSum / rags-block-zeroClipSum / rags-block-maxSum / rags-block-expSum / rags-traj-linearSum / rags-traj-zeroClipSum / rags-traj-maxSum / rags-traj-expSum / tsds / lp-mirror；seeds [0, 1, 2]。
- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，递归最小 kappa 分裂，优先最右，预算严格耗尽。
- 新意见聚合协议：LESS参考max；RAGS按Block/Traj及四聚合显式命名，不指定主方法。
- 原生逐窗口梯度沿用已有字段校验缓存；V5 训练与评测独立；V5轮间复用静态概率和orig，训练seed逐轮变化。
- V1/V2 内容完整归档，继续参考：[MAIN-v1](archive/MAIN-v1.md)、[MAIN-v2](archive/MAIN-v2.md)。

## 结果展示与执行顺序（强制约束）

1. 所有结果表（任务模型表、末尾跨任务汇总表及后续新增结果表）严格按相对 rand 改善从好到差降序排列，改善率为 100×(1−方法 MASE 均值/rand MASE 均值)，正值更好。rand 自身为 0%；orig 也按同一改善率排序。无结果或缺少 rand 基准的行统一置于末尾；改善率完全相同时按固定顺序稳定排列。不能用绝对 MASE、方法名或完成时间代替此排序。
2. 执行优先级为：种子 → 下游任务 → 该任务的全部方法 cell。先完成 seed0 的全部对应实验，再完成 seed1，最后 seed2；当前种子的全部模型、任务、方法完成训练与评测后才允许进入下一种子。轮内两模型并行；每个模型严格按配置的任务顺序，完成当前任务的全部方法 cell 后再推进下一任务，不跨任务轮流跑同一种方法。两模型各自按任务推进，种子整轮屏障覆盖两模型。已完成 cell 不重跑；orig 是只测一次的种子无关基线，已提前完成的基线保留，后续种子只引用。


完成 666/666 cells。

| 模型 | 已完成 cells | 当前状态 |
|---|---:|---|
| chronos-base | 333 | ////complete |
| timesfm-2.5 | 333 | ////complete |
<!-- v5-patrol:start -->
## 巡检记录与解读（每 30 分钟巡检更新）

- 巡检时间：2026-09-20T00:55:33+08:00（北京时间）；以实际 cells/state 为准。
- 总进度：666/666 cells；已授权并行推进。

### 结果（按任务分节，每节各模型一表；两列百分比分别为均值相对 orig 与 rand 的改善，正=更好；胜/n 为逐 seed 配对胜 rand）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-zeroClipSum | 3 | 1.323450 | 0.066163 | +23.39% | +4.26% | 2/3 |
| rags-block-linearSum | 3 | 1.327354 | 0.115937 | +23.16% | +3.98% | 2/3 |
| rags-traj-expSum | 3 | 1.357953 | 0.089838 | +21.39% | +1.76% | 3/3 |
| rags-traj-linearSum | 3 | 1.379673 | 0.066397 | +20.14% | +0.19% | 2/3 |
| rand | 3 | 1.382337 | 0.103409 | +19.98% | +0.00% | — |
| tsds | 3 | 1.382384 | 0.084658 | +19.98% | -0.00% | 1/3 |
| rags-traj-maxSum | 3 | 1.418465 | 0.054267 | +17.89% | -2.61% | 1/3 |
| rags-block-expSum | 3 | 1.436664 | 0.045787 | +16.84% | -3.93% | 1/3 |
| less-main | 3 | 1.441543 | 0.037287 | +16.55% | -4.28% | 1/3 |
| rags-traj-zeroClipSum | 3 | 1.456554 | 0.066228 | +15.68% | -5.37% | 0/3 |
| rags-block-maxSum | 3 | 1.497766 | 0.090333 | +13.30% | -8.35% | 1/3 |
| orig | 1 | 1.727507 | — | +0.00% | -24.97% | — |
| lp-mirror | 3 | 2.172320 | 0.181392 | -25.75% | -57.15% | 0/3 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-expSum | 3 | 0.685112 | 0.011305 | +22.32% | +7.57% | 3/3 |
| rags-block-linearSum | 3 | 0.694575 | 0.007059 | +21.24% | +6.30% | 3/3 |
| rags-traj-maxSum | 3 | 0.710861 | 0.006832 | +19.40% | +4.10% | 3/3 |
| rags-traj-linearSum | 3 | 0.712236 | 0.010111 | +19.24% | +3.91% | 3/3 |
| rags-traj-zeroClipSum | 3 | 0.717496 | 0.012561 | +18.64% | +3.20% | 3/3 |
| tsds | 3 | 0.721019 | 0.001882 | +18.24% | +2.73% | 3/3 |
| less-main | 3 | 0.721196 | 0.004319 | +18.22% | +2.70% | 3/3 |
| rags-traj-expSum | 3 | 0.722459 | 0.012771 | +18.08% | +2.53% | 3/3 |
| rags-block-zeroClipSum | 3 | 0.724503 | 0.002566 | +17.85% | +2.26% | 3/3 |
| rags-block-maxSum | 3 | 0.726064 | 0.020073 | +17.67% | +2.05% | 2/3 |
| rand | 3 | 0.741243 | 0.016305 | +15.95% | +0.00% | — |
| lp-mirror | 3 | 0.877323 | 0.000192 | +0.52% | -18.36% | 0/3 |
| orig | 1 | 0.881911 | — | +0.00% | -18.98% | — |

### time_azure_i_cpu_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-zeroClipSum | 3 | 1.182011 | 0.008908 | +11.24% | +3.92% | 2/3 |
| rags-traj-zeroClipSum | 3 | 1.191853 | 0.029867 | +10.50% | +3.12% | 3/3 |
| rags-block-linearSum | 3 | 1.205612 | 0.015714 | +9.46% | +2.00% | 2/3 |
| rags-traj-linearSum | 3 | 1.211428 | 0.012685 | +9.03% | +1.53% | 2/3 |
| tsds | 3 | 1.212413 | 0.042393 | +8.95% | +1.45% | 2/3 |
| rags-traj-expSum | 3 | 1.213672 | 0.022960 | +8.86% | +1.35% | 2/3 |
| lp-mirror | 3 | 1.216375 | 0.028541 | +8.66% | +1.13% | 2/3 |
| less-main | 3 | 1.219202 | 0.009377 | +8.44% | +0.90% | 2/3 |
| rags-block-maxSum | 3 | 1.220678 | 0.021710 | +8.33% | +0.78% | 2/3 |
| rags-traj-maxSum | 3 | 1.224140 | 0.016007 | +8.07% | +0.50% | 2/3 |
| rags-block-expSum | 3 | 1.227666 | 0.049358 | +7.81% | +0.21% | 2/3 |
| rand | 3 | 1.230251 | 0.043551 | +7.61% | +0.00% | — |
| orig | 1 | 1.331634 | — | +0.00% | -8.24% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 3 | 0.762043 | 0.001374 | +0.23% | +0.33% | 2/3 |
| rags-block-maxSum | 3 | 0.763522 | 0.005380 | +0.04% | +0.14% | 2/3 |
| less-main | 3 | 0.763533 | 0.003292 | +0.04% | +0.14% | 2/3 |
| rags-traj-maxSum | 3 | 0.763798 | 0.002214 | +0.00% | +0.10% | 2/3 |
| orig | 1 | 0.763820 | — | +0.00% | +0.10% | — |
| rags-block-zeroClipSum | 3 | 0.763992 | 0.001641 | -0.02% | +0.08% | 2/3 |
| rags-traj-zeroClipSum | 3 | 0.764064 | 0.003582 | -0.03% | +0.07% | 1/3 |
| rand | 3 | 0.764581 | 0.002946 | -0.10% | +0.00% | — |
| rags-traj-linearSum | 3 | 0.766217 | 0.001643 | -0.31% | -0.21% | 0/3 |
| rags-block-linearSum | 3 | 0.766367 | 0.001361 | -0.33% | -0.23% | 2/3 |
| tsds | 3 | 0.766826 | 0.001165 | -0.39% | -0.29% | 1/3 |
| rags-traj-expSum | 3 | 0.767493 | 0.001745 | -0.48% | -0.38% | 0/3 |
| rags-block-expSum | 3 | 0.768138 | 0.003192 | -0.57% | -0.47% | 0/3 |

### time_darwin_water_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-maxSum | 3 | 1.656349 | 0.004625 | +0.46% | +0.38% | 2/3 |
| rags-block-expSum | 3 | 1.657247 | 0.005153 | +0.41% | +0.32% | 3/3 |
| rags-block-zeroClipSum | 3 | 1.659809 | 0.003826 | +0.26% | +0.17% | 2/3 |
| less-main | 3 | 1.660169 | 0.003378 | +0.24% | +0.15% | 2/3 |
| rags-traj-zeroClipSum | 3 | 1.660336 | 0.008302 | +0.23% | +0.14% | 2/3 |
| rags-block-linearSum | 3 | 1.660981 | 0.002592 | +0.19% | +0.10% | 3/3 |
| tsds | 3 | 1.661259 | 0.004549 | +0.17% | +0.08% | 1/3 |
| rand | 3 | 1.662618 | 0.002768 | +0.09% | +0.00% | — |
| orig | 1 | 1.664086 | — | +0.00% | -0.09% | — |
| rags-traj-expSum | 3 | 1.665946 | 0.005450 | -0.11% | -0.20% | 0/3 |
| rags-traj-maxSum | 3 | 1.668158 | 0.003966 | -0.24% | -0.33% | 0/3 |
| rags-traj-linearSum | 3 | 1.669625 | 0.009725 | -0.33% | -0.42% | 1/3 |
| lp-mirror | 3 | 1.688070 | 0.077659 | -1.44% | -1.53% | 2/3 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 3 | 1.001466 | 0.004124 | +6.38% | +2.99% | 3/3 |
| rags-block-expSum | 3 | 1.004411 | 0.004390 | +6.10% | +2.71% | 3/3 |
| rags-block-zeroClipSum | 3 | 1.008161 | 0.003795 | +5.75% | +2.35% | 3/3 |
| rags-traj-zeroClipSum | 3 | 1.023760 | 0.005129 | +4.30% | +0.83% | 3/3 |
| rags-traj-maxSum | 3 | 1.023887 | 0.003982 | +4.28% | +0.82% | 3/3 |
| rags-block-maxSum | 3 | 1.026001 | 0.001023 | +4.09% | +0.62% | 3/3 |
| rags-traj-linearSum | 3 | 1.027278 | 0.003369 | +3.97% | +0.49% | 3/3 |
| less-main | 3 | 1.027568 | 0.001079 | +3.94% | +0.47% | 2/3 |
| rags-traj-expSum | 3 | 1.029867 | 0.004354 | +3.72% | +0.24% | 2/3 |
| rand | 3 | 1.032380 | 0.006643 | +3.49% | +0.00% | — |
| tsds | 3 | 1.039987 | 0.002520 | +2.78% | -0.74% | 0/3 |
| lp-mirror | 3 | 1.069666 | 0.000241 | +0.00% | -3.61% | 0/3 |
| orig | 1 | 1.069706 | — | +0.00% | -3.62% | — |

### time_finland_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| tsds | 3 | 1.021053 | 0.043874 | +50.33% | +17.28% | 3/3 |
| rags-block-expSum | 3 | 1.104193 | 0.109595 | +46.28% | +10.54% | 3/3 |
| less-main | 3 | 1.137035 | 0.128787 | +44.69% | +7.88% | 2/3 |
| rags-traj-linearSum | 3 | 1.141705 | 0.257522 | +44.46% | +7.51% | 2/3 |
| rags-traj-maxSum | 3 | 1.141705 | 0.257522 | +44.46% | +7.51% | 2/3 |
| rags-traj-expSum | 3 | 1.141705 | 0.257522 | +44.46% | +7.51% | 2/3 |
| rags-block-maxSum | 3 | 1.152989 | 0.271234 | +43.91% | +6.59% | 2/3 |
| rags-traj-zeroClipSum | 3 | 1.221960 | 0.181537 | +40.55% | +1.00% | 2/3 |
| rags-block-zeroClipSum | 3 | 1.233872 | 0.200841 | +39.97% | +0.04% | 2/3 |
| rand | 3 | 1.234354 | 0.062098 | +39.95% | +0.00% | — |
| rags-block-linearSum | 3 | 1.335114 | 0.198919 | +35.05% | -8.16% | 1/3 |
| lp-mirror | 3 | 1.587298 | 0.121479 | +22.78% | -28.59% | 0/3 |
| orig | 1 | 2.055572 | — | +0.00% | -66.53% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 3 | 0.609725 | 0.002420 | +1.95% | +1.02% | 3/3 |
| rags-traj-zeroClipSum | 3 | 0.610231 | 0.002732 | +1.87% | +0.93% | 3/3 |
| rags-traj-linearSum | 3 | 0.610333 | 0.000888 | +1.86% | +0.92% | 3/3 |
| rags-traj-maxSum | 3 | 0.610333 | 0.000888 | +1.86% | +0.92% | 3/3 |
| rags-traj-expSum | 3 | 0.610333 | 0.000888 | +1.86% | +0.92% | 3/3 |
| rags-block-expSum | 3 | 0.611438 | 0.001785 | +1.68% | +0.74% | 2/3 |
| rags-block-zeroClipSum | 3 | 0.612653 | 0.001008 | +1.48% | +0.54% | 3/3 |
| rags-block-maxSum | 3 | 0.612945 | 0.004107 | +1.44% | +0.49% | 2/3 |
| tsds | 3 | 0.614979 | 0.000692 | +1.11% | +0.16% | 2/3 |
| rand | 3 | 0.615987 | 0.003555 | +0.95% | +0.00% | — |
| less-main | 3 | 0.616636 | 0.002182 | +0.84% | -0.11% | 2/3 |
| orig | 1 | 0.621873 | — | +0.00% | -0.96% | — |
| lp-mirror | 3 | 0.625578 | 0.001012 | -0.60% | -1.56% | 0/3 |

### time_eweld_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 3 | 1.175806 | 0.035272 | +21.40% | +9.68% | 3/3 |
| rags-block-expSum | 3 | 1.203214 | 0.059509 | +19.57% | +7.57% | 2/3 |
| rags-traj-maxSum | 3 | 1.228749 | 0.072498 | +17.86% | +5.61% | 2/3 |
| less-main | 3 | 1.235223 | 0.042140 | +17.43% | +5.11% | 2/3 |
| rags-block-linearSum | 3 | 1.241541 | 0.088262 | +17.01% | +4.63% | 2/3 |
| rags-block-zeroClipSum | 3 | 1.242002 | 0.072458 | +16.98% | +4.59% | 2/3 |
| rags-traj-expSum | 3 | 1.256428 | 0.052334 | +16.01% | +3.48% | 2/3 |
| tsds | 3 | 1.273446 | 0.040967 | +14.88% | +2.17% | 2/3 |
| rand | 3 | 1.301758 | 0.119424 | +12.98% | +0.00% | — |
| rags-traj-zeroClipSum | 3 | 1.302969 | 0.064181 | +12.90% | -0.09% | 2/3 |
| rags-traj-linearSum | 3 | 1.317095 | 0.018932 | +11.96% | -1.18% | 2/3 |
| rags-block-maxSum | 3 | 1.373724 | 0.203239 | +8.17% | -5.53% | 1/3 |
| orig | 1 | 1.495984 | — | +0.00% | -14.92% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-main | 3 | 0.721271 | 0.001098 | +0.96% | +0.24% | 3/3 |
| rags-block-linearSum | 3 | 0.721899 | 0.000427 | +0.87% | +0.15% | 2/3 |
| rags-block-expSum | 3 | 0.722798 | 0.000506 | +0.75% | +0.03% | 1/3 |
| rags-traj-zeroClipSum | 3 | 0.722861 | 0.001092 | +0.74% | +0.02% | 2/3 |
| rags-traj-expSum | 3 | 0.722889 | 0.000901 | +0.74% | +0.01% | 1/3 |
| rand | 3 | 0.722988 | 0.001693 | +0.72% | +0.00% | — |
| rags-traj-linearSum | 3 | 0.723484 | 0.001942 | +0.65% | -0.07% | 1/3 |
| rags-block-zeroClipSum | 3 | 0.723570 | 0.000187 | +0.64% | -0.08% | 1/3 |
| rags-block-maxSum | 3 | 0.723782 | 0.000801 | +0.61% | -0.11% | 1/3 |
| tsds | 3 | 0.724220 | 0.002607 | +0.55% | -0.17% | 2/3 |
| rags-traj-maxSum | 3 | 0.724653 | 0.002027 | +0.49% | -0.23% | 0/3 |
| orig | 1 | 0.728248 | — | +0.00% | -0.73% | — |
| lp-mirror | 3 | 0.729592 | 0.002234 | -0.18% | -0.91% | 0/3 |

### time_cphl_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 3 | 1.508442 | 0.001773 | +1.97% | +3.65% | 3/3 |
| rags-traj-zeroClipSum | 3 | 1.513568 | 0.040629 | +1.64% | +3.32% | 3/3 |
| rags-block-maxSum | 3 | 1.514595 | 0.030408 | +1.57% | +3.26% | 3/3 |
| orig | 1 | 1.538772 | — | +0.00% | +1.71% | — |
| rags-traj-linearSum | 3 | 1.539250 | 0.038053 | -0.03% | +1.68% | 2/3 |
| rags-block-expSum | 3 | 1.539277 | 0.039354 | -0.03% | +1.68% | 2/3 |
| tsds | 3 | 1.543695 | 0.019645 | -0.32% | +1.40% | 3/3 |
| rags-traj-expSum | 3 | 1.545664 | 0.026843 | -0.45% | +1.27% | 2/3 |
| rags-block-zeroClipSum | 3 | 1.546878 | 0.033455 | -0.53% | +1.19% | 3/3 |
| rags-traj-maxSum | 3 | 1.555795 | 0.011564 | -1.11% | +0.62% | 1/3 |
| lp-mirror | 3 | 1.561549 | 0.058348 | -1.48% | +0.26% | 2/3 |
| less-main | 3 | 1.562286 | 0.015816 | -1.53% | +0.21% | 2/3 |
| rand | 3 | 1.565579 | 0.017958 | -1.74% | +0.00% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-maxSum | 3 | 1.329569 | 0.003257 | +2.68% | +1.29% | 3/3 |
| rags-block-expSum | 3 | 1.333185 | 0.002204 | +2.42% | +1.02% | 3/3 |
| rags-block-zeroClipSum | 3 | 1.333501 | 0.005359 | +2.39% | +1.00% | 3/3 |
| rags-block-linearSum | 3 | 1.335261 | 0.003390 | +2.27% | +0.86% | 3/3 |
| less-main | 3 | 1.345672 | 0.007804 | +1.50% | +0.09% | 1/3 |
| rags-traj-maxSum | 3 | 1.346804 | 0.002875 | +1.42% | +0.01% | 2/3 |
| lp-mirror | 3 | 1.346897 | 0.003127 | +1.41% | +0.00% | 2/3 |
| rand | 3 | 1.346911 | 0.001671 | +1.41% | +0.00% | — |
| rags-traj-zeroClipSum | 3 | 1.348653 | 0.004292 | +1.29% | -0.13% | 2/3 |
| rags-traj-linearSum | 3 | 1.349335 | 0.001386 | +1.24% | -0.18% | 1/3 |
| rags-traj-expSum | 3 | 1.351715 | 0.001343 | +1.06% | -0.36% | 0/3 |
| tsds | 3 | 1.364836 | 0.005639 | +0.10% | -1.33% | 0/3 |
| orig | 1 | 1.366210 | — | +0.00% | -1.43% | — |

### time_nem_price_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-expSum | 3 | 1.032367 | 0.003353 | +2.81% | +1.75% | 3/3 |
| less-main | 3 | 1.040161 | 0.002707 | +2.07% | +1.01% | 3/3 |
| rags-block-linearSum | 3 | 1.040698 | 0.001987 | +2.02% | +0.96% | 2/3 |
| rags-traj-expSum | 3 | 1.042299 | 0.005468 | +1.87% | +0.81% | 2/3 |
| tsds | 3 | 1.043533 | 0.010296 | +1.76% | +0.69% | 2/3 |
| rags-block-maxSum | 3 | 1.045123 | 0.005538 | +1.61% | +0.54% | 2/3 |
| rags-traj-zeroClipSum | 3 | 1.046404 | 0.006250 | +1.49% | +0.42% | 2/3 |
| rags-block-zeroClipSum | 3 | 1.046861 | 0.009176 | +1.44% | +0.37% | 1/3 |
| rags-traj-maxSum | 3 | 1.049906 | 0.004722 | +1.16% | +0.08% | 1/3 |
| rand | 3 | 1.050788 | 0.014902 | +1.07% | +0.00% | — |
| rags-traj-linearSum | 3 | 1.052576 | 0.006229 | +0.91% | -0.17% | 1/3 |
| lp-mirror | 3 | 1.054419 | 0.006381 | +0.73% | -0.35% | 1/3 |
| orig | 1 | 1.062202 | — | +0.00% | -1.09% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-expSum | 3 | 0.710460 | 0.003316 | +5.17% | +1.95% | 3/3 |
| rags-traj-linearSum | 3 | 0.710718 | 0.002332 | +5.14% | +1.92% | 3/3 |
| rags-traj-zeroClipSum | 3 | 0.712162 | 0.003544 | +4.94% | +1.72% | 3/3 |
| rags-block-expSum | 3 | 0.716131 | 0.000700 | +4.41% | +1.17% | 3/3 |
| less-main | 3 | 0.716408 | 0.000599 | +4.38% | +1.13% | 3/3 |
| rags-block-linearSum | 3 | 0.716635 | 0.001839 | +4.35% | +1.10% | 3/3 |
| rags-traj-maxSum | 3 | 0.716659 | 0.004233 | +4.34% | +1.10% | 2/3 |
| rags-block-zeroClipSum | 3 | 0.716862 | 0.003424 | +4.32% | +1.07% | 3/3 |
| tsds | 3 | 0.720954 | 0.002474 | +3.77% | +0.51% | 2/3 |
| rags-block-maxSum | 3 | 0.722611 | 0.003252 | +3.55% | +0.28% | 2/3 |
| rand | 3 | 0.724624 | 0.006311 | +3.28% | +0.00% | — |
| orig | 1 | 0.749205 | — | +0.00% | -3.39% | — |
| lp-mirror | 3 | 0.755532 | 0.000901 | -0.84% | -4.27% | 0/3 |

### time_solar_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 3 | 1.204102 | 0.060388 | +1.57% | +0.00% | — |
| rags-block-maxSum | 3 | 1.207896 | 0.245254 | +1.26% | -0.32% | 2/3 |
| rags-block-expSum | 3 | 1.217178 | 0.016173 | +0.50% | -1.09% | 2/3 |
| rags-block-linearSum | 3 | 1.221579 | 0.101530 | +0.15% | -1.45% | 1/3 |
| orig | 1 | 1.223355 | — | +0.00% | -1.60% | — |
| lp-mirror | 3 | 1.228822 | 0.158462 | -0.45% | -2.05% | 1/3 |
| less-main | 3 | 1.244282 | 0.056064 | -1.71% | -3.34% | 1/3 |
| rags-block-zeroClipSum | 3 | 1.270936 | 0.142540 | -3.89% | -5.55% | 1/3 |
| tsds | 3 | 1.292194 | 0.080047 | -5.63% | -7.32% | 1/3 |
| rags-traj-zeroClipSum | 3 | 1.308656 | 0.041455 | -6.97% | -8.68% | 0/3 |
| rags-traj-linearSum | 3 | 1.319936 | 0.272292 | -7.89% | -9.62% | 1/3 |
| rags-traj-maxSum | 3 | 1.319936 | 0.272292 | -7.89% | -9.62% | 1/3 |
| rags-traj-expSum | 3 | 1.319936 | 0.272292 | -7.89% | -9.62% | 1/3 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-expSum | 3 | 1.114610 | 0.005932 | +1.30% | +1.15% | 3/3 |
| lp-mirror | 3 | 1.117018 | 0.011846 | +1.09% | +0.93% | 2/3 |
| rags-block-linearSum | 3 | 1.118433 | 0.009260 | +0.96% | +0.81% | 3/3 |
| rags-traj-zeroClipSum | 3 | 1.118650 | 0.011620 | +0.94% | +0.79% | 2/3 |
| rags-block-zeroClipSum | 3 | 1.120632 | 0.010944 | +0.77% | +0.61% | 2/3 |
| less-main | 3 | 1.121734 | 0.009768 | +0.67% | +0.51% | 1/3 |
| rags-traj-linearSum | 3 | 1.127492 | 0.014421 | +0.16% | +0.00% | 1/3 |
| rags-traj-maxSum | 3 | 1.127492 | 0.014421 | +0.16% | +0.00% | 1/3 |
| rags-traj-expSum | 3 | 1.127492 | 0.014421 | +0.16% | +0.00% | 1/3 |
| rand | 3 | 1.127534 | 0.007553 | +0.16% | +0.00% | — |
| orig | 1 | 1.129299 | — | +0.00% | -0.16% | — |
| rags-block-maxSum | 3 | 1.131380 | 0.014981 | -0.18% | -0.34% | 1/3 |
| tsds | 3 | 1.132088 | 0.001585 | -0.25% | -0.40% | 0/3 |

### time_port_activity_d

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.882133 | — | +0.00% | +1.31% | — |
| rags-traj-maxSum | 3 | 0.883752 | 0.009543 | -0.18% | +1.13% | 2/3 |
| lp-mirror | 3 | 0.883800 | 0.002488 | -0.19% | +1.12% | 2/3 |
| rags-block-zeroClipSum | 3 | 0.886293 | 0.005916 | -0.47% | +0.84% | 2/3 |
| rags-block-expSum | 3 | 0.888557 | 0.006190 | -0.73% | +0.59% | 2/3 |
| less-main | 3 | 0.892152 | 0.011043 | -1.14% | +0.19% | 2/3 |
| rags-traj-zeroClipSum | 3 | 0.893739 | 0.019669 | -1.32% | +0.01% | 1/3 |
| rand | 3 | 0.893838 | 0.015555 | -1.33% | +0.00% | — |
| rags-traj-expSum | 3 | 0.896194 | 0.015326 | -1.59% | -0.26% | 1/3 |
| rags-block-maxSum | 3 | 0.897754 | 0.018922 | -1.77% | -0.44% | 2/3 |
| rags-block-linearSum | 3 | 0.899795 | 0.004386 | -2.00% | -0.67% | 2/3 |
| rags-traj-linearSum | 3 | 0.901212 | 0.024135 | -2.16% | -0.82% | 1/3 |
| tsds | 3 | 0.917647 | 0.005653 | -4.03% | -2.66% | 0/3 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.716205 | — | +0.00% | +0.90% | — |
| rand | 3 | 0.722692 | 0.000984 | -0.91% | +0.00% | — |
| lp-mirror | 3 | 0.722733 | 0.000367 | -0.91% | -0.01% | 2/3 |
| rags-traj-maxSum | 3 | 0.727924 | 0.000993 | -1.64% | -0.72% | 0/3 |
| rags-traj-zeroClipSum | 3 | 0.728760 | 0.001995 | -1.75% | -0.84% | 0/3 |
| rags-traj-linearSum | 3 | 0.728903 | 0.000890 | -1.77% | -0.86% | 0/3 |
| less-main | 3 | 0.728935 | 0.000851 | -1.78% | -0.86% | 0/3 |
| rags-traj-expSum | 3 | 0.729109 | 0.001311 | -1.80% | -0.89% | 0/3 |
| rags-block-maxSum | 3 | 0.730437 | 0.001239 | -1.99% | -1.07% | 0/3 |
| rags-block-linearSum | 3 | 0.733143 | 0.001494 | -2.36% | -1.45% | 0/3 |
| rags-block-expSum | 3 | 0.733374 | 0.000704 | -2.40% | -1.48% | 0/3 |
| rags-block-zeroClipSum | 3 | 0.734667 | 0.000179 | -2.58% | -1.66% | 0/3 |
| tsds | 3 | 0.736037 | 0.000884 | -2.77% | -1.85% | 0/3 |

**解读**：

以本版真实落盘结果更新；没有结果时不作效果判断。

<!-- v5-patrol:end -->

| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |
|---|---|---|---:|---:|---:|
| chronos-base | time_finland_15t | tsds | 3 | 1.021053 | 0.043874 |
| chronos-base | time_finland_15t | rags-block-expSum | 3 | 1.104193 | 0.109595 |
| chronos-base | time_eweld_15t | lp-mirror | 3 | 1.175806 | 0.035272 |
| chronos-base | time_finland_15t | less-main | 3 | 1.137035 | 0.128787 |
| timesfm-2.5 | time_nem_demand_5t | rags-block-expSum | 3 | 0.685112 | 0.011305 |
| chronos-base | time_eweld_15t | rags-block-expSum | 3 | 1.203214 | 0.059509 |
| chronos-base | time_finland_15t | rags-traj-expSum | 3 | 1.141705 | 0.257522 |
| chronos-base | time_finland_15t | rags-traj-linearSum | 3 | 1.141705 | 0.257522 |
| chronos-base | time_finland_15t | rags-traj-maxSum | 3 | 1.141705 | 0.257522 |
| chronos-base | time_finland_15t | rags-block-maxSum | 3 | 1.152989 | 0.271234 |
| timesfm-2.5 | time_nem_demand_5t | rags-block-linearSum | 3 | 0.694575 | 0.007059 |
| chronos-base | time_eweld_15t | rags-traj-maxSum | 3 | 1.228749 | 0.072498 |
| chronos-base | time_eweld_15t | less-main | 3 | 1.235223 | 0.042140 |
| chronos-base | time_eweld_15t | rags-block-linearSum | 3 | 1.241541 | 0.088262 |
| chronos-base | time_eweld_15t | rags-block-zeroClipSum | 3 | 1.242002 | 0.072458 |
| chronos-base | time_nem_demand_5t | rags-block-zeroClipSum | 3 | 1.323450 | 0.066163 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-maxSum | 3 | 0.710861 | 0.006832 |
| chronos-base | time_nem_demand_5t | rags-block-linearSum | 3 | 1.327354 | 0.115937 |
| chronos-base | time_azure_i_cpu_5t | rags-block-zeroClipSum | 3 | 1.182011 | 0.008908 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-linearSum | 3 | 0.712236 | 0.010111 |
| chronos-base | time_cphl_h | rags-block-linearSum | 3 | 1.508442 | 0.001773 |
| chronos-base | time_eweld_15t | rags-traj-expSum | 3 | 1.256428 | 0.052334 |
| chronos-base | time_cphl_h | rags-traj-zeroClipSum | 3 | 1.513568 | 0.040629 |
| chronos-base | time_cphl_h | rags-block-maxSum | 3 | 1.514595 | 0.030408 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-zeroClipSum | 3 | 0.717496 | 0.012561 |
| chronos-base | time_azure_i_cpu_5t | rags-traj-zeroClipSum | 3 | 1.191853 | 0.029867 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-linearSum | 3 | 1.001466 | 0.004124 |
| timesfm-2.5 | time_nem_demand_5t | tsds | 3 | 0.721019 | 0.001882 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-expSum | 3 | 1.004411 | 0.004390 |
| timesfm-2.5 | time_nem_demand_5t | less-main | 3 | 0.721196 | 0.004319 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-expSum | 3 | 0.722459 | 0.012771 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-zeroClipSum | 3 | 1.008161 | 0.003795 |
| timesfm-2.5 | time_nem_demand_5t | rags-block-zeroClipSum | 3 | 0.724503 | 0.002566 |
| chronos-base | time_eweld_15t | tsds | 3 | 1.273446 | 0.040967 |
| timesfm-2.5 | time_nem_demand_5t | rags-block-maxSum | 3 | 0.726064 | 0.020073 |
| chronos-base | time_azure_i_cpu_5t | rags-block-linearSum | 3 | 1.205612 | 0.015714 |
| timesfm-2.5 | time_nem_price_5t | rags-traj-expSum | 3 | 0.710460 | 0.003316 |
| timesfm-2.5 | time_nem_price_5t | rags-traj-linearSum | 3 | 0.710718 | 0.002332 |
| chronos-base | time_nem_demand_5t | rags-traj-expSum | 3 | 1.357953 | 0.089838 |
| chronos-base | time_nem_price_5t | rags-block-expSum | 3 | 1.032367 | 0.003353 |
| timesfm-2.5 | time_nem_price_5t | rags-traj-zeroClipSum | 3 | 0.712162 | 0.003544 |
| chronos-base | time_cphl_h | orig | 1 | 1.538772 | — |
| chronos-base | time_cphl_h | rags-traj-linearSum | 3 | 1.539250 | 0.038053 |
| chronos-base | time_cphl_h | rags-block-expSum | 3 | 1.539277 | 0.039354 |
| chronos-base | time_azure_i_cpu_5t | rags-traj-linearSum | 3 | 1.211428 | 0.012685 |
| chronos-base | time_azure_i_cpu_5t | tsds | 3 | 1.212413 | 0.042393 |
| chronos-base | time_cphl_h | tsds | 3 | 1.543695 | 0.019645 |
| chronos-base | time_azure_i_cpu_5t | rags-traj-expSum | 3 | 1.213672 | 0.022960 |
| chronos-base | time_port_activity_d | orig | 1 | 0.882133 | — |
| timesfm-2.5 | time_cphl_h | rags-block-maxSum | 3 | 1.329569 | 0.003257 |
| chronos-base | time_cphl_h | rags-traj-expSum | 3 | 1.545664 | 0.026843 |
| chronos-base | time_cphl_h | rags-block-zeroClipSum | 3 | 1.546878 | 0.033455 |
| timesfm-2.5 | time_nem_price_5t | rags-block-expSum | 3 | 0.716131 | 0.000700 |
| timesfm-2.5 | time_solar_h | rags-block-expSum | 3 | 1.114610 | 0.005932 |
| timesfm-2.5 | time_nem_price_5t | less-main | 3 | 0.716408 | 0.000599 |
| chronos-base | time_port_activity_d | rags-traj-maxSum | 3 | 0.883752 | 0.009543 |
| chronos-base | time_azure_i_cpu_5t | lp-mirror | 3 | 1.216375 | 0.028541 |
| chronos-base | time_port_activity_d | lp-mirror | 3 | 0.883800 | 0.002488 |
| timesfm-2.5 | time_nem_price_5t | rags-block-linearSum | 3 | 0.716635 | 0.001839 |
| timesfm-2.5 | time_nem_price_5t | rags-traj-maxSum | 3 | 0.716659 | 0.004233 |
| timesfm-2.5 | time_nem_price_5t | rags-block-zeroClipSum | 3 | 0.716862 | 0.003424 |
| timesfm-2.5 | time_cphl_h | rags-block-expSum | 3 | 1.333185 | 0.002204 |
| timesfm-2.5 | time_finland_15t | rags-block-linearSum | 3 | 0.609725 | 0.002420 |
| chronos-base | time_nem_price_5t | less-main | 3 | 1.040161 | 0.002707 |
| chronos-base | time_finland_15t | rags-traj-zeroClipSum | 3 | 1.221960 | 0.181537 |
| timesfm-2.5 | time_cphl_h | rags-block-zeroClipSum | 3 | 1.333501 | 0.005359 |
| chronos-base | time_nem_price_5t | rags-block-linearSum | 3 | 1.040698 | 0.001987 |
| timesfm-2.5 | time_finland_15t | rags-traj-zeroClipSum | 3 | 0.610231 | 0.002732 |
| timesfm-2.5 | time_solar_h | lp-mirror | 3 | 1.117018 | 0.011846 |
| timesfm-2.5 | time_finland_15t | rags-traj-expSum | 3 | 0.610333 | 0.000888 |
| timesfm-2.5 | time_finland_15t | rags-traj-linearSum | 3 | 0.610333 | 0.000888 |
| timesfm-2.5 | time_finland_15t | rags-traj-maxSum | 3 | 0.610333 | 0.000888 |
| chronos-base | time_azure_i_cpu_5t | less-main | 3 | 1.219202 | 0.009377 |
| timesfm-2.5 | time_port_activity_d | orig | 1 | 0.716205 | — |
| timesfm-2.5 | time_cphl_h | rags-block-linearSum | 3 | 1.335261 | 0.003390 |
| chronos-base | time_port_activity_d | rags-block-zeroClipSum | 3 | 0.886293 | 0.005916 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-zeroClipSum | 3 | 1.023760 | 0.005129 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-maxSum | 3 | 1.023887 | 0.003982 |
| chronos-base | time_nem_price_5t | rags-traj-expSum | 3 | 1.042299 | 0.005468 |
| timesfm-2.5 | time_solar_h | rags-block-linearSum | 3 | 1.118433 | 0.009260 |
| timesfm-2.5 | time_solar_h | rags-traj-zeroClipSum | 3 | 1.118650 | 0.011620 |
| chronos-base | time_azure_i_cpu_5t | rags-block-maxSum | 3 | 1.220678 | 0.021710 |
| timesfm-2.5 | time_finland_15t | rags-block-expSum | 3 | 0.611438 | 0.001785 |
| chronos-base | time_nem_price_5t | tsds | 3 | 1.043533 | 0.010296 |
| chronos-base | time_cphl_h | rags-traj-maxSum | 3 | 1.555795 | 0.011564 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-maxSum | 3 | 1.026001 | 0.001023 |
| timesfm-2.5 | time_solar_h | rags-block-zeroClipSum | 3 | 1.120632 | 0.010944 |
| chronos-base | time_port_activity_d | rags-block-expSum | 3 | 0.888557 | 0.006190 |
| timesfm-2.5 | time_finland_15t | rags-block-zeroClipSum | 3 | 0.612653 | 0.001008 |
| chronos-base | time_nem_price_5t | rags-block-maxSum | 3 | 1.045123 | 0.005538 |
| timesfm-2.5 | time_solar_h | less-main | 3 | 1.121734 | 0.009768 |
| timesfm-2.5 | time_nem_price_5t | tsds | 3 | 0.720954 | 0.002474 |
| chronos-base | time_azure_i_cpu_5t | rags-traj-maxSum | 3 | 1.224140 | 0.016007 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-linearSum | 3 | 1.027278 | 0.003369 |
| timesfm-2.5 | time_finland_15t | rags-block-maxSum | 3 | 0.612945 | 0.004107 |
| timesfm-2.5 | time_darwin_water_15t | less-main | 3 | 1.027568 | 0.001079 |
| chronos-base | time_nem_price_5t | rags-traj-zeroClipSum | 3 | 1.046404 | 0.006250 |
| chronos-base | time_darwin_water_15t | rags-block-maxSum | 3 | 1.656349 | 0.004625 |
| chronos-base | time_nem_price_5t | rags-block-zeroClipSum | 3 | 1.046861 | 0.009176 |
| timesfm-2.5 | time_azure_i_cpu_5t | lp-mirror | 3 | 0.762043 | 0.001374 |
| chronos-base | time_darwin_water_15t | rags-block-expSum | 3 | 1.657247 | 0.005153 |
| timesfm-2.5 | time_nem_price_5t | rags-block-maxSum | 3 | 0.722611 | 0.003252 |
| chronos-base | time_cphl_h | lp-mirror | 3 | 1.561549 | 0.058348 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-expSum | 3 | 1.029867 | 0.004354 |
| timesfm-2.5 | time_eweld_15t | less-main | 3 | 0.721271 | 0.001098 |
| chronos-base | time_cphl_h | less-main | 3 | 1.562286 | 0.015816 |
| chronos-base | time_azure_i_cpu_5t | rags-block-expSum | 3 | 1.227666 | 0.049358 |
| chronos-base | time_nem_demand_5t | rags-traj-linearSum | 3 | 1.379673 | 0.066397 |
| chronos-base | time_port_activity_d | less-main | 3 | 0.892152 | 0.011043 |
| chronos-base | time_darwin_water_15t | rags-block-zeroClipSum | 3 | 1.659809 | 0.003826 |
| timesfm-2.5 | time_finland_15t | tsds | 3 | 0.614979 | 0.000692 |
| timesfm-2.5 | time_eweld_15t | rags-block-linearSum | 3 | 0.721899 | 0.000427 |
| chronos-base | time_darwin_water_15t | less-main | 3 | 1.660169 | 0.003378 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-maxSum | 3 | 0.763522 | 0.005380 |
| chronos-base | time_darwin_water_15t | rags-traj-zeroClipSum | 3 | 1.660336 | 0.008302 |
| timesfm-2.5 | time_azure_i_cpu_5t | less-main | 3 | 0.763533 | 0.003292 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-maxSum | 3 | 0.763798 | 0.002214 |
| timesfm-2.5 | time_azure_i_cpu_5t | orig | 1 | 0.763820 | — |
| chronos-base | time_darwin_water_15t | rags-block-linearSum | 3 | 1.660981 | 0.002592 |
| timesfm-2.5 | time_cphl_h | less-main | 3 | 1.345672 | 0.007804 |
| chronos-base | time_nem_price_5t | rags-traj-maxSum | 3 | 1.049906 | 0.004722 |
| chronos-base | time_darwin_water_15t | tsds | 3 | 1.661259 | 0.004549 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-zeroClipSum | 3 | 0.763992 | 0.001641 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-zeroClipSum | 3 | 0.764064 | 0.003582 |
| chronos-base | time_finland_15t | rags-block-zeroClipSum | 3 | 1.233872 | 0.200841 |
| timesfm-2.5 | time_eweld_15t | rags-block-expSum | 3 | 0.722798 | 0.000506 |
| timesfm-2.5 | time_eweld_15t | rags-traj-zeroClipSum | 3 | 0.722861 | 0.001092 |
| timesfm-2.5 | time_eweld_15t | rags-traj-expSum | 3 | 0.722889 | 0.000901 |
| chronos-base | time_port_activity_d | rags-traj-zeroClipSum | 3 | 0.893739 | 0.019669 |
| timesfm-2.5 | time_cphl_h | rags-traj-maxSum | 3 | 1.346804 | 0.002875 |
| timesfm-2.5 | time_solar_h | rags-traj-expSum | 3 | 1.127492 | 0.014421 |
| timesfm-2.5 | time_solar_h | rags-traj-linearSum | 3 | 1.127492 | 0.014421 |
| timesfm-2.5 | time_solar_h | rags-traj-maxSum | 3 | 1.127492 | 0.014421 |
| timesfm-2.5 | time_cphl_h | lp-mirror | 3 | 1.346897 | 0.003127 |
| chronos-base | time_azure_i_cpu_5t | rand | 3 | 1.230251 | 0.043551 |
| chronos-base | time_cphl_h | rand | 3 | 1.565579 | 0.017958 |
| chronos-base | time_darwin_water_15t | rand | 3 | 1.662618 | 0.002768 |
| chronos-base | time_eweld_15t | rand | 3 | 1.301758 | 0.119424 |
| chronos-base | time_finland_15t | rand | 3 | 1.234354 | 0.062098 |
| chronos-base | time_nem_demand_5t | rand | 3 | 1.382337 | 0.103409 |
| chronos-base | time_nem_price_5t | rand | 3 | 1.050788 | 0.014902 |
| chronos-base | time_port_activity_d | rand | 3 | 0.893838 | 0.015555 |
| chronos-base | time_solar_h | rand | 3 | 1.204102 | 0.060388 |
| timesfm-2.5 | time_azure_i_cpu_5t | rand | 3 | 0.764581 | 0.002946 |
| timesfm-2.5 | time_cphl_h | rand | 3 | 1.346911 | 0.001671 |
| timesfm-2.5 | time_darwin_water_15t | rand | 3 | 1.032380 | 0.006643 |
| timesfm-2.5 | time_eweld_15t | rand | 3 | 0.722988 | 0.001693 |
| timesfm-2.5 | time_finland_15t | rand | 3 | 0.615987 | 0.003555 |
| timesfm-2.5 | time_nem_demand_5t | rand | 3 | 0.741243 | 0.016305 |
| timesfm-2.5 | time_nem_price_5t | rand | 3 | 0.724624 | 0.006311 |
| timesfm-2.5 | time_port_activity_d | rand | 3 | 0.722692 | 0.000984 |
| timesfm-2.5 | time_solar_h | rand | 3 | 1.127534 | 0.007553 |
| chronos-base | time_nem_demand_5t | tsds | 3 | 1.382384 | 0.084658 |
| timesfm-2.5 | time_port_activity_d | lp-mirror | 3 | 0.722733 | 0.000367 |
| timesfm-2.5 | time_eweld_15t | rags-traj-linearSum | 3 | 0.723484 | 0.001942 |
| timesfm-2.5 | time_eweld_15t | rags-block-zeroClipSum | 3 | 0.723570 | 0.000187 |
| chronos-base | time_darwin_water_15t | orig | 1 | 1.664086 | — |
| chronos-base | time_eweld_15t | rags-traj-zeroClipSum | 3 | 1.302969 | 0.064181 |
| timesfm-2.5 | time_finland_15t | less-main | 3 | 0.616636 | 0.002182 |
| timesfm-2.5 | time_eweld_15t | rags-block-maxSum | 3 | 0.723782 | 0.000801 |
| timesfm-2.5 | time_cphl_h | rags-traj-zeroClipSum | 3 | 1.348653 | 0.004292 |
| timesfm-2.5 | time_solar_h | orig | 1 | 1.129299 | — |
| chronos-base | time_nem_price_5t | rags-traj-linearSum | 3 | 1.052576 | 0.006229 |
| timesfm-2.5 | time_eweld_15t | tsds | 3 | 0.724220 | 0.002607 |
| timesfm-2.5 | time_cphl_h | rags-traj-linearSum | 3 | 1.349335 | 0.001386 |
| chronos-base | time_darwin_water_15t | rags-traj-expSum | 3 | 1.665946 | 0.005450 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-linearSum | 3 | 0.766217 | 0.001643 |
| timesfm-2.5 | time_eweld_15t | rags-traj-maxSum | 3 | 0.724653 | 0.002027 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-linearSum | 3 | 0.766367 | 0.001361 |
| chronos-base | time_port_activity_d | rags-traj-expSum | 3 | 0.896194 | 0.015326 |
| timesfm-2.5 | time_azure_i_cpu_5t | tsds | 3 | 0.766826 | 0.001165 |
| chronos-base | time_solar_h | rags-block-maxSum | 3 | 1.207896 | 0.245254 |
| chronos-base | time_darwin_water_15t | rags-traj-maxSum | 3 | 1.668158 | 0.003966 |
| timesfm-2.5 | time_solar_h | rags-block-maxSum | 3 | 1.131380 | 0.014981 |
| chronos-base | time_nem_price_5t | lp-mirror | 3 | 1.054419 | 0.006381 |
| timesfm-2.5 | time_cphl_h | rags-traj-expSum | 3 | 1.351715 | 0.001343 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-expSum | 3 | 0.767493 | 0.001745 |
| timesfm-2.5 | time_solar_h | tsds | 3 | 1.132088 | 0.001585 |
| chronos-base | time_darwin_water_15t | rags-traj-linearSum | 3 | 1.669625 | 0.009725 |
| chronos-base | time_port_activity_d | rags-block-maxSum | 3 | 0.897754 | 0.018922 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-expSum | 3 | 0.768138 | 0.003192 |
| chronos-base | time_port_activity_d | rags-block-linearSum | 3 | 0.899795 | 0.004386 |
| timesfm-2.5 | time_port_activity_d | rags-traj-maxSum | 3 | 0.727924 | 0.000993 |
| timesfm-2.5 | time_eweld_15t | orig | 1 | 0.728248 | — |
| timesfm-2.5 | time_darwin_water_15t | tsds | 3 | 1.039987 | 0.002520 |
| chronos-base | time_port_activity_d | rags-traj-linearSum | 3 | 0.901212 | 0.024135 |
| timesfm-2.5 | time_port_activity_d | rags-traj-zeroClipSum | 3 | 0.728760 | 0.001995 |
| timesfm-2.5 | time_port_activity_d | rags-traj-linearSum | 3 | 0.728903 | 0.000890 |
| timesfm-2.5 | time_port_activity_d | less-main | 3 | 0.728935 | 0.000851 |
| timesfm-2.5 | time_port_activity_d | rags-traj-expSum | 3 | 0.729109 | 0.001311 |
| timesfm-2.5 | time_eweld_15t | lp-mirror | 3 | 0.729592 | 0.002234 |
| timesfm-2.5 | time_finland_15t | orig | 1 | 0.621873 | — |
| timesfm-2.5 | time_port_activity_d | rags-block-maxSum | 3 | 0.730437 | 0.001239 |
| chronos-base | time_solar_h | rags-block-expSum | 3 | 1.217178 | 0.016173 |
| chronos-base | time_nem_price_5t | orig | 1 | 1.062202 | — |
| chronos-base | time_eweld_15t | rags-traj-linearSum | 3 | 1.317095 | 0.018932 |
| timesfm-2.5 | time_cphl_h | tsds | 3 | 1.364836 | 0.005639 |
| timesfm-2.5 | time_cphl_h | orig | 1 | 1.366210 | — |
| timesfm-2.5 | time_port_activity_d | rags-block-linearSum | 3 | 0.733143 | 0.001494 |
| chronos-base | time_solar_h | rags-block-linearSum | 3 | 1.221579 | 0.101530 |
| timesfm-2.5 | time_port_activity_d | rags-block-expSum | 3 | 0.733374 | 0.000704 |
| chronos-base | time_darwin_water_15t | lp-mirror | 3 | 1.688070 | 0.077659 |
| timesfm-2.5 | time_finland_15t | lp-mirror | 3 | 0.625578 | 0.001012 |
| chronos-base | time_solar_h | orig | 1 | 1.223355 | — |
| timesfm-2.5 | time_port_activity_d | rags-block-zeroClipSum | 3 | 0.734667 | 0.000179 |
| timesfm-2.5 | time_port_activity_d | tsds | 3 | 0.736037 | 0.000884 |
| chronos-base | time_solar_h | lp-mirror | 3 | 1.228822 | 0.158462 |
| chronos-base | time_nem_demand_5t | rags-traj-maxSum | 3 | 1.418465 | 0.054267 |
| chronos-base | time_port_activity_d | tsds | 3 | 0.917647 | 0.005653 |
| chronos-base | time_solar_h | less-main | 3 | 1.244282 | 0.056064 |
| timesfm-2.5 | time_nem_price_5t | orig | 1 | 0.749205 | — |
| timesfm-2.5 | time_darwin_water_15t | lp-mirror | 3 | 1.069666 | 0.000241 |
| timesfm-2.5 | time_darwin_water_15t | orig | 1 | 1.069706 | — |
| chronos-base | time_nem_demand_5t | rags-block-expSum | 3 | 1.436664 | 0.045787 |
| timesfm-2.5 | time_nem_price_5t | lp-mirror | 3 | 0.755532 | 0.000901 |
| chronos-base | time_nem_demand_5t | less-main | 3 | 1.441543 | 0.037287 |
| chronos-base | time_nem_demand_5t | rags-traj-zeroClipSum | 3 | 1.456554 | 0.066228 |
| chronos-base | time_eweld_15t | rags-block-maxSum | 3 | 1.373724 | 0.203239 |
| chronos-base | time_solar_h | rags-block-zeroClipSum | 3 | 1.270936 | 0.142540 |
| chronos-base | time_solar_h | tsds | 3 | 1.292194 | 0.080047 |
| chronos-base | time_finland_15t | rags-block-linearSum | 3 | 1.335114 | 0.198919 |
| chronos-base | time_azure_i_cpu_5t | orig | 1 | 1.331634 | — |
| chronos-base | time_nem_demand_5t | rags-block-maxSum | 3 | 1.497766 | 0.090333 |
| chronos-base | time_solar_h | rags-traj-zeroClipSum | 3 | 1.308656 | 0.041455 |
| chronos-base | time_solar_h | rags-traj-expSum | 3 | 1.319936 | 0.272292 |
| chronos-base | time_solar_h | rags-traj-linearSum | 3 | 1.319936 | 0.272292 |
| chronos-base | time_solar_h | rags-traj-maxSum | 3 | 1.319936 | 0.272292 |
| chronos-base | time_eweld_15t | orig | 1 | 1.495984 | — |
| timesfm-2.5 | time_nem_demand_5t | lp-mirror | 3 | 0.877323 | 0.000192 |
| timesfm-2.5 | time_nem_demand_5t | orig | 1 | 0.881911 | — |
| chronos-base | time_nem_demand_5t | orig | 1 | 1.727507 | — |
| chronos-base | time_finland_15t | lp-mirror | 3 | 1.587298 | 0.121479 |
| chronos-base | time_nem_demand_5t | lp-mirror | 3 | 2.172320 | 0.181392 |
| chronos-base | time_finland_15t | orig | 1 | 2.055572 | — |

## 最新跨模型汇报（2026-09-21）

- v5 主实验仍为完整结果：666/666 cells，Chronos-Base 与 TimesFM 2.5，9 个任务、13 个臂、3 seeds。
- v5-uni 已完成：342/342 cells；两个 uni 臂在 `UNI.md` 中有完整的 2 模型 × 9 任务 × 3 seeds 结果。
- TimerXL 已纳入同一份 `v5-timerxl` 正式矩阵：`timerxl-84m` 使用 v5 的 13 臂加两个 uni 臂，共 15 臂、9 任务、3 seeds、387 cells；截至 2026-09-21 02:18 已去重落盘 272 个 cell（seed0=135、seed1=126、seed2=11）。训练配方已冻结为 lr=1e-4、4 epochs、batch=32、cosine、weight_decay=0.01；当前正式 run 已完成 seed0/seed1 并进入 seed2，第三模型最终数值尚未封存。
- v5/v5-uni 的跨模型判断和 TimerXL 已落盘部分见 [ALL-RESULTS.md](ALL-RESULTS.md)。该总表严格区分已完成数值与 TimerXL 尚未完成的单元。

### 当前阶段判断

基于完整的 v5+v5-uni 结果，RAGS 对 rand 有总体正向倾向，但尚不能称为“非常稳定地优于 rand”：含 uni 的 RAGS 任务-模型-方法均值格中 112/180 优于 rand，逐 seed 配对胜 316/540；原始 v5 的 8 个 RAGS 臂为 100/144 均值格、267/432 seed 配对胜。相对 orig，含 uni 的 RAGS 有 132/180 个均值格不差于 orig，说明多数情况下不差，但仍不是保证。

在 rand 明显差于 orig 的 4 个完整模型×任务场景中，RAGS 至少一个臂达到或超过 orig 的有 2 个；合计为 6/40 个 RAGS 臂格，说明存在拉回能力，但不稳定。TimerXL 当前只作为部分正式证据，待 seed2 完成后再更新第三模型的最终判断。完整三模型、三张总表和“还需补什么”见 [ALL-RESULTS.md](ALL-RESULTS.md)。

相对其他基线，原始 v5 的 RAGS 8 臂在 18 个模型×任务场景中取最佳臂时赢过最佳非 RAGS 基线 11/18，平均改善 +0.41%；逐臂比较时相对 LESS 平均 -0.10%、相对 TSDS 平均 -0.21%，说明 RAGS 的优势主要不是全面压过所有基线，而是相对较弱的 rand、orig 和 LP-Mirror 在部分场景取得收益。详细比较见总表中的“RAGS 相对其他基线到底好多少”。

<!-- v5-notes:start -->
## 固定方法协议（V5 冻结方案）

| 方法 | 单位 | 聚合 | 竞争 | 温度 |
|---|---|---|---|---|
| rand | — | — | — | — |
| orig | — | — | — | — |
| less-main | — | — | global | 1/(2ln4)（Rank） |
| rags-block-linearSum | — | — | global | 1/(2ln4)（Rank） |
| rags-block-zeroClipSum | — | — | global | 1/(2ln4)（Rank） |
| rags-block-maxSum | — | — | global | 1/(2ln4)（Rank） |
| rags-block-expSum | — | — | global | 1/(2ln4)（Rank） |
| rags-traj-linearSum | — | — | global | 1/(2ln4)（Rank） |
| rags-traj-zeroClipSum | — | — | global | 1/(2ln4)（Rank） |
| rags-traj-maxSum | — | — | global | 1/(2ln4)（Rank） |
| rags-traj-expSum | — | — | global | 1/(2ln4)（Rank） |
| tsds | — | — | — | — |
| lp-mirror | — | — | — | .05（原生） |

Block/Traj 四种聚合均保留，未指定 RAGS 主方法。LESS-Main 采用参考轨迹加权后 max。
参考话语权只进入评分：轨迹 1/N；块 (1/N)×块合法起点占所属参考轨迹比例。Max 为 max(alpha*cos)，Linear 为 sum(alpha*cos)，ReLU 为 sum(alpha*relu(cos))，Exp 为 sum(alpha*exp(cos/1.0))，内层T=1.00。
LESS/RAGS 最终分数只保留对象排序及精确并列。冻结 F_T(x)=(1-exp(-x/T))/(1-exp(-1/T))，T=1/(2ln4)；每个对象获得其等宽排名区间的积分质量，并列组获得合并区间质量后组内等分。随后乘源对象合法窗口起点数量并归一化。前 50% 对象获得 80% 质量的约束作用于源长度先验之前，边界并列或奇数对象数需按冻结协议解释。没有最终 Softmax；LP-Mirror 和 TSDS 保持原生分布。
CRB 递归选择最小 kappa 区间，精确并列优先最右，实际探测取 floor，中点代表使用原始端点梯度线性插值，不追加 BP，无 P10，严格耗尽可用预算。轨迹代表保留范数后平均，再归一化进入匹配。
仅汇总独立 V5 正式 cells；orig 只计 seed0 一次，后续轮基线引用不重复计数。样本标准差使用 ddof=1，单 seed 显示 —；胜/seeds 仅使用同 seed 的 rand 配对，MASE 越低越好。

## 材料、执行与追溯

# 当前实验：V5

唯一配置：`configs/rags/main.json`。继承停止前 V4 的两模型、九共享任务、13 方法、seeds 0/1/2 和训练配方；V5 改动限定为 [冻结方法](method.md)。没有选定唯一 RAGS 主方法；Block/Traj 四种聚合均保留。三个 seed 按整轮完成屏障串行，轮内两模型并行。

V4 已按用户 2026-09-18 授权停止，停止记录位于既有结果目录 `STOPPED_FOR_DEVELOPMENT.json`；结果、检查点、原始梯度缓存保留。旧选集、CRB 重建资产、训练 cells 不能作为 V5 结果续跑。逐窗口梯度只有在模型、投影、精度、窗口及数据字段一致时才允许复用；新 CRB 和新概率必须重新构造。

## 唯一入口

在 `/root/TS3`、`rags` 分支执行：

```bash
python -m rags.experiments.formal_main_entry --config configs/rags/main.json --run /root/autodl-tmp/TS3/outputs/main-rank-volume --dry-run
python -m rags.experiments.prepare_suite --config configs/rags/main.json --run /root/autodl-tmp/TS3/outputs/main-rank-volume
python -m rags.experiments.full_pool_assets --run /root/autodl-tmp/TS3/outputs/main-rank-volume --student chronos-base
python -m rags.experiments.suite_report --run /root/autodl-tmp/TS3/outputs/main-rank-volume
```

准备入口默认不启用正式选择、训练、评测。用户实际决定启动正式实验后，在 run 的 `execution_control.json` 显式启用三项控制，再运行 `formal_main_entry`（不带 `--dry-run`）。2026-09-18 用户已明确授权启动，正式 V5 已于 2026-09-20 全部完成（666/666 cells，seeds 0/1/2 三轮屏障通过，supervisor exit_code=0），产物位于 `/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918`。科学代码版本 `6d3ead6`；执行优化参数独立记录在 run 的 `execution.json`，GPU 连续采样见 `gpu.csv`/`telemetry.json`。

各模型 worker 使用配置内的独立 Python 环境。输出默认在数据盘；源码临时备份和测试日志在仓库 `temp/v5-final/`。字段比对隔离缓存，不计算数据/模型/资产 hash。写缓存先同目录临时文件，再原子替换，单写者锁。选集 NPZ 保存原始分数、最终对象概率、窗口所有权和完整冻结协议，不保存全体参考×源配对矩阵。

## 验证

验证覆盖排序数值间隔去除、对象数量无关区间质量、精确并列、源窗口量先验、完整窗口所有权、冻结配置拒绝变更、无 P10/调度 RNG、精确预算、准备后零额外 BP、保存/抽样一致、旧资产拒绝、worker 训练与恢复、seed 轮次屏障及真实模型小规模训练/评测。实际通过数和运行产物见 `Progress.md`；开发样本结果不能说明正式效果。

旧 V1–V4 入口和版本后缀配置已清理，完整清理前源码备份是 `temp/v5-final/pre-cleanup-code.tar.gz`，不纳入 Git。历史结果和旧方法说明位于 `docs/archive/`。
## 2026-09-18 巡检责任交接：WSL tmux moni

用户明确要求本机停止巡检，由 SSH `wsl` 主机上的 `tmux moni` 中 Codex 接管巡检与总结。模型固定为 `gpt-5.6-luna`，推理强度 `high`。本机 heartbeat `v5` 已暂停，不重新启用。WSL 工作目录 `/home/rogers/temp/v5-monitor`；完整交接材料 `HANDOFF.md`，巡检记录 `monitor.log`，最新总结 `latest-summary.md`，接管确认 `ACK.json`，当前状态 `STATUS.json` / `monitor.json`。同一 Codex 任务每三十分钟续跑，全部 666 独立 cell 和三轮 seed 完成且报告核验后停止。

此交接只改变监督责任，不停止或重启实验。【2026-09-20 更新：V5 已 666/666 完成，巡检闭环；后续实验为 v5-uni，见 docs/UNI.md。】冻结科学版本、种子整轮屏障、轮内任务优先级、两模型并行、全部结果表按相对 rand 改善降序及产物保留边界不变。已有原生 checkpoint 保留协议不因交接改变。WSL 负责人每次巡检更新 MAIN 和 Progress，故障先定位，在既有授权内恢复并避免重复 cell，必要时报告用户。当前实验已完成。
<!-- v5-notes:end -->
