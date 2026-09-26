# MAIN-v3 — 主实验

- 更新时间：2026-09-18T00:38:04+08:00
- 配置：`configs/rags/main_v3_t005.json`；运行：`/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-t005-20260917`
- 所有使用采样温度的默认方法（lp-mirror / less-main / rags-main / rags-traj）统一 T=0.05。
- 九任务、3模型；矩阵 orig / rand / tsds / lp-mirror / less-linearSum / rags-block-linearSum / rags-traj-linearSum；seeds [0]。
- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，批量分裂规则不变。
- 本批仍为冻结reference-mean线性协议；旧主方法以-linearSum展示，尚无新聚合规则训练结果。
- 原生逐窗口梯度沿用已有字段校验缓存；V3 训练与评测独立；V4轮间复用静态概率和orig，训练seed逐轮变化。
- V1/V2 内容完整归档，继续参考：[MAIN-v1](archive/MAIN-v1.md)、[MAIN-v2](archive/MAIN-v2.md)。

完成 189/189 cells。

| 模型 | 已完成 cells | 当前状态 |
|---|---:|---|
| chronos-base | 63 | ////complete |
| timesfm-2.5 | 63 | ////complete |
| timemoe-200m | 63 | ////complete |
<!-- v3-patrol:start -->
## 巡检记录与解读（每 30 分钟自动更新）

- 巡检时间：2026-09-18T00:38:04+08:00（北京时间）；以实际 cells/state 为准。
- 总进度：189/189 cells；已授权并行推进。

### 结果（按任务分节，每节各模型一表；两列百分比分别为均值相对 orig 与 rand 的改善，正=更好；胜/n 为逐 seed 配对胜 rand）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-linearSum | 1 | 1.245335 | — | +27.91% | +11.92% | 1/1 |
| less-linearSum | 1 | 1.295673 | — | +25.00% | +8.36% | 1/1 |
| tsds | 1 | 1.302271 | — | +24.62% | +7.89% | 1/1 |
| rand | 1 | 1.413860 | — | +18.16% | +0.00% | — |
| rags-block-linearSum | 1 | 1.441458 | — | +16.56% | -1.95% | 0/1 |
| orig | 1 | 1.727507 | — | +0.00% | -22.18% | — |
| lp-mirror | 1 | 2.060057 | — | -19.25% | -45.70% | 0/1 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-linearSum | 1 | 0.689784 | — | +21.79% | +5.07% | 1/1 |
| less-linearSum | 1 | 0.692374 | — | +21.49% | +4.72% | 1/1 |
| rags-block-linearSum | 1 | 0.713524 | — | +19.09% | +1.81% | 1/1 |
| tsds | 1 | 0.721514 | — | +18.19% | +0.71% | 1/1 |
| rand | 1 | 0.726650 | — | +17.61% | +0.00% | — |
| lp-mirror | 1 | 0.877396 | — | +0.51% | -20.75% | 0/1 |
| orig | 1 | 0.881911 | — | +0.00% | -21.37% | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 1.065605 | — | +0.00% | +10.59% | — |
| lp-mirror | 1 | 1.075979 | — | -0.97% | +9.72% | 1/1 |
| rand | 1 | 1.191778 | — | -11.84% | +0.00% | — |
| rags-block-linearSum | 1 | 1.208490 | — | -13.41% | -1.40% | 0/1 |
| rags-traj-linearSum | 1 | 1.303941 | — | -22.37% | -9.41% | 0/1 |
| less-linearSum | 1 | 1.330100 | — | -24.82% | -11.61% | 0/1 |
| tsds | 1 | 1.381358 | — | -29.63% | -15.91% | 0/1 |

### time_azure_i_cpu_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-linearSum | 1 | 1.202050 | — | +9.73% | +4.87% | 1/1 |
| tsds | 1 | 1.220121 | — | +8.37% | +3.44% | 1/1 |
| rags-traj-linearSum | 1 | 1.232285 | — | +7.46% | +2.48% | 1/1 |
| lp-mirror | 1 | 1.242592 | — | +6.69% | +1.66% | 1/1 |
| rags-block-linearSum | 1 | 1.245812 | — | +6.44% | +1.41% | 1/1 |
| rand | 1 | 1.263619 | — | +5.11% | +0.00% | — |
| orig | 1 | 1.331634 | — | +0.00% | -5.38% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 1 | 0.761271 | — | +0.33% | +0.00% | — |
| lp-mirror | 1 | 0.762703 | — | +0.15% | -0.19% | 0/1 |
| orig | 1 | 0.763820 | — | +0.00% | -0.33% | — |
| tsds | 1 | 0.767492 | — | -0.48% | -0.82% | 0/1 |
| rags-traj-linearSum | 1 | 0.768144 | — | -0.57% | -0.90% | 0/1 |
| rags-block-linearSum | 1 | 0.771433 | — | -1.00% | -1.33% | 0/1 |
| less-linearSum | 1 | 0.775556 | — | -1.54% | -1.88% | 0/1 |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.909997 | — | +0.00% | +17.20% | — |
| rand | 1 | 1.099002 | — | -20.77% | +0.00% | — |
| lp-mirror | 1 | 1.156283 | — | -27.06% | -5.21% | 0/1 |
| rags-block-linearSum | 1 | 1.170832 | — | -28.66% | -6.54% | 0/1 |
| rags-traj-linearSum | 1 | 1.203653 | — | -32.27% | -9.52% | 0/1 |
| less-linearSum | 1 | 1.206149 | — | -32.54% | -9.75% | 0/1 |
| tsds | 1 | 1.212800 | — | -33.28% | -10.35% | 0/1 |

### time_darwin_water_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-linearSum | 1 | 1.656504 | — | +0.46% | +0.18% | 1/1 |
| rand | 1 | 1.659562 | — | +0.27% | +0.00% | — |
| tsds | 1 | 1.663311 | — | +0.05% | -0.23% | 0/1 |
| orig | 1 | 1.664086 | — | +0.00% | -0.27% | — |
| less-linearSum | 1 | 1.666030 | — | -0.12% | -0.39% | 0/1 |
| rags-block-linearSum | 1 | 1.667221 | — | -0.19% | -0.46% | 0/1 |
| lp-mirror | 1 | 1.777741 | — | -6.83% | -7.12% | 0/1 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 1 | 1.009940 | — | +5.59% | +1.61% | 1/1 |
| less-linearSum | 1 | 1.022888 | — | +4.38% | +0.34% | 1/1 |
| rags-traj-linearSum | 1 | 1.022924 | — | +4.37% | +0.34% | 1/1 |
| rand | 1 | 1.026424 | — | +4.05% | +0.00% | — |
| tsds | 1 | 1.042631 | — | +2.53% | -1.58% | 0/1 |
| lp-mirror | 1 | 1.069587 | — | +0.01% | -4.21% | 0/1 |
| orig | 1 | 1.069706 | — | +0.00% | -4.22% | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 1.139189 | — | +0.00% | +17.47% | — |
| lp-mirror | 1 | 1.154240 | — | -1.32% | +16.38% | 1/1 |
| less-linearSum | 1 | 1.313450 | — | -15.30% | +4.84% | 1/1 |
| rags-traj-linearSum | 1 | 1.325334 | — | -16.34% | +3.98% | 1/1 |
| rags-block-linearSum | 1 | 1.370825 | — | -20.33% | +0.68% | 1/1 |
| rand | 1 | 1.380258 | — | -21.16% | +0.00% | — |
| tsds | 1 | 1.453512 | — | -27.59% | -5.31% | 0/1 |

### time_finland_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| tsds | 1 | 1.016375 | — | +50.56% | +12.64% | 1/1 |
| rags-traj-linearSum | 1 | 1.143717 | — | +44.36% | +1.70% | 1/1 |
| rand | 1 | 1.163479 | — | +43.40% | +0.00% | — |
| rags-block-linearSum | 1 | 1.225996 | — | +40.36% | -5.37% | 0/1 |
| less-linearSum | 1 | 1.275432 | — | +37.95% | -9.62% | 0/1 |
| lp-mirror | 1 | 1.455847 | — | +29.18% | -25.13% | 0/1 |
| orig | 1 | 2.055572 | — | +0.00% | -76.67% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 1 | 0.612023 | — | +1.58% | +1.17% | 1/1 |
| rags-traj-linearSum | 1 | 0.612115 | — | +1.57% | +1.16% | 1/1 |
| tsds | 1 | 0.614404 | — | +1.20% | +0.79% | 1/1 |
| less-linearSum | 1 | 0.618772 | — | +0.50% | +0.08% | 1/1 |
| rand | 1 | 0.619276 | — | +0.42% | +0.00% | — |
| orig | 1 | 0.621873 | — | +0.00% | -0.42% | — |
| lp-mirror | 1 | 0.625917 | — | -0.65% | -1.07% | 0/1 |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.840590 | — | +0.00% | +3.16% | — |
| lp-mirror | 1 | 0.854423 | — | -1.65% | +1.56% | 1/1 |
| less-linearSum | 1 | 0.861845 | — | -2.53% | +0.71% | 1/1 |
| rags-block-linearSum | 1 | 0.862970 | — | -2.66% | +0.58% | 1/1 |
| rand | 1 | 0.867992 | — | -3.26% | +0.00% | — |
| rags-traj-linearSum | 1 | 0.896354 | — | -6.63% | -3.27% | 0/1 |
| tsds | 1 | 0.987759 | — | -17.51% | -13.80% | 0/1 |

### time_eweld_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 1 | 1.139507 | — | +23.83% | +2.62% | 1/1 |
| rand | 1 | 1.170147 | — | +21.78% | +0.00% | — |
| less-linearSum | 1 | 1.253983 | — | +16.18% | -7.16% | 0/1 |
| tsds | 1 | 1.293124 | — | +13.56% | -10.51% | 0/1 |
| rags-block-linearSum | 1 | 1.421912 | — | +4.95% | -21.52% | 0/1 |
| rags-traj-linearSum | 1 | 1.447604 | — | +3.23% | -23.71% | 0/1 |
| orig | 1 | 1.495984 | — | +0.00% | -27.85% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-linearSum | 1 | 0.721916 | — | +0.87% | +0.42% | 1/1 |
| rags-block-linearSum | 1 | 0.722485 | — | +0.79% | +0.34% | 1/1 |
| rags-traj-linearSum | 1 | 0.722496 | — | +0.79% | +0.34% | 1/1 |
| tsds | 1 | 0.724468 | — | +0.52% | +0.07% | 1/1 |
| rand | 1 | 0.724941 | — | +0.45% | +0.00% | — |
| orig | 1 | 0.728248 | — | +0.00% | -0.46% | — |
| lp-mirror | 1 | 0.732160 | — | -0.54% | -1.00% | 0/1 |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.835714 | — | +0.00% | +4.12% | — |
| rags-block-linearSum | 1 | 0.838264 | — | -0.31% | +3.83% | 1/1 |
| rags-traj-linearSum | 1 | 0.848054 | — | -1.48% | +2.70% | 1/1 |
| less-linearSum | 1 | 0.864023 | — | -3.39% | +0.87% | 1/1 |
| rand | 1 | 0.871619 | — | -4.30% | +0.00% | — |
| lp-mirror | 1 | 0.871655 | — | -4.30% | -0.00% | 0/1 |
| tsds | 1 | 0.964504 | — | -15.41% | -10.66% | 0/1 |

### time_cphl_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-linearSum | 1 | 1.485280 | — | +3.48% | +3.94% | 1/1 |
| rags-traj-linearSum | 1 | 1.499640 | — | +2.54% | +3.01% | 1/1 |
| tsds | 1 | 1.521871 | — | +1.10% | +1.58% | 1/1 |
| rags-block-linearSum | 1 | 1.530302 | — | +0.55% | +1.03% | 1/1 |
| lp-mirror | 1 | 1.531299 | — | +0.49% | +0.97% | 1/1 |
| orig | 1 | 1.538772 | — | +0.00% | +0.48% | — |
| rand | 1 | 1.546225 | — | -0.48% | +0.00% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 1 | 1.333771 | — | +2.37% | +0.90% | 1/1 |
| less-linearSum | 1 | 1.343836 | — | +1.64% | +0.15% | 1/1 |
| rand | 1 | 1.345847 | — | +1.49% | +0.00% | — |
| lp-mirror | 1 | 1.349813 | — | +1.20% | -0.29% | 0/1 |
| rags-traj-linearSum | 1 | 1.356075 | — | +0.74% | -0.76% | 0/1 |
| tsds | 1 | 1.362809 | — | +0.25% | -1.26% | 0/1 |
| orig | 1 | 1.366210 | — | +0.00% | -1.51% | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 1.435433 | — | +0.00% | +2.93% | — |
| rags-traj-linearSum | 1 | 1.444709 | — | -0.65% | +2.30% | 1/1 |
| less-linearSum | 1 | 1.453958 | — | -1.29% | +1.67% | 1/1 |
| lp-mirror | 1 | 1.458943 | — | -1.64% | +1.34% | 1/1 |
| tsds | 1 | 1.466999 | — | -2.20% | +0.79% | 1/1 |
| rand | 1 | 1.478726 | — | -3.02% | +0.00% | — |
| rags-block-linearSum | 1 | 1.478792 | — | -3.02% | -0.00% | 0/1 |

### time_nem_price_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-linearSum | 1 | 1.037644 | — | +2.31% | +2.82% | 1/1 |
| rags-traj-linearSum | 1 | 1.042492 | — | +1.86% | +2.36% | 1/1 |
| rags-block-linearSum | 1 | 1.046400 | — | +1.49% | +2.00% | 1/1 |
| tsds | 1 | 1.047329 | — | +1.40% | +1.91% | 1/1 |
| lp-mirror | 1 | 1.048575 | — | +1.28% | +1.79% | 1/1 |
| orig | 1 | 1.062202 | — | +0.00% | +0.52% | — |
| rand | 1 | 1.067739 | — | -0.52% | +0.00% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-linearSum | 1 | 0.708308 | — | +5.46% | +1.26% | 1/1 |
| rags-traj-linearSum | 1 | 0.712027 | — | +4.96% | +0.75% | 1/1 |
| rand | 1 | 0.717374 | — | +4.25% | +0.00% | — |
| rags-block-linearSum | 1 | 0.718022 | — | +4.16% | -0.09% | 0/1 |
| tsds | 1 | 0.721010 | — | +3.76% | -0.51% | 0/1 |
| orig | 1 | 0.749205 | — | +0.00% | -4.44% | — |
| lp-mirror | 1 | 0.754650 | — | -0.73% | -5.20% | 0/1 |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.788303 | — | +0.00% | +10.13% | — |
| lp-mirror | 1 | 0.813672 | — | -3.22% | +7.24% | 1/1 |
| rand | 1 | 0.877180 | — | -11.27% | +0.00% | — |
| rags-block-linearSum | 1 | 0.879736 | — | -11.60% | -0.29% | 0/1 |
| rags-traj-linearSum | 1 | 0.942162 | — | -19.52% | -7.41% | 0/1 |
| tsds | 1 | 0.960980 | — | -21.90% | -9.55% | 0/1 |
| less-linearSum | 1 | 0.976498 | — | -23.87% | -11.32% | 0/1 |

### time_solar_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 1 | 1.102482 | — | +9.88% | +2.81% | 1/1 |
| less-linearSum | 1 | 1.118182 | — | +8.60% | +1.43% | 1/1 |
| rand | 1 | 1.134373 | — | +7.27% | +0.00% | — |
| rags-traj-linearSum | 1 | 1.206659 | — | +1.36% | -6.37% | 0/1 |
| orig | 1 | 1.223355 | — | +0.00% | -7.84% | — |
| tsds | 1 | 1.273342 | — | -4.09% | -12.25% | 0/1 |
| lp-mirror | 1 | 1.311835 | — | -7.23% | -15.64% | 0/1 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 1 | 1.103397 | — | +2.29% | +2.42% | 1/1 |
| rags-traj-linearSum | 1 | 1.119860 | — | +0.84% | +0.97% | 1/1 |
| orig | 1 | 1.129299 | — | +0.00% | +0.13% | — |
| rand | 1 | 1.130794 | — | -0.13% | +0.00% | — |
| less-linearSum | 1 | 1.131229 | — | -0.17% | -0.04% | 0/1 |
| tsds | 1 | 1.131475 | — | -0.19% | -0.06% | 0/1 |
| rags-block-linearSum | 1 | 1.153879 | — | -2.18% | -2.04% | 0/1 |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 1 | 1.123140 | — | +2.32% | +0.00% | — |
| lp-mirror | 1 | 1.127984 | — | +1.90% | -0.43% | 0/1 |
| rags-block-linearSum | 1 | 1.137131 | — | +1.11% | -1.25% | 0/1 |
| orig | 1 | 1.149861 | — | +0.00% | -2.38% | — |
| tsds | 1 | 1.152869 | — | -0.26% | -2.65% | 0/1 |
| rags-traj-linearSum | 1 | 1.160453 | — | -0.92% | -3.32% | 0/1 |
| less-linearSum | 1 | 1.182440 | — | -2.83% | -5.28% | 0/1 |

### time_port_activity_d

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.882133 | — | +0.00% | +1.54% | — |
| lp-mirror | 1 | 0.885762 | — | -0.41% | +1.14% | 1/1 |
| less-linearSum | 1 | 0.894835 | — | -1.44% | +0.13% | 1/1 |
| rags-block-linearSum | 1 | 0.895667 | — | -1.53% | +0.03% | 1/1 |
| rand | 1 | 0.895971 | — | -1.57% | +0.00% | — |
| rags-traj-linearSum | 1 | 0.896399 | — | -1.62% | -0.05% | 0/1 |
| tsds | 1 | 0.911446 | — | -3.32% | -1.73% | 0/1 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.716205 | — | +0.00% | +1.02% | — |
| lp-mirror | 1 | 0.722498 | — | -0.88% | +0.15% | 1/1 |
| rand | 1 | 0.723588 | — | -1.03% | +0.00% | — |
| tsds | 1 | 0.736563 | — | -2.84% | -1.79% | 0/1 |
| rags-block-linearSum | 1 | 0.741228 | — | -3.49% | -2.44% | 0/1 |
| rags-traj-linearSum | 1 | 0.752374 | — | -5.05% | -3.98% | 0/1 |
| less-linearSum | 1 | 0.754699 | — | -5.37% | -4.30% | 0/1 |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 1 | 0.751227 | — | +2.13% | +0.00% | — |
| rags-block-linearSum | 1 | 0.760421 | — | +0.93% | -1.22% | 0/1 |
| tsds | 1 | 0.761662 | — | +0.77% | -1.39% | 0/1 |
| lp-mirror | 1 | 0.763467 | — | +0.53% | -1.63% | 0/1 |
| orig | 1 | 0.767551 | — | +0.00% | -2.17% | — |
| rags-traj-linearSum | 1 | 0.791824 | — | -3.16% | -5.40% | 0/1 |
| less-linearSum | 1 | 0.801004 | — | -4.36% | -6.63% | 0/1 |

**解读**：

以本版真实落盘结果更新；没有结果时不作效果判断。

<!-- v3-patrol:end -->

| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |
|---|---|---|---:|---:|---:|
| timemoe-200m | time_darwin_water_15t | orig | 1 | 1.139189 | — |
| timemoe-200m | time_azure_i_cpu_5t | orig | 1 | 0.909997 | — |
| timemoe-200m | time_darwin_water_15t | lp-mirror | 1 | 1.154240 | — |
| chronos-base | time_finland_15t | tsds | 1 | 1.016375 | — |
| chronos-base | time_nem_demand_5t | rags-traj-linearSum | 1 | 1.245335 | — |
| timemoe-200m | time_nem_demand_5t | orig | 1 | 1.065605 | — |
| timemoe-200m | time_nem_price_5t | orig | 1 | 0.788303 | — |
| timemoe-200m | time_nem_demand_5t | lp-mirror | 1 | 1.075979 | — |
| chronos-base | time_nem_demand_5t | less-linearSum | 1 | 1.295673 | — |
| chronos-base | time_nem_demand_5t | tsds | 1 | 1.302271 | — |
| timemoe-200m | time_nem_price_5t | lp-mirror | 1 | 0.813672 | — |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-linearSum | 1 | 0.689784 | — |
| chronos-base | time_azure_i_cpu_5t | less-linearSum | 1 | 1.202050 | — |
| timemoe-200m | time_darwin_water_15t | less-linearSum | 1 | 1.313450 | — |
| timesfm-2.5 | time_nem_demand_5t | less-linearSum | 1 | 0.692374 | — |
| timemoe-200m | time_eweld_15t | orig | 1 | 0.835714 | — |
| timemoe-200m | time_darwin_water_15t | rags-traj-linearSum | 1 | 1.325334 | — |
| chronos-base | time_cphl_h | less-linearSum | 1 | 1.485280 | — |
| timemoe-200m | time_eweld_15t | rags-block-linearSum | 1 | 0.838264 | — |
| chronos-base | time_azure_i_cpu_5t | tsds | 1 | 1.220121 | — |
| timemoe-200m | time_finland_15t | orig | 1 | 0.840590 | — |
| chronos-base | time_cphl_h | rags-traj-linearSum | 1 | 1.499640 | — |
| timemoe-200m | time_cphl_h | orig | 1 | 1.435433 | — |
| chronos-base | time_nem_price_5t | less-linearSum | 1 | 1.037644 | — |
| chronos-base | time_solar_h | rags-block-linearSum | 1 | 1.102482 | — |
| timemoe-200m | time_eweld_15t | rags-traj-linearSum | 1 | 0.848054 | — |
| chronos-base | time_eweld_15t | lp-mirror | 1 | 1.139507 | — |
| chronos-base | time_azure_i_cpu_5t | rags-traj-linearSum | 1 | 1.232285 | — |
| timesfm-2.5 | time_solar_h | lp-mirror | 1 | 1.103397 | — |
| chronos-base | time_nem_price_5t | rags-traj-linearSum | 1 | 1.042492 | — |
| timemoe-200m | time_cphl_h | rags-traj-linearSum | 1 | 1.444709 | — |
| chronos-base | time_nem_price_5t | rags-block-linearSum | 1 | 1.046400 | — |
| chronos-base | time_nem_price_5t | tsds | 1 | 1.047329 | — |
| timesfm-2.5 | time_nem_demand_5t | rags-block-linearSum | 1 | 0.713524 | — |
| chronos-base | time_nem_price_5t | lp-mirror | 1 | 1.048575 | — |
| chronos-base | time_finland_15t | rags-traj-linearSum | 1 | 1.143717 | — |
| timemoe-200m | time_cphl_h | less-linearSum | 1 | 1.453958 | — |
| chronos-base | time_azure_i_cpu_5t | lp-mirror | 1 | 1.242592 | — |
| timesfm-2.5 | time_darwin_water_15t | rags-block-linearSum | 1 | 1.009940 | — |
| chronos-base | time_cphl_h | tsds | 1 | 1.521871 | — |
| timemoe-200m | time_finland_15t | lp-mirror | 1 | 0.854423 | — |
| chronos-base | time_port_activity_d | orig | 1 | 0.882133 | — |
| chronos-base | time_solar_h | less-linearSum | 1 | 1.118182 | — |
| chronos-base | time_azure_i_cpu_5t | rags-block-linearSum | 1 | 1.245812 | — |
| timemoe-200m | time_cphl_h | lp-mirror | 1 | 1.458943 | — |
| timesfm-2.5 | time_nem_price_5t | less-linearSum | 1 | 0.708308 | — |
| timesfm-2.5 | time_finland_15t | rags-block-linearSum | 1 | 0.612023 | — |
| timesfm-2.5 | time_finland_15t | rags-traj-linearSum | 1 | 0.612115 | — |
| chronos-base | time_port_activity_d | lp-mirror | 1 | 0.885762 | — |
| chronos-base | time_cphl_h | rags-block-linearSum | 1 | 1.530302 | — |
| timesfm-2.5 | time_port_activity_d | orig | 1 | 0.716205 | — |
| timesfm-2.5 | time_solar_h | rags-traj-linearSum | 1 | 1.119860 | — |
| chronos-base | time_cphl_h | lp-mirror | 1 | 1.531299 | — |
| timesfm-2.5 | time_cphl_h | rags-block-linearSum | 1 | 1.333771 | — |
| timemoe-200m | time_eweld_15t | less-linearSum | 1 | 0.864023 | — |
| timemoe-200m | time_cphl_h | tsds | 1 | 1.466999 | — |
| timesfm-2.5 | time_finland_15t | tsds | 1 | 0.614404 | — |
| timesfm-2.5 | time_nem_price_5t | rags-traj-linearSum | 1 | 0.712027 | — |
| timemoe-200m | time_finland_15t | less-linearSum | 1 | 0.861845 | — |
| timesfm-2.5 | time_nem_demand_5t | tsds | 1 | 0.721514 | — |
| timemoe-200m | time_darwin_water_15t | rags-block-linearSum | 1 | 1.370825 | — |
| timemoe-200m | time_finland_15t | rags-block-linearSum | 1 | 0.862970 | — |
| chronos-base | time_nem_price_5t | orig | 1 | 1.062202 | — |
| chronos-base | time_cphl_h | orig | 1 | 1.538772 | — |
| timesfm-2.5 | time_eweld_15t | less-linearSum | 1 | 0.721916 | — |
| timesfm-2.5 | time_darwin_water_15t | less-linearSum | 1 | 1.022888 | — |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-linearSum | 1 | 1.022924 | — |
| timesfm-2.5 | time_eweld_15t | rags-block-linearSum | 1 | 0.722485 | — |
| timesfm-2.5 | time_eweld_15t | rags-traj-linearSum | 1 | 0.722496 | — |
| chronos-base | time_darwin_water_15t | rags-traj-linearSum | 1 | 1.656504 | — |
| timesfm-2.5 | time_port_activity_d | lp-mirror | 1 | 0.722498 | — |
| timesfm-2.5 | time_cphl_h | less-linearSum | 1 | 1.343836 | — |
| timesfm-2.5 | time_solar_h | orig | 1 | 1.129299 | — |
| chronos-base | time_port_activity_d | less-linearSum | 1 | 0.894835 | — |
| timesfm-2.5 | time_finland_15t | less-linearSum | 1 | 0.618772 | — |
| timesfm-2.5 | time_eweld_15t | tsds | 1 | 0.724468 | — |
| chronos-base | time_port_activity_d | rags-block-linearSum | 1 | 0.895667 | — |
| chronos-base | time_azure_i_cpu_5t | rand | 1 | 1.263619 | — |
| chronos-base | time_cphl_h | rand | 1 | 1.546225 | — |
| chronos-base | time_darwin_water_15t | rand | 1 | 1.659562 | — |
| chronos-base | time_eweld_15t | rand | 1 | 1.170147 | — |
| chronos-base | time_finland_15t | rand | 1 | 1.163479 | — |
| chronos-base | time_nem_demand_5t | rand | 1 | 1.413860 | — |
| chronos-base | time_nem_price_5t | rand | 1 | 1.067739 | — |
| chronos-base | time_port_activity_d | rand | 1 | 0.895971 | — |
| chronos-base | time_solar_h | rand | 1 | 1.134373 | — |
| timemoe-200m | time_azure_i_cpu_5t | rand | 1 | 1.099002 | — |
| timemoe-200m | time_cphl_h | rand | 1 | 1.478726 | — |
| timemoe-200m | time_darwin_water_15t | rand | 1 | 1.380258 | — |
| timemoe-200m | time_eweld_15t | rand | 1 | 0.871619 | — |
| timemoe-200m | time_finland_15t | rand | 1 | 0.867992 | — |
| timemoe-200m | time_nem_demand_5t | rand | 1 | 1.191778 | — |
| timemoe-200m | time_nem_price_5t | rand | 1 | 0.877180 | — |
| timemoe-200m | time_port_activity_d | rand | 1 | 0.751227 | — |
| timemoe-200m | time_solar_h | rand | 1 | 1.123140 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | rand | 1 | 0.761271 | — |
| timesfm-2.5 | time_cphl_h | rand | 1 | 1.345847 | — |
| timesfm-2.5 | time_darwin_water_15t | rand | 1 | 1.026424 | — |
| timesfm-2.5 | time_eweld_15t | rand | 1 | 0.724941 | — |
| timesfm-2.5 | time_finland_15t | rand | 1 | 0.619276 | — |
| timesfm-2.5 | time_nem_demand_5t | rand | 1 | 0.726650 | — |
| timesfm-2.5 | time_nem_price_5t | rand | 1 | 0.717374 | — |
| timesfm-2.5 | time_port_activity_d | rand | 1 | 0.723588 | — |
| timesfm-2.5 | time_solar_h | rand | 1 | 1.130794 | — |
| timemoe-200m | time_eweld_15t | lp-mirror | 1 | 0.871655 | — |
| timemoe-200m | time_cphl_h | rags-block-linearSum | 1 | 1.478792 | — |
| timesfm-2.5 | time_solar_h | less-linearSum | 1 | 1.131229 | — |
| chronos-base | time_port_activity_d | rags-traj-linearSum | 1 | 0.896399 | — |
| timesfm-2.5 | time_solar_h | tsds | 1 | 1.131475 | — |
| timesfm-2.5 | time_nem_price_5t | rags-block-linearSum | 1 | 0.718022 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | lp-mirror | 1 | 0.762703 | — |
| chronos-base | time_darwin_water_15t | tsds | 1 | 1.663311 | — |
| chronos-base | time_darwin_water_15t | orig | 1 | 1.664086 | — |
| timemoe-200m | time_nem_price_5t | rags-block-linearSum | 1 | 0.879736 | — |
| timesfm-2.5 | time_cphl_h | lp-mirror | 1 | 1.349813 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | orig | 1 | 0.763820 | — |
| chronos-base | time_darwin_water_15t | less-linearSum | 1 | 1.666030 | — |
| timesfm-2.5 | time_finland_15t | orig | 1 | 0.621873 | — |
| timemoe-200m | time_solar_h | lp-mirror | 1 | 1.127984 | — |
| timesfm-2.5 | time_eweld_15t | orig | 1 | 0.728248 | — |
| chronos-base | time_darwin_water_15t | rags-block-linearSum | 1 | 1.667221 | — |
| timesfm-2.5 | time_nem_price_5t | tsds | 1 | 0.721010 | — |
| timesfm-2.5 | time_cphl_h | rags-traj-linearSum | 1 | 1.356075 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | tsds | 1 | 0.767492 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-linearSum | 1 | 0.768144 | — |
| timesfm-2.5 | time_eweld_15t | lp-mirror | 1 | 0.732160 | — |
| timesfm-2.5 | time_finland_15t | lp-mirror | 1 | 0.625917 | — |
| timemoe-200m | time_port_activity_d | rags-block-linearSum | 1 | 0.760421 | — |
| timemoe-200m | time_solar_h | rags-block-linearSum | 1 | 1.137131 | — |
| timesfm-2.5 | time_cphl_h | tsds | 1 | 1.362809 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-linearSum | 1 | 0.771433 | — |
| timemoe-200m | time_port_activity_d | tsds | 1 | 0.761662 | — |
| timemoe-200m | time_nem_demand_5t | rags-block-linearSum | 1 | 1.208490 | — |
| timesfm-2.5 | time_cphl_h | orig | 1 | 1.366210 | — |
| timesfm-2.5 | time_darwin_water_15t | tsds | 1 | 1.042631 | — |
| timemoe-200m | time_port_activity_d | lp-mirror | 1 | 0.763467 | — |
| chronos-base | time_port_activity_d | tsds | 1 | 0.911446 | — |
| timesfm-2.5 | time_port_activity_d | tsds | 1 | 0.736563 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | less-linearSum | 1 | 0.775556 | — |
| chronos-base | time_nem_demand_5t | rags-block-linearSum | 1 | 1.441458 | — |
| timesfm-2.5 | time_solar_h | rags-block-linearSum | 1 | 1.153879 | — |
| timemoe-200m | time_port_activity_d | orig | 1 | 0.767551 | — |
| timemoe-200m | time_solar_h | orig | 1 | 1.149861 | — |
| timesfm-2.5 | time_port_activity_d | rags-block-linearSum | 1 | 0.741228 | — |
| timemoe-200m | time_solar_h | tsds | 1 | 1.152869 | — |
| timemoe-200m | time_finland_15t | rags-traj-linearSum | 1 | 0.896354 | — |
| timemoe-200m | time_solar_h | rags-traj-linearSum | 1 | 1.160453 | — |
| timesfm-2.5 | time_port_activity_d | rags-traj-linearSum | 1 | 0.752374 | — |
| timesfm-2.5 | time_darwin_water_15t | lp-mirror | 1 | 1.069587 | — |
| timesfm-2.5 | time_darwin_water_15t | orig | 1 | 1.069706 | — |
| timesfm-2.5 | time_port_activity_d | less-linearSum | 1 | 0.754699 | — |
| timesfm-2.5 | time_nem_price_5t | orig | 1 | 0.749205 | — |
| timesfm-2.5 | time_nem_price_5t | lp-mirror | 1 | 0.754650 | — |
| timemoe-200m | time_azure_i_cpu_5t | lp-mirror | 1 | 1.156283 | — |
| timemoe-200m | time_solar_h | less-linearSum | 1 | 1.182440 | — |
| timemoe-200m | time_darwin_water_15t | tsds | 1 | 1.453512 | — |
| chronos-base | time_finland_15t | rags-block-linearSum | 1 | 1.225996 | — |
| chronos-base | time_azure_i_cpu_5t | orig | 1 | 1.331634 | — |
| timemoe-200m | time_port_activity_d | rags-traj-linearSum | 1 | 0.791824 | — |
| chronos-base | time_solar_h | rags-traj-linearSum | 1 | 1.206659 | — |
| timemoe-200m | time_azure_i_cpu_5t | rags-block-linearSum | 1 | 1.170832 | — |
| timemoe-200m | time_port_activity_d | less-linearSum | 1 | 0.801004 | — |
| chronos-base | time_darwin_water_15t | lp-mirror | 1 | 1.777741 | — |
| chronos-base | time_eweld_15t | less-linearSum | 1 | 1.253983 | — |
| timemoe-200m | time_nem_price_5t | rags-traj-linearSum | 1 | 0.942162 | — |
| chronos-base | time_solar_h | orig | 1 | 1.223355 | — |
| timemoe-200m | time_nem_demand_5t | rags-traj-linearSum | 1 | 1.303941 | — |
| timemoe-200m | time_azure_i_cpu_5t | rags-traj-linearSum | 1 | 1.203653 | — |
| timemoe-200m | time_nem_price_5t | tsds | 1 | 0.960980 | — |
| chronos-base | time_finland_15t | less-linearSum | 1 | 1.275432 | — |
| timemoe-200m | time_azure_i_cpu_5t | less-linearSum | 1 | 1.206149 | — |
| timemoe-200m | time_azure_i_cpu_5t | tsds | 1 | 1.212800 | — |
| chronos-base | time_eweld_15t | tsds | 1 | 1.293124 | — |
| timemoe-200m | time_eweld_15t | tsds | 1 | 0.964504 | — |
| timemoe-200m | time_nem_price_5t | less-linearSum | 1 | 0.976498 | — |
| timemoe-200m | time_nem_demand_5t | less-linearSum | 1 | 1.330100 | — |
| chronos-base | time_solar_h | tsds | 1 | 1.273342 | — |
| timemoe-200m | time_finland_15t | tsds | 1 | 0.987759 | — |
| chronos-base | time_solar_h | lp-mirror | 1 | 1.311835 | — |
| timemoe-200m | time_nem_demand_5t | tsds | 1 | 1.381358 | — |
| timesfm-2.5 | time_nem_demand_5t | lp-mirror | 1 | 0.877396 | — |
| timesfm-2.5 | time_nem_demand_5t | orig | 1 | 0.881911 | — |
| chronos-base | time_eweld_15t | rags-block-linearSum | 1 | 1.421912 | — |
| chronos-base | time_nem_demand_5t | orig | 1 | 1.727507 | — |
| chronos-base | time_eweld_15t | rags-traj-linearSum | 1 | 1.447604 | — |
| chronos-base | time_finland_15t | lp-mirror | 1 | 1.455847 | — |
| chronos-base | time_eweld_15t | orig | 1 | 1.495984 | — |
| chronos-base | time_nem_demand_5t | lp-mirror | 1 | 2.060057 | — |
| chronos-base | time_finland_15t | orig | 1 | 2.055572 | — |

<!-- v3-notes:start -->
## V3 固定方法协议

| 方法 | 评分单位 | 参考权重 | 竞争与概率 |
|---|---|---|---|
| orig | 不选择；预训练模型直接评测 | 不适用 | 不训练 |
| rand | 全部合法起点 | 不适用 | 延续均匀窗口基线 |
| tsds | 原生 TSDS | 原方法不变 | 原生分布不变 |
| lp-mirror | 窗口 | 原方法不变 | 延续 V2 lpv_t025 的 raw-value 分布 |
| less-main | 轨迹 | 每参考轨迹 1/N | 无门控、Global、平均余弦、Softmax |
| rags-main | 块 | (1/N)×块合法起点占所属轨迹比例 | 无门控、Global、两级评委、Softmax |
| rags-traj | 轨迹 | 每参考轨迹 1/N | 无门控、Global、平均余弦、Softmax |

N 指当前参考数据集的参考轨迹数；参考话语权总和为 1。权重只用于分数相加，
最终 P_j=softmax(S_j/T)，不乘源单位长度，不对 Softmax 输出再次分配参考预算。
less-main/rags-main/rags-traj 以及 lp-mirror 统一 T005（0.05）；所有使用采样温度的默认方法均为 T005。旧版所有方法展示名加 -T025 后缀，新版使用原定名称；orig/rand/tsds 无此温度，历史后缀仅标识旧协议。

less-main 对按 rounded linspace 均匀选取的原生窗口投影梯度做 FP32 算术平均，
再分别归一化轨迹代表并算余弦。rags-traj 同样直接平均实际计算的原生 CRB 锚点梯度，
不使用旧版的全窗口线性重构均值；每轨迹梯度数量核对同一配置预算，唯一设计差异是取点位置。
rags-main 保留块级线性中点代表。CRB 覆盖阶段最长区间二分，误差阶段积分风险优先，
分裂阶段允许批量最小模长 frontier，耗尽可用预算，p10 只审计，不筛分裂区间、不丢窗。

源块等权竞争不保证源轨迹等总概率：同分且块数不同的源轨迹可累积不同质量。
参考等权及长度先验删除是适应目标 Benchmark 评估单位的设计；用户在真实应用中仍主张按长度/时间点频次加权。

正式 V3 名称只使用本矩阵；旧 Global/Gateless/T025 别名不进入结果表。
不再另设与主公式相同的 Global 臂。旧概率/训练 cells/checkpoints 不复用，
仅复用逐窗口原生梯度、锚点及块资产；新评分、概率版本由 worker 验证。

## 材料、执行与追溯

- 已有三模型×九任务参考资产 27/27 完成；当前方法窗口并集缓存复用，未计算全部合法窗口梯度。
- grad_cache 使用字段校验和单写者锁；新选集/cells/checkpoints 位于独立 V3 run，旧成果完整保留。
- 两级权重、轨迹原生锚点均值、V3 标准名称及汇总流程有单元和 CPU 端到端回归验证。
- 用户先要求只准备，随后已明确授权三个模型并行推进；授权状态记录在 run/execution_control.json。
- 三模型训练配方、源池、九任务、单 seed 0 沿用既有设置；不把旧 V1/V2 数字填进 V3 结果。

### 温度切换
所有涉及采样温度的默认方法统一 T005（0.05）：lp-mirror、less-main、rags-main、rags-traj。新方法沿用原定名称，旧版所有方法对照展示加 -T025 后缀；orig/rand/tsds无该温度，旧后缀仅标记历史批次。四温度臂重新训练；其余已完成同协议结果有显式复用来源。旧结果见 [T025快照](archive/MAIN-v3-t025-snapshot-20260917.md)。


### 2026-09-17：RAGS-Main 与 RAGS-Traj 首批配对诊断（T005）

现有完整配对四组：NEM Chronos(main1.441458/traj1.245335)、NEM TimesFM(.713524/.689784)是Traj优；NEM TimeMoE(1.208490/1.303941)及Azure TimeMoE(1.170832/1.203653)是Main优。均seed0、固定最终epoch，不能说所有模型块级都较弱，也不据此宣称显著性。旧T025 ChronosNEM1.499006/1.494969近似无差；主差距在降低温度后出现。

本轮CPU只读分析原始选集、score与已存NEM参考梯度，未启动新训练、未修改正式方法。发现两项确定的机制差异：

1. 源池1082轨迹、360774块；每源轨迹3–4180块。块数与源合法起点数量相关系数.99999945。Main无显式长度乘数，但块等权基础竞争使源轨迹总基础质量正比块数，近似重新引入长度先验；每块同分时P(traj_j)=B_j/ΣB。这与Traj每源轨迹一个单位不同。Main的NEM实际源轨迹组成到“全块均匀”的TV距离Chronos.0495/TimesFM.0149，接近该先验；不是“轨迹均匀”。块内合法起点未必均匀间隔，故也不能说全窗口概率完全等于rand。

2. Main先单位化每个参考块、按起点比例加权；Traj先平均原生锚点梯度、再单位化轨迹，两者不可交换。NEM参考汇总向量范数Main/Traj为Chronos.2251/.8829、TimesFM.1588/.7109、TimeMoE.3998/.8350。Chronos两汇总方向余弦.9948，方向几乎相同，Main主要更弱；Main/Traj源余弦std分别.01499/.07608、.01165/.05136、.05669/.11835。统一T=.05并不保证相同logit方差或选择强度。仅参考汇总范数缩小，不代表重建错误。

| NEM模型 | Main源轨迹ESS | Traj源轨迹ESS | Main top10%质量 | Traj top10%质量 |
|---|---:|---:|---:|---:|
| Chronos | 386.77 | 30.58 | 31.94% | 80.31% |
| TimesFM | 357.40 | 46.21 | 30.92% | 68.36% |
| TimeMoE | 201.24 | 14.90 | 48.47% | 94.69% |

此ESS统一按最终源轨迹概率计算，避免360774块与1082轨迹直接比较。Chronos重复窗口率Main .65%/Traj7.84%，两个方法都训练1018步；Traj在四个checkpoint都优。更有益的选择而非更大预算/单终点偶然排序是当前相容解释，但重复率本身不是效果原因。TimeMoE对强选择反而更差，说明高预训练梯度对齐不保证下游收益。

只除源块数的无训练反事实：Chronos Main源轨迹ESS变1057.62、TimesFM1074.38，几乎轨迹均匀；因此单独修基础先验并不能恢复Traj的评分偏好。建议后续分别验证源基础先验、参考单位化顺序/评分尺度、源整轨迹选择vs块内选择；保持锚点和训练预算配对。不要一口气改变三处。

这些数据尚不能证明线性插值/中点重建误差导致劣势、近零方向噪声是主因、或轨迹平均在所有任务更好；均需固定竞争机制/参考表示后对源表示做消融。证据[JSON](../temp/rags-main-traj-audit-20260917.json)。



分析范围补充：用户指出TimeMoE几乎所有训练方法弱于orig、仅LP-Mirror相对可行，怀疑训练异常。主方法比较优先聚焦Chronos/TimesFM完整T005配对，二者NEM均Traj优。TimeMoE应单列训练健康/负迁移审计，其反向排序暂不作为Main优势证据；同时当前数据不足以确定数值发散或训练链路错误，仅确认训练后评测退化。前述源基础先验与参考方向/尺度诊断在Chronos/TimesFM仍成立。

### 2026-09-17：参考块归一化顺序复核与因果边界

CPU审计NEM Chronos/TimesFM冻结参考梯度，证据 `temp/rags-reference-order-audit-20260917.json`。Main为Q=(1/N)Σ_rΣ_b alpha_rb unit(m_rb)，alpha是本轨迹合法起点占比；权重总和1，没有额外除块数。Traj为Q=(1/N)Σ_r unit(mean_actual_anchors_r)，实际锚点等权均值，也不同于块长度加权均值。

Main/Traj参考汇总范数分别为Chronos .225068/.882898、TimesFM .158771/.710910，汇总方向余弦 .994828/.859855。Main轨迹内块方向平均范数均值 .253187/.174477，跨参考轨迹方向一致性 .888941/.909985，主要压缩发生在轨迹内部。对同一单位块方向平均的每轨迹结果重新单位化，汇总范数恢复 .888546/.903013；未恢复原始梯度模长，因此之前把压缩直接归因为丢掉模长的解释过强。原始块中点长度加权平均后单位化，汇总范数 .889045/.910850，与Traj方向余弦 .994774/.911303。Chronos参考侧主要是尺度差异，TimesFM还存在方向变化。模长、锚点等权与块长度权重贡献尚未完整隔离。

固定Main源块和参考方向，仅按 ||Qtraj||/||Qmain|| 放大分数：Chronos源轨迹ESS386.77→264.35、概率最高10%轨迹质量31.94%→54.31%；TimesFM ESS357.40→100.27、质量30.92%→55.12%。等价Main温度 .012746/.011167。只证明评分尺度影响选择集中度；仍不同于Traj ESS30.58/46.21，未作下游训练，不能声称MASE改善或解释全部差距。归一化顺序是两种评分目标的差异，非已确认的权重或温度实现错误。重建误差主因亦未验证。生产协议未改、未新开训练，T005继续并行推进；下一步可固定源竞争机制配对训练隔离因果。

### 2026-09-17：LESS原论文多子任务冲突协议核实

核读LESS原论文Section4.2 https://arxiv.org/html/2402.04333v3#S4.SS2 及官方 matching.py / write_selected_data.py。论文对子任务内部形成平均梯度代表，跨warmup checkpoint累加余弦影响，跨子任务取max，再全候选Top-k。官方matching.py以checkpoint加权影响矩阵reshape后mean(-1).max(-1)[0]，即子任务内样例影响平均、子任务间max；与论文平均梯度再余弦的内部写法存在区别，不混同。官方选集为全局topk，不用采样Softmax，也无显式正分门控。max不保证无负迁移或子任务配额，只避免其他子任务低分抵消最强子任务匹配。

当前less-main与rags-traj按用户V3协议平均参考轨迹余弦并Softmax，属于时间序列改编协议，不是原论文跨子任务max+Top-k的完全复现。历史把原LESS描述为全部参考求和+Softmax不准确；历史V2求和指本项目实现，且原论文checkpoint求和不是参考评委求和。当前rags-traj仍会有不同参考轨迹方向抵消；其较好结果不能证明冲突已解决。轨迹内部先原始锚点梯度平均再单位化，保留模长影响并重新放大净方向；Main为局部方向单位化后按合法起点占比平均。已有NEM审计支持参考尺度差异，另有源轨迹/块竞争差异；尚未完成性能因果隔离。未改实现/配置/实验批次，T005继续运行。下一步如测试max应明确专家单位（参考轨迹/语义模式），避免把任意细块直接当独立子任务；新臂需单独命名。

### 2026-09-17：专家意见聚合新协议reference-opinion-v1（旧批次不改）

用户决定LESS主规则恢复跨参考max，原V3线性LESS展示less-linearSum；本轮RAGS-Main暂定zeroClipSum，Traj同规则以便粒度对照，预登记设计理由而非按测试集选主项。新增四评分组件linearSum/zeroClipSum/maxSum/expSum，代码src/rags/experiments/reference_opinions.py、full_pool.py、baselines/less/runner.py；expSum严格exp(raw cosine)加权求和后外层Softmax，内层尺度1，不是LSE。max忽略alpha分数乘子；zeroClip不等于源支持硬门控。Local为行Softmax后固定alpha混合，等价同核Global采用alpha/Z有效先验，非相同alpha下同分布；local-max退化linear。当前linearSum-global还不同于同核Global，指数位置不同。

本表当前run继续冻结reference-mean-t005线性协议；旧产物原slug不改，仅展示less-linearSum/rags-main-linearSum/rags-traj-linearSum。已启动的训练不使用新规则；守护重启worker按protocol显式解析旧聚合，缓存检查聚合/竞争/温度/版本字段。新配置configs/rags/main_reference_opinion_20260917.json（7臂）及ablation_reference_opinion_20260917.json（14候选名称）验证通过，definition_only_not_launched；没有新增run/训练/梯度准备。候选矩阵不是自动启动14×9×3训练授权。锚点并集原始梯度可复用，非线性需要双方单位代表而非旧最终线性分数。

消融另外两轴为block/trajectory和adaptive/uniform；历史覆盖/分裂/块边界/窗口返回/硬门控/排名温度等在新设计中分类并标记兼容性，不恢复旧过滤。方法与实验决策镜像：docs/experiments/reference_opinion_method_20260917.md、reference_opinion_ablation_20260917.md；用户指定NDSync _method-0903.md/_exp-0907.md已原子更新并保留历史正文。101项回归通过，包括GPU有界配对、四公式、Local等价偏置、旧协议保护、缓存和窗口支持；未提交/Push。
## 2026-09-17 当前实验矩阵：Block/Traj × 四种意见聚合（中立比较）

**本节覆盖前一轮“暂定RAGS-Main为zeroClipSum”的方案，该暂定主项撤回。** 在得到实验结果之前，不将任何RAGS变体宣称为主方法；八个变体地位相同。LESS-Main仍采用参考轨迹间max合并。历史线性run、已有结果与之前未启动的方案均保留身份，不改名冒充新规则。

| 展示方法 | 执行名称 | 聚合 |
|---|---|---|
| rand | rand | 合法窗口均匀 |
| orig | orig | 不训练 |
| LESS-Main | less-main | maxSum |
| RAGS-Block-linearSum | rags-block-linearSum | 加权有符号和 |
| RAGS-Block-zeroClipSum | rags-block-zeroClipSum | 子分数截零后加权和 |
| RAGS-Block-maxSum | rags-block-maxSum | 参考块余弦最大值 |
| RAGS-Block-expSum | rags-block-expSum | 子分数exp后加权和 |
| RAGS-Traj-linearSum | rags-traj-linearSum | 加权有符号和 |
| RAGS-Traj-zeroClipSum | rags-traj-zeroClipSum | 子分数截零后加权和 |
| RAGS-Traj-maxSum | rags-traj-maxSum | 参考轨迹余弦最大值 |
| RAGS-Traj-expSum | rags-traj-expSum | 子分数exp后加权和 |
| TSDS | tsds | 原生分布 |
| LP-Mirror | lp-mirror | 原生proxy评分→Softmax |

共13方法，沿用九任务×三模型×seed0，完整矩阵351 cells=27 orig+324训练格。所有涉及采样Softmax的臂统一T005=.05；orig/rand/tsds没有该温度。统一使用Global竞争，不追加Local、LESS-linearSum、Uniform、覆盖/分裂或块构造臂；这些只留历史/候选，不是已完成的研究成果。

本轮消融研究主轴只有表示粒度（Block/Traj）和参考意见聚合（linearSum/zeroClipSum/maxSum/expSum）。其余固定：C512/H64/D1、模型/池/参考/训练配方、4096维投影seed42、CRB 2%/1%/1%、批量closest_zero_v2预算用尽且全窗保留。Block用线性插值中点代表；Traj用实际不同CRB锚点原始梯度FP32算术均值后余弦；源单位内部均匀抽合法窗，无额外源长度乘子。

Sum组件参考轨迹权重1/N_ref，参考块权重为(1/N_ref)×所属轨迹合法起点占比；max直接比较余弦、不乘alpha。zeroClip不把源零评分概率强制为零。expSum明确为exp(raw cosine)→加权和→外层Softmax(.05)，内层尺度1；不替换为LSE或exp(c/.05)直接归一化。用户书写maxsum统一记maxSum，仅名称大小写规范。

新配置 `configs/rags/matrix_reference_opinion_20260917.json`，协议 `reference-opinion-v2-neutral`，显式rags_main_method=null、rags_main_method_selected=false；校验拒绝任何主方法声明或矩阵增删。前一轮main_reference_opinion_20260917/ablation_reference_opinion_20260917标为superseded_not_launched，不删除。新矩阵定义与执行入口已接入，当前尚未启动此351格批次；在跑V3线性批次继续按其冻结协议，不静默切换。得到结果也应区分经验排序和独立验证，不能将旧结果作为新规则证据。

本轮验证：87项相关回归通过，包含GPU配对一致性、中立13方法校验及旧V3冻结协议保护；未提交/Push。
### 2026-09-17 精确定义补充：RAGS先双重加权，再处理专家意见

本轮用户明确修正Max/Exp的权重顺序；本段覆盖此前“max不乘alpha”“exp后乘alpha”的RAGS定义。参考专家i对源对象j的余弦为c_ij，先得到q_ij=alpha_i c_ij。Block参考alpha=(1/N_ref)×参考块合法起点占所属参考轨迹比例；Traj参考alpha=1/N_ref（内部长度占比为1）。不乘源对象长度。

| RAGS聚合 | Softmax前源对象评分 |
|---|---|
| linearSum | s_j=Σ_i q_ij |
| zeroClipSum | s_j=Σ_i ReLU(q_ij) |
| maxSum | s_j=max_i q_ij，不相加 |
| expSum | s_j=Σ_i exp(q_ij)，exp之后不再乘alpha |

四者最后统一p_j=Softmax_j(s_j/.05)。ZeroClip由于alpha>0，与Σalpha_i ReLU(c_ij)等价；Max与Exp不等价于上一版原始余弦max和Σalpha_i exp(c_ij)。RAGS所有八臂仍为中立变体，不选主方法；13方法/九任务/三模型/seed0、351cells及其余配方不变。

**LESS-Main基线例外。** LESS-Main保留上一轮用户要求的原论文跨参考max原则：max_i c_ij，不在max前乘1/N_ref。本段是对RAGS四聚合变体的定义，不将原生LESS的max悄悄改为加权max。若锚点及代表相同，RAGS-Traj-maxSum得分恰为LESS-Main得分/N_ref；同配置T=.05对应相对LESS的有效温度.05×N_ref，不能宣称两者仅差锚点、选择强度相同。

**尺度与粒度的直接推论。** 加权Max可能选择权重更大但余弦较低的专家；参考块alpha小会压小源候选间分差。同向块细分虽然不改变linear/zeroClip评分，但会改变加权max及exp评分。因此“所有组件都不受块细分影响”不成立。

expSum可展开为M_ref+Σq_ij+(1/2)Σq_ij²+…，M_ref是参考专家总数，为所有源对象共同常数，最终Softmax消掉。参考块alpha很小时，expSum近似linearSum，非线性奖励可能很弱；负意见仍通过一阶项抵消正意见。不能沿用“指数汇总一定有效避免抵消”的论断。实现用Σexpm1(q_ij)进行Softmax，严格等价于Σexp(q_ij)减共同常数M_ref，避免大量接近1的项淹没数值精度；审计同时记录原始s_j、Softmax用中心分数与共同常数。

当前配置协议更新为reference-opinion-v3-weighted-first，RAGS metadata记录weight_placement=before_transform；LESS记录after_transform且Max输入raw_cosine。上一版中立配置保留为configs/rags/matrix_reference_opinion_v2_neutral_20260917.json（superseded_not_launched），不同顺序概率版本隔离。既有正在运行的V3线性protocol显式保持旧顺序，线性语义不变。本轮不启动新批次。

验证：94项相关回归及1项LESS/加权Traj-Max尺度验证通过；旧V3冻结线性批次不变，新批次尚未启动，未提交/Push。
## 2026-09-18 当前聚合协议：Linear/ZeroClip/Max先加权，Exp后加权

本节是现行定义，覆盖2026-09-17的“所有变体先加权”方案；历史推导、配置和结果保留，不覆盖或冒充新结果。仍为13方法、九任务、三模型、seed0，共351 cells；八个RAGS变体均中立，得到结果之前不指定RAGS主方法。本轮只更新准备材料，未启动新批次。

参考专家i与源对象j的原始余弦为c_ij。Block双层权重alpha_i=(1/N_ref)×该块拥有的合法起点数/所属参考轨迹全部合法起点数；Traj内部占比为1，即alpha_i=1/N_ref。权重总和为1，不乘源对象长度，不对最终概率再次加权。

| RAGS变体 | 外层Softmax前得分 |
|---|---|
| linearSum | S_j=Σ_i alpha_i c_ij |
| zeroClipSum | S_j=Σ_i ReLU(alpha_i c_ij)=Σ_i alpha_i ReLU(c_ij) |
| maxSum | S_j=max_i(alpha_i c_ij)，不相加 |
| expSum | S_j=Σ_i alpha_i exp(c_ij/1.00)，先exp，再双层加权求和 |

全部最后统一p_j=Softmax_j(S_j/0.05)。Exp内层温度明确固定T_inner=1.00；外层温度T_outer=0.05。不得将内层偷偷改成0.05，不得改成log-sum-exp或指数混合后直接归一化。配置和审计显式记录exp_inner_temperature=1.00、weight_placement=after_transform；其余RAGS三变体为before_transform。除Exp外，其他规则不变。

选择理由：Linear与ZeroClip保留参考内容占比；Max保留加权后择优，既尊重比例也允许长块影响胜者。Traj等权Max的1/N不改变胜者/排序，但会改变固定温度下的概率集中度，不能说概率完全不受影响。Block专家很多时原始Max可能接近饱和，各源对象最大值趋同；加权Max也可能因块权重小而缩小分差，本轮把它视作待验证变体。

Exp先加权后指数的旧式Σexp(alpha*c)，在alpha小时近似M_ref+Σalpha*c，外层Softmax消掉共同M_ref后接近linearSum。现改为Σalpha*exp(c)，alpha控制意见测度，指数负责非线性评价。同方向块细分且总权重保持不变时，新Exp得分不变。其展开为1+Σalpha*c+(1/2)Σalpha*c²+…：外层Softmax消掉共同常数1，但高阶项不再被alpha平方压小；内层T=1仍可能较温和，是否提高性能必须等待新实验，不能声称已证明优于Linear。

LESS版本边界：正在运行的旧V3 LESS是(1/N_ref)Σ_i c_ij，内核求和后在调用层除以len(reference)，不是未归一化总和。已准备的新LESS-Main仍保留此前指定的原始余弦Max，即max_i c_ij，作为原论文多子任务Max原则的适配基线。本次不更改该基线。若表示相同，RAGS-Traj-maxSum分数为LESS-Main的1/N_ref，固定T=.05下选择强度不同。新旧LESS主要变化为Mean→Max，不是新增1/N；旧线性结果展示为LESS-linearSum。

执行配置仍为configs/rags/matrix_reference_opinion_20260917.json，聚合协议更新为reference-opinion-v4-mixed-weight-order（聚合版本号，不表示已启动V4实验）。上一版保存为configs/rags/matrix_reference_opinion_v3_weighted_first_20260917.json，状态superseded_not_launched，继续通过校验。已有V3运行协议与缓存身份保留；只为Exp切换权重位置，Max/Linear/ZeroClip语义不变。没有新梯度准备、训练或评估启动，未提交/Push。

验证：99项相关回归测试通过（3条既有Swig弃用警告）。覆盖精确公式、内层T=1/外层T=.05、同向参考块细分不变性、Block/Traj生产入口、历史V3与上一版配置保护；未启动新批次。
## 2026-09-18 V4范围与编排：两模型、自然数seed逐轮串行

本节为最新实验范围与编排决策，覆盖此前“三模型、只做seed0”的新意见聚合矩阵规模。聚合公式不变：RAGS Linear/ZeroClip/Max先按双层alpha加权，Exp先exp(c/1.00)再按alpha加权加和，外层Softmax .05；LESS-Main保留原始余弦Max。13方法、九任务不变，不指定任何RAGS主方法。

V4暂时只做chronos-base和timesfm-2.5。TimeMoE-200M因当前微调困难退出V4，不运行它的准备、训练或评估；历史结果、材料和实现保留。替代模型尚未确定，本轮不自行选择或加入新模型。此前正在运行的V3批次沿其原冻结配置处理，本次不停止或修改历史批次。

编排制度：seed固定自然数0、1、2、3、4……，从0开始逐轮串行；必须先完成同一seed下两个模型×九任务×全部训练方法的训练和评估，再放行seed+1。并行放在当前seed的一轮内部，两模型并行，不能让较快模型提前进入下个seed。失败、缺项、身份不符或非有限MASE都会阻止跨轮；续跑从最早未完成的轮次继续，已完成项按原有续跑逻辑跳过。父套件使用单例锁，避免两个编排进程分别推进不同seed。

没有指定总seed数，不擅自把01234限定为五轮。默认持续自然数序列；--max-seeds只提供可选的运行次数上限，不是当前科学协议的一部分。各轮位于独立rounds/seedK目录，配置seeds只包含[K]，训练checkpoint、结果和seed身份独立。

第一轮为234格=18 orig基线+216训练/评估格（两个模型×九任务×12训练方法）。orig与训练seed无关，在第一轮训练前测量，后续只引用首次测量结果。每增加一个seed，新增216训练/评估格；S个seed的唯一测量总数为18+216×S。每轮的完整展示仍包含18个基线引用。原351格规模是旧三模型seed0设计，不能再作为V4规模。

训练seed控制PyTorch随机数、CUDA随机数与训练时源窗口抽样。源池pool_seed=0、4096维投影seed42、固定参考、梯度材料及静态选择概率保持不变；后续seed复用已完成seed0的静态概率选集，并采用自己的训练/抽样seed，不是重复引用训练结果。已有字段校验梯度缓存按轮链接复用，不重新为每个seed计算梯度，不新增TimeMoE梯度任务；现有缓存的单写者锁仍有效。

远程配置：configs/rags/main_v4.json。入口：python -m rags.experiments.v4_seed_rounds --config configs/rags/main_v4.json --run <数据盘V4父目录>。--dry-run仅打印计划，不加载模型、不启动任务。正式入口须遵守父目录execution_control.json运行开关；本次修改与验证不启动V4实验。父state.json记录当前seed、完成seed及下一seed；各轮结果沿用原cell JSON/selection NPZ格式，无hash，协议与字段明确校验、JSON原子发布。

验证状态：77项相关回归测试通过（3条既有Swig弃用警告）。验证了跨模型轮次屏障、自然数seed配置、失败/缺项阻止下一轮、续跑、静态材料与orig复用、实际seed汇总及旧V3兼容。编排dry-run为两模型、首轮234格、后续每seed新增216格、jobs_launched=0。本次未启动V4实验，未提交/Push。
<!-- v3-notes:end -->
