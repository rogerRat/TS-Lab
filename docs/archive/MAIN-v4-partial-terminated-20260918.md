# MAIN-v4 — 主实验

- 更新时间：2026-09-18T16:56:55+08:00
- 配置：`configs/rags/main_v4.json`；运行：`/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918`
- 所有使用采样温度的默认方法（lp-mirror / less-main / rags-main / rags-traj）统一 T=0.05。
- 九任务、2模型；矩阵 rand / orig / less-main / rags-block-linearSum / rags-block-zeroClipSum / rags-block-maxSum / rags-block-expSum / rags-traj-linearSum / rags-traj-zeroClipSum / rags-traj-maxSum / rags-traj-expSum / tsds / lp-mirror；seeds [0, 1, 2]。
- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，批量分裂规则不变。
- 新意见聚合协议：LESS参考max；RAGS按Block/Traj及四聚合显式命名，不指定主方法。
- 原生逐窗口梯度沿用已有字段校验缓存；V4 训练与评测独立；V4轮间复用静态概率和orig，训练seed逐轮变化。
- V1/V2 内容完整归档，继续参考：[MAIN-v1](archive/MAIN-v1.md)、[MAIN-v2](archive/MAIN-v2.md)。

完成 395/666 cells。

| 模型 | 已完成 cells | 当前状态 |
|---|---:|---|
| chronos-base | 210 | training/time_solar_h/rags-traj-expSum/1/running |
| timesfm-2.5 | 185 | training/time_cphl_h/rags-traj-maxSum/1/running |
<!-- v4-patrol:start -->
## 巡检记录与解读（每 30 分钟自动更新）

- 巡检时间：2026-09-18T16:56:55+08:00（北京时间）；以实际 cells/state 为准。
- 总进度：395/666 cells；已授权并行推进。

### 结果（按任务分节，每节各模型一表；两列百分比分别为均值相对 orig 与 rand 的改善，正=更好；胜/n 为逐 seed 配对胜 rand）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-expSum | 2 | 1.176176 | 0.051399 | +31.91% | +12.25% | 2/2 |
| rags-traj-linearSum | 2 | 1.248892 | 0.005031 | +27.71% | +6.82% | 2/2 |
| less-main | 2 | 1.252777 | 0.044459 | +27.48% | +6.53% | 2/2 |
| rags-traj-zeroClipSum | 2 | 1.294794 | 0.028800 | +25.05% | +3.40% | 1/2 |
| tsds | 2 | 1.338099 | 0.050668 | +22.54% | +0.17% | 1/2 |
| rand | 2 | 1.340348 | 0.103962 | +22.41% | +0.00% | — |
| rags-block-zeroClipSum | 2 | 1.426203 | 0.035300 | +17.44% | -6.41% | 1/2 |
| rags-block-linearSum | 2 | 1.431217 | 0.014484 | +17.15% | -6.78% | 0/2 |
| rags-block-maxSum | 2 | 1.456659 | 0.023758 | +15.68% | -8.68% | 0/2 |
| rags-traj-maxSum | 2 | 1.493788 | 0.026335 | +13.53% | -11.45% | 0/2 |
| rags-block-expSum | 2 | 1.517179 | 0.009562 | +12.18% | -13.19% | 0/2 |
| orig | 1 | 1.727507 | — | +0.00% | -28.89% | — |
| lp-mirror | 2 | 2.067686 | 0.010789 | -19.69% | -54.26% | 0/2 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-expSum | 2 | 0.680408 | 0.011941 | +22.85% | +7.10% | 2/2 |
| less-main | 2 | 0.686324 | 0.006088 | +22.18% | +6.30% | 2/2 |
| rags-traj-linearSum | 2 | 0.693223 | 0.004864 | +21.40% | +5.35% | 2/2 |
| rags-traj-zeroClipSum | 2 | 0.708700 | 0.003700 | +19.64% | +3.24% | 2/2 |
| rags-block-linearSum | 2 | 0.718740 | 0.007377 | +18.50% | +1.87% | 2/2 |
| tsds | 2 | 0.720226 | 0.001821 | +18.33% | +1.67% | 2/2 |
| rags-block-expSum | 2 | 0.720764 | 0.014983 | +18.27% | +1.59% | 1/2 |
| rand | 2 | 0.732444 | 0.008195 | +16.95% | +0.00% | — |
| rags-traj-maxSum | 2 | 0.733116 | 0.012797 | +16.87% | -0.09% | 1/2 |
| rags-block-zeroClipSum | 2 | 0.746136 | 0.019016 | +15.40% | -1.87% | 0/2 |
| rags-block-maxSum | 2 | 0.760397 | 0.012405 | +13.78% | -3.82% | 0/2 |
| lp-mirror | 2 | 0.877250 | 0.000205 | +0.53% | -19.77% | 0/2 |
| orig | 1 | 0.881911 | — | +0.00% | -20.41% | — |

### time_azure_i_cpu_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-zeroClipSum | 2 | 1.185293 | 0.038746 | +10.99% | +3.03% | 1/2 |
| tsds | 2 | 1.193408 | 0.037778 | +10.38% | +2.36% | 2/2 |
| rand | 2 | 1.222301 | 0.058431 | +8.21% | +0.00% | — |
| rags-traj-linearSum | 2 | 1.227964 | 0.006111 | +7.79% | -0.46% | 1/2 |
| lp-mirror | 2 | 1.231576 | 0.015578 | +7.51% | -0.76% | 1/2 |
| rags-block-expSum | 2 | 1.232651 | 0.001499 | +7.43% | -0.85% | 1/2 |
| rags-block-linearSum | 2 | 1.233582 | 0.017297 | +7.36% | -0.92% | 1/2 |
| rags-block-maxSum | 2 | 1.240410 | 0.022072 | +6.85% | -1.48% | 1/2 |
| rags-traj-maxSum | 2 | 1.250247 | 0.016523 | +6.11% | -2.29% | 1/2 |
| rags-traj-zeroClipSum | 2 | 1.260238 | 0.012514 | +5.36% | -3.10% | 1/2 |
| rags-traj-expSum | 2 | 1.275371 | 0.019141 | +4.23% | -4.34% | 0/2 |
| orig | 1 | 1.331634 | — | +0.00% | -8.94% | — |
| less-main | 2 | 1.333616 | 0.025134 | -0.15% | -9.11% | 0/2 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 2 | 0.761583 | 0.001583 | +0.29% | +0.24% | 1/2 |
| rags-traj-maxSum | 2 | 0.761741 | 0.001344 | +0.27% | +0.22% | 2/2 |
| rand | 2 | 0.763412 | 0.003027 | +0.05% | +0.00% | — |
| orig | 1 | 0.763820 | — | +0.00% | -0.05% | — |
| tsds | 2 | 0.767499 | 0.000010 | -0.48% | -0.54% | 0/2 |
| rags-block-maxSum | 2 | 0.767530 | 0.002536 | -0.49% | -0.54% | 0/2 |
| rags-traj-zeroClipSum | 2 | 0.767645 | 0.002165 | -0.50% | -0.55% | 0/2 |
| rags-block-zeroClipSum | 2 | 0.768065 | 0.003000 | -0.56% | -0.61% | 0/2 |
| rags-traj-linearSum | 2 | 0.769786 | 0.002322 | -0.78% | -0.83% | 0/2 |
| rags-block-linearSum | 2 | 0.770488 | 0.001337 | -0.87% | -0.93% | 0/2 |
| rags-traj-expSum | 2 | 0.771951 | 0.001859 | -1.06% | -1.12% | 0/2 |
| rags-block-expSum | 2 | 0.772725 | 0.000968 | -1.17% | -1.22% | 0/2 |
| less-main | 2 | 0.787026 | 0.005646 | -3.04% | -3.09% | 0/2 |

### time_darwin_water_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-main | 2 | 1.655063 | 0.004153 | +0.54% | +0.43% | 2/2 |
| rags-block-maxSum | 2 | 1.655702 | 0.011663 | +0.50% | +0.39% | 2/2 |
| rags-block-zeroClipSum | 2 | 1.655838 | 0.010615 | +0.50% | +0.39% | 1/2 |
| rags-block-expSum | 2 | 1.656438 | 0.002080 | +0.46% | +0.35% | 2/2 |
| rags-traj-zeroClipSum | 2 | 1.658846 | 0.007015 | +0.31% | +0.21% | 2/2 |
| tsds | 2 | 1.659678 | 0.005137 | +0.26% | +0.16% | 1/2 |
| rand | 2 | 1.662259 | 0.003814 | +0.11% | +0.00% | — |
| rags-traj-linearSum | 2 | 1.662789 | 0.008889 | +0.08% | -0.03% | 1/2 |
| orig | 1 | 1.664086 | — | +0.00% | -0.11% | — |
| rags-block-linearSum | 2 | 1.665764 | 0.002060 | -0.10% | -0.21% | 1/2 |
| rags-traj-expSum | 2 | 1.666936 | 0.002778 | -0.17% | -0.28% | 0/2 |
| rags-traj-maxSum | 2 | 1.667128 | 0.000526 | -0.18% | -0.29% | 0/2 |
| lp-mirror | 2 | 1.710288 | 0.095393 | -2.78% | -2.89% | 1/2 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 2 | 1.012590 | 0.003748 | +5.34% | +1.58% | 2/2 |
| rags-block-expSum | 2 | 1.015458 | 0.001583 | +5.07% | +1.30% | 2/2 |
| less-main | 2 | 1.015761 | 0.003217 | +5.04% | +1.27% | 2/2 |
| rags-block-zeroClipSum | 2 | 1.018188 | 0.008418 | +4.82% | +1.03% | 2/2 |
| rags-traj-expSum | 2 | 1.021355 | 0.000380 | +4.52% | +0.72% | 2/2 |
| rags-traj-linearSum | 2 | 1.023426 | 0.000709 | +4.33% | +0.52% | 2/2 |
| rags-traj-zeroClipSum | 2 | 1.026400 | 0.002712 | +4.05% | +0.23% | 2/2 |
| rand | 2 | 1.028798 | 0.003357 | +3.82% | +0.00% | — |
| rags-block-maxSum | 2 | 1.033366 | 0.001890 | +3.40% | -0.44% | 0/2 |
| rags-traj-maxSum | 2 | 1.034423 | 0.005452 | +3.30% | -0.55% | 0/2 |
| tsds | 2 | 1.040122 | 0.003548 | +2.77% | -1.10% | 0/2 |
| orig | 1 | 1.069706 | — | +0.00% | -3.98% | — |
| lp-mirror | 2 | 1.069761 | 0.000247 | -0.01% | -3.98% | 0/2 |

### time_finland_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| tsds | 2 | 0.998040 | 0.025930 | +51.45% | +18.28% | 2/2 |
| rags-traj-linearSum | 2 | 1.051900 | 0.129849 | +48.83% | +13.87% | 2/2 |
| rags-traj-maxSum | 2 | 1.051900 | 0.129849 | +48.83% | +13.87% | 2/2 |
| rags-traj-expSum | 2 | 1.100540 | 0.030268 | +46.46% | +9.89% | 2/2 |
| less-main | 2 | 1.137678 | 0.194813 | +44.65% | +6.85% | 1/2 |
| rags-traj-zeroClipSum | 2 | 1.200104 | 0.278911 | +41.62% | +1.74% | 1/2 |
| rand | 2 | 1.221344 | 0.081833 | +40.58% | +0.00% | — |
| rags-block-linearSum | 2 | 1.229442 | 0.004874 | +40.19% | -0.66% | 1/2 |
| rags-block-maxSum | 2 | 1.233797 | 0.266671 | +39.98% | -1.02% | 1/2 |
| rags-block-expSum | 2 | 1.237789 | 0.080779 | +39.78% | -1.35% | 0/2 |
| rags-block-zeroClipSum | 2 | 1.515952 | 0.054390 | +26.25% | -24.12% | 0/2 |
| lp-mirror | 2 | 1.575634 | 0.169404 | +23.35% | -29.01% | 0/2 |
| orig | 1 | 2.055572 | — | +0.00% | -68.30% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-expSum | 2 | 0.608930 | 0.000385 | +2.08% | +1.11% | 2/2 |
| rags-block-maxSum | 2 | 0.611421 | 0.000266 | +1.68% | +0.70% | 2/2 |
| rags-traj-linearSum | 2 | 0.611640 | 0.000671 | +1.65% | +0.67% | 2/2 |
| rags-traj-maxSum | 2 | 0.611640 | 0.000671 | +1.65% | +0.67% | 2/2 |
| rags-block-expSum | 2 | 0.612495 | 0.001830 | +1.51% | +0.53% | 1/2 |
| rags-block-linearSum | 2 | 0.613370 | 0.001905 | +1.37% | +0.39% | 1/2 |
| tsds | 2 | 0.615075 | 0.000949 | +1.09% | +0.11% | 1/2 |
| rags-traj-zeroClipSum | 2 | 0.615094 | 0.002919 | +1.09% | +0.11% | 1/2 |
| rand | 2 | 0.615746 | 0.004993 | +0.99% | +0.00% | — |
| rags-block-zeroClipSum | 2 | 0.616187 | 0.000821 | +0.91% | -0.07% | 1/2 |
| less-main | 2 | 0.616436 | 0.003304 | +0.87% | -0.11% | 1/2 |
| orig | 1 | 0.621873 | — | +0.00% | -1.00% | — |
| lp-mirror | 2 | 0.626147 | 0.000324 | -0.69% | -1.69% | 0/2 |

### time_eweld_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 2 | 1.174730 | 0.049813 | +21.47% | +6.10% | 2/2 |
| rags-block-expSum | 2 | 1.239987 | 0.139806 | +17.11% | +0.88% | 1/2 |
| rand | 2 | 1.251029 | 0.114384 | +16.37% | +0.00% | — |
| rags-traj-maxSum | 2 | 1.259141 | 0.116442 | +15.83% | -0.65% | 1/2 |
| tsds | 2 | 1.259739 | 0.047214 | +15.79% | -0.70% | 1/2 |
| rags-traj-expSum | 2 | 1.306140 | 0.054417 | +12.69% | -4.41% | 1/2 |
| rags-block-linearSum | 2 | 1.345481 | 0.108089 | +10.06% | -7.55% | 1/2 |
| rags-block-maxSum | 2 | 1.359386 | 0.061046 | +9.13% | -8.66% | 1/2 |
| rags-traj-zeroClipSum | 2 | 1.372773 | 0.104323 | +8.24% | -9.73% | 1/2 |
| less-main | 2 | 1.403154 | 0.037629 | +6.21% | -12.16% | 0/2 |
| rags-traj-linearSum | 2 | 1.436728 | 0.015380 | +3.96% | -14.84% | 0/2 |
| rags-block-zeroClipSum | 2 | 1.443781 | 0.024499 | +3.49% | -15.41% | 0/2 |
| orig | 1 | 1.495984 | — | +0.00% | -19.58% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-maxSum | 2 | 0.720008 | 0.001672 | +1.13% | +0.48% | 2/2 |
| rags-block-maxSum | 2 | 0.720151 | 0.001446 | +1.11% | +0.46% | 2/2 |
| rags-traj-zeroClipSum | 2 | 0.721428 | 0.000299 | +0.94% | +0.29% | 2/2 |
| rags-block-linearSum | 2 | 0.722275 | 0.000296 | +0.82% | +0.17% | 2/2 |
| tsds | 2 | 0.722983 | 0.002100 | +0.72% | +0.07% | 2/2 |
| rand | 2 | 0.723508 | 0.002027 | +0.65% | +0.00% | — |
| rags-block-expSum | 2 | 0.723596 | 0.001226 | +0.64% | -0.01% | 1/2 |
| rags-block-zeroClipSum | 2 | 0.723629 | 0.002235 | +0.63% | -0.02% | 1/2 |
| rags-traj-linearSum | 2 | 0.723749 | 0.001772 | +0.62% | -0.03% | 1/2 |
| rags-traj-expSum | 2 | 0.723892 | 0.000582 | +0.60% | -0.05% | 1/2 |
| orig | 1 | 0.728248 | — | +0.00% | -0.66% | — |
| less-main | 2 | 0.728531 | 0.000048 | -0.04% | -0.69% | 0/2 |
| lp-mirror | 2 | 0.730128 | 0.002873 | -0.26% | -0.92% | 0/2 |

### time_cphl_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-linearSum | 2 | 1.488860 | 0.015245 | +3.24% | +4.41% | 2/2 |
| lp-mirror | 2 | 1.527918 | 0.004782 | +0.71% | +1.90% | 2/2 |
| orig | 1 | 1.538772 | — | +0.00% | +1.20% | — |
| tsds | 2 | 1.540917 | 0.026936 | -0.14% | +1.07% | 2/2 |
| rags-block-expSum | 2 | 1.543637 | 0.034706 | -0.32% | +0.89% | 2/2 |
| rags-block-linearSum | 2 | 1.543750 | 0.019019 | -0.32% | +0.88% | 2/2 |
| rags-traj-zeroClipSum | 2 | 1.549067 | 0.009643 | -0.67% | +0.54% | 1/2 |
| rags-traj-maxSum | 2 | 1.550763 | 0.028610 | -0.78% | +0.43% | 1/2 |
| rags-block-zeroClipSum | 2 | 1.551827 | 0.000947 | -0.85% | +0.37% | 1/2 |
| rags-block-maxSum | 2 | 1.554507 | 0.037378 | -1.02% | +0.19% | 1/2 |
| rand | 2 | 1.557517 | 0.015970 | -1.22% | +0.00% | — |
| rags-traj-expSum | 2 | 1.578647 | 0.060730 | -2.59% | -1.36% | 1/2 |
| less-main | 2 | 1.604966 | 0.000451 | -4.30% | -3.05% | 0/2 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-block-linearSum | 2 | 1.333654 | 0.000166 | +2.38% | +1.02% | 2/2 |
| rags-block-zeroClipSum | 2 | 1.334819 | 0.003711 | +2.30% | +0.93% | 2/2 |
| rags-block-expSum | 2 | 1.335257 | 0.006192 | +2.27% | +0.90% | 2/2 |
| rags-block-maxSum | 2 | 1.335566 | 0.008688 | +2.24% | +0.87% | 2/2 |
| rags-traj-maxSum | 1 | 1.344525 | — | +1.59% | +0.21% | 1/1 |
| rand | 2 | 1.347342 | 0.002114 | +1.38% | +0.00% | — |
| less-main | 2 | 1.347518 | 0.000230 | +1.37% | -0.01% | 1/2 |
| rags-traj-expSum | 1 | 1.347547 | — | +1.37% | -0.02% | 0/1 |
| rags-traj-zeroClipSum | 2 | 1.347935 | 0.002943 | +1.34% | -0.04% | 0/2 |
| lp-mirror | 1 | 1.349813 | — | +1.20% | -0.18% | 0/1 |
| rags-traj-linearSum | 2 | 1.352928 | 0.004450 | +0.97% | -0.41% | 0/2 |
| tsds | 1 | 1.362809 | — | +0.25% | -1.15% | 0/1 |
| orig | 1 | 1.366210 | — | +0.00% | -1.40% | — |

### time_nem_price_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-expSum | 2 | 1.042847 | 0.002718 | +1.82% | +1.03% | 1/2 |
| rags-traj-maxSum | 2 | 1.043475 | 0.019720 | +1.76% | +0.97% | 1/2 |
| rags-traj-linearSum | 2 | 1.045415 | 0.004134 | +1.58% | +0.79% | 1/2 |
| rags-block-expSum | 2 | 1.045741 | 0.010449 | +1.55% | +0.76% | 1/2 |
| rags-block-zeroClipSum | 2 | 1.046792 | 0.005605 | +1.45% | +0.66% | 1/2 |
| less-main | 2 | 1.048658 | 0.005938 | +1.28% | +0.48% | 1/2 |
| rags-traj-zeroClipSum | 2 | 1.049180 | 0.015637 | +1.23% | +0.43% | 1/2 |
| tsds | 2 | 1.049361 | 0.002873 | +1.21% | +0.42% | 1/2 |
| lp-mirror | 2 | 1.051015 | 0.003450 | +1.05% | +0.26% | 1/2 |
| rags-block-maxSum | 2 | 1.052840 | 0.035689 | +0.88% | +0.09% | 1/2 |
| rand | 2 | 1.053744 | 0.019792 | +0.80% | +0.00% | — |
| rags-block-linearSum | 2 | 1.054669 | 0.011694 | +0.71% | -0.09% | 1/2 |
| orig | 1 | 1.062202 | — | +0.00% | -0.80% | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rags-traj-expSum | 1 | 0.707365 | — | +5.58% | +1.40% | 1/1 |
| rags-traj-zeroClipSum | 1 | 0.708384 | — | +5.45% | +1.25% | 1/1 |
| rags-traj-linearSum | 1 | 0.712027 | — | +4.96% | +0.75% | 1/1 |
| less-main | 1 | 0.715924 | — | +4.44% | +0.20% | 1/1 |
| rand | 1 | 0.717374 | — | +4.25% | +0.00% | — |
| rags-block-linearSum | 1 | 0.718022 | — | +4.16% | -0.09% | 0/1 |
| rags-block-expSum | 1 | 0.718489 | — | +4.10% | -0.16% | 0/1 |
| rags-block-zeroClipSum | 1 | 0.720940 | — | +3.77% | -0.50% | 0/1 |
| tsds | 1 | 0.721010 | — | +3.76% | -0.51% | 0/1 |
| rags-traj-maxSum | 1 | 0.723276 | — | +3.46% | -0.82% | 0/1 |
| rags-block-maxSum | 1 | 0.724004 | — | +3.36% | -0.92% | 0/1 |
| orig | 1 | 0.749205 | — | +0.00% | -4.44% | — |
| lp-mirror | 1 | 0.754650 | — | -0.73% | -5.20% | 0/1 |

### time_solar_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| less-main | 2 | 1.092254 | 0.065338 | +10.72% | +7.96% | 1/2 |
| rags-traj-expSum | 1 | 1.138957 | — | +6.90% | +4.02% | 0/1 |
| rags-traj-linearSum | 2 | 1.181255 | 0.035926 | +3.44% | +0.46% | 1/2 |
| rags-traj-maxSum | 2 | 1.181255 | 0.035926 | +3.44% | +0.46% | 1/2 |
| rand | 2 | 1.186699 | 0.074000 | +3.00% | +0.00% | — |
| orig | 1 | 1.223355 | — | +0.00% | -3.09% | — |
| rags-traj-zeroClipSum | 2 | 1.249834 | 0.060986 | -2.16% | -5.32% | 1/2 |
| rags-block-expSum | 2 | 1.268980 | 0.000049 | -3.73% | -6.93% | 0/2 |
| tsds | 1 | 1.273342 | — | -4.09% | -7.30% | 0/1 |
| rags-block-linearSum | 2 | 1.303766 | 0.284659 | -6.57% | -9.86% | 1/2 |
| lp-mirror | 1 | 1.311835 | — | -7.23% | -10.54% | 0/1 |
| rags-block-zeroClipSum | 2 | 1.340535 | 0.310787 | -9.58% | -12.96% | 1/2 |
| rags-block-maxSum | 2 | 1.357154 | 0.092345 | -10.94% | -14.36% | 0/2 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| lp-mirror | 1 | 1.103397 | — | +2.29% | +2.42% | 1/1 |
| rags-traj-zeroClipSum | 1 | 1.118377 | — | +0.97% | +1.10% | 1/1 |
| rags-traj-linearSum | 1 | 1.119860 | — | +0.84% | +0.97% | 1/1 |
| rags-traj-maxSum | 1 | 1.119860 | — | +0.84% | +0.97% | 1/1 |
| orig | 1 | 1.129299 | — | +0.00% | +0.13% | — |
| rand | 1 | 1.130794 | — | -0.13% | +0.00% | — |
| less-main | 1 | 1.131229 | — | -0.17% | -0.04% | 0/1 |
| tsds | 1 | 1.131475 | — | -0.19% | -0.06% | 0/1 |
| rags-block-maxSum | 1 | 1.146978 | — | -1.57% | -1.43% | 0/1 |
| rags-block-zeroClipSum | 1 | 1.148907 | — | -1.74% | -1.60% | 0/1 |
| rags-traj-expSum | 1 | 1.149650 | — | -1.80% | -1.67% | 0/1 |
| rags-block-linearSum | 1 | 1.153879 | — | -2.18% | -2.04% | 0/1 |
| rags-block-expSum | 1 | 1.163095 | — | -2.99% | -2.86% | 0/1 |

### time_port_activity_d

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.882133 | — | +0.00% | +1.54% | — |
| rags-block-expSum | 1 | 0.884721 | — | -0.29% | +1.26% | 1/1 |
| lp-mirror | 1 | 0.885762 | — | -0.41% | +1.14% | 1/1 |
| rags-traj-maxSum | 1 | 0.889470 | — | -0.83% | +0.73% | 1/1 |
| rags-traj-zeroClipSum | 1 | 0.894842 | — | -1.44% | +0.13% | 1/1 |
| rags-traj-expSum | 1 | 0.895186 | — | -1.48% | +0.09% | 1/1 |
| rags-block-linearSum | 1 | 0.895667 | — | -1.53% | +0.03% | 1/1 |
| rand | 1 | 0.895971 | — | -1.57% | +0.00% | — |
| rags-traj-linearSum | 1 | 0.896399 | — | -1.62% | -0.05% | 0/1 |
| rags-block-zeroClipSum | 1 | 0.903792 | — | -2.46% | -0.87% | 0/1 |
| tsds | 1 | 0.911446 | — | -3.32% | -1.73% | 0/1 |
| less-main | 1 | 0.918136 | — | -4.08% | -2.47% | 0/1 |
| rags-block-maxSum | 1 | 0.927945 | — | -5.19% | -3.57% | 0/1 |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 1 | 0.716205 | — | +0.00% | +1.02% | — |
| rags-traj-maxSum | 1 | 0.721531 | — | -0.74% | +0.28% | 1/1 |
| lp-mirror | 1 | 0.722498 | — | -0.88% | +0.15% | 1/1 |
| rand | 1 | 0.723588 | — | -1.03% | +0.00% | — |
| rags-block-maxSum | 1 | 0.724205 | — | -1.12% | -0.09% | 0/1 |
| tsds | 1 | 0.736563 | — | -2.84% | -1.79% | 0/1 |
| rags-block-zeroClipSum | 1 | 0.739047 | — | -3.19% | -2.14% | 0/1 |
| rags-block-linearSum | 1 | 0.741228 | — | -3.49% | -2.44% | 0/1 |
| rags-block-expSum | 1 | 0.743162 | — | -3.76% | -2.71% | 0/1 |
| less-main | 1 | 0.746831 | — | -4.28% | -3.21% | 0/1 |
| rags-traj-linearSum | 1 | 0.752374 | — | -5.05% | -3.98% | 0/1 |
| rags-traj-zeroClipSum | 1 | 0.753158 | — | -5.16% | -4.09% | 0/1 |
| rags-traj-expSum | 1 | 0.755281 | — | -5.46% | -4.38% | 0/1 |

**解读**：

以本版真实落盘结果更新；没有结果时不作效果判断。

<!-- v4-patrol:end -->

| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |
|---|---|---|---:|---:|---:|
| chronos-base | time_finland_15t | tsds | 2 | 0.998040 | 0.025930 |
| chronos-base | time_finland_15t | rags-traj-linearSum | 2 | 1.051900 | 0.129849 |
| chronos-base | time_finland_15t | rags-traj-maxSum | 2 | 1.051900 | 0.129849 |
| chronos-base | time_nem_demand_5t | rags-traj-expSum | 2 | 1.176176 | 0.051399 |
| chronos-base | time_finland_15t | rags-traj-expSum | 2 | 1.100540 | 0.030268 |
| chronos-base | time_solar_h | less-main | 2 | 1.092254 | 0.065338 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-expSum | 2 | 0.680408 | 0.011941 |
| chronos-base | time_finland_15t | less-main | 2 | 1.137678 | 0.194813 |
| chronos-base | time_nem_demand_5t | rags-traj-linearSum | 2 | 1.248892 | 0.005031 |
| chronos-base | time_nem_demand_5t | less-main | 2 | 1.252777 | 0.044459 |
| timesfm-2.5 | time_nem_demand_5t | less-main | 2 | 0.686324 | 0.006088 |
| chronos-base | time_eweld_15t | lp-mirror | 2 | 1.174730 | 0.049813 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-linearSum | 2 | 0.693223 | 0.004864 |
| chronos-base | time_cphl_h | rags-traj-linearSum | 2 | 1.488860 | 0.015245 |
| chronos-base | time_solar_h | rags-traj-expSum | 1 | 1.138957 | — |
| chronos-base | time_nem_demand_5t | rags-traj-zeroClipSum | 2 | 1.294794 | 0.028800 |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-zeroClipSum | 2 | 0.708700 | 0.003700 |
| chronos-base | time_azure_i_cpu_5t | rags-block-zeroClipSum | 2 | 1.185293 | 0.038746 |
| timesfm-2.5 | time_solar_h | lp-mirror | 1 | 1.103397 | — |
| chronos-base | time_azure_i_cpu_5t | tsds | 2 | 1.193408 | 0.037778 |
| chronos-base | time_cphl_h | lp-mirror | 2 | 1.527918 | 0.004782 |
| timesfm-2.5 | time_nem_demand_5t | rags-block-linearSum | 2 | 0.718740 | 0.007377 |
| chronos-base | time_finland_15t | rags-traj-zeroClipSum | 2 | 1.200104 | 0.278911 |
| timesfm-2.5 | time_nem_demand_5t | tsds | 2 | 0.720226 | 0.001821 |
| timesfm-2.5 | time_nem_demand_5t | rags-block-expSum | 2 | 0.720764 | 0.014983 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-linearSum | 2 | 1.012590 | 0.003748 |
| chronos-base | time_port_activity_d | orig | 1 | 0.882133 | — |
| timesfm-2.5 | time_nem_price_5t | rags-traj-expSum | 1 | 0.707365 | — |
| timesfm-2.5 | time_darwin_water_15t | rags-block-expSum | 2 | 1.015458 | 0.001583 |
| timesfm-2.5 | time_darwin_water_15t | less-main | 2 | 1.015761 | 0.003217 |
| chronos-base | time_port_activity_d | rags-block-expSum | 1 | 0.884721 | — |
| timesfm-2.5 | time_nem_price_5t | rags-traj-zeroClipSum | 1 | 0.708384 | — |
| chronos-base | time_cphl_h | orig | 1 | 1.538772 | — |
| chronos-base | time_port_activity_d | lp-mirror | 1 | 0.885762 | — |
| timesfm-2.5 | time_finland_15t | rags-traj-expSum | 2 | 0.608930 | 0.000385 |
| timesfm-2.5 | time_solar_h | rags-traj-zeroClipSum | 1 | 1.118377 | — |
| chronos-base | time_cphl_h | tsds | 2 | 1.540917 | 0.026936 |
| chronos-base | time_nem_price_5t | rags-traj-expSum | 2 | 1.042847 | 0.002718 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-zeroClipSum | 2 | 1.018188 | 0.008418 |
| timesfm-2.5 | time_port_activity_d | orig | 1 | 0.716205 | — |
| timesfm-2.5 | time_cphl_h | rags-block-linearSum | 2 | 1.333654 | 0.000166 |
| chronos-base | time_nem_price_5t | rags-traj-maxSum | 2 | 1.043475 | 0.019720 |
| timesfm-2.5 | time_solar_h | rags-traj-linearSum | 1 | 1.119860 | — |
| timesfm-2.5 | time_solar_h | rags-traj-maxSum | 1 | 1.119860 | — |
| timesfm-2.5 | time_cphl_h | rags-block-zeroClipSum | 2 | 1.334819 | 0.003711 |
| timesfm-2.5 | time_cphl_h | rags-block-expSum | 2 | 1.335257 | 0.006192 |
| chronos-base | time_cphl_h | rags-block-expSum | 2 | 1.543637 | 0.034706 |
| chronos-base | time_cphl_h | rags-block-linearSum | 2 | 1.543750 | 0.019019 |
| chronos-base | time_eweld_15t | rags-block-expSum | 2 | 1.239987 | 0.139806 |
| timesfm-2.5 | time_cphl_h | rags-block-maxSum | 2 | 1.335566 | 0.008688 |
| chronos-base | time_nem_price_5t | rags-traj-linearSum | 2 | 1.045415 | 0.004134 |
| chronos-base | time_nem_price_5t | rags-block-expSum | 2 | 1.045741 | 0.010449 |
| timesfm-2.5 | time_nem_price_5t | rags-traj-linearSum | 1 | 0.712027 | — |
| chronos-base | time_port_activity_d | rags-traj-maxSum | 1 | 0.889470 | — |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-expSum | 2 | 1.021355 | 0.000380 |
| timesfm-2.5 | time_finland_15t | rags-block-maxSum | 2 | 0.611421 | 0.000266 |
| timesfm-2.5 | time_finland_15t | rags-traj-linearSum | 2 | 0.611640 | 0.000671 |
| timesfm-2.5 | time_finland_15t | rags-traj-maxSum | 2 | 0.611640 | 0.000671 |
| chronos-base | time_nem_price_5t | rags-block-zeroClipSum | 2 | 1.046792 | 0.005605 |
| chronos-base | time_cphl_h | rags-traj-zeroClipSum | 2 | 1.549067 | 0.009643 |
| timesfm-2.5 | time_finland_15t | rags-block-expSum | 2 | 0.612495 | 0.001830 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-linearSum | 2 | 1.023426 | 0.000709 |
| timesfm-2.5 | time_eweld_15t | rags-traj-maxSum | 2 | 0.720008 | 0.001672 |
| chronos-base | time_nem_price_5t | less-main | 2 | 1.048658 | 0.005938 |
| timesfm-2.5 | time_eweld_15t | rags-block-maxSum | 2 | 0.720151 | 0.001446 |
| chronos-base | time_solar_h | rags-traj-linearSum | 2 | 1.181255 | 0.035926 |
| chronos-base | time_solar_h | rags-traj-maxSum | 2 | 1.181255 | 0.035926 |
| chronos-base | time_cphl_h | rags-traj-maxSum | 2 | 1.550763 | 0.028610 |
| chronos-base | time_nem_price_5t | rags-traj-zeroClipSum | 2 | 1.049180 | 0.015637 |
| chronos-base | time_darwin_water_15t | less-main | 2 | 1.655063 | 0.004153 |
| chronos-base | time_nem_price_5t | tsds | 2 | 1.049361 | 0.002873 |
| chronos-base | time_darwin_water_15t | rags-block-maxSum | 2 | 1.655702 | 0.011663 |
| chronos-base | time_darwin_water_15t | rags-block-zeroClipSum | 2 | 1.655838 | 0.010615 |
| timesfm-2.5 | time_finland_15t | rags-block-linearSum | 2 | 0.613370 | 0.001905 |
| chronos-base | time_cphl_h | rags-block-zeroClipSum | 2 | 1.551827 | 0.000947 |
| chronos-base | time_darwin_water_15t | rags-block-expSum | 2 | 1.656438 | 0.002080 |
| timesfm-2.5 | time_eweld_15t | rags-traj-zeroClipSum | 2 | 0.721428 | 0.000299 |
| timesfm-2.5 | time_port_activity_d | rags-traj-maxSum | 1 | 0.721531 | — |
| chronos-base | time_nem_price_5t | lp-mirror | 2 | 1.051015 | 0.003450 |
| timesfm-2.5 | time_azure_i_cpu_5t | lp-mirror | 2 | 0.761583 | 0.001583 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-zeroClipSum | 2 | 1.026400 | 0.002712 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-maxSum | 2 | 0.761741 | 0.001344 |
| timesfm-2.5 | time_cphl_h | rags-traj-maxSum | 1 | 1.344525 | — |
| chronos-base | time_darwin_water_15t | rags-traj-zeroClipSum | 2 | 1.658846 | 0.007015 |
| timesfm-2.5 | time_nem_price_5t | less-main | 1 | 0.715924 | — |
| chronos-base | time_cphl_h | rags-block-maxSum | 2 | 1.554507 | 0.037378 |
| timesfm-2.5 | time_eweld_15t | rags-block-linearSum | 2 | 0.722275 | 0.000296 |
| chronos-base | time_nem_demand_5t | tsds | 2 | 1.338099 | 0.050668 |
| chronos-base | time_darwin_water_15t | tsds | 2 | 1.659678 | 0.005137 |
| timesfm-2.5 | time_port_activity_d | lp-mirror | 1 | 0.722498 | — |
| timesfm-2.5 | time_solar_h | orig | 1 | 1.129299 | — |
| chronos-base | time_port_activity_d | rags-traj-zeroClipSum | 1 | 0.894842 | — |
| timesfm-2.5 | time_finland_15t | tsds | 2 | 0.615075 | 0.000949 |
| timesfm-2.5 | time_finland_15t | rags-traj-zeroClipSum | 2 | 0.615094 | 0.002919 |
| chronos-base | time_port_activity_d | rags-traj-expSum | 1 | 0.895186 | — |
| chronos-base | time_nem_price_5t | rags-block-maxSum | 2 | 1.052840 | 0.035689 |
| timesfm-2.5 | time_eweld_15t | tsds | 2 | 0.722983 | 0.002100 |
| chronos-base | time_port_activity_d | rags-block-linearSum | 1 | 0.895667 | — |
| chronos-base | time_azure_i_cpu_5t | rand | 2 | 1.222301 | 0.058431 |
| chronos-base | time_cphl_h | rand | 2 | 1.557517 | 0.015970 |
| chronos-base | time_darwin_water_15t | rand | 2 | 1.662259 | 0.003814 |
| chronos-base | time_eweld_15t | rand | 2 | 1.251029 | 0.114384 |
| chronos-base | time_finland_15t | rand | 2 | 1.221344 | 0.081833 |
| chronos-base | time_nem_demand_5t | rand | 2 | 1.340348 | 0.103962 |
| chronos-base | time_nem_price_5t | rand | 2 | 1.053744 | 0.019792 |
| chronos-base | time_port_activity_d | rand | 1 | 0.895971 | — |
| chronos-base | time_solar_h | rand | 2 | 1.186699 | 0.074000 |
| timesfm-2.5 | time_azure_i_cpu_5t | rand | 2 | 0.763412 | 0.003027 |
| timesfm-2.5 | time_cphl_h | rand | 2 | 1.347342 | 0.002114 |
| timesfm-2.5 | time_darwin_water_15t | rand | 2 | 1.028798 | 0.003357 |
| timesfm-2.5 | time_eweld_15t | rand | 2 | 0.723508 | 0.002027 |
| timesfm-2.5 | time_finland_15t | rand | 2 | 0.615746 | 0.004993 |
| timesfm-2.5 | time_nem_demand_5t | rand | 2 | 0.732444 | 0.008195 |
| timesfm-2.5 | time_nem_price_5t | rand | 1 | 0.717374 | — |
| timesfm-2.5 | time_port_activity_d | rand | 1 | 0.723588 | — |
| timesfm-2.5 | time_solar_h | rand | 1 | 1.130794 | — |
| timesfm-2.5 | time_eweld_15t | rags-block-expSum | 2 | 0.723596 | 0.001226 |
| timesfm-2.5 | time_cphl_h | less-main | 2 | 1.347518 | 0.000230 |
| timesfm-2.5 | time_cphl_h | rags-traj-expSum | 1 | 1.347547 | — |
| timesfm-2.5 | time_eweld_15t | rags-block-zeroClipSum | 2 | 0.723629 | 0.002235 |
| chronos-base | time_darwin_water_15t | rags-traj-linearSum | 2 | 1.662789 | 0.008889 |
| timesfm-2.5 | time_eweld_15t | rags-traj-linearSum | 2 | 0.723749 | 0.001772 |
| timesfm-2.5 | time_solar_h | less-main | 1 | 1.131229 | — |
| timesfm-2.5 | time_cphl_h | rags-traj-zeroClipSum | 2 | 1.347935 | 0.002943 |
| chronos-base | time_port_activity_d | rags-traj-linearSum | 1 | 0.896399 | — |
| timesfm-2.5 | time_eweld_15t | rags-traj-expSum | 2 | 0.723892 | 0.000582 |
| timesfm-2.5 | time_azure_i_cpu_5t | orig | 1 | 0.763820 | — |
| timesfm-2.5 | time_solar_h | tsds | 1 | 1.131475 | — |
| timesfm-2.5 | time_finland_15t | rags-block-zeroClipSum | 2 | 0.616187 | 0.000821 |
| timesfm-2.5 | time_port_activity_d | rags-block-maxSum | 1 | 0.724205 | — |
| chronos-base | time_nem_price_5t | rags-block-linearSum | 2 | 1.054669 | 0.011694 |
| timesfm-2.5 | time_nem_price_5t | rags-block-linearSum | 1 | 0.718022 | — |
| timesfm-2.5 | time_nem_demand_5t | rags-traj-maxSum | 2 | 0.733116 | 0.012797 |
| chronos-base | time_darwin_water_15t | orig | 1 | 1.664086 | — |
| timesfm-2.5 | time_finland_15t | less-main | 2 | 0.616436 | 0.003304 |
| timesfm-2.5 | time_nem_price_5t | rags-block-expSum | 1 | 0.718489 | — |
| timesfm-2.5 | time_cphl_h | lp-mirror | 1 | 1.349813 | — |
| chronos-base | time_darwin_water_15t | rags-block-linearSum | 2 | 1.665764 | 0.002060 |
| chronos-base | time_darwin_water_15t | rags-traj-expSum | 2 | 1.666936 | 0.002778 |
| chronos-base | time_darwin_water_15t | rags-traj-maxSum | 2 | 1.667128 | 0.000526 |
| timesfm-2.5 | time_cphl_h | rags-traj-linearSum | 2 | 1.352928 | 0.004450 |
| timesfm-2.5 | time_darwin_water_15t | rags-block-maxSum | 2 | 1.033366 | 0.001890 |
| chronos-base | time_azure_i_cpu_5t | rags-traj-linearSum | 2 | 1.227964 | 0.006111 |
| timesfm-2.5 | time_nem_price_5t | rags-block-zeroClipSum | 1 | 0.720940 | — |
| timesfm-2.5 | time_nem_price_5t | tsds | 1 | 0.721010 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | tsds | 2 | 0.767499 | 0.000010 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-maxSum | 2 | 0.767530 | 0.002536 |
| timesfm-2.5 | time_darwin_water_15t | rags-traj-maxSum | 2 | 1.034423 | 0.005452 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-zeroClipSum | 2 | 0.767645 | 0.002165 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-zeroClipSum | 2 | 0.768065 | 0.003000 |
| chronos-base | time_eweld_15t | rags-traj-maxSum | 2 | 1.259141 | 0.116442 |
| timesfm-2.5 | time_eweld_15t | orig | 1 | 0.728248 | — |
| chronos-base | time_finland_15t | rags-block-linearSum | 2 | 1.229442 | 0.004874 |
| timesfm-2.5 | time_eweld_15t | less-main | 2 | 0.728531 | 0.000048 |
| chronos-base | time_eweld_15t | tsds | 2 | 1.259739 | 0.047214 |
| chronos-base | time_azure_i_cpu_5t | lp-mirror | 2 | 1.231576 | 0.015578 |
| chronos-base | time_nem_price_5t | orig | 1 | 1.062202 | — |
| timesfm-2.5 | time_nem_price_5t | rags-traj-maxSum | 1 | 0.723276 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-linearSum | 2 | 0.769786 | 0.002322 |
| chronos-base | time_azure_i_cpu_5t | rags-block-expSum | 2 | 1.232651 | 0.001499 |
| chronos-base | time_port_activity_d | rags-block-zeroClipSum | 1 | 0.903792 | — |
| timesfm-2.5 | time_eweld_15t | lp-mirror | 2 | 0.730128 | 0.002873 |
| chronos-base | time_azure_i_cpu_5t | rags-block-linearSum | 2 | 1.233582 | 0.017297 |
| timesfm-2.5 | time_nem_price_5t | rags-block-maxSum | 1 | 0.724004 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-linearSum | 2 | 0.770488 | 0.001337 |
| timesfm-2.5 | time_finland_15t | orig | 1 | 0.621873 | — |
| chronos-base | time_finland_15t | rags-block-maxSum | 2 | 1.233797 | 0.266671 |
| timesfm-2.5 | time_darwin_water_15t | tsds | 2 | 1.040122 | 0.003548 |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-traj-expSum | 2 | 0.771951 | 0.001859 |
| timesfm-2.5 | time_cphl_h | tsds | 1 | 1.362809 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | rags-block-expSum | 2 | 0.772725 | 0.000968 |
| chronos-base | time_finland_15t | rags-block-expSum | 2 | 1.237789 | 0.080779 |
| chronos-base | time_cphl_h | rags-traj-expSum | 2 | 1.578647 | 0.060730 |
| timesfm-2.5 | time_cphl_h | orig | 1 | 1.366210 | — |
| timesfm-2.5 | time_solar_h | rags-block-maxSum | 1 | 1.146978 | — |
| chronos-base | time_azure_i_cpu_5t | rags-block-maxSum | 2 | 1.240410 | 0.022072 |
| timesfm-2.5 | time_solar_h | rags-block-zeroClipSum | 1 | 1.148907 | — |
| timesfm-2.5 | time_solar_h | rags-traj-expSum | 1 | 1.149650 | — |
| timesfm-2.5 | time_finland_15t | lp-mirror | 2 | 0.626147 | 0.000324 |
| chronos-base | time_port_activity_d | tsds | 1 | 0.911446 | — |
| timesfm-2.5 | time_port_activity_d | tsds | 1 | 0.736563 | — |
| timesfm-2.5 | time_nem_demand_5t | rags-block-zeroClipSum | 2 | 0.746136 | 0.019016 |
| timesfm-2.5 | time_solar_h | rags-block-linearSum | 1 | 1.153879 | — |
| timesfm-2.5 | time_port_activity_d | rags-block-zeroClipSum | 1 | 0.739047 | — |
| chronos-base | time_azure_i_cpu_5t | rags-traj-maxSum | 2 | 1.250247 | 0.016523 |
| timesfm-2.5 | time_port_activity_d | rags-block-linearSum | 1 | 0.741228 | — |
| chronos-base | time_port_activity_d | less-main | 1 | 0.918136 | — |
| timesfm-2.5 | time_port_activity_d | rags-block-expSum | 1 | 0.743162 | — |
| timesfm-2.5 | time_solar_h | rags-block-expSum | 1 | 1.163095 | — |
| chronos-base | time_darwin_water_15t | lp-mirror | 2 | 1.710288 | 0.095393 |
| chronos-base | time_cphl_h | less-main | 2 | 1.604966 | 0.000451 |
| chronos-base | time_solar_h | orig | 1 | 1.223355 | — |
| timesfm-2.5 | time_azure_i_cpu_5t | less-main | 2 | 0.787026 | 0.005646 |
| chronos-base | time_azure_i_cpu_5t | rags-traj-zeroClipSum | 2 | 1.260238 | 0.012514 |
| timesfm-2.5 | time_port_activity_d | less-main | 1 | 0.746831 | — |
| chronos-base | time_port_activity_d | rags-block-maxSum | 1 | 0.927945 | — |
| timesfm-2.5 | time_nem_demand_5t | rags-block-maxSum | 2 | 0.760397 | 0.012405 |
| timesfm-2.5 | time_darwin_water_15t | orig | 1 | 1.069706 | — |
| timesfm-2.5 | time_port_activity_d | rags-traj-linearSum | 1 | 0.752374 | — |
| timesfm-2.5 | time_darwin_water_15t | lp-mirror | 2 | 1.069761 | 0.000247 |
| timesfm-2.5 | time_port_activity_d | rags-traj-zeroClipSum | 1 | 0.753158 | — |
| chronos-base | time_azure_i_cpu_5t | rags-traj-expSum | 2 | 1.275371 | 0.019141 |
| timesfm-2.5 | time_port_activity_d | rags-traj-expSum | 1 | 0.755281 | — |
| chronos-base | time_eweld_15t | rags-traj-expSum | 2 | 1.306140 | 0.054417 |
| timesfm-2.5 | time_nem_price_5t | orig | 1 | 0.749205 | — |
| timesfm-2.5 | time_nem_price_5t | lp-mirror | 1 | 0.754650 | — |
| chronos-base | time_solar_h | rags-traj-zeroClipSum | 2 | 1.249834 | 0.060986 |
| chronos-base | time_nem_demand_5t | rags-block-zeroClipSum | 2 | 1.426203 | 0.035300 |
| chronos-base | time_nem_demand_5t | rags-block-linearSum | 2 | 1.431217 | 0.014484 |
| chronos-base | time_solar_h | rags-block-expSum | 2 | 1.268980 | 0.000049 |
| chronos-base | time_solar_h | tsds | 1 | 1.273342 | — |
| chronos-base | time_eweld_15t | rags-block-linearSum | 2 | 1.345481 | 0.108089 |
| chronos-base | time_eweld_15t | rags-block-maxSum | 2 | 1.359386 | 0.061046 |
| chronos-base | time_nem_demand_5t | rags-block-maxSum | 2 | 1.456659 | 0.023758 |
| chronos-base | time_azure_i_cpu_5t | orig | 1 | 1.331634 | — |
| chronos-base | time_azure_i_cpu_5t | less-main | 2 | 1.333616 | 0.025134 |
| chronos-base | time_eweld_15t | rags-traj-zeroClipSum | 2 | 1.372773 | 0.104323 |
| chronos-base | time_solar_h | rags-block-linearSum | 2 | 1.303766 | 0.284659 |
| chronos-base | time_solar_h | lp-mirror | 1 | 1.311835 | — |
| chronos-base | time_nem_demand_5t | rags-traj-maxSum | 2 | 1.493788 | 0.026335 |
| chronos-base | time_eweld_15t | less-main | 2 | 1.403154 | 0.037629 |
| chronos-base | time_solar_h | rags-block-zeroClipSum | 2 | 1.340535 | 0.310787 |
| chronos-base | time_nem_demand_5t | rags-block-expSum | 2 | 1.517179 | 0.009562 |
| chronos-base | time_solar_h | rags-block-maxSum | 2 | 1.357154 | 0.092345 |
| chronos-base | time_eweld_15t | rags-traj-linearSum | 2 | 1.436728 | 0.015380 |
| chronos-base | time_eweld_15t | rags-block-zeroClipSum | 2 | 1.443781 | 0.024499 |
| chronos-base | time_eweld_15t | orig | 1 | 1.495984 | — |
| timesfm-2.5 | time_nem_demand_5t | lp-mirror | 2 | 0.877250 | 0.000205 |
| timesfm-2.5 | time_nem_demand_5t | orig | 1 | 0.881911 | — |
| chronos-base | time_finland_15t | rags-block-zeroClipSum | 2 | 1.515952 | 0.054390 |
| chronos-base | time_nem_demand_5t | orig | 1 | 1.727507 | — |
| chronos-base | time_finland_15t | lp-mirror | 2 | 1.575634 | 0.169404 |
| chronos-base | time_nem_demand_5t | lp-mirror | 2 | 2.067686 | 0.010789 |
| chronos-base | time_finland_15t | orig | 1 | 2.055572 | — |

<!-- v4-notes:start -->
## 固定方法协议（reference-opinion-v4-mixed-weight-order）

| 方法 | 单位 | 聚合 | 竞争 | 温度 |
|---|---|---|---|---|
| rand | — | — | — | — |
| orig | — | — | — | — |
| less-main | trajectory | maxSum | global | .05 |
| rags-block-linearSum | block | linearSum | global | .05 |
| rags-block-zeroClipSum | block | zeroClipSum | global | .05 |
| rags-block-maxSum | block | maxSum | global | .05 |
| rags-block-expSum | block | expSum | global | .05 |
| rags-traj-linearSum | trajectory | linearSum | global | .05 |
| rags-traj-zeroClipSum | trajectory | zeroClipSum | global | .05 |
| rags-traj-maxSum | trajectory | maxSum | global | .05 |
| rags-traj-expSum | trajectory | expSum | global | .05 |
| tsds | — | — | — | — |
| lp-mirror | — | — | — | .05 |

RAGS-Block/Traj各四种聚合均为未排序变体；未指定RAGS主方法。LESS-Main采用max。
RAGS linear/zeroClip/max先乘alpha；maxSum=max(alpha*cos)。expSum=sum(alpha*exp(cos/1.0))，内层T=1.00、外层Softmax T=.05。LESS-Main仍取原始余弦max。
历史reference-mean线性批次保留原产物，以-linearSum展示；本批独立选集/训练，不改旧批次。
## 2026-09-18 V4正式启动：固定三seed、V3归档与巡检

最新决策覆盖前一版不限seed数的计划：V4固定仅使用seed0、seed1、seed2，总共三轮。轮间串行，当前轮两个模型×九任务×12训练方法全部完成训练与评估后才开始下一seed；轮内Chronos Base和TimesFM 2.5并行，TimeMoE暂停。13方法及聚合公式不变，外层Softmax T=.05；Exp内层T=1.00、先exp再双层加权，Max保留加权后择优。新seed不是新源池，pool_seed0和projection_seed42仍固定。

总规模666个独立结果项=18个orig基线+216训练评估项×3seed。orig首轮在训练前测量，后续引用；静态选择概率和已有字段校验梯度缓存复用，各seed独立训练/抽样与checkpoint，不复用训练成绩。父协议seeds=[0,1,2]，子轮seeds=[K]，seed_schedule.stop=3，默认第三轮结束后停止，禁止自动进入seed3。

V3已完成189/189。已整理全部既有输出目录，归档746个cell JSON及配置、文档、日志、run元数据，生成完整all_cells.jsonl与按各原run/model/task相对rand改进降序的all_results_ranked.csv。归档位于/root/autodl-tmp/TS3/outputs/archives/results-through-v3-20260918，含results-docs-configs.tar.gz及明确字段inventory.json，无hash。原结果、选集、梯度缓存及保留checkpoint留在原路径；大梯度不重复复制。最终V3报告另存docs/archive/MAIN-v3-final-20260918.md。

归档后停止已结束的V3 scheduler/report loop，清理其过期PID文件及可丢弃报告partial，不删除正式成果或V4依赖材料。V3 execution_control切换archived，防止旧守护误续跑。V4已按用户授权真实启动：configs/rags/main_v4.json；run=/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918；父入口rags.experiments.v4_seed_rounds，先推进seed0。恢复仍须用父入口，不能绕过跨模型seed屏障。

MAIN-v4.md直接复用MAIN-v3的报告生成模板，保持任务分节、模型表、状态/巡检、末尾汇总与固定方法说明的版式，只更换内容及版本标识。所有结果表（任务模型表、末尾聚合汇总表）均按相对rand改进降序排序，正值更好，缺数据/缺rand基准列末；不是按绝对MASE或方法名排序。末尾汇总按方法聚合seed，orig仅计一次，不把基线引用算成新实验。三seed汇总包含均值、样本标准差及逐seed胜rand配对信息。

79项相关回归通过，覆盖固定三seed止步、全矩阵屏障、失败/续跑、基线去重、表格降序和同模板报告、旧V3兼容。已建立本任务每30分钟巡检，并维护远程30分钟报告更新；正常进度保持安静，整轮/全批完成或明显异常时通知。本机不关机，已启用保持空闲唤醒，远程独立运行；不执行关机操作。当前实验进行中，尚不对V4效果作结论；未提交/Push。

### 2026-09-18 巡检责任交接

用户明确要求巡检交给本机tmux Claude，Codex不再负责。Codex heartbeat automation v4已暂停（PAUSED），已向claude:0提交接管巡检、异常处理、MAIN-v4维护和本机镜像同步的消息。V4现有实验父进程和远程报告更新继续运行，没有停止或重启实验；固定seed0/1/2和其他科学协议不变。本机保持开机。此决定覆盖先前由Codex定时巡检的记录。


### 2026-09-18 当前seed0即时吞吐优化与恢复确认

source矩阵按原FP64/source_batch分块上传一次并驻留GPU，参考批次复用GPU切片，避免重复H2D上传。用户要求全力推进后，安全从父入口恢复当前seed0，使优化立即生效；新父PID839955，Chronos worker839965，TimesFM worker839966。已完成cells、原梯度缓存和选集保留，Chronos TSDS在恢复前已落盘。未改温度、科学协议、种子屏障或聚合顺序。

恢复后Chronos进入NEM rand训练，TimesFM进入NEM TSDS选择；连续10秒GPU采样全部100%，显存26890MiB，功耗460.25–483.78W。这是当前阶段实测，不能据此推断全流程恒定100%。tests/test_reference_opinions.py回归49项全部通过（1.66s），包含GPU/CPU数值一致性及聚合协议验证。性能恢复快照位于run/performance_reload.json。后续巡检与资源异常处理继续由本机tmux Claude负责，Codex heartbeat仍PAUSED。
<!-- v4-notes:end -->
