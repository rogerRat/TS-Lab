# MAIN-v3 — 主实验

- 更新时间：2026-09-18T00:14:31+08:00
- 配置：`configs/rags/matrix_reference_opinion_20260917.json`；运行：`temp/exp-after-weight-20260918/report-preview`
- 所有使用采样温度的默认方法（lp-mirror / less-main / rags-main / rags-traj）统一 T=0.05。
- 九任务、三模型；矩阵 rand / orig / less-main / rags-block-linearSum / rags-block-zeroClipSum / rags-block-maxSum / rags-block-expSum / rags-traj-linearSum / rags-traj-zeroClipSum / rags-traj-maxSum / rags-traj-expSum / tsds / lp-mirror；仅 seed 0。
- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，批量分裂规则不变。
- 新意见聚合协议：LESS参考max；RAGS按Block/Traj及四聚合显式命名，不指定主方法。
- 原生逐窗口梯度沿用已有字段校验缓存；V3 选集、训练和评测新生，不复制旧 cells/checkpoints。
- V1/V2 内容完整归档，继续参考：[MAIN-v1](archive/MAIN-v1.md)、[MAIN-v2](archive/MAIN-v2.md)。

完成 0/351 cells。

| 模型 | 已完成 cells | 当前状态 |
|---|---:|---|
| chronos-base | 0 | 准备完成，等待授权，尚未启动 |
| timesfm-2.5 | 0 | 准备完成，等待授权，尚未启动 |
| timemoe-200m | 0 | 准备完成，等待授权，尚未启动 |
<!-- v3-patrol:start -->
## 巡检记录与解读（每 30 分钟自动更新）

- 巡检时间：2026-09-18T00:14:31+08:00（北京时间）；以实际 cells/state 为准。
- 总进度：0/351 cells；尚未启动。

### 结果（按任务分节，每节三模型各一表；两列百分比分别为均值相对 orig 与 rand 的改善，正=更好；胜/n 为逐 seed 配对胜 rand）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_azure_i_cpu_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_darwin_water_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_finland_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_eweld_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_cphl_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_nem_price_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_solar_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

### time_port_activity_d

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| rand | 0 | — | — | — | — | — |
| orig | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rags-block-linearSum | 0 | — | — | — | — | — |
| rags-block-zeroClipSum | 0 | — | — | — | — | — |
| rags-block-maxSum | 0 | — | — | — | — | — |
| rags-block-expSum | 0 | — | — | — | — | — |
| rags-traj-linearSum | 0 | — | — | — | — | — |
| rags-traj-zeroClipSum | 0 | — | — | — | — | — |
| rags-traj-maxSum | 0 | — | — | — | — | — |
| rags-traj-expSum | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |

**解读**：

以本版真实落盘结果更新；没有结果时不作效果判断。

<!-- v3-patrol:end -->

| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |
|---|---|---|---:|---:|---:|
| — | — | — | 0 | — | — |

<!-- v3-notes:start -->
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
<!-- v3-notes:end -->
