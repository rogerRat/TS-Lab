# MAIN-v3 — 主实验

- 更新时间：2026-09-17T18:18:13+08:00
- 配置：`configs/rags/main_v3.json`；运行：`/root/autodl-tmp/TS3/outputs/main-shared-nine-v3-20260917`
- 九任务、三模型；矩阵 orig-T025 / rand-T025 / tsds-T025 / lp-mirror-T025 / less-main-T025 / rags-main-T025 / rags-traj-T025；仅 seed 0。
- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，批量分裂规则不变。
- LESS/RAGS 全家采用参考平均评分：轨迹权重 1/N，块在所属轨迹内部按合法起点比例划分。
- 原生逐窗口梯度沿用已有字段校验缓存；V3 选集、训练和评测新生，不复制旧 cells/checkpoints。
- V1/V2 内容完整归档，继续参考：[MAIN-v1](MAIN-v1.md)、[MAIN-v2](MAIN-v2.md)。

完成 13/189 cells。

| 模型 | 已完成 cells | 当前状态 |
|---|---:|---|
| chronos-base | 6 | training/time_nem_demand_5t/rags-traj-T025/0/running |
| timesfm-2.5 | 3 | training/time_nem_demand_5t/lp-mirror-T025/0/running |
| timemoe-200m | 4 | training/time_nem_demand_5t/less-main-T025/0/running |
<!-- v3-patrol:start -->
## 巡检记录与解读（每 30 分钟自动更新）

- 巡检时间：2026-09-17T18:18:13+08:00（北京时间）；以实际 cells/state 为准。
- 总进度：13/189 cells；已授权并行推进。

### 结果（按任务分节，每节三模型各一表；两列百分比分别为均值相对 orig-T025 与 rand-T025 的改善，正=更好；胜/n 为逐 seed 配对胜 rand-T025）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 1 | 1.727507 | — | +0.00% | -22.18% | — |
| rand-T025 | 1 | 1.413860 | — | +18.16% | +0.00% | — |
| tsds-T025 | 1 | 1.302271 | — | +24.62% | +7.89% | 1/1 |
| lp-mirror-T025 | 1 | 1.494751 | — | +13.47% | -5.72% | 0/1 |
| less-main-T025 | 1 | 1.539824 | — | +10.86% | -8.91% | 0/1 |
| rags-main-T025 | 1 | 1.499006 | — | +13.23% | -6.02% | 0/1 |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 1 | 0.881911 | — | +0.00% | -21.37% | — |
| rand-T025 | 1 | 0.726650 | — | +17.61% | +0.00% | — |
| tsds-T025 | 1 | 0.721514 | — | +18.19% | +0.71% | 1/1 |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 1 | 1.065605 | — | +0.00% | +10.59% | — |
| rand-T025 | 1 | 1.191778 | — | -11.84% | +0.00% | — |
| tsds-T025 | 1 | 1.381358 | — | -29.63% | -15.91% | 0/1 |
| lp-mirror-T025 | 1 | 1.254806 | — | -17.76% | -5.29% | 0/1 |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_azure_i_cpu_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_darwin_water_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_finland_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_eweld_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_cphl_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_nem_price_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_solar_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

### time_port_activity_d

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig-T025 | vs rand-T025 | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig-T025 | 0 | — | — | — | — | — |
| rand-T025 | 0 | — | — | — | — | — |
| tsds-T025 | 0 | — | — | — | — | — |
| lp-mirror-T025 | 0 | — | — | — | — | — |
| less-main-T025 | 0 | — | — | — | — | — |
| rags-main-T025 | 0 | — | — | — | — | — |
| rags-traj-T025 | 0 | — | — | — | — | — |

**解读**：

以本版真实落盘结果更新；没有结果时不作效果判断。

<!-- v3-patrol:end -->

| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |
|---|---|---|---:|---:|---:|
| chronos-base | time_nem_demand_5t | orig-T025 | 1 | 1.727507 | — |
| chronos-base | time_nem_demand_5t | rand-T025 | 1 | 1.413860 | — |
| chronos-base | time_nem_demand_5t | tsds-T025 | 1 | 1.302271 | — |
| chronos-base | time_nem_demand_5t | lp-mirror-T025 | 1 | 1.494751 | — |
| chronos-base | time_nem_demand_5t | less-main-T025 | 1 | 1.539824 | — |
| chronos-base | time_nem_demand_5t | rags-main-T025 | 1 | 1.499006 | — |
| timesfm-2.5 | time_nem_demand_5t | orig-T025 | 1 | 0.881911 | — |
| timesfm-2.5 | time_nem_demand_5t | rand-T025 | 1 | 0.726650 | — |
| timesfm-2.5 | time_nem_demand_5t | tsds-T025 | 1 | 0.721514 | — |
| timemoe-200m | time_nem_demand_5t | orig-T025 | 1 | 1.065605 | — |
| timemoe-200m | time_nem_demand_5t | rand-T025 | 1 | 1.191778 | — |
| timemoe-200m | time_nem_demand_5t | tsds-T025 | 1 | 1.381358 | — |
| timemoe-200m | time_nem_demand_5t | lp-mirror-T025 | 1 | 1.254806 | — |

<!-- v3-notes:start -->
## V3 固定方法协议

| 方法 | 评分单位 | 参考权重 | 竞争与概率 |
|---|---|---|---|
| orig-T025 | 不选择；预训练模型直接评测 | 不适用 | 不训练 |
| rand-T025 | 全部合法起点 | 不适用 | 延续均匀窗口基线 |
| tsds-T025 | 原生 TSDS | 原方法不变 | 原生分布不变 |
| lp-mirror-T025 | 窗口 | 原方法不变 | 延续 V2 lpv_t025 的 raw-value 分布 |
| less-main-T025 | 轨迹 | 每参考轨迹 1/N | 无门控、Global、平均余弦、Softmax |
| rags-main-T025 | 块 | (1/N)×块合法起点占所属轨迹比例 | 无门控、Global、两级评委、Softmax |
| rags-traj-T025 | 轨迹 | 每参考轨迹 1/N | 无门控、Global、平均余弦、Softmax |

N 指当前参考数据集的参考轨迹数；参考话语权总和为 1。权重只用于分数相加，
最终 P_j=softmax(S_j/T)，不乘源单位长度，不对 Softmax 输出再次分配参考预算。
less-main-T025/rags-main-T025/rags-traj-T025 统一 T=0.25；该温度沿用此前数值，尚未针对平均评分重新调参。

less-main-T025 对按 rounded linspace 均匀选取的原生窗口投影梯度做 FP32 算术平均，
再分别归一化轨迹代表并算余弦。rags-traj-T025 同样直接平均实际计算的原生 CRB 锚点梯度，
不使用旧版的全窗口线性重构均值；每轨迹梯度数量核对同一配置预算，唯一设计差异是取点位置。
rags-main-T025 保留块级线性中点代表。CRB 覆盖阶段最长区间二分，误差阶段积分风险优先，
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


### 2026-09-17：LESS V2→V3定量核查（保持正式实验继续运行）

以各模型NEM seed0的原始选集与cells配对。V2真实落盘LESS为lessvg_t025，V3为less-main。改变两项：V2 P_j∝m_j·exp(Σ_r cos_jr/.25)，V3 P_j∝exp((Σ_r cos_jr/N)/.25)。m_j为源轨迹合法起点数；不是参考长度权重。NEM N=5，均值评分温度.25在求和尺度等效温度1.25，较旧版本升温5倍。若只希望在均值尺度恢复旧对齐力度，应使用.05；这仍不恢复已删除的源长度先验，因此不能宣称完全恢复V2。

原始梯度/均匀取点预算/参考及源sample_counts一致，V3分数等于V2/5（最大误差1.20e-8）；分别重建两版本概率，最大误差均0；无重复除N、门控或rank映射误入。优化器/学习率/epochs/batch/schedule均一致；rand-T025/tsds-T025/lp-mirror三模型NEM seed0结果与V2相同。

Chronos NEM LESS最终MASE1.130484→1.539824，恶化36.21%；TimeMoE1.334236→1.261603，改善5.44%。TimesFM新LESS当时尚未落盘，不用概率统计充当成绩。当前不足以说所有模型/所有任务都恶化。

| 模型 | V2 top10%轨迹质量 | V3 top10%轨迹质量 | V2 ESS | V3 ESS |
|---|---:|---:|---:|---:|
| chronos-base | 88.09% | 19.82% | 33.12 | 932.28 |
| timesfm-2.5 | 98.89% | 18.49% | 8.71 | 944.75 |
| timemoe-200m | 96.35% | 24.14% | 21.54 | 785.54 |

ESS=1/Σp_j²，是分布相当于多少条等权轨迹，非实际抽到数量；总源轨迹1082。top10%取ceil(1082*.1)=109条最高概率轨迹。V3对轨迹接近均匀但不等于rand（rand对全合法窗口均匀）。

只改均值/仍保留长度的Chronos ESS315.93；只删长度/仍使用求和 ESS16.35；两项都改ESS932.28。说明温度尺度压平已由无GPU反事实概率重算确认；删除长度也显著改变源混合。Chronos≤1500合法起点短轨迹总质量0.080%→10.620%。两项分别对下游精度的因果贡献尚未通过训练消融分离。

证据 [概率核查JSON](../../temp/less-v2-v3-audit-20260917.json)。本次仅CPU只读重算和文档记录，不改变正在运行的V3协议/温度，不重启worker；最小后续消融是固定轨迹等权，只对三梯度方法共同比较均值温度.25/.05，另设源长度开关分离效应。

<!-- v3-notes:end -->
