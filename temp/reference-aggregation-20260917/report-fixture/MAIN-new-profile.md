# MAIN-v3 — 主实验

- 更新时间：2026-09-17T22:51:09+08:00
- 配置：`configs/rage/main_reference_opinion_20260917.json`；运行：`temp/reference-aggregation-20260917/report-fixture`
- 所有使用采样温度的默认方法（lp-mirror / less-main / rage-main / rage-traj）统一 T=0.05。
- 九任务、三模型；矩阵 orig / rand / tsds / lp-mirror / less-main / rage-main / rage-traj；仅 seed 0。
- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，批量分裂规则不变。
- 新reference-opinion-v1：LESS参考max；RAGE主zeroClipSum；显式后缀记录聚合规则。
- 原生逐窗口梯度沿用已有字段校验缓存；V3 选集、训练和评测新生，不复制旧 cells/checkpoints。
- V1/V2 内容完整归档，继续参考：[MAIN-v1](archive/MAIN-v1.md)、[MAIN-v2](archive/MAIN-v2.md)。

完成 0/189 cells。

| 模型 | 已完成 cells | 当前状态 |
|---|---:|---|
| chronos-base | 0 | 准备完成，等待授权，尚未启动 |
| timesfm-2.5 | 0 | 准备完成，等待授权，尚未启动 |
| timemoe-200m | 0 | 准备完成，等待授权，尚未启动 |
<!-- v3-patrol:start -->
## 巡检记录与解读（每 30 分钟自动更新）

- 巡检时间：2026-09-17T22:51:09+08:00（北京时间）；以实际 cells/state 为准。
- 总进度：0/189 cells；尚未启动。

### 结果（按任务分节，每节三模型各一表；两列百分比分别为均值相对 orig 与 rand 的改善，正=更好；胜/n 为逐 seed 配对胜 rand）

### time_nem_demand_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_azure_i_cpu_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_darwin_water_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_finland_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_eweld_15t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_cphl_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_nem_price_5t

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_solar_h

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

### time_port_activity_d

#### chronos-base

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timesfm-2.5

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

#### timemoe-200m

| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |
|---|---:|---:|---:|---:|---:|---:|
| orig | 0 | — | — | — | — | — |
| rand | 0 | — | — | — | — | — |
| tsds | 0 | — | — | — | — | — |
| lp-mirror | 0 | — | — | — | — | — |
| less-main | 0 | — | — | — | — | — |
| rage-main | 0 | — | — | — | — | — |
| rage-traj | 0 | — | — | — | — | — |

**解读**：

以本版真实落盘结果更新；没有结果时不作效果判断。

<!-- v3-patrol:end -->

| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |
|---|---|---|---:|---:|---:|
| — | — | — | 0 | — | — |

<!-- v3-notes:start -->
## 固定方法协议（reference-opinion-v1）

| 方法 | 单位 | 聚合 | 竞争 | 温度 |
|---|---|---|---|---|
| orig | — | — | — | — |
| rand | — | — | — | — |
| tsds | — | — | — | — |
| lp-mirror | — | — | — | .05 |
| less-main | trajectory | maxSum | global | .05 |
| rage-main | block | zeroClipSum | global | .05 |
| rage-traj | trajectory | zeroClipSum | global | .05 |

RAGE-Main暂定zeroClipSum，LESS-Main恢复跨参考max；没有用最终测试集优选主项。
expSum内层exp(raw cosine)，加权求和后外层Softmax；max不乘参考alpha，zeroClip不等于源支持门控。
历史reference-mean线性批次保留原产物，以-linearSum展示；本批独立选集/训练，不改旧批次。
<!-- v3-notes:end -->
