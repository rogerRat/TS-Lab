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
