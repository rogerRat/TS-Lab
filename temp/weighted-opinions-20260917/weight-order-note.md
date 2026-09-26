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
