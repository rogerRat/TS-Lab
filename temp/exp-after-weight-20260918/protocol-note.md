## 2026-09-18 当前聚合协议：Linear/ZeroClip/Max先加权，Exp后加权

本节是现行定义，覆盖2026-09-17的“所有变体先加权”方案；历史推导、配置和结果保留，不覆盖或冒充新结果。仍为13方法、九任务、三模型、seed0，共351 cells；八个RAGE变体均中立，得到结果之前不指定RAGE主方法。本轮只更新准备材料，未启动新批次。

参考专家i与源对象j的原始余弦为c_ij。Block双层权重alpha_i=(1/N_ref)×该块拥有的合法起点数/所属参考轨迹全部合法起点数；Traj内部占比为1，即alpha_i=1/N_ref。权重总和为1，不乘源对象长度，不对最终概率再次加权。

| RAGE变体 | 外层Softmax前得分 |
|---|---|
| linearSum | S_j=Σ_i alpha_i c_ij |
| zeroClipSum | S_j=Σ_i ReLU(alpha_i c_ij)=Σ_i alpha_i ReLU(c_ij) |
| maxSum | S_j=max_i(alpha_i c_ij)，不相加 |
| expSum | S_j=Σ_i alpha_i exp(c_ij/1.00)，先exp，再双层加权求和 |

全部最后统一p_j=Softmax_j(S_j/0.05)。Exp内层温度明确固定T_inner=1.00；外层温度T_outer=0.05。不得将内层偷偷改成0.05，不得改成log-sum-exp或指数混合后直接归一化。配置和审计显式记录exp_inner_temperature=1.00、weight_placement=after_transform；其余RAGE三变体为before_transform。除Exp外，其他规则不变。

选择理由：Linear与ZeroClip保留参考内容占比；Max保留加权后择优，既尊重比例也允许长块影响胜者。Traj等权Max的1/N不改变胜者/排序，但会改变固定温度下的概率集中度，不能说概率完全不受影响。Block专家很多时原始Max可能接近饱和，各源对象最大值趋同；加权Max也可能因块权重小而缩小分差，本轮把它视作待验证变体。

Exp先加权后指数的旧式Σexp(alpha*c)，在alpha小时近似M_ref+Σalpha*c，外层Softmax消掉共同M_ref后接近linearSum。现改为Σalpha*exp(c)，alpha控制意见测度，指数负责非线性评价。同方向块细分且总权重保持不变时，新Exp得分不变。其展开为1+Σalpha*c+(1/2)Σalpha*c²+…：外层Softmax消掉共同常数1，但高阶项不再被alpha平方压小；内层T=1仍可能较温和，是否提高性能必须等待新实验，不能声称已证明优于Linear。

LESS版本边界：正在运行的旧V3 LESS是(1/N_ref)Σ_i c_ij，内核求和后在调用层除以len(reference)，不是未归一化总和。已准备的新LESS-Main仍保留此前指定的原始余弦Max，即max_i c_ij，作为原论文多子任务Max原则的适配基线。本次不更改该基线。若表示相同，RAGE-Traj-maxSum分数为LESS-Main的1/N_ref，固定T=.05下选择强度不同。新旧LESS主要变化为Mean→Max，不是新增1/N；旧线性结果展示为LESS-linearSum。

执行配置仍为configs/rage/matrix_reference_opinion_20260917.json，聚合协议更新为reference-opinion-v4-mixed-weight-order（聚合版本号，不表示已启动V4实验）。上一版保存为configs/rage/matrix_reference_opinion_v3_weighted_first_20260917.json，状态superseded_not_launched，继续通过校验。已有V3运行协议与缓存身份保留；只为Exp切换权重位置，Max/Linear/ZeroClip语义不变。没有新梯度准备、训练或评估启动，未提交/Push。

验证：99项相关回归测试通过（3条既有Swig弃用警告）。覆盖精确公式、内层T=1/外层T=.05、同向参考块细分不变性、Block/Traj生产入口、历史V3与上一版配置保护；未启动新批次。
