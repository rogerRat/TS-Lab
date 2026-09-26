> LATEST split freeze (2026-09-18): [minimum kappa / full budget](v5_kappa_split_20260918.md), config v5_crb_kappa_protocol.json. V5 has NO P10, endpoint eligibility or score threshold: minimize raw-endpoint kappa, exact ties rightmost, recurse after every split and consume the entire capped budget. Zero/zero kappa is 1. Previous P10 and optional-unused-budget V5 statements below are historical. Frozen ranking/score-tie probabilities and floor/geometric-midpoint definitions are unchanged.

> FINAL decision (2026-09-18): [rightmost ties and midpoint definitions](v5_rightmost_midpoints_20260918.md), config v5_crb_rightmost_protocol.json. Exact scheduling ties prefer the rightmost interval without RNG. Real anchor probes use floor, free representatives use the geometric midpoint; error probes interpolate at the actual integer position. Earlier random-priority and pending-midpoint statements are superseded. Frozen score-tie/rank probability rules remain unchanged.

> Latest tie-policy decision (2026-09-18): [random scheduling ties](v5_random_ties_20260918.md), config v5_crb_random_ties_protocol.json. Exact priority ties use seeded random choices in cover/error/split. Earlier fixed-left V5 scheduling statements are superseded; final score ties remain equal-mass groups. Midpoint terminology is a recommendation awaiting decision, not a new representative rule.

> Latest CRB override (2026-09-18): [approved decisions](v5_decisions_20260918.md). V5 now uses closest_zero_v5: frozen P10, strict endpoint > P10, recursive current-minimum splitting; eligible-pool exhaustion may leave budget unused. Ranking freeze is unchanged. Earlier CRB budget-exhaustion descriptions below are historical; V4 remains unchanged.

## 2026-09-18 用户批准：V5方法与排名概率协议正式冻结

用户已明确批准当前方案。本节确认前文V5方法定义、参考评分权重、源数据量先验及共同排名函数为已批准的冻结方案，覆盖此前“建议/待决定”的相关表述。协议身份为v5-rank-probability-frozen-20260918；批准方法并不表示已经启动正式V5训练，在途V4保持冻结原协议。

冻结内容：

- 原始锚点梯度保留模长，LESS轨迹内原始均值、RAGS块两端原始梯度插值中点；只有构造代表后进入匹配时才方向归一化。BP预算仅在锚点准备阶段使用，耗尽后绝不追加任何BP，缺准备资产直接报错。
- 参考双层权重仅进入评分：LESS先1/N_ref再Max；RAGS Max为max(alpha*c)，Linear为Σalpha*c，ReLU为Σalpha*ReLU(c)，Exp为Σalpha*exp(c)，Exp内层温度1。源长度不进入这些评分。
- 完成各方法评分后，只保留对象排名和精确并列关系，分位按对象数定义，不按源长度加权。共同累计曲线F_T(x)=(1-exp(-x/T))/(1-exp(-1/T))；按对象排名区间分配中间质量，无第二次Softmax，无新增正分门控。
- 默认T_rank=1/(2ln4)=0.36067376022224085，长度前前50%对象目标质量80%，各方法/模型/任务共用，不按对象数调整。精确并列整组区间分配后组内均分，不按ID或随机扰动打破；边界并列偏差报告，不为达成80%制造虚假次序。
- 源侧q_j=m_j/Σm按对象实际拥有的不同合法起点计数，仅最后进入一次：P_j=q_j*p_tilde_j/Σ(q*p_tilde)。必须再次归一化，随后对象内均匀抽合法起点。80%只约束中间排名曲线，不保证长度加权后的最终对象概率仍满足此比例。

机器可读冻结配置：远程/root/TS3/configs/rags/v5_rank_probability_protocol.json。实现src/rags/experiments/rank_v5.py新增frozen_rank_protocol_v5、validate_frozen_rank_protocol_v5与正式调用入口apply_frozen_rank_v5；入口逐字段校验冻结配置并保存完整协议身份，拒绝静默改变温度/先验/并列规则/BP边界。正式选集与新run必须携带该协议及原始评分/资产身份，不能沿用V4或此前加权分位重放的概率身份。不使用hash。

更改默认温度、曲线、排名测度、权重位置或并列政策，应建立新协议身份和新产物目录，不能回写本冻结协议。底层函数的显式temperature参数仅用于隔离的敏感性分析，不允许通过它覆盖正式配置。后续完整V5矩阵/训练入口接入仍须明确，不能将概率重放当作训练成绩。

验证：21项测试通过（0.25s），包括原19项数学/采样不变性验证，以及冻结参数偏差拒绝和完整冻结身份序列化验证。已有90份缓存排名概率重放、BP=0/训练=0的记录保持不变。本次未启动正式训练、未改在途V4，未提交/Push。

---


### 2026-09-18 Acceptance clarification approved: review items 8/9

Mathematical text corrected: normalized weighted Exp has zero-opinion baseline 1, ReLU is nonnegative. These are text corrections, not new scoring or temperature revisions. V5 acceptance follows score aggregation, pure object-percentile interval probabilities, source counts applied once with normalization, and uniform owned-window sampling. Per-reference probability budgets/row conservation, sign gating, no-positive-row infeasibility and conditional beta marginalization are historical requirements only. Full checklist: [V5 method definition](v5_method_definition_20260918.md).

All nine reviewed differences now have user decisions. Frozen ranking protocol and latest rightmost CRB identity unchanged. Formal V5 training matrix and end-to-end entry remain unlaunched; no training outcome is implied by document or pure-array validation.
