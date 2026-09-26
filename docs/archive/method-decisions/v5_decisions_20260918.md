> LATEST split freeze (2026-09-18): [minimum kappa / full budget](v5_kappa_split_20260918.md), config v5_crb_kappa_protocol.json. V5 has NO P10, endpoint eligibility or score threshold: minimize raw-endpoint kappa, exact ties rightmost, recurse after every split and consume the entire capped budget. Zero/zero kappa is 1. Previous P10 and optional-unused-budget V5 statements below are historical. Frozen ranking/score-tie probabilities and floor/geometric-midpoint definitions are unchanged.

> FINAL decision (2026-09-18): [rightmost ties and midpoint definitions](v5_rightmost_midpoints_20260918.md), config v5_crb_rightmost_protocol.json. Exact scheduling ties prefer the rightmost interval without RNG. Real anchor probes use floor, free representatives use the geometric midpoint; error probes interpolate at the actual integer position. Earlier random-priority and pending-midpoint statements are superseded. Frozen score-tie/rank probability rules remain unchanged.

> Latest tie-policy decision (2026-09-18): [random scheduling ties](v5_random_ties_20260918.md), config v5_crb_random_ties_protocol.json. Exact priority ties use seeded random choices in cover/error/split. Earlier fixed-left V5 scheduling statements are superseded; final score ties remain equal-mass groups. Midpoint terminology is a recommendation awaiting decision, not a new representative rule.

## V5 最新决策：恢复冻结排名概率，更新 P10 分裂（2026-09-18）

本节覆盖下方历史记录中相冲突的 V5 描述，不改动正在运行的 V4。

- **恢复排名温度冻结方案。** 先计算各自完整分数，再只保留对象排序与完全相等的并列关系。对象平权分位轴 $[0,1]$、指数累计函数 $F_T(x)=(1-e^{-x/T})/(1-e^{-1/T})$、默认 $T=1/(2\ln4)$。无并列时每个对象取得其 $1/Q$ 分位区间的概率差；并列组取得整组区间的概率差并按对象数均分。协议仍为 `v5-rank-probability-frozen-20260918` / `object-rank-exponential-interval-v5`，不改为整数秩 Softmax 或平均秩指数。
- **强度表述修正。** “前 50% 对象获得 80%”仅指完整对象及并列组边界可对齐时的中间排名概率；最终概率 $P_j=m_jp_j/\sum_km_kp_k$ 仍受合法起点量与并列/对象粒度影响。排名分位不按长度加权，源长度只在最后乘一次并归一化。
- **新分裂规则。** 独立声明 `split_policy=closest_zero_v5`。仅用该轨迹前两阶段不同真实锚点梯度模长冻结线性插值 P10；至少一个端点模长**严格大于** P10、内部仍可采集的区间允许分裂。持续选当前端点平均梯度模长最小的候选，新增一次真实中点锚点梯度后二分，子区间按其真实端点重新检查资格并进入下一次竞争。阈值不更新、所有区间保留。新增 BP 仅限锚点准备预算；准备结束后代表、匹配、概率阶段无 BP。
- **预算不制造空转。** 候选存在则继续使用预算；预算用尽或有资格的可分裂区间耗尽则停止，记录 used/unused/stop reason，三阶段不挪用。硬门槛不能保证任何梯度场都耗尽预算；所有锚点模长相等时无端点严格大于 P10，分裂为零。若需严格等实际 BP 的 LESS 对照，需另行决定是否按实测锚点数配置其预算；否则报告各自实际 BP，不把未用预算假记为已使用。
- **范围。** `closest_zero_v2/v3/v4` 保留精确复现；本轮不启动正式 V5、不切换在跑 V4。其它尚待决差异保留待决。

### 实现与验证

独立 CRB 配置：`configs/rags/v5_crb_split_protocol.json`，身份 `v5-crb-strict-p10-recursive-20260918`。实现为 `src/rags/crb.py` 的 `closest_zero_v5`，其它路径保留。每次二分后立即重选当前最小，新子区间可能抢先于旧并列候选。完全相同优先级暂沿用左端位置顺序；本次不改覆盖/风险阶段并列规则。跨轨迹 CRBEngineMulti 保持批量执行与逐轨迹语义一致。

测试：`tests/test_crb_split_v5.py`、`test_crb_split_one_shot.py`、`test_crb_split_antagonist.py`、`test_crb_split_budget.py`、`test_crb_trajectory_p10.py`、`test_rags_crb_multi.py`、`test_rank_v5.py`、`test_crb.py` 共 106 项通过（3.21s）。包括独立递归最小值 oracle、多轨迹等价、阈值冻结、零/等模长候选耗尽、新子区间抢先、无丢弃与窗口无重漏，以及旧版本/排名回归。旧版本的未知 policy 测试改用真正未知名称，以便允许新增 v5。

没有设置跨阶段余量回填。可供后续决策：候选耗尽后将余量转给风险探测，在有剩余合法位置时用满总锚点预算。这属于预算转移规则变更，本轮未落地。可进一步比较抵消比例 `||gL+gR||/(||gL||+||gR||)` 与绝对中点模长优先级，以区分一强一弱和方向相反；该替代亦未落地。原审阅其它待决项不在本次决策范围；§5.2 的数学表述错误仅随该段恢复直接纠正。正式 V5 配置/训练尚未接入，未启动训练，未提交/Push。
