> LATEST split freeze (2026-09-18): [minimum kappa / full budget](v5_kappa_split_20260918.md), config v5_crb_kappa_protocol.json. V5 has NO P10, endpoint eligibility or score threshold: minimize raw-endpoint kappa, exact ties rightmost, recurse after every split and consume the entire capped budget. Zero/zero kappa is 1. Previous P10 and optional-unused-budget V5 statements below are historical. Frozen ranking/score-tie probabilities and floor/geometric-midpoint definitions are unchanged.

> FINAL decision (2026-09-18): [rightmost ties and midpoint definitions](v5_rightmost_midpoints_20260918.md), config v5_crb_rightmost_protocol.json. Exact scheduling ties prefer the rightmost interval without RNG. Real anchor probes use floor, free representatives use the geometric midpoint; error probes interpolate at the actual integer position. Earlier random-priority and pending-midpoint statements are superseded. Frozen score-tie/rank probability rules remain unchanged.

> Latest tie-policy decision (2026-09-18): [random scheduling ties](v5_random_ties_20260918.md), config v5_crb_random_ties_protocol.json. Exact priority ties use seeded random choices in cover/error/split. Earlier fixed-left V5 scheduling statements are superseded; final score ties remain equal-mass groups. Midpoint terminology is a recommendation awaiting decision, not a new representative rule.

> Latest CRB override (2026-09-18): [approved decisions](v5_decisions_20260918.md). V5 now uses closest_zero_v5: frozen P10, strict endpoint > P10, recursive current-minimum splitting; eligible-pool exhaustion may leave budget unused. Ranking freeze is unchanged. Earlier CRB budget-exhaustion descriptions below are historical; V4 remains unchanged.

## 2026-09-18 V5方法定义：原始梯度代表、参考评分权重、对象排名与源数据量先验

> 用户已批准并正式冻结（2026-09-18）。冻结确认见[v5_approved_freeze_20260918.md](v5_approved_freeze_20260918.md)，机器配置见[协议配置](../../configs/rags/v5_rank_probability_protocol.json)。后续参数变更须新协议身份；正式V5训练尚未启动。

本节按用户最新重述冻结V5方法的结构，覆盖此前相关建议中的未定状态；它是新版本定义，不修改或重新解释在途V4。统一排名温度函数现已在[v5_rank_temperature_20260918.md](v5_rank_temperature_20260918.md)确定：指数对象排名区间质量、默认T=1/(2ln4)，排名阶段前半质量80%；并列整体分配、源长度最后相乘并必须归一化。独立函数已实现，但V5正式训练尚未配置或运行。完整训练矩阵、RAGS-Traj是否纳入本批次等编排信息也不从本条重述自动推断。

### 1. 公共定义与梯度表示

源对象拥有m_j个不同合法窗口起点，源数据量先验q_j=m_j/Σ_k m_k。LESS源对象是一条可行源轨迹；本节RAGS源对象是一个CRB源块。窗口数均按对象实际拥有的合法起点计数，不按原始点数、576窗口当量或梯度锚点数计数。源块无重复覆盖共同窗口起点，参考块也按独占合法起点计数。

每条参考轨迹等权，参考块权重alpha_rb=(1/N_ref)×(L_rb/L_r)；LESS一条参考轨迹就是一个评委，内部占比为1，因此alpha_r=1/N_ref。参考alpha只用于参考意见得分阶段，源q只用于排名部署之后的曝光质量阶段。可提前计算/存储权重，但不将它们预乘锚点梯度，不改变代表构造，也不将源q加入原始得分或排名分位。

锚点梯度来自该合法窗口的实际BP。当前共用资产保存固定投影的梯度特征，所谓保留范数是“不对每个锚点先做单位方向归一化”，不是声称投影特征范数严格等于全参数梯度范数。源/参考同一模型、loss、投影与梯度协议，保留现行数值策略；梯度代理与实际AdamW更新不混称。

### 2. LESS-Main V5

对源池与参考池的每条可行轨迹，按配置预算在合法起点列表内均匀播撒锚点。沿用现实现时，K为CRBConfig.budgets(M)各阶段预算之和，各阶段使用round并做可行数量截断；均匀锚点序号使用np.rint(linspace(0,M-1,K))，包含两端。round/rint的恰好半整数按就近偶数取整，不是ceil/floor任选；M=1时仅一个起点。V5如更改取整，须另行明确，而不是让不同执行入口任意选择。

取得锚点真实BP梯度特征后，轨迹代表为原始锚点特征算术平均：gbar_t=(1/K_t)Σ_k g_tk。刻意保留各锚点模长影响，不先平均单位方向。进入匹配时再归一化轨迹代表；零向量沿用零方向约定。

对源轨迹j，计算每个参考轨迹余弦c_rj，先乘alpha_r=1/N_ref，再取最大：S_j^LESS=max_r(alpha_r c_rj)。这覆盖前版LESS-Main“Max前不乘1/N_ref”的定义，仅适用于V5。因所有参考alpha相同，S_j^LESS=(1/N_ref)max_r c_rj，与旧原始Max排序和并列关系完全相同；后续只取排名时，不改变最终部署概率。仍称LESS适配基线，保留其子任务级归一化/跨子任务Max逻辑，不宣称完整原生复现。

### 3. RAGS-Block V5

沿用现CRB配置与预算耗尽策略取得实际锚点。按时间/合法起点顺序排列，相邻的两个锚点界定一个最小块；不是在所有锚点对中寻找全局距离最小的一对。块采用左闭右开归属，最后一块包含右端点；单起点轨迹为退化单点块。合法起点完整保留，不重复归属。

块代表由两端原始锚点梯度特征的线性插值中点给出：g_mid=(g_left+g_right)/2，然后d_mid=g_mid/||g_mid||。中点是几何区间中点的插值代表，不必是整数起点；全部BP预算在锚点准备阶段耗尽；代表构造及之后所有评分/排名/概率步骤绝不允许追加任何BP，缺资产直接报错，最终代表只用已有锚点插值。如果CRB过程中已对某整数中点BP，它成为锚点并参与后续细分，但不把该操作混称为最终块中点代表。端点先单位化再平均不属于本版定义。此项与当前CRBEngine._interval_mean/_make_block一致。

令c_ij为参考块i与源块j方向的余弦，alpha_i为上述参考双层权重，四种最终分数：

| 变体 | 最终源块得分 |
|---|---|
| Max | S_j=max_i(alpha_i c_ij) |
| Linear | S_j=Σ_i alpha_i c_ij |
| ReLU | S_j=Σ_i alpha_i max(c_ij,0) |
| Exp | S_j=Σ_i alpha_i exp(c_ij) |

Exp按用户重述保留内层exp(c)，即内层温度1；排名温度另行定义。ReLU先截断原始余弦再乘正alpha，与先乘alpha再ReLU数学等价。Max明确为加权择优，有参考块细分依赖，不把它宣称为严格覆盖测度的累计聚合；保留这一性质作为变体语义。

### 4. 共同部署：最终得分→纯对象排名→共同函数→源长度权重→再次归一化

对每个方法先按上述规则完整计算S_j，只保留其对象排名与并列关系，丢弃分数间距。排名分位按对象个数定义，不以m_j再次加权排名。排名概率函数记为R_T，得到中间对象质量p_tilde_j=R_T(rank(S))_j。

最终可采样的对象概率必须是

\[
P_j=\frac{q_j\widetilde p_j}{\sum_kq_k\widetilde p_k}
=\frac{m_j\widetilde p_j}{\sum_km_k\widetilde p_k}.
\]

q_j×p_tilde_j只是尚未归一化的采样质量，不叫新的“最终得分”，也不能直接当作总和为1的概率。选中对象后，在其m_j个起点内均匀抽窗。若匹配排名质量相同，对象总曝光正比m_j，每起点曝光相同；q仅应用一次，训练抽窗不再乘长度。

共同排名函数已按v5_rank_temperature_20260918.md确定并验证，使用对象排名区间指数曲线，默认T_rank≈.3606737602；长度前的50%→80%成为默认设计目标，边界并列偏差显式报告。此前T_rank=.05重放保持旧身份，不是V5正式默认。加入q后，固定对象排名曲线不保证固定百分比对象仍占固定最终概率，需将排名偏好曲线与最终数据量加权曝光分开报告。

现rank_probability(counts=m)同时将m用于分位与最终质量，尚不符合本版“纯对象排名后才乘q”的分离定义。其324份已有缓存重放保留历史身份，独立V5排名模块rank_v5.py已实现，正式训练入口接入/配置/训练待实施，不将现有重放冒充V5结果。

---


<!-- V5 current acceptance approved (items 8/9) -->

## V5 当前验收与历史边界（2026-09-18，第 8、9 项最终决策）

本节适用于已冻结的 LESS/RAGS 评分与排名概率路径，不自动更改外部基线原生算法、V4 在途实验或正式 V5 矩阵范围。

### 数学表述修正

参考权重 $\alpha_i\ge0$ 且 $\sum_i\alpha_i=1$。加权 Exp 是 $s_j^{\mathrm{exp}}=\sum_i\alpha_i e^{c_{ij}}=1+\sum_i\alpha_i\operatorname{expm1}(c_{ij})$，零意见 $c_{ij}=0$ 时得分为 1，不是 $N_{\mathrm{ref}}$ 量级的常数。余弦在 $[-1,1]$ 时，Exp 得分在 $[e^{-1},e]$，ReLU 得分在 $[0,1]$；Linear 和 Max 则可能为负。ReLU 对每个意见截断属于评分定义，不是对最终采样概率设置正分门槛。数学错误直接修复，不登记为方法或温度变更。

### 当前正确性验收

1. **资产与对象。** 源/参考池、模型状态、窗口协议、投影、评分和 CRB/排名版本身份完整且一致。源对象拥有的合法起点不重不漏，$m_j$ 为正整数；源/参考为空、得分或资产非有限、所需资产缺失时明确报错。缺失资产不得以额外 BP 补齐。整数 BP 点与几何插值代表按最新决定区分。
2. **评分与参考话语权。** 验证参考轨迹等权、块内合法起点占比，以及每条参考轨迹权重之和为 $1/N_{\mathrm{ref}}$、全部专家权重之和为 1。验证 LESS 加权 Max 与四种 RAGS 聚合的准确公式和加权顺序。参考权重只影响意见得分；不要求每位参考在最终源采样概率中拥有固定贡献份额。
3. **排名与温度。** 先完整评分，再只保留对象排序和精确并列关系；长度不参与分位轴。验证对象分位区间指数概率、默认 $T=1/(2\ln4)$、并列组整体区间质量均分、单调变换和对象排列不变性，以及中间质量总和为 1。前半对象 80% 只在完整对象/并列组边界对齐时严格成立；报告奇数对象和边界并列偏差。评分并列不按最右规则打破。
4. **源先验与最终概率。** 验证 $P_j=m_jp_j/\sum_km_kp_k$，源合法起点量只乘一次，必须重新归一化；最终概率非负、有限且总和为 1。保存/加载须保持概率、对象身份、合法起点归属和完整冻结配置。全部同分时 $P_j=m_j/\sum_km_k$，同分对象每窗偏好相同。全非正得分也是合法排名输入，不额外设置正分门槛、top-k 或第二次 Softmax。
5. **实际采样。** 直接按 $P_j$ 抽源对象，内部均匀抽其拥有的合法起点，窗口概率为 $P_j/m_j$。检查对象/窗口频率与该分布一致，允许有限抽样波动，不要求每个窗口都访问。报告中间排名强度、长度加权后的最终对象/窗口集中度、实际唯一曝光和重复率；正确性验收不证明 MASE 必然更好。
6. **成本与实施状态。** 记录各阶段预算、实际不同锚点数、未用量、停止原因与梯度/评分/训练耗时。严格 P10 门槛下不假记预算已经用满。正式矩阵、成本配平和训练范围另行登记，不因更新验收自动启动实验。

### 降级为历史的条款

以下不再约束 V5：逐参考行条件概率及归一化；参考总曝光固定为 $\pi_j$；联合质量 $a_{jk}=\pi_jP(k\mid j)$ 和逐行守恒 $\sum_k a_{jk}=\pi_j$；三级参考入口抽样与 $\beta=\sum_j\pi_jP(\cdot\mid j)$ 边缘化的等价性验收；非正配对概率置零；全非正参考行报告不可行。它们仅用于追溯当时版本，旧函数、旧测试和旧结果保留对应身份，不删除或改名冒充 V5。

V5 中所有参考意见完成评分聚合后，再对源对象排名构造概率；没有逐参考概率入口，因此也不存在必须通过的逐参考采样份额检查。旧条款中的真实通用要求——归一化、保存加载一致、合法窗口身份和频率检查——按上面的 V5 公式重新验收。
