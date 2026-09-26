import json,datetime
from pathlib import Path
import numpy as np
base=Path("/root/TS3/temp/v4-explanation-20260918")
d=json.loads((base/"diagnostics.json").read_text())
snap=json.loads((base/"results_snapshot.json").read_text())
sampling=json.loads((base/"finland_sampling_replay.json").read_text())
norms=json.loads((base/"uniform_gradient_norm_audit.json").read_text())
text="""# V4 结果解释与梯度诊断（2026-09-18）

结论：当前证据支持“Block 的局部单位方向汇总减弱净参考信号，加上源块竞争先验及固定温度导致实际采样分布不同”。NEM 的共同表示梯度诊断符合 Traj 更好这一机制；其他任务有明确反例，不能把所有 MASE 差异归结为平均方向对齐。V4 仍在 seed0，三seed结论尚未形成。

本诊断只读取既有 v4 cells/selection NPZ 与字段匹配的原生梯度缓存、冻结参考 CRB 资产。源 CRB 以生产配置进行模型无关 CPU 重放，逐块核对 owned-start 数量和起点与 v4 选集一致；两模型均 360,774 源块。未调用模型、未新增梯度、未启动训练或评测，未改变实验与巡检。临时脚本和产物全部保留在 temp/v4-explanation-20260918/。

## 1. 当前结果并不支持“Block 一直不如 Traj”

以下为配对完备 cells 的描述统计，以快照为准，不挑每个任务最好的聚合。改善=100×(1−Traj MASE/Block MASE)，正值为Traj更好。不同聚合已完成任务数不同，不能直接跨行比较总体胜率。

"""
text+=f"快照 UTC {snap['time']}；{len(snap['cells'])}/666 cells，均为seed0。\n\n"
text+="| 模型 | 聚合 | Traj胜/配对任务 | 平均改善 |\n|---|---|---:|---:|\n"
for r in snap["matched"]:text+=f"| {r['model']} | {r['aggregation']} | {r['traj_wins']}/{r['n']} | {r['mean_traj_improvement_pct']:+.2f}% |\n"
text+="""
Chronos Azure：Block-zeroClipSum 1.157895，Traj-zeroClipSum 1.251389。
Chronos EWeld：Block-expSum 1.141130，Traj-expSum 1.344618。
Chronos NEM：LESS 1.284215、Traj-maxSum 1.512409，Traj-maxSum本身反而差于LESS。
这些反例需要和Traj-linear/exp在NEM与Finland的优势一起解释。

## 2. 项目适配 LESS 与 Traj-maxSum 除了锚点，还差尺度

以R表示参考轨迹数，c为双方原始轨迹均值单位化后的余弦：
LESS评分=max_r c_rt；Traj-maxSum评分=max_r(c_rt/R)=(max_r c_rt)/R。
双方外层温度同为.05，Traj-maxSum对原始max余弦的有效温度是.05R。
R=5的NEM为.25；R=10的EWeld为.5；R=492的Azure为24.6。
因此排序即使完全相同，分布也可以差很多；Azure的Traj-maxSum几乎是轨迹均匀抽样。

以下用现有评分重新算概率，仅是分布反事实，不是新的训练成绩：
“匹配LESS尺度”把Traj分数乘R再softmax；
“匹配Traj尺度”把LESS分数除R再softmax。
TV=0表示相同，TV=1表示完全不重叠。匹配尺度后的差异仍来自两侧锚点选择及其均值表示，不能进一步区分源侧/参考侧的因果贡献。

| 模型 | 任务 | R | 实际LESS/Traj TV | 匹配LESS尺度 TV | 匹配Traj尺度 TV |
|---|---|---:|---:|---:|---:|
"""
for m in d["models"]:
 for r in m["probabilities"]:
  if r["method"]=="rags-traj-maxSum":text+=f"| {m['model']} | {r['task']} | {r['reference_count']} | {r['tv_to_less_actual']:.4f} | {r['tv_anchor_only_at_less_scale']:.4f} | {r['tv_anchor_only_at_traj_scale']:.4f} |\n"
text+="""
Finland只有一个参考：尺度因素消失，linearSum与maxSum相同；其LESS/Traj概率差异确实来自锚点（Chronos TV=.282，TimesFM TV=.209）。CRB选择误差/分裂处的自适应锚点，当前Traj对所有不同锚点等权算术平均，未按窗口覆盖测度重建/校正；它与均匀抽样的LESS估计目标并不严格相同。相同K只保证计算预算相同，不能保证估计偏差相同。

## 3. 参考异质性：归一化和平均的次序改变有效信号

Block线性评分写为 S_b=u(g_b)·Q_B，Q_B=Σ_r(1/R)Σ_k(L_rk/L_r)u(g_rk)。
Traj评分使用 Q_T=Σ_r(1/R)u(mean_anchor g_r)，归一化发生在轨迹内部平均之后。
定义轨迹内coherence=||Σ_k(L_rk/L_r)u(g_rk)||；单位方向一致时为1，分散/抵消时下降。
Q范数是净参考选择信号强度，不是原始平均梯度模长，也不是梯度估计误差。
同向参考块细分并保持权重总和，对linear/zeroClip及当前exp评分不产生稀释；方向异质性会减弱linear净信号。

| 模型 | 任务 | 平均轨迹内coherence | ||Q_B|| | ||Q_T|| | cos(Q_B,Q_T) |
|---|---|---:|---:|---:|---:|
"""
for m in d["models"]:
 for task,r in m["references"].items():text+=f"| {m['model']} | {task} | {np.mean(r['within_coherence']):.4f} | {r['block_Q_norm']:.4f} | {r['traj_Q_norm']:.4f} | {r['Q_direction_cosine']:.4f} |\n"
text+="""
NEM在同一参考轨迹内已经高度分散；Chronos两种Q方向近乎相同(.995)，但Block模长只有Traj约1/4，说明“指向哪里”相似而选择信号强度显著不同。TimesFM既有模长差异，也有方向变化(.860)。Traj消除了轨迹内部平均后净方向的模长差异，这可能放大稳定净模式，也可能放大极弱净方向，不能先验认定一定更可靠。

零截断消除负意见后仍可能缺乏候选间差异。Exp内层温度1，Σαexp(c)=1+Σαc+0.5Σαc²+…；共同常数1被外层softmax消去，较小余弦下仍可能接近线性。Max先乘块权重，参考块权重小会让logits几乎完全平坦。这些因素不能通过“分数全为正”判断选择信号是否充足。

## 4. 分布证据与源先验

以下为NEM，std(S)/T衡量logit差异，TV对各方法原生单位均匀分布，ESS为聚合到轨迹后的有效轨迹数。
原生单位不同，TV仅用于判断各分支是否缺乏选择强度；公平跨方法比较需统一到窗口/轨迹空间。

| 模型 | 方法 | std(S)/T | TV原生单位均匀 | 轨迹ESS | top10%轨迹质量 |
|---|---|---:|---:|---:|---:|
"""
for m in d["models"]:
 for r in m["probabilities"]:
  if r["task"]=="time_nem_demand_5t":text+=f"| {m['model']} | {r['method']} | {r['logit_std']:.6f} | {r['tv_unit_uniform']:.6f} | {r['trajectory_ess']:.1f} | {r['top10pct_mass']:.3f} |\n"
text+="""
Block实际按块竞争，轨迹总质量P_t=Σ_b∈t p_b；分数接近相同时，P_t约为该轨迹块数/总块数。源轨迹长度不同，块数多的轨迹会获得更多总质量。Traj按轨迹竞争，轨迹内部按合法窗口均匀抽样。两侧固定温度相同不等于采样集中程度/源先验相同。
NEM Chronos Block-linear选中轨迹的期望合法长度约24,232，Traj-linear约8,823。没有显式源长度乘子并不消除这种竞争单位带来的影响。
Block-max也不等于rand：它是近块均匀，rand是全体合法窗口均匀；块大小不完全相同。Chronos NEM两者窗口分布TV约.244。

## 5. 同一梯度表示下，源池与选择分布是否对齐

为避免“用Traj自己的评分证明Traj更好”，所有方法使用同一生产源块原始中点梯度作为窗口常值代理。
参考 G_R=(1/R)Σ_r Σ_b∈r(L_b/L_r)g_b。
Block的选择期望 G_p=Σ_b p_b g_b；
Traj/LESS的选择期望 G_p=Σ_t p_t Σ_b∈t(L_b/L_t)g_b。
这对应概率分布的期望梯度，当前训练并没有一次性固定的硬子集。
逐窗真实梯度未全部计算，本指标是重建代理；它不包含实际抽样噪声、训练后梯度变化或AdamW状态。

| 模型 | 任务 | 窗口均匀源池 cos | 轨迹均匀源池 cos | Block-linear cos | Traj-linear cos | Block-exp cos | Traj-exp cos |
|---|---|---:|---:|---:|---:|---:|---:|
"""
for m in d["models"]:
 for task in m["references"]:
  rows={r["method"]:r for r in m["alignment"] if r["task"]==task}
  vals=[rows[k]["raw_mean_cosine"] for k in ["uniform_windows","uniform_trajectories","rags-block-linearSum","rags-traj-linearSum","rags-block-expSum","rags-traj-expSum"]]
  text+=f"| {m['model']} | {task} | "+" | ".join(f"{v:.4f}" for v in vals)+" |\n"
text+="""
NEM支持Traj选择后的梯度更对齐：Chronos均匀池.268→Block-linear.436→Traj-linear.550；TimesFM为−.052→−.048→+.174，exp下Traj为+.182。
这也表明源池平均方向接近正交/反向不等于源池不可用，重加权仍能选出更对齐的方向；池均值不是可行性上限。
Finland Chronos反例：Block-linear平均梯度余弦.521，高于Traj-linear的.486，但MASE更差。Traj原始点积更大(.03086 vs .02026)，余弦单独忽略模长。
EWeld的Traj-exp期望原始点积更大，却明显输给Block-exp。源池均值本身较对齐，过度集中及遗漏其他目标模式都是待验证假说，不能由单个初始方向指标判定原因。

局部方向/模長差异还需区分：E_p[cos(g,G_R)]与cos(E_p[g],G_R)不相同；前者单位化后平均，后者先按原始模长平均。
缓存4096维投影余弦并非精确全参数几何。对于初始参考平均损失的一步SGD，小步长下原始点积G_R·G_p给出一阶下降代理；真实训练是AdamW、梯度裁剪、多步BF16训练，最终评估目标是MASE，当前梯度诊断不能直接等同最终成绩。

## 6. 原始梯度模长审计

从相同预算均匀源锚点读取原生逐窗口缓存梯度，无模型调用。下表是投影梯度模长，不是输入/输出值模长。

| 模型 | 样本数 | median ||g|| | p99 ||g|| | max ||g|| | top1%占Σ||g|| |
|---|---:|---:|---:|---:|---:|
"""
for model,r in norms.items():
 q=r["gradient_norm_quantiles"];text+=f"| {model} | {r['n']} | {q[1]:.5g} | {q[3]:.5g} | {q[5]:.5g} | {r['top1pct_share_sum_norm']:.2%} |\n"
text+="""
TimesFM未裁剪平均梯度的解释需要结合极端模长审计。余弦评分丢失模长，原始均值又可能被少数巨大梯度支配；训练的裁剪和AdamW会改变其影响。下一步要比较裁剪后批梯度或实际参数更新与参考的内积。模长跨度本身不证明缓存损坏或当前实现错误；补充输入尺度审计给出了幅度来源线索，具体影响仍待验证。

## 7. 有些大成绩差异来自不同训练轨迹，初始均值解释不足

Chronos Finland Traj-linear/zeroClip概率TV=.0012107，期望梯度cos均约.4855，原始点积相差约.16%；最终MASE却是1.143717 vs1.397324。
按生产CDF和轨迹内均匀抽样、同一seed0重放1018×32次，窗口序列同位置相同率95.79%，约4.21%窗口不同，第一批已有1/32窗口不同。CDF同随机数并非最大耦合，小概率质量变化可在很多单位边界改变抽样。
这提供了数据序列分化的直接证据，但没有证明单凭这4.21%就导致全部成绩差异。没有发现两方法温度/训练预算不同；兩者均32576样本暴露/1018步。checkpoint成绩亦明显分化，linear在.5/1/1.5/2 epoch的MASE约1.046/1.472/1.113/1.144；zeroClip约1.556/1.544/1.638/1.397。当前无法归因于确定的理论机制，应等待配对多seed分布并进一步核查训练轨迹。

## 8. 最小可解释消融（建议，未启动）

1. LESS/Traj-max先匹配有效温度：Traj设T=.05/R匹配LESS当前强度；或LESS设T=.05R匹配Traj当前强度。R=1任务是锚点差异的现有对照。固定预算、训练配方并做多seed配对。
2. 源先验与粒度分开：源先按轨迹竞争，再在轨迹内按块或窗条件概率抽样，保留Block局部方向。或诊断性地除源块数，区分“局部匹配”与“长轨迹/多块轨迹质量”贡献；此项改变目标先验，需显式单独协议。
3. 参考内部聚合与源粒度分开：Block源×Traj参考、Traj源×Block参考，与现有两臂形成2×2。对参考汇总范数或logit集中程度做匹配，用来拆开归一化次序、方向变化和尺度因素。
4. 同一组未用于选择的合法参考/源窗口上计算真实原生梯度，检查重建误差、范数异常、选择后的真实均值/裁剪后批均值；记录若干步真实更新的参考损失改善及checkpoint MASE。训练后目标漂移需要重新测量，不能只看起点。
5. 不以某个任务事后最优聚合定义主方法。三seed先报告同聚合、同任务配对差及不确定性。

本轮完整结构化结果：diagnostics.json、results_snapshot.json、uniform_gradient_norm_audit.json、finland_sampling_replay.json；复现脚本diagnose.py和audit_norms.py。完整各任务/方法概率与梯度数值保留在JSON；本报告仅摘取解释需要的数值。尚未提交/Push，未更改正式代码、配置、结果或在途实验。
"""
scale=json.loads((base/"timesfm_input_scale_audit.json").read_text())
text+="""
## 补充：TimesFM大梯度与源输入尺度的直接关联

从1,191,062个已缓存源窗口中，取固定分布控制样本2000个及最大模长100个，共2100个窗口，仅读取输入和缓存。log10梯度模长与log10窗口标准差的Pearson相关约.946，与log10最大绝对输入约.911；这不是随机抽样总体相关，且不构成干预因果证据。最大的10个梯度都来自fid799/entity100848__ind_3，其窗口标准差约445万–454万，最大绝对输入1490万，缓存投影梯度模长可达约2.00e15。

本地实现src/gradient/timesfm.py:per_window_losses直接对原始尺度mean_predictions与future计算MSE并加pinball；src/adapters/foundation/timesfm.py训练使用固定版本Transformers官方原生loss，在已安装modeling_timesfm2_5.py:849亦为预测与原始future的MSE。由此推断，源序列单位/幅度差异和原始尺度MSE很可能是极端梯度的重要来源，尚未做归一化干预验证。MASE是按序列尺度归一化的评估，原生MSE与平均MASE的一阶梯度目标并不相同。建议按输入尺度分层报告选择质量，并检查裁剪后的批更新是否被这些序列主导；不能直接把原始梯度均值方向解释为平均MASE改善方向。完整样本身份及数值见timesfm_input_scale_audit.json；复现audit_input_scale.py。
"""
(base/"report.md").write_text(text)
print(base/"report.md")

