from pathlib import Path

path = Path('/root/TS3/docs/MAIN-v2.md')
content = path.read_text()
heading = '## NEM 各臂概率距离与 LESS 求和尺度混杂（2026-09-17）'
note = '''

## NEM 各臂概率距离与 LESS 求和尺度混杂（2026-09-17）

用户要求明确哪些 RAGE 变体接近随机。CPU只读检查当前NEM静态选集，
脚本temp/nem_probability_audit.py，结果temp/nem-probability-audit-20260917.json。
比较对象是全池9052777个合法窗口的均匀概率，不是1082条轨迹均匀概率；
完整轨迹长度不同，uniform-window本来就对长轨迹赋予更大总质量。
TVD=0.5*sum_start(abs(q_start-1/W))，0为完全均匀，百分数可理解为
需重新分配的概率质量，或任一窗口集合概率差的最大值；它不直接等于性能差。

| 当前NEM臂，全部T=.25、无hard gating | Chronos TVD | TimesFM TVD | TimeMoE TVD |
| --- | ---: | ---: | ---: |
| rage_gateless，块级独立参考预算 | 2.492% | 1.720% | 9.510% |
| rage_global，块级Global | 2.299% | 1.538% | 9.194% |
| rage_traj_gls，轨迹级独立参考预算 | 13.692% | 20.805% | 20.589% |
| rage_traj_global，轨迹级Global | 13.669% | 20.664% | 20.500% |
| lessvg_t025 | 72.613% | 84.666% | 82.896% |

结论只针对该任务、当前参数与上述四个RAGE臂，不能泛化到有hard gating的
历史原RAGE或所有温度。块级最接近均匀，轨迹级有明显偏好，TimeMoE块级
也并非完全随机。Global未使NEM明显锐化，故“独立ref预算导致平滑”不足以
解释所有现象。Global块级的窗口加权评分std分别.015211/.011704/.058419，
除以T=.25仅.06084/.04682/.23368；softmax指数变化小，q_start容易接近均匀。
块质量beta_source正比legal_count*exp(score/T)，块内再除legal_count均匀抽样，
在评分没有区分度时严格退化为uniform-window。这不是legal_count额外强制均衡。
以Chronos块Global为例，80%的窗口概率/均匀概率位于约.923–1.073，
最强块对应每窗口概率也仅约均匀的1.415倍。

关键比较混杂：baselines/less/runner.py与algorithms.py把归一化轨迹均值的
逐参考cosine求和，RAGE-Global则按参考合法窗口比例加权平均；二者同用T=.25。
NEM有5条等长reference（每条39456点），因此sum对比mean多5倍评分尺度，
相当于对reference mean使用有效温度.05。求和与均值对排名等价，对固定温度
的value softmax并不等价；不能只因名义温度一致便视为同等尖锐度。

反事实只改已有LESS评分：score/=5、T仍.25（等价原sum评分T=1.25）。
无需梯度或重新训练，其他所有因素保持：
| 模型 | 原LESS→uniform TVD | mean5 LESS→uniform TVD | mean5 LESS→RAGE-Traj TVD | mean5 LESS→RAGE-Traj-Global TVD |
| --- | ---: | ---: | ---: | ---: |
| Chronos |72.613%|13.675%|1.638%|1.623%|
| TimesFM |84.666%|21.319%|3.577%|3.605%|
| TimeMoE |82.896%|20.739%|2.729%|2.408%|
mean5 LESS窗口ESS/全池窗口数分别87.053%/71.649%/74.394%，
与RAGE轨迹级约87.1%/73.3%/74.4%接近。比较已验证单位窗口数量和每个
轨迹单位首个global窗口row一致，不能把不同单位顺序错当概率差异。

这支持NEM的LESS与RAGE-Traj最终分布差异很大部分来自尺度/有效温度；
不证明统一尺度后下游性能相等，未重训且只分析单任务。块级与轨迹级的评分
分布宽度仍显著不同，可能与局部方向噪声、参考平均抵消、表示误差有关；
需要每reference条件概率TVD/评分std与最终marginal比较，区分单行已平和
混合后变平。不可直接把低global score解读为源池不存在可行匹配。

修正此前解释：不能把LESS更尖/RAGE更平及性能差主要归因为ref预算强制多样性，
当前还存在sum与weighted-mean的量纲混杂，以及块/轨迹评分分布对同T的不同响应。
建议独立诊断臂统一参考聚合尺度，并在验证集上做温度或匹配分布集中度消融；
保留原正式臂及元数据。本次未改主实验概率、温度或已有选集。
'''
marker = '<!-- v2-notes:end -->'
if marker not in content:
    raise RuntimeError('Missing persistent notes marker')
if heading not in content:
    path.write_text(content.replace(marker, note+'\n'+marker))
print('Recorded all-arm TVD, score scale and LESS mean5 counterfactual')
