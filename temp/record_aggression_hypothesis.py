from pathlib import Path

path = Path('/root/TS3/docs/MAIN-v2.md')
content = path.read_text()
heading = '## 用户解释：筛选激进性与迁移收益符号（2026-09-17）'
note = '''

## 用户解释：筛选激进性与迁移收益符号（2026-09-17）

用户提出：当候选源的迁移收益整体为正，LESS更尖的分布集中有益模式并放大收益；
当迁移收益为负，更尖的分布可能放大负迁移，而RAGE较平的分布减轻该集中损害。
这是值得验证的解释，但“接近Random”只直接预测接近Random，并不保证比Random
更安全；原轨迹是否有益也不能由Random整体提升推出所有源轨迹都有益。

只读核对当前v2 seed0已完成且同时有Random/LESS的5个model-task结果。
收益定义100*(1-MASE_arm/MASE_original)，正值改善、负值退化。
脚本temp/check_selection_hypothesis.py，结果
temp/selection-aggression-hypothesis-20260917.json。

| model/task | Random | LESS | RAGE块级gateless | 块级Global |
| --- | ---: | ---: | ---: | ---: |
| Chronos / NEM |+18.16%|+34.56%|+10.39%|+21.31%|
| TimesFM / NEM |+17.61%|+23.72%|+17.64%|+15.96%|
| TimeMoE / NEM |-11.84%|-25.21%|-14.31%|-16.00%|
| TimeMoE / Azure |-20.77%|-26.82%|-27.37%|-28.84%|
| TimeMoE / Darwin |-21.16%|-22.90%|-20.25%|未完成|

NEM三模型符合用户描述的方向：正迁移两模型LESS更好，负迁移TimeMoE
LESS退化更大，RAGE介于LESS与Random之间。但Azure块级RAGE比LESS退化略大，
Darwin只有小幅方向差异，不能称为所有任务的统一规律。
Chronos NEM块级RAGE与uniform TVD仅2.49%，性能却与Random相差约7.76个百分点，
也说明概率接近不能等同单seed训练结果相同：有限随机抽样、单位映射、优化非线性
及参考/表示差异仍存在。已有反事实仅隔离求和/平均对概率尖锐度的影响，尚未
隔离其对最终训练性能的影响。当前证据不能区分有益梯度集中、过拟合或负迁移。

下一因果检验应保持同一表示与参考聚合规则，仅改变softmax有效温度/评分尺度，
固定源池、训练曝光、学习率序列、评估口径，并保留当前正式结果。可分别将
LESS评分改mean（等价T乘参考轨迹数），及RAGE-Traj-Global在NEM改sum
（等价T除5），比较选择集中度和验证集收益响应。块级仍需按每参考轨迹
内合法起点比例加权、轨迹之间求和，而非直接乘全部参考块数。
本次未修改主实验科学参数或增加训练。
'''
marker = '<!-- v2-notes:end -->'
if marker not in content:
    raise RuntimeError('Missing persistent notes marker')
if heading not in content:
    path.write_text(content.replace(marker,note+'\n'+marker))
print('Recorded hypothesis, current results and counterexamples')
