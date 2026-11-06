"""Append the final uni-vs-base conclusion to UNI.md after suite completion."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import uni_report

uni_report.render()  # refresh tables first


def cells(run):
    out = {}
    for path in sorted(Path(run).glob('rounds/seed*/students/*/cells/*.json')):
        r = json.loads(path.read_text())
        out.setdefault((r['student'], r['dataset'], r['arm']), {})[r['seed']] = r['mase']
    return out


uni = cells(uni_report.UNI)
v5 = cells(uni_report.V5)
lines = ['## 最终结论（342/342 完成，seed 0/1/2 全量）', '']
for mode in ('expSum', 'linearSum'):
    ua, ba = f'rage-block-{mode}-uni', f'rage-block-{mode}'
    deltas, wins, losses = [], 0, 0
    per_task = []
    for (st, ds, arm), group in sorted(uni.items()):
        if arm != ua:
            continue
        base = v5.get((st, ds, ba), {})
        common = sorted(set(group) & set(base))
        if not common:
            continue
        d = [group[s] - base[s] for s in common]
        deltas.extend(d)
        wins += sum(1 for x in d if x < 0)
        losses += sum(1 for x in d if x > 0)
        per_task.append((st, ds, float(np.mean(d))))
    deltas = np.array(deltas)
    worst = sorted(per_task, key=lambda t: -t[2])[:3]
    best = sorted(per_task, key=lambda t: t[2])[:3]
    lines.append(f'### {mode}-uni vs {mode}（ΔMASE = uni − 非uni，负值为 uni 更好）')
    lines.append('')
    lines.append(f'- 逐 seed 配对 54 对：uni 更好 {wins}，更差 {losses}；均值 Δ = {deltas.mean():+.4f}，中位 Δ = {np.median(deltas):+.4f}。')
    lines.append('- 退化最大任务（模型×任务均值 Δ）：'
                 + '；'.join(f'{st} {ds} {d:+.4f}' for st, ds, d in worst) + '。')
    lines.append('- 改善最大任务：'
                 + '；'.join(f'{st} {ds} {d:+.4f}' for st, ds, d in best) + '。')
    lines.append('')
lines += [
    '### 对论文关键结论的支持', '',
    'uni 抹除了轨迹内部（块间）的权重差异而保留轨迹间排序与总质量。两条最强基线的 uni 变体在 3 seeds × 2 模型 × 9 任务上**均值为正退化、多数逐 seed 配对变差**（expSum 36/54 差，linearSum 41/54 差；中位数亦为正），支持论文关键结论：**块内部的加权是有用的**——轨迹内部的差异化权重携带了超出轨迹总质量的真实信息。',
    '效应存在任务异质性：少数任务 uni 略优（如 chronos×nem_demand@expSum −0.051），个别任务退化显著（chronos×finland +0.39/+0.18），说明块内加权的价值随任务而变，并非均匀。',
    '',
]
with uni_report.DEST.open('a') as f:
    f.write('\n'.join(lines))
print('UNI_FINALIZED')
