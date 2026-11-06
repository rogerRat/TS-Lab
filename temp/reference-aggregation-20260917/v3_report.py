"""Publish MAIN-v3 in MAIN-v2's task/model table layout."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import tempfile
from pathlib import Path

import numpy as np

from .full_pool_run import summarize
from .v3_protocol import V3_MATRIX, v3_temperatures


METHOD_NOTES = '''
## V3 固定方法协议

| 方法 | 评分单位 | 参考权重 | 竞争与概率 |
|---|---|---|---|
| orig | 不选择；预训练模型直接评测 | 不适用 | 不训练 |
| rand | 全部合法起点 | 不适用 | 延续均匀窗口基线 |
| tsds | 原生 TSDS | 原方法不变 | 原生分布不变 |
| lp-mirror | 窗口 | 原方法不变 | 延续 V2 lpv_t025 的 raw-value 分布 |
| less-main | 轨迹 | 每参考轨迹 1/N | 无门控、Global、平均余弦、Softmax |
| rage-main | 块 | (1/N)×块合法起点占所属轨迹比例 | 无门控、Global、两级评委、Softmax |
| rage-traj | 轨迹 | 每参考轨迹 1/N | 无门控、Global、平均余弦、Softmax |

N 指当前参考数据集的参考轨迹数；参考话语权总和为 1。权重只用于分数相加，
最终 P_j=softmax(S_j/T)，不乘源单位长度，不对 Softmax 输出再次分配参考预算。
less-main/rage-main/rage-traj 统一 T=0.25；该温度沿用此前数值，尚未针对平均评分重新调参。

less-main 对按 rounded linspace 均匀选取的原生窗口投影梯度做 FP32 算术平均，
再分别归一化轨迹代表并算余弦。rage-traj 同样直接平均实际计算的原生 CRB 锚点梯度，
不使用旧版的全窗口线性重构均值；每轨迹梯度数量核对同一配置预算，唯一设计差异是取点位置。
rage-main 保留块级线性中点代表。CRB 覆盖阶段最长区间二分，误差阶段积分风险优先，
分裂阶段允许批量最小模长 frontier，耗尽可用预算，p10 只审计，不筛分裂区间、不丢窗。

源块等权竞争不保证源轨迹等总概率：同分且块数不同的源轨迹可累积不同质量。
参考等权及长度先验删除是适应目标 Benchmark 评估单位的设计；用户在真实应用中仍主张按长度/时间点频次加权。

正式 V3 名称只使用本矩阵；旧 Global/Gateless/T025 别名不进入结果表。
不再另设与主公式相同的 Global 臂。旧概率/训练 cells/checkpoints 不复用，
仅复用逐窗口原生梯度、锚点及块资产；新评分、概率版本由 worker 验证。

## 材料、执行与追溯

- 已有三模型×九任务参考资产 27/27 完成；当前方法窗口并集缓存复用，未计算全部合法窗口梯度。
- grad_cache 使用字段校验和单写者锁；新选集/cells/checkpoints 位于独立 V3 run，旧成果完整保留。
- 两级权重、轨迹原生锚点均值、V3 标准名称及汇总流程有单元和 CPU 端到端回归验证。
- 用户先要求只准备，随后已明确授权三个模型并行推进；授权状态记录在 run/execution_control.json。
- 三模型训练配方、源池、九任务、单 seed 0 沿用既有设置；不把旧 V1/V2 数字填进 V3 结果。
'''



def display_method(method, config):
    if not config.get('opinion_aggregation_protocol'):
        return {'less-main': 'less-linearSum', 'rage-main': 'rage-main-linearSum',
                'rage-traj': 'rage-traj-linearSum'}.get(method,method)
    return method


def opinion_method_notes(config):
    lines=['## 固定方法协议（reference-opinion-v1）','',
        '| 方法 | 单位 | 聚合 | 竞争 | 温度 |','|---|---|---|---|---|']
    for name,p in config['method_protocols'].items():
        lines.append(f"| {name} | {p.get('scoring_unit','—')} | {p.get('reference_aggregation','—')} | {p.get('competition','—')} | {'.05' if name not in ('orig','rand','tsds') else '—'} |")
    lines += ['', 'RAGE-Main暂定zeroClipSum，LESS-Main恢复跨参考max；没有用最终测试集优选主项。',
        'expSum内层exp(raw cosine)，加权求和后外层Softmax；max不乘参考alpha，zeroClip不等于源支持门控。',
        '历史reference-mean线性批次保留原产物，以-linearSum展示；本批独立选集/训练，不改旧批次。']
    return '\n'.join(lines)

def render(run_dir, destination):
    run = Path(run_dir)
    config = json.loads((run/'protocol.json').read_text())
    if config['experiment_version'] != 'v3':
        raise ValueError('MAIN-v3 requires a V3 run')
    result = summarize(run, config)
    methods = tuple(config['arms'])
    total_cells = len(methods)*len(config['students'])*len(config['datasets'])*len(config['seeds'])
    opinion_profile = bool(config.get('opinion_aggregation_protocol'))
    control = json.loads((run/'execution_control.json').read_text())
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    active = control.get('training_enabled') is True
    temperatures = v3_temperatures(config)
    gradient_temperature = temperatures['less-main']
    config_label = ('configs/rage/main_reference_opinion_20260917.json' if opinion_profile else
        'configs/rage/main_v3_t005.json' if gradient_temperature == .05 else 'configs/rage/main_v3.json')
    lines = ['# MAIN-v3 — 主实验', '', f'- 更新时间：{stamp}',
        f'- 配置：`{config_label}`；运行：`{run}`',
        f'- 所有使用采样温度的默认方法（lp-mirror / less-main / rage-main / rage-traj）统一 T={gradient_temperature}。',
        '- 九任务、三模型；矩阵 '+ ' / '.join(display_method(m,config) for m in methods)+'；仅 seed 0。',
        '- C512/H64/D1，CRB 2%/1%/1%；全部合法起点保留，批量分裂规则不变。',
        ('- 新reference-opinion-v1：LESS参考max；RAGE主zeroClipSum；显式后缀记录聚合规则。' if opinion_profile else
         '- 本批仍为冻结reference-mean线性协议；旧主方法以-linearSum展示，尚无新聚合规则训练结果。'),
        '- 原生逐窗口梯度沿用已有字段校验缓存；V3 选集、训练和评测新生，不复制旧 cells/checkpoints。',
        '- V1/V2 内容完整归档，继续参考：[MAIN-v1](archive/MAIN-v1.md)、[MAIN-v2](archive/MAIN-v2.md)。', '',
        f"完成 {result['completed_cells']}/{total_cells} cells。", '',
        '| 模型 | 已完成 cells | 当前状态 |', '|---|---:|---|']
    for student in config['students']:
        count = sum(r['student'] == student for r in result['cells'])
        state = run/'students'/student/'state.json'
        data = json.loads(state.read_text()) if state.exists() else {}
        status = '/'.join(str(data.get(k, '')) for k in ['stage', 'dataset', 'arm', 'seed', 'status'])
        if not data:
            status = '已授权，等待进程启动' if active else '准备完成，等待授权，尚未启动'
        lines.append(f'| {student} | {count} | {status} |')
    lines += ['<!-- v3-patrol:start -->', '## 巡检记录与解读（每 30 分钟自动更新）', '',
        f'- 巡检时间：{stamp}（北京时间）；以实际 cells/state 为准。',
        f"- 总进度：{result['completed_cells']}/{total_cells} cells；" + ('已授权并行推进。' if active else '尚未启动。'), '',
        '### 结果（按任务分节，每节三模型各一表；两列百分比分别为均值相对 orig 与 rand 的改善，正=更好；胜/n 为逐 seed 配对胜 rand）', '']
    for task in config['datasets']:
        lines += [f'### {task}', '']
        for student in config['students']:
            lines += [f'#### {student}', '',
                '| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |',
                '|---|---:|---:|---:|---:|---:|---:|']
            group = [r for r in result['cells'] if r['dataset'] == task and r['student'] == student]
            original = next((r['mase'] for r in group if r['arm'] == 'orig'), None)
            random = {r['seed']: r['mase'] for r in group if r['arm'] == 'rand'}
            entries = []
            for method in methods:
                label = display_method(method,config)
                rows = [r for r in group if r['arm'] == method]
                if not rows:
                    entries.append((-1e18, methods.index(method),
                                    f'| {label} | 0 | — | — | — | — | — |'))
                    continue
                values = [r['mase'] for r in rows]
                mean = float(np.mean(values))
                sd = f'{np.std(values, ddof=1):.6f}' if len(values) > 1 else '—'
                vs_orig = f'{100*(1-mean/original):+.2f}%' if original else '—'
                random_mean = float(np.mean(list(random.values()))) if random else None
                vs_rand = f'{100*(1-mean/random_mean):+.2f}%' if random_mean else '—'
                paired = [r['mase'] < random[r['seed']] for r in rows if r['seed'] in random]
                wins = f'{sum(paired)}/{len(paired)}' if paired and method not in ('rand', 'orig') else '—'
                # 行序：按 vs rand 改善从好到差（用户裁定 2026-09-17）；rand 自身为 0 基准，
                # 无 rand 基准或无数据的行垫底，同分按 canonical 序稳定排列
                key = (1 - mean/random_mean) if (random_mean and method != 'rand') else (0.0 if method == 'rand' else -1e18)
                entries.append((key, methods.index(method),
                                f'| {label} | {len(rows)} | {mean:.6f} | {sd} | {vs_orig} | {vs_rand} | {wins} |'))
            for _, _, row_line in sorted(entries, key=lambda e: (-e[0], e[1])):
                lines.append(row_line)
            lines += ['']
    lines += ['**解读**：', '', '以本版真实落盘结果更新；没有结果时不作效果判断。', '',
        '<!-- v3-patrol:end -->', '', '| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |',
        '|---|---|---|---:|---:|---:|']
    for r in result['cells']:
        lines.append(f"| {r['student']} | {r['dataset']} | {display_method(r['arm'],config)} | 1 | {r['mase']:.6f} | — |")
    if not result['cells']:
        lines.append('| — | — | — | 0 | — | — |')
    destination = Path(destination)
    notes = opinion_method_notes(config) if opinion_profile else METHOD_NOTES.strip()
    if gradient_temperature == .05 and not opinion_profile:
        notes = notes.replace('统一 T=0.25；该温度沿用此前数值，尚未针对平均评分重新调参。',
            '以及 lp-mirror 统一 T005（0.05）；所有使用采样温度的默认方法均为 T005。旧版所有方法展示名加 -T025 后缀，新版使用原定名称；orig/rand/tsds 无此温度，历史后缀仅标识旧协议。')
    if destination.exists():
        old = destination.read_text()
        start, end = '<!-- v3-notes:start -->', '<!-- v3-notes:end -->'
        if start in old and end in old:
            notes = old.split(start, 1)[1].split(end, 1)[0].strip()
    lines += ['', '<!-- v3-notes:start -->', notes, '<!-- v3-notes:end -->', '']
    Path('temp').mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir='temp', prefix='main-v3-', suffix='.partial', delete=False) as handle:
        handle.write('\n'.join(lines))
        temporary = Path(handle.name)
    temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', default='docs/MAIN-v3.md')
    args = parser.parse_args()
    result = render(args.run, args.out)
    print(f"MAIN-v3 updated: {result['completed_cells']}/189")


if __name__ == '__main__':
    main()
