"""Publish V5 using the exact archived V4 result-table rendering function."""
import ast
from datetime import datetime, timezone, timedelta
import fcntl
import json
from pathlib import Path
import tempfile
import numpy as np
from rags.experiments.suite_report import collect

RUN = Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918')
DEST = Path('docs/MAIN.md')
CONSTRAINTS = '''## 结果展示与执行顺序（强制约束）

1. 所有结果表（任务模型表、末尾跨任务汇总表及后续新增结果表）严格按相对 rand 改善从好到差降序排列，改善率为 100×(1−方法 MASE 均值/rand MASE 均值)，正值更好。rand 自身为 0%；orig 也按同一改善率排序。无结果或缺少 rand 基准的行统一置于末尾；改善率完全相同时按固定顺序稳定排列。不能用绝对 MASE、方法名或完成时间代替此排序。
2. 执行优先级为：种子 → 下游任务 → 该任务的全部方法 cell。先完成 seed0 的全部对应实验，再完成 seed1，最后 seed2；当前种子的全部模型、任务、方法完成训练与评测后才允许进入下一种子。轮内两模型并行；每个模型严格按配置的任务顺序，完成当前任务的全部方法 cell 后再推进下一任务，不跨任务轮流跑同一种方法。两模型各自按任务推进，种子整轮屏障覆盖两模型。已完成 cell 不重跑；orig 是只测一次的种子无关基线，已提前完成的基线保留，后续种子只引用。
'''

def notes(config):
    lines = ['## 固定方法协议（V5 冻结方案）', '',
             '| 方法 | 单位 | 聚合 | 竞争 | 温度 |', '|---|---|---|---|---|']
    for name in config['arms']:
        p = config['method_protocols'][name]
        temperature = '—' if name in ('orig', 'rand', 'tsds') else '.05（原生）' if name == 'lp-mirror' else '1/(2ln4)（Rank）'
        lines.append(f"| {name} | {p.get('scoring_unit', '—')} | {p.get('reference_aggregation', '—')} | {p.get('competition', '—')} | {temperature} |")
    lines += ['', 'Block/Traj 四种聚合均保留，未指定 RAGS 主方法。LESS-Main 采用参考轨迹加权后 max。',
              '参考话语权只进入评分：轨迹 1/N；块 (1/N)×块合法起点占所属参考轨迹比例。Max 为 max(alpha*cos)，Linear 为 sum(alpha*cos)，ReLU 为 sum(alpha*relu(cos))，Exp 为 sum(alpha*exp(cos/1.0))，内层T=1.00。',
              'LESS/RAGS 最终分数只保留对象排序及精确并列。冻结 F_T(x)=(1-exp(-x/T))/(1-exp(-1/T))，T=1/(2ln4)；每个对象获得其等宽排名区间的积分质量，并列组获得合并区间质量后组内等分。随后乘源对象合法窗口起点数量并归一化。前 50% 对象获得 80% 质量的约束作用于源长度先验之前，边界并列或奇数对象数需按冻结协议解释。没有最终 Softmax；LP-Mirror 和 TSDS 保持原生分布。',
              'CRB 递归选择最小 kappa 区间，精确并列优先最右，实际探测取 floor，中点代表使用原始端点梯度线性插值，不追加 BP，无 P10，严格耗尽可用预算。轨迹代表保留范数后平均，再归一化进入匹配。',
              '仅汇总独立 V5 正式 cells；orig 只计 seed0 一次，后续轮基线引用不重复计数。样本标准差使用 ddof=1，单 seed 显示 —；胜/seeds 仅使用同 seed 的 rand 配对，MASE 越低越好。', '',
              '## 材料、执行与追溯', '', DEST.read_text().strip()]
    return '\n'.join(lines)

def build_renderer():
    source = Path('temp/v3_report.py').read_text()
    tree = ast.parse(source)
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'render')
    text = ast.get_source_segment(source, function)
    text = text.replace("if version not in ('v3','v4'):", "if version != 'v5':")
    text = text.replace("MAIN report requires a V3 or V4 run", "MAIN report requires a V5 run")
    start = text.index('    config_label = ')
    end = text.index('    lines = ', start)
    text = text[:start] + "    config_label = 'configs/rags/main.json'\n" + text[end:]
    text = text.replace("f'- 所有使用采样温度的默认方法（lp-mirror / less-main / rags-main / rags-traj）统一 T={gradient_temperature}。'", "f'- LESS/RAGS 统一对象 Rank 温度 T=1/(2ln4)={gradient_temperature:.12f}；LP-Mirror 原生 T=.05，TSDS 原生分布不变。'")
    text = text.replace('全部合法起点保留，批量分裂规则不变。', '全部合法起点保留，递归最小 kappa 分裂，优先最右，预算严格耗尽。')
    text = text.replace('V4轮间复用', 'V5轮间复用')
    text = text.replace('每 30 分钟自动更新', '每 30 分钟巡检更新')
    text = text.replace("if version=='v4':", "if version=='v5':")
    text = text.replace("'<\x21-- v4-'", "'<\x21-- v5-'")
    text = text.replace("prefix='main-v3-'", "prefix='main-'")
    # The V4 renderer's table body and sort/statistics expressions stay intact.
    def summarize(run, config):
        result = collect(run)
        states = {}
        for seed in config['seeds']:
            for student in config['students']:
                state = run/'rounds'/f'seed{seed}'/'students'/student/'state.json'
                if state.exists():
                    states[student] = state
        # Redirect only the old template's state reads, never create result links.
        return result
    text = text.replace("state = run/'students'/student/'state.json'", "state = next((run/'rounds'/f'seed{seed}'/'students'/student/'state.json' for seed in reversed(config['seeds']) if (run/'rounds'/f'seed{seed}'/'students'/student/'state.json').exists()), run/'students'/student/'state.json')")
    namespace = dict(Path=Path, json=json, np=np, tempfile=tempfile,
        datetime=datetime, timezone=timezone, timedelta=timedelta,
        summarize=summarize, display_method=lambda m,c:m,
        v3_temperatures=lambda c:{'less-main':1/(2*np.log(4))},
        opinion_method_notes=notes, METHOD_NOTES='')
    exec(compile(text, 'archived-V4-table-template', 'exec'), namespace)
    return namespace['render']

def verify(result, config):
    text = DEST.read_text()
    assert text.startswith('# MAIN-v5 — 主实验')
    assert '<!-- v5-patrol:start -->' in text and '<!-- v5-notes:start -->' in text
    header = '| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |'
    assert text.count(header) == len(config['datasets'])*len(config['students']) == 18
    assert f"完成 {result['completed_cells']}/666 cells。" in text
    assert sum(x['arm']=='orig' for x in result['cells']) == 18
    task = student = None
    count = 0
    for section in text.split(header)[1:]:
        table = section.split('\n\n',1)[0]
        rows = [line for line in table.splitlines() if line.startswith('| ') and not line.startswith('|---')]
        assert len(rows) == 13
        values = [float(line.split('|')[6].strip().rstrip('%')) for line in rows if line.split('|')[6].strip()!='—']
        assert values == sorted(values, reverse=True)
        count += len(rows)
    assert count == 234
    assert '| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |' in text
    summary = text.split('| 模型 | 任务 | 方法 | 完成 seeds | MASE 均值 | sd |',1)[1].split('<!-- v5-notes:start -->',1)[0]
    random = {}
    for row in result['cells']:
        if row['arm']=='rand':
            random.setdefault((row['student'],row['dataset']),[]).append(row['mase'])
    keys = []
    for line in summary.splitlines():
        if not line.startswith('| '):
            continue
        columns = [x.strip() for x in line.split('|')[1:-1]]
        student, task, method = columns[:3]
        values = [x['mase'] for x in result['cells'] if (x['student'],x['dataset'],x['arm'])==(student,task,method)]
        assert int(columns[3])==len(values)
        assert columns[4]==f'{np.mean(values):.6f}'
        baseline = random.get((student, task))
        keys.append(1-float(np.mean(values))/float(np.mean(baseline)) if baseline else -1e18)
    assert keys == sorted(keys,reverse=True)
    return dict(completed_cells=result['completed_cells'], training_cells=sum(x['arm']!='orig' for x in result['cells']), task_model_tables=18, method_rows=count)

def main():
    with (RUN/'main_report.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = json.loads((RUN/'protocol.json').read_text())
        result = build_renderer()(RUN, DEST)
        content = DEST.read_text()
        anchor = f"\n完成 {result['completed_cells']}/666 cells。"
        assert anchor in content
        content = content.replace(anchor, '\n'+CONSTRAINTS+'\n'+anchor, 1)
        with tempfile.NamedTemporaryFile(mode='w', dir='temp', prefix='main-constraints-', suffix='.partial', delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.replace(DEST)
        checked = verify(result, config)
        stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
        Path('docs/Progress.md').open('a').write(f"\n### {stamp} MAIN V5 results updated in exact V4 table layout\n- Reused the archived V4/V3 rendering function's table columns, task/model sections, mean/sample-sd, vs-orig/vs-rand formula, paired wins, missing entries and descending-vs-rand sorts. V5 collect reads only current formal cells and deduplicates orig. Snapshot: {checked}. Preserved prior MAIN operational/protocol notes at the end; changed metadata to actual frozen V5 rank/volume/kappa rules. No experiment process, selection, protocol or training change. Generator: temp/v5-report/main_report.py; future patrols can refresh the same file.\n")
        print(json.dumps(checked))

if __name__ == '__main__':
    main()
