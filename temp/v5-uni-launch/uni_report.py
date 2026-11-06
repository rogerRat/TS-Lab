"""Build docs/UNI.md in the exact MAIN.md layout: per-task sections, per-model tables."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

UNI = Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920')
V5 = Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918')
DEST = Path('docs/UNI.md')
UNI_ARMS = ['rage-block-expSum-uni', 'rage-block-linearSum-uni']
V5_BASES = ['rage-block-expSum', 'rage-block-linearSum']
BASELINES = ['rand', 'orig', 'less-main', 'tsds', 'lp-mirror']
ALL_ARMS = UNI_ARMS + V5_BASES + BASELINES


def cells(run):
    """Collect (student, dataset, arm) -> {seed: mase} across seed rounds."""
    out = {}
    for path in sorted(run.glob('rounds/seed*/students/*/cells/*.json')):
        record = json.loads(path.read_text())
        key = (record['student'], record['dataset'], record['arm'])
        out.setdefault(key, {})[record['seed']] = record['mase']
    return out


def render():
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    uni_cells, v5_cells = cells(UNI), cells(V5)
    config = json.loads((UNI / 'protocol.json').read_text()) if (UNI / 'protocol.json').exists() \
        else json.loads(Path('configs/rage/main_v5_uni.json').read_text())
    students, datasets = config['students'], config['datasets']
    state = json.loads((UNI / 'state.json').read_text()) if (UNI / 'state.json').exists() else {}
    total = len(students) * len(datasets) * (1 + 6 * 3)
    done = sum(len(v) for v in uni_cells.values())
    lines = [
        '# UNI — v5-uni 实验三：轨迹内部均匀化（RAGE-uni）', '',
        f'- 更新时间：{now}',
        '- 配置：`configs/rage/main_v5_uni.json`；运行：`/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920`',
        f'- 进度：{done}/{total} cells；状态：{state.get("status", "starting")}（已完成 seeds：{state.get("completed_seeds", [])}）',
        '- 定义：uni 不是块也不是轨迹方法。先按基线块方法得到全部块概率（等价于块概率加和到轨迹），再把每条轨迹内部各块差异抹除——轨迹总质量在其块间均分，块内仍均匀抽合法窗口。轨迹间排名、冻结 Rank 曲线、训练配方、种子、模型、ckpt 与 v5 完全一致，只改变方法。',
        '- 新臂：`rage-block-expSum-uni`、`rage-block-linearSum-uni`（v5 两个最强块方法的 uni 变体）；同跑保留 rand/orig/less-main/tsds/lp-mirror 基线。',
        '- 梯度资产：硬链接复用 v5 同模型/同投影/同字段缓存（零新增 BP），`gradient_reuse.json` 强制缓存命中。',
        '- 对照：`docs/MAIN.md`（v5 最终结果，666/666 完成）。',
        '',
        '## 结果展示（与 MAIN.md 同格式）', '',
        '1. 表格按相对 rand 改善降序，改善率 = 100×(1−方法 MASE 均值/rand MASE 均值)，正值更好；rand 自身 0%，orig 同列排序。',
        '2. uni 行为本实验（v5-uni）结果；带 † 的行为 MAIN.md（v5）最终结果，用于逐格对照。vs orig / vs rand 优先使用本 run 自身基线，缺失时回退 v5 基线（两者协议、种子、模型、ckpt 完全一致）。',
        '3. 胜/seeds 为逐 seed 配对胜 rand；单 seed 时 sd 显示 —。',
        '',
        '### 结果（按任务分节，每节各模型一表）', '']
    for dataset in datasets:
        lines.append(f'#### {dataset}')
        lines.append('')
        for student in students:
            lines.append(f'##### {student}')
            lines.append('')
            lines.append('| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|')
            uni_rand = uni_cells.get((student, dataset, 'rand'), {})
            v5_rand = v5_cells.get((student, dataset, 'rand'), {})
            rand = uni_rand or v5_rand
            rand_by_seed = uni_rand if uni_rand else v5_rand
            uni_orig = uni_cells.get((student, dataset, 'orig'), {})
            v5_orig = v5_cells.get((student, dataset, 'orig'), {})
            orig = uni_orig or v5_orig
            rand_mean = float(np.mean(list(rand.values()))) if rand else None
            orig_mean = float(np.mean(list(orig.values()))) if orig else None
            rows = []
            for arm in ALL_ARMS:
                from_uni = arm in UNI_ARMS or (arm in BASELINES and (student, dataset, arm) in uni_cells)
                group = (uni_cells if from_uni else v5_cells).get((student, dataset, arm), {})
                if not group:
                    continue
                values = list(group.values())
                mean = float(np.mean(values))
                sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
                vs_rand = 100 * (1 - mean / rand_mean) if rand_mean else None
                vs_orig = 100 * (1 - mean / orig_mean) if orig_mean else None
                wins = (sum(1 for seed, mase in group.items()
                            if seed in rand_by_seed and mase < rand_by_seed[seed])
                        if arm != 'rand' else None)
                rows.append(dict(arm=arm, from_uni=from_uni, n=len(values), mean=mean, sd=sd,
                                 vs_rand=vs_rand, vs_orig=vs_orig, wins=wins))
            rows.sort(key=lambda r: (-(r['vs_rand'] if r['vs_rand'] is not None else float('-inf')), r['arm']))
            for r in rows:
                name = r['arm'] if r['from_uni'] else r['arm'] + ' †'
                sd_text = f"{r['sd']:.6f}" if r['sd'] is not None else '—'
                vs_orig_text = f"{r['vs_orig']:+.2f}%" if r['vs_orig'] is not None else '—'
                vs_rand_text = f"{r['vs_rand']:+.2f}%" if r['vs_rand'] is not None else '—'
                wins_text = f"{r['wins']}/{r['n']}" if r['wins'] is not None else '—'
                lines.append(f"| {name} | {r['n']} | {r['mean']:.6f} | {sd_text} | {vs_orig_text} | {vs_rand_text} | {wins_text} |")
            lines.append('')
    lines += ['## 执行与追溯', '',
        '- 代码：`f699b7f`（rage 分支）；uni 变换 `rage-intra-trajectory-uniform-v1`；测试 `tests/test_uni_selection.py` 12 项 + 既有选集/恢复测试扩展。',
        '- 身份隔离：v5-uni cell 记录 `experiment_version=v5-uni`；v5 冻结矩阵（13 臂）不被 uni 扩展；uni 选集带 `intra_trajectory_uniformization` 元数据，缓存校验区分基线/uni 身份。',
        '- 预算与 CRB：cover/error/split 2%/1%/1%，min_kappa、最右并列、预算严格耗尽，与 v5 逐字段一致。', '']
    DEST.write_text('\n'.join(lines) + '\n')
    return dict(uni_cells=done, total=total)


if __name__ == '__main__':
    print(render())
