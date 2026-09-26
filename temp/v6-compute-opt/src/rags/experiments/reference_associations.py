"""Exploratory task-level associations; windows/seeds are never independent tasks."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .reference_census import read_json, publish, SCHEMA

FEATURES = (
    'within_directional_variance', 'between_directional_variance', 'total_directional_variance',
    'within_raw_variance_fraction', 'between_raw_variance_fraction',
    'trajectory_mean_cancellation', 'trajectory_mean_norm_cv', 'trajectory_mean_norm_top1_mass',
    'trajectory_mean_early_late_relative_shift',
    'trajectory_mean_lag1_direction_change_mean', 'trajectory_mean_lag64_direction_change_mean',
    'trajectory_mean_lag576_direction_change_mean', 'trajectory_mean_relative_curvature_mean',
    'trajectory_mean_crb_nonanchor_relative_rmse', 'trajectory_mean_crb_anchor_mean_relative_bias',
    'trajectory_mean_signal_normalized_step_rms', 'trajectory_mean_signal_lag1_correlation',
    'trajectory_mean_signal_standardized_early_late_mean_shift', 'trajectory_mean_signal_zero_fraction',
    'windows', 'trajectories')
ALIGNMENT_FEATURES = ('mean_pair_cosine', 'cosine_of_raw_means', 'alignment_gain_vs_rand',
                      'window_ess_fraction', 'sampled_max_fragment_share')
AGGREGATIONS = ('linearSum', 'zeroClipSum', 'maxSum', 'expSum')


def fdr_bh(pvalues):
    values = np.asarray([np.nan if p is None else p for p in pvalues], dtype=float)
    valid = np.flatnonzero(np.isfinite(values)); out = [None]*len(values)
    order = valid[np.argsort(values[valid])]
    q = np.minimum.accumulate((values[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
    for i, v in zip(order, q):
        out[int(i)] = float(min(1., v))
    return out


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x-x.mean(), y-y.mean()
    denom = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.clip(x@y/denom, -1, 1)) if denom > 1e-14 else None


def association(x, y, *, permutations=9999, bootstraps=1999, seed=20260921, control=None):
    """Two-sided task permutation; paired task bootstrap and leave-one-task-out.

    Optional control is rank-residualized and reported as descriptive only:
    observational nuisance-adjusted permutations need stronger assumptions.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 5:
        return dict(n_tasks=n, rho=None, p=None, status='insufficient_tasks_minimum_5')
    rx, ry = rankdata(x), rankdata(y)
    rho = correlation(rx, ry)
    if rho is None:
        return dict(n_tasks=n, rho=None, p=None, status='constant_predictor_or_outcome')
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.broadcast_to(ry, (permutations,n)), axis=1)
    statistics = ((shuffled-ry.mean()) @ (rx-rx.mean())) / (np.linalg.norm(rx-rx.mean())*np.linalg.norm(ry-ry.mean()))
    extreme = int(np.sum(np.abs(statistics) >= abs(rho)-1e-12))
    p = (extreme+1)/(permutations+1)
    rows = rng.integers(0, n, size=(bootstraps,n))
    bx, by = rankdata(x[rows], axis=1), rankdata(y[rows], axis=1)
    bx -= bx.mean(1,keepdims=True); by -= by.mean(1,keepdims=True)
    denom = np.linalg.norm(bx,axis=1)*np.linalg.norm(by,axis=1)
    boots = (np.einsum('ij,ij->i',bx,by)[denom>1e-14]/denom[denom>1e-14]).tolist()
    loo = [correlation(rankdata(np.delete(x,i)), rankdata(np.delete(y,i))) for i in range(n)]
    loo = [r for r in loo if r is not None]
    result = dict(n_tasks=n, rho=rho, p=float(p), status='exploratory',
        bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist() if boots else None,
        valid_bootstraps=len(boots), leave_one_task_out_range=[min(loo),max(loo)] if loo else None,
        sign_stable_leave_one_out=all(r*rho>0 for r in loo) if loo else None,
        permutations=permutations, seed=seed)
    if control is not None:
        design = np.column_stack([np.ones(n), rankdata(control)])
        ex = rx-design@np.linalg.lstsq(design, rx, rcond=None)[0]
        ey = ry-design@np.linalg.lstsq(design, ry, rcond=None)[0]
        result['partial_rho_controlling_log_reference_windows_descriptive'] = correlation(ex, ey)
    return result


def read_cells(run):
    run = Path(run); config = read_json(run/'protocol.json')
    records, missing = {}, []
    for student in config['students']:
        for task in config['datasets']:
            for arm in config['arms']:
                for seed in ([0] if arm == 'orig' else config['seeds']):
                    name = f'{task}__{arm}__seed{seed}.json'
                    candidates = [run/'students'/student/'cells'/name,
                                  run/'rounds'/f'seed{seed}'/'students'/student/'cells'/name]
                    paths = [p for p in candidates if p.is_file()]
                    if not paths:
                        missing.append(dict(student=student, dataset=task, arm=arm, seed=seed)); continue
                    values = []
                    for path in paths:
                        cell = read_json(path)
                        expected = dict(student=student, dataset=task, arm=arm, seed=seed,
                                        experiment_version=config['experiment_version'])
                        if any(cell.get(k) != v for k,v in expected.items()):
                            raise ValueError('result identity mismatch: '+str(path))
                        value = float(cell['mase'])
                        if not np.isfinite(value) or value < 0:
                            raise ValueError('invalid MASE: '+str(path))
                        values.append(value)
                    if len(set(values)) != 1:
                        raise ValueError('conflicting duplicate result cell')
                    records[(student,task,arm,seed)] = values[0]
    return config, records, missing


def paired_effect(records, student, task, arm, baseline, seeds):
    keys = [(student,task,arm,s) for s in seeds]+[(student,task,baseline,s) for s in seeds]
    if any(k not in records for k in keys):
        return None
    a = np.asarray([records[(student,task,arm,s)] for s in seeds])
    b = np.asarray([records[(student,task,baseline,s)] for s in seeds])
    if np.any(b <= 0):
        return None
    return dict(relative_gain_percent=float(100*(1-a.mean()/b.mean())),
                seed_relative_gains=(100*(1-a/b)).tolist(), seed_wins=int(np.sum(a<b)),
                paired_seeds=list(seeds), arm_mase_mean=float(a.mean()), baseline_mase_mean=float(b.mean()))


def analyze(runs, census, *, permutations=9999, bootstraps=1999, seed=20260921):
    census = Path(census); tests, effects, missing, observations = [], [], [], []
    for run in map(Path, runs):
        config, cells, absent = read_cells(run)
        missing.extend(dict(r, run=str(run), kind='cell') for r in absent)
        pairs = [(f'rags-block-{a}', f'rags-traj-{a}') for a in AGGREGATIONS]
        pairs += [(a,b) for a in config['arms'] if a.startswith('rags-')
                  for b in ('tsds','less-main','lp-mirror','rand')]
        pairs += [(f'rags-block-{a}', f'rags-block-{a}-uni') for a in ('linearSum','expSum')]
        pairs = [(a,b) for a,b in pairs if a in config['arms'] and b in config['arms']]
        for student in config['students']:
            geometry, alignments = {}, {}
            for task in config['datasets']:
                path = census/student/f'{task}__geometry.json'
                if not path.exists():
                    missing.append(dict(run=str(run), student=student, dataset=task, kind='geometry')); continue
                payload = read_json(path)
                if (payload.get('student') != student or payload.get('dataset') != task
                        or not payload.get('coverage',{}).get('complete')):
                    raise ValueError('incomplete/wrong geometry identity')
                geometry[task] = payload['summary']
                path = census/student/f'{task}__alignment__{run.name}.json'
                if path.exists():
                    alignment_record = read_json(path)
                    if (alignment_record.get('student') != student or alignment_record.get('dataset') != task
                            or Path(alignment_record.get('run','')).resolve() != run.resolve()):
                        raise ValueError('alignment run/model/task identity mismatch')
                    alignments[task] = alignment_record['alignment']
                    for arm in config['arms']:
                        if arm != 'orig' and arm not in alignments[task]:
                            missing.append(dict(run=str(run),student=student,dataset=task,arm=arm,kind='alignment_arm'))
                else:
                    missing.append(dict(run=str(run),student=student,dataset=task,kind='alignment'))
            for arm, baseline in pairs:
                current = []
                for task in config['datasets']:
                    effect = paired_effect(cells,student,task,arm,baseline,config['seeds'])
                    if effect is None:
                        continue
                    item = dict(run=str(run), student=student, dataset=task, arm=arm, baseline=baseline, **effect)
                    effects.append(item)
                    if task not in geometry:
                        continue
                    predictors = {k: geometry[task].get(k) for k in FEATURES}
                    rand = [cells.get((student,task,'rand',s)) for s in config['seeds']]
                    orig = cells.get((student,task,'orig',0))
                    predictors['orig_mase'] = orig
                    predictors['rand_mase'] = float(np.mean(rand)) if all(v is not None for v in rand) else None
                    predictors['rand_gain_vs_orig'] = (100*(1-predictors['rand_mase']/orig)
                        if orig is not None and orig>0 and predictors['rand_mase'] is not None else None)
                    aa = alignments.get(task,{}).get(arm,{}).get('metrics',{})
                    ab = alignments.get(task,{}).get(baseline,{}).get('metrics',{})
                    for key in ALIGNMENT_FEATURES:
                        predictors['arm_'+key] = aa.get(key)
                        predictors['contrast_'+key] = (aa[key]-ab[key] if aa.get(key) is not None and ab.get(key) is not None else None)
                    row = dict(item, predictors=predictors, reference_windows=geometry[task]['windows'])
                    current.append(row); observations.append(row)
                keys = sorted(set().union(*(r['predictors'].keys() for r in current)))
                for feature in keys:
                    rows = [r for r in current if r['predictors'].get(feature) is not None
                            and np.isfinite(r['predictors'][feature])]
                    result = association([r['predictors'][feature] for r in rows],
                        [r['relative_gain_percent'] for r in rows], permutations=permutations,
                        bootstraps=bootstraps, seed=seed,
                        control=np.log1p([r['reference_windows'] for r in rows]))
                    tests.append(dict(run=str(run), student=student, arm=arm, baseline=baseline,
                        feature=feature, tasks=[r['dataset'] for r in rows], **result))
    # One correction family includes ALL attempted valid associations, not
    # only favorable methods, models, predictors or signs.
    for test, q in zip(tests, fdr_bh([t['p'] for t in tests])):
        test['q_bh_all_tests'] = q
        test['significant_fdr_05'] = q is not None and q < .05
    return dict(schema=SCHEMA, inference='exploratory_association_not_causal',
        independent_unit='task_within_model_and_run; all seeds paired then aggregated',
        no_independent_pooling_of_models=True, multiple_testing='BH across every valid reported test',
        runs=list(map(str,runs)), effects=effects, tests=tests, observations=observations, missing=missing)


def write_report(result, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    publish(out/'associations.json', result)
    for key in ('effects','tests'):
        rows = result[key]
        if rows:
            columns = list(dict.fromkeys(k for r in rows for k in r))
            stream = io.StringIO(); writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
            temporary = out/(key+'.csv.partial'); temporary.write_text(stream.getvalue()); temporary.replace(out/(key+'.csv'))
    valid = [t for t in result['tests'] if t['p'] is not None]
    significant = [t for t in valid if t['significant_fdr_05']]
    lines = ['# 参考梯度与方法收益：探索性分析', '',
        f"有效检验 {len(valid)} 项；全家族 BH q<0.05 为 {len(significant)} 项；缺失记录 {len(result['missing'])} 项。", '',
        '独立统计单位为各模型/实验内的任务。同 seed 配对后计算任务收益；重叠窗口和重复 seed 不充当独立任务。',
        '各模型分别检验，不把相同任务在三个模型的观测当作 27 个独立任务。完整检验（含失败/反例）见 tests.csv 和 associations.json。',
        '这些假设已受到已有实验结果启发，所有 p/q 均为探索性证据，不能解释为因果或新的确认性验证。',
        '需一起阅读 bootstrap 区间、逐任务留一范围及控制参考长度后的描述性相关。九任务的统计功效有限。', '',
        '## 相对 rand 的任务结果', '',
        '| Run | 模型 | 任务 | 方法 | vs rand % | 配对 seed 胜数 |',
        '|---|---|---|---|---:|---:|']
    effects = [r for r in result['effects'] if r['baseline']=='rand']
    effects.sort(key=lambda r:(r['run'],r['student'],r['dataset'],-r['relative_gain_percent']))
    for r in effects:
        lines.append(f"| {Path(r['run']).name} | {r['student']} | {r['dataset']} | {r['arm']} | {r['relative_gain_percent']:+.4f} | {r['seed_wins']}/{len(r['paired_seeds'])} |")
    lines += ['', '## 检验阅读方式', '',
        'rho>0 表示该特征越高，arm 相对 baseline 的收益越大；rho<0 为反向关联。',
        'block-vs-traj 固定相同聚合器；RAGS-vs-TSDS/LESS/LP/rand 包含全部 RAGS 变体，不按任务挑最佳方法。',
        '对齐是全部参考窗口的精确期望与源采样均值的内积；源侧 Monte Carlo 误差在 alignment 产物单列。',
        '下一轮应先记录假设/所需资源/判定规则，再补资源与复核；保留负结果，新增检验需扩充校正家族。', '']
    temporary = out/'REPORT.md.partial'; temporary.write_text('\n'.join(lines)); temporary.replace(out/'REPORT.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True); p.add_argument('--census',required=True); p.add_argument('--out',required=True)
    p.add_argument('--permutations',type=int,default=9999); p.add_argument('--bootstraps',type=int,default=1999)
    p.add_argument('--seed',type=int,default=20260921)
    args = p.parse_args()
    if args.permutations < 1 or args.bootstraps < 1:
        p.error('positive permutation/bootstrap counts required')
    result = analyze(args.runs,args.census,permutations=args.permutations,bootstraps=args.bootstraps,seed=args.seed)
    write_report(result,args.out)


if __name__ == '__main__':
    main()
