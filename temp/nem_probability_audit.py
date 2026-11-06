import json
from pathlib import Path
import numpy as np

run = Path('/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917')
rows = []
for student in ['chronos-base', 'timesfm-2.5', 'timemoe-200m']:
    for arm in ['rage_gateless','rage_global','rage_traj_gls','rage_traj_global','lessvg_t025']:
        path = run/'students'/student/'selections'/f'time_nem_demand_5t__{arm}.npz'
        with np.load(path, allow_pickle=False) as data:
            p = data['probabilities']
            c = np.diff(data['unit_offsets']).astype(np.float64)
            total = c.sum()
            ratio = p*total/c
            order = np.argsort(ratio)
            cumulative = np.cumsum(c[order])/total
            weighted_q = ratio[order][np.searchsorted(cumulative, [.01,.1,.5,.9,.99])]
            row = dict(student=student, arm=arm,
                tv_to_uniform=float(np.abs(p-c/total).sum()/2),
                ess_fraction=float(1/(total*(p*p/c).sum())),
                probability_ratio_to_uniform_quantiles=weighted_q.tolist(),
                min_ratio=float(ratio.min()), max_ratio=float(ratio.max()))
            if arm == 'lessvg_t025':
                metadata = json.loads(str(data['resampling_metadata']))
                scores = np.asarray(metadata['native_scores'])
                mean = float((scores*c).sum()/total)
                std = float(np.sqrt(((scores-mean)**2*c).sum()/total))
                logits = scores/5/.25 + np.log(c)
                new_p = np.exp(logits-logits.max())
                new_p /= new_p.sum()
                row.update(less_score_std=std, less_score_std_over_temperature=std/.25,
                    counterfactual_less_mean5_tv=float(np.abs(new_p-c/total).sum()/2),
                    counterfactual_less_mean5_ess_fraction=float(1/(total*(new_p*new_p/c).sum())))
                starts = data['unit_positions'][data['unit_offsets'][:-1]]
                comparisons = {}
                for target in ['rage_traj_gls', 'rage_traj_global']:
                    with np.load(path.with_name(f'time_nem_demand_5t__{target}.npz'),
                                 allow_pickle=False) as other:
                        other_counts = np.diff(other['unit_offsets'])
                        other_starts = other['unit_positions'][other['unit_offsets'][:-1]]
                        assert np.array_equal(c, other_counts)
                        assert np.array_equal(starts, other_starts)
                        comparisons[target] = float(np.abs(new_p-other['probabilities']).sum()/2)
                row['tv_less_mean5_to_rage_trajectories'] = comparisons
            if arm in ['rage_global', 'rage_traj_global']:
                audit = json.loads(str(data['diagnostics']))['reference_audit']
                scores = np.asarray(audit['source_scores'])
                mean = float((scores*c).sum()/total)
                std = float(np.sqrt(((scores-mean)**2*c).sum()/total))
                row.update(window_weighted_score_mean=mean, window_weighted_score_std=std,
                           score_std_over_temperature=std/.25)
            rows.append(row)
            print(json.dumps(row), flush=True)
Path('temp/nem-probability-audit-20260917.json').write_text(json.dumps(rows,indent=2))
