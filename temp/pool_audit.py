import gc
import json
from pathlib import Path
import numpy as np

run = Path('/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917')
inputs = Path(json.loads((run/'protocol.json').read_text())['inputs'])
manifest = json.loads((inputs/'pool_manifest.json').read_text())
summary = {k:v for k,v in manifest.items() if not isinstance(v, (list, dict))}
summary['manifest_keys'] = list(manifest)
summary['trajectories'] = len(manifest['lengths'])
summary['length_quantiles'] = np.quantile(manifest['lengths'], [0,.25,.5,.75,1]).tolist()
with np.load(inputs/'pool.npz', allow_pickle=False) as data:
    summary['entity_examples'] = data['entities'][:12].tolist()
refs = {}
for file in sorted(inputs.glob('*__reference.npz')):
    with np.load(file, allow_pickle=False) as data:
        lengths = np.diff(data['offsets'])
        refs[file.stem] = dict(trajectories=len(lengths), points=int(lengths.sum()),
            legal_windows=int(np.maximum(lengths-575,0).sum()))
config = json.loads((run/'protocol.json').read_text())
rows = []
for student in config['students']:
    settings = config['settings'][student]
    draws = int(np.ceil(settings['epochs']*manifest['total_equivalents']/settings['batch_size']))*settings['batch_size']
    for path in sorted((run/'students'/student/'selections').glob('*.npz')):
        if not any(task in path.name for task in ['time_nem_demand_5t', 'time_azure_i_cpu_5t']):
            continue
        with np.load(path, allow_pickle=False) as data:
            p = data['probabilities']
            counts = np.diff(data['unit_offsets'])
            positions = data['unit_positions']
            ids = data['window_ids'][:,0]
            metadata = json.loads(str(data['resampling_metadata']))
            q = np.zeros(len(ids), dtype=np.float64)
            q[positions] = np.repeat(p/counts, counts)
            _, inverse = np.unique(ids, return_inverse=True)
            trajectory_p = np.bincount(inverse, weights=q)
            positive = p>0
            expected_unique = float((counts[positive] * -np.expm1(draws*np.log1p(-p[positive]/counts[positive]))).sum())
            row = dict(student=student, selection=path.stem, scoring_unit=metadata.get('scoring_unit'),
                draws=draws, supported_windows=int(counts[positive].sum()),
                window_ess=float(1/(p*p/counts).sum()), trajectory_ess=float(1/(trajectory_p**2).sum()),
                top10_trajectory_mass=float(np.sort(trajectory_p)[-10:].sum()),
                top10pct_trajectory_mass=float(np.sort(trajectory_p)[-int(np.ceil(len(trajectory_p)*.1)):].sum()),
                expected_unique_starts=expected_unique,
                draws_duplicate_fraction=1-expected_unique/draws)
            diagnostic = json.loads(str(data['diagnostics']))
            if 'reference_audit' in diagnostic:
                audit = diagnostic['reference_audit']
                if 'rows' in audit:
                    ar = audit['rows']
                    if ar and 'nonpositive_mass' in ar[0]:
                        row['nonpositive_joint_mass'] = sum(a['nonpositive_mass'] for a in ar)
                if 'source_scores' in audit:
                    scores = np.asarray(audit['source_scores'])
                    row['global_score_quantiles'] = np.quantile(scores, [0,.5,.9,.99,1]).tolist()
            cell = run/'students'/student/'cells'/(path.stem+'__seed0.json')
            if cell.exists():
                result = json.loads(cell.read_text())
                row['mase'] = result.get('mase')
                row['actual_train'] = result.get('train',{}).get('source',{})
                row['actual_train'] = {k:v for k,v in row['actual_train'].items() if k in
                    ('steps','sample_exposures','budget_windows','epochs','resumed_from_step')}
            rows.append(row)
            print(json.dumps(row), flush=True)
        del p, counts, positions, ids, inverse, q, trajectory_p, metadata, diagnostic
        gc.collect()
output = dict(pool=summary, references=refs, selections=rows)
Path('temp/pool-audit-20260917.json').write_text(json.dumps(output, indent=2))
print('POOL', json.dumps(summary))
print('REFS', json.dumps(refs))
