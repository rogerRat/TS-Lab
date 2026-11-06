import json
from pathlib import Path

run = Path('/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917')
protocol = json.loads((run/'protocol.json').read_text())
rows = []
for student in protocol['students']:
    cells = run/'students'/student/'cells'
    for task in protocol['datasets']:
        original = cells/f'{task}__original__seed0.json'
        if not original.exists():
            continue
        baseline = json.loads(original.read_text())['mase']
        row = dict(student=student, task=task, original_mase=baseline, improvement_pct={})
        for arm in ['random','lessvg_t025','rage_gateless','rage_global','rage_traj_gls','rage_traj_global']:
            cell = cells/f'{task}__{arm}__seed0.json'
            if cell.exists():
                result = json.loads(cell.read_text())
                if result.get('experiment_version') != 'v2':
                    continue
                row['improvement_pct'][arm] = 100*(1-result['mase']/baseline)
        if 'random' in row['improvement_pct'] and 'lessvg_t025' in row['improvement_pct']:
            rows.append(row)
            print(json.dumps(row), flush=True)
Path('temp/selection-aggression-hypothesis-20260917.json').write_text(json.dumps(rows,indent=2))
