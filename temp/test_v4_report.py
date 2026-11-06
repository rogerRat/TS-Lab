import json
from pathlib import Path

from rage.experiments.students import atomic_json
from rage.experiments.v4_seed_rounds import round_config
from rage.experiments.v4_report import render
from test_v4_seed_rounds import publish


def test_identical_template_sorted_results_and_unique_baseline(tmp_path):
    c=json.loads(Path('configs/rage/main_v4.json').read_text())
    atomic_json(tmp_path/'protocol.json',c)
    atomic_json(tmp_path/'execution_control.json',dict(training_enabled=True))
    for seed in (0,1):
        folder=tmp_path/'rounds'/f'seed{seed}';publish(round_config(c,seed),folder)
        for p in folder.glob('students/*/cells/*.json'):
            row=json.loads(p.read_text())
            row['mase']=10. if row['arm']=='rand' else 12. if row['arm']=='rage-block-maxSum' else 7.
            atomic_json(p,row)
    output=tmp_path/'MAIN-v4.md';result=render(tmp_path,output);text=output.read_text()
    assert result['completed_cells']==450
    assert '450/666 cells' in text
    assert text.startswith('# MAIN-v4 — 主实验')
    assert '<!-- v4-patrol:start -->' in text
    assert '| 方法 | seeds | MASE 均值 | sd | vs orig | vs rand | 胜/seeds |' in text
    assert '内层T=1.00' in text
    assert len([r for r in result['cells'] if r['arm']=='orig'])==18
    in_table=False;values=[];checked=0
    for line in text.splitlines():
        if line.startswith('| 方法 | seeds |'):
            in_table=True;values=[]
        elif in_table and line.startswith('|') and not line.startswith('|---'):
            value=line.split('|')[6].strip()
            if value!='—':values.append(float(value.rstrip('%')))
        elif in_table and not line.startswith('|'):
            assert values==sorted(values,reverse=True);checked+=1;in_table=False
    assert checked==18
