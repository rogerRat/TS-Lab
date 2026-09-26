from pathlib import Path
# Readable imports in the actual smoke entry.
p=Path('src/rags/experiments/full_pool_smoke.py');s=p.read_text().replace('import argparse\n','import argparse\nfrom dataclasses import asdict\nimport json\n').replace('from .full_pool import ARMS, ARM_BASE, select_full_pool','from .full_pool import ARMS, ARM_BASE, select_full_pool\nfrom .ranking import DEFAULT_TEMPERATURE').replace('__import__("dataclasses").asdict(config)','asdict(config)').replace('__import__("json").loads(protocol_path.read_text())','json.loads(protocol_path.read_text())').replace('__import__("rags.experiments.ranking",fromlist=["DEFAULT_TEMPERATURE"]).DEFAULT_TEMPERATURE','DEFAULT_TEMPERATURE');p.write_text(s)
p=Path('src/rags/experiments/seed_rounds.py');s=p.read_text().replace("    cache=folder/'grad_cache';source=Path(template.get('gradient_cache_source') or parent/'grad_cache').resolve();source.mkdir(parents=True,exist_ok=True)","    cache=folder/'grad_cache'\n    source=Path(template.get('gradient_cache_source') or parent/'grad_cache').resolve()\n    if not template.get('gradient_cache_source'):source.mkdir(parents=True,exist_ok=True)")
p.write_text(s)
p=Path('tests/test_full_pool_resampling.py');p.write_text(p.read_text().rstrip()+'\n')
# A historical completed cell must never cause an early return in a V5 worker.
p=Path('src/rags/experiments/full_pool_worker.py');s=p.read_text();a=s.index('\ndef run_worker(');s=s[:a]+'''
def _validate_completed_cell(path,*,student,dataset,arm,seed):
    record=json.loads(Path(path).read_text())
    expected=dict(student=student,dataset=dataset,arm=arm,seed=seed,
        experiment_version='v5',resampling_revision=REVISION)
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed cell identity differs from V5; use a fresh run directory')

'''+s[a:]
s=s.replace('                if cell.exists():\n                    if clean_completed:', '                if cell.exists():\n                    _validate_completed_cell(cell,student=student,dataset=dataset,arm=arm,seed=seed)\n                    if clean_completed:')
s=s.replace('        if not original.exists():\n            costs.original', '        if original.exists():\n            _validate_completed_cell(original,student=student,dataset=dataset,arm=original_arm,seed=0)\n        if not original.exists():\n            costs.original');p.write_text(s)
# Exercise actual worker training/evaluation+resume for every gradient arm in one preparation.
p=Path('tests/test_selection.py');s=p.read_text()+'''

def test_all_gradient_arms_worker_training_evaluation_and_completed_resume(tmp_path,monkeypatch):
    from test_full_pool_worker import fixture
    from rags.experiments.full_pool_worker import run_worker
    from rags.experiments.protocol import method_protocols
    from rags.experiments.students import atomic_json
    config,bundle=fixture(tmp_path)
    config.update(arms=list(METHODS),seeds=[0],method_protocols=method_protocols(METHODS))
    config['crb']=dict(cover_density=.3,error_density=.1,split_density=.1)
    atomic_json(tmp_path/'protocol.json',config)
    out=run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    paths=sorted((out/'cells').glob('*.json'));assert len(paths)==10
    before={p.name:p.read_bytes() for p in paths}
    def forbidden(*args,**kwargs):raise AssertionError('Resume must reuse completed cells and prepared distributions')
    bundle[1].batch_gradient_sketch=forbidden
    monkeypatch.setattr('rags.experiments.full_pool_worker.run_arm_training',forbidden)
    run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    assert before=={p.name:p.read_bytes() for p in paths}
    for p in (out/'selections').glob('*.npz'):
        selected=load_selection(p).selected
        assert selected.metadata['frozen_rank_protocol']['additional_bp_allowed'] is False
        assert selected.metadata['source_weighting']=='legal_window_counts'


def test_worker_rejects_copied_historical_completed_cell(tmp_path):
    from test_full_pool_worker import fixture
    from rags.experiments.full_pool_worker import run_worker
    from rags.experiments.students import atomic_json
    config,bundle=fixture(tmp_path);out=run_worker(tmp_path,'chronos-base',model_bundle=bundle)
    path=out/'cells'/'tiny__rand__seed0.json';record=json.loads(path.read_text())
    record['experiment_version']='v4';atomic_json(path,record)
    with pytest.raises(ValueError,match='Completed cell identity differs'):
        run_worker(tmp_path,'chronos-base',model_bundle=bundle)
''';p.write_text(s)
# Suite reporter also refuses mixed historical files rather than reporting their scores.
p=Path('src/rags/experiments/full_pool_run.py');s=p.read_text();s=s.replace('                records.append(json.loads(original.read_text()))','                record=json.loads(original.read_text())\n                if config.get("experiment_version")=="v5" and record.get("experiment_version")!="v5":\n                    raise ValueError("Historical original cell cannot enter V5 summary")\n                records.append(record)').replace('                        records.append(json.loads(path.read_text()))','                        record=json.loads(path.read_text())\n                        if config.get("experiment_version")=="v5" and record.get("experiment_version")!="v5":\n                            raise ValueError("Historical trained cell cannot enter V5 summary")\n                        records.append(record)');p.write_text(s)
