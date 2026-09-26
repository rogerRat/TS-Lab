"""Terminate-and-archive for MAIN-v4 (user-terminated partial run, 2026-09-18).

Mirrors temp/archive_and_launch.py's archive stage (V3) but only for the V4 run,
marks the suite archived, and launches NOTHING.
"""
import csv, json, os, shutil, tarfile, time
from pathlib import Path

repo = Path('/root/TS3'); os.chdir(repo)
from rags.experiments.students import atomic_json

outputs = Path('/root/autodl-tmp/TS3/outputs')
run = outputs / 'main-v4-three-seeds-20260918'
archive = outputs / 'archives' / 'results-v4-partial-20260918'
archive.mkdir(parents=True, exist_ok=True)
assert not (archive / 'archive_complete.json').exists(), 'Archive already completed; inspect before rerunning'

terminated_at = time.strftime('%Y-%m-%d %H:%M:%S')

def counted(p, seen):
    """V4 unique-cell rule: orig counted only at seed0; rounds reuse seed0's orig
    (records carry seed=0), so dedupe by (student, dataset, arm, seed)."""
    rec = json.loads(p.read_text())
    key = (rec.get('student'), rec.get('dataset'), rec.get('arm'), rec.get('seed'))
    if rec.get('arm') == 'orig' and rec.get('seed') != 0:
        return None
    if key in seen:
        return None
    seen.add(key)
    return rec

seen = set()
per_seed = {}
for seed_dir in sorted(run.glob('rounds/seed*')):
    before = len(seen)
    for p in seed_dir.glob('students/*/cells/*.json'):
        counted(p, seen)
    per_seed[seed_dir.name] = len(seen) - before
total_unique = sum(per_seed.values())

target = archive / 'runs' / run.name
target.mkdir(parents=True, exist_ok=True)
item = dict(run=str(run), retention='all original artifacts retained in place', files=[], cells=0)

for p in list(run.glob('*.json')) + list(run.glob('*.log')):
    if p.is_file():
        shutil.copy2(p, target / p.name)
        item['files'].append(dict(path=str(p), bytes=p.stat().st_size))

rows = []
seen = set()
for p in sorted(run.glob('rounds/seed*/students/*/cells/*.json')):
    rec = counted(p, seen)
    if rec is None:
        continue
    dest = target / p.relative_to(run)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
    rows.append(dict(run=run.name, source=str(p), record=rec))
    item['cells'] += 1
assert item['cells'] == total_unique, (item['cells'], total_unique)

for pat in ('model.json', 'state.json', 'selection_metrics.jsonl'):
    for p in run.glob(f'rounds/seed*/students/*/{pat}'):
        dest = target / p.relative_to(run)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

with (archive / 'all_cells.jsonl').open('w') as stream:
    for row in rows:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')

tables = []
for row in rows:
    rec = row['record']; student = rec['student']; dataset = rec['dataset']
    rand = [r['record']['mase'] for r in rows if r['run'] == row['run']
            and r['record']['student'] == student and r['record']['dataset'] == dataset
            and r['record']['arm'] in ('rand', 'random')]
    baseline = sum(rand) / len(rand) if rand else None
    tables.append(dict(run=row['run'], student=student, dataset=dataset, arm=rec['arm'], seed=rec['seed'],
                       mase=rec['mase'], improvement_vs_rand=(1 - rec['mase'] / baseline) if baseline else None,
                       source=row['source']))
tables.sort(key=lambda r: (-(r['improvement_vs_rand'] if r['improvement_vs_rand'] is not None else -1e18),
                           r['run'], r['student'], r['dataset'], r['arm'], r['seed']))
with (archive / 'all_results_ranked.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['run', 'student', 'dataset', 'arm', 'seed', 'mase', 'improvement_vs_rand', 'source'])
    writer.writeheader(); writer.writerows(tables)

shutil.copytree(repo / 'docs', archive / 'docs', dirs_exist_ok=True)
shutil.copytree(repo / 'configs/rags', archive / 'configs/rags', dirs_exist_ok=True)
shutil.copy2(repo / 'docs/MAIN-v4.md', repo / 'docs/archive/MAIN-v4-partial-terminated-20260918.md')

seed_summary = ', '.join(f'{k}={v} unique cells' for k, v in per_seed.items())
(archive / 'README.md').write_text(
    '# MAIN-v4 partial results archive (user-terminated)\n\n'
    f'Run terminated by user at {terminated_at} Beijing time: {total_unique}/666 unique cells complete ({", ".join(f"{k}: {v}" for k, v in per_seed.items())}). '
    'Seed 0 round complete; seed 1 round partial; seed 2 round never started. In-flight training cells at termination are absent by design.\n\n'
    'all_results_ranked.csv is sorted descending by relative improvement over rand within each model/task.\n'
    'all_cells.jsonl preserves complete raw result records. Configs, docs, root logs and run metadata are snapshotted.\n'
    'Original outputs, selections, gradient caches and checkpoints remain at the source run path; nothing was deleted.\n'
    'Predecessor archive: results-through-v3-20260918.\n')

control = json.loads((run / 'execution_control.json').read_text())
control.update(mode='archived', selection_enabled=False, training_enabled=False,
               evaluation_enabled=False, assets_enabled=False,
               terminated_by_user=terminated_at, archive=str(archive))
atomic_json(run / 'execution_control.json', control)

removed = []
for name in ['v4_report_loop.pid', 'v4_report_loop.lock', 'v4_suite.pid']:
    p = repo / 'temp' / name
    if p.exists():
        shutil.copy2(p, archive / name); p.unlink(); removed.append(str(p))

atomic_json(archive / 'inventory.json', dict(runs=[item], raw_artifacts_deleted=False,
             raw_gradients_duplicated=False, source_outputs=str(outputs), cells=len(rows),
             per_seed_unique=per_seed))
with tarfile.open(archive / 'results-docs-configs.tar.gz', 'w:gz') as tar:
    for name in ['runs', 'docs', 'configs', 'all_cells.jsonl', 'all_results_ranked.csv', 'inventory.json', 'README.md']:
        tar.add(archive / name, arcname=name)
atomic_json(archive / 'archive_complete.json', dict(complete=False, terminated_by_user=True,
             terminated_at=terminated_at, v4_cells_unique=total_unique, target_cells=666,
             per_seed=per_seed, run=run.name, archive=str(archive), predecessor='results-through-v3-20260918',
             processes_stopped_before_archive='v4_seed_rounds launcher + 2x full_pool_worker + v4_report_loop (SIGTERM, verified GPU idle)',
             removed_disposable_files=removed))
print(f'V4 partial archive complete: {total_unique}/666 unique cells, {len(rows)} cell JSONs, seeds: {per_seed}')
