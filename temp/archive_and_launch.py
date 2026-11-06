from pathlib import Path
import csv,json,os,signal,shutil,subprocess,tarfile,time,math

repo=Path('/root/TS3');os.chdir(repo)
from rage.experiments.full_pool_run import summarize
from rage.experiments.students import atomic_json
from rage.experiments.v3_report import render
from rage.experiments.v4_protocol import validate_v4_config
from rage.experiments.v4_report import render as render_v4

outputs=Path('/root/autodl-tmp/TS3/outputs')
v3=outputs/'main-shared-nine-v3-t005-20260917'
config=json.loads((v3/'protocol.json').read_text());result=summarize(v3,config)
assert result['complete'] and result['completed_cells']==189
for row in result['cells']:
    assert row['experiment_version']=='v3' and math.isfinite(row['mase'])
render(v3,repo/'docs/MAIN-v3.md')
archive=outputs/'archives'/'results-through-v3-20260918';archive.mkdir(parents=True,exist_ok=True)
assert not (archive/'archive_complete.json').exists(), 'Archive already completed; inspect before rerunning transition'
rows=[];inventory=[];runs=[]
for run in sorted(outputs.iterdir()):
    if not run.is_dir() or run.name=='archives':continue
    target=archive/'runs'/run.name;target.mkdir(parents=True,exist_ok=True)
    item=dict(run=str(run),retention='all original artifacts retained in place',files=[],cells=0)
    candidates=list(run.glob('*.json'))+list(run.glob('*.log'))
    candidates+=list(run.glob('students/*/model.json'))+list(run.glob('students/*/state.json'))
    candidates+=list(run.glob('students/*/selection_metrics.jsonl'))
    for p in candidates:
        if p.is_file():
            destination=target/p.relative_to(run);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,destination);item['files'].append(dict(path=str(p),bytes=p.stat().st_size,archived=str(destination)))
    for p in sorted(run.glob('students/*/cells/*.json')):
        record=json.loads(p.read_text());destination=target/p.relative_to(run)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
        rows.append(dict(run=run.name,source=str(p),record=record));item['cells']+=1
    inventory.append(item)
    if (run/'protocol.json').exists():runs.append(run.name)
with (archive/'all_cells.jsonl').open('w') as stream:
    for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
tables=[]
for row in rows:
    record=row['record'];student=record['student'];dataset=record['dataset']
    random=[r['record']['mase'] for r in rows if r['run']==row['run'] and r['record']['student']==student
        and r['record']['dataset']==dataset and r['record']['arm'] in ('rand','random')]
    baseline=sum(random)/len(random) if random else None
    improvement=(1-record['mase']/baseline) if baseline else None
    tables.append(dict(run=row['run'],student=student,dataset=dataset,arm=record['arm'],seed=record['seed'],
        mase=record['mase'],improvement_vs_rand=improvement,source=row['source']))
tables.sort(key=lambda r:(-(r['improvement_vs_rand'] if r['improvement_vs_rand'] is not None else -1e18),r['run'],r['student'],r['dataset'],r['arm'],r['seed']))
with (archive/'all_results_ranked.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['run','student','dataset','arm','seed','mase','improvement_vs_rand','source'])
    writer.writeheader();writer.writerows(tables)
shutil.copytree(repo/'docs',archive/'docs',dirs_exist_ok=True)
shutil.copytree(repo/'configs/rage',archive/'configs/rage',dirs_exist_ok=True)
shutil.copy2(repo/'docs/MAIN-v3.md',repo/'docs/archive/MAIN-v3-final-20260918.md')
atomic_json(archive/'inventory.json',dict(runs=inventory,raw_artifacts_deleted=False,
    raw_gradients_duplicated=False,source_outputs=str(outputs),cells=len(rows)))
(archive/'README.md').write_text('# Results archive through V3\n\n'
    f'V3 final: 189/189 complete. Registered {len(inventory)} output directories and archived {len(rows)} cell JSONs.\n\n'
    'all_results_ranked.csv is sorted descending by relative improvement over rand within each original run/model/task.\n'
    'all_cells.jsonl preserves complete raw result records. Configs, docs, root logs and run metadata are snapshotted.\n'
    'Original outputs, selections, gradient caches and any retained checkpoints remain at inventory source paths.\n'
    'Large raw gradients/checkpoints are not duplicated; no hashes are used.\n')
with tarfile.open(archive/'results-docs-configs.tar.gz','w:gz') as tar:
    for name in ['runs','docs','configs','all_cells.jsonl','all_results_ranked.csv','inventory.json','README.md']:
        tar.add(archive/name,arcname=name)
atomic_json(archive/'archive_complete.json',dict(complete=True,v3_cells=189,archived_cells=len(rows),runs=runs,
    archive=str(archive),completed_at=time.strftime('%Y-%m-%d %H:%M:%S')))

# Cleanup only retired controllers and disposable report partial/PID files.
stopped=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        args=(proc/'cmdline').read_bytes().split(b'\0')
        if any(a.decode(errors='ignore') in ('/root/TS3/temp/v3_t005_scheduler.sh','/root/TS3/temp/v3_t005_report_loop.sh') for a in args):
            os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
    except (OSError,ProcessLookupError):pass
control=json.loads((v3/'execution_control.json').read_text())
control.update(mode='archived',selection_enabled=False,training_enabled=False,evaluation_enabled=False)
atomic_json(v3/'execution_control.json',control)
removed=[]
for name in ['v3_t005_scheduler.pid','v3_t005_report.pid','v3_t005_bootstrap.pid']:
    p=repo/'temp'/name
    if p.exists():shutil.copy2(p,archive/name);p.unlink();removed.append(str(p))
for p in (repo/'temp').glob('main-v3-*.partial'):
    if p.is_file():p.unlink();removed.append(str(p))
atomic_json(archive/'cleanup.json',dict(stopped_pids=stopped,removed_disposable_files=removed,
    original_result_data_preserved=True))

v4config=json.loads((repo/'configs/rage/main_v4.json').read_text());validate_v4_config(v4config)
parent=outputs/'main-v4-three-seeds-20260918';parent.mkdir(exist_ok=True)
assert not (parent/'protocol.json').exists(), 'Use the existing suite resume entry rather than transition again'
atomic_json(parent/'protocol.json',v4config)
atomic_json(parent/'execution_control.json',dict(mode='run',selection_enabled=True,training_enabled=True,
    evaluation_enabled=True,assets_enabled=True,authorized_by='user: archive V3, cleanup, start V4',
    fixed_seeds=[0,1,2],predecessor_archive=str(archive)))
# Replace the prior preparation-only MAIN-v4 placeholder with the exact template.
old=repo/'docs/MAIN-v4.md'
if old.exists():shutil.copy2(old,repo/'docs/archive/MAIN-v4-preparation-20260918.md');old.unlink()
render_v4(parent,old)
env=dict(os.environ,PYTHONPATH=str(repo/'src'),HF_HOME='/root/autodl-tmp/hf-cache',
    PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
python='/root/miniconda3/envs/ts3/bin/python'
with (parent/'suite.log').open('a') as log:
    process=subprocess.Popen([python,'-m','rage.experiments.v4_seed_rounds','--config','configs/rage/main_v4.json',
        '--run',str(parent)],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(parent/'launcher.json',dict(pid=process.pid,seeds=[0,1,2],started_at=time.strftime('%Y-%m-%d %H:%M:%S')))
(repo/'temp/v4_suite.pid').write_text(str(process.pid)+'\n')
Path(repo/'docs/Progress.md').open('a').write('\n## 2026-09-18 V3归档清理后启动V4\n\n'
    f'- V3 189/189完成；全部既有输出目录与{len(rows)}个cell归档至{archive}，汇总CSV按vs rand降序。\n'
    f'- 停止V3守护进程{stopped}，清理{len(removed)}个可丢弃临时文件；历史结果/选集/梯度/保留checkpoint未删除。\n'
    f'- V4固定seed0/1/2、两模型、13方法、九任务、666唯一cells，轮间串行轮内并行；已授权启动，PID={process.pid}，run={parent}。\n'
    '- MAIN-v4复用MAIN-v3生成模板，任务表和汇总表按相对rand改进降序；正在推进，尚无V4效果结论。\n')
print(json.dumps(dict(v3_complete=189,archive=str(archive),archived_cells=len(rows),stopped=stopped,
    v4_run=str(parent),v4_pid=process.pid,seeds=[0,1,2]),indent=2))
