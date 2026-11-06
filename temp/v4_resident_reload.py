from pathlib import Path
import os,json,signal,time,subprocess

repo=Path('/root/TS3');os.chdir(repo)
from rage.experiments.students import atomic_json
from rage.experiments.v4_protocol import validate_v4_config
run=Path('/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918')
config=json.loads((run/'protocol.json').read_text());validate_v4_config(config)
assert json.loads((run/'state.json').read_text())['seed']==0
assert 'source_gpu = torch.empty' in (repo/'src/rage/experiments/reference_opinions.py').read_text()
round_dir=run/'rounds/seed0'
def snapshot():
    return {s:dict(cells=sorted(p.name for p in (round_dir/'students'/s/'cells').glob('*.json')),
                   selections=sorted(p.name for p in (round_dir/'students'/s/'selections').glob('*.npz')))
            for s in config['students']}
before=snapshot()
pids=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
    except (OSError,UnicodeError):continue
    if 'rage.experiments.v4_seed_rounds' in args and str(run) in args:
        pids.append((0,int(p.name)))
    elif 'rage.experiments.full_pool_worker' in args and str(round_dir) in args:
        pids.append((1,int(p.name)))
assert sum(kind==0 for kind,_ in pids)==1,(pids,'Parent changed; coordinate before reload')
assert sum(kind==1 for kind,_ in pids)==2,(pids,'Workers changed; coordinate before reload')
for _,pid in sorted(pids):
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
def alive(pid):
    try:return (Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z')
    except OSError:return False
deadline=time.monotonic()+20
while any(alive(pid) for _,pid in pids) and time.monotonic()<deadline:time.sleep(.25)
assert not any(alive(pid) for _,pid in pids),'Old worker still alive; do not launch duplicate'
after=snapshot()
for s in config['students']:
    assert set(before[s]['cells'])<=set(after[s]['cells'])
    assert set(before[s]['selections'])<=set(after[s]['selections'])
env=dict(os.environ,PYTHONPATH=str(repo/'src'),HF_HOME='/root/autodl-tmp/hf-cache',
    PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
python='/root/miniconda3/envs/ts3/bin/python'
with (run/'suite.log').open('a') as log:
    log.write('\n2026-09-18 PERFORMANCE_RELOAD source-resident optimization for current seed0\n');log.flush()
    proc=subprocess.Popen([python,'-m','rage.experiments.v4_seed_rounds','--config','configs/rage/main_v4.json',
        '--run',str(run)],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
old=json.loads((run/'launcher.json').read_text())
atomic_json(run/'launcher.json',dict(old,pid=proc.pid,previous_pid=old['pid'],
    reloaded_at=time.strftime('%Y-%m-%d %H:%M:%S'),reason='source matrix resident on GPU; user requested full throughput'))
(repo/'temp/v4_suite.pid').write_text(str(proc.pid)+'\n')
atomic_json(run/'performance_reload.json',dict(old_pids=pids,new_parent_pid=proc.pid,
    before=before,after=after,scientific_protocol_unchanged=True,inflight_work_recomputed=True))
Path('docs/Progress.md').open('a').write('\n## 2026-09-18 当前seed0立即加载GPU常驻优化\n\n'
    f'- 用户要求全力推进、吃满计算资源。Claude已完成source-resident FP64配对评分优化，原worker仍持旧模块。安全停止旧父/worker {pids}，由v4_seed_rounds父入口恢复，新父PID={proc.pid}。\n'
    '- 已完成cells及选集逐字段/文件身份保留，原始梯度缓存继续复用；未落盘的特征/评分重做。种子屏障、三seed、温度、双层权重、GPU FP64及全部科学协议不变；未清除任何结果。\n'
    '- 即时性能工作由Codex实施，后续巡检仍由Claude负责；Codex heartbeat保持暂停。\n')
print(json.dumps(dict(new_parent_pid=proc.pid,stopped_pids=pids,preserved=after),indent=2))
