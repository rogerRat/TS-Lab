from pathlib import Path
import json,os,subprocess,sys
from datetime import datetime,timezone,timedelta
from rags.experiments.prepare_suite import prepare_suite
from rags.experiments.protocol import validate_config
from rags.experiments.students import atomic_json
repo=Path('/root/TS3');os.chdir(repo)
run=Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918')
config=json.loads(Path('configs/rags/main.json').read_text());validate_config(config)
if run.exists() and any(run.iterdir()):raise RuntimeError('Choose a fresh V5 launch directory; existing contents preserved')
prepare_suite(config,run)
old=Path('/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918/rounds/seed0/grad_cache').resolve()
if not old.is_dir():raise FileNotFoundError(old)
(run/'grad_cache').symlink_to(old,target_is_directory=True)
profile=dict(student_parallelism=2,student_selection_execution={
    'chronos-base':dict(gradient_batch=8,gradient_workers=4,inference_batch=64,cpu_threads=4,generation='eager',memory_fraction=.40),
    'timesfm-2.5':dict(gradient_batch=16,gradient_workers=1,inference_batch=64,cpu_threads=4,generation='eager',memory_fraction=.40)})
atomic_json(run/'execution.json',profile)
now=datetime.now(timezone(timedelta(hours=8))).isoformat()
atomic_json(run/'execution_control.json',dict(mode='run',selection_enabled=True,training_enabled=True,
    evaluation_enabled=True,authorized_by='user: 测试完毕直接你本人主管开始V5，尽可能优化跑满GPU',authorized_at=now))
atomic_json(run/'launch.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    run=str(run),started_at=now,supervision='current Codex thread personally; no delegation',
    raw_gradient_cache=str(old),reuse='field validation only; reconstruct V5 CRB and probabilities; no old cells',
    expected_unique_cells=666,expected_training_cells=648,seed_schedule=[0,1,2],execution=profile))
script=repo/'temp/v5-launch/supervisor.py'
script.write_text('''from pathlib import Path
import json,os
from rags.experiments.seed_rounds import run_rounds
from rags.experiments.full_pool_run import run_suite
from rags.experiments.students import atomic_json
run=Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918')
config=json.loads((run/'protocol.json').read_text())
atomic_json(run/'supervisor.json',dict(pid=os.getpid(),status='running',supervision='current Codex thread'))
def runner(config,folder,execution):
    profile=json.loads((run/'execution.json').read_text())
    return run_suite(config,folder,execution=dict(profile,**execution))
try:
    code=run_rounds(config,run,runner=runner)
    atomic_json(run/'supervisor.json',dict(pid=os.getpid(),status='complete',exit_code=code))
except BaseException as e:
    atomic_json(run/'supervisor.json',dict(pid=os.getpid(),status='failed',error=repr(e)))
    raise
''')
import shlex
command=shlex.join([sys.executable,'-u',str(script)])+' > '+shlex.quote(str(run/'supervisor.log'))+' 2>&1'
env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
subprocess.run(['tmux','new-session','-d','-s','main-rank-volume','-c',str(repo),command],check=True,env=env)
with Path('docs/Progress.md').open('a') as f:f.write('\n## '+now+' Formal V5 authorized launch\n- User explicitly authorized personal supervision and GPU optimization after tests. Launched frozen commit 6d3ead6 into '+str(run)+'. Matrix unchanged: 2 models, 9 tasks, 13 arms, seeds 0/1/2 with round barriers, 666 unique cells / 648 training cells. tmux main-rank-volume runs the standard seed-round supervisor with explicit execution-only overrides. Chronos 4 independent single-window gradient processes (request batch8), TimesFM verified batch16, two models parallel, memory_fraction .40 each, training batch32 unchanged. Raw cache '+str(old)+' reused only through field comparison; V5 CRB/probabilities rebuilt, old cells not reused. Current state: launched, no formal V5 result yet. Next: verify both workers, cache field equality, GPU/memory utilization and first cells.\n')
print(json.dumps(dict(run=str(run),tmux='main-rank-volume',execution=profile),indent=2))
