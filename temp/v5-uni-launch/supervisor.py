from pathlib import Path
import json,os
from rags.experiments.seed_rounds import run_rounds
from rags.experiments.full_pool_run import run_suite
from rags.experiments.students import atomic_json
run=Path('/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920')
config=json.loads(Path('/root/TS3/configs/rags/main_v5_uni.json').read_text())
atomic_json(run/'supervisor.json',dict(pid=os.getpid(),status='running',supervision='v5-uni experiment 3'))
def runner(config,folder,execution):
    profile=json.loads((run/'execution.json').read_text())
    return run_suite(config,folder,execution=dict(profile,**execution))
try:
    code=run_rounds(config,run,runner=runner)
    atomic_json(run/'supervisor.json',dict(pid=os.getpid(),status='complete',exit_code=code))
except BaseException as e:
    atomic_json(run/'supervisor.json',dict(pid=os.getpid(),status='failed',error=repr(e)))
    raise
