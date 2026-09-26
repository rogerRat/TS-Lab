import json,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
config=json.loads(Path('configs/rags/main.json').read_text())
methods=[a for a in config['arms'] if a!='orig']

def run(student):
    out=Path('/root/autodl-tmp/TS3/outputs')/('development-rank-volume-20260918-'+student)
    env=config['student_environments'][student]
    cmd=[env,'-m','rags.experiments.full_pool_smoke','--student',student,'--model-path',config['model_paths'][student],
         '--out',str(out),'--arms',*methods]
    log=Path('temp/v5-final')/(student+'-real-smoke.log')
    with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    print(student,'exit=',r.returncode,'out=',out,'log=',log,flush=True)
    return r.returncode
with ThreadPoolExecutor(max_workers=2) as ex:
    codes=list(ex.map(run,config['students']))
raise SystemExit(1 if any(codes) else 0)
