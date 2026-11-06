from pathlib import Path
import subprocess,json
p=Path('docs/Progress.md')
with p.open('a') as f:f.write('\n### V5 remote Git freeze completed\n- All authorized development/cleanup/testing is complete. The implementation, canonical configuration, regression tests and documentation are frozen together in one commit on remote `rage`, titled `Freeze V5 ranking, source-volume prior and recursive kappa CRB`. Final commit ID is obtained with `git log -1`; no Push performed. V4 experiment processes are stopped, historical outputs preserved, and no formal V5 experiment is running. The sole remaining untracked review note predates this work and was left untouched.\n')
subprocess.run(['git','add','docs/Progress.md'],check=True)
subprocess.run(['git','commit','--amend','--no-edit'],check=True,stdout=subprocess.DEVNULL)
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
Path('temp/v5-final/freeze-result.json').write_text(json.dumps(dict(commit=commit,branch='rage',tests_passed=586,
    tests_skipped=2,real_model_training_arms_passed=24,matrix_changed=False,formal_jobs_launched=0,pushed=False),indent=2)+'\n')
print(subprocess.check_output(['git','log','-1','--format=%h %s'],text=True).strip())
print(subprocess.check_output(['git','status','--short'],text=True).strip())
subprocess.run(['git','diff','HEAD','--check'],check=True)
