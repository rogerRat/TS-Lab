from pathlib import Path
import json,subprocess
for name in ('prior_clarification_20260918.md','v5_method_definition_20260918.md','v5_random_ties_20260918.md'):
    p=Path('docs/archive/method-decisions')/name;p.write_text(p.read_text().rstrip()+'\n')
with Path('docs/Progress.md').open('a') as f:f.write('\n### Final freeze checks completed\n- Final full-repository regression after all code changes: **586 passed, 2 skipped**, 6 pre-existing fixture/dependency warnings, 57.74 s. Both actual-model smoke subprocesses exited 0, 24/24 arms complete. Field audit confirmed matrix and training recipes unchanged from V4. Staged 87 explicit implementation/config/test/document files; unrelated pre-existing review note is not included. Formal V5 jobs launched: 0.\n')
paths=json.loads(Path('temp/v5-final/commit-paths.json').read_text())['paths']
subprocess.run(['git','add','--',*paths],check=True)
subprocess.run(['git','diff','--cached','--check'],check=True)
print(subprocess.check_output(['git','diff','--cached','--stat'],text=True))
print('STAGED SOURCE PATHS')
print(subprocess.check_output(['git','diff','--cached','--name-only','src','baselines'],text=True))
