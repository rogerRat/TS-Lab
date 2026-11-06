from pathlib import Path
import json,tarfile
from rage.experiments.protocol import validate_config
from rage.experiments.students import atomic_json
config=json.loads(Path('configs/rage/main.json').read_text());validate_config(config)
with tarfile.open('temp/v5-final/pre-cleanup-code.tar.gz') as tar:
    previous=json.load(tar.extractfile('configs/rage/main_v4.json'))
fields=['students','datasets','arms','seeds','pool_equivalents','pool_seed','context_length','prediction_length',
    'source_load','reference_root','inputs','model_paths','student_environments','student_parallelism','seed_schedule',
    'train_batch','eval_batch','checkpoint_retention','original_first','online_evaluation','student_selection_execution']
for key in fields:
    if config.get(key)!=previous.get(key):raise ValueError('Matrix/execution field changed: '+key)
for student in config['students']:
    old=previous['settings'][student];new=config['settings'][student]
    for key,value in old.items():
        if key not in ('temperature','temperature_evidence','epsilon_calibration') and new.get(key)!=value:
            raise ValueError('Training recipe field changed: '+student+'.'+key)
    if old['temperature']['lp-mirror']!=new['temperature']['lp-mirror']:
        raise ValueError('LP-Mirror native temperature changed')
for key in ('cover_density','error_density','split_density','curvature_cap'):
    if config['crb'][key]!=previous['crb'][key]:raise ValueError('Budget field changed: '+key)
atomic_json(Path('temp/v5-final/matrix-audit.json'),dict(compared_to='pre-cleanup main_v4.json',
    matrix_unchanged=True,training_recipes_unchanged=True,native_baseline_temperatures_unchanged=True,
    budget_densities_unchanged=True,fields=fields))
for student in config['students']:
    root=Path('/root/autodl-tmp/TS3/outputs')/('development-rank-volume-full-20260918-'+student)
    report=json.loads((root/'smoke.json').read_text())
    expected=set(config['arms'])-{'orig'}
    if set(report['arms'])!=expected:raise ValueError('Incomplete actual-model smoke: '+student)
    for arm,record in report['arms'].items():
        if record['arm']!=arm:raise ValueError('Smoke arm mismatch')
print('Matrix unchanged; real-model smoke 24/24 training arms complete')
# Record concrete outcomes before freezing the Git version.
with Path('docs/Progress.md').open('a') as f:f.write('''
## 2026-09-18 V5 final validation and freeze
- Current scientific methods now use canonical module/config names: `configs/rage/main.json`, `rank_protocol.json`, `crb_protocol.json`; no `_vN` executable aliases. CRB only recursive minimum-kappa, all candidates, rightmost ties, no P10, exact capped budgets. Raw representatives keep norms; scoring-only weighted opinions -> object-rank interval mass -> source legal-start prior -> required normalization. No post-preparation BP.
- Compared to the pre-cleanup V4 snapshot: two models, nine tasks, thirteen arms, seeds 0/1/2, source pool N=16284, C=512/H=64, training recipes/optimizer/schedule, native LP-Mirror temperature and CRB budget densities are unchanged. Field-by-field audit: `temp/v5-final/matrix-audit.json`. Only the method/probability implementation changed.
- Repository regression: 584 passed, 2 skipped before final cell-resume guard; subsequent targeted final integration: 48 passed (includes all nine gradient arms through actual fixture worker training/evaluation and completed resume, no additional gradient calls, rejection of copied V4 cells). Final complete repository run is recorded in `temp/v5-final/final-all-tests.log`.
- Real pretrained Chronos and TimesFM smoke: each 12/12 training arms completed, 24/24 total, with shared actual-window BP preparation, save/load probability identity, one real training step and evaluation. New isolated output roots: `/root/autodl-tmp/TS3/outputs/development-rank-volume-full-20260918-chronos-base` and `...-timesfm-2.5`. Fixtures: 20 source trajectories, 2 references, synthetic length 632; not formal effectiveness results. Native TSDS emitted small-fixture clustering warnings, but its rules and probability kernel were preserved. Logs: `temp/v5-final/{student}-real-smoke.log`.
- Formal entry dry-run reports `jobs_launched=0`; no formal V5 launched. Prior V4 outputs/checkpoints/raw caches preserved. Source-only pre-cleanup snapshot stays ignored in `temp/v5-final/pre-cleanup-code.tar.gz`; old protocols and scientific decisions remain historical in `docs/archive/`.
- Preparing one Git commit for implementation, dependencies, canonical config, regressions and current documentation. No Push requested or performed.
''')
# Persist explicit staging list. Existing unrelated review note stays untouched/untracked.
import subprocess
tracked=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
new=['configs/rage/crb_protocol.json','configs/rage/main.json','configs/rage/rank_protocol.json','configs/rage/shared_tasks.json',
    'docs/MAIN.md','docs/method.md','src/rage/experiments/full_pool_assets.py','src/rage/experiments/prepare_suite.py',
    'src/rage/experiments/protocol.py','src/rage/experiments/ranking.py','src/rage/experiments/seed_rounds.py','src/rage/experiments/suite_report.py',
    'tests/test_crb_kappa.py','tests/test_full_pool_assets.py','tests/test_gpu_marginal.py','tests/test_ranking.py','tests/test_reference_scores.py',
    'tests/test_seed_rounds.py','tests/test_selection.py','tests/test_selection_memory.py']
new+=subprocess.check_output(['git','ls-files','--others','--exclude-standard','docs/archive'],text=True).splitlines()
stage=tracked+new
atomic_json(Path('temp/v5-final/commit-paths.json'),dict(paths=stage))
subprocess.run(['git','add','--',*stage],check=True)
print('Staged',len(stage),'explicit files')
