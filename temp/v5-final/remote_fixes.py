from pathlib import Path
import ast,json
p=Path('src/rage/experiments/protocol.py');s=p.read_text().replace("    for student in config['students']:\n        temperatures=", "    for student in config['students']:\n        if student not in config.get('settings',{}):raise ValueError('Missing model settings: '+student)\n        temperatures=");p.write_text(s)
p=Path('baselines/trajectory.py');s=p.read_text().replace('[(fid, index.fragment_lengths[fid]) for fid in fids]','[[fid, index.fragment_lengths[fid]] for fid in fids]');p.write_text(s)
# Retire old split/P10 oracle tests: scientific replacement is test_crb_kappa.
for name in ('test_crb_trajectory_p10.py','test_crb_split_budget.py'):Path('tests',name).unlink()
p=Path('tests/test_anchor_budget_cap.py');s=p.read_text();a=s.index('def test_real_crb_cache_replay_still_passes_self_test');s=s[:a]+'''def test_retired_replay_protocol_is_rejected(tmp_path):
    import pytest
    with pytest.raises(RuntimeError,match='Retired block-constant budget protocol'):
        run_self_test(tmp_path,log=lambda *args:None)
''';p.write_text(s)
p=Path('src/rage/experiments/anchor_budget_simulate.py');s=p.read_text().replace('    for block in result.blocks:\n        left, right = int(block.left_index)', '    for block in ([] if single_window else result.blocks):\n        left, right = int(block.left_index)');p.write_text(s)
p=Path('tests/test_anchor_integer.py');s=p.read_text().replace('test_strong_cancellation_dropped_windows_remain_in_denominator','test_strong_cancellation_keeps_all_windows').replace('assert metrics["dropped_windows"] == len(truth)','assert metrics["dropped_windows"] == 0').replace('    assert not result.blocks','    assert len(result.blocks)==1');p.write_text(s)
p=Path('tests/test_resampling.py');s=p.read_text().replace('match="Nonfinite"','match="finite"');p.write_text(s)
# Retire old source hard-delivery path: never silently run unbudgeted BP.
p=Path('src/rage/experiments/arms.py');s=p.read_text();a=s.index('def trajectory_level_select(');b=s.index('\n\ndef ',a) if '\n\ndef ' in s[a:] else s.index('\n__all__',a);sig=s[a:s.index('\n',s.index(') -> ArmOutput:',a))+1] if ') -> ArmOutput:' in s[a:b] else None
# AST isolates signature/body despite multiline declaration.
tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='trajectory_level_select');lines=s.splitlines(keepends=True)
body_start=node.body[0].lineno-1;lines[body_start:node.end_lineno]=['    raise RuntimeError("Retired hard trajectory delivery; use canonical V5 full_pool selection with prepared assets")\n'];p.write_text(''.join(lines))
p=Path('tests/test_rage_protocol.py');s=p.read_text();s=s.replace('    out = trajectory_level_select(reference, pool, index,','    with pytest.raises(RuntimeError,match="Retired hard trajectory delivery"):\n        trajectory_level_select(reference, pool, index,');a=s.index('    with pytest.raises(RuntimeError,match="Retired hard trajectory delivery")');node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='test_trajectory_level_select_with_stub_engine');lines=s.splitlines(keepends=True);end=next(i for i in range(a and node.lineno,len(lines)) if 'context_length=512, prediction_length=64)' in lines[i] and i>=body_start) if False else None
# Drop obsolete assertions following that retired call in this test only.
start=s.index('    with pytest.raises(RuntimeError,match="Retired hard trajectory delivery")');lineend=s.index('\n',s.index('context_length=512, prediction_length=64)',start));lines=s.splitlines(keepends=True);node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='test_trajectory_level_select_with_stub_engine');endoffset=sum(len(x) for x in lines[:node.end_lineno]);s=s[:lineend+1]+s[endoffset:];p.write_text(s)
p=Path('tests/test_rage_ablation.py');s=p.read_text().replace('    assert variant_config(BASE, "random_cover").initial_placement == "random"','    with pytest.raises(ValueError,match="rightmost ties"):\n        variant_config(BASE,"random_cover")');p.write_text(s)
# Generic main defaults and V5 launches share the canonical round supervisor.
p=Path('src/rage/experiments/run.py');s=p.read_text().replace('default="configs/rage/full_pool.yaml"','default="configs/rage/main.json"').replace('        return run_suite(config, run_dir, dry_run=args.dry_run)','        if args.dry_run:return run_suite(config,run_dir,dry_run=True)\n        from .seed_rounds import run_rounds\n        return run_rounds(config,run_dir)');p.write_text(s)
p=Path('tests/test_full_pool_run.py');s=p.read_text().replace('configs/rage/full_pool.yaml','configs/rage/main.json');a=s.index('def test_formal_config_requires_exact_matrix_and_frozen_development');b=s.index('\ndef test_explicit_environment',a);s=s[:a]+'''def test_formal_config_requires_frozen_v5_matrix():
    config=formal_config();validate_config(config)
    for key,value in (("pool_equivalents",16384),("students",["chronos-small"]),
            ("datasets",["boomlet_1676"]),("arms",["random"]),("seeds",[3])):
        with pytest.raises(ValueError):validate_config(dict(config,**{key:value}))

'''+s[b:];p.write_text(s)
# Remove retired alias dispatch test, preserve useful kernel parity test.
p=Path('tests/test_full_pool_global.py');s=p.read_text();tree=ast.parse(s);n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_formal_global_dispatch_does_not_call_original_kernel');lines=s.splitlines(keepends=True);start=min([n.lineno]+[d.lineno for d in n.decorator_list]);del lines[start-1:n.end_lineno];p.write_text(''.join(lines))
# Retired hard LESS delivery assertions are replaced by test_selection for all V5 methods.
p=Path('tests/test_trajectory_baselines.py');s=p.read_text();tree=ast.parse(s);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('test_less_trajectory_cosines_and_whole_trajectory_deployment','test_less_score_ties_follow_frozen_random_order')];lines=s.splitlines(keepends=True)
for n in reversed(nodes):del lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]
s=''.join(lines).replace('for kind in ("less", "tsds"):', 'for kind in ("tsds",):').replace('for name in ("less", "tsds"):', 'for name in ("tsds",):')
s=s.replace('        _, cold = trajectory_features(engine, index, changed, kind=kind,\n                                     cache_dir=tmp_path, pool_manifest=manifest)\n        assert not cold["pool_cached"]','        with pytest.raises(ValueError,match="cache fields differ"):\n            trajectory_features(engine,index,changed,kind=kind,cache_dir=tmp_path,pool_manifest=manifest)')
p.write_text(s)
# Tighten window ownership and current protocol resume checks.
p=Path('src/rage/experiments/full_pool.py');s=p.read_text().replace('    rank,probabilities,audit=rank_mass(', '    if sum(len(unit) for unit in units)!=len(index):raise ValueError("Source units must own the full legal-start pool")\n    rank,probabilities,audit=rank_mass(').replace("        validate_frozen_rank_protocol(metadata.get('frozen_rank_protocol'))","        validate_frozen_rank_protocol(metadata.get('frozen_rank_protocol'))\n        if metadata.get('scoring_revision')!=REVISION or metadata.get('probability_mapping')!='object-rank-exponential-interval-v5':\n            raise ValueError('Cached scoring/probability identity differs')")
p.write_text(s)
with Path('docs/Progress.md').open('a') as f:f.write('\n### V5 broad regression migration\n- First all-repository run: 589 passed, 34 failed, 2 skipped. Failures exposed retired P10/minimum-norm/random-hard-delivery assertions, a field-cache tuple/list mismatch, a single-window diagnostic overlap, and a config error type. Fixed actual issues and migrated/retired assertions against the current approved protocol; replacement independent kappa and all-gradient-arm integration tests remain active. No formal jobs launched.\n')
