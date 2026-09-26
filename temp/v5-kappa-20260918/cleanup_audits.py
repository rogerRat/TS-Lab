from pathlib import Path
p=Path('src/assets/gradient/blocks.py')
s=p.read_text().replace('        counters["split_audits"] = []', '        counters.pop("epsilon_audits")\n        counters["split_audits"] = []')
p.write_text(s)
p=Path('src/rags/experiments/full_pool_assets.py')
old='''audit=json.dumps(dict(counts=trajectory_counts(result), epsilon=result.epsilon_audit,
                             **({"split":result.split_audit} if result.split_audit is not None else {}))))'''
new='''audit=json.dumps(dict(counts=trajectory_counts(result),
                             **({"split":result.split_audit} if result.split_audit is not None
                                else {"epsilon":result.epsilon_audit}))))'''
s=p.read_text();assert old in s;s=s.replace(old,new);p.write_text(s)
