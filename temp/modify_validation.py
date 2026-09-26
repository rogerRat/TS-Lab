from pathlib import Path
p=Path("src/rags/experiments/full_pool_run.py")
s=p.read_text()
s=s.replace('''required = dict(pool_equivalents=16284, context_length=512, prediction_length=64,
                        students=list(STUDENTS), datasets=list(MAIN_DATASETS),
                        arms=list(ARMS), seeds=[0])''','''required = dict(pool_equivalents=16284, context_length=512, prediction_length=64,
                        students=list(STUDENTS), datasets=list(MAIN_DATASETS),
                        arms=list(ARMS), seeds=[0, 1, 2])''')
s=s.replace('''            required['datasets'] = decisions['datasets']
            # The linked temperature supplement''','''            required['datasets'] = decisions['datasets']
            required['seeds'] = [0]
            # The linked temperature supplement''')
p.write_text(s)
