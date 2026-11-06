from pathlib import Path
p=Path("src/rage/experiments/full_pool.py")
s=p.read_text()
s=s.replace('rage_global="rage_global", rage_traj_global="rage_traj_global")',
            'rage_global="rage", rage_traj_global="rage_traj")')
s=s.replace('''    elif base == 'rage_global':
        output = _rage_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif base == 'rage_traj_global':
        output = _rage_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
    elif base == 'rage_uniform':''',
'''    elif name == 'rage_global':
        output = _rage_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif name == 'rage_traj_global':
        output = _rage_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
    elif base == 'rage_uniform':''')
p.write_text(s)

p=Path("src/rage/experiments/full_pool_run.py")
s=p.read_text().replace('arms=list(ARMS), seeds=[0, 1, 2])','arms=list(ARMS), seeds=[0])')
p.write_text(s)

p=Path("configs/rage/main_v2.json")
import json
c=json.loads(p.read_text())
c["arms"]=["rage_gateless","random","lessvg_t025","tsds","lpv_t025","rage_traj_gls","rage_global","rage_traj_global"]
c["seeds"]=[0]
for st in c["settings"].values():
    t=st["temperature"]
    t["rage_global"]=0.25
    t["rage_traj_global"]=0.25
c["batch"]="main-shared-nine-v2-global-20260917"
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+"\n")
