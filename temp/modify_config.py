from pathlib import Path
p=Path("src/rags/experiments/full_pool.py")
s=p.read_text()
s=s.replace('rags_global="rags_global", rags_traj_global="rags_traj_global")',
            'rags_global="rags", rags_traj_global="rags_traj")')
s=s.replace('''    elif base == 'rags_global':
        output = _rags_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif base == 'rags_traj_global':
        output = _rags_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
    elif base == 'rags_uniform':''',
'''    elif name == 'rags_global':
        output = _rags_global(engine, reference_fragments, pool, index, pool_manifest,
                              crb_config, gradient_assets, metadata, temperature)
    elif name == 'rags_traj_global':
        output = _rags_traj_global(index, pool_manifest, crb_config, gradient_assets,
                                   metadata, temperature)
    elif base == 'rags_uniform':''')
p.write_text(s)

p=Path("src/rags/experiments/full_pool_run.py")
s=p.read_text().replace('arms=list(ARMS), seeds=[0, 1, 2])','arms=list(ARMS), seeds=[0])')
p.write_text(s)

p=Path("configs/rags/main_v2.json")
import json
c=json.loads(p.read_text())
c["arms"]=["rags_gateless","random","lessvg_t025","tsds","lpv_t025","rags_traj_gls","rags_global","rags_traj_global"]
c["seeds"]=[0]
for st in c["settings"].values():
    t=st["temperature"]
    t["rags_global"]=0.25
    t["rags_traj_global"]=0.25
c["batch"]="main-shared-nine-v2-global-20260917"
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+"\n")
