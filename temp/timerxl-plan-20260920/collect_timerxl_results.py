import json, glob, collections, sys
R="/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920"
files=sorted(glob.glob(R+"/rounds/seed*/students/timerxl-84m/cells/*.json"))
print("files",len(files))

def find_mase(c):
    # try common locations
    for path in [("metrics","mase"),("mase",),("result","mase"),("evaluation","mase"),("exam","mase")]:
        cur=c
        ok=True
        for k in path:
            if isinstance(cur,dict) and k in cur: cur=cur[k]
            else: ok=False; break
        if ok and isinstance(cur,(int,float)): return cur
    return None

c0=json.load(open(files[0]))
print("first-cell keys:",list(c0.keys()))
m0=find_mase(c0)
print("first-cell mase:",m0)
if m0 is None:
    print(json.dumps(c0,indent=1)[:2000]); sys.exit(1)

def get(c,*names):
    for n in names:
        if n in c: return c[n]
    raise KeyError(names)

rows=collections.defaultdict(dict)
for f in files:
    c=json.load(open(f))
    ds=get(c,"dataset","task")
    arm=get(c,"arm")
    seed=int(get(c,"seed"))
    rows[(ds,arm)][seed]=find_mase(c)
arms=sorted({a for _,a in rows})
tasks=sorted({d for d,_ in rows})
# 'orig' is seed-invariant: evaluated once per task; broadcast its single value to all seeds
for t in tasks:
    v=rows.get((t,"orig"),{})
    if v and len(v)<3:
        val=next(iter(v.values()))
        rows[(t,"orig")]={0:val,1:val,2:val}
bad=[k for k,v in rows.items() if len(v)!=3 or any(x is None for x in v.values())]
print("TASKS",len(tasks),"ARMS",len(arms),"CELLKEYS",len(rows),"INCOMPLETE",len(bad))
if bad: print("BAD",bad[:10])
out={}
for t in tasks:
    out[t]={a:{"seeds":[rows[(t,a)][s] for s in (0,1,2)],"mean":sum(rows[(t,a)].values())/3} for a in arms}
json.dump(out,open(R+"/results_matrix.json","w"),indent=1)
print("wrote results_matrix.json")
