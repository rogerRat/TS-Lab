import json, glob, collections
RUNS={
 "main":{"dir":"/root/autodl-tmp/TS3/outputs/main-rank-volume-20260918","students":["chronos-base","timesfm-2.5"]},
 "uni":{"dir":"/root/autodl-tmp/TS3/outputs/main-rank-volume-uni-20260920","students":["chronos-base","timesfm-2.5"]},
}
out={}
for rn,rc in RUNS.items():
    for st in rc["students"]:
        rows=collections.defaultdict(dict)
        for f in glob.glob(f"{rc['dir']}/rounds/seed*/students/{st}/cells/*.json"):
            c=json.load(open(f))
            rows[(c["dataset"],c["arm"])][int(c["seed"])]=c["mase"]
        arms=sorted({a for _,a in rows})
        tasks=sorted({d for d,_ in rows})
        for t in tasks:
            v=rows.get((t,"orig"),{})
            if v and len(v)<3:
                val=next(iter(v.values())); rows[(t,"orig")]={0:val,1:val,2:val}
        bad=[k for k,v in rows.items() if len(v)!=3]
        print(rn,st,"tasks",len(tasks),"arms",len(arms),"bad",len(bad), bad[:5])
        g={a: sum(sum(rows[(t,a)].values())/3 for t in tasks)/len(tasks) for a in arms if len(rows[(t,a)])==3}
        out[f"{rn}:{st}"]=g
json.dump(out,open("/root/autodl-tmp/TS3/outputs/main-rank-volume-timerxl-20260920/cross_model_means.json","w"),indent=1)
for k,g in out.items():
    print(k, json.dumps({a:round(v,4) for a,v in sorted(g.items(),key=lambda x:x[1])}))
