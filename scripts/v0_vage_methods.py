"""Compare VAGE allocation methods with fixed one-anchor initialization."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from configs.logger import configure_logger, format_duration
from configs.paths import PROJECT_ROOT, output_path
from scripts.v0_vage_static import allocate, metrics, spaced, split_gap, vector_variance

SOURCE = PROJECT_ROOT / "outputs/train/v0-gradient-proxy-test"
DEST = output_path("train", "outputs/train/v0-vage-methods-nmin1")
METHODS = ("uniform", "vage_static", "vage_quota", "vage_fusion", "vage_residual", "oracle_vage")
KAPPA = 8.0


def blocks_for(n):
    return [(lo, min(512, n-lo)) for lo in range(0, n, 512)]


def proxy_samples(blocks, budget):
    counts = allocate(budget, [length for _, length in blocks], 2)
    return [spaced(lo, length, count) for (lo, length), count in zip(blocks, counts)]


def initial(blocks, proxy=None):
    result = []
    for i, (lo, length) in enumerate(blocks):
        center = lo + (length-1)//2
        result.append([min(proxy[i], key=lambda x: (abs(x-center), x))] if proxy is not None
                      else [center])
    return result


def add_one(anchors, block):
    lo, length = block
    point = split_gap(anchors, lo, length)
    anchors.append(point)
    anchors.sort()
    return point


def static(blocks, budget, variance):
    anchors = initial(blocks)
    while sum(map(len, anchors)) < budget:
        scores = [length * variance[i] / (len(a)*(len(a)+1)) if len(a)<length else -1
                  for i, ((_, length), a) in enumerate(zip(blocks, anchors))]
        add_one(anchors[int(np.argmax(scores))], blocks[int(np.argmax(scores))])
    return sorted(x for group in anchors for x in group)


def fusion(blocks, budget, prior, full, scale):
    anchors = initial(blocks)
    while sum(map(len, anchors)) < budget:
        scores=[]
        for i, ((_, length), a) in enumerate(zip(blocks, anchors)):
            if len(a)==1:
                estimate=scale*prior[i]
            else:
                sample=vector_variance(full,a)
                estimate=(KAPPA*scale*prior[i]+(len(a)-1)*sample)/(KAPPA+len(a)-1)
            scores.append(length*estimate/(len(a)*(len(a)+1)) if len(a)<length else -1)
        chosen=int(np.argmax(scores))
        add_one(anchors[chosen],blocks[chosen])
    return sorted(x for group in anchors for x in group)


def quota(blocks, budget, variance):
    lengths=[length for _,length in blocks]
    counts=np.ones(len(blocks),dtype=int)
    weights=np.asarray(lengths)*np.sqrt(np.maximum(variance,0)+1e-12)
    while int(counts.sum())<budget:
        eligible=counts<lengths
        target=budget*weights/weights.sum()
        score=np.where(eligible,target-counts,-np.inf)
        counts[int(np.argmax(score))]+=1
    return sorted(x for (lo,length),count in zip(blocks,counts) for x in spaced(lo,length,int(count)))


def proxy_error(head, points, anchors, length):
    p=np.asarray(points,dtype=int)
    a=np.asarray(sorted(anchors),dtype=int)
    idx=np.searchsorted(a,p,side="right")
    left=a[np.clip(idx-1,0,len(a)-1)]
    right=a[np.clip(idx,0,len(a)-1)]
    frac=np.divide(p-left,right-left,out=np.zeros(len(p),dtype=float),where=right!=left)
    x,y=1-frac,frac
    sq=np.diag(head)[p]+x*x*head[left,left]+y*y*head[right,right]+2*x*y*head[left,right]-2*x*head[p,left]-2*y*head[p,right]
    return length*float(np.maximum(0,sq).mean())


def residual(blocks,budget,head,proxy):
    anchors=initial(blocks,proxy)
    while sum(map(len,anchors))<budget:
        best=None
        for h,((_,length),points) in enumerate(zip(blocks,proxy)):
            current=proxy_error(head,points,anchors[h],length)
            for point in points:
                if point in anchors[h]:
                    continue
                benefit=current-proxy_error(head,points,anchors[h]+[point],length)
                candidate=(benefit,-h,-point,h,point)
                if best is None or candidate[:3]>best[:3]:
                    best=candidate
        if best is None:
            raise ValueError("proxy candidate pool exhausted")
        anchors[best[3]].append(best[4])
        anchors[best[3]].sort()
    return sorted(x for group in anchors for x in group)


def main():
    if (DEST/".complete").exists():
        raise ValueError("completed output exists")
    DEST.mkdir(parents=True,exist_ok=True)
    log=configure_logger("v0.vage.methods","v0-vage","methods","replay")
    start=time.monotonic()
    log.info("Stage start: method comparison; n_min=1 Bg=Bp=ceil(N/64) L=512 kappa=%s output=%s",KAPPA,DEST)
    cache=[]
    for t in range(10):
        name="trajectory-%02d"%t
        directory=SOURCE/name
        manifest=json.loads((directory/"manifest.json").read_text(encoding="ascii"))
        n=int(manifest["windows"])
        grad=PROJECT_ROOT/"outputs/train/v0-crb-reconst-test/gradients"/(name+".bf16")
        if not (directory/".complete").exists() or not grad.with_suffix(".complete").exists() or grad.stat().st_size!=n*8394496*2:
            raise ValueError("incomplete source: "+name)
        with np.load(directory/"gram_matrices.npz") as z:
            full=np.array(z["full"]);head=np.array(z["head"])
        if full.shape!=(n,n) or head.shape!=(n,n) or not np.isfinite(full).all() or not np.isfinite(head).all():
            raise ValueError("invalid Gram: "+name)
        blocks=blocks_for(n);bg=math.ceil(n/64);bp=math.ceil(n/64)
        proxy=proxy_samples(blocks,bp)
        observed=np.array([vector_variance(head,p) for p in proxy])
        prior=float(np.average(observed,weights=[length for _,length in blocks]))
        variance=np.array([min(1,(len(p)-1)/7)*v+(1-min(1,(len(p)-1)/7))*prior for p,v in zip(proxy,observed)])
        true=np.array([max(0,float(np.mean(np.diag(full)[lo:lo+length])-np.mean(full[lo:lo+length,lo:lo+length]))) for lo,length in blocks])
        cache.append(dict(name=name,n=n,full=full,head=head,blocks=blocks,bg=bg,bp=bp,proxy=proxy,variance=variance,true=true))
    rows=[];anchor_records=[];scale_records=[]
    for i,item in enumerate(cache):
        # Calibrate only on the other nine trajectories, never on held-out full gradients.
        ratios=[float(v/p) for j,other in enumerate(cache) if j!=i for v,p in zip(other["true"],other["variance"]) if p>0]
        scale=float(np.median(ratios))
        scale_records.append(dict(track=item["name"],scale=scale,kappa=KAPPA))
        blocks=item["blocks"];bg=item["bg"];n=item["n"]
        selections={
            "uniform":spaced(0,n,bg),
            "vage_static":static(blocks,bg,item["variance"]),
            "vage_quota":quota(blocks,bg,item["variance"]),
            "vage_fusion":fusion(blocks,bg,item["variance"],item["full"],scale),
            "vage_residual":residual(blocks,bg,item["head"],item["proxy"]),
            "oracle_vage":static(blocks,bg,item["true"]),
        }
        for method,anchors in selections.items():
            if len(anchors)!=bg or len(set(anchors))!=bg:
                raise ValueError("anchor budget mismatch: "+method)
            rows.append(dict(track=item["name"],method=method,windows=n,gradient_budget=bg,
                             proxy_budget=0 if method in ("uniform","oracle_vage") else item["bp"],
                             **metrics(item["full"],anchors)))
            anchor_records.append(dict(track=item["name"],method=method,anchors=anchors))
        log.info("Track complete: %s Bg=%d Bp=%d scale=%.4f",item["name"],bg,item["bp"],scale)
    with (DEST/"trajectory_metrics.csv").open("w",newline="",encoding="ascii") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (DEST/"fusion_calibration.csv").open("w",newline="",encoding="ascii") as f:
        w=csv.DictWriter(f,fieldnames=list(scale_records[0]));w.writeheader();w.writerows(scale_records)
    (DEST/"anchors.json").write_text(json.dumps(anchor_records,indent=2)+"\n",encoding="ascii")
    summary={m:{k:float(np.mean([r[k] for r in rows if r["method"]==m])) for k in ("mean_l2","mean_cosine","relative_rmse","p95_l2","p05_cosine")} for m in METHODS}
    (DEST/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="ascii")
    (DEST/".complete").write_text("complete\n",encoding="ascii")
    log.info("Stage complete: method comparison; duration=%s output=%s",format_duration(time.monotonic()-start),DEST)

if __name__=="__main__":
    main()
