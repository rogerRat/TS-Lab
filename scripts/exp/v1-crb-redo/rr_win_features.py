"""Create P1 filtered per-window direction features for win-level RR."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
from assets.gradient.dataset import GradientDataset
from configs.paths import PROJECT_ROOT,output_path
from rage.experiments.development_inputs import load_fragments
from rr_features import SRC_ROOT,REF_ROOT,REF_INPUTS,record_anchors,source_layout,fetch,TASKS
ROOT=PROJECT_ROOT
def interpolate(anchors,grads,threshold,target,offset=0):
    norms=np.linalg.norm(grads,axis=1)
    unit=np.divide(grads,norms[:,None],out=np.zeros_like(grads),where=norms[:,None]>0)
    unit[norms<=threshold]=0.
    for j,width in enumerate(np.diff(anchors)):
        count=int(width+(1 if j==len(anchors)-2 else 0))
        ts=np.arange(count,dtype=np.float64)/float(width)
        if j==len(anchors)-2: ts=np.linspace(0.,1.,count)
        vals=unit[j][None,:]*(1-ts[:,None])+unit[j+1][None,:]*ts[:,None]
        norms2=np.linalg.norm(vals,axis=1)
        vals=np.divide(vals,norms2[:,None],out=np.zeros_like(vals),where=norms2[:,None]>0)
        left=int(offset+anchors[j]); target[left:left+count]=vals.astype(np.float32)
def build_source_windows(method):
    config,legal,offsets,files=source_layout()
    out=output_path("selection",ROOT/"outputs/selection/v1-crb-redo/features"/method/"win")
    path=out/"source_windows.npy"; meta=out/"source_windows.json"
    if path.is_file() and meta.is_file(): return path
    out.mkdir(parents=True,exist_ok=True)
    norms=[]
    for fid in range(96):
        record,cache=files[(method,fid)]; anchors=record_anchors(record,method)
        vals=fetch(cache,"src-96",fid,anchors)
        norms.extend(np.linalg.norm(vals,axis=1).tolist())
    threshold=float(np.quantile(norms,.01))
    values=np.load(ROOT/yaml.safe_load((ROOT/"configs/data/src-96.yaml").read_text())["values_path"],mmap_mode="r")
    dest=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(int(legal.sum()),4096))
    for fid in range(96):
        record,cache=files[(method,fid)]; anchors=record_anchors(record,method)
        vals=fetch(cache,"src-96",fid,anchors)
        interpolate(anchors,vals,threshold,dest,int(offsets[fid]))
    dest.flush(); del dest
    meta.write_text(json.dumps(dict(method=method,threshold=threshold,windows=int(legal.sum()),dimension=4096))+"\n",encoding="ascii")
    return path
def build_reference_windows(task,method):
    out=output_path("selection",ROOT/"outputs/selection/v1-crb-redo/features"/method/"win"/task)
    path=out/"reference_windows.npy"; meta=out/"reference_windows.json"
    if path.is_file() and meta.is_file(): return path
    out.mkdir(parents=True,exist_ok=True)
    base=REF_ROOT/task; manifest=json.loads((base/"manifest.json").read_text())
    cache=GradientDataset(Path(manifest["cache"])); refs=load_fragments(REF_INPUTS/task/"references.npz")
    records=[]; norms=[]
    for frag in refs:
        a=record_anchors(base/method/("%05d.json"%frag.fragment_id),method)
        g=fetch(cache,"reference:"+task,frag.fragment_id,a); records.append((frag,a,g))
        norms.extend(np.linalg.norm(g,axis=1).tolist())
    threshold=float(np.quantile(norms,.01)); total=int(manifest["legal_windows"])
    dest=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(total,4096)); cur=0
    for frag,a,g in records:
        count=frag.length-575
        interpolate(a,g,threshold,dest,cur); cur+=count
    if cur!=total: raise ValueError("Reference window count mismatch")
    dest.flush(); del dest
    meta.write_text(json.dumps(dict(task=task,method=method,threshold=threshold,windows=total))+"\n",encoding="ascii")
    return path
