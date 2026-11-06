import json,time,os
from pathlib import Path
import numpy as np
from collections import OrderedDict
from rage.crb import CRBConfig,CRBEngineMulti
from rage.experiments.development_inputs import load_fragments
from assets.gradient.dataset import GradientDataset,decode_bf16
from assets.gradient.cache import FragmentGradients
run=Path("/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918/rounds/seed0")
old=Path("/root/autodl-tmp/TS3/outputs/main-shared-nine-v2-global-20260917")
out=Path("/root/TS3/temp/v4-explanation-20260918");out.mkdir(exist_ok=True)
protocol=json.loads((run/"protocol.json").read_text())
tasks=["time_nem_demand_5t","time_finland_15t","time_eweld_15t","time_azure_i_cpu_5t"]
methods=["less-main"]+["rage-"+g+"-"+a for g in ["block","traj"] for a in ["linearSum","zeroClipSum","maxSum","expSum"]]
def unit(x):
 n=np.linalg.norm(x,axis=-1,keepdims=True);return np.divide(x,n,out=np.zeros_like(x,dtype=float),where=n>0)
def cos(x,y):return float(unit(x)@unit(y))
def soft(x):
 x=x-x.max();p=np.exp(x);return p/p.sum()
result={"run":str(run),"estimator":"shared production midpoint raw gradients; reconstructed proxy, not exact window truth or optimizer update","models":[]}
for model in protocol["students"]:
 folder=run/"students"/model
 sels={}
 for task in tasks:
  for method in methods:
   path=folder/"selections"/(task+"__"+method+".npz")
   if not path.exists():continue
   with np.load(path) as d:
    p=d["probabilities"];off=d["unit_offsets"];count=np.diff(off);pos=d["unit_positions"][off[:-1]]
    meta=json.loads(str(d["resampling_metadata"]));audit=meta.get("reference_audit",{})
    score=np.asarray(audit.get("source_scores_for_softmax",audit.get("source_scores",meta.get("native_scores",[]))))
    sels[(task,method)]=dict(p=p,count=count,pos=pos,meta=meta,score=score)
 # pool geometry from common input only
 with np.load(Path(protocol["inputs"])/"pool.npz") as d:
  fids=d["ids"];legal=np.diff(d["offsets"])-575
 lo=np.cumsum(np.r_[0,legal[:-1]]);total=legal.sum()
 blockexample=next(s for (t,m),s in sels.items() if "block" in m)
 blockfidindex=np.searchsorted(lo,blockexample["pos"],side="right")-1
 probrows=[]
 for (task,method),s in sels.items():
  p=s["p"];count=s["count"];ix=np.searchsorted(lo,s["pos"],side="right")-1
  tp=np.bincount(ix,weights=p,minlength=len(fids))
  row=dict(task=task,method=method,units=len(p),reference_count=s["meta"].get("reference_audit",{}).get("reference_count",s["meta"].get("reference_count")),score_std=float(s["score"].std()),logit_std=float(s["score"].std()/.05),tv_unit_uniform=float(abs(p-1/len(p)).sum()/2),unit_ess=float(1/(p@p)),trajectory_ess=float(1/(tp@tp)),top10pct_mass=float(np.sort(tp)[-int(np.ceil(len(tp)/10)):].sum()),expected_legal_length=float(tp@legal),tv_window_uniform=float(abs(p-count/total).sum()/2))
  cell=folder/"cells"/(task+"__"+method+"__seed0.json")
  if cell.exists(): row["mase"]=json.loads(cell.read_text())["mase"]
  if method=="rage-traj-maxSum":
   less=sels[(task,"less-main")]
   a=soft(s["score"]*row["reference_count"]/.05);b=soft(less["score"]/row["reference_count"]/.05)
   row.update(tv_to_less_actual=float(abs(p-less["p"]).sum()/2),tv_anchor_only_at_less_scale=float(abs(a-less["p"]).sum()/2),tv_anchor_only_at_traj_scale=float(abs(p-b).sum()/2))
  probrows.append(row)
 print(model,"probability",json.dumps(probrows),flush=True)
 # readonly indexed cache, no lock, no publication, no missing gradient computation
 cachepath=Path(json.loads((folder/(tasks[0]+"__gradient_assets.json")).read_text())["path"])
 reader=GradientDataset(cachepath)
 class Cache:
  context_length=512;prediction_length=64
  def __init__(self):self.lru=OrderedDict();self.bytes=0
  def get_many(self,requests):
   grouped={};outs=[None]*len(requests)
   for i,r in enumerate(requests):
    for target,(fid,start) in enumerate(r.ids):
     j=reader._positions[(r.dataset,int(fid),int(start))]
     grouped.setdefault(int(reader.shard_ids[j]),[]).append((i,target,int(reader.shard_rows[j])))
   for sh,locs in grouped.items():
    if sh not in self.lru:
     with np.load(cachepath/str(reader.shards[sh])) as d:v=decode_bf16(d["gradients_bf16"])
     self.lru[sh]=v;self.bytes+=v.nbytes
     while self.bytes>2*2**30 and len(self.lru)>1:self.bytes-=self.lru.popitem(last=False)[1].nbytes
    v=self.lru[sh];self.lru.move_to_end(sh)
    for i,target,row in locs:
     if outs[i] is None:outs[i]=np.empty((len(requests[i].ids),4096),np.float32)
     outs[i][target]=v[row]
   return outs
 pool=load_fragments(Path(protocol["inputs"])/"pool.npz")
 config=CRBConfig(context_length=512,prediction_length=64,**protocol["crb"])
 engine=CRBEngineMulti(config,FragmentGradients(Cache(),"source"))
 keys=list(sels);sums=np.zeros((len(keys)+2,4096));dirs=np.zeros_like(sums)
 anchorpool=np.zeros(4096);anchortraj=[]
 cursor=0;tic=time.time()
 for begin in range(0,len(pool),64):
  batch=engine.process_fragments(pool[begin:begin+64])
  for fi,r in enumerate(batch,begin):
   cnt=np.asarray([len(b.start_positions) for b in r.blocks]);n=len(cnt)
   assert np.array_equal(cnt,blockexample["count"][cursor:cursor+n])
   assert np.array_equal(lo[fi]+np.asarray([b.start_positions[0]-1 for b in r.blocks]),blockexample["pos"][cursor:cursor+n])
   raw=np.asarray([b.mean_gradient for b in r.blocks],dtype=np.float64);dr=unit(raw)
   coeff=np.zeros((len(keys)+2,n))
   for k,key in enumerate(keys):
    s=sels[key]
    if "block" in key[1]:coeff[k]=s["p"][cursor:cursor+n]
    else:coeff[k]=s["p"][fi]*cnt/legal[fi]
   coeff[-2]=cnt/total;coeff[-1]=cnt/legal[fi]/len(pool)
   sums+=coeff@raw;dirs+=coeff@dr
   aa=np.mean(np.asarray(list(r.anchors.values()),dtype=np.float64),axis=0);anchortraj.append(aa);anchorpool+=aa/len(pool)
   cursor+=n
  print(model,"replay",min(begin+64,len(pool)),"seconds",round(time.time()-tic,1),flush=True)
 assert cursor==len(blockexample["p"])
 refs={};alignment=[]
 for task in tasks:
  path=old/"students"/model/"reference_assets"/task
  fields=json.loads((path/"protocol.json").read_text())
  assert fields["model"]==json.loads((folder/"model.json").read_text())
  assert fields["crb"]==json.loads((old/"protocol.json").read_text())["crb"] or all(fields["crb"][k]==v for k,v in protocol["crb"].items())
  files=sorted(path.glob("trajectory-*.npz"));rawref=np.zeros(4096);qb=np.zeros(4096);qt=np.zeros(4096);coh=[];refaa=[]
  for f in files:
   with np.load(f) as d:
    cnt=np.diff(d["block_offsets"]);w=cnt/cnt.sum()
    rawref+=w@d["block_means"]/len(files);q=w@d["block_directions"];qb+=q/len(files);coh.append(float(np.linalg.norm(q)))
    a=np.mean(d["anchors"],axis=0,dtype=np.float64);qt+=unit(a)/len(files);refaa.append(a)
  refs[task]=dict(within_coherence=coh,block_Q_norm=float(np.linalg.norm(qb)),traj_Q_norm=float(np.linalg.norm(qt)),Q_direction_cosine=cos(qb,qt),raw_reference_vs_trajQ=cos(rawref,qt),raw_ref_vs_uniform_window_pool=cos(rawref,sums[-2]),raw_ref_vs_uniform_trajectory_pool=cos(rawref,sums[-1]),block_direction_ref_vs_window_pool=cos(qb,dirs[-2]))
  for k,key in enumerate(keys):
   if key[0]!=task:continue
   alignment.append(dict(task=task,method=key[1],raw_mean_cosine=cos(rawref,sums[k]),raw_mean_dot=float(rawref@sums[k]),raw_mean_norm=float(np.linalg.norm(sums[k])),direction_mean_cosine=cos(qb,dirs[k]),mean_window_cosine_to_rawref=float(dirs[k]@unit(rawref)),reference_linear_score=float(dirs[k]@qb)))
  for k,label in [(-2,"uniform_windows"),(-1,"uniform_trajectories")]:
   alignment.append(dict(task=task,method=label,raw_mean_cosine=cos(rawref,sums[k]),raw_mean_dot=float(rawref@sums[k]),raw_mean_norm=float(np.linalg.norm(sums[k])),direction_mean_cosine=cos(qb,dirs[k]),mean_window_cosine_to_rawref=float(dirs[k]@unit(rawref)),reference_linear_score=float(dirs[k]@qb)))
 record=dict(model=model,probabilities=probrows,references=refs,alignment=alignment,source_blocks=cursor)
 result["models"].append(record)
 (out/"diagnostics.json").write_text(json.dumps(result,indent=2))
 print(model,"DONE",json.dumps(dict(references=refs,alignment=alignment)),flush=True)

