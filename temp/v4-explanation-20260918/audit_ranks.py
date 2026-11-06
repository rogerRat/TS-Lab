import json
from pathlib import Path
import numpy as np
run=Path("/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918/rounds/seed0")
rows=[]
def ranks(x):
 vals,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
 mid=np.cumsum(cnt)-.5*(cnt+1)
 return mid[inv]
for model in ["chronos-base","timesfm-2.5"]:
 for task in ["time_nem_demand_5t","time_finland_15t","time_eweld_15t","time_azure_i_cpu_5t"]:
  arrays=[];ids=[]
  for method in ["less-main","rage-traj-maxSum"]:
   with np.load(run/"students"/model/"selections"/(task+"__"+method+".npz")) as d:
    meta=json.loads(str(d["resampling_metadata"]));diag=json.loads(str(d["diagnostics"]))
   audit=meta.get("reference_audit",{})
   arrays.append(np.asarray(audit.get("source_scores_for_softmax",audit.get("source_scores",meta.get("native_scores",[])))))
   ids.append(diag["source_sampling"]["trajectory_ids"] if method=="less-main" else [r["fragment_id"] for r in audit["source_trajectory_counts"]])
  assert ids[0]==ids[1]
  x,y=arrays;k=int(np.ceil(.1*len(x)))
  # Fractional treatment of boundary ties; common trajectory weights are uniform.
  def topweights(a):
   bound=np.sort(a)[-k];out=(a>bound).astype(float)
   ties=a==bound;out[ties]=(k-out.sum())/ties.sum()
   return out
  wx,wy=topweights(x),topweights(y)
  row=dict(model=model,task=task,trajectories=len(x),spearman=float(np.corrcoef(ranks(x),ranks(y))[0,1]),top10pct_overlap_fraction=float(np.minimum(wx,wy).sum()/k),definition="overlap relative to top10pct set size, fractional boundary ties")
  rows.append(row);print(json.dumps(row),flush=True)
out=Path("/root/TS3/temp/v4-explanation-20260918");(out/"less_traj_rank_audit.json").write_text(json.dumps(rows,indent=2))

