import json
from pathlib import Path
import numpy as np
from assets.gradient.dataset import decode_bf16
root=Path("/root/TS3/temp/v4-explanation-20260918")
run=Path("/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918/rounds/seed0")
proto=json.loads((run/"protocol.json").read_text())
with np.load(Path(proto["inputs"])/"pool.npz") as d:
 fids=d["ids"];legal=np.diff(d["offsets"])-575
# Freeze source uniform requirement from production formula, without importing a model/cache writer.
from rags.crb import CRBConfig
config=CRBConfig(context_length=512,prediction_length=64,**proto["crb"])
wanted=set()
for fid,n in zip(fids,legal):
 k=sum(config.budgets(int(n)))
 for s in np.rint(np.linspace(0,n-1,k)).astype(int)+1:wanted.add((int(fid),int(s)))
report={}
for model in proto["students"]:
 cache=Path(json.loads((run/"students"/model/"time_nem_demand_5t__gradient_assets.json").read_text())["path"])
 with np.load(cache/"window_index.npz") as d:
  ds=d["datasets"];di=d["dataset_ids"];ids=d["window_ids"];sh=d["shard_ids"];row=d["shard_rows"];names=d["shards"]
 use=np.asarray([i for i in np.flatnonzero(di==np.flatnonzero(ds=="source")[0]) if tuple(ids[i]) in wanted])
 norms=[]
 for sid in np.unique(sh[use]):
  ix=use[sh[use]==sid]
  with np.load(cache/str(names[sid])) as d:g=decode_bf16(d["gradients_bf16"])[row[ix]].astype(np.float64)
  norms.extend(np.linalg.norm(g,axis=1).tolist())
 a=np.asarray(norms);s=np.sort(a)
 record=dict(n=len(a),gradient_norm_quantiles=np.quantile(a,[0,.5,.9,.99,.999,1]).tolist(),top1pct_share_sum_norm=float(s[-int(np.ceil(len(a)*.01)):].sum()/s.sum()),top01pct_share_sum_norm=float(s[-int(np.ceil(len(a)*.001)):].sum()/s.sum()))
 report[model]=record;print(model,record,flush=True)
(root/"uniform_gradient_norm_audit.json").write_text(json.dumps(report,indent=2))

