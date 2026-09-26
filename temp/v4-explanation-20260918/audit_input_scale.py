import json
from pathlib import Path
import numpy as np
from assets.gradient.dataset import decode_bf16
root=Path("/root/TS3/temp/v4-explanation-20260918")
run=Path("/root/autodl-tmp/TS3/outputs/main-v4-three-seeds-20260918/rounds/seed0")
proto=json.loads((run/"protocol.json").read_text())
with np.load(Path(proto["inputs"])/"pool.npz") as d:
 fids=d["ids"];offsets=d["offsets"];values=d["values"];entities=d["entities"]
fidmap={int(fid):i for i,fid in enumerate(fids)}
cache=Path(json.loads((run/"students"/"timesfm-2.5"/"time_nem_demand_5t__gradient_assets.json").read_text())["path"])
with np.load(cache/"window_index.npz") as d:
 ds=d["datasets"];di=d["dataset_ids"];ids=d["window_ids"];sh=d["shard_ids"];row=d["shard_rows"];names=d["shards"]
# Bound by cached source shards; retain the largest norm windows plus a fixed evenly spread control sample.
use=np.flatnonzero(di==np.flatnonzero(ds=="source")[0])
norms=np.zeros(len(use))
for sid in np.unique(sh[use]):
 ix=np.flatnonzero(sh[use]==sid)
 with np.load(cache/str(names[sid])) as d:g=decode_bf16(d["gradients_bf16"])[row[use[ix]]].astype(np.float64)
 norms[ix]=np.linalg.norm(g,axis=1)
test=np.unique(np.r_[np.argsort(norms)[-100:],np.linspace(0,len(use)-1,2000).astype(int)])
std=[];maximum=[];records=[]
topten=set(np.argsort(norms)[-10:])
for j in test:
 fid,start=ids[use[j]];fi=fidmap[int(fid)];window=values[offsets[fi]+start-1:offsets[fi]+start-1+576].astype(float)
 std.append(float(window.std()));maximum.append(float(abs(window).max()))
 if j in topten:records.append(dict(fid=int(fid),start=int(start),entity=str(entities[fi]),norm=float(norms[j]),input_std=std[-1],input_absmax=maximum[-1]))
std=np.asarray(std);maximum=np.asarray(maximum);n=norms[test]
corr=lambda a,b:float(np.corrcoef(np.log10(np.maximum(a,1e-20)),np.log10(np.maximum(b,1e-20)))[0,1])
report=dict(model="timesfm-2.5",source_cached_windows=len(use),sample_windows=len(test),lognorm_logstd_correlation=corr(n,std),lognorm_logabsmax_correlation=corr(n,maximum),input_std_quantiles=np.quantile(std,[.5,.9,.99,1]).tolist(),top_gradient_examples=records,interpretation="bounded fixed control sample plus 100 largest cached source gradient norms; descriptive association only")
(root/"timesfm_input_scale_audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

