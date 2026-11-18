"""Queue a fixed top-five-percent selection batch after full-pool resampling."""
from __future__ import annotations
import json, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from configs.logger import configure_logger
BATCH="data-select-005-linear"; SOURCE="q16-src256-eight-task-resample"
TASKS=("FEV09","TIME1","TIME2","TIME3","TIME4","TIME6","TIME7","TIME8")
SHARED=("rand","grand-static")
CONDITIONAL=("reft","crb","uni","crb-traj","uni-traj","tsds","lp_mirror")
SEL=ROOT/"outputs/selection/main-crb-v0"/BATCH; SRCSEL=ROOT/"outputs/selection/main-crb-v0"/SOURCE
SRCEVAL=ROOT/"outputs/eval/main-crb-v0"/SOURCE; SRCTRAIN=ROOT/"outputs/train/main-crb-v0"/SOURCE
POOL=SRCSEL/"shared/pool"; TRAIN=ROOT/"outputs/train/main-crb-v0"/BATCH
EVAL=ROOT/"outputs/eval/main-crb-v0"/BATCH; CFG=ROOT/"configs/experiments/main-crb-v0"/BATCH
NSELECT=319101
ENV=dict(os.environ); ENV["TSL_TOOL_ENV_DISPATCHED"]="1"
ENV["PYTHONPATH"]="/root/autodl-tmp/rage/third_parties/TIME/src"+os.pathsep+str(ROOT/"src")+os.pathsep+ENV.get("PYTHONPATH","")
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): ENV[k]="4"

def load_yaml(p): return yaml.safe_load(p.read_text(encoding="utf-8"))
def save_yaml(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(yaml.safe_dump(x,sort_keys=False,allow_unicode=False),encoding="ascii")
def save_ids(method,scope,ids,meta):
    d=SEL/method/scope; d.mkdir(parents=True,exist_ok=True); p=d/"fixed_subset.npz"
    np.savez_compressed(p,selected_indices=np.asarray(ids,dtype=np.int64),metadata=np.asarray(json.dumps(meta,sort_keys=True)))
    (d/".complete").write_text("complete\n",encoding="ascii"); return p
def top_ids(p,k):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0): raise ValueError("invalid marginal probabilities")
    pivot=len(p)-k; part=np.argpartition(p,pivot); threshold=float(p[part[pivot]])
    higher=np.flatnonzero(p>threshold); tied=np.flatnonzero(p==threshold)
    ids=np.sort(np.concatenate((higher,tied[:k-len(higher)])).astype(np.int64))
    if len(ids)!=k: raise ValueError("top-k selection size mismatch")
    return ids,threshold
def marginals(path,counts):
    with np.load(path,allow_pickle=False) as z:
        p=np.asarray(z["probabilities"],dtype=np.float64); layout=str(z["unit_layout"])
        meta=json.loads(str(z["resampling_metadata"]))
        offsets=np.asarray(z["unit_offsets"],dtype=np.int64) if "unit_offsets" in z.files else None
    total=int(counts.sum())
    if layout=="window":
        if len(p)!=total: raise ValueError("window probability size mismatch")
        return p,meta
    q=np.empty(total,dtype=np.float64)
    if layout=="trajectory":
        if len(p)!=len(counts) or len(offsets)!=len(counts)+1: raise ValueError("trajectory layout mismatch")
        cursor=0
        for i,n in enumerate(counts):
            stop=cursor+int(n)
            if offsets[i+1]-offsets[i]!=int(n): raise ValueError("trajectory window counts mismatch")
            q[cursor:stop]=p[i]/int(n); cursor=stop
    elif layout=="interval":
        if len(offsets)!=len(p)+1 or offsets[-1]!=total: raise ValueError("interval layout mismatch")
        for i,mass in enumerate(p):
            n=int(offsets[i+1]-offsets[i])
            if n<=0: raise ValueError("empty interval")
            q[offsets[i]:offsets[i+1]]=mass/n
    else: raise ValueError("unsupported unit layout "+layout)
    return q,meta
def prepare_selection(log):
    manifest=json.loads((POOL/"manifest.json").read_text(encoding="ascii"))
    counts=np.asarray(manifest["trajectory_legal_windows"],dtype=np.int64); total=int(manifest["total_equivalents"])
    if total!=6382020 or len(counts)!=256 or int(total*.05)!=NSELECT: raise ValueError("src pool differs from registration")
    for method in SHARED:
        p,source_meta=marginals(SRCSEL/method/"shared/resampling_payload.npz",counts)
        if method=="rand":
            ids=np.sort(np.random.default_rng(0).choice(total,NSELECT,replace=False)); cutoff=float(p[0])
            rule="uniform fixed subset with rng seed 0"
        else:
            ids,cutoff=top_ids(p,NSELECT); rule="rank legal windows by interval mass divided by owned-window count"
        save_ids(method,"shared",ids,dict(batch=BATCH,method=method,selected_windows=len(ids),pool_windows=total,
            fraction=.05,rule=rule,probability_cutoff=cutoff,tie_break="ascending global window index",seed=0,
            source_probability_batch=SOURCE,source_metadata=source_meta))
        log.info("stage=selection method=%s scope=shared n=%d cutoff=%.12g",method,len(ids),cutoff)
    for task in TASKS:
        for method in CONDITIONAL:
            if method=="reft":
                src=ROOT/"outputs/selection/main-crb-v0/q16-src256-eight-task"/task/"reference/training_windows.npz"
                with np.load(src,allow_pickle=False) as z: windows=np.asarray(z["windows"],dtype=np.float32)
                k=max(1,int(len(windows)*.05)); ids=np.sort(np.random.default_rng(0).choice(len(windows),k,replace=False))
                d=SEL/method/task; d.mkdir(parents=True,exist_ok=True); np.savez_compressed(d/"reference_subset.npz",windows=windows[ids])
                (d/"metadata.json").write_text(json.dumps({"batch":BATCH,"method":method,"seed":0,"selected_windows":k,
                    "pool_windows":len(windows),"rule":"uniform fixed five-percent of safe ref windows"},indent=2)+"\n",encoding="ascii")
                (d/".complete").write_text("complete\n",encoding="ascii"); log.info("stage=selection method=reft task=%s n=%d",task,k); continue
            p,source_meta=marginals(SRCSEL/method/task/"resampling_payload.npz",counts); ids,cutoff=top_ids(p,NSELECT)
            rule="rank legal windows by marginal draw probability; unit mass divided uniformly by legal-window count"
            if method in ("crb","uni"): rule+="; ref q8 gradients aggregated by linearSum"
            save_ids(method,task,ids,dict(batch=BATCH,method=method,task=task,selected_windows=len(ids),pool_windows=total,
                fraction=.05,rule=rule,probability_cutoff=cutoff,tie_break="ascending global window index",seed=0,
                source_probability_batch=SOURCE,source_metadata=source_meta))
            log.info("stage=selection method=%s task=%s n=%d cutoff=%.12g",method,task,len(ids),cutoff)
def prepare_configs():
    templates=ROOT/"configs/experiments/main-crb-v0/eight-task"
    for method in SHARED:
        tr=load_yaml(templates/"TIME1/crb-train.yaml"); tr["run_id"]="main-crb-v0-data-select-005-linear-shared-"+method
        data=dict(tr["data"]); data.pop("windows",None); data.pop("windows_key",None)
        data.update(id="src-256-top005-"+method,fixed_subset_payload=str(SEL/method/"shared/fixed_subset.npz"),resampling_pool=str(POOL))
        tr["data"]=data; tr["output_dir"]=str(TRAIN/"shared"/method); save_yaml(CFG/"shared"/(method+"-train.yaml"),tr)
        for task in TASKS:
            ev=load_yaml(templates/task/"ori-eval.yaml"); ev["checkpoint"]=str(TRAIN/"shared"/method/"checkpoint")
            ev["output_dir"]=str(EVAL/task/method); save_yaml(CFG/task/(method+"-eval.yaml"),ev)
    for task in TASKS:
        for method in CONDITIONAL:
            tr=load_yaml(templates/task/"reft-train.yaml"); tr["run_id"]="main-crb-v0-data-select-005-linear-"+task.lower()+"-"+method
            data=dict(tr["data"])
            if method=="reft": data.update(id=task.lower()+"-safe-reference-top005",windows=str(SEL/method/task/"reference_subset.npz"),windows_key="windows")
            else:
                data.pop("windows",None); data.pop("windows_key",None)
                data.update(id="src-256-top005-"+method+"-"+task.lower(),fixed_subset_payload=str(SEL/method/task/"fixed_subset.npz"),resampling_pool=str(POOL))
            tr["data"]=data; tr["output_dir"]=str(TRAIN/task/method); save_yaml(CFG/task/(method+"-train.yaml"),tr)
            ev=load_yaml(templates/task/"ori-eval.yaml"); ev["checkpoint"]=str(TRAIN/task/method/"checkpoint")
            ev["output_dir"]=str(EVAL/task/method); save_yaml(CFG/task/(method+"-eval.yaml"),ev)
        ori=load_yaml(templates/task/"ori-eval.yaml")
        ori["checkpoint"]=str(ROOT/"outputs/train/main-crb-v0/q16-src256-eight-task"/task/"ori/checkpoint")
        ori["output_dir"]=str(EVAL/task/"ori"); save_yaml(CFG/task/"ori-eval.yaml",ori)
        for method in ("ori",)+SHARED+CONDITIONAL:
            if method=="ori": stages=[{"name":"eval","config":str((CFG/task/"ori-eval.yaml").relative_to(ROOT))}]
            elif method in SHARED: stages=[{"name":"eval","config":str((CFG/task/(method+"-eval.yaml")).relative_to(ROOT))}]
            else: stages=[{"name":"train","config":str((CFG/task/(method+"-train.yaml")).relative_to(ROOT))},{"name":"eval","config":str((CFG/task/(method+"-eval.yaml")).relative_to(ROOT))}]
            save_yaml(CFG/task/(method+"-pipeline.yaml"),{"schema_version":1,"run_id":"main-crb-v0-data-select-005-linear-"+task.lower()+"-"+method,"stages":stages})
def command(task,method): return [sys.executable,"-u","-m","scripts.tools.run","--config",str((CFG/task/(method+"-pipeline.yaml")).relative_to(ROOT))]
def train_shared(method): return [sys.executable,"-u","-m","scripts.tools.train","--config",str((CFG/"shared"/(method+"-train.yaml")).relative_to(ROOT))]
def parallel(jobs,limit,log):
    pending=list(jobs); active=[]
    while pending or active:
        while pending and len(active)<limit:
            name,cmd=pending.pop(0); proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,start_new_session=True)
            active.append((name,proc)); log.info("stage=job-start name=%s",name)
        for name,proc in list(active):
            code=proc.poll()
            if code is None: continue
            active.remove((name,proc)); log.info("stage=job-end name=%s code=%d",name,code)
            if code:
                for _,other in active:
                    if other.poll() is None: os.killpg(other.pid,signal.SIGINT)
                for _,other in active: other.wait()
                raise RuntimeError("job failed: "+name)
        if active: time.sleep(1)
def execute(log):
    prepare_selection(log); prepare_configs()
    jobs=[("train-"+m,train_shared(m)) for m in SHARED if not (TRAIN/"shared"/m/".complete").is_file()]
    parallel(jobs,2,log)
    jobs=[(t+"-"+m,command(t,m)) for m in SHARED+("ori",) for t in TASKS if not (EVAL/t/m/".complete").is_file()]
    parallel(jobs,8,log)
    parallel([(t,[sys.executable,"-u",str(Path(__file__)),"--task",t]) for t in TASKS],8,log)
    (EVAL/".complete").write_text("complete\n",encoding="ascii"); log.info("stage=batch-complete train=%s eval=%s",TRAIN,EVAL)
def run_task(task):
    log=configure_logger("main_crb_data_select_005_linear","main-crb-v0",BATCH,"task-"+task.lower(),enable_heartbeat=True)
    for method in CONDITIONAL:
        if (EVAL/task/method/".complete").is_file(): continue
        code=subprocess.call(command(task,method),cwd=ROOT,env=ENV)
        if code or not (EVAL/task/method/".complete").is_file(): raise RuntimeError("task arm failed: "+method)
    (EVAL/task/".task-complete").write_text("complete\n",encoding="ascii")
def wait_prior(log):
    marker=SRCEVAL/"native-task-conditioned.complete"
    while True:
        neval=sum(1 for _ in SRCEVAL.glob("*/*/.complete")); ntrain=sum(1 for _ in SRCTRAIN.glob("*/*/.complete"))
        if marker.is_file() and neval>=80 and ntrain>=58: log.info("stage=prior-batch-complete eval=%d train=%d",neval,ntrain); return
        log.info("stage=waiting-prior-batch native=%s eval=%d/80 train=%d/58",marker.is_file(),neval,ntrain); time.sleep(60)
def main():
    log=configure_logger("main_crb_data_select_005_linear","main-crb-v0",BATCH,"queue",enable_heartbeat=True)
    log.info("stage=config source=%s tasks=%s methods=ori,%s,%s fraction=0.05 top_windows=%d linearSum=true parallelism=8",SOURCE,",".join(SHARED),",".join(CONDITIONAL),NSELECT)
    wait_prior(log); execute(log)
if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--task": run_task(sys.argv[2])
    else: main()
