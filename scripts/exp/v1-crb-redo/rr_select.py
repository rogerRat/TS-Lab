"""Build one RR window set from representative cosine scores."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from configs.logger import configure_logger,format_duration
from configs.paths import PROJECT_ROOT,output_path
from rr_features import build_reference,build_source
from rr_selection import round_robin_units
ROOT=PROJECT_ROOT
METHODS={"base-uni":"uniform","ours-rage101":"crb"}
TASKS=("fev-09","time-01","time-02","time-03","time-04","time-06","time-07","time-08")
WINDOW_BUDGET=266631
def build_scores(refs,sources,path,logger):
    refs=np.asarray(refs,dtype=np.float32); sources=np.asarray(sources,dtype=np.float32)
    scores=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(refs),len(sources)))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_gpu=torch.as_tensor(sources,device=device); started=time.monotonic()
    for rb in range(0,len(refs),16):
        re=min(rb+16,len(refs)); ref_gpu=torch.as_tensor(refs[rb:re],device=device)
        scores[rb:re]=torch.matmul(ref_gpu,source_gpu.T).cpu().numpy()
        if rb==0 or re==len(refs) or re%128==0:
            elapsed=time.monotonic()-started
            logger.info("stage=score-progress refs=%d/%d device=%s eta=%s",re,len(refs),device,
                        format_duration(elapsed*(len(refs)-re)/max(re,1)))
    scores.flush()
    return scores
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--family",choices=METHODS,required=True)
    parser.add_argument("--granularity",choices=("traj","block"),required=True)
    parser.add_argument("--task",choices=TASKS,required=True)
    args=parser.parse_args(); method=METHODS[args.family]; started=time.monotonic()
    batch="timerXL-%s-01-%s-rr"%(args.family,args.granularity)
    logger=configure_logger("v1_crb_redo_rr_select","v1-crb-redo",batch+"-"+args.task,"selection")
    source_dir=ROOT/"outputs/selection/v1-crb-redo"/"features"/method/args.granularity
    reference_dir=source_dir/args.task
    build_source(method,args.granularity,source_dir)
    build_reference(args.task,method,args.granularity,reference_dir)
    with np.load(source_dir/"source_units.npz",allow_pickle=False) as x:
        src_dirs=x["directions"]; src_counts=x["counts"]; starts=x["starts"]
    with np.load(reference_dir/"reference_units.npz",allow_pickle=False) as x:
        ref_dirs=x["directions"]
    root=output_path("selection",ROOT/"outputs/selection/v1-crb-redo"/batch/args.task)
    if (root/".complete").exists():
        logger.info("stage=skip reason=complete"); return
    root.mkdir(parents=True,exist_ok=True)
    scores=build_scores(ref_dirs,src_dirs,root/"scores.npy",logger)
    units,take=round_robin_units(scores,src_counts,WINDOW_BUDGET)
    chosen=[]
    for unit,n in zip(units,take):
        chosen.extend(range(int(starts[unit]),int(starts[unit])+int(n)))
    chosen=np.asarray(chosen,dtype=np.int64)
    if len(chosen)!=WINDOW_BUDGET or len(np.unique(chosen))!=WINDOW_BUDGET:
        raise ValueError("RR window budget or uniqueness check failed")
    tmp=root/"window_indices.partial.npy"; np.save(tmp,chosen,allow_pickle=False)
    tmp.replace(root/"window_indices.npy")
    selection_elapsed=time.monotonic()-started
    selection_bytes=(root/"window_indices.npy").stat().st_size
    score_path=root/"scores.npy"
    if score_path.exists(): score_path.unlink()
    manifest=dict(batch=batch,task=args.task,family=args.family,gradient_method=method,
        granularity=args.granularity,selection_algorithm="per_reference_unit_greedy_round_robin",
        window_budget=WINDOW_BUDGET,source_units=int(len(src_dirs)),reference_units=int(len(ref_dirs)),
        selected_units=units.tolist(),selected_unit_window_counts=take.tolist(),
        selected_windows=int(len(chosen)),selection_elapsed_seconds=selection_elapsed,
        selection_file_bytes=selection_bytes,window_order="unit_rr_order_then_global_start_ascending",
        tie_break="source_unit_order_then_global_window_start")
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="ascii")
    (root/".complete").write_text("complete\n",encoding="ascii")
    logger.info("stage=complete windows=%d units=%d output=%s",len(chosen),len(units),root)
if __name__=="__main__":
    main()
