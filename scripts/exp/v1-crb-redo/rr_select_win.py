"""Build one BF16 full window-score matrix and select its RR window set."""
from __future__ import annotations
import argparse,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from configs.logger import configure_logger,format_duration
from configs.paths import PROJECT_ROOT,output_path
from rr_win_features import build_reference_windows,build_source_windows,TASKS
ROOT=PROJECT_ROOT; BUDGET=266631; SOURCE_CHUNK=32768
METHODS={"base-uni":"uniform","ours-rage101":"crb"}

def write_json(path,value):
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="ascii")
    tmp.replace(path)

def build_score_matrix(refs,sources,path,logger):
    n_refs,n_sources=len(refs),len(sources)
    expected=n_refs*n_sources*2
    if path.is_file() and path.stat().st_size==expected:
        logger.info("stage=score-matrix-reuse refs=%d sources=%d dtype=bfloat16 bytes=%d",n_refs,n_sources,expected)
        return expected
    if path.exists(): path.unlink()
    partial=path.with_suffix(path.suffix+".partial")
    if partial.exists(): partial.unlink()
    free=shutil.disk_usage(path.parent).free
    if free<expected+1024**3:
        raise OSError("Insufficient free disk for BF16 score matrix and 1 GiB headroom")
    logger.info("stage=score-matrix-start refs=%d sources=%d shape=%dx%d dtype=bfloat16 bytes=%d chunk=%d",
                n_refs,n_sources,n_refs,n_sources,expected,SOURCE_CHUNK)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_gpu=torch.tensor(np.asarray(refs),device=device,dtype=torch.bfloat16)
    matrix=np.memmap(partial,mode="w+",dtype=np.uint16,shape=(n_refs,n_sources))
    started=time.monotonic()
    for left in range(0,n_sources,SOURCE_CHUNK):
        right=min(left+SOURCE_CHUNK,n_sources)
        source_gpu=torch.tensor(np.asarray(sources[left:right]),device=device,dtype=torch.bfloat16)
        score=ref_gpu @ source_gpu.T
        if score.dtype is not torch.bfloat16:
            raise TypeError("BF16 score output was not produced")
        bits=score.contiguous().cpu().view(torch.uint16).numpy()
        matrix[:,left:right]=bits
        done=right
        if left==0 or (left//SOURCE_CHUNK+1)%4==0 or right==n_sources:
            if device.type=="cuda": torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            logger.info("stage=score-matrix-progress source_windows=%d/%d device=%s eta=%s",
                        done,n_sources,device,format_duration(elapsed*(n_sources-done)/max(done,1)))
        del source_gpu,score,bits
    matrix.flush(); del matrix,ref_gpu
    with partial.open("r+b") as stream: os.fsync(stream.fileno())
    partial.replace(path)
    logger.info("stage=score-matrix-complete bytes=%d output=%s",expected,path)
    return expected

def rank_candidates(matrix,k,logger):
    n_refs,n_sources=matrix.shape
    candidate_indices=np.empty((n_refs,k),dtype=np.int32)
    started=time.monotonic()
    for ri in range(n_refs):
        bits=np.array(matrix[ri],dtype=np.uint16,copy=True)
        scores=torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(torch.float32).numpy()
        if not np.isfinite(scores).all(): raise ValueError("Non-finite BF16 score in matrix")
        threshold=np.partition(scores,n_sources-k)[n_sources-k]
        above=np.flatnonzero(scores>threshold)
        needed=k-len(above)
        tied=np.flatnonzero(scores==threshold)[:needed]
        candidates=np.concatenate((above,tied))
        order=np.lexsort((candidates,-scores[candidates]))
        candidate_indices[ri]=candidates[order].astype(np.int32,copy=False)
        if ri==0 or (ri+1)%32==0 or ri+1==n_refs:
            elapsed=time.monotonic()-started
            logger.info("stage=window-rank refs=%d/%d k=%d eta=%s",ri+1,n_refs,k,
                        format_duration(elapsed*(n_refs-ri-1)/(ri+1)))
    return candidate_indices

def select(candidates):
    selected=np.zeros(2666312,dtype=bool); cursors=np.zeros(len(candidates),dtype=np.int64)
    out=[]; progressed=True
    while len(out)<BUDGET and progressed:
        progressed=False
        for r,row in enumerate(candidates):
            while cursors[r]<len(row) and selected[row[cursors[r]]]: cursors[r]+=1
            if cursors[r]>=len(row): continue
            idx=int(row[cursors[r]]); cursors[r]+=1
            selected[idx]=True; out.append(idx); progressed=True
            if len(out)==BUDGET: break
    return np.asarray(out,dtype=np.int64)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--family",choices=METHODS,required=True);p.add_argument("--task",choices=TASKS,required=True)
    a=p.parse_args();method=METHODS[a.family];started=time.monotonic();batch="timerXL-%s-01-win-rr"%a.family
    logger=configure_logger("v1_crb_redo_rr_win","v1-crb-redo",batch+"-"+a.task,"selection")
    srcp=build_source_windows(method);refp=build_reference_windows(a.task,method)
    src=np.load(srcp,mmap_mode="r");ref=np.load(refp,mmap_mode="r")
    out=output_path("selection",ROOT/"outputs/selection/v1-crb-redo"/batch/a.task)
    out.mkdir(parents=True,exist_ok=True)
    if (out/".complete").is_file(): logger.info("stage=skip reason=complete");return
    n_refs,n_sources=len(ref),len(src)
    matrix_path=out/"score_matrix.bf16.bin"
    matrix_bytes=build_score_matrix(ref,src,matrix_path,logger)
    score_matrix=np.memmap(matrix_path,mode="r",dtype=np.uint16,shape=(n_refs,n_sources))
    k=min(n_sources,max(512,int(np.ceil(BUDGET/n_refs))+512))
    while True:
        logger.info("stage=window-rr-start refs=%d sources=%d candidates_per_ref=%d score_dtype=bfloat16",n_refs,n_sources,k)
        ci=rank_candidates(score_matrix,k,logger);chosen=select(ci)
        if len(chosen)==BUDGET: break
        if k>=n_sources: raise RuntimeError("Window RR exhausted candidate ranks")
        k=min(n_sources,k*2)
        logger.info("stage=window-rr-expand next_candidates_per_ref=%d reuse=score_matrix",k)
    if len(np.unique(chosen))!=BUDGET or chosen.min()<0 or chosen.max()>=n_sources:
        raise ValueError("RR window budget, uniqueness, or range check failed")
    tmp=out/"window_indices.partial.npy";np.save(tmp,chosen,allow_pickle=False);tmp.replace(out/"window_indices.npy")
    selection_bytes=(out/"window_indices.npy").stat().st_size
    manifest=dict(batch=batch,task=a.task,granularity="win",family=a.family,
        algorithm="per_reference_window_greedy_round_robin",window_budget=BUDGET,
        reference_windows=n_refs,source_windows=n_sources,candidates_per_ref=k,
        score_dtype="bfloat16",score_matrix_logical_shape=[n_refs,n_sources],
        score_matrix_storage="row_major_ref_by_source_raw_bfloat16",
        score_matrix_bytes=matrix_bytes,score_matrix_removed_after_selection=True,
        selected_windows=len(chosen),selection_file_bytes=selection_bytes,
        selection_elapsed_seconds=time.monotonic()-started,
        tie_break="descending_bfloat16_score_then_source_window_index")
    write_json(out/"manifest.json",manifest)
    del score_matrix
    matrix_path.unlink()
    (out/".complete").write_text("complete\n",encoding="ascii")
    logger.info("stage=complete windows=%d score_matrix_bytes=%d score_matrix_removed=true output=%s",
                len(chosen),matrix_bytes,out)
if __name__=="__main__":main()
