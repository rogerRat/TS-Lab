"""Train and evaluate one TimerXL RR selection cell."""
from __future__ import annotations
import argparse,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import GenerationConfig
from peft import PeftModel
from configs.logger import configure_logger,format_duration
from configs.paths import PROJECT_ROOT,output_path
from rage.experiments.full_pool_models import build_model
from rage.experiments.lora import apply_student_lora
from rage.experiments.train_eval import TrainingRecipe,fine_tune
ROOT=PROJECT_ROOT
TASKS=("fev-09","time-01","time-02","time-03","time-04","time-06","time-07","time-08")
FAMILIES=("base-uni","ours-rage101")
GRANULARITIES=("traj","block","win")
MODEL_PATH=Path("/root/autodl-tmp/pretrained_models/timer-base-84m-84m")
def write_json(path,value):
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="ascii")
    tmp.replace(path)
def materialize(indices,path):
    config=__import__("yaml").safe_load((ROOT/"configs/data/src-96.yaml").read_text())
    source=np.load(ROOT/config["values_path"],mmap_mode="r")
    lengths=np.asarray([int(x["length"]) for x in config["trajectories"]],dtype=np.int64)
    legal=lengths-575; offsets=np.cumsum(np.r_[0,legal[:-1]])
    if int(legal.sum())!=2666312: raise ValueError("Source window pool changed")
    rows=np.asarray(indices,dtype=np.int64)
    if len(rows)!=266631 or len(np.unique(rows))!=len(rows) or rows.min()<0 or rows.max()>=legal.sum():
        raise ValueError("RR selected-window manifest is invalid")
    out=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(rows),576))
    owners=np.searchsorted(np.cumsum(legal),rows,side="right")
    for fid in np.unique(owners):
        pos=np.flatnonzero(owners==fid); starts=rows[pos]-offsets[fid]
        for p,start in zip(pos,starts):
            begin=int(offsets[fid]+start)
            out[p]=source[begin:begin+576]
    out.flush()
    return out
def export_model(adapter_dir,destination,task,method,granularity):
    base,_,identity=build_model("timerxl-84m",model_path=str(MODEL_PATH),device="cpu",
                                context_length=480,prediction_length=96)
    wrapped=PeftModel.from_pretrained(base.model,adapter_dir,is_trainable=False)
    merged=wrapped.merge_and_unload()
    destination.mkdir(parents=True,exist_ok=False)
    merged.save_pretrained(destination,safe_serialization=True,max_shard_size="5GB")
    GenerationConfig.from_pretrained(MODEL_PATH,local_files_only=True).save_pretrained(destination)
    record=dict(schema_version=1,experiment_version="v1-crb-redo/"+method+"/"+granularity+"/"+task,
        model_key="timerxl-84m",model="thuml/timer-base-84m",training_step=4000,precision="bfloat16")
    if task.startswith("fev"): record["reference_task_record"]=dict(benchmark="fev",task_indices=[9])
    write_json(destination/"rage_checkpoint_manifest.json",record)
    del base,wrapped,merged
def evaluate(checkpoint,task,batch,logger):
    folder="fev" if task.startswith("fev") else "time"
    config=ROOT/"configs/eval"/folder/("v1-crb-redo-"+task+".yaml")
    out=output_path("eval",ROOT/"outputs/eval/v1-crb-redo"/batch/task)
    cmd=[sys.executable,"-u","-m","evaluate.run","--checkpoint",str(checkpoint),
         "--config",str(config),"--output-dir",str(out)]
    env=dict(os.environ);env["PYTHONPATH"]="src:third_parties/fev/src:third_parties/TIME/src:."
    logger.info("stage=evaluation-start task=%s config=%s",task,config)
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    primary=out/("fev_results.csv" if folder=="fev" else "time_metadata.json")
    if not primary.is_file(): raise RuntimeError("Evaluation primary output missing")
    if folder=="time":
        meta=json.loads(primary.read_text())
        if not meta.get("completed_at"): raise RuntimeError("TIME evaluation incomplete")
    else:
        with primary.open(newline="") as f:
            if len(list(csv.DictReader(f)))!=1: raise RuntimeError("FEV task result count differs")
        if not (out/"fev_metadata.json").is_file(): raise RuntimeError("FEV metadata missing")
    (out/".complete").write_text("complete\n",encoding="ascii")
    return out
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--family",choices=FAMILIES,required=True)
    p.add_argument("--granularity",choices=GRANULARITIES,required=True)
    p.add_argument("--task",choices=TASKS,required=True)
    a=p.parse_args();batch="timerXL-%s-01-%s-rr"%(a.family,a.granularity)
    logger=configure_logger("v1_crb_redo_rr_train_eval","v1-crb-redo",batch+"-"+a.task,"train-eval")
    started=time.monotonic()
    selection=ROOT/"outputs/selection/v1-crb-redo"/batch/a.task
    if not (selection/".complete").is_file(): raise FileNotFoundError("RR selection incomplete")
    phase=output_path("train",ROOT/"outputs/train/v1-crb-redo"/batch/a.task)
    if (phase/".complete").is_file(): logger.info("stage=skip reason=complete");return
    phase.mkdir(parents=True,exist_ok=True)
    indices=np.load(selection/"window_indices.npy",allow_pickle=False)
    pool_path=phase/"selected_windows.npy"
    if not pool_path.exists(): materialize(indices,pool_path)
    windows=np.load(pool_path,mmap_mode="r")
    engine,_,identity=build_model("timerxl-84m",model_path=str(MODEL_PATH),device="cuda",
                                  context_length=480,prediction_length=96)
    lora=apply_student_lora(engine,"timerxl-84m",dict(r=8,alpha=16,dropout=0.,
        target_modules=["q_proj","v_proj"]))
    recipe=TrainingRecipe(learning_rate=1e-4,epochs=1.,batch_size=32,micro_batch_size=8,
        weight_decay=0.,betas=(.9,.999),eps=1e-8,max_grad_norm=1.,warmup_ratio=.03,
        schedule="cosine",min_lr_ratio=0.,schedule_epochs=4000.)
    resume=phase/"resume.pt";adapter=phase/"adapter-step-4000";export=phase/"export-step-4000"
    training_manifest=phase/"training_manifest.json"
    if not training_manifest.exists():
        def on_step(step):
            if step==4000 or step%100==0:
                logger.info("stage=train-progress task=%s step=%d/4000",a.task,step)
        result=fine_tune(engine,windows,steps=4000,batch_size=32,learning_rate=1e-4,
            seed=0,recipe=recipe,schedule_total_steps=4000,resume_path=resume,
            checkpoint_steps=list(range(500,4001,500)),on_step=on_step)
        if result["steps"]!=4000: raise RuntimeError("Training stopped before 4000 updates")
        engine.model.save_pretrained(adapter,safe_serialization=True)
        write_json(training_manifest,dict(batch=batch,task=a.task,identity=identity,lora=lora,
            recipe=recipe.as_dict(),result=result,selection=str(selection/"manifest.json")))
    del engine,windows
    torch.cuda.empty_cache()
    if not export.exists(): export_model(adapter,export,a.task,a.family,a.granularity)
    eval_out=evaluate(export,a.task,batch,logger)
    if not (eval_out/".complete").is_file(): raise RuntimeError("Evaluation incomplete")
    for path in (resume,adapter,export,pool_path):
        if path.is_dir(): shutil.rmtree(path)
        elif path.exists(): path.unlink()
    write_json(phase/"completion_manifest.json",dict(batch=batch,task=a.task,
        selection=str(selection/"manifest.json"),evaluation=str(eval_out),checkpoints_deleted=True))
    (phase/".complete").write_text("complete\n",encoding="ascii")
    logger.info("stage=complete task=%s elapsed=%s output=%s",a.task,
                format_duration(time.monotonic()-started),phase)
if __name__=="__main__": main()
