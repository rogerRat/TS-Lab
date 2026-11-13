"""Prepare single-task FEV/TIME evaluation configs and exact TIME manifests."""
from __future__ import annotations
import json
import copy
from pathlib import Path
import yaml
from configs.paths import PROJECT_ROOT
from evaluate.schemas import EvaluationRunConfig
ROOT=PROJECT_ROOT
TASKS={"fev-09":9,"time-01":"CPHL/15T","time-02":"CPHL/30T","time-03":"CPHL/H",
"time-04":"ECDC_COVID/D","time-06":"epf_electricity_price/H","time-07":"Commodity_Import/M","time-08":"Commodity_Production/M"}
def main():
    fevbase=yaml.safe_load((ROOT/"configs/eval/fev/v1-crb-redo-timerxl.yaml").read_text())
    timebase=yaml.safe_load((ROOT/"configs/eval/time/v1-rage-trev-timerxl.yaml").read_text())
    for task,value in TASKS.items():
        if task.startswith("fev"):
            c=dict(fevbase); c["bench_config"]=dict(fevbase["bench_config"],expected_tasks=1,task_indices=[value])
            dest=ROOT/"configs/eval/fev"/("v1-crb-redo-"+task+".yaml")
            dest.write_text(yaml.safe_dump(c,sort_keys=False),encoding="ascii")
        else:
            c=copy.deepcopy(timebase)
            c["bench_config"]["tasks"]={"mode":"explicit","items":[{"dataset":value,"terms":["short"]}]}
            data=c["bench_config"]["dataset"]; original=ROOT/data["root"]
            manifest=json.loads((original.parent/"data_manifest.json").read_text())
            wanted={"dataset":value,"term":"short"}
            if wanted not in manifest["verified_tasks"]: raise ValueError("TIME task not verified")
            subset=ROOT/"data/bench/time/v1-crb-redo"/task
            subset.mkdir(parents=True,exist_ok=True)
            newroot=subset/"dataset"; newroot.mkdir(exist_ok=True)
            src=original/value; link=newroot/value; link.parent.mkdir(parents=True,exist_ok=True)
            if link.exists() or link.is_symlink():
                if link.resolve()!=src.resolve(): raise ValueError("TIME link conflict")
            else: link.symlink_to(src.resolve(),target_is_directory=True)
            m=dict(manifest); m["verified_tasks"]=[wanted]
            (subset/"data_manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n",encoding="ascii")
            data["root"]=str(newroot.relative_to(ROOT))
            dest=ROOT/"configs/eval/time"/("v1-crb-redo-"+task+".yaml")
            dest.write_text(yaml.safe_dump(c,sort_keys=False),encoding="ascii")
        parsed=EvaluationRunConfig.from_file(dest)
        print("EVAL_CONFIG_READY",task,dest,parsed.benchmark,flush=True)
if __name__=="__main__": main()
