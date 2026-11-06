from pathlib import Path
import json,time
from dataclasses import replace
import numpy as np,torch
from rage.experiments.full_pool_models import build_model
from rage.experiments.development_inputs import load_fragments
from rage.execution import prepare_contexts,selection_execution,SelectionExecution
from rage.generation import generate_tokens
from rage.experiments.students import atomic_json
config=json.loads(Path('configs/rage/main.json').read_text())
engine,probe,identity=build_model('chronos-base',model_path=config['model_paths']['chronos-base'])
probe.execution=SelectionExecution(gradient_batch=8,gradient_workers=1,inference_batch=32,cpu_threads=4,generation='eager')
pool=load_fragments(Path(config['inputs'])/'pool.npz')
windows=np.stack([pool[i].extract_window(1,512,64) for i in range(32)])
prepared=prepare_contexts(probe,windows)
result=dict(student='chronos-base',dtype='BF16',rows=32,horizon=64,fixture='first 32 actual source trajectories first legal start',additional_bp=0)
with selection_execution(probe):
    times={};tokens={}
    for mode in ('eager','cuda_graph'):
        probe.execution=replace(probe.execution,generation=mode)
        generate_tokens(probe,prepared.input_ids,prepared.attention_mask)
        torch.cuda.synchronize();started=time.perf_counter()
        for _ in range(3):tokens[mode]=generate_tokens(probe,prepared.input_ids,prepared.attention_mask)[0]
        torch.cuda.synchronize();times[mode]=(time.perf_counter()-started)/3
    result.update(seconds=times,tokens_exactly_equal=bool(torch.equal(tokens['eager'],tokens['cuda_graph'])),
        differing_tokens=int((tokens['eager']!=tokens['cuda_graph']).sum()),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
atomic_json(Path('temp/v5-launch/graph-check.json'),result)
print(json.dumps(result,indent=2),flush=True)
