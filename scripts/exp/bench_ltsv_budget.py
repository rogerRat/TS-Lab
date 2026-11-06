"""Disclosed LTSV budget tradeoff: original stride288/Ref64 vs stride576/Ref16.

Scores use matched block updates; source-prefix pilot, never exam data.
All timing rows are retained for later original/budget cost accounting.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np
import torch

from baselines.common import reference_windows
from baselines.ltsv.fast import BlockScorer, block_rng, block_starts, reference_mse, run_fragments
from baselines.ltsv.algorithms import ltsv_window_scores
from baselines.ltsv.runner import _ltsv_training_loss
from rage.execution import SelectionExecution, prepare_contexts, selection_execution
from rage.experiments.development_inputs import load_fragments
from rage.experiments.full_pool_models import build_model
from rage.experiments.students import atomic_json
from scripts.exp.bench_ltsv_execution import timed, compare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--student',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path)
    p.add_argument('--fragments',type=int,default=8)
    p.add_argument('--blocks-per-fragment',type=int,default=4)
    p.add_argument('--tasks',type=int,default=3)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--reference-windows',type=int,default=16)
    args=p.parse_args()
    if args.output.exists() or args.run.resolve() in args.output.resolve().parents:
        raise ValueError('Choose new independent diagnostic report')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.run/'protocol.json').read_text())
    old=json.loads(args.baseline_report.read_text()) if args.baseline_report else None
    pool=load_fragments(Path(config['inputs'])/'pool.npz')
    c,h=config['context_length'],config['prediction_length']
    width,stride=c+h,(c+h)//2
    length=width+(args.blocks_per_fragment-1)*stride
    eligible=[f for f in pool if f.length>=length]
    chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,args.fragments,dtype=int)]
    datasets=[config['datasets'][i] for i in np.linspace(0,len(config['datasets'])-1,args.tasks,dtype=int)]
    torch.set_num_threads(4)
    _,probe,identity=build_model(args.student,model_path=config['model_paths'][args.student])
    if old and (old['model']!=identity or old['source_fragments']!=[int(f.fragment_id) for f in chosen]
                or old['source_prefix_length']!=length):
        raise ValueError('Baseline pilot identities differ')
    original=dict(config['selection_execution'])
    original.update(config.get('student_selection_execution',{}).get(args.student,{}))
    original['gradient_workers']=1
    probe.execution=SelectionExecution(**original)
    native=callable(getattr(probe,'forecast',None))
    generation='eager'
    inference_batch = min(args.reference_windows,32) if native else args.reference_windows
    vals,ids,raw={}, {}, {}
    for dataset in datasets:
        refs=load_fragments(Path(config['inputs'])/f'{dataset}__reference.npz')
        val,refids=reference_windows(refs,context_length=c,prediction_length=h,stride=16,max_windows=64,seed=0)
        subset=np.linspace(0,len(val)-1,min(args.reference_windows,len(val)),dtype=int)
        raw[dataset]=val[subset]
        ids[dataset]=dict(original=refids,budget=[refids[i] for i in subset])
        for mode,v in [('baseline',val),('budget',val[subset])]:
            vals[mode,dataset]=v if native else prepare_contexts(probe,v)
    scorer=BlockScorer(probe,raw[datasets[0]],inference_batch=inference_batch,generation=generation)
    cpu=scorer.base.cpu().clone()
    views=list(cpu.split([p.numel() for p in scorer.parameters]))
    def restore_cpu():
        with torch.no_grad():
            for p,v in zip(scorer.parameters,views): p.copy_(v.view_as(p))
    def forecast(mode,dataset):
        probe.execution=replace(probe.execution,
            inference_batch=original.get('inference_batch',0) if mode=='baseline' else inference_batch,
            generation='eager' if mode=='baseline' else generation)
        return reference_mse(probe,vals[mode,dataset],32 if mode=='baseline' else inference_batch)
    rows,bases=[],{}
    torch.cuda.reset_peak_memory_stats()
    with selection_execution(probe):
        for mode in ('baseline','budget'):
            for dataset in datasets:
                bases[mode,dataset]=timed(lambda:forecast(mode,dataset))[0]
        ordinal=0
        for fragment in chosen:
            for begin in block_starts(length,width,stride):
                scorer.restore()
                window=np.asarray(fragment.data[begin:begin+width],dtype=np.float32)[None]
                with block_rng(probe.device,20260916,ordinal):
                    def update():
                        optimizer=torch.optim.Adam(probe.model.parameters(),lr=1e-5,foreach=False,
                            betas=(.9,.999),eps=1e-8,weight_decay=.1)
                        probe.model.train(); optimizer.zero_grad(set_to_none=True)
                        _ltsv_training_loss(probe,window).backward()
                        torch.nn.utils.clip_grad_norm_(probe.model.parameters(),1.)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); probe.model.eval()
                        return optimizer
                    optimizer,update_s=timed(update)
                    current=[]
                    try:
                        for dataset in datasets:
                            modes=['baseline','budget'] if ordinal%2 else ['budget','baseline']
                            for mode in modes:
                                if mode=='baseline' and old and begin!=0:
                                    source=next(r for r in old['rows'] if r['mode']=='baseline' and r['ordinal']==ordinal and r['dataset']==dataset)
                                    row=dict(source, timing_reused=True)
                                else:
                                    mse,forecast_s=timed(lambda:forecast(mode,dataset))
                                    row=dict(mode=mode,dataset=dataset,ordinal=ordinal,
                                        fragment_id=int(fragment.fragment_id),start=int(begin),
                                        value=bases[mode,dataset]-mse,mse=mse,forecast_seconds=forecast_s,
                                        update_seconds=update_s,timing_reused=False)
                                current.append(row)
                    finally:
                        _,cpu_s=timed(restore_cpu)
                        _,gpu_s=timed(scorer.restore)
                        del optimizer
                    for row in current:
                        if not row.get('timing_reused'):
                            row.update(cpu_restore_seconds=cpu_s,gpu_restore_seconds=gpu_s)
                    rows.extend(current)
                ordinal+=1
            print(f'{args.student}: budget comparison {ordinal} blocks, {len(datasets)} Ref tasks',flush=True)
            atomic_json(args.output.with_suffix('.progress.json'),dict(rows=rows))
    peak=torch.cuda.max_memory_allocated()/2**30
    scorer.close()
    tasks=[]
    reduced_starts=set(block_starts(length,width,width).tolist())
    for dataset in datasets:
        scores={}
        for mode in ('baseline','budget'):
            result=[]
            for f in chosen:
                sums,counts=np.zeros(length+1),np.zeros(length+1,dtype=np.int64)
                for row in rows:
                    if row['dataset']!=dataset or row['mode']!=mode or row['fragment_id']!=f.fragment_id: continue
                    begin=row['start']
                    if mode=='budget' and begin not in reduced_starts: continue
                    sums[begin:begin+width]+=row['value']; counts[begin:begin+width]+=1
                result.extend(ltsv_window_scores(sums,counts,np.arange(length-width+1),width).astype(np.float32))
            scores[mode]=result
        matched=[r for r in rows if r['dataset']==dataset and not r.get('timing_reused')]
        base=[r for r in matched if r['mode']=='baseline']
        budget=[r for r in matched if r['mode']=='budget']
        original_s=float(np.mean([r['update_seconds']+r['forecast_seconds']+r['cpu_restore_seconds'] for r in base]))
        budget_s=float(np.mean([r['update_seconds']+r['forecast_seconds']+r['gpu_restore_seconds'] for r in budget]))
        tasks.append(dict(dataset=dataset,**compare(scores['baseline'],scores['budget']),
            baseline_seconds_per_block=original_s,budget_seconds_per_block=budget_s,
            original_timing_blocks=len(base),budget_timing_blocks=len(budget)))
    original_k=sum(len(block_starts(f.length,width,stride)) for f in pool)
    budget_k=sum(len(block_starts(f.length,width,width)) for f in pool)
    # Measure real spawn/MPS execution at 1 and requested concurrency, including
    # initialization, then separately report summed shard compute / wall time.
    concurrency=[]
    jobs=[(int(f.fragment_id),f.data[:length],i*args.blocks_per_fragment,None) for i,f in enumerate(chosen)]
    options=dict(lr=1e-5,inference_batch=inference_batch,generation=generation,seed=20260916)
    outputs={}
    for workers in sorted(set((1,2,args.workers))):
        start=time.perf_counter()
        result=list(run_fragments(probe,identity,raw[datasets[0]],jobs,width=width,stride=width,
            workers=workers,options=options))
        seconds=time.perf_counter()-start
        outputs[workers]={r['fragment_id']:r['values'] for r in result}
        concurrency.append(dict(workers=workers,wall_seconds=seconds,
            compute_seconds_sum=sum(r['compute_seconds'] for r in result),
            active_span_seconds=max(r['work_finished'] for r in result)-min(r['work_started'] for r in result),
            blocks=sum(r['computed_blocks'] for r in result),
            max_worker_peak_gpu_gib=max(r['peak_gpu_gib'] for r in result)))
        print(f'{args.student}: workers={workers} {seconds:.2f}s',flush=True)
    exact=all(np.array_equal(outputs[1][key],other[key]) for count,other in outputs.items() for key in outputs[1])
    baseline_s=float(np.mean([t['baseline_seconds_per_block'] for t in tasks]))
    budget_s=float(np.mean([t['budget_seconds_per_block'] for t in tasks]))
    report=dict(schema='ltsv-budget-pilot-v1',model=identity,student=args.student,
        source_fragments=[int(f.fragment_id) for f in chosen],source_prefix_length=length,
        reference_ids=ids,rows=rows,tasks=tasks,peak_gpu_gib=peak,
        original_blocks_per_task=original_k,budget_blocks_per_task=budget_k,
        baseline_seconds_per_block=baseline_s,budget_seconds_per_block=budget_s,
        original_serial_seconds_estimate=original_k*baseline_s,
        budget_serial_seconds_estimate=budget_k*budget_s,
        estimated_serial_speedup=original_k*baseline_s/(budget_k*budget_s),
        concurrency=concurrency,exact_concurrency_scores=exact,
        execution=dict(inference_batch=inference_batch,generation=generation,context_windows=args.reference_windows,
            block_stride=width,block_seed=20260916),
        limitation='Small source prefixes and 3 safe Ref tasks under live MPS. No downstream accuracy measurement. Parallel setup included; do not linearly extrapolate worker count.')
    atomic_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('student','tasks','estimated_serial_speedup','concurrency','exact_concurrency_scores')}),flush=True)

if __name__=='__main__': main()
