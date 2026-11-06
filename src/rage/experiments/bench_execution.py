"""Small synthetic execution benchmarks. Never load formal data or weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import T5Config, T5ForConditionalGeneration

from rage.generation import GreedyGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--suite", choices=["greedy", "projection", "cpu", "baselines", "rage"], default="greedy")
    args = parser.parse_args()
    if args.suite != "greedy":
        report = globals()["bench_" + args.suite](args)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    torch.manual_seed(42)
    torch.set_num_threads(4)
    config = T5Config(vocab_size=4096, d_model=512 if args.full_size else 64,
                      d_ff=2048 if args.full_size else 128, d_kv=64 if args.full_size else 16,
                      num_layers=6 if args.full_size else 1,
                      num_heads=8 if args.full_size else 2,
                      decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
                      dropout_rate=0.0)
    model = T5ForConditionalGeneration(config).cuda().eval()
    ids = torch.randint(2,4096,(32,513),device="cuda")
    mask = torch.ones_like(ids)
    mask[0,:80] = 0
    kwargs = dict(input_ids=ids, attention_mask=mask, min_new_tokens=64, max_new_tokens=64,
                  do_sample=False, num_beams=1, num_return_sequences=1)
    def timed(function, repeats=3):
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(repeats):
            result=function()
        torch.cuda.synchronize()
        return result, (time.perf_counter()-start)/repeats
    with torch.no_grad():
        expected, eager_seconds=timed(lambda:model.generate(**kwargs)[:,1:])
        graph, cold_seconds=timed(lambda:GreedyGraph(model,ids,mask,64),repeats=1)
        actual, graph_seconds=timed(lambda:graph(ids,mask))
        mismatches=int((expected!=actual).sum())
        # A new set of histories and in-place model updates must be observed.
        ids.random_(2,4096)
        model.shared.weight.mul_(.9)
        updated_expected=model.generate(**kwargs)[:,1:]
        updated_actual=graph(ids,mask)
        updated_mismatches=int((updated_expected!=updated_actual).sum())
    report=dict(gpu=torch.cuda.get_device_name(),full_size=args.full_size,
                parameters=sum(p.numel() for p in model.parameters()),
                eager_seconds=eager_seconds,graph_seconds=graph_seconds,
                capture_seconds=cold_seconds,speedup=eager_seconds/graph_seconds,
                token_mismatches=mismatches,updated_token_mismatches=updated_mismatches)
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert mismatches==updated_mismatches==0


def bench_projection(args):
    from gradient.count_sketch import add_count_sketch
    torch.manual_seed(42)
    report = {"gpu": torch.cuda.get_device_name(), "measurements": []}
    for rows, columns in [(1, 262144), (16, 262144), (16, 2097152), (64, 262144)]:
        values = torch.randn(rows, columns, device="cuda")
        hashes = torch.randint(0, 4096, (columns,), device="cuda")
        signs = torch.randint(0, 2, (columns,), device="cuda").float() * 2 - 1
        results, times = [], []
        for fused in (False, True):
            out = torch.zeros(rows, 4096, device="cuda")
            for _ in range(3):
                out.zero_()
                add_count_sketch(out, values, hashes, signs, fused=fused)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                out.zero_()
                add_count_sketch(out, values, hashes, signs, fused=fused)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)/20)
            results.append(out.clone())
        torch.testing.assert_close(results[0], results[1], rtol=1e-4, atol=2e-4)
        report["measurements"].append(dict(rows=rows, columns=columns,
            torch_seconds=times[0], fused_seconds=times[1], speedup=times[0]/times[1]))
    return report


def legacy_module(path, name):
    """Read-only oracle from the last delivered revision, for this benchmark."""
    import subprocess
    import sys
    import types
    source = subprocess.check_output(["git", "show", "cde56ec:"+path], text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, "cde56ec:"+path, "exec"), module.__dict__)
    return module


def bench_cpu(args):
    from dataclasses import replace
    import numpy as np
    from baselines.common import reference_windows
    from rage.core import TimeSeriesFragment
    from rage import crb
    old = legacy_module("src/rage/crb.py", "rage._benchmark_old_crb")
    old_common = legacy_module("baselines/common.py", "baselines._benchmark_old_common")
    fragments = [TimeSeriesFragment(np.arange(10000, dtype=np.float32) + i/10, fragment_id=i)
                 for i in range(12)]
    def field(fragment, starts):
        x = np.stack([fragment.extract_window(int(s), 8, 4)[:8] for s in starts])
        return np.sin(x/20).astype(np.float32)
    report = {"reference_revision": "cde56ec", "crb": []}
    base = crb.CRBConfig(context_length=8, prediction_length=4, cover_density=.03,
                         error_density=.04, split_density=.01, mean_epsilon=.01,
                         endpoint_epsilon=.01)
    for initial, error, constant in [("longest", "risk", False), ("random", "risk", False),
                                      ("random", "risk", True), ("longest", "uniform", False)]:
        config = replace(base, initial_placement=initial, error_placement=error)
        computer = (lambda f,s:np.ones((len(s),8),dtype=np.float32)) if constant else field
        outputs, times = [], []
        for module in (old, crb):
            start = time.perf_counter()
            outputs.append(module.CRBEngineMulti(config, computer).process_fragments(fragments))
            times.append(time.perf_counter()-start)
        for a,b in zip(*outputs):
            assert sorted(a.anchors)==sorted(b.anchors)
            assert (a.cover_computations,a.error_computations,a.split_computations,
                    a.discarded_no_signal,a.discarded_antagonistic,a.banned_budget)==(
                    b.cover_computations,b.error_computations,b.split_computations,
                    b.discarded_no_signal,b.discarded_antagonistic,b.banned_budget)
            assert len(a.blocks)==len(b.blocks)
            for aa,bb in zip(a.blocks,b.blocks):
                np.testing.assert_array_equal(aa.start_positions,bb.start_positions)
                np.testing.assert_array_equal(aa.direction,bb.direction)
        report["crb"].append(dict(initial=initial,error=error,constant=constant,
                                  old_seconds=times[0],new_seconds=times[1],speedup=times[0]/times[1],
                                  identical_anchors_blocks_counters=True))
    reference_times, reference_outputs = [], []
    for function in (old_common.reference_windows, reference_windows):
        start = time.perf_counter()
        reference_outputs.append(function(fragments,context_length=512,prediction_length=64,
                                           stride=1,max_windows=64,seed=7))
        reference_times.append(time.perf_counter()-start)
    assert reference_outputs[0][1]==reference_outputs[1][1]
    np.testing.assert_array_equal(reference_outputs[0][0],reference_outputs[1][0])
    report["reference_windows"] = dict(old_seconds=reference_times[0],new_seconds=reference_times[1],
                                        speedup=reference_times[0]/reference_times[1],identical=True)
    return report


def bench_baselines(args):
    import importlib
    import numpy as np
    from chronos import ChronosConfig
    from rage.core import TimeSeriesFragment
    from rage.engines import ChronosT5Config, ChronosT5Engine, _FullModelProjector
    from rage.execution import SelectionExecution, selection_execution
    from rage.experiments.arms import WindowIndex
    torch.manual_seed(42)
    torch.set_num_threads(4)
    old_engine_module = legacy_module("src/rage/engines.py", "rage._benchmark_old_engines")
    old_common = legacy_module("baselines/common.py", "baselines._benchmark_old_common")
    old_arms = legacy_module("src/rage/experiments/arms.py", "rage.experiments._benchmark_old_arms")
    config = T5Config(vocab_size=4096, d_model=512 if args.full_size else 64,
                      d_ff=2048 if args.full_size else 128, d_kv=64 if args.full_size else 16,
                      num_layers=6 if args.full_size else 1,
                      num_heads=8 if args.full_size else 2,
                      decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
                      dropout_rate=.1, initializer_factor=.05)
    model = T5ForConditionalGeneration(config).cuda().eval()
    tokenizer = ChronosConfig(tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15., "high_limit": 15.}, n_tokens=4096,
        n_special_tokens=2, pad_token_id=0, eos_token_id=1, use_eos_token=True,
        model_type="seq2seq", context_length=512, prediction_length=64,
        num_samples=1, temperature=1., top_k=50, top_p=1.).create_tokenizer()
    engines=[]
    for module in (old_engine_module, importlib.import_module("rage.engines")):
        engine=module.ChronosT5Engine.__new__(module.ChronosT5Engine)
        engine.model=model
        engine.tokenizer=tokenizer
        engine.config=module.ChronosT5Config(projection_dimension=4096)
        engine.device=torch.device("cuda")
        engine.parameter_count=sum(p.numel() for p in model.parameters())
        engine._named_trainable=list(model.named_parameters())
        engine._parameters=[p for _,p in engine._named_trainable]
        engine.projector=module._FullModelProjector(engine._named_trainable,4096,42,engine.device)
        engine.execution=SelectionExecution()
        engines.append(engine)
    pool=[TimeSeriesFragment(np.sin(np.arange(1152)/13+i).astype(np.float32),fragment_id=i)
          for i in range(2)]
    reference=[TimeSeriesFragment(np.sin(np.arange(900)/11+i).astype(np.float32),fragment_id=10+i)
               for i in range(2)]
    indexes=[old_arms.WindowIndex(pool,context_length=512,prediction_length=64),
             WindowIndex(pool,context_length=512,prediction_length=64)]
    report={"reference_revision":"cde56ec", "gpu":torch.cuda.get_device_name(),
            "parameters":engines[0].parameter_count,"pool_windows":len(indexes[0]),"arms":{}}
    # LESS/TSDS changed algorithmic sample units in decisions 11/12. Comparing
    # them to the old window-level selectors is no longer an execution parity test.
    report["excluded_redesigned_arms"] = ["less", "tsds"]
    for name in ("lp_mirror","ltsv"):
        old=legacy_module(f"baselines/{name}/runner.py", f"baselines.{name}._benchmark_old_runner")
        old.reference_windows=old_common.reference_windows
        current=importlib.import_module(f"baselines.{name}.runner")
        outputs,times,steady_times=[],[],[]
        for i,module in enumerate((old,current)):
            function=getattr(module,name+"_select")
            torch.manual_seed(7)
            torch.cuda.synchronize()
            start=time.perf_counter()
            with selection_execution(engines[i]):
                result=function(engines[i],reference_fragments=reference,index=indexes[i],
                                context_length=512,prediction_length=64,n_select=8,seed=7)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            outputs.append(result)
            print(f"{name} {'old' if i==0 else 'new'}: {times[-1]:.3f}s",flush=True)
            # Recompute the complete arm, not a selected-set cache hit. This
            # separates process/kernel/graph initialization from throughput.
            repeats = []
            for _ in range(2):
                torch.manual_seed(7)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with selection_execution(engines[i]):
                    repeated = function(engines[i], reference_fragments=reference,index=indexes[i],
                        context_length=512,prediction_length=64,n_select=8,seed=7)
                torch.cuda.synchronize()
                repeats.append(time.perf_counter()-start)
                assert repeated.window_ids == result.window_ids
            steady_times.append(float(np.median(repeats)))
        assert outputs[0].window_ids==outputs[1].window_ids, name
        np.testing.assert_array_equal(outputs[0].selected,outputs[1].selected)
        for counter in ("gradient_computations","feature_computations","proxy_training_steps"):
            assert getattr(outputs[0].cost,counter)==getattr(outputs[1].cost,counter),(name,counter)
        report["arms"][name]=dict(old_seconds=times[0],new_seconds=times[1],speedup=times[0]/times[1],
                                  steady_old_seconds=steady_times[0],steady_new_seconds=steady_times[1],
                                  steady_speedup=steady_times[0]/steady_times[1],
                                  identical_selected_ids_windows_work_counts=True)
    # Compare actual projected model gradients, not only a synthetic kernel.
    gradients = []
    for engine in engines:
        with selection_execution(engine):
            gradients.append(engine.batch_gradient_sketch(indexes[1].windows_at(np.arange(8)))[1])
    np.testing.assert_allclose(gradients[0], gradients[1], rtol=2e-3, atol=2e-5)
    report["gradient_relative_l2"] = float(np.linalg.norm(gradients[0]-gradients[1]) /
                                            np.linalg.norm(gradients[0]))
    return report


def bench_rage(args):
    import numpy as np
    from rage.core import TimeSeriesFragment
    from rage.experiments import arms
    pool = [TimeSeriesFragment(np.arange(2400, dtype=np.float32)+i/7,fragment_id=i)
            for i in range(16)]
    reference = [TimeSeriesFragment(np.arange(1600,dtype=np.float32)+i/3,fragment_id=100+i)
                 for i in range(8)]
    def field(fragment, starts):
        windows = np.stack([fragment.extract_window(int(s),8,4)[:8] for s in starts])
        return np.sin(windows/20).astype(np.float32)
    outputs, times, cache = [], [], {}
    kwargs = dict(gradient_computer=field,context_length=8,prediction_length=4,
                  window_stride=1,cover_density=.03,error_density=.04,split_density=.01,
                  block_budget=8,n_select=64,mean_epsilon=.01,endpoint_epsilon=.01,
                  tie_seed=7,device="cpu")
    for _ in range(2):
        index = arms.WindowIndex(pool,context_length=8,prediction_length=4)
        start = time.perf_counter()
        outputs.append(arms.rage_crb_select(reference,pool,index,pool_block_cache=cache,**kwargs))
        times.append(time.perf_counter()-start)
    assert outputs[0].window_ids == outputs[1].window_ids
    np.testing.assert_array_equal(outputs[0].selected.regions,outputs[1].selected.regions)
    np.testing.assert_array_equal(outputs[0].selected[:],outputs[1].selected[:])
    assert outputs[0].cost.gradient_computations == outputs[1].cost.gradient_computations
    for key in ("reference_queries","candidate_blocks","returned_windows","score_stats"):
        assert outputs[0].diagnostics[key] == outputs[1].diagnostics[key],key
    assert outputs[1].cost.notes["pool_cached"] and not outputs[0].cost.notes["pool_cached"]
    return dict(cold_seconds=times[0],warm_seconds=times[1],
                delivery_revision="point-budget-regions-v1", identical_selection_scores_counts=True,
                pool_memory_reuse_identical=True,
                note="Synthetic field, cold/warm same-protocol comparison. Historical fixed-window "
                     "deployment is not an equivalent speed baseline; use cpu suite for CRB parity.")
