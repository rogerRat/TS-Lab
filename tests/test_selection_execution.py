"""Execution optimisations preserve windows, forecasts and gradient projections."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.common import reference_windows
from gradient.count_sketch import add_count_sketch
from rags.core import TimeSeriesFragment
from rags.execution import SelectionExecution, forecast_mse, prepare_contexts, selection_execution
from rags.experiments.arms import WindowIndex
from rags.generation import generate_tokens


@pytest.mark.parametrize("stride,cap", [(1, None), (1, 11), (3, 11), (3, 1000)])
def test_reference_sampling_matches_enumerate_then_sample(stride, cap):
    fragments = [TimeSeriesFragment(np.arange(n, dtype=np.float64) + i * 100, fragment_id=i)
                 for i, n in enumerate([7, 200, 31])]
    ids, windows = [], []
    for fragment in fragments:
        for start in fragment.valid_starts(8, 4, stride):
            ids.append((fragment.fragment_id, int(start)))
            windows.append(fragment.extract_window(int(start), 8, 4))
    keep = np.arange(len(ids))
    if cap is not None and len(ids) > cap:
        keep = np.sort(np.random.default_rng(7).choice(len(ids), size=cap, replace=False))
    actual, actual_ids = reference_windows(fragments, context_length=8, prediction_length=4,
                                          stride=stride, max_windows=cap, seed=7)
    assert actual_ids == [ids[i] for i in keep]
    np.testing.assert_array_equal(actual, np.asarray(windows, dtype=np.float32)[keep])


def test_window_gather_preserves_arbitrary_order_repeats_and_trajectory_edges():
    fragments = [TimeSeriesFragment(np.arange(37) + 100 * i, fragment_id=10 + 7 * i)
                 for i in range(3)]
    index = WindowIndex(fragments, context_length=8, prediction_length=4, stride=3)
    positions = np.array([len(index)-1, 0, 0, 12, 8, 1, 7])
    expected = [index.by_fragment[int(index.fragment_ids[p])].extract_window(
        int(index.starts[p]), 8, 4) for p in positions]
    np.testing.assert_array_equal(index.windows_at(positions), expected)


def test_selection_threads_are_shared_and_restored():
    before = torch.get_num_threads()
    engine = SimpleNamespace(execution=SelectionExecution(cpu_threads=2))
    with selection_execution(engine):
        assert torch.get_num_threads() == 2
    assert torch.get_num_threads() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_fused_count_sketch_same_signed_accumulation(strided, compact):
    torch.manual_seed(8)
    x = torch.randn((7, 4099), device="cuda")
    if strided:
        x = x.T.contiguous().T
    hashes = torch.randint(0, 127, (x.shape[1],), device="cuda")
    signs = torch.randint(0, 2, (x.shape[1],), device="cuda").float() * 2 - 1
    if compact:
        hashes, signs = hashes.int(), signs.to(torch.int8)
    expected, actual = torch.zeros(7, 127, device="cuda"), torch.zeros(7, 127, device="cuda")
    add_count_sketch(expected, x, hashes, signs, fused=False)
    add_count_sketch(actual, x, hashes, signs, fused=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def native_engine():
    from chronos import ChronosConfig
    from transformers import T5Config, T5ForConditionalGeneration
    torch.manual_seed(17)
    config = T5Config(vocab_size=32, d_model=32, d_ff=64, d_kv=16, num_layers=1,
                      num_heads=2, decoder_start_token_id=0, pad_token_id=0,
                      eos_token_id=1, tie_word_embeddings=False, dropout_rate=0.0)
    model = T5ForConditionalGeneration(config).cuda().eval()
    tokenizer = ChronosConfig(tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0}, n_tokens=32,
        n_special_tokens=2, pad_token_id=0, eos_token_id=1, use_eos_token=True,
        model_type="seq2seq", context_length=512, prediction_length=64,
        num_samples=1, temperature=1., top_k=50, top_p=1.).create_tokenizer()
    return SimpleNamespace(model=model, tokenizer=tokenizer, device=torch.device("cuda"),
                           config=SimpleNamespace(context_length=512, prediction_length=64),
                           execution=SelectionExecution(inference_batch=8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_tokens_scores_masks_and_current_weights_match_native_generate():
    engine = native_engine()
    x = np.random.default_rng(5).normal(size=(8, 576)).astype(np.float32)
    prepared = prepare_contexts(engine, x)
    prepared.attention_mask[0, :40] = 0
    with selection_execution(engine):
        for update in range(2):
            engine.execution = replace(engine.execution, generation="eager")
            tokens, scores = generate_tokens(engine, prepared.input_ids, prepared.attention_mask,
                                             output_scores=True)
            expected_mse = forecast_mse(engine, prepared, batch_size=3)
            engine.execution = replace(engine.execution, generation="cuda_graph")
            actual, actual_scores = generate_tokens(engine, prepared.input_ids, prepared.attention_mask,
                                                    output_scores=True)
            torch.testing.assert_close(actual, tokens, rtol=0, atol=0)
            torch.testing.assert_close(actual_scores, scores, rtol=1e-4, atol=1e-5)
            assert tokens.unique().numel() > 1
            assert forecast_mse(engine, prepared, batch_size=3) == expected_mse
            with torch.no_grad():
                engine.model.lm_head.weight.normal_(std=.2)
                prepared.input_ids.random_(2,32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nondefault_generation_rules_use_official_generate():
    engine = native_engine()
    engine.model.generation_config.suppress_tokens = [2, 3]
    prepared = prepare_contexts(engine, np.ones((2,576),dtype=np.float32))
    with selection_execution(engine):
        tokens, _ = generate_tokens(engine, prepared.input_ids, prepared.attention_mask)
    assert not hasattr(engine, "_greedy_graphs")
    assert not torch.isin(tokens, torch.tensor([2,3],device="cuda")).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_training_four_checkpoints_graph_evaluation_and_restore(tmp_path):
    from rags.engines import ChronosT5Engine, ChronosT5Config
    from rags.experiments.train_eval import run_arm_training
    from rags.execution import clear_execution_cache
    native = native_engine()
    engine = ChronosT5Engine.__new__(ChronosT5Engine)
    engine.__dict__.update(native.__dict__)
    engine.config = ChronosT5Config(context_length=512,prediction_length=64)
    x = np.random.default_rng(15).normal(size=(4,576)).astype(np.float32)
    base = {k:v.clone() for k,v in engine.model.state_dict().items()}
    results = []
    with selection_execution(engine):
        for mode in ("eager", "cuda_graph"):
            engine.execution = replace(engine.execution,generation=mode)
            result = run_arm_training(engine,source_windows=x,source_epochs=4,
                batch_size=4,learning_rate=1e-5,seed=7,test_windows=x[:2],
                test_entities=["ETT_1H"]*2,context_length=512,
                test_scales=np.ones(2),base_state=base,checkpoint_dir=tmp_path/mode)
            assert len(result["checkpoints"]) == 4
            results.append(result)
            for key,value in engine.model.state_dict().items():
                torch.testing.assert_close(value,base[key],rtol=0,atol=0)
        for a,b in zip(results[0]["checkpoints"],results[1]["checkpoints"]):
            assert a["metrics"] == b["metrics"]
            left = torch.load(a["checkpoint"],weights_only=True)
            right = torch.load(b["checkpoint"],weights_only=True)
            for key in left:
                torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        assert engine._greedy_graphs
        clear_execution_cache(engine)
        assert not hasattr(engine,"_greedy_graphs")
