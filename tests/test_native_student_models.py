"""Real upstream model code, tiny random configs; run in each isolated env.

No pretrained downloads and no formal data. Exercises native loss backward,
context-only forecasts, four saved/evaluated checkpoints and state restoration.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers
from dataclasses import replace
import json
import subprocess
import sys

from adapters.foundation import build_adapter
from rage.experiments.students import adapter_engine
from rage.experiments.train_eval import run_arm_training


def exercise(adapter, tmp_path):
    # Exercise the same from_pretrained loader used by formal jobs, with tiny weights.
    adapter._model.save_pretrained(tmp_path / 'pretrained')
    adapter.spec=replace(adapter.spec,model_id=str(tmp_path / 'pretrained'))
    adapter._model=None
    adapter.load()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adapter._model.to(device).float()
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(2,576)).astype(np.float32)
    model = adapter.load()
    model.train()
    loss = adapter.training_loss(windows,context_length=512,prediction_length=64)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    model.zero_grad(set_to_none=True)
    predictions = adapter.forecast(windows[:,:512],prediction_length=64)
    assert predictions.shape==(2,64) and np.isfinite(predictions).all()
    np.testing.assert_allclose(predictions,adapter.forecast(windows[:,:512],prediction_length=64),rtol=0,atol=0)
    base={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    from rage.experiments.regions import RegionWindows, save_selection
    from rage.experiments.arms import ArmOutput, SelectionCost
    trajectory = rng.normal(size=600).astype(np.float32)
    regions = RegionWindows(np.tile(trajectory, 2), [0, 600, 1200], [[0, 0, 600], [0, 0, 600]],
                            window_length=576, budget_windows=2)
    result=run_arm_training(adapter_engine(adapter,512,64),source_windows=regions,
        source_epochs=4,batch_size=2,learning_rate=1e-5,seed=0,checkpoint_dir=tmp_path,
        test_windows=windows,test_entities=['ETT_1H:a','ETT_1H:b'],context_length=512,
        test_scales=np.ones(2),base_state=base)
    assert len(result['checkpoints'])==4
    assert [r['step'] for r in result['checkpoints']] == [1, 2, 3, 4]
    assert result['selection_coverage']['repeated_points'] == 600
    assert result['mase']==result['checkpoints'][-1]['metrics']['mase']
    assert 'sql' not in result
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v.cpu(),base[k],rtol=0,atol=0)
    assert len(list(tmp_path.glob('epoch-*.pt')))==4
    # Run the actual CLI in this environment as a fresh process (no injected engine).
    save_selection(tmp_path/'selection.npz', ArmOutput('tsds', regions, [], SelectionCost(), {}))
    np.savez(tmp_path/'exam.npz', windows=windows,entities=np.array(['ETT_1H:a','ETT_1H:b']),scales=np.ones(2))
    manifest={'dataset':'ETT_1H','context_length':512,'prediction_length':64,
              'model_paths':{adapter.spec.key:adapter.spec.model_id}, 'device':device,
              'training':{'epochs':4,'batch_size':2,'learning_rate':1e-5,'seeds':[0],
                          'weight_decay':.01,'gradient_clipping':None,'dtype':'float32'},
              'exam':{'path':str(tmp_path/'exam.npz')},
              'selections':{'random':{'path':str(tmp_path/'selection.npz'),'cost':{'wall_seconds':0}}}}
    (tmp_path/'task.json').write_text(json.dumps(manifest))
    run=subprocess.run([sys.executable,'-m','scripts.exp.model_train','--model',adapter.spec.key,
                        '--task-manifest',str(tmp_path/'task.json'),'--output',str(tmp_path/'worker')],
                       capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    cell=json.loads((tmp_path/'worker'/'cells'/'ETT_1H__random__seed0.json').read_text())
    assert len(cell['checkpoints'])==4 and cell['student']==adapter.spec.key


@pytest.mark.skipif(transformers.__version__!='5.3.0',reason='isolated rage-timesfm profile')
def test_timesfm_official_tiny_fp32_512_64(tmp_path):
    from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction
    adapter=build_adapter('timesfm-2.5',device='cpu')
    config=TimesFm2_5Config(hidden_size=32,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=4,head_dim=8,
        context_length=512,output_quantile_len=128)
    adapter._model=TimesFm2_5ModelForPrediction(config)
    exercise(adapter,tmp_path)


@pytest.mark.skipif(transformers.__version__!='4.40.1',reason='isolated rage-timemoe profile')
def test_timemoe_official_tiny_fp32_512_64(tmp_path):
    from adapters.foundation.timemoe import official_classes,native_training_batch
    config_class,model_class=official_classes()
    config=config_class(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,num_experts=2,num_experts_per_tok=1,
        horizon_lengths=[1,8,32,64],max_position_embeddings=4096,
        attn_implementation='eager')
    adapter=build_adapter('timemoe-200m',device='cpu')
    adapter._model=model_class(config)
    # Exact input/label/mask parity with the upstream dataset on selected sequences.
    from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset
    row=np.arange(576,dtype=np.float32)
    row=(row-row.mean())/row.std()
    class Dataset:
        def __len__(self): return 1
        def __getitem__(self,index): return row
        def get_sequence_length_by_idx(self,index): return len(row)
    original=TimeMoEWindowDataset(Dataset(),context_length=575)[0]
    batch=native_training_batch(np.arange(576,dtype=np.float32)[None],device='cpu')
    for key in original:
        np.testing.assert_allclose(batch[key][0],original[key],rtol=0,atol=0)
    exercise(adapter,tmp_path)
