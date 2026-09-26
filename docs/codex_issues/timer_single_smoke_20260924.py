from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoModelForCausalLM, StoppingCriteriaList, MaxLengthCriteria
from adapters.foundation.timerxl import TimerXLAdapter

path='/root/autodl-tmp/hf-integration/hub/models--thuml--timer-base-84m/snapshots/70077a71acce1b4c00d98332fcaabc694255d8e5'
torch.set_num_threads(4)
model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16).cuda().eval()
adapter=object.__new__(TimerXLAdapter)
adapter.spec=SimpleNamespace(require=lambda capability:None,max_context=2880)
adapter.load=lambda:model
values=np.random.default_rng(17).normal(size=(2,480)).astype(np.float32)
actual=adapter.forecast(values,prediction_length=96)
assert actual.shape==(2,96) and np.isfinite(actual).all()
x=torch.tensor(values,device='cuda',dtype=torch.bfloat16)
with torch.no_grad():
    # Reference harness only: restore the removed legacy extraction helper.
    # use_cache=False means there is no cached state; no numerical code changes.
    model._extract_past_from_model_output=lambda outputs, **kwargs: outputs.past_key_values
    expected=model._greedy_search(x,stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(576)]),
        attention_mask=torch.ones_like(x,dtype=torch.long),use_cache=False,
        synced_gpus=False,return_dict_in_generate=False)
np.testing.assert_array_equal(actual,expected[:,-96:].float().cpu().numpy())
print('REAL_TIMER_SMOKE_PASS: BF16 batch2 C480/H96 equals pinned _greedy_search bit-exact (reference-only legacy cache extractor shim, use_cache=False); no BP/cache writes')
