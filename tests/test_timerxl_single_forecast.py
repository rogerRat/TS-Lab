from types import SimpleNamespace
import numpy as np
import pytest
import torch
from adapters.foundation.timerxl import TimerXLAdapter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(input_token_len=96, output_token_lens=[96])
    def prepare_inputs_for_generation(self, inputs, attention_mask, use_cache):
        assert attention_mask.shape == (len(inputs), 5)
        assert not use_cache
        return dict(input_ids=inputs, revin=True, use_cache=False)
    def forward(self, input_ids, revin, use_cache, return_dict, max_output_length):
        assert revin and not use_cache and return_dict and max_output_length == 96
        return SimpleNamespace(logits=input_ids[:, -96:]*self.weight)
    def generate(self, *args, **kwargs):
        raise AssertionError('single patch must not enter token generation')


def test_single_patch_uses_official_forward_without_cache():
    adapter = object.__new__(TimerXLAdapter)
    adapter.spec = SimpleNamespace(require=lambda capability: None, max_context=2880)
    model = Model()
    adapter.load = lambda: model
    values = np.arange(960,dtype=np.float32).reshape(2,480)
    np.testing.assert_array_equal(adapter.forecast(values,prediction_length=96),values[:,-96:])
    model.config.output_token_lens = [192]
    with pytest.raises(ValueError,match='pinned 96-point head'):
        adapter.forecast(values,prediction_length=96)
