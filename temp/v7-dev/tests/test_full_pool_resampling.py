from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rage.core import TimeSeriesFragment
from rage.crb import CRBConfig
from rage.execution import SelectionExecution, selection_execution
from rage.experiments.arms import WindowIndex
from rage.experiments.full_pool import ARMS, select_full_pool
from assets.gradient import GradientCache
from rage.experiments.resampling import ResampledWindows, trajectory_distribution




class Probe:
    device = "cpu"
    execution = SelectionExecution(gradient_batch=8, inference_batch=8, cpu_threads=2)

    def __init__(self):
        self.config = SimpleNamespace(model_path="fixture", context_length=4,
            prediction_length=2, projection_dimension=3, projection_seed=42)
        self.model = torch.nn.Linear(4, 2)

    def batch_gradient_sketch(self, windows):
        return np.zeros(len(windows)), self.features(windows)

    @staticmethod
    def features(windows):
        return np.stack([windows[:, 0], windows.mean(axis=1), windows[:, -1]], axis=1)

    def training_loss(self, windows):
        data = torch.from_numpy(windows)
        return (self.model(data[:, :4]) - data[:, 4:]).square().mean()

    def forecast(self, contexts):
        with torch.no_grad():
            return self.model(torch.from_numpy(contexts)).numpy()


def inputs():
    rng = np.random.default_rng(12)
    pool = [TimeSeriesFragment(rng.uniform(.1, 2., 30 + i % 7).astype(np.float32),
                              fragment_id=i) for i in range(40)]
    refs = [TimeSeriesFragment(rng.uniform(.1, 2., 30).astype(np.float32),
                              fragment_id=100 + i) for i in range(8)]
    index = WindowIndex(pool, context_length=4, prediction_length=2)
    manifest = dict(total_equivalents=sum(f.length // 6 for f in pool),
                    total_points=sum(f.length for f in pool), seed=12)
    config = CRBConfig(context_length=4, prediction_length=2)
    return dict(reference_fragments=refs, pool=pool, index=index, pool_manifest=manifest,
                model_identity={"model_id": "fixture", "parameter_count": 10}, crb_config=config)
