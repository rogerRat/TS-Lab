"""Small synthetic CPU benchmarks; never open production caches or a GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from assets.gradient.cache import GradientCache, WindowRequest
from rags.execution import SelectionExecution
from rags.experiments.reference_opinions import reference_scores_many, AGGREGATIONS


def old_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Probe:
    device = 'cpu'
    config = SimpleNamespace(projection_dimension=4096, projection_seed=42)
    execution = SelectionExecution(gradient_batch=8)
    last_gradient_execution = dict(backend='fixture', batch_size=1, oom_retries=0)
    def batch_gradient_sketch(self, windows):
        return windows.mean(1), np.repeat(windows[:, :1], 4096, axis=1)


def main():
    before = Path('/root/TS3/temp/ct-cpu-opt/before')
    baseline = old_module('assets.gradient._benchmark_before', before / 'cache.py')
    old_scores = old_module('rags.experiments._benchmark_before', before / 'reference_opinions.py')
    with tempfile.TemporaryDirectory(prefix='ct-cpu-test-', dir='/root/autodl-tmp') as root:
        for label, cls in [('before', baseline.GradientCache), ('after', GradientCache)]:
            cache = cls(Path(root) / label, Probe(), model_identity={'model_id':'fixture'},
                        data_fields={'pool':'synthetic'}, context_length=4, prediction_length=2)
            start = time.perf_counter()
            for row in range(256):
                cache.get(WindowRequest('source', [[1, row+1]],
                    lambda indices, row=row: np.full((len(indices), 6), row/7, dtype=np.float32)))
            cache.close()
            print(json.dumps(dict(phase='sparse_durable_write_read', version=label,
                seconds=time.perf_counter()-start, shards=len(cache.shards),
                disk_bytes=sum(path.stat().st_size for path in cache.shards))), flush=True)
    rng = np.random.default_rng(121)
    refs = rng.normal(size=(256, 1024)); sources = rng.normal(size=(2048, 1024))
    refs /= np.linalg.norm(refs,axis=1)[:,None]; sources /= np.linalg.norm(sources,axis=1)[:,None]
    weights = np.full(len(refs), 1/len(refs))
    for repeat in range(3):
        start = time.perf_counter()
        expected = {mode: old_scores.reference_scores(refs, sources, weights, aggregation=mode, device='cpu')
                    for mode in AGGREGATIONS}
        old_seconds = time.perf_counter() - start
        start = time.perf_counter()
        actual = reference_scores_many(refs, sources, weights, device='cpu')
        new_seconds = time.perf_counter() - start
        for mode in AGGREGATIONS:
            np.testing.assert_array_equal(actual[mode][0], expected[mode][0])
            assert actual[mode][1] == expected[mode][1]
        print(json.dumps(dict(phase='four_arm_matching', repeat=repeat,
            before=old_seconds, after=new_seconds, speedup=old_seconds/new_seconds,
            scores_and_audits='bitwise_equal')), flush=True)


if __name__ == '__main__':
    main()
