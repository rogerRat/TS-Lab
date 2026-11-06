from threading import Barrier, Lock
import numpy as np
import pytest

from rage.experiments.reference_execution import gradient_agreement, choose_execution
from rage.experiments.reference_queue import execute_streams, commands


def test_agreement_tiny_gradients_and_identity_shape():
    g = np.array([[1e-30, 2e-30]], dtype=np.float32)
    assert gradient_agreement([1], g, [1], g)['passed']
    assert not gradient_agreement([1], -g, [1], g)['passed']
    assert not gradient_agreement([1], g.T, [1], g)['passed']
    assert not gradient_agreement([1], g*np.nan, [1], g)['passed']


def test_choose_valid_throughput_and_low_memory():
    def row(speed, memory, valid=True, oom=0):
        return dict(windows_per_second=speed, peak_bytes=memory, workers=1,
                    batch=8, agreement=dict(passed=valid), oom_retries=oom)
    rows = [row(100, 100), row(96, 50), row(1000, 1, False), row(200, 1, oom=1)]
    assert choose_execution(rows) is rows[1]
    with pytest.raises(ValueError):
        choose_execution(rows[2:])


def test_stream_overlap_and_per_model_order():
    barrier = Barrier(3, timeout=3)
    seen = []; lock = Lock()
    def run(job):
        if job.endswith('ref'):
            barrier.wait()  # Fails if models are globally serialized.
        with lock:
            seen.append(job)
    execute_streams({m: [m+'ref', m+'align'] for m in 'abc'}, run)
    for m in 'abc':
        assert seen.index(m+'ref') < seen.index(m+'align')


def test_failure_blocks_only_own_dependents():
    seen = []
    def run(job):
        seen.append(job)
        if job == 'bad':
            raise ValueError('test')
    with pytest.raises(RuntimeError, match='failed'):
        execute_streams(dict(a=['bad', 'forbidden'], b=['ok', 'ok2']), run)
    assert 'forbidden' not in seen
    assert 'ok2' in seen


def test_queue_has_one_stream_per_cache_and_no_stage_markers():
    import json
    from pathlib import Path
    config = json.loads(Path('configs/rage/reference_census_execution.json').read_text())
    streams = commands(config)
    assert set(streams) == {'chronos-base', 'timesfm-2.5'}
    assert [len(jobs) for jobs in streams.values()] == [3, 3]
    assert all('timerxl' not in run for m in config['models'] for run in m['runs'])
    assert config['report_subdir'] == 'report-chronos-timesfm'
    for jobs in streams.values():
        assert jobs[0]['command'][-3] == 'references'
        assert all(job['command'][-3] == 'alignment' for job in jobs[1:])


def test_published_reader_and_async_geometry_match_sync(tmp_path):
    from types import SimpleNamespace
    from assets.gradient.cache import GradientCache
    from rage.core import TimeSeriesFragment
    from rage.crb import CRBConfig
    from rage.execution import SelectionExecution
    from rage.experiments.reference_census import collect_references
    from rage.experiments.reference_execution import PublishedGradientRows
    class Probe:
        device = 'cpu'
        execution = SelectionExecution(gradient_batch=3)
        config = SimpleNamespace(projection_dimension=2, projection_seed=1)
        count = 0
        def batch_gradient_sketch(self, windows):
            self.count += len(windows)
            return windows.mean(1), windows[:, [0, -1]]
    probe = Probe()
    refs = [TimeSeriesFragment(np.arange(14)+i, entity_id=str(i), fragment_id=i)
            for i in range(3)]
    cache = GradientCache(tmp_path, probe, model_identity={'model_id':'fixture'},
                          data_fields={}, context_length=4, prediction_length=2)
    try:
        crb = CRBConfig(context_length=4, prediction_length=2)
        asynchronous = collect_references(cache, refs, 'task', 4, 2, crb, chunk=4)
        synchronous = collect_references(cache, refs, 'task', 4, 2, crb, chunk=4,
                                         overlap_geometry=False)
        assert asynchronous == synchronous
        assert probe.count == 27  # Geometry, CRB and resume perform no new BP.
        reader = PublishedGradientRows(cache, 'reference:task', 0, 9, budget=1)
        np.testing.assert_array_equal(reader.read(0, 9), np.column_stack((np.arange(9), np.arange(9)+5)))
        with pytest.raises(ValueError):
            reader.read(-1, 2)
    finally:
        cache.close()


def test_tuner_preserves_science_and_rejects_wrong_batch():
    from rage.execution import SelectionExecution
    from rage.experiments.reference_execution import tune_execution
    class Probe:
        device = 'cpu'
        execution = SelectionExecution(gradient_batch=1, gradient_workers=1)
        max_gradient_batch = 4
        def batch_gradient_sketch(self, windows):
            values = windows[:, :2].copy()
            if self.execution.gradient_batch == 4:
                values *= -1  # Deliberate semantic violation, even if faster.
            self.last_gradient_execution = dict(oom_retries=0)
            return windows.mean(1), values
    probe = Probe()
    result = tune_execution(probe, np.arange(24.).reshape(6, 4), [2, 4, 8], [1], repeats=1)
    assert result['chosen']['batch'] == 2
    assert probe.execution.gradient_batch == 2
    assert len(result['candidates']) == 2  # Native cap respected.
    assert not result['candidates'][1]['agreement']['passed']
