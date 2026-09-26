import sys
import subprocess
import threading

import numpy as np
import pytest

from rags.experiments.resampling import rags_global_marginal, rags_marginal


class GuardedRows(list):
    """Fail if scoring requests a full direction matrix instead of batches."""
    def __init__(self, rows, maximum):
        super().__init__(rows)
        self.maximum = maximum
        self.observed = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('Full direction matrix conversion is forbidden')

    def __getitem__(self, key):
        selected = super().__getitem__(key)
        if isinstance(key, slice):
            self.observed.append(len(selected))
            assert len(selected) <= self.maximum
        return selected


@pytest.mark.parametrize('global_competition', [False, True])
def test_batched_native_directions_preserve_probabilities_without_full_conversion(global_competition):
    rng = np.random.default_rng(73)
    refs = rng.normal(size=(33, 64)).astype(np.float32)
    sources = rng.normal(size=(77, 64)).astype(np.float32)
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    sources /= np.linalg.norm(sources, axis=1, keepdims=True)
    n = rng.integers(1, 17, len(refs)); m = rng.integers(1, 17, len(sources))
    cosines = refs.astype(np.float64) @ sources.astype(np.float64).T
    if global_competition:
        scores = (n/n.sum()) @ cosines
        logits = (scores-scores.max())/.25 + np.log(m)
        expected = np.exp(logits-logits.max()); expected /= expected.sum()
        actual, _ = rags_global_marginal(GuardedRows(refs, 11), GuardedRows(sources, 11),
            n, m, temperature=.25, gating=False, direction_batch=11)
    else:
        logits = (cosines-cosines.max(axis=1, keepdims=True))/.25 + np.log(m)
        weights = np.exp(logits-logits.max(axis=1, keepdims=True))
        expected = (n/n.sum()) @ (weights/weights.sum(axis=1, keepdims=True))
        actual, _ = rags_marginal(GuardedRows(refs, 7), GuardedRows(sources, 11),
            n, m, temperature=.25, gating=False, device='cpu', row_batch=7, source_batch=11)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_duplicate_launcher_waits_before_worker_body(tmp_path, monkeypatch):
    from rags.experiments import full_pool_worker
    out = tmp_path/'students'/'chronos-base'
    out.mkdir(parents=True)
    lock_path = out/'worker.lock'
    body_started = threading.Event()
    result = []
    monkeypatch.setattr(full_pool_worker, '_run_worker',
        lambda *args, **kwargs: (body_started.set(), out)[1])
    incumbent = subprocess.Popen([sys.executable, '-c',
        "import fcntl,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); "
        "print('locked',flush=True); sys.stdin.readline()", str(lock_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    thread = None
    try:
        assert incumbent.stdout.readline().strip() == 'locked'
        thread = threading.Thread(target=lambda: result.append(full_pool_worker.run_worker(tmp_path, 'chronos-base')))
        thread.start()
        assert not body_started.wait(.2)
        incumbent.communicate('\n', timeout=5)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert body_started.is_set() and result == [out]
    finally:
        if incumbent.poll() is None:
            incumbent.kill(); incumbent.wait()
        if thread is not None:
            thread.join(timeout=5)
