import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
from rage.experiments.v7_train_leases import model_lease, train_once


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_shared_training_excludes_assets_and_legacy(tmp_path):
    with model_lease(tmp_path,'model','train',parallel_train=True):
        with model_lease(tmp_path,'model','train',parallel_train=True):
            pass
        with pytest.raises(BlockingIOError):
            with model_lease(tmp_path,'model','assets'):pass
        with pytest.raises(BlockingIOError):
            with model_lease(tmp_path,'model','train'):pass
    with model_lease(tmp_path,'model','train'):
        with pytest.raises(BlockingIOError):
            with model_lease(tmp_path,'model','train',parallel_train=True):pass
    with pytest.raises(ValueError):
        with model_lease(tmp_path,'model','select',parallel_train=True):pass


def test_same_task_executes_once_different_tasks_parallel(tmp_path):
    entered,release=Event(),Event()
    calls=[]
    def run(task,execute):
        return train_once(tmp_path,'model',0,task,tmp_path/f'{task}.json',
                          {'id':task},lambda a,b:a==b,execute,save)
    def slow():
        calls.append('first');entered.set()
        assert release.wait(5)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(run,'a',slow)
        assert entered.wait(5)
        second=pool.submit(run,'a',lambda:calls.append('duplicate'))
        independent=pool.submit(run,'b',lambda:calls.append('independent'))
        try:
            assert independent.result(timeout=5)
        finally:release.set()
        assert first.result(timeout=5)
        assert second.result(timeout=5) is False
    assert calls==['first','independent']


def test_failed_task_releases_lease_without_marker(tmp_path):
    marker=tmp_path/'done.json'
    def fail():raise RuntimeError('failure')
    args=(tmp_path,'model',0,'task',marker,{'x':1},lambda a,b:a==b)
    with pytest.raises(RuntimeError):train_once(*args,fail,save)
    assert not marker.exists()
    assert train_once(*args,lambda:None,save)
    with pytest.raises(ValueError):
        train_once(tmp_path,'model',0,'task',marker,{'x':2},lambda a,b:a==b,lambda:None,save)
