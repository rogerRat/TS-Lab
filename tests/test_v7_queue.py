import multiprocessing as mp
from rage.experiments.v7_queue import claim, cell_config


def contender(path, output):
    with claim(path) as acquired:
        output.put(acquired)


def test_busy_claim_skips_without_waiting(tmp_path):
    ctx = mp.get_context('spawn')
    output = ctx.Queue()
    path = tmp_path/'cell.lock'
    with claim(path) as acquired:
        assert acquired
        child = ctx.Process(target=contender,args=(path,output))
        child.start()
        assert output.get(timeout=5) is False
        child.join(timeout=5)
        assert child.exitcode == 0
    with claim(path) as acquired:
        assert acquired
    assert path.exists()


def test_shared_task_claims_but_legacy_exclusive_is_skipped(tmp_path):
    path = tmp_path/'task.lock'
    with claim(path,shared=True) as first:
        with claim(path,shared=True) as second:
            assert first and second
            with claim(path) as exclusive:
                assert not exclusive
    with claim(path) as exclusive:
        assert exclusive
        with claim(path,shared=True) as queue:
            assert not queue


def test_queue_slice_preserves_scientific_identity():
    config = dict(arms=['orig','rand','grad-cos'],asset_protocol={'arms':['old']},
                  training_protocol={'max_epochs':{'chronos-base':3}},settings={'batch':32})
    result = cell_config(config,'grad-cos')
    assert result == config
    rand = cell_config(config,'rand')
    assert rand['arms'] == ['orig','rand']
    for key in ('asset_protocol','training_protocol','settings'):
        assert rand[key] == config[key]
    assert config['arms'] == ['orig','rand','grad-cos']
