import json
import numpy as np
import pytest
from rage.experiments.v7_costs import (selection_storage, collect, render,
                                     update_markdown, BEGIN, END, endpoint_mase,
                                     shared_inventory)


def selection(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=np.arange(20,dtype=np.float64),
        probabilities=np.ones(20)/20, descending_order=np.arange(20),
        starts=np.empty(0,dtype=np.int64), counts=np.empty(0,dtype=np.int64),
        cost=json.dumps({'wall_seconds': 2.0}))


def dump(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))


def test_storage_payload_and_header_only(tmp_path, monkeypatch):
    p=tmp_path/'selection.npz'; selection(p)
    original=np.load
    def guarded(stream, **kwargs):
        # Only tiny JSON cost is allowed through np.load, not the score vectors.
        result=original(stream, **kwargs)
        assert result.ndim == 0
        return result
    monkeypatch.setattr(np, 'load', guarded)
    s=selection_storage(p)
    assert s['arrays']['scores']['payload_bytes']==160
    assert s['arrays']['counts']['payload_bytes']==0
    assert s['file_bytes']==p.stat().st_size


def test_accounting_scope_missing_and_no_duplicate_assets(tmp_path):
    root=tmp_path/'seeds/seed0/model'
    for arm in ['grand-static','grad-dot']:
        selection(root/'selections'/f'active__{arm}.npz')
    selection(root/'selections/excluded__grand-static.npz')
    dump(root/'select/active.json',dict(complete=True,selections=[dict(
        arm='grand-static',elapsed_seconds=5,selection_new_bp=0)]))
    dump(root/'assets/active.json',dict(audit=dict(new_bp=7, imported_rows=2,
        cached_rows=3,elapsed_seconds=10,plans=dict(
            source_audit=dict(required_bp=9),reference_audit=dict(required_bp=4)))))
    dump(root/'cells/active__grand-static.json',dict(arm='grand-static',mase=9,
        final_epoch=4,elapsed_seconds=12,test_curve=[dict(epoch=3,metrics=dict(mase=2))]))
    dump(root/'cells/active__rand.json',dict(arm='rand',final_epoch=3,mase=4))
    config=dict(seeds=[0],settings={'model':{'epochs':3}},datasets=['active'],
                arms=['grand-static','grad-dot','rand'])
    report=collect(config,tmp_path)
    assert len(report['shared_assets'])==1
    rows={r['arm']:r for r in report['rows']}
    assert rows['grand-static']['required_reference_bp']==0
    assert rows['grad-dot']['required_reference_bp']==4
    assert rows['grad-dot']['selection_elapsed_s'] is None
    assert rows['grand-static']['mase']==2
    assert rows['grand-static']['executed_final_epoch']==4
    assert rows['grand-static']['peak_cpu_rss_bytes'] is None
    md=render(report)
    assert '50.00%' in md and '未测' in md
    assert 'excluded' not in md


def test_update_preserves_results_and_is_idempotent(tmp_path):
    path=tmp_path/'v7.md';path.write_text('# User results\n\nKeep me\n')
    block=BEGIN+'\nCosts\n'+END+'\n'
    update_markdown(path,block);first=path.read_text()
    update_markdown(path,block)
    assert path.read_text()==first
    assert '# User results\n\nKeep me' in first
    path.write_text(BEGIN)
    with pytest.raises(ValueError):update_markdown(path,block)


def test_missing_endpoint_is_not_best_or_final_four():
    assert endpoint_mase(dict(arm='x',final_epoch=4,mase=1),3) is None


def test_legacy_cosine_is_display_only():
    from rage.experiments.v7_display import display_arm
    assert display_arm('grad-cos') == 'gradcos-legacy'
    assert display_arm('grad-dot') == 'grad-dot'
    report = {'generated_at': 'now', 'rows': [dict(model='m', arm='grad-cos',
        selection_elapsed_s=None)], 'shared_assets': []}
    markdown = render(report)
    assert '| gradcos-legacy |' in markdown
    assert '不是双侧逐窗口余弦平均' in markdown
    assert report['rows'][0]['arm'] == 'grad-cos'


def test_shared_inventory_deduplicates_links(tmp_path):
    import os
    root=tmp_path/'cache';root.mkdir()
    (root/'data').write_bytes(b'1234')
    os.link(root/'data',root/'hardlink')
    (root/'symlink').symlink_to(root/'data')
    result=shared_inventory([root,root])
    assert len(result)==1
    assert result[0]['files']==1
    assert result[0]['logical_bytes']==4
    assert result[0]['skipped_links']==1
