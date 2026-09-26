import json
from rags.experiments.v7_figures import load_snapshot,mean_curves,rank_arms,GROUPS


def test_legacy_display_preserves_raw_group_identity():
    from rags.experiments.v7_display import display_arm
    assert 'grad-cos' in GROUPS['main']
    assert 'gradcos-legacy' not in GROUPS['main']
    assert display_arm('grad-cos') == 'gradcos-legacy'


def test_snapshot_mean_truncates_and_excludes(tmp_path):
    config=dict(settings={'m':{'epochs':3}},seeds=[0,1,2],datasets=['task'],arms=['orig','rand','grad-dot'])
    for seed,value in [(0,2),(1,4)]:
        root=tmp_path/f'seeds/seed{seed}/m/cells';root.mkdir(parents=True)
        for arm in config['arms']:
            cell=dict(student='m',seed=seed,task='task',arm=arm,mase=99,
                test_curve=[dict(epoch=i/2,metrics={'mase':value if arm=='grad-dot' else 8}) for i in range(1,9)])
            (root/f'task__{arm}.json').write_text(json.dumps(cell))
        (root/'excluded__grad-dot.json').write_text('{}')
    snap=load_snapshot(config,tmp_path);curves=mean_curves(snap)
    assert len(snap['rows'])==6
    assert list(curves['m','task','grad-dot'])==[.5,1,1.5,2,2.5,3]
    assert curves['m','task','grad-dot'][3]==3
    ranked,improvement=rank_arms(snap,['rand','grad-dot'])
    assert ranked==['grad-dot','rand']
    assert improvement['grad-dot']==3
    assert all(a in GROUPS['main'] for a in ('grad-dot','grad-cos','grand-static'))


def test_incomplete_grid_not_averaged_as_completed(tmp_path):
    root=tmp_path/'seeds/seed0/m/cells';root.mkdir(parents=True)
    (root/'t__rand.json').write_text(json.dumps(dict(student='m',seed=0,task='t',arm='rand',
        test_curve=[dict(epoch=.5,metrics={'mase':1})])))
    snap=load_snapshot(dict(settings={'m':{'epochs':4}},seeds=[0],datasets=['t'],arms=['rand']),tmp_path)
    assert not snap['rows'] and len(snap['skipped'])==1


def test_original_plasma_endpoints_and_global_order():
    import matplotlib.pyplot as plt
    from rags.experiments.v7_figures import color_map_for
    colors=color_map_for(['worst','orig','best','middle'],['best','middle','worst'])
    assert colors['best']==plt.get_cmap('plasma')(0.)
    assert colors['worst']==plt.get_cmap('plasma')(1.)
    assert colors['middle']==plt.get_cmap('plasma')(.5)
    assert 'orig' not in colors


def test_symlog_preserves_extremes_and_color():
    import matplotlib.pyplot as plt
    import numpy as np
    from rags.experiments.v7_figures import symlog_axis
    fig,ax=plt.subplots()
    line,=ax.plot([0,1,2,3],[-2,0,.5,1e6],color='purple')
    before=line.get_ydata().copy()
    symlog_axis(ax)
    assert ax.get_yscale()=='symlog'
    np.testing.assert_array_equal(before,line.get_ydata())
    assert line.get_color()=='purple'
    assert ax.get_ylim()[0]<=-2 and ax.get_ylim()[1]>=1e6
    plt.close(fig)


def test_focus_keeps_extreme_in_overview_and_discloses_clipping():
    import matplotlib.pyplot as plt
    import numpy as np
    from rags.experiments.v7_figures import focus_with_overview,focus_limits,draw_overview
    fig,ax=plt.subplots()
    line,=ax.plot([.5,1,1.5,2,2.5,3],[1,1.1,1,1.2,1.1,1e6],color='purple')
    original=line.get_ydata().copy()
    audit=focus_with_overview(ax)
    assert audit['full_max']==1e6
    assert ax.get_ylim()[1]<2
    assert not ax.child_axes
    inset=fig.add_axes([.6,.6,.2,.2]);draw_overview(inset,audit,'test')
    assert inset.get_ylim()[1]>=1e6
    np.testing.assert_array_equal(inset.lines[0].get_ydata(),original)
    np.testing.assert_array_equal(line.get_ydata(),original)
    assert any(l.get_marker()=='^' for l in ax.lines)
    assert focus_limits([1,1.1,1.2,1.3]) is None
    plt.close(fig)


def test_overview_grid_preserves_empty_task_slots_and_does_not_cover_main():
    import matplotlib.pyplot as plt
    from rags.experiments.v7_figures import focus_with_overview,overview_area
    fig=plt.figure();main=fig.add_axes([.1,.6,.8,.3])
    main.plot([0,1,2,3,4],[1,1.1,1.2,1,1e6],label='method')
    focus=focus_with_overview(main)
    cells=overview_area(fig,(.1,.1,.8,.3),['a','b','c','d','e','f','g'],
                        {'c':focus,'g':focus},*main.get_legend_handles_labels())
    assert len(cells)==9
    assert all(ax.axison for ax in cells)
    assert [i for i,ax in enumerate(cells) if ax.lines]==[2,6]
    assert all(ax.get_title().splitlines()[0]==task for ax,task in zip(cells,['a','b','c','d','e','f','g']))
    assert cells[0].texts[0].get_text()=='No zoom needed'
    assert cells[7].get_title()=='Unused slot'
    assert not main.child_axes and main.get_legend() is None
    assert all(not ax.get_position().overlaps(main.get_position()) for ax in cells)
    plt.close(fig)


def test_gradient_labels_hidden_but_symlog_retained():
    import matplotlib.pyplot as plt
    from rags.experiments.v7_figures import gradient_axis
    fig,ax=plt.subplots();ax.plot([0,1,2],[0,1,1e6])
    gradient_axis(ax);fig.canvas.draw()
    assert ax.get_yscale()=='symlog'
    assert not any(t.get_visible() for t in ax.get_yticklabels())
    assert ax.get_ylim()[1]>=1e6
    plt.close(fig)
