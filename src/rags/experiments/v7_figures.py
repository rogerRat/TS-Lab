"""V7 completed-cell mean curves and read-only frozen-gradient diagnostics."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from .v7_display import display_arm

GROUPS = {
    'main': ['rags-block-expSum','rags-block-linearSum','rand','less-main','tsds',
             'lp-mirror','grad-dot','grad-cos','grand-static','orig'],
    'aggregation': [f'rags-{level}-{agg}' for level in ('block','traj')
                    for agg in ('expSum','linearSum','maxSum','zeroClipSum')],
    'uni': ['rags-block-expSum','rags-block-expSum-win-uni','rags-block-expSum-block-uni',
            'rags-block-linearSum','rags-block-linearSum-win-uni','rags-block-linearSum-block-uni'],
}
TITLES = {'main':'Main comparison', 'aggregation':'Block / trajectory aggregation',
          'uni':'Within-unit weighting'}
SYMLOG_LINTHRESH = 0.01


def symlog_axis(ax):
    """Transform only the display; keep every data point and autoscaled limit."""
    from matplotlib.ticker import MaxNLocator
    ax.set_yscale('symlog',base=10,linthresh=SYMLOG_LINTHRESH,linscale=1)
    low,high=ax.get_ylim()
    # Matplotlib's decade locator can leave narrow positive MASE panels unlabeled.
    if low>0 and high/low<10:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def focus_limits(values):
    """Only zoom when Tukey outliers more than double the bulk's visible span."""
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if len(values)<4:return None
    q1,q3=np.quantile(values,[.25,.75])
    spread=max(q3-q1,abs(float(np.median(values)))*.01,1e-12)
    bulk=values[(values>=q1-1.5*spread)&(values<=q3+1.5*spread)]
    if not len(bulk) or len(bulk)==len(values):return None
    low,high=float(bulk.min()),float(bulk.max())
    width=max(high-low,spread)
    if np.ptp(values)<2*width:return None
    return low-.08*width,high+.08*width


def focus_with_overview(ax, *, debug=False):
    """Focus main panel and return full traces for the separate overview area."""
    lines=list(ax.lines)
    values=np.concatenate([np.asarray(line.get_ydata(),dtype=float) for line in lines]) if lines else []
    limits=focus_limits(values)
    if limits is None:return None
    traces=[]
    for line in lines:
        x=np.asarray(line.get_xdata(),dtype=float)
        y=np.asarray(line.get_ydata(),dtype=float)
        traces.append(dict(x=x.tolist(),y=y.tolist(),color=line.get_color(),
                           linestyle=line.get_linestyle(),orig=line.get_label()=='orig'))
        for mask,boundary,symbol in ((y>limits[1],limits[1],'^'),(y<limits[0],limits[0],'v')):
            if not np.any(mask):continue
            # Orig is a horizontal axes-relative line, not an epoch series.
            if line.get_label()=='orig':
                ax.plot([.98],[boundary],marker=symbol,ls='none',color=line.get_color(),
                        transform=ax.get_yaxis_transform(),ms=5,clip_on=False)
            else:
                ax.plot(x[mask],np.full(np.count_nonzero(mask),boundary),marker=symbol,ls='none',
                        color=line.get_color(),ms=5,clip_on=False)
    ax.set_ylim(*limits)
    ax.text(.01,.99,'Zoomed; triangles = outside range',transform=ax.transAxes,
            va='top',fontsize=5.5,bbox=dict(facecolor='white',edgecolor='none',alpha=.75),zorder=7)
    return dict(focus=list(limits),full_min=float(np.min(values)),full_max=float(np.max(values)),
                xlim=list(ax.get_xlim()),traces=traces)


def draw_overview(ax, overview, title):
    from matplotlib.ticker import MaxNLocator
    for trace in overview['traces']:
        if trace['orig']:
            ax.axhline(trace['y'][0],color=trace['color'],ls=trace['linestyle'],lw=.8)
        else:
            ax.plot(trace['x'],trace['y'],color=trace['color'],ls=trace['linestyle'],
                    lw=.8,marker='o',ms=1.4)
    ax.set_xlim(overview['xlim'])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.tick_params(labelsize=4.5,pad=1)
    ax.grid(alpha=.2)
    ax.set_title(f'{title}\nmax={overview["full_max"]:.3g}',fontsize=5,pad=2)


def overview_area(fig, bounds, tasks, overviews, handles, labels, gradient_legend=None):
    """Dedicated bottom-right legend + 3x3 overview grid, never over data panels."""
    x,y,width,height=bounds
    legend_ax=fig.add_axes([x,y,width*.29,height]);legend_ax.axis('off')
    legend_ax.legend(handles,labels,loc='upper left',fontsize=7,frameon=False,
                     borderaxespad=0,title='Methods',title_fontsize=8)
    if gradient_legend:
        gh,gl=gradient_legend
        grad_ax=fig.add_axes([x,y,width*.29,height*.31]);grad_ax.axis('off')
        grad_ax.legend(gh,gl,loc='lower left',fontsize=5,ncol=2,frameon=False,
                       borderaxespad=0,title='Reference traces (first task)',title_fontsize=5)
    grid_x=x+width*.36;grid_width=width*.64
    fig.text(grid_x,y+height,'Full-range views (same task positions as main grid)',
             fontsize=7,va='bottom')
    cells=[]
    for i in range(9):
        row,col=divmod(i,3)
        cell_width=grid_width/3;cell_height=height/3
        ax=fig.add_axes([grid_x+col*cell_width+cell_width*.10,
                         y+(2-row)*cell_height+cell_height*.17,
                         cell_width*.80,cell_height*.57])
        if i<len(tasks) and tasks[i] in overviews:
            draw_overview(ax,overviews[tasks[i]],tasks[i].removeprefix('time_'))
        else:
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color('0.65')
            ax.set_title(tasks[i].removeprefix('time_') if i<len(tasks) else 'Unused slot',
                         fontsize=5,pad=2)
            ax.text(.5,.5,'No zoom needed' if i<len(tasks) else '—',
                    transform=ax.transAxes,ha='center',va='center',fontsize=5,color='0.5')
        cells.append(ax)
    return cells


def gradient_axis(ax):
    symlog_axis(ax)
    ax.tick_params(axis='y',which='both',labelleft=False,labelright=False)
    ax.yaxis.get_offset_text().set_visible(False)


def read(path):
    return json.loads(Path(path).read_text())


def load_snapshot(config, run):
    """Use only completed cell files and complete prescribed grids, never best-test."""
    rows=[]; skipped=[]
    for model,settings in config['settings'].items():
        epochs=[i/2 for i in range(1, int(settings['epochs']*2)+1)]
        for seed in config['seeds']:
            for task in config['datasets']:
                for arm in config['arms']:
                    path=Path(run)/f'seeds/seed{seed}/{model}/cells/{task}__{arm}.json'
                    if not path.exists():continue
                    cell=read(path)
                    if (cell.get('student'),cell.get('seed'),cell.get('task'),cell.get('arm')) != (model,seed,task,arm):
                        raise ValueError('Cell identity mismatch: '+str(path))
                    if arm=='orig':
                        curve={0.:float(cell['mase'])}
                    else:
                        records=cell.get('test_curve',cell.get('checkpoints',[]))
                        curve={float(r['epoch']):float(r['metrics']['mase']) for r in records
                               if 0<float(r['epoch'])<=settings['epochs']}
                        if sorted(curve)!=epochs:
                            skipped.append(dict(path=str(path),reason='incomplete prescribed curve'))
                            continue
                    if not np.isfinite(list(curve.values())).all():
                        raise ValueError('Nonfinite curve: '+str(path))
                    rows.append(dict(model=model,seed=seed,task=task,arm=arm,
                                     path=str(path),curve=curve))
    return dict(created_at=datetime.now(timezone.utc).isoformat(),run=str(run),
                tasks=config['datasets'],rows=rows,skipped=skipped,
                policy='available completed seeds mean; fixed prescribed epoch; no best selection')


def mean_curves(snapshot):
    grouped=defaultdict(list)
    for row in snapshot['rows']:
        grouped[(row['model'],row['task'],row['arm'])].append(row['curve'])
    return {key:{epoch:float(np.mean([c[epoch] for c in curves]))
                 for epoch in sorted(curves[0])} for key,curves in grouped.items()}


def rank_arms(snapshot, arms):
    """Original plot color ranking: pooled final-epoch MASE, lower is darker."""
    values=defaultdict(list)
    for row in snapshot['rows']:
        if row['arm']!='orig':values[row['arm']].append(row['curve'][max(row['curve'])])
    means={arm:float(np.mean(v)) for arm,v in values.items()}
    return sorted(arms,key=lambda a:means.get(a,float('inf'))),means


def color_map_for(arms, ranked):
    """Original plasma normalization: best present=0, worst present=1."""
    import matplotlib.pyplot as plt
    ordered=sorted([a for a in arms if a in ranked],key=ranked.index)
    return {a:plt.get_cmap('plasma')(i/max(len(ordered)-1,1)) for i,a in enumerate(ordered)}


def reference_norms(run, model, tasks, seed=0, trajectories=6):
    """Read only selected reference cache shards; no BP, no cache writes."""
    pool_by_task={}
    for task in tasks:
        path=Path(run)/f'seeds/seed{seed}/{model}/assets/{task}.json'
        if path.exists():pool_by_task[task]=Path(read(path)['audit']['path'])
    result={};evidence=[]
    for pool in sorted(set(pool_by_task.values())):
        with np.load(pool/'window_index.npz',allow_pickle=False) as z:
            datasets=[str(x) for x in z['datasets']]
            did=z['dataset_ids'];wids=z['window_ids'];sids=z['shard_ids'];srows=z['shard_rows']
            shards=[str(x) for x in z['shards']]
        requests=defaultdict(list)
        for task in tasks:
            if pool_by_task.get(task)!=pool:continue
            key='reference:'+task
            if key not in datasets:
                raise ValueError('Reference missing from prepared cache: '+key)
            indices=np.flatnonzero(did==datasets.index(key))
            ids=np.unique(wids[indices,0])
            picked=ids[np.linspace(0,len(ids)-1,min(trajectories,len(ids))).astype(int)] if len(ids) else []
            selected=indices[np.isin(wids[indices,0],picked)]
            for i in selected:
                requests[int(sids[i])].append((int(srows[i]),task,int(wids[i,0]),int(wids[i,1])))
            evidence.append(dict(task=task,pool=str(pool),trajectories=[int(x) for x in picked],
                                 cached_windows=len(selected)))
        grouped=defaultdict(list)
        for sid,requests_in_shard in requests.items():
            with np.load(pool/shards[sid],allow_pickle=False) as z:
                indices=[q[0] for q in requests_in_shard]
                if 'gradients_bf16' in z:
                    bits=z['gradients_bf16'][indices].astype(np.uint32)<<16
                    gradients=bits.view(np.float32)
                else:
                    gradients=z['gradients'][indices]
                norms=np.linalg.norm(gradients.astype(np.float64),axis=1)
            if not np.isfinite(norms).all():raise ValueError('Nonfinite cached gradients')
            for (_,task,fid,start),norm in zip(requests_in_shard,norms):
                grouped[(task,fid)].append((start,float(norm)))
        for (task,fid),values in grouped.items():
            result.setdefault(task,{})[fid]=sorted(values)
    return result,evidence


def render(config, run, destination, snapshot=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    snapshot=load_snapshot(config,run) if snapshot is None else snapshot
    for row in snapshot['rows']:
        row['curve']={float(k):v for k,v in row['curve'].items()}
    curves=mean_curves(snapshot)
    arms=sorted({a for group in GROUPS.values() for a in group if a!='orig'})
    ranked,means=rank_arms(snapshot,arms)
    tasks=config['datasets']
    evidence={}
    focus_audit=[]
    def draw(ax,model,task,group,colors,debug=False):
        for arm in group:
            curve=curves.get((model,task,arm))
            if not curve:continue
            if arm=='orig':
                ax.axhline(curve[0.],color='black',ls='--',lw=1.3 if debug else 1.4,label='orig')
            else:
                epochs=sorted(curve)
                ax.plot(epochs,[curve[e] for e in epochs],color=colors[arm],ls='-',
                        lw=1.6 if debug else 1.8,marker='o',ms=2.5 if debug else 3,
                        label=display_arm(arm) if debug else f'#{ranked.index(arm)+1} {display_arm(arm)}')
        ax.set_title(task,fontsize=10 if debug else 9)
        focus=focus_with_overview(ax,debug=debug)
        if focus:focus_audit.append(dict(model=model,task=task,arms=group,debug=debug,
                                         **{k:v for k,v in focus.items() if k!='traces'}))
        ax.grid(alpha=.3);ax.tick_params(labelsize=7)
        if debug:ax.set_ylabel('MASE',fontsize=7)
        return focus
    for model in config['settings']:
        for group,group_arms in GROUPS.items():
            fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True)
            colors=color_map_for(group_arms,ranked)
            overviews={}
            for ax,task in zip(axes.flat,tasks):
                focus=draw(ax,model,task,group_arms,colors)
                if focus:overviews[task]=focus
            for ax in list(axes.flat)[len(tasks):]:ax.axis('off')
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.suptitle(f'V7 | {model} — {TITLES[group]} (mean; plasma: darkest = best ... brightest = worst; #n = global rank)',fontsize=12)
            fig.text(.5,.02,'epoch (checkpoint every 0.5)',ha='center')
            fig.text(.01,.5,'MASE (lower=better; full-range views at bottom right)',va='center',rotation='vertical')
            fig.tight_layout(rect=[.02,.04,1,.97])
            left,right=axes.flat[7].get_position(),axes.flat[8].get_position()
            overview_area(fig,(left.x0,left.y0,right.x1-left.x0,left.height),
                          tasks,overviews,handles,labels)
            fig.savefig(destination/f'{model}__{group}.png',dpi=140)
            plt.close(fig)
        gradients,ev=reference_norms(run,model,tasks)
        evidence[model]=ev
        # Original debug script ranks within each model and uses viridis for trajectories.
        model_rank,_=rank_arms({'rows':[r for r in snapshot['rows'] if r['model']==model]},arms)
        debug_colors=color_map_for(GROUPS['main'],model_rank)
        fig=plt.figure(figsize=(17,15))
        grid=GridSpec(6,3,figure=fig,height_ratios=[4,1,4,1,4,1],hspace=.55,wspace=.3)
        overviews={};gradient_legend=None
        for i,task in enumerate(tasks):
            row,col=2*(i//3),i%3
            ax=fig.add_subplot(grid[row,col])
            focus=draw(ax,model,task,GROUPS['main'],debug_colors,debug=True)
            if focus:overviews[task]=focus
            if i==0:handles,labels=ax.get_legend_handles_labels()
            ax=fig.add_subplot(grid[row+1,col])
            series=sorted(gradients.get(task,{}).items())
            for j,(fid,points) in enumerate(series):
                array=np.asarray(points)
                ax.plot(array[:,0],array[:,1],lw=.9,color=plt.get_cmap('viridis')(j/max(len(series)-1,1)),label=f'traj{fid}')
            ax.set_xlabel('window position',fontsize=6)
            gradient_axis(ax)
            ax.set_ylabel('||g|| (symlog)',fontsize=6)
            ax.grid(alpha=.25);ax.tick_params(labelsize=6)
            if i==0:gradient_legend=ax.get_legend_handles_labels()
        bounds=grid[4:6,1:3].get_position(fig)
        overview_area(fig,bounds.bounds,tasks,overviews,handles,labels,gradient_legend)
        fig.suptitle(f'V7 DEBUG — performance (top) + frozen projected-gradient L2 per trajectory (bottom), {model}',fontsize=13)
        fig.savefig(destination/f'{model}__main_debug.png',dpi=130)
        plt.close(fig)
    snapshot.update(display_aliases={'grad-cos': 'gradcos-legacy'},
                    y_scale=dict(performance='linear; bulk focus + full-range inset when compressed',
                                gradient='symlog',base=10,linthresh=SYMLOG_LINTHRESH,linscale=1,
                                gradient_numeric_y_labels=False),focus_audit=focus_audit,
                    color_order=ranked,color_rank_mean_mase=means,
                    color_policy='original plasma; per-figure endpoints; main global / debug per-model pooled final MASE',gradient_evidence=evidence)
    (destination/'v7_plot_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    return snapshot
