"""Dense cached-gradient audit with equal-cost controls and held-out rows."""
import argparse,json,pathlib,dataclasses
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rags.crb import CRBConfig,CRBEngine
from rags.core import TimeSeriesFragment
from rags.window_reconstruction import interpolate_window_gradients
from assets.gradient.variants import aggregate_retained
from visualize.grad_reconst.reviewer_audit import save

def unit(g):return g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-30)
def metrics(g,h,mask):
    g=g[mask].astype(float); h=h[mask].astype(float)
    if not len(g):return dict(n=0,relative_l2=None,mean_cosine=None)
    return dict(n=len(g),relative_l2=float(np.linalg.norm(g-h)/np.linalg.norm(g)),mean_cosine=float(np.mean(np.sum(unit(g)*unit(h),axis=1))))
def interp(g,anchors):
    aa=np.array(sorted(anchors)); rr=np.clip(np.searchsorted(aa,np.arange(len(g)),side='right'),1,len(aa)-1);lo=aa[rr-1];hi=aa[rr]; w=((np.arange(len(g))-lo)/(hi-lo))[:,None]
    return (1-w)*g[lo]+w*g[hi]
def blockfield(r,n,d):
    h=np.zeros((n,d),np.float32);mask=np.zeros(n,bool)
    for b in r.blocks:h[b.start_positions-1]=b.direction;mask[b.start_positions-1]=True
    return h,mask

def main():
    p=argparse.ArgumentParser();p.add_argument('--dense',required=True);p.add_argument('--out',required=True);p.add_argument('--cases',default='');p.add_argument('--skip-retrieval',action='store_true');a=p.parse_args();root=pathlib.Path(a.dense);out=pathlib.Path(a.out);out.mkdir(exist_ok=True,parents=True)
    ranks=json.loads((root/'ranking.json').read_text());reports=[];assets=[];visual=[]
    if a.cases:ranks=[r for r in ranks if f"{r['side']}/{r['fragment']}" in a.cases.split(',')]
    cfg=CRBConfig(cover_density=.04,error_density=.02,split_density=.02,epsilon_policy='trajectory_p10')
    for rec in ranks:
        g=np.load(root/rec['dense_file'])['gradients'];n,d=g.shape;fid=rec['fragment'];frag=TimeSeriesFragment(np.zeros(n+575),fragment_id=fid)
        def run(c):return CRBEngine(c,lambda f,s:g[s-1]).process_fragment(frag)
        full=run(cfg);cover=run(dataclasses.replace(cfg,error_density=0,split_density=0,detection='none'));error=run(dataclasses.replace(cfg,split_density=0,detection='none'));drop=run(dataclasses.replace(cfg,detection='drop_only'));uniform=run(dataclasses.replace(cfg,error_placement='uniform'))
        k=full.computed_gradients;cap=sum(cfg.budgets(n));uni=np.rint(np.linspace(0,n-1,k)).astype(int);uni_cap=np.rint(np.linspace(0,n-1,cap)).astype(int)
        hs={name:interpolate_window_gradients(r) for name,r in [('cover',cover),('error',error),('rags',full),('rags_uniform',uniform)]}
        hs.update(uniform_equal_K=interp(g,uni),uniform_full_cap=interp(g,uni_cap),trajectory_oracle=np.broadcast_to(g.mean(axis=0),(n,d)))
        anchor_union=set(uni)|set(uni_cap)
        for r in (cover,error,full,uniform):anchor_union.update(r.anchors)
        held=np.ones(n,bool);held[list(anchor_union)]=False
        mm={key:dict(all=metrics(g,h,np.ones(n,bool)),heldout=metrics(g,h,held)) for key,h in hs.items()}
        split=sorted(set(full.anchors)-set(error.anchors));new_error=sorted(set(error.anchors)-set(cover.anchors))
        before=sorted(error.anchors);splits=[]
        for s in split:
            i=np.searchsorted(before,s);l,r=before[i-1:i+1];mask=np.zeros(n,bool);mask[l:r+1]=True;mask&=held
            parent=(g[l].astype(float)+g[r])/2
            children=[]
            for b in full.blocks:
                if l<=b.left_index and b.right_index<=r:
                    truth=g[b.start_positions-1].mean(axis=0);children.append(dict(left=b.left_index,right=b.right_index,windows=b.size,true_mean_norm=float(np.linalg.norm(truth)),direction_cosine=float(unit(truth)@b.direction)))
            splits.append(dict(left=l,right=r,mid=s,endpoint_cosine=float(unit(g[l])@unit(g[r])),parent_mean_norm=float(np.linalg.norm(parent)),before=metrics(g,hs['error'],mask),after=metrics(g,hs['rags'],mask),children=children))
        report=dict(**rec,metrics=mm,uniform_variant_used=[uniform.cover_computations,uniform.error_computations,uniform.split_computations],heldout_rule='Exclude union of all compared anchor sets',split_events=splits,retained_before_split=sum(b.size for b in drop.blocks),retained_after_split=sum(b.size for b in full.blocks))
        reports.append(report)
        # Same fixed display coordinate for every method, never used for decisions.
        v=np.random.default_rng(0).normal(size=d);v=unit(v)
        for _ in range(30):v=unit(g.T@(g@v))
        pc=g@v;energy=float(np.sum(pc**2)/np.sum(g.astype(float)**2))
        stem=f"{rec['side']}-{fid}"
        visual.append(dict(fragment=f"{rec['side']} {fid}",starts=n,anchors=k,displayed_relative_error=mm['rags']['heldout']['relative_l2'],representation=dict(retained_energy_fraction=energy),curve=[dict(point=i+1,actual=float(pc[i]),estimated=float(hs['rags'][i]@v),full_l2_norm=float(np.linalg.norm(g[i]))) for i in range(n)]))
        # Full-domain views plus a split-rich fixed-width view selected on anchor counts only.
        width=min(240,n); scores=np.convolve(np.isin(np.arange(n),split).astype(int),np.ones(width,dtype=int),'valid');left=int(np.argmax(scores));right=left+width
        fig,axs=plt.subplots(4,1,figsize=(15,12),constrained_layout=True)
        colors={'cover':'#888888','error':'#f0a202','rags':'#228877','uniform_equal_K':'#bb5566','trajectory_oracle':'#332288'}
        for ax,(lo,hi,title) in zip(axs[:2],[(0,n,'Full trajectory'),(left,right,'Split-rich crop (selected by split count, not error gain)')]):
            xx=np.arange(lo,hi)+1;ax.plot(xx,pc[lo:hi],color='#4477aa',lw=1,alpha=.8,label='Actual window gradient')
            for name in ('cover','error','rags','uniform_equal_K','trajectory_oracle'):ax.plot(xx,(hs[name]@v)[lo:hi],color=colors[name],lw=1.2,label=name)
            ax.set(title=title,ylabel='Signed PC1 coordinate');ax.legend(ncol=3,fontsize=8)
        for name in ('cover','error','rags','uniform_equal_K'):
            err=np.linalg.norm(g-hs[name],axis=1);axs[2].plot(np.arange(left,right)+1,err[left:right],color=colors[name],label=name)
        axs[2].set(ylabel='4096-D L2 error',title='Reconstruction error in the same crop');axs[2].legend(ncol=4)
        for yy,points,col,label in [(0,cover.anchors,'#888888','Coverage'),(1,new_error,'#f0a202','Error'),(2,split,'#7b2cbf','Split')]:axs[3].scatter(np.array(list(points))+1,np.full(len(points),yy),s=12,color=col,label=label)
        axs[3].set(yticks=[0,1,2],yticklabels=['Coverage','Error','Split'],xlabel='Legal physical window start',title=f'Used {rec["used"]} / allocated {rec["budgets"]}; epsilon={rec["epsilon"]:.4f}')
        fig.suptitle(f'TimeMoE-200M | {stem} | PC1 energy {energy:.1%} | current trajectory P10 / 4:2:2',fontsize=15)
        fig.savefig(out/(stem+'.png'),dpi=130);fig.savefig(out/(stem+'.svg'));plt.close(fig)
        bf,mask=blockfield(full,n,d);uf,umask=blockfield(uniform,n,d);agg=aggregate_retained(full)
        assets.append(dict(rec=rec,g=g,rags=bf,mask=mask,uniform=uf,umask=umask,traj=unit(agg.mean_gradient),oracle=unit(g.mean(axis=0))))
        print(stem,'K',k,'split',len(split),'L2',{x:round(y['heldout']['relative_l2'],4) for x,y in mm.items()},flush=True)
    save(out/'reconstruction.json',reports);save(out/'grad_reconst_input.json',visual)
    if a.skip_retrieval:return
    # Retrieval audit: deterministic 128 query windows per full reference, all four source trajectories.
    sources=[x for x in assets if x['rec']['side']=='source'];refs=[x for x in assets if x['rec']['side']=='reference']
    sg=np.concatenate([unit(x['g']) for x in sources]);sb=np.concatenate([x['rags'] for x in sources]);su=np.concatenate([x['uniform'] for x in sources]);st=np.concatenate([np.broadcast_to(x['traj'],x['g'].shape) for x in sources]);so=np.concatenate([np.broadcast_to(x['oracle'],x['g'].shape) for x in sources]);sm=np.concatenate([x['mask'] for x in sources]);um=np.concatenate([x['umask'] for x in sources]);retr=[]
    for ref in refs:
        queries=np.unique(np.rint(np.linspace(0,len(ref['g'])-1,128)).astype(int));queries=queries[ref['mask'][queries]&ref['umask'][queries]]
        truth=unit(ref['g'][queries])@sg.T
        predictions={'rags_blocks':ref['rags'][queries]@sb.T,'uniform_blocks':ref['uniform'][queries]@su.T,'rags_traj':np.broadcast_to(ref['traj']@st.T,truth.shape),'oracle_full_trajectory_mean':np.broadcast_to(ref['oracle']@so.T,truth.shape)}
        # Common support isolates averaging from filtering; ties use expected tie fractions.
        common=sm&um; t=truth[:,common]; detail={}; k=max(1,round(t.shape[1]*.1)); threshold=np.partition(t,-k,axis=1)[:,-k];target=t>=threshold[:,None]
        for name,pred in predictions.items():
            pp=pred[:,common];positive=pp>0; mass=np.where(positive,np.exp(pp),0);mass/=np.maximum(mass.sum(axis=1,keepdims=True),1e-30)
            cut=np.partition(pp,-k,axis=1)[:,-k];above=pp>cut[:,None];ties=pp==cut[:,None];w=above.astype(float)+ties*((k-above.sum(axis=1))/np.maximum(1,ties.sum(axis=1)))[:,None]
            detail[name]=dict(cosine_MAE=float(np.mean(abs(pp-t))),sign_disagreement=float(np.mean((pp>0)!=(t>0))),top10_recall_tie_expected=float(np.mean(np.sum(w*target,axis=1)/target.sum(axis=1))),negative_true_cosine_probability_mass=float(np.mean(np.sum(mass*(t<0),axis=1))),expected_true_cosine=float(np.mean(np.sum(mass*t,axis=1))))
        row=dict(reference=ref['rec']['fragment'],queries=(queries+1).tolist(),source_windows=int(common.sum()),metrics=detail);retr.append(row);print('RETRIEVAL',json.dumps(row['metrics']),flush=True)
    save(out/'retrieval.json',dict(protocol='128 evenly spaced queries before common retained-support filtering; all source windows on common RAGS/uniform retained support; theta=1 positive-cosine weights; tie-aware top10; diagnostic only',records=retr))
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True);names=list(retr[0]['metrics']);labels=['RAGS blocks','Uniform blocks','RAGS trajectory','Full-scan mean']
    for ax,key,title in zip(axs,['cosine_MAE','sign_disagreement','top10_recall_tie_expected'],['Cosine score MAE (lower)','Sign disagreement (lower)','Top-10% recall (higher)']):
        vals=[np.mean([r['metrics'][name][key] for r in retr]) for name in names];ax.bar(labels,vals,color=['#228877','#bb5566','#332288','#999999']);ax.set_title(title);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Real ref-to-source retrieval | two references / four sources | common retained support');fig.savefig(out/'retrieval.png',dpi=150);fig.savefig(out/'retrieval.svg');plt.close(fig)

if __name__=='__main__':main()
