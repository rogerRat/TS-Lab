"""One explicit real split witness and a fixed-cost error-phase comparison."""
import argparse,json,pathlib
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rage.crb import CRBConfig,CRBEngine
from rage.core import TimeSeriesFragment
from rage.window_reconstruction import interpolate_window_gradients
from visualize.grad_reconst.reviewer_compare import metrics,unit,interp
from visualize.grad_reconst.reviewer_audit import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--dense',required=True);p.add_argument('--out',required=True);p.add_argument('--side',default='reference');p.add_argument('--fragment',type=int,default=1000);a=p.parse_args();root=pathlib.Path(a.dense);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rec=next(r for r in json.loads((root/'ranking.json').read_text()) if r['side']==a.side and r['fragment']==a.fragment);g=np.load(root/rec['dense_file'])['gradients'];n=len(g);f=TimeSeriesFragment(np.zeros(n+575),fragment_id=a.fragment)
    c=CRBConfig(cover_density=.04,error_density=.02,split_density=.02,epsilon_policy='trajectory_p10');run=lambda cc:CRBEngine(cc,lambda f,s:g[s-1]).process_fragment(f)
    r=run(c);e=run(replace(c,split_density=0,detection='none'));u=run(replace(c,split_density=0,detection='none',error_placement='uniform'));he=interpolate_window_gradients(e);hr=interpolate_window_gradients(r);hu=interpolate_window_gradients(u)
    held=np.ones(n,bool);held[list(set(r.anchors)|set(u.anchors))]=False
    uniform_points=np.rint(np.linspace(0,n-1,e.computed_gradients)).astype(int);hl=interp(g,uniform_points);phase_held=held.copy();phase_held[uniform_points]=False
    phase=dict(risk=metrics(g,he,phase_held),uniform=metrics(g,hu,phase_held),uniform_linspace=metrics(g,hl,phase_held),K=e.computed_gradients,uniform_K=u.computed_gradients)
    fig,axs=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
    axs[0].plot(np.arange(n)+1,np.linalg.norm(g,axis=1),label='Actual norm',color='#4477aa',lw=1)
    for h,label,col in [(he,'Risk allocation','#228877'),(hu,'Uniform longest-interval','#bb5566'),(hl,'Uniform linspace','#999999')]:
        axs[0].plot(np.arange(n)+1,np.linalg.norm(h,axis=1),label=label,color=col,lw=1)
        axs[1].plot(np.arange(n)+1,np.where(phase_held,np.linalg.norm(g-h,axis=1),np.nan),label=label,color=col,lw=1)
    axs[0].set(ylabel='Gradient norm',title=f'{a.side} {a.fragment} | error-phase comparison: all methods K={e.computed_gradients}');axs[0].legend(ncol=2)
    axs[1].set(ylabel='Non-anchor L2 error',xlabel='Physical window start',title='Risk %.3f | uniform longest %.3f | uniform linspace %.3f (held-out relative L2)'%(phase['risk']['relative_l2'],phase['uniform']['relative_l2'],phase['uniform_linspace']['relative_l2']));axs[1].legend(ncol=3)
    fig.savefig(out/'error_phase.png',dpi=150);fig.savefig(out/'error_phase.svg');plt.close(fig)
    # Match the stricter reconstruction table's union of every compared anchor set.
    uf=run(replace(c,error_placement='uniform'));witness_held=held.copy()
    witness_held[list(uf.anchors)]=False
    for count in (r.computed_gradients,sum(c.budgets(n))):
        witness_held[np.rint(np.linspace(0,n-1,count)).astype(int)]=False
    aa=sorted(e.anchors);events=[]
    for mid in sorted(set(r.anchors)-set(e.anchors)):
        j=np.searchsorted(aa,mid);l,rr=aa[j-1:j+1];mask=witness_held.copy();mask[:l]=False;mask[rr+1:]=False
        if not mask.any():continue
        m1=metrics(g,he,mask);m2=metrics(g,hr,mask);events.append(dict(left=l,right=rr,mid=mid,before=m1,after=m2,gain=m1['relative_l2']-m2['relative_l2']))
    ex=max(events,key=lambda x:x['gain']);l,rr,mid=ex['left'],ex['right'],ex['mid'];xx=np.arange(l,rr+1)+1;eps=r.epsilon_audit['epsilon'];fig,axs=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    axs[0].plot(xx,np.linalg.norm(g[l:rr+1],axis=1),'o-',ms=3,label='Actual window gradient norm',color='#4477aa');axs[0].plot(xx,np.linalg.norm(he[l:rr+1],axis=1),label='Before split',color='#f0a202');axs[0].plot(xx,np.linalg.norm(hr[l:rr+1],axis=1),label='After one midpoint probe',color='#228877');axs[0].axhline(eps,color='#7b2cbf',ls='--',label=f'Frozen epsilon = {eps:.3f}');axs[0].axvline(mid+1,color='#7b2cbf',lw=.8);axs[0].set(ylabel='4096-D gradient L2 norm',title=f'{a.side} {a.fragment}: parent would be rejected; one real midpoint reveals signal');axs[0].legend()
    axs[1].plot(xx,np.linalg.norm(g[l:rr+1]-he[l:rr+1],axis=1),label='Before split',color='#f0a202');axs[1].plot(xx,np.linalg.norm(g[l:rr+1]-hr[l:rr+1],axis=1),label='After split',color='#228877');axs[1].set(xlabel='Physical window start',ylabel='4096-D reconstruction error',title=f'Held-out relative error {ex["before"]["relative_l2"]:.3f} -> {ex["after"]["relative_l2"]:.3f}; {ex["before"]["n"]} non-anchor windows');axs[1].legend()
    fig.suptitle('Real split witness (selected for largest held-out error reduction among split events)',fontsize=12);fig.savefig(out/'split_witness.png',dpi=160);fig.savefig(out/'split_witness.svg');plt.close(fig)
    blocks=[]
    for b in r.blocks:
        if l<=b.left_index and b.right_index<=rr:
            true=g[b.start_positions-1].mean(axis=0);blocks.append(dict(start=int(b.start_positions[0]),end=int(b.start_positions[-1]),windows=b.size,true_mean_norm=float(np.linalg.norm(true)),cosine=float(unit(true)@b.direction)))
    ex.update(epsilon=eps,endpoint_cosine=float(unit(g[l])@unit(g[rr])),parent_mean_norm=float(np.linalg.norm((g[l].astype(float)+g[rr])/2)),retained_children=blocks,selection='Largest error reduction among actual split events; illustrative selected witness, not average effect',indices='left/right/mid are zero-based; figure uses physical starts = index+1')
    save(out/'split_witness.json',dict(record=rec,witness=ex,events=events,error_phase=phase));print(json.dumps(phase),flush=True)

if __name__=='__main__':main()
