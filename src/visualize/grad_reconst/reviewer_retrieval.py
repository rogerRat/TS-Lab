"""Retrieval verification on non-anchor windows with an explicit sign-change example."""
import argparse,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rage.crb import CRBConfig,CRBEngine
from rage.core import TimeSeriesFragment
from assets.gradient.variants import aggregate_retained
from dataclasses import replace
from visualize.grad_reconst.reviewer_compare import unit,blockfield
from visualize.grad_reconst.reviewer_audit import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--dense',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=pathlib.Path(a.dense);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assets=[];cfg=CRBConfig(cover_density=.04,error_density=.02,split_density=.02,epsilon_policy='trajectory_p10')
    for rec in json.loads((root/'ranking.json').read_text()):
        g=np.load(root/rec['dense_file'])['gradients'];n,d=g.shape;f=TimeSeriesFragment(np.zeros(n+575),fragment_id=rec['fragment']);run=lambda c:CRBEngine(c,lambda f,s:g[s-1]).process_fragment(f)
        r=run(cfg);u=run(replace(cfg,error_placement='uniform'));bf,mask=blockfield(r,n,d);uf,um=blockfield(u,n,d);mask&=um;mask[list(set(r.anchors)|set(u.anchors))]=False
        assets.append(dict(rec=rec,g=g,bf=bf,uf=uf,mask=mask,traj=unit(aggregate_retained(r).mean_gradient),oracle=unit(g.mean(axis=0))))
    src=[x for x in assets if x['rec']['side']=='source'];ref=[x for x in assets if x['rec']['side']=='reference'];common=np.concatenate([x['mask'] for x in src]);sg=np.concatenate([unit(x['g']) for x in src])[common];sb=np.concatenate([x['bf'] for x in src])[common];su=np.concatenate([x['uf'] for x in src])[common];st=np.concatenate([np.broadcast_to(x['traj'],x['g'].shape) for x in src])[common];so=np.concatenate([np.broadcast_to(x['oracle'],x['g'].shape) for x in src])[common];records=[];example=None
    for rr in ref:
        qs=np.unique(np.rint(np.linspace(0,len(rr['g'])-1,128)).astype(int));qs=qs[rr['mask'][qs]]
        truth=unit(rr['g'][qs])@sg.T;preds={'rage_blocks':rr['bf'][qs]@sb.T,'uniform_blocks':rr['uf'][qs]@su.T,'rage_traj':np.broadcast_to(rr['traj']@st.T,truth.shape),'full_scan_mean':np.broadcast_to(rr['oracle']@so.T,truth.shape)}
        k=max(1,round(truth.shape[1]*.1));tt=np.partition(truth,-k,axis=1)[:,-k];target=truth>=tt[:,None];mm={}
        for name,pp in preds.items():
            cut=np.partition(pp,-k,axis=1)[:,-k];above=pp>cut[:,None];ties=pp==cut[:,None];w=above+ties*((k-above.sum(axis=1))/np.maximum(1,ties.sum(axis=1)))[:,None]
            mass=np.where(pp>0,np.exp(pp),0);mass/=np.maximum(mass.sum(axis=1,keepdims=True),1e-30)
            mm[name]=dict(cosine_MAE=float(abs(pp-truth).mean()),sign_disagreement=float(np.mean((pp>0)!=(truth>0))),top10_recall=float(np.mean((w*target).sum(axis=1)/target.sum(axis=1))),negative_cosine_probability_mass=float(np.mean((mass*(truth<0)).sum(axis=1))))
        records.append(dict(reference=rr['rec']['fragment'],queries=(qs+1).tolist(),metrics=mm))
        # Choose example by true within-trajectory variability, independently of approximation errors.
        for ss in src:
            t=unit(rr['g'][qs])@unit(ss['g']).T;variation=np.quantile(t,.9,axis=1)-np.quantile(t,.1,axis=1);j=int(np.argmax(variation));score=float(variation[j])
            if example is None or score>example['score']:
                example=dict(score=score,ref=rr['rec']['fragment'],query=int(qs[j]+1),source=ss['rec']['fragment'],actual=t[j],rage=rr['bf'][qs[j]]@ss['bf'].T,uniform=rr['uf'][qs[j]]@ss['uf'].T,trajectory=float(rr['traj']@ss['traj']),full_scan=float(rr['oracle']@ss['oracle']),mask=ss['mask'])
    summary=dict(protocol='Current 4:2:2, trajectory P10. 128 predetermined even query starts/reference; remove RAGE and uniform anchor starts and rejected windows on both sides. Common source support. Exact cached 4096-D cosine truth. Uniform variant uses same phase caps, actual K differs. Top10 ties receive expected random tie allocation.',source_windows=int(common.sum()),records=records)
    save(out/'retrieval_heldout.json',summary)
    fig,axs=plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True);names=list(records[0]['metrics']);labels=['RAGE blocks','Uniform blocks','RAGE trajectory','Full-scan mean']
    for ax,key,title in zip(axs,['cosine_MAE','sign_disagreement','top10_recall'],['Cosine score MAE (lower)','Sign disagreement (lower)','Top-10% recall (higher)']):
        vals=[np.mean([r['metrics'][name][key] for r in records]) for name in names];bars=ax.bar(labels,vals,color=['#228877','#bb5566','#332288','#999999']);ax.bar_label(bars,fmt='%.3f');ax.set_title(title);ax.tick_params(axis='x',rotation=20);ax.set_ylim(0,max(vals)*1.18)
    fig.suptitle(f'Non-anchor ref-to-source retrieval | {sum(len(r["queries"]) for r in records)} queries, {common.sum()} source windows | common support');fig.savefig(out/'retrieval_heldout.png',dpi=150);fig.savefig(out/'retrieval_heldout.svg');plt.close(fig)
    ex=example;xx=np.arange(len(ex['actual']))+1
    fig,ax=plt.subplots(figsize=(14,4.8),constrained_layout=True);ax.plot(xx,ex['actual'],label='True window cosine',color='#4477aa',lw=1);ax.plot(xx,np.where(ex['mask'],ex['rage'],np.nan),label='RAGE blocks',color='#228877',lw=1);ax.axhline(ex['trajectory'],label='RAGE trajectory mean',color='#332288');ax.axhline(ex['full_scan'],label='Full-scan trajectory mean',color='#999999',ls='--');ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Source legal window start',ylabel='Cosine with reference query',title=f'Ref {ex["ref"]}, query {ex["query"]} -> source {ex["source"]}: opposite signs inside one trajectory');ax.legend(ncol=2);fig.savefig(out/'trajectory_sign_failure.png',dpi=160);fig.savefig(out/'trajectory_sign_failure.svg');plt.close(fig)
    for k in ('actual','rage','uniform','mask'):ex[k]=ex[k].tolist()
    ex['selection']='Largest true P90-P10 cosine range across predetermined held-out queries and four sources, independent of method errors.';save(out/'trajectory_sign_failure.json',ex)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
