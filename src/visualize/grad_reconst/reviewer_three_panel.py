"""Three-row full-length/200-point views using saved grad_reconst curves."""
import argparse,json,pathlib,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
def save(path, value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

COLORS=dict(truth='#4477aa',mean='#725496',uniform='#da8531',rage='#208477')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args();src=pathlib.Path(a.input);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    old=json.loads((src/'manifest.json').read_text());cases=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#bdc8d1','axes.titleweight':'semibold','axes.labelcolor':'#34495b','savefig.facecolor':'white'})
    for case in old['cases']:
        stem=case['stem'];data=json.loads((src/(stem+'.json')).read_text());c=data['curve'];x=np.asarray(c['starts']);truth=np.asarray(c['truth']);n=len(x)
        curves=dict(truth=truth,mean=np.full(n,c['mean']),uniform=np.asarray(c['uniform']),rage=np.asarray(c['rage']))
        residual={k:v-truth for k,v in curves.items() if k!='truth'}
        width=min(200,n)
        change=np.diff(truth)**2;cum=np.r_[0.,np.cumsum(change)];scores=cum[width-1:]-cum[:n-width+1];lo=int(np.argmax(scores));hi=lo+width
        fig=plt.figure(figsize=(18,10));gs=fig.add_gridspec(3,2,width_ratios=[4,1],height_ratios=[3,2.6,1.25],left=.066,right=.98,top=.86,bottom=.085,wspace=.065,hspace=.43)
        af=fig.add_subplot(gs[0,0]);az=fig.add_subplot(gs[0,1],sharey=af);bf=fig.add_subplot(gs[1,0],sharex=af);bz=fig.add_subplot(gs[1,1],sharex=az,sharey=bf);anchors=fig.add_subplot(gs[2,0],sharex=af);anchors_zoom=fig.add_subplot(gs[2,1],sharex=az,sharey=anchors)
        fig.suptitle('TimeMoE-200M  |  '+case['title'],y=.98,fontsize=17,fontweight='bold',color='#243444')
        fig.text(.5,.945,f'RAGE and uniform: K = {data["K"]:,} each  |  4% / 2% / 2%  |  shaded range: {lo+1:,}–{hi:,} ({width:,}/{n:,} starts)',ha='center',color='#586b7a')
        handles=[Line2D([],[],color=COLORS[k],lw=1.6,ls='--' if k=='mean' else '-',label=l) for k,l in [('truth','True gradient (B: zero line)'),('mean','Whole-trajectory mean'),('uniform','Uniform reconstruction (same K)'),('rage','RAGE: coverage + error + split')]]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.924),ncol=4,frameon=False,fontsize=10)
        for ax,left,right in [(af,0,n),(az,lo,hi)]:
            for key in ('truth','mean','uniform','rage'):
                ax.plot(x[left:right],curves[key][left:right],color=COLORS[key],ls='--' if key=='mean' else '-',lw=1.15 if key=='truth' else 1.4,alpha=.85 if key=='truth' else 1)
            ax.set_xlim(left+1,right);ax.grid(axis='y',alpha=.15)
        ylo=min(float(np.min(v)) for v in curves.values());yhi=max(float(np.max(v)) for v in curves.values());pad=max(float(yhi-ylo)*.04,1e-10);af.set_ylim(ylo-pad,yhi+pad)
        af.set_title('A  Gradient curves — full length',loc='left');az.set_title('A  Most variable 200 points',fontsize=10);af.set_ylabel('Signed gradient coordinate')
        for ax,left,right in [(bf,0,n),(bz,lo,hi)]:
            for key in ('mean','uniform','rage'):
                ax.plot(x[left:right],residual[key][left:right],color=COLORS[key],ls='--' if key=='mean' else '-',lw=1.1 if key=='mean' else 1.3)
            ax.axhline(0,color=COLORS['truth'],lw=1.25,zorder=5);ax.grid(axis='y',alpha=.15)
        bf.set_title('B  Signed error: estimate minus truth  |  closer to zero = more accurate',loc='left',fontsize=11);bz.set_title('B  Same 200-point interval',fontsize=10);bf.set_ylabel('Estimate − true gradient')
        lim=max(float(np.max(np.abs(v))) for v in residual.values())*1.04;bf.set_ylim(-max(lim,1e-10),max(lim,1e-10))
        for ax in (az,bz):ax.tick_params(labelleft=False);ax.xaxis.set_major_locator(MaxNLocator(3));ax.set_xlabel('Window start')
        bf.set_xlabel('Legal physical window start')
        for y,points,col in [(0,data['uniform_anchors'],COLORS['uniform']),(1,data['anchors']['coverage'],'#8c969f'),(2,data['anchors']['error'],'#d4ac47'),(3,data['anchors']['split'],COLORS['rage'])]:
            anchors.scatter(points,np.full(len(points),y),color=col,s=8,alpha=.8,linewidths=0)
            selected=[q for q in points if lo+1<=q<=hi]
            anchors_zoom.scatter(selected,np.full(len(selected),y),color=col,s=14,alpha=.8,linewidths=0)
        anchors.set(yticks=[0,1,2,3],yticklabels=['Uniform','RAGE cover','RAGE error','RAGE split'],ylim=(-.5,3.5),xlabel='Legal physical window start')
        anchors.set_title(f'C  Anchor placement — full length  |  used {data["case"]["used"]} / allocated {data["case"]["budgets"]}',loc='left',fontsize=10)
        anchors_zoom.set_title('C  Same 200-point interval',fontsize=10);anchors_zoom.tick_params(labelleft=False);anchors_zoom.set_xlabel('Window start');anchors_zoom.xaxis.set_major_locator(MaxNLocator(3))
        for ax in (af,bf,anchors):ax.axvspan(lo+1,hi,color='#6699bb',alpha=.10,zorder=0)
        fig.text(.5,.024,'B uses the same display coordinate as A: positive = overestimate, negative = underestimate; true gradient is exactly zero.\n200-point interval maximizes squared consecutive changes of the true displayed gradient. Full and zoom share y-scales covering all values, including extremes.',ha='center',fontsize=9,color='#586b7a')
        for ext in ('png','svg','pdf'):fig.savefig(out/(stem+'.'+ext),dpi=160)
        plt.close(fig)
        result=dict(schema='grad-reconst-three-panel-200-full-range-v4',source=str(src/(stem+'.json')),title=case['title'],K=data['K'],layout=dict(rows=['A gradients','B signed residual','C anchors'],width_ratio=[4,1],zoom_C=True,shared_y_scales=True),crop=dict(first=lo+1,last=hi,count=width,total=n,fraction=width/n,rule='min(200,N) starts; maximize sum of squared consecutive changes in true displayed PC1; earliest tie',change_score=float(scores[lo])),highlight=dict(color='#6699bb',alpha=.10),axis_limits=dict(A=[float(ylo-pad),float(yhi+pad)],B=[-max(lim,1e-10),max(lim,1e-10)],rule='A: min/max of all curves plus 4% margin; B: maximum absolute residual plus 4% margin; shared full/zoom; all extrema included'),curve=dict(starts=x.tolist(),**{k:v.tolist() for k,v in curves.items()}),signed_error={k:v.tolist() for k,v in residual.items()},anchors=data['anchors'],uniform_anchors=data['uniform_anchors'],note='Signed error is in the unchanged 1D display coordinate, not a 4096-D norm. No model calls, no refit, no new gradient estimates.')
        save(out/(stem+'.json'),result);cases.append(dict(stem=stem,title=case['title'],crop=result['crop']));print(stem,'done',flush=True)
    save(out/'manifest.json',dict(schema='grad-reconst-three-panel-200-full-range-v4',cases=cases,model_calls=0))
    sections=[]
    for case in cases:
        s=html.escape(case['stem']);title=html.escape(case['title']);sections.append(f'<section><h2>{title}</h2><p><a href="{s}.png">PNG</a> · <a href="{s}.svg">SVG</a> · <a href="{s}.pdf">PDF</a> · <a href="{s}.json">曲线与选区数据</a></p><a href="{s}.png"><img src="{s}.png" loading="lazy" alt="{title}"></a></section>')
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>九例 · 全长与200点放大</title><style>body{max-width:1700px;margin:30px auto;padding:0 20px;font:16px/1.8 system-ui;color:#243444;background:#fafbfd}img{width:100%}section{margin:40px 0}a{color:#26728a}p{max-width:1300px}</style><body><h1>九例：全长与变化最大的200点区间</h1><p>A保留真梯度、轨迹均值、均匀重建、完整RAGE四条线。B将真梯度拉平为蓝色零线，紫/橙/绿线分别表示三种估计减去真值；越靠近零越准，正为高估，负为低估。B是同一展示投影下的有符号误差，不是4096维误差范数。</p><p>每行左80%为全长，右20%为200点放大，A/B使用同一区间；左右纵轴尺度一致；A按全部曲线的最小值/最大值缩放，B按全部误差的最大绝对值对称缩放，均加4%边距。全部极端值纳入显示范围。浅蓝色高透明度色块标出放大区间。区间长度固定为200个合法起点（不足200时取全长），按真梯度展示曲线相邻差的平方和最大选择，与方法优劣无关。C也采用相同8:2布局，右侧显示同一区间的锚点；三行左右两列横轴分别对齐。</p>'+''.join(sections)+'</body></html>')

if __name__=='__main__':main()
