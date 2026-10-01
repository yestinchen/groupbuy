"""Explicit fresh scalability sources: catalog/length sweeps, length-budget
tiers, and the large-catalog ladders, drawn with the manuscript's own panel
code but read only from validated revision records."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from gquery.figures import plot_exp4 as plots

REPO=Path(__file__).resolve().parents[2]
CAMPAIGN=Path(os.environ.get('CAMPAIGN_ROOT',REPO/'results'/'campaign'))
FIGURES=Path(os.environ.get('FIGURES_ROOT',REPO/'data_figures'/'exp7'))
MEMORY_GRIDS=Path(os.environ.get('MEMORY_GRIDS_ROOT',REPO/'results'/'memory_grids'))
ROOT=Path(os.environ.get('CAMPAIGN_SCALABILITY_ROOT',CAMPAIGN/'scalability'))
OUT=Path(os.environ.get('CAMPAIGN_SCALABILITY_OUT',FIGURES/'exp7_scale'))
FILES={'scale':'scale.csv','length_budget':'length_budget.csv','xl':'scale_xl.csv'}
SOURCES={(d,k):ROOT/f'{d}_{k}'/FILES[k] for d in ['amazon','airbnb'] for k in FILES}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Separate ladder jobs: the recalibrated 330K and 1.1M rungs (all arms) and the candidate-cap
# sweep at the recalibrated 2.14M rung (only the k=10, 25, 100 arms; its base arms duplicate
# amazon_xl and are excluded).
# Depth-first policy jobs supply every PTS series (a 'PTS-DFS' entry keeps every method with that prefix); present() renames them to PTS.
EXTRA={('amazon','xl'):[(ROOT/'amazon_xl_recal'/'scale_xl.csv',None),(ROOT/'amazon_xl_kappa'/'scale_xl.csv',('PTS k=10','PTS k=25','PTS k=100')),
                        (ROOT/'amazon_xl_dfs'/'scale_xl.csv','PTS-DFS'),(ROOT/'amazon_xl_recal_dfs'/'scale_xl.csv','PTS-DFS'),(ROOT/'amazon_xl_kappa_dfs'/'scale_xl.csv',('PTS-DFS k=10','PTS-DFS k=25','PTS-DFS k=100')),
                        (ROOT/'amazon_xl_dpconv'/'scale_xl.csv','Pareto-DP'),  # conventional Pareto DP (no candidate reduction), measured alone on the presented rungs
                        (ROOT/'amazon_xl_pbf60'/'scale_xl.csv','PBF')],  # PBF rerun with a 60 s limit, a 16 GiB memory growth limit and no combination guard; replaces the PBF rows
       ('airbnb','xl'):[(ROOT/'airbnb_xl_dfs'/'scale_xl.csv','PTS-DFS'),(ROOT/'airbnb_xl_dpconv'/'scale_xl.csv','Pareto-DP')],
       ('amazon','scale'):[(ROOT/'amazon_scale_dfs'/'scale.csv','PTS-DFS')],('airbnb','scale'):[(ROOT/'airbnb_scale_dfs'/'scale.csv','PTS-DFS')],
       ('amazon','length_budget'):[(ROOT/'amazon_length_budget_dfs'/'length_budget.csv','PTS-DFS')],('airbnb','length_budget'):[(ROOT/'airbnb_length_budget_dfs'/'length_budget.csv','PTS-DFS')]}

# PBS arms rerun with the reachability bound in FeasibleExtensions (2026-10-01). Their rows replace every PBS row
# (bounded PBS and PBS without beam truncation) of the jobs above; the *_xl*_reach jobs measure only these arms, the
# scale and length-budget reach jobs rerun the whole runner and contribute only their PBS rows. Required unless
# PBS_REACH_OPTIONAL=1 (only for testing the builder before the jobs finish).
PBS_METHODS=('PACS','PACS unbounded')
PBS_REACH={('amazon','xl'):[ROOT/'amazon_xl_reach'/'scale_xl.csv',ROOT/'amazon_xl_recal_reach'/'scale_xl.csv'],
           ('airbnb','xl'):[ROOT/'airbnb_xl_reach'/'scale_xl.csv'],
           ('amazon','scale'):[ROOT/'amazon_scale_reach'/'scale.csv'],('airbnb','scale'):[ROOT/'airbnb_scale_reach'/'scale.csv'],
           ('amazon','length_budget'):[ROOT/'amazon_length_budget_reach'/'length_budget.csv'],
           ('airbnb','length_budget'):[ROOT/'airbnb_length_budget_reach'/'length_budget.csv']}

def read_one(path,kind,methods=None):
    proof=json.loads((path.parent/'reconstruction.json').read_text())
    assert proof['files'][path.name]==sha(path) and proof['kind']==('scale_xl' if kind=='xl' else kind)
    df=pd.read_csv(path);assert len(df)==proof['rows'];df['source_file']=str(path)
    if methods is None:return df
    return df[df.method.str.startswith(methods)] if isinstance(methods,str) else df[df.method.isin(methods)]

def read(dataset,kind):
    # the 60 s PBF rerun is required unless PBF60_OPTIONAL=1 (only for testing before the job finishes)
    assert dataset!='amazon' or kind!='xl' or (ROOT/'amazon_xl_pbf60'/'reconstruction.json').exists() or os.environ.get('PBF60_OPTIONAL')=='1','missing validated amazon_xl_pbf60'
    extras=[(p,m) for p,m in EXTRA.get((dataset,kind),[]) if (p.parent/'reconstruction.json').exists()]
    parts=[read_one(SOURCES[(dataset,kind)],kind)]+[read_one(p,kind,m) for p,m in extras if m!='PBF']
    pbf=[read_one(p,kind,m) for p,m in extras if m=='PBF']
    if pbf:   # a validated PBF rerun replaces the PBF rows of every other job
        parts=[d[d.method!='PBF'] for d in parts]+pbf
    reach=PBS_REACH.get((dataset,kind),[])
    if os.environ.get('PBS_REACH_OPTIONAL')=='1':
        reach=[p for p in reach if (p.parent/'reconstruction.json').exists()]
    if reach:   # validated PBS reruns replace the PBS rows of every other job, setting by setting
        pbs=pd.concat([read_one(p,kind,PBS_METHODS) for p in reach],ignore_index=True)
        old=pd.concat([d[d.method.isin(PBS_METHODS)] for d in parts],ignore_index=True)
        covered=set(map(tuple,pbs[['setting','method']].drop_duplicates().values))
        if os.environ.get('PBS_REACH_OPTIONAL')!='1':
            assert covered==set(map(tuple,old[['setting','method']].drop_duplicates().values)),'PBS reruns do not cover the presented PBS rows'
        parts=[d[~(d.method.isin(PBS_METHODS)&d.setting.isin(pbs.setting.unique()))] for d in parts]+[pbs]
    df=present(pd.concat(parts,ignore_index=True))
    key=['setting','method','query_id']+[c for c in ('beam','expansion_budget') if c in df.columns]  # length-budget rows repeat a query per search budget
    assert not df.duplicated(key).any(),'overlapping settings between ladder jobs'
    return df

def present(frame):
    """Manuscript presentation: PTS is the depth-first policy. UCT rows (runner labels PTS, PTS uncapped, PTS k=...)
    are dropped and the depth-first rows (PTS-DFS...) take the PTS names; the raw validated records keep both."""
    if not frame.method.str.startswith('PTS-DFS').any():return frame.copy()   # fresh job: PTS rows are depth first already
    uct=frame.method.str.match(r'^PTS(?:$| |-)')&~frame.method.str.startswith('PTS-DFS')
    frame=frame[~uct].copy();frame['method']=frame.method.str.replace('PTS-DFS','PTS',regex=False);return frame

def with_source(frame):
    """Route the manuscript panel code to one explicit validated frame."""
    class _Reader:
        def __init__(self,frame):self.frame=frame
        def __call__(self,src,name):return self.frame.copy()
    return _Reader(frame)

def denominators(df,keys,out):
    rows=[]
    for key,g in df.groupby(keys,sort=False):
        row=dict(zip(keys,key if isinstance(key,tuple) else (key,)))
        eligible=g[g.metric_eligible==1]
        row.update(n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=len(eligible),n_returned=int(g.success.sum()),n_failure=int((g.solver_status=='solver_failure').sum()),n_timeout=int(g.timed_out.fillna(0).sum()),n_timed=int(g.method_wall_s.notna().sum()),
                   recall_median=eligible.recall.median(),recall_mean=eligible.recall.mean(),hv_median=eligible.hv_ratio.median(),total_s_median=g.total_s.median(),method_wall_s_median=g.method_wall_s.median(),retrieval_s_median=g.retrieval_s.median(),n_candidates_median=g.n_candidates.median())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out,index=False)


# The manuscript's catalog-time panel puts its legend in the upper left corner, where the PTS point at 10K items sits.
_band_panel_orig=plots._band_panel
def _band_panel_override(exp_name,folder,stem,*args,**kwargs):
    if stem=='scale_dist_catalog_time':kwargs['legend']='lower right'
    return _band_panel_orig(exp_name,folder,stem,*args,**kwargs)
plots._band_panel=_band_panel_override

def dp_table(frames):
    """Supplementary table: the conventional Pareto DP against PBS without beam truncation per catalog rung."""
    rows=[];lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'catalog & cand. & \multicolumn{4}{c}{DP} & \multicolumn{2}{c}{PBS unbounded} \\',r'\cmidrule(lr){3-6}\cmidrule(lr){7-8}',r' & median & median & p90 & max & unfinished & median & p90 \\',r'\midrule']
    fmt=lambda s:(f'${s*1000:.1f}$\\,ms' if s<1 else f'${s:.1f}$\\,s')
    for dataset,ladder,label in [('amazon',plots.XL_LADDER,'{}'),('airbnb',plots.AIRBNB_LADDER,'Airbnb {}')]:
        df=frames.get(dataset)
        if df is None or not df.method.eq('Pareto-DP').any():continue
        for setting,lab,n in ladder:
            g=df[(df.setting==setting)&(df.method=='Pareto-DP')];u=df[(df.setting==setting)&(df.method=='PACS unbounded')]
            if g.empty:continue
            unfinished=int(g.timed_out.fillna(0).sum())
            rows.append(dict(dataset=dataset,setting=setting,catalog=lab,candidates_median=float(g.n_before.median()),dp_median_s=float(g.total_s.median()),dp_p90_s=float(g.total_s.quantile(.9)),dp_max_s=float(g.total_s.max()),dp_unfinished=unfinished,dp_recall_mean=float(g[g.metric_eligible==1].recall.mean()),pbs_unbounded_median_s=float(u.total_s.median()),pbs_unbounded_p90_s=float(u.total_s.quantile(.9))))
            lines.append(' & '.join([label.format(lab),f'{int(g.n_before.median()):,}',fmt(g.total_s.median()),fmt(g.total_s.quantile(.9)),fmt(g.total_s.max()),f'{unfinished} / {len(g)}',fmt(u.total_s.median()),fmt(u.total_s.quantile(.9))])+r' \\')
    if not rows:return
    lines+=[r'\bottomrule',r'\end{tabular}']
    (OUT/'table_scale_dp.tex').write_text('% generated by scripts/exp7/build_scalability.py: conventional Pareto DP (no candidate reduction) against PBS without beam truncation per catalog rung; times are fresh retrieval plus method wall time over the 100 queries, unfinished counts queries reaching the 60 s limit, candidates are the median retrieved count per query before reduction\n'+'\n'.join(lines)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'scale_dp_summary.csv',index=False)

def xl_panels(df,folder,prefix,ladder):
    """Latency and recall over the ladder for PBS, PTS (depth first), unbounded PBS, and retrieval."""
    def _n(s,fallback):
        v=df[df.setting==s].catalog_items.dropna();return int(v.iloc[0]) if len(v) else fallback
    order=[(s,lbl,_n(s,n)) for s,lbl,n in ladder if s in set(df.setting)];labs=[l for _,l,_ in order];ns=[n for _,_,n in order]
    fig,ax=plots.paper_panel();rows=[]
    for meth,lbl,ls in [('PACS','PBS','-'),('PTS','PTS','-'),('PACS unbounded','PBS unbounded','--'),('Pareto-DP','DP','--')]:
        if meth not in set(df.method):continue
        vals=[df[(df.setting==s)&(df.method==meth)].total_s.median()*1000 for s,_,_ in order]
        ax.plot(ns,vals,marker='o',linestyle=ls,color=plots.color(meth),label=lbl);rows+=[dict(setting=s,method=meth,e2e_median_ms=float(v)) for (s,_,_),v in zip(order,vals)]
    ret=[df[(df.setting==s)&(df.method=='PTS')].retrieval_s.median()*1000 for s,_,_ in order]
    ax.plot(ns,ret,':',marker='s',color='#888888',label='retrieval');rows+=[dict(setting=s,method='retrieval',e2e_median_ms=float(v)) for (s,_,_),v in zip(order,ret)]
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xticks(ns);ax.set_xticklabels(labs);ax.xaxis.set_minor_formatter(plots.plt.NullFormatter())
    allv=[r['e2e_median_ms'] for r in rows if np.isfinite(r['e2e_median_ms'])];lo,hi=(min(allv),max(allv)) if allv else (1,20)
    ticks=[t for t in (0.5,1,2,5,10,20,50,100,200) if lo/1.6<=t<=hi*1.6];ticks=ticks[::2] if len(ticks)>5 else ticks  # at most five labels on the short axis
    ax.set_yticks(ticks);ax.yaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'));ax.yaxis.set_minor_formatter(plots.plt.NullFormatter())
    ax.set_xlabel('catalog items');ax.set_ylabel('median time (ms)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=26,columnspacing=.8,handlelength=1.3)  # above the axes in two rows (five series): the retrieval markers occupy every lower corner
    plots.save_panel(fig,str(OUT),folder,f'{prefix}_latency',pd.DataFrame(rows))
    fig,ax=plots.paper_panel();rows=[]
    for meth,lbl in [('PACS','PBS'),('PTS','PTS')]:
        if meth not in set(df.method):continue
        med,lo,hi=[],[],[]
        for s,_,n in order:
            g=df[(df.setting==s)&(df.method==meth)&(df.metric_eligible==1)];m_,l_,h_=plots._quartiles(g,'recall');med.append(m_);lo.append(l_);hi.append(h_)
            rows.append(dict(method=meth,catalog_items=n,recall_median=float(m_),q25=float(l_),q75=float(h_),recall_mean=float(g.recall.mean()),n_metric=len(g)))
        plots._band(ax,np.array(ns),np.array(med),np.array(lo),np.array(hi),plots.color(meth),lbl)
    ax.set_xscale('log');ax.set_xticks(ns);ax.set_xticklabels(labs);ax.xaxis.set_minor_formatter(plots.plt.NullFormatter())
    ax.set_xlabel('catalog items');ax.set_ylabel('frontier recall');ax.set_ylim(-.03,1.05);ax.legend(loc='lower left' if folder=='amazon' else 'upper right')
    plots.save_panel(fig,str(OUT),folder,f'{prefix}_recall',pd.DataFrame(rows))

def main():
    OUT.mkdir(parents=True,exist_ok=True);plots.OUT_ROOT=OUT.parent;original=plots.read
    plots.COLORS.update({'Pareto-DP':'#9673A6'});xl_frames={}
    try:
        for dataset in ['amazon','airbnb']:
            folder=dataset
            scale=read(dataset,'scale');plots.read=with_source(scale)
            if dataset=='amazon':
                plots.fig_scale_dist(str(OUT),src='validated',folder=folder,prefix='scale_dist',sweeps=('catalog','length'))
            else:
                plots.fig_scale_dist(str(OUT),src='validated',folder=folder,prefix='scale_dist_length',sweeps=('length',))  # single sweep: the panel code omits the sweep tag, so the prefix carries it
            denominators(scale,['sweep','setting','param','method'],OUT/f'{dataset}_scale_denominators.csv')
            length=read(dataset,'length_budget');plots.read=with_source(length)
            plots.fig_length_budget(str(OUT),src='validated',folder=folder,prefix='length_budget')
            denominators(length.assign(tier=length.beam.fillna(length.expansion_budget)),['setting','n','method','tier'],OUT/f'{dataset}_length_budget_denominators.csv')
            xl=read(dataset,'xl');plots.read=with_source(xl);xl_frames[dataset]=xl
            if dataset=='amazon':
                plots.fig_scale_xl(str(OUT),src='validated',folder=folder,prefix='scale_xl');xl_panels(xl,folder,'scale_xl',plots.XL_LADDER)
            else:
                plots.fig_scale_xl(str(OUT),src='validated',ladder=plots.AIRBNB_LADDER,fixed=plots.AIRBNB_FIXED,top_setting='airbnb_lad231k_recal',folder=folder,prefix='scale_xl',recall_legend='upper right',tightness_legend_above=True)
                xl_panels(xl,folder,'scale_xl',plots.AIRBNB_LADDER)
            denominators(xl,['setting','method'],OUT/f'{dataset}_xl_denominators.csv')
    finally:
        plots.read=original
    dp_table(xl_frames)
    code=REPO;diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    (OUT/'generation_source.diff').write_bytes(diff)
    manifest=dict(builder_sha256=sha(__file__),presentation='PTS is the depth-first policy: every PTS series (including uncapped and capped arms) comes from the *_dfs jobs and the UCT rows of the base jobs are excluded from every panel; both remain in the raw records',generation_command=sys.executable+' '+str(Path(__file__).resolve()),
        helpers={str(code/'gquery/figures/plot_exp4.py'):sha(code/'gquery/figures/plot_exp4.py')},source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
        sources={f'{d}.{k}':dict(path=str(p),sha256=sha(p),reconstruction_sha256=sha(p.parent/'reconstruction.json')) for (d,k),p in list(SOURCES.items())+[((d,k+'_extra_'+p.parent.name),p) for (d,k),ps in EXTRA.items() for p,_ in ps if p.exists()]+[((d,k+'_reach_'+p.parent.name),p) for (d,k),ps in PBS_REACH.items() for p in ps if (p.parent/'reconstruction.json').exists()]},
        quality_denominator='metric-eligible queries (feasible, validated complete nonempty reference); failed feasible methods score zero; medians and IQR over eligible queries',
        time_denominator='every attempted measured invocation; end-to-end is measured retrieval plus measured method wall time; the large-catalog unbounded PBS row is a separate measured invocation under the ordinary limit, not the 600 s reference',
        generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith(('manifest','candidates_','timing_')) and not p.stem.endswith('_pbs_pts')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
