"""Explicit fresh ablation, deceptive, budget split, and case study sources with status denominators."""
import hashlib
import importlib.util
import json
import os
import shutil
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
# The environment overrides exist only for exercising the builder on synthetic pilot outputs.
ROOT=Path(os.environ.get('CAMPAIGN_ABLATION_ROOT',CAMPAIGN/'sensitivity'))
OUT=Path(os.environ.get('CAMPAIGN_ABLATION_OUT',FIGURES/'exp7_ablation'))
# PTS is the depth-first policy: the diagnostics come from the *_dfs jobs (their runners record the policy), the PBS ablation arms from the base ablation job.
JOBS={('amazon','ablation'):'amazon_ablation',('airbnb','ablation'):'airbnb_ablation',('amazon','deceptive'):'amazon_deceptive_dfs',('airbnb','deceptive'):'airbnb_deceptive_dfs',
      ('amazon','budget_split'):'amazon_budget_split_dfs',('airbnb','budget_split'):'airbnb_budget_split_dfs',('amazon','case'):'amazon_case_gap',('airbnb','case'):'airbnb_case_gap'}
# Depth-first ablation jobs (bench_ablation --pts-traversal depth_first) supply every PTS arm; the policy has no close-set arm.
DFS_JOBS={'amazon':'amazon_ablation_dfs','airbnb':'airbnb_ablation_dfs'}
DFS_ARMS=['-bound','-prefixdom','-bound-prefixdom']
# PBS rerun with the reachability bound in FeasibleExtensions (2026-10-01): the PBS ablation arms and the case study
# (whose table compares PBS and PTS) come from these jobs. Required unless PBS_REACH_OPTIONAL=1 (builder tests only).
# *_ablation_reach2: the first ablation reruns used a PBS copy (variants.py) without the bound; only reach2 is valid
REACH_JOBS={('amazon','ablation'):'amazon_ablation_reach2',('airbnb','ablation'):'airbnb_ablation_reach2',
            ('amazon','case'):'amazon_case_gap_reach',('airbnb','case'):'airbnb_case_gap_reach'}
FILES={'ablation':'ablation.csv','deceptive':'deceptive.csv','budget_split':'budget_split.csv','case':'case_study.tex'}
FOLDER={'amazon':'amazon','airbnb':'airbnb'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reach(dataset,kind):
    """True when the PBS reachability rerun supplies this source; a missing rerun is an error unless PBS_REACH_OPTIONAL=1."""
    if (dataset,kind) not in REACH_JOBS:return False
    if (ROOT/REACH_JOBS[(dataset,kind)]/'reconstruction.json').exists():return True
    assert os.environ.get('PBS_REACH_OPTIONAL')=='1',f'missing validated PBS rerun {REACH_JOBS[(dataset,kind)]}'
    return False

def source(dataset,kind,rerun=False):
    directory=ROOT/(REACH_JOBS[(dataset,kind)] if rerun else JOBS[(dataset,kind)]);path=directory/FILES[kind]
    proof=json.loads((directory/'reconstruction.json').read_text())
    assert proof['files'][path.name]==sha(path) and proof['kind']==('case_study' if kind=='case' else kind)
    return path,proof

def read(dataset,kind):
    path,proof=source(dataset,kind);df=pd.read_csv(path);df['source_file']=str(path);return df


def available(dataset,kind):
    """The deceptive and budget-split workloads are not part of the current job matrix."""
    return (ROOT/JOBS[(dataset,kind)]/'reconstruction.json').exists()

def dfs_source(dataset):
    directory=ROOT/DFS_JOBS[dataset];path=directory/FILES['ablation']
    if not (directory/'reconstruction.json').exists() or not path.exists(): return None,None
    proof=json.loads((directory/'reconstruction.json').read_text())
    assert proof['files'][path.name]==sha(path) and proof['kind']=='ablation'
    return path,proof

def read_dfs(dataset):
    path,proof=dfs_source(dataset)
    if path is None: return None
    df=pd.read_csv(path);df['source_file']=str(path);df=df[df.family=='PTS-DFS'].copy();df['family']='PTS';return df  # presented as the manuscript's PTS

def ablation(dataset):
    df=read(dataset,'ablation');folder=FOLDER[dataset]
    beta=plots.ABLATION_BETA if plots.ABLATION_BETA in set(df.beta) else sorted(df.beta)[-1]
    # PBS arms from the reachability rerun (base job only for builder tests); every PTS arm from the depth-first job
    base=df[df.beta==beta];df=base[base.family=='PACS']
    if reach(dataset,'ablation'):
        path,proof=source(dataset,'ablation',rerun=True);rerun=pd.read_csv(path);rerun['source_file']=str(path)
        rerun=rerun[(rerun.beta==beta)&(rerun.family=='PACS')]
        # the current runner measures only the presented PBS arms (full and the raw cost ablation)
        shown=set(plots.DOCUMENTED_ARMS['PACS']);rerun=rerun[rerun.arm.isin(shown)]
        assert set(rerun.arm)==shown and rerun.groupby('arm').size().eq(100).all() and set(rerun.query_id)==set(df.query_id),set(rerun.arm)
        df=rerun
    dfs=read_dfs(dataset)
    if dfs is None:dfs=base[base.family=='PTS'].copy()   # fresh job: its PTS family is depth first already
    else:dfs=dfs[dfs.beta==beta]
    df=pd.concat([df,dfs],ignore_index=True)
    arms=[('PACS',arm) for arm in plots.DOCUMENTED_ARMS['PACS'] if arm!='full']+[('PTS',arm) for arm in DFS_ARMS if arm in set(dfs.arm)]
    ticks=dict(plots.ABLATION_TICK)
    denominators=[]
    for stem,col,ylab,ratio in [('ablation_dist_recall','recall','change in recall',False),('ablation_dist_hv','hv_ratio','change in hypervolume',False),('ablation_dist_work','evals','work, ablated / full',True)]:
        data,labs,colours,frames=[],[],{},[]
        for fam,arm in arms:
            full=df[(df.family==fam)&(df.arm=='full')].set_index('query_id');abl=df[(df.family==fam)&(df.arm==arm)].set_index('query_id')
            common=full.index.intersection(abl.index)
            if not ratio:
                # paired on metric-eligible queries; failed feasible arms keep zero quality
                common=[q for q in common if full.loc[q,'metric_eligible']==1 and abl.loc[q,'metric_eligible']==1]
                v=(abl.loc[common,col]-full.loc[common,col])
            else:
                base=full.loc[common,col].replace(0,np.nan);v=(abl.loc[common,col]/base).dropna()
            lbl=ticks.get((fam,arm),arm);data.append(v.values);labs.append(lbl);colours[lbl]=plots.color(fam)
            frames.append(pd.DataFrame({'family':fam,'arm':arm,'value':v.values}))
            denominators.append(dict(panel=stem,family=fam,arm=arm,beta=beta,n_requested=len(abl),n_paired=len(v),n_metric=int(abl.metric_eligible.sum()),n_failure=int((abl.solver_status=='solver_failure').sum()),n_timeout=int(abl.timed_out.sum()),median=float(np.median(v)) if len(v) else np.nan))
        plots._box_panel(str(OUT),folder,stem,data,labs,ylab,colours,logy=ratio,hline=1. if ratio else 0.,yticks=[.25,.5,1,2,4] if ratio else None,df=pd.concat(frames,ignore_index=True))
    pd.DataFrame(denominators).to_csv(OUT/f'{dataset}_ablation_denominators.csv',index=False)
    summary=df.groupby(['beta','family','arm'],sort=False).agg(n_requested=('query_id','size'),n_metric=('metric_eligible','sum'),n_returned=('success','sum'),n_failure=('solver_status',lambda s:(s=='solver_failure').sum()),recall=('recall','mean'),hv=('hv_ratio','mean'),evals_median=('evals','median'),time_median=('method_wall_s','median')).reset_index()
    summary.to_csv(OUT/f'{dataset}_ablation_summary.csv',index=False)

def deceptive(dataset):
    df=read(dataset,'deceptive');folder=FOLDER[dataset]
    cat=df[(df.workload=='catalog')&(df.method=='PACS')]
    series,rows=[],[]
    for rule,lbl,col in [('raw_cost','raw cost $C$',plots.color('PACS')),('overage',r'overhead $\Delta C$',plots.color('PACS_dC'))]:
        s=cat[cat.rule==rule];xs,med,lo,hi=[],[],[],[]
        for prm,g in s.groupby('param'):
            eligible=g[g.metric_eligible==1];m,l,h=plots._quartiles(eligible,'recall');xs.append(prm);med.append(m);lo.append(l);hi.append(h)
            rows.append(dict(rule=rule,beam=prm,median=m,q25=l,q75=h,n_requested=len(g),n_metric=len(eligible),n_failure=int((g.solver_status=='solver_failure').sum())))
        o=np.argsort(xs);series.append((np.array(xs)[o],np.array(med)[o],np.array(lo)[o],np.array(hi)[o],col,lbl))
    plots._band_panel(str(OUT),folder,'deceptive_rule',series,'beam width $b$','frontier recall',logx=True,ylim=(0,1.05),legend='lower right',xticks=plots._sparse_ticks(sorted(cat.param.unique())),df=pd.DataFrame(rows))
    syn=df[df.workload=='synthetic'];fig,ax=plots.paper_panel();rows=[]
    for meth,rule,prm,lbl,col in [('PACS','raw_cost',50,'PBS, raw cost',plots.color('PACS')),('PACS','overage',50,r'PBS, $\Delta C$',plots.color('PACS_dC')),('PTS','raw_cost',100,'PTS',plots.color('PTS'))]:
        g=syn[(syn.method==meth)&(syn.rule==rule)&(syn.param==prm)&(syn.metric_eligible==1)]
        if g.empty: continue
        m=g.groupby('k').recall.median();ax.plot(m.index.values,m.values,marker='o',color=col,label=lbl)
        rows+=[dict(method=meth,rule=rule,param=prm,k=float(k),median=float(v),n=int((g.k==k).sum())) for k,v in m.items()]
    ax.set_xlabel('frontier width $k$');ax.set_ylabel('frontier recall');ax.set_ylim(-.03,1.05);ax.legend(loc='center left')
    plots.save_panel(fig,str(OUT),folder,'deceptive_width',pd.DataFrame(rows))
    summary=df.groupby(['workload','method','rule','param'],sort=False).agg(n_requested=('query_id','size'),n_metric=('metric_eligible','sum'),n_returned=('success','sum'),n_recovered=('recovered','sum'),recall=('recall','mean'),hv=('hv_ratio','mean'),evals_median=('evals','median')).reset_index()
    summary.to_csv(OUT/f'{dataset}_deceptive_summary.csv',index=False)

def budget(dataset):
    df=read(dataset,'budget_split');folder=FOLDER[dataset]
    arms=[(m,lbl) for m,lbl in plots.BUDGET_ARMS if m in set(df.method)];labs=[lbl for _,lbl in arms];colours={lbl:plots.color(m) for m,lbl in arms}
    feasible=df[df.query_feasible==1]
    succ=[float(feasible[feasible.method==m].success.mean()) for m,_ in arms]
    plots._bar_panel(str(OUT),folder,'budget_success',labs,succ,'success rate',colours,ylim=(0,1.15),df=pd.DataFrame({'method':[m for m,_ in arms],'success':succ,'n_feasible':[int((feasible.method==m).sum()) for m,_ in arms]}))
    data,ls,cs,frames=[],[],{},[]
    for m,lbl in arms:
        v=df[(df.method==m)&(df.metric_eligible==1)].recall
        if not len(v): continue
        data.append(v.values);ls.append(lbl);cs[lbl]=plots.color(m);frames.append(pd.DataFrame({'method':m,'value':v.values}))
    plots._box_panel(str(OUT),folder,'budget_recall',data,ls,'frontier recall',cs,ylim=(-.03,1.05),df=pd.concat(frames,ignore_index=True))
    summary=df.groupby('method',sort=False).agg(n_requested=('query_id','size'),n_feasible=('query_feasible','sum'),n_metric=('metric_eligible','sum'),n_returned=('success','sum'),n_starved=('split_starved',lambda s:s.fillna(0).sum()),recall=('recall','mean'),hv=('hv_ratio','mean'),wratio=('wratio','mean'),time_median=('method_wall_s','median')).reset_index()
    summary.to_csv(OUT/f'{dataset}_budget_split_summary.csv',index=False)

def case(dataset):
    path,proof=source(dataset,'case',rerun=reach(dataset,'case'));target=OUT/f'table_case_study_{dataset}.tex'
    shutil.copyfile(path,target);shutil.copyfile(path.parent/'case_study_provenance.json',target.with_suffix('.provenance.json'))

def main():
    OUT.mkdir(parents=True,exist_ok=True);plots.OUT_ROOT=OUT.parent
    for dataset in ['amazon','airbnb']:
        ablation(dataset);case(dataset)
        if available(dataset,'deceptive'):deceptive(dataset)
        if available(dataset,'budget_split'):budget(dataset)
    code=REPO;diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    (OUT/'generation_source.diff').write_bytes(diff)
    manifest=dict(builder_sha256=sha(__file__),presentation='PBS ablation arms and the case study come from the *_reach jobs (PBS with the reachability bound). PTS is the depth-first policy: PTS ablation arms from the *_ablation_dfs jobs, deceptive and budget-split from the *_dfs jobs and the case study from the *_case_gap jobs (depth-first PTS, query chosen where PTS recovers reference points PBS misses), whose runners record pts_traversal; UCT measurements remain only in the raw records',generation_command=sys.executable+' '+str(Path(__file__).resolve()),
        helpers={str(code/'gquery/figures/plot_exp4.py'):sha(code/'gquery/figures/plot_exp4.py')},source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
        sources={**{f'{d}.{k}':dict(path=str(source(d,k)[0]),sha256=sha(source(d,k)[0]),reconstruction_sha256=sha(source(d,k)[0].parent/'reconstruction.json')) for d,k in JOBS if available(d,k)},
                 **{f'{d}.{k}_reach':dict(path=str(source(d,k,True)[0]),sha256=sha(source(d,k,True)[0]),reconstruction_sha256=sha(source(d,k,True)[0].parent/'reconstruction.json')) for d,k in REACH_JOBS if reach(d,k)},**{f'{d}.ablation_dfs':dict(path=str(dfs_source(d)[0]),sha256=sha(dfs_source(d)[0]),reconstruction_sha256=sha(dfs_source(d)[0].parent/'reconstruction.json')) for d in DFS_JOBS if dfs_source(d)[0] is not None}},
        quality_denominator='metric-eligible queries (feasible, validated complete nonempty reference); failed feasible arms score zero; ablation panels pair each arm with the full method on the same eligible query (PTS is the depth-first policy, whose arms and full method come from the depth-first ablation job); work ratios use timed arms with nonzero full-method evaluations',
        ablation_threshold=plots.ABLATION_BETA,generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
