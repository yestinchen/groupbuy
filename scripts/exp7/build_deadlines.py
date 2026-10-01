"""Explicit fresh deadline sources: recall and HV at common elapsed budgets,
time to first point, and time to K, drawn with the manuscript panel code but
read only from validated revision records. Observation, fresh-stage, and hard
modes are kept as separate panels and never pooled."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from gquery.figures import plot_exp4 as plots

REPO=Path(__file__).resolve().parents[2]
CAMPAIGN=Path(os.environ.get('CAMPAIGN_ROOT',REPO/'results'/'campaign'))
FIGURES=Path(os.environ.get('FIGURES_ROOT',REPO/'data_figures'/'exp7'))
MEMORY_GRIDS=Path(os.environ.get('MEMORY_GRIDS_ROOT',REPO/'results'/'memory_grids'))
ROOT=Path(os.environ.get('CAMPAIGN_DEADLINES_ROOT',CAMPAIGN/'deadlines'))
OUT=Path(os.environ.get('CAMPAIGN_DEADLINES_OUT',FIGURES/'exp7_deadlines'))
JOBS=[(d,m,c) for d in ['amazon','airbnb'] for m,c in [('observation','method'),('fresh_stages','fresh-stages'),('hard','method')]]
PREFIX={(d,m):(d,p) for d in ['amazon','airbnb'] for m,p in [('observation','anytime'),('fresh_stages','anytime_fresh'),('hard','anytime_hard')]}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

# The Airbnb hard-cutoff job was rerun after an external interruption; the failed attempt is retained beside it.
DIRS={('airbnb','hard'):'airbnb_hard_rerun'}
# The main text presents epsilon grids 4 and 16: the g=4 arm was measured alone in *_g4 jobs (same queries, deadlines,
# clocks, and modes) and is merged here; the campaign's g=8 rows stay in the summaries but leave the panels.
GRID_DIRS={(d,m):f'{d}_{m}_g4' for d in ['amazon','airbnb'] for m in ['observation','fresh_stages','hard']}
# The conventional Pareto DP arm (R6) was measured alone in *_dpconv jobs and merges the same way; optional until validated.
CONV_DIRS={(d,m):f'{d}_{m}_dpconv' for d in ['amazon','airbnb'] for m in ['observation','fresh_stages','hard']}
# PBS and unbounded PBS were rerun with the reachability bound in FeasibleExtensions (*_reach jobs, 2026-10-01) and replace
# the base job's PBS rows. Required unless PBS_REACH_OPTIONAL=1 (builder testing only).
REACH_DIRS={(d,m):f'{d}_{m}_reach' for d in ['amazon','airbnb'] for m in ['observation','fresh_stages','hard']}
PBS_METHODS={'PACS','PBS-unbounded'}
PANEL_DROP={'eps-ILP g=8'}

def read(dataset,name):
    path=ROOT/DIRS.get((dataset,name),f'{dataset}_{name}')/'anytime.csv';proof=json.loads((path.parent/'reconstruction.json').read_text())
    assert proof['files'][path.name]==sha(path) and proof['kind']=='deadlines' and proof['rows']==sum(1 for _ in open(path))-1
    df=pd.read_csv(path);df['source_file']=str(path);return df,path

def read_grid(dataset,name):
    path=ROOT/GRID_DIRS[(dataset,name)]/'anytime.csv';proof=json.loads((path.parent/'reconstruction.json').read_text())
    assert proof['files'][path.name]==sha(path) and proof['kind']=='deadlines' and proof['rows']==sum(1 for _ in open(path))-1
    params=json.loads((path.parent.parent/(path.parent.name+'.job')/'execution_params.json').read_text())
    assert params['grid']==4 and params['methods']=='eps-ILP g=4' and params['mode']==('hard' if name=='hard' else 'observation')
    df=pd.read_csv(path);assert set(df.method)=={'eps-ILP g=4'};df['source_file']=str(path);return df,path

# The manuscript panel code places its legends above the axes at 14 pt, which renders far smaller than the
# context-scaled text; raise every dict legend to the house legend size.
_band_panel_orig=plots._band_panel;_bar_panel_orig=plots._bar_panel
def _legend_fix(kwargs):
    if isinstance(kwargs.get('legend'),dict):kwargs['legend']={**kwargs['legend'],'fontsize':26,'columnspacing':1.,'handlelength':1.6}
    return kwargs
def _band_panel_override(*args,**kwargs):return _band_panel_orig(*args,**_legend_fix(kwargs))
def _bar_panel_override(*args,**kwargs):return _bar_panel_orig(*args,**_legend_fix(kwargs))
plots._band_panel=_band_panel_override;plots._bar_panel=_bar_panel_override
_save_panel_orig=plots.save_panel
def _save_panel_override(fig,exp_name,folder,stem,df=None):
    # the time-to-K panel draws its own 14 pt legend and 12 pt reach labels inline; raise both to house sizes
    if stem.endswith('_ttok'):
        for ax in fig.axes:
            if ax.get_legend() is not None:
                handles,labels=ax.get_legend_handles_labels();ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,fontsize=26,columnspacing=1.,handlelength=1.6)
            for text in ax.texts:text.set_fontsize(20)
    return _save_panel_orig(fig,exp_name,folder,stem,df)
plots.save_panel=_save_panel_override

def read_conv(dataset,name):
    path=ROOT/CONV_DIRS[(dataset,name)]/'anytime.csv'
    if not (path.parent/'reconstruction.json').exists():return None,None
    proof=json.loads((path.parent/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path) and proof['kind']=='deadlines'
    params=json.loads((path.parent.parent/(path.parent.name+'.job')/'execution_params.json').read_text());assert params['methods']=='Pareto-DP-conv'
    df=pd.read_csv(path);assert set(df.method)=={'Pareto-DP-conv'};df['source_file']=str(path);return df,path

def read_reach(dataset,name):
    path=ROOT/REACH_DIRS[(dataset,name)]/'anytime.csv'
    if not (path.parent/'reconstruction.json').exists() and os.environ.get('PBS_REACH_OPTIONAL')=='1':return None,None
    proof=json.loads((path.parent/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path) and proof['kind']=='deadlines'
    params=json.loads((path.parent.parent/(path.parent.name+'.job')/'execution_params.json').read_text())
    assert params['methods']=='PACS,PBS-unbounded' and params['mode']==('hard' if name=='hard' else 'observation')
    df=pd.read_csv(path);assert set(df.method)==PBS_METHODS;df['source_file']=str(path);return df,path

def present(frame):
    """PTS is the depth-first policy (runner label DFS); the UCT rows (runner label PTS) stay only in the raw records."""
    frame=frame.copy()
    if frame.method.eq('DFS').any():   # archived two-policy job; a fresh job has depth-first PTS rows only
        frame=frame[frame.method!='PTS'].copy();frame['method']=frame.method.replace({'DFS':'PTS'})
    # Pareto DP is the conventional DP without candidate reduction (Pareto-DP-conv); the reduced DP arm is not presented.
    if frame.method.eq('Pareto-DP-conv').any():
        frame=frame[frame.method!='Pareto-DP'];frame['method']=frame.method.replace({'Pareto-DP-conv':'Pareto-DP'})
    return frame

def main():
    OUT.mkdir(parents=True,exist_ok=True);plots.OUT_ROOT=OUT.parent;original=plots.read;sources={}
    # Section 3 presents one PTS policy (depth first): the runner's DFS rows are presented as PTS and its UCT rows are excluded.
    plots.LABELS.update({'PTS':'PTS','PBS-unbounded':'PBS unbounded','Pareto-DP':'DP'})
    try:
        for dataset,name,clock in JOBS:
            frame,path=read(dataset,name);grid,gpath=read_grid(dataset,name);sources[f'{dataset}.{name}']=path;sources[f'{dataset}.{name}_g4']=gpath
            reach,rpath=read_reach(dataset,name)
            if reach is not None:
                for m in PBS_METHODS:
                    assert set(reach[reach.method==m].query_id)==set(frame.query_id) and set(reach.elapsed_budget_s)==set(frame.elapsed_budget_s)
                first={m:i for i,m in enumerate(dict.fromkeys(frame.method))}  # keep the base job's method order
                frame=pd.concat([frame[~frame.method.isin(PBS_METHODS)],reach],ignore_index=True)
                frame=frame.sort_values('method',key=lambda c:c.map(first),kind='stable').reset_index(drop=True);sources[f'{dataset}.{name}_reach']=rpath
            assert set(grid.query_id)==set(frame.query_id) and set(grid.elapsed_budget_s)==set(frame.elapsed_budget_s)
            conv,cpath=read_conv(dataset,name)
            if conv is not None:
                assert set(conv.query_id)==set(frame.query_id) and set(conv.elapsed_budget_s)==set(frame.elapsed_budget_s);sources[f'{dataset}.{name}_dpconv']=cpath;grid=pd.concat([grid,conv],ignore_index=True)
            frame=present(pd.concat([frame,grid],ignore_index=True));assert not frame.duplicated(['query_id','method','elapsed_budget_s']).any()
            order={m:i for i,m in enumerate(['PACS','PTS','eps-ILP g=4','eps-ILP g=16','PBS-unbounded','Pareto-DP'])}  # panel and legend order
            panel=frame[~frame.method.isin(PANEL_DROP)].sort_values('method',key=lambda c:c.map(order).fillna(99),kind='stable')
            mode='hard' if name=='hard' else 'observation';folder,prefix=PREFIX[(dataset,name)]
            plots.read=lambda src,kind,frame=panel:frame.copy()
            plots.fig_anytime_dist(str(OUT),src='validated',folder=folder,prefix=prefix,clock=clock,mode=mode)
            deadlines=sorted({float(c.split('@')[1]) for c in frame.columns if c.startswith('recall@')})
            rows=[]
            for (method,limit),g in frame.groupby(['method','elapsed_budget_s'],sort=False):
                e=g[g.metric_eligible==1]
                row=dict(dataset=dataset,job=name,mode=mode,clock=clock,method=method,elapsed_budget_s=limit,n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=len(e),
                         n_completed=int((g.solver_status=='completed').sum()),n_timeout=int((g.solver_status=='timeout').sum()),n_terminated=int((g.solver_status=='terminated').sum()),n_failed=int((g.solver_status=='failed').sum()),
                         n_not_run=int(g.solver_status.isin(['not_run','retrieval_exceeds_budget']).sum()),t_first_median=e.t_first.median(),overshoot_median=g.overshoot_s.median() if 'overshoot_s' in g else None)
                for d in deadlines:
                    if mode=='hard' and d!=limit: continue
                    row[f'recall@{d}_median']=e[f'recall@{d}'].median();row[f'recall@{d}_mean']=e[f'recall@{d}'].mean();row[f'hv@{d}_median']=e[f'hv@{d}'].median();row[f'hv@{d}_mean']=e[f'hv@{d}'].mean()
                for k in sorted({int(c.split('_')[-1]) for c in frame.columns if c.startswith('t_to_')}):
                    row[f't_to_{k}_median']=e[f't_to_{k}'].median();row[f'reached_{k}']=e[f'reached_{k}'].mean();row[f'censored_{k}']=e[f'censored_{k}'].mean()
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/f'{dataset}_{name}_summary.csv',index=False)
    finally:
        plots.read=original
    code=REPO;diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    (OUT/'generation_source.diff').write_bytes(diff)
    manifest=dict(builder_sha256=sha(__file__),presentation='PTS is the depth-first policy (runner label DFS, same budgets and references); UCT rows are excluded from every panel and summary and remain only in the raw records. Epsilon grids: the panels show g=4 (from the *_g4 jobs, merged on query and cutoff) and g=16; the campaign g=8 rows remain in the summary CSVs only. Pareto DP is the conventional DP without candidate reduction (*_dpconv jobs); the reduced DP arm remains only in the raw records. PBS and unbounded PBS rows come from the *_reach jobs (reachability bound in FeasibleExtensions, 2026-10-01); the base job PBS rows remain only in the raw records',generation_command=sys.executable+' '+str(Path(__file__).resolve()),
        helpers={str(code/'gquery/figures/plot_exp4.py'):sha(code/'gquery/figures/plot_exp4.py')},source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
        sources={k:dict(path=str(p),sha256=sha(p),reconstruction_sha256=sha(p.parent/'reconstruction.json')) for k,p in sources.items()},
        quality_denominator='metric-eligible queries (feasible, complete validated reference); a method with no snapshot by the cutoff scores zero; observation, fresh-stage, and hard modes are separate panels',
        timing='observation and hard modes use the method clock; fresh-stage panels add the measured fresh retrieval offset; snapshot publication costs are inside the method clock; time-to-K is right censored at the observation horizon',
        generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and not p.stem.startswith('elapsed_')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
