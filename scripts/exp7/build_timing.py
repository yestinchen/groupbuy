"""Candidate counts and latency components across catalog sizes in the house panel style (medians with
interquartile bands over the 100 queries), from the validated scalability rows presented by the scalability builder
(PTS is the depth-first policy). Single panels only: the manuscript arranges them with subfigure captions."""
import hashlib
import importlib.util
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
OUT=FIGURES/'exp7_scale'  # shares the scale topic directory with the scalability builder; own manifest
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
_spec=importlib.util.spec_from_file_location('scalability',Path(__file__).resolve().parent/'build_scalability.py')
scal=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(scal)
GROUPS=[('category','amazon','items per category (K)',[(f'e11_cat{k}k',str(k),k) for k in [10,15,20,25,30]],False,('amazon','scale')),
        ('catalog','amazon','catalog items',list(plots.XL_LADDER),True,('amazon','xl')),
        ('catalog','airbnb','catalog items',list(plots.AIRBNB_LADDER),True,('airbnb','xl'))]
COMPONENTS=[('retrieval_s','retrieval','#6C8EBF','o','-'),('search_s','search','#D79B00','^','-'),('total_s','total','#5A6472','x','--')]

def counts_of(frame):
    before=frame.candidates_before_by_requirement.map(lambda s:sum(json.loads(s)));after=frame.candidates_after_by_requirement.map(lambda s:sum(json.loads(s)));return before,after

def stats(values):
    v=values.dropna().to_numpy();return float(np.median(v)),float(np.quantile(v,.25)),float(np.quantile(v,.75)),len(v)

def draw_counts(ax,frame,ladder,xlabel,logx):
    rows=[];xs=[];series={k:([],[],[]) for k in ['before','after']}
    for setting,label,n in ladder:
        g=frame[(frame.setting==setting)&(frame.method=='PACS')];before,after=counts_of(g)
        for key,vals in [('before',before),('after',after)]:
            m,lo,hi,count=stats(vals);series[key][0].append(m);series[key][1].append(lo);series[key][2].append(hi)
            rows.append(dict(setting=setting,label=label,x=n,quantity='candidates_'+key,median=m,q25=lo,q75=hi,n=count))
        xs.append(n)
    x=np.array(xs)
    for key,label,color,marker in [('before','before reduction','#5A6472','o'),('after','after reduction','#82B366','s')]:
        plots._band(ax,x,np.array(series[key][0]),np.array(series[key][1]),np.array(series[key][2]),color,label,marker=marker)
    if logx:ax.set_xscale('log')
    ax.set_xticks(xs);ax.set_xticklabels([l for _,l,_ in ladder]);ax.xaxis.set_minor_formatter(plots.plt.NullFormatter());ax.set_yscale('log')
    ax.set_xlabel(xlabel);ax.set_ylabel('candidate count');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,columnspacing=1.,handlelength=1.6)
    return pd.DataFrame(rows)

def draw_timing(ax,frame,method,ladder,xlabel,logx):
    rows=[];xs=[n for _,_,n in ladder];x=np.array(xs)
    for field,label,color,marker,ls in COMPONENTS:
        med,lo,hi=[],[],[]
        for setting,lab,n in ladder:
            g=frame[(frame.setting==setting)&(frame.method==method)];m,l,h,count=stats(g[field]*1000);med.append(m);lo.append(l);hi.append(h)
            rows.append(dict(setting=setting,label=lab,x=n,method=method,component=field,median_ms=m,q25_ms=l,q75_ms=h,n=count))
        plots._band(ax,x,np.array(med),np.array(lo),np.array(hi),color,label,marker=marker,ls=ls)
    if logx:ax.set_xscale('log')
    ax.set_xticks(xs);ax.set_xticklabels([l for _,l,_ in ladder]);ax.xaxis.set_minor_formatter(plots.plt.NullFormatter());ax.set_yscale('log')
    ax.set_yticks([.5,1,2,5,10,20,50]);ax.yaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'));ax.yaxis.set_minor_formatter(plots.plt.NullFormatter());ax.set_ylim(.4,60)
    ax.set_xlabel(xlabel);ax.set_ylabel('time (ms)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,columnspacing=1.,handlelength=1.6)
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True);plots.OUT_ROOT=OUT.parent
    frames={key:scal.read(*key) for key in {g[5] for g in GROUPS}}
    records=[];sources={}
    for key,df in frames.items():
        for src in df.source_file.unique():
            p=Path(src);sources[str(p.resolve())]=sha(p);sources[str((p.parent/'reconstruction.json').resolve())]=sha(p.parent/'reconstruction.json')
    # single panels
    for name,folder,xlabel,ladder,logx,key in GROUPS:
        df=frames[key];assert all(s in set(df.setting) for s,_,_ in ladder),(name,[s for s,_,_ in ladder if s not in set(df.setting)])
        fig,ax=plots.paper_panel();t=draw_counts(ax,df,ladder,xlabel,logx);t['group']=f'{folder}_{name}';records.append(t);plots.save_panel(fig,str(OUT),folder,f'candidates_{name}',t)
        for method,tag in [('PACS','pbs'),('PTS','pts')]:
            fig,ax=plots.paper_panel();t=draw_timing(ax,df,method,ladder,xlabel,logx);t['group']=f'{folder}_{name}';records.append(t);plots.save_panel(fig,str(OUT),folder,f'timing_{tag}_{name}',t)
    pd.concat(records,ignore_index=True).to_csv(OUT/'timing_panels_summary.csv',index=False)
    code=REPO;diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    manifest=dict(builder_sha256=sha(__file__),generation_command=sys.executable+' '+str(Path(__file__).resolve()),
        helpers={str(code/'gquery/figures/plot_exp4.py'):sha(code/'gquery/figures/plot_exp4.py'),str(Path(__file__).resolve().parent/'build_scalability.py'):sha(Path(__file__).resolve().parent/'build_scalability.py')},
        source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),sources=sources,
        presentation='House panel style (10x6 in, paper font scale, IQR bands) for candidate counts and retrieval/search/total latency; medians and 25th/75th percentiles over the 100 queries of each setting; PTS is the depth-first policy; candidate counts are taken from the PBS rows (identical for PTS); total is the per-query retrieval plus method time; single panels only, arranged by subfigure captions in the manuscript',
        generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name.startswith(('candidates_','timing_'))})
    (OUT/'manifest_timing.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
