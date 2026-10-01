"""Complete search against bounded search on Airbnb as the matching threshold drops and queries grow.

Main-text panels: retrieval + method latency (median solid, p90 dashed) against beta and against query length for
PBS (b=100), PTS (M=500) and PBS without beam truncation. Supplement panels: sampled search memory increase of the
isolated per-method processes against beta and length, and the number of nondominated partial assignments retained
at each level by PBS without beam truncation.

Sources (2026-10-01, PBS with the reachability bound in FeasibleExtensions):
- PBS rows and the timed reference call of PBS without beam truncation (reference_s + retrieval_s per query):
  sensitivity/airbnb_reach (beta sweep) and scalability/airbnb_scale_reach (length sweep).
- PTS rows: the depth-first jobs sensitivity/airbnb_dfs and scalability/airbnb_scale_dfs (runner label PTS-DFS),
  as in the other builders.
- Memory: memory/airbnb_beta{07,075,08}_reach and memory/airbnb_gs{2,4,6,8}_reach.
- Level widths: computed here, without timing, by PBS without beam truncation on the archived candidate sets of
  the reach jobs.
Every reach and memory source is required unless PBS_REACH_OPTIONAL=1 (builder testing only), which falls back to
the pre-fix jobs sensitivity/airbnb and scalability/airbnb_scale and skips missing memory panels.
"""
import gzip
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
OUT=FIGURES/'exp7_complete_search'
FOLDER='airbnb'
OPTIONAL=os.environ.get('PBS_REACH_OPTIONAL')=='1'
# MEMORY_OPTIONAL=1 skips memory panels whose runs are not validated yet, without the pre-fix PBS fallback
MEMORY_OPTIONAL=OPTIONAL or os.environ.get('MEMORY_OPTIONAL')=='1'
BETAS=[0.7,0.75,0.8];LENGTHS=[2,4,6,8]
PBS_SOURCES={'beta':(CAMPAIGN/'sensitivity/airbnb_reach/sensitivity.csv',CAMPAIGN/'sensitivity/airbnb/sensitivity.csv'),
             'length':(CAMPAIGN/'scalability/airbnb_scale_reach/scale.csv',CAMPAIGN/'scalability/airbnb_scale/scale.csv')}
PTS_SOURCES={'beta':CAMPAIGN/'sensitivity/airbnb_dfs/sensitivity.csv','length':CAMPAIGN/'scalability/airbnb_scale_dfs/scale.csv'}
MEMORY={'beta':{b:CAMPAIGN/f"memory/airbnb_beta{str(b).replace('.','')}_reach/mem_airbnb_12c.json" for b in BETAS},
        'length':{g:CAMPAIGN/f'memory/airbnb_gs{g}_reach/mem_airbnb_gs{g}.json' for g in LENGTHS}}
COMPLETE='PBS-unbounded'
LABELS={'PACS':'PBS','PTS':'PTS',COMPLETE:'PBS unbounded'}
COLORS={'PACS':plots.color('PACS'),'PTS':plots.color('PTS'),COMPLETE:'#2E5E8E'}   # PBS-unbounded colour of build_default
XLABEL={'beta':r'matching threshold $\beta$','length':'query length $n$'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def top_legend(ax,ncol=3):
    # one row above the axes: inside the axes it covers the curves
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.0),ncol=ncol,frameon=False,fontsize='small',handlelength=1.5,columnspacing=1.0)
sources={}


def validated(path):
    proof=json.loads((path.parent/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path)
    sources[str(path)]=dict(sha256=sha(path),reconstruction_sha256=sha(path.parent/'reconstruction.json'));return path


def pick(reach,fallback):
    if (reach.parent/'reconstruction.json').exists():return validated(reach),False
    if not OPTIONAL:raise FileNotFoundError(f'PBS reachability rerun not validated: {reach}')
    return validated(fallback),True


def sweep_rows(kind):
    """Per-query latency rows for PBS, PTS and the timed reference call of PBS without beam truncation."""
    path,fallback=pick(*PBS_SOURCES[kind])
    df=pd.read_csv(path,low_memory=False);df=df[df.sweep==kind]
    params=BETAS if kind=='beta' else LENGTHS
    df=df[df.param.round(4).isin([round(float(p),4) for p in params])]
    pbs=df[df.method=='PACS'].copy();assert pbs.groupby('param').size().eq(100).all() and pbs.param.nunique()==len(params)
    # one timed reference call per query and setting: PBS without beam truncation on the same candidate sets
    ref=pbs.drop_duplicates(['param','query_id']).copy()
    assert (ref.query_status=='feasible').all() and (ref.reference_status=='complete').all()
    ref['total_s']=ref.reference_s+ref.retrieval_s;ref['method']=COMPLETE
    pts_path=validated(PTS_SOURCES[kind]);pts=pd.read_csv(pts_path,low_memory=False)
    pts=pts[(pts.sweep==kind)&(pts.method=='PTS-DFS')&pts.param.round(4).isin([round(float(p),4) for p in params])].copy()
    assert pts.groupby('param').size().eq(100).all() and pts.param.nunique()==len(params);pts['method']='PTS'
    rows=pd.concat([pbs,pts,ref],ignore_index=True)[['param','method','query_id','total_s','method_wall_s','retrieval_s','reference_s','ref_size','recall','metric_eligible']]
    rows['source_file']=[str(path) if m!='PTS' else str(pts_path) for m in rows.method];rows['pbs_fallback']=fallback
    return rows,path,fallback


def latency_panel(rows,kind):
    summary=(rows.groupby(['param','method']).total_s.agg(median='median',p90=lambda x:x.quantile(.9),n='size')
             .reset_index());summary[['median_ms','p90_ms']]=summary[['median','p90']]*1000
    fig,ax=plots.paper_panel()
    for method in ['PACS','PTS',COMPLETE]:
        g=summary[summary.method==method].sort_values('param')
        ax.plot(g.param,g.median_ms,marker='o',color=COLORS[method],label=LABELS[method])
        ax.plot(g.param,g.p90_ms,marker='o',linestyle='--',color=COLORS[method],markerfacecolor='white')
    ax.set_yscale('log');ax.set_xlabel(XLABEL[kind]);ax.set_ylabel('latency (ms)')
    ax.set_xticks(BETAS if kind=='beta' else LENGTHS)
    ax.yaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'))
    top_legend(ax)
    plots.save_panel(fig,str(OUT),FOLDER,f'latency_{kind}',summary)
    return summary


def memory_panel(kind):
    rows=[]
    for param,path in MEMORY[kind].items():
        if not (path.parent/'reconstruction.json').exists():
            if MEMORY_OPTIONAL:continue
            raise FileNotFoundError(f'memory run not validated: {path}')
        rec=json.loads(validated(path).read_text())
        assert rec['n_queries_requested']==100 and (kind=='length' or abs(rec['beta']-param)<1e-9)
        for method,label in [('PBS','PACS'),('PTS','PTS'),('PBS-unbounded',COMPLETE)]:
            c=rec['search_processes'][method]
            rows.append(dict(param=param,method=label,sampled_search_increase_mb=c['sampled_search_increase_mb'],sampled_search_peak_rss_mb=c['sampled_search_peak_rss_mb'],candidate_resident_baseline_mb=c['candidate_resident_baseline_mb'],requested=c['requested'],returned=c['returned'],timeouts=c['timeouts'],source_file=str(path)))
    if not rows:return None
    data=pd.DataFrame(rows);fig,ax=plots.paper_panel()
    for method in ['PACS','PTS',COMPLETE]:
        g=data[data.method==method].sort_values('param')
        ax.plot(g.param,g.sampled_search_increase_mb,marker='o',color=COLORS[method],label=LABELS[method])
    # every sampled increase is at least 1 MB, so a log axis with plain labels, as in the latency panels
    assert (data.sampled_search_increase_mb>0).all()
    ax.set_yscale('log');ax.set_xlabel(XLABEL[kind]);ax.set_ylabel('peak memory (MB)')  # increase over the candidate-resident baseline; stated in the caption
    ax.yaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'))
    ax.set_xticks(BETAS if kind=='beta' else LENGTHS);top_legend(ax)
    plots.save_panel(fig,str(OUT),FOLDER,f'memory_{kind}',data)
    return data


def archived_inputs(kind):
    """(param, path) of the archived candidate sets of each setting in the reach job (or its fallback)."""
    path,_=pick(*PBS_SOURCES[kind]);job=path.parent
    if kind=='length':return [(g,job/'inputs'/f'length-airbnb_gs{g}'/'frontiers.jsonl.gz') for g in LENGTHS]
    settings=pd.read_csv(job/'sensitivity_inputs.csv');out=[];previous=None
    # the runner archives inputs once per (alpha, beta) change, under the index of the first setting that loads them
    for index,row in settings.iterrows():
        key=(row.alpha,row.beta)
        if key!=previous:loaded=index;previous=key
        if row.sweep=='beta' and round(row.param,4) in [round(b,4) for b in BETAS]:out.append((row.param,job/'inputs'/str(loaded)/'frontiers.jsonl.gz'))
    return out


def level_widths(kind):
    from gquery.alg.common import Item
    from gquery.alg.pareto.pbs import solve_pbs
    rows=[]
    for param,path in archived_inputs(kind):
        sources[str(path)]=dict(sha256=sha(path))
        for line in gzip.open(path,'rt'):
            r=json.loads(line)
            if r['query_status']!='feasible':continue
            slots=[[(Item(f'{i}-{j}',[],c[1],c[2],None,''),c[0]) for j,c in enumerate(s)] for i,s in enumerate(r['candidates'])]
            result=solve_pbs(slots,r['budget'],r['alpha'],beam_size=0,prefix_rule='raw_cost',use_reduction=True)
            assert len(result.frontier)==len(r['reference'])   # same complete frontier as the archived reference
            for level,width in enumerate(result.level_widths,1):
                rows.append(dict(param=param,query_id=r['query_id'],n_requirements=len(slots),level=level,width=width))
    widths=pd.DataFrame(rows);widths.to_csv(OUT/FOLDER/f'level_widths_{kind}_per_query.csv',index=False)
    summary=widths.groupby(['param','level']).width.agg(median='median',p90=lambda x:x.quantile(.9),max='max',n='size').reset_index()
    fig,ax=plots.paper_panel();palette=plots.plt.get_cmap('viridis')
    params=sorted(summary.param.unique())
    for i,param in enumerate(params):
        g=summary[summary.param==param].sort_values('level');colour=palette(i/max(1,len(params)-1)*.85)
        label=(rf'$\beta={param:g}$' if kind=='beta' else f'$n={int(param)}$')
        ax.plot(g.level,g['median'],marker='o',color=colour,label=label)
        ax.plot(g.level,g.p90,linestyle='--',color=colour)
    ax.set_yscale('log');ax.set_xlabel('search level');ax.set_ylabel('partial assignments')
    ax.yaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'));top_legend(ax,ncol=len(params))
    plots.save_panel(fig,str(OUT),FOLDER,f'level_widths_{kind}',summary)
    return summary


def main():
    (OUT/FOLDER).mkdir(parents=True,exist_ok=True);plots.OUT_ROOT=OUT.parent;fallbacks={}
    for kind in ['beta','length']:
        rows,_,fallbacks[kind]=sweep_rows(kind)
        rows.to_csv(OUT/FOLDER/f'latency_{kind}_per_query.csv',index=False)
        latency_panel(rows,kind);memory_panel(kind);level_widths(kind)
    code=REPO;diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    (OUT/'generation_source.diff').write_bytes(diff)
    manifest=dict(builder_sha256=sha(__file__),generation_command=sys.executable+' '+str(Path(__file__).resolve()),
                  source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
                  dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),sources=sources,pbs_fallback=fallbacks,
                  presentation='Latency is fresh retrieval plus method wall time per query (median solid, p90 dashed). PBS without beam truncation is the timed reference call of the sweep job (reference_s) plus the same retrieval_s; PTS is the depth-first policy (PTS-DFS jobs). Memory is the sampled search increase of each isolated per-method process over its candidate-resident baseline (one value per process, 1 ms sampling). Level widths are the nondominated partial assignments retained per level by PBS without beam truncation on the archived candidate sets, computed without timing.',
                  generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
