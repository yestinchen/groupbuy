"""Generate only default artifacts from explicit fresh campaign source mappings."""
import hashlib
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import re
import numpy as np
import pandas as pd

from gquery.alg.exp4.make_tables import table_pareto, _fmt_time
from gquery.figures import plot_exp4 as plots
from gquery.figures.default_status import distributions

REPO=Path(__file__).resolve().parents[2]
CAMPAIGN=Path(os.environ.get('CAMPAIGN_ROOT',REPO/'results'/'campaign'))
FIGURES=Path(os.environ.get('FIGURES_ROOT',REPO/'data_figures'/'exp7'))
MEMORY_GRIDS=Path(os.environ.get('MEMORY_GRIDS_ROOT',REPO/'results'/'memory_grids'))
ROOT=CAMPAIGN/'default'
OUT=FIGURES/'exp7_default'
SOURCES={('amazon','default'):ROOT/'amazon/default.csv',('amazon','baselines'):ROOT/'amazon_baselines/baselines.csv',
         ('airbnb','default'):ROOT/'airbnb/default.csv',('airbnb','baselines'):ROOT/'airbnb_baselines/baselines.csv'}
# The conventional Pareto DP arm (no candidate reduction, overhead label dominance) was measured alone on the same
# queries after the campaign; its rows merge into the default frame when the job has validated.
EXTRA={('amazon','default'):ROOT/'amazon_dpconv/default.csv',('airbnb','default'):ROOT/'airbnb_dpconv/default.csv'}
# PBF rerun alone with a 60 s limit (the limit of every other method), a 16 GiB limit on its memory growth and no
# combination guard; once validated its rows replace the PBF rows of the default job.
PBF_EXTRA={('amazon','default'):ROOT/'amazon_pbf60/default.csv',('airbnb','default'):ROOT/'airbnb_pbf60/default.csv'}
# PBS arms rerun with the reachability bound in FeasibleExtensions (2026-10-01); their rows replace the PBS rows of the
# default job. Required unless PBS_REACH_OPTIONAL=1 (only for testing the builder before the jobs finish).
PBS_EXTRA={('amazon','default'):ROOT/'amazon_reach/default.csv',('airbnb','default'):ROOT/'airbnb_reach/default.csv'}
PBS_METHODS={'PACS','PACS_dC','PBS-unbounded'}
LABELS={'PACS':'PBS','PTS':'PTS','PBS-unbounded':'PBS unbounded','Pareto-DP':'DP','DFS':'PTS-DFS',
        'PBF':'PBF','PACS_dC':r'PBS$_{\Delta C}$','Greedy':'Greedy','WS-28':'WS(28)','WS-253':'WS(253)',
        'NSGA2-2000':'NSGA-II(2k)','NSGA2-10000':'NSGA-II(10k)',
        **{f'eps-ILP g={g}':rf'$\epsilon$-ILP $g={g}$' for g in [2,4,8,16]}}
# Presented epsilon grids: 4 and 16 in the main table, 4, 8, and 16 in the supplementary sweep; g=2 is measured but not presented.
MAIN_GRIDS=[4,16];SWEEP_GRIDS=[4,8,16]
ORDER=['PACS','PTS','PBS-unbounded','Pareto-DP',*[f'eps-ILP g={g}' for g in SWEEP_GRIDS],'PBF','WS-28','WS-253','Greedy','NSGA2-2000','NSGA2-10000']
plots.COLORS.update({'PBS-unbounded':'#2E5E8E','Pareto-DP':'#9673A6','Pareto-DP-conv':'#6F4E8C','DFS':'#C17D11','Greedy':'#82B366','WS-28':'#B8A16B','WS-253':'#8F7A3F','NSGA2-2000':'#B85450','NSGA2-10000':'#7A2E2B'})


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(dataset,kind):
    # No interpretation of historical paths or silent fallback.
    path=SOURCES[(dataset,kind)]
    proof=json.loads((path.parent/'reconstruction.json').read_text())
    assert proof['queries']==100 and proof['files'][path.name]==sha(path)
    frame=pd.read_csv(path)
    grids=[f'eps-ILP g={g}' for g in [2,4,8,16]]
    expected = ([{'PACS','PACS_dC','PTS','PBS-unbounded','Pareto-DP','DFS','PBF',*grids},   # archived two-policy campaign job
                 {'PACS','PACS_dC','PTS','PBS-unbounded','Pareto-DP-conv','PBF',*grids}]   # fresh job of the current runner
                if kind=='default' else [{'PBS-unbounded','Pareto-DP','DFS','WS-28','WS-253','Greedy','NSGA2-2000','NSGA2-10000'}])
    assert set(frame.method) in expected,set(frame.method)
    assert frame.groupby('method').size().eq(100).all()
    frame['source_file']=str(path)
    extra=EXTRA.get((dataset,kind))
    if extra is not None and (extra.parent/'reconstruction.json').exists():
        proof=json.loads((extra.parent/'reconstruction.json').read_text());assert proof['files'][extra.name]==sha(extra)
        more=pd.read_csv(extra);assert set(more.method)=={'Pareto-DP-conv'} and len(more)==100 and set(more.query_id)==set(frame.query_id)
        more['source_file']=str(extra);frame=pd.concat([frame,more],ignore_index=True)
    extra=PBF_EXTRA.get((dataset,kind))
    # the 60 s PBF rows are required unless PBF60_OPTIONAL=1 (only for testing before the jobs finish)
    if extra is not None and ((extra.parent/'reconstruction.json').exists() or os.environ.get('PBF60_OPTIONAL')!='1'):
        proof=json.loads((extra.parent/'reconstruction.json').read_text());assert proof['files'][extra.name]==sha(extra)
        more=pd.read_csv(extra);assert set(more.method)=={'PBF'} and len(more)==100 and set(more.query_id)==set(frame.query_id)
        more['source_file']=str(extra);frame=pd.concat([frame[frame.method!='PBF'],more],ignore_index=True)
    extra=PBS_EXTRA.get((dataset,kind))
    if extra is not None and ((extra.parent/'reconstruction.json').exists() or os.environ.get('PBS_REACH_OPTIONAL')!='1'):
        proof=json.loads((extra.parent/'reconstruction.json').read_text());assert proof['files'][extra.name]==sha(extra)
        more=pd.read_csv(extra);assert set(more.method)==PBS_METHODS and more.groupby('method').size().eq(100).all() and set(more.query_id)==set(frame.query_id)
        more['source_file']=str(extra);frame=pd.concat([frame[~frame.method.isin(PBS_METHODS)],more],ignore_index=True)
    return frame


def present(frame):
    """Manuscript presentation: PTS is the depth-first policy, measured as the runner's DFS rows (same M=500, kappa=50,
    pruning, and references); the UCT rows (runner label PTS) stay only in the raw validated records."""
    frame=frame.copy()
    if 'expansion_budget' in frame and frame.method.eq('DFS').any():
        # the DFS rows ran with the CLI iteration budget (baseline_arms receives it) but the runner stores it on its PTS rows only
        budgets=frame.loc[frame.method=='PTS','expansion_budget'].dropna().unique();assert len(budgets)==1,budgets
        frame.loc[frame.method=='DFS','expansion_budget']=frame.loc[frame.method=='DFS','expansion_budget'].fillna(budgets[0])
    if frame.method.eq('DFS').any():   # archived two-policy job: the UCT rows are dropped; a fresh job has depth-first PTS rows only
        frame=frame[frame.method!='PTS'];frame['method']=frame.method.replace({'DFS':'PTS'})
    # Pareto DP is the conventional DP without candidate reduction (runner label Pareto-DP-conv); the campaign's reduced
    # DP arm (runner label Pareto-DP, same recurrence as PBS unbounded) is not presented.
    if frame.method.eq('Pareto-DP-conv').any():
        frame=frame[frame.method!='Pareto-DP'];frame['method']=frame.method.replace({'Pareto-DP-conv':'Pareto-DP'})
    return frame


def has_baselines(dataset):
    """The extra frontier-recovery baselines (weighted-sum sweep, greedy, NSGA-II) are not part of the current job matrix."""
    return (SOURCES[(dataset,'baselines')].parent/'reconstruction.json').exists()


def check_pair(dataset):
    loaded=[]
    for kind in ['default','baselines']:
        with gzip.open(SOURCES[(dataset,kind)].parent/'frontiers.jsonl.gz','rt') as f:
            loaded.append({str(r['query_id']):r for r in map(json.loads,f)})
    assert loaded[0].keys()==loaded[1].keys()
    for q,primary in loaded[0].items():
        alternative=loaded[1][q]
        assert all(primary[k]==alternative[k] for k in ['budget','alpha','query_status','candidates'])
        if primary['reference_complete'] and alternative['reference_complete']:
            keys=lambda records:{tuple(round(p[k],9) for k in ['quality','cost_overage','rating']) for p in records}
            assert keys(primary['reference'])==keys(alternative['reference'])
    return dict(dataset=dataset,identical_fresh_inputs=len(loaded[0]),
                complete_reference_agreements=sum(a['reference_complete'] and loaded[1][q]['reference_complete'] for q,a in loaded[0].items()))


def interval(values,stat,seed=42):
    values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
    if not len(values):return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    samples=rng.choice(values,(2000,len(values)),replace=True)
    estimates=np.mean(samples,axis=1) if stat=='mean' else np.median(samples,axis=1)
    return tuple(np.quantile(estimates,[.025,.975]))


def summarize(frame):
    rows=[]
    for method in ORDER:
        s=frame[frame.method==method]
        if s.empty:continue
        eligible=s[s.metric_eligible==1]
        row=dict(method=method,n_requested=len(s),n_feasible=int(s.query_feasible.sum()),n_metric=len(eligible),
                 n_returned=int(s.success.sum()),n_completed=int(s.completed.fillna(False).astype(bool).sum()),
                 n_timed=int(s.method_wall_s.notna().sum()),n_timeout=int(s.timed_out.sum()),
                 n_guard=int((s.termination_reason=='combination_limit').sum()),
                 n_failure=int((s.solver_status=='solver_failure').sum()),
                 n_work_limit=int(s.termination_reason.isin(['iteration_limit','combination_limit','work_limit']).sum()),
                 recall=eligible.recall.mean(),hv=eligible.hv_ratio.mean(),reported_count=s.solves.mean(),
                 counter_kind='scalar_invocations' if method.startswith('WS') else 'objective_evaluations' if method.startswith('NSGA') else 'selection_criteria' if method=='Greedy' else 'optimizer_invocations',
                 method_median=s.method_wall_s.median(),method_p90=s.method_wall_s.quantile(.9),
                 total_median=s.total_s.median(),total_p90=s.total_s.quantile(.9),source_file=s.source_file.iloc[0])
        for name,values,stat in [('recall',eligible.recall,'mean'),('hv',eligible.hv_ratio,'mean'),
                                  ('method_median',s.method_wall_s,'median'),('total_median',s.total_s,'median')]:
            row[name+'_ci_lo'],row[name+'_ci_hi']=interval(values,stat)
        rows.append(row)
    return pd.DataFrame(rows)


def tradeoff(summary,dataset):
    fig,ax=plots.paper_panel(figsize=(10.,7.5))
    for index,row in summary.iterrows():
        x=row.total_median*1000;y=row.recall
        if not np.isfinite(x+y):continue
        ax.errorbar(x,y,xerr=[[max(0,x-row.total_median_ci_lo*1000)],[max(0,row.total_median_ci_hi*1000-x)]],
                    yerr=[[max(0,y-row.recall_ci_lo)],[max(0,row.recall_ci_hi-y)]],fmt='o',
                    color=plots.color(row.method),capsize=3,label=LABELS[row.method],markersize=8)
    ax.set_xscale('log');ax.set_xlabel('median retrieval + method time (ms)');ax.set_ylabel('mean frontier recall');ax.set_ylim(-.03,1.05)
    ax.legend(fontsize=20,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False)
    plots.save_panel(fig,str(OUT),dataset,'baselines_tradeoff',summary)


# Two renderings of the epsilon-grid sweep: the house 10 x 6 in panels (for 0.49 columnwidth placement) and a
# quarter-column variant for four panels in one column-wide row (0.235 columnwidth each). At that width a panel
# is 0.8 in wide, so the variant uses a smaller canvas, a reduced font scale, short labels, and one shared legend
# row (eps_grid_legend) instead of per-panel legends.
QUARTER_FIGSIZE=(4.8,3.3);QUARTER_FONT_SCALE=plots.PAPER_FONT_SCALE  # same font-to-canvas ratio as the house panel, so text renders at the same size

def quarter_panel():
    import seaborn as sns
    sns.set_style('whitegrid');sns.set_context('paper',font_scale=QUARTER_FONT_SCALE)
    plots.plt.rcParams['lines.linewidth']=plots.PAPER_LINEWIDTH*.6;plots.plt.rcParams['lines.markersize']=plots.PAPER_MARKERSIZE*.6
    fig,ax=plots.plt.subplots(figsize=QUARTER_FIGSIZE);return fig,ax

def grid_sweep(default,dataset,folder):
    """Supplementary epsilon-grid sweep from the default job: median recall and median latency with interquartile
    bands against g, PBS and PTS medians as reference lines; HV stays in table_eps_grid."""
    rows=[]
    for g in SWEEP_GRIDS:
        s=default[default.method==f'eps-ILP g={g}'];e=s[s.metric_eligible==1]
        q=lambda v,p:float(v.quantile(p))
        rows.append(dict(grid=g,n_requested=len(s),n_metric=len(e),n_returned=int(s.success.sum()),n_timeout=int(s.timed_out.sum()),runs_mean=float(s.solves.mean()),
                         recall_mean=float(e.recall.mean()),recall_median=q(e.recall,.5),recall_q25=q(e.recall,.25),recall_q75=q(e.recall,.75),
                         hv_mean=float(e.hv_ratio.mean()),hv_median=q(e.hv_ratio,.5),hv_q25=q(e.hv_ratio,.25),hv_q75=q(e.hv_ratio,.75),
                         total_median_s=q(s.total_s,.5),total_q25_s=q(s.total_s,.25),total_q75_s=q(s.total_s,.75),total_p90_s=q(s.total_s,.9)))
    sweep=pd.DataFrame(rows);x=sweep.grid.to_numpy()
    refs={m:default[default.method==m] for m in ['PACS','PTS']}
    def recall_panel(ax,legend):
        plots._band(ax,x,sweep.recall_median.to_numpy(),sweep.recall_q25.to_numpy(),sweep.recall_q75.to_numpy(),plots.color('eps-ILP g=16'),r'$\epsilon$-ILP')
        for m,ls in [('PACS',':'),('PTS','-.')]:
            e=refs[m][refs[m].metric_eligible==1];ax.axhline(e.recall.median(),color=plots.color(m),ls=ls,lw=plots.plt.rcParams['lines.linewidth']*.8,label=LABELS[m])
        ax.set_ylim(-.03,1.05)
        if legend:ax.legend(loc='upper left',handlelength=1.4)
    def time_panel(ax,legend,annotate_size):
        plots._band(ax,x,sweep.total_median_s.to_numpy()*1000,sweep.total_q25_s.to_numpy()*1000,sweep.total_q75_s.to_numpy()*1000,plots.color('eps-ILP g=16'),r'$\epsilon$-ILP')
        for m,ls in [('PACS',':'),('PTS','-.')]:
            ax.axhline(refs[m].total_s.median()*1000,color=plots.color(m),ls=ls,lw=plots.plt.rcParams['lines.linewidth']*.8,label=LABELS[m])
        for g,r,t in zip(sweep.grid,sweep.runs_mean,sweep.total_q75_s*1000):ax.annotate(f'{r:.0f}',(g,t),textcoords='offset points',xytext=(0,3),ha='center',fontsize=annotate_size)
        ax.set_yscale('log')
        if legend:ax.legend(loc='lower right',handlelength=1.4)
    def axis(ax,xlabel):
        ax.set_xscale('log',base=2);ax.set_xticks(SWEEP_GRIDS);ax.set_xticklabels([str(g) for g in SWEEP_GRIDS]);ax.minorticks_off();ax.set_xlabel(xlabel)
    # house panels
    fig,ax=plots.paper_panel();recall_panel(ax,True);axis(ax,'grid size $g$');ax.set_ylabel('frontier recall');plots.save_panel(fig,str(OUT),folder,'eps_grid_quality',sweep)
    fig,ax=plots.paper_panel();time_panel(ax,True,'x-small');axis(ax,'grid size $g$');ax.set_ylabel('time (ms)');plots.save_panel(fig,str(OUT),folder,'eps_grid_time',sweep)
    # quarter-column variant
    fig,ax=quarter_panel();recall_panel(ax,False);axis(ax,'$g$');ax.set_ylabel('recall');ax.set_yticks([0,.5,1]);plots.save_panel(fig,str(OUT),folder,'eps_grid_quality_quarter',sweep)
    fig,ax=quarter_panel();time_panel(ax,False,'xx-small');axis(ax,'$g$');ax.set_ylabel('time (ms)');ax.set_yticks([10,100,1000,10000]);ax.set_yticklabels(['10','100','1k','10k']);ax.yaxis.set_minor_formatter(plots.plt.NullFormatter());ax.set_ylim(3,3e4);plots.save_panel(fig,str(OUT),folder,'eps_grid_time_quarter',sweep)
    if folder=='amazon':
        from matplotlib.lines import Line2D
        _f,_a=quarter_panel();plots.plt.close(_f)
        handles=[Line2D([],[],color=plots.color('eps-ILP g=16'),marker='o',label=r'$\epsilon$-ILP (median, IQR band)'),Line2D([],[],color=plots.color('PACS'),ls=':',label='PBS median'),Line2D([],[],color=plots.color('PTS'),ls='-.',label='PTS median')]
        fig=plots.plt.figure(figsize=(10,.7));fig.legend(handles=handles,loc='center',ncol=3,frameon=False,handlelength=1.6,columnspacing=1.2,fontsize=24)
        for ext in ['pdf','png']:fig.savefig(OUT/folder/f'eps_grid_legend.{ext}',bbox_inches='tight',pad_inches=.03)
        plots.plt.close(fig)
    plots.paper_panel()  # restore the house context for later panels
    return sweep


def grid_table(sweeps):
    lines=[r'\setlength{\tabcolsep}{3.5pt}%',r'\begin{tabular}{lrrrrrrrrr}',r'\toprule',r'& & \multicolumn{4}{c}{Amazon} & \multicolumn{4}{c}{Airbnb} \\',r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',r'grid & runs & recall & HV & median & p90 & recall & HV & median & p90 \\',r'\midrule']
    for g in SWEEP_GRIDS:
        cells=[f'$g{{=}}{g}$',f"{int(sweeps['amazon'].set_index('grid').loc[g,'runs_mean'])}"]
        for dataset in ['amazon','airbnb']:
            r=sweeps[dataset].set_index('grid').loc[g];cells+=[f'{r.recall_mean:.3f}',f'{r.hv_mean:.3f}',_fmt_time(r.total_median_s),_fmt_time(r.total_p90_s)]
        lines.append(' & '.join(cells)+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}']
    (OUT/'table_eps_grid.tex').write_text('% generated by scripts/exp7/build_default.py: epsilon-ILP grid sweep from the default jobs; recall and HV are means over metric-eligible queries, runs is the mean number of ILP solves per query, times are median and p90 fresh retrieval plus method wall time\n'+'\n'.join(lines)+'\n')



def house_table(table):
    """Restyle make_tables output to the manuscript convention (paper commits 4981f45, b976fd2):
    proposed methods (PBS, PTS, PBS unbounded) above one midrule, every baseline below it, no bold,
    no colour."""
    lines=table.split('\n')
    head=lines[:lines.index(r'\midrule')+1]
    body=[l for l in lines[len(head):] if l not in (r'\midrule',r'\bottomrule',r'\end{tabular}')]
    tail=[r'\bottomrule',r'\end{tabular}']
    def key(l):
        s=l.split('&')[0]
        return 0 if (r'\texttt{PBS}' in s or r'\texttt{PTS}' in s or 'unbounded' in s) else 1
    proposed=[l for l in body if key(l)==0];baselines=[l for l in body if key(l)==1]
    assert len(proposed)==3 and baselines, body
    out=[]
    for l in proposed+[r'\midrule']+baselines:
        l=re.sub(r'\\textbf\{([^}]*)\}',r'\1',l)  # no emphasis and no revision colour in the tables (user request 2026-09-19)
        out.append(l)
    return '\n'.join(head+out+tail).replace('Pareto DP &','DP &')  # the tables and text call the baseline DP (2026-09-19)

def baseline_table(summary,dataset):
    chosen=summary[summary.method.isin(['PBS-unbounded','Pareto-DP','WS-28','WS-253','Greedy','NSGA2-2000','NSGA2-10000'])]
    lines=[r'\setlength{\tabcolsep}{3.5pt}%',r'\begin{tabular}{lrrrr}',r'\toprule',r'method & recall & HV & median & p90 \\',r'\midrule']
    for row in chosen.itertuples():
        lines.append(' & '.join([LABELS[row.method],f'{row.recall:.3f}' if np.isfinite(row.recall) else '--',f'{row.hv:.3f}' if np.isfinite(row.hv) else '--',_fmt_time(row.total_median),_fmt_time(row.total_p90)])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/f'table_baselines_{dataset}.tex').write_text('\n'.join(lines)+'\n')


# Five short tick labels fit the panel horizontally at the house size; the shared panel code would rotate and shrink them.
_box_panel_orig=plots._box_panel
def _box_panel_override(*args,**kwargs):
    kwargs['tick_rotation']=None
    return _box_panel_orig(*args,**kwargs)
plots._box_panel=_box_panel_override

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plots.OUT_ROOT=OUT.parent
    # Every potential shared reader must use this exact explicit source map.
    aliases={('exp4_pplus_default','default'):('amazon','default'),('exp4_airbnb_default','default'):('airbnb','default'),
             ('exp4_pplus_frontier_baselines','baselines'):('amazon','baselines'),('exp4_airbnb_frontier_baselines','baselines'):('airbnb','baselines')}
    plots.read=lambda exp,name:present(read(*aliases[(exp,name)]))
    reference_counts=[]
    pairs=[];sweeps={}
    for dataset in ['amazon','airbnb']:
        default=present(read(dataset,'default'));baselines=present(read(dataset,'baselines')) if has_baselines(dataset) else None
        if baselines is not None:pairs.append(check_pair(dataset))
        combined=pd.concat([default,baselines[~baselines.method.isin(default.method.unique())]],ignore_index=True) if baselines is not None else default.copy()
        assert not combined.duplicated(['query_id','method']).any()
        summary=summarize(combined)
        summary.to_csv(OUT/f'{dataset}_summary.csv',index=False)
        combined.to_csv(OUT/f'{dataset}_source.csv',index=False)
        queries=default.drop_duplicates('query_id');eligible=queries[queries.metric_eligible==1]
        reference_counts.append(dict(dataset=dataset,n_requested=len(queries),n_complete_reference=len(eligible),
                                     median_size=float(eligible.ref_size.median()) if len(eligible) else None,
                                     maximum_size=int(eligible.ref_size.max()) if len(eligible) else None,
                                     source_file=str(SOURCES[(dataset,'default')])))
        folder=dataset;prefix='default_dist'
        # main-text distributions: the bounded searches, the epsilon grid, the exact Pareto DP (short label DP), and PBF
        distributions(default,str(OUT),folder,prefix,[('PACS','PBS'),('PTS','PTS'),('eps-ILP g=16',r'$\epsilon$-ILP'),('Pareto-DP','DP'),('PBF','PBF')])
        # All original arms and new methods remain available in supplemental panels.
        distributions(combined,str(OUT),dataset,'all_methods',[(m,LABELS[m]) for m in ORDER if m in set(combined.method)])
        # Section 3 presents one PTS policy (depth first); plots.read returns the presented frame, so the PTS row is the DFS measurement.
        presented_csv=OUT/f'{dataset}_default_presented.csv';default.to_csv(presented_csv,index=False)  # table_pareto reads a CSV path directly
        main_csv=OUT/f'{dataset}_default_main_table.csv';default[~default.method.str.startswith('eps-ILP')|default.method.isin([f'eps-ILP g={g}' for g in MAIN_GRIDS])].to_csv(main_csv,index=False)
        table=house_table(table_pareto(beam=100,src=str(main_csv)).replace(r'\texttt{BF}',r'\texttt{PBF}'))
        assert 'DFS' not in table and 'UCT' not in table and 'g{=}2' not in table and 'g{=}8' not in table
        sweeps[dataset]=grid_sweep(default,dataset,folder)
        (OUT/f'table_pareto_{dataset}.tex').write_text(table+'\n')
        if baselines is None:continue
        baseline_table(summary,dataset)
        tradeoff(summary,dataset)
        diagnostic=baselines.groupby('query_id').supported_frac.first()
        (OUT/f'{dataset}_supported_lattice.json').write_text(json.dumps(dict(lattice_resolution=50,mean=float(diagnostic.mean()) if diagnostic.notna().any() else None,median=float(diagnostic.median()) if diagnostic.notna().any() else None,n=int(diagnostic.notna().sum()),interpretation='approximate supported fraction sampled on the finite weight lattice with 1e-9 objective tie tolerance, not a supported-point ceiling'),indent=2)+'\n')
    grid_table(sweeps)
    pd.DataFrame(reference_counts).to_csv(OUT/'reference_summary.csv',index=False)
    code=REPO
    diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    (OUT/'generation_source.diff').write_bytes(diff)
    generation=dict(builder_sha256=sha(__file__),source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),code_files={name:sha(code/name) for name in ['gquery/alg/exp4/make_tables.py','gquery/figures/plot_exp4.py','gquery/figures/default_status.py']})
    manifest=dict(generation=generation,paired_inputs=pairs,sources={f'{d}.{k}':dict(path=str(p),sha256=sha(p),reconstruction_sha256=sha(p.parent/'reconstruction.json'),job_manifest=str(p.parent.parent/(p.parent.name+'.job')/'source/manifest.json')) for (d,k),p in SOURCES.items() if (p.parent/'reconstruction.json').exists()},
                  shared_method_precedence='default job; baseline duplicate measurements retained in raw source only',
                  presentation='PTS is the depth-first policy (runner label DFS, same budgets and references); the UCT rows (runner label PTS) are excluded from every panel and table and remain only in the raw records. Pareto DP is the conventional DP without candidate reduction (runner label Pareto-DP-conv, measured in the *_dpconv jobs); the reduced DP arm (runner label Pareto-DP) is excluded and remains only in the raw records. Epsilon grids: the main table shows g=4 and g=16, the supplementary sweep (eps_grid_quality: median recall with interquartile band, eps_grid_time: median latency with interquartile band and the ILP run count per query annotated; house 10x6 panels plus *_quarter variants and eps_grid_legend for four panels in one column-wide row; table_eps_grid: means, HV, and median/p90 times as in the main table) shows g=4, 8, 16; g=2 is measured but not presented',
                  uncertainty='95 percent percentile bootstrap interval across saved queries, 2000 resamples, seed 42; no repeated-run noise estimate',
                  deferred=['Amazon and Airbnb envelope panels require fresh sensitivity measurements'],
                  generation_command=sys.executable+' '+str(Path(__file__).resolve()),
                  generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
