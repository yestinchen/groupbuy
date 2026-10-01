"""Explicit fresh sensitivity sources, status denominators, and envelope panels."""
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
ROOT=CAMPAIGN
OUT=FIGURES/'exp7_sensitivity'
# Amazon is the union of the interrupted full job (a whole-setting checkpoint) and its continuation job.
SOURCES={'amazon':[ROOT/'sensitivity'/name/'sensitivity.csv' for name in ['amazon','amazon_resume']],
         'airbnb':[ROOT/'sensitivity/airbnb/sensitivity.csv'],'airbnb_msweep':[ROOT/'sensitivity/airbnb_msweep/sensitivity.csv']}
# Depth-first policy jobs supply every PTS series (the manuscript's single PTS policy); their PBS rows duplicate the base jobs.
DFS_SOURCES={'amazon':ROOT/'sensitivity/amazon_dfs/sensitivity.csv','airbnb':ROOT/'sensitivity/airbnb_dfs/sensitivity.csv','airbnb_msweep':ROOT/'sensitivity/airbnb_msweep_dfs/sensitivity.csv'}
# PBS rerun with the reachability bound in FeasibleExtensions (2026-10-01): every PBS row (runner label PACS) of the
# beam, alpha and beta sweeps comes from these jobs; every other row stays from the sources above. Required unless
# PBS_REACH_OPTIONAL=1 (only for testing the builder before the jobs finish).
REACH_SOURCES={'amazon':ROOT/'sensitivity/amazon_reach/sensitivity.csv','airbnb':ROOT/'sensitivity/airbnb_reach/sensitivity.csv'}
PBS_METHODS={'PACS'}
_spec=importlib.util.spec_from_file_location('reconstruct',Path(__file__).resolve().parent/'reconstruct_sensitivity.py')
reconstruct=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(reconstruct)
LABELS={'PACS':'PBS','PTS':'PTS','PBF':'PBF','eps-ILP g=8':r'$\epsilon$-ILP $g=8$'}
COMPLETE_LABEL='PBS unbounded';COMPLETE_COLOR='#2E5E8E'   # label and colour of PBS-unbounded in build_default
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(name):
    parts=[]
    for path in SOURCES[name]:
        if not (path.parent/'reconstruction.json').exists():continue   # the archived Amazon continuation job has no fresh counterpart
        proof=json.loads((path.parent/'reconstruction.json').read_text())
        assert proof['files'][path.name]==sha(path)
        part=pd.read_csv(path);part['source_file']=str(path);parts.append(part)
    df=pd.concat(parts,ignore_index=True)
    assert not df.duplicated(['sweep','param','method','query_id']).any()
    assert df.groupby(['sweep','param','method']).size().eq(100).all()
    # The first source's execution parameters carry the complete planned schedule; the union must cover exactly that.
    first=SOURCES[name][0];params=json.loads((first.parent.parent/(first.parent.name+'.job')/'execution_params.json').read_text())
    groups=reconstruct.planned_groups(params)
    assert set(map(tuple,df[['sweep','param','method']].drop_duplicates().to_numpy()))==groups
    assert len(df)==100*len(groups),(name,len(df),100*len(groups))
    return df

def read_dfs(name):
    """PTS-DFS rows of the validated depth-first job, or None when it has not run yet."""
    path=DFS_SOURCES[name]
    if not (path.parent/'reconstruction.json').exists():return None
    proof=json.loads((path.parent/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path)
    params=json.loads((path.parent.parent/(path.parent.name+'.job')/'execution_params.json').read_text());assert params['pts_traversal']=='depth_first'
    df=pd.read_csv(path);df['source_file']=str(path);df=df[df.method.str.startswith('PTS-DFS')]
    assert set(map(tuple,df[['sweep','param','method']].drop_duplicates().to_numpy()))=={g for g in reconstruct.planned_groups(params) if g[2].startswith('PTS-DFS')}
    assert df.groupby(['sweep','param','method']).size().eq(100).all();return df

def read_reach(name):
    """PBS rows of the validated reachability rerun, or None when it is absent and PBS_REACH_OPTIONAL=1."""
    path=REACH_SOURCES[name]
    if not (path.parent/'reconstruction.json').exists():
        if os.environ.get('PBS_REACH_OPTIONAL')=='1':return None
        raise FileNotFoundError(f'PBS reachability rerun not validated: {path}')
    proof=json.loads((path.parent/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path)
    params=json.loads((path.parent.parent/(path.parent.name+'.job')/'execution_params.json').read_text())
    df=pd.read_csv(path);df['source_file']=str(path);df=df[df.method.isin(PBS_METHODS)]
    assert set(map(tuple,df[['sweep','param','method']].drop_duplicates().to_numpy()))=={g for g in reconstruct.planned_groups(params) if g[2] in PBS_METHODS}
    assert not df.duplicated(['sweep','param','method','query_id']).any()
    assert df.groupby(['sweep','param','method']).size().eq(100).all();return df

def present(frame):
    """Manuscript presentation: PTS is the depth-first policy. The UCT rows (runner label PTS) are dropped and the
    depth-first rows (runner label PTS-DFS) take the PTS name; the raw validated records keep both."""
    if not frame.method.str.startswith('PTS-DFS').any():return frame.copy()   # fresh job: PTS rows are depth first already
    uct=frame.method.str.match(r'^PTS(?:$| |-)')&~frame.method.str.startswith('PTS-DFS')
    frame=frame[~uct].copy();frame['method']=frame.method.str.replace('PTS-DFS','PTS',regex=False);return frame

def summary(frame):
    rows=[]
    for (sweep,param,method),g in frame.groupby(['sweep','param','method'],sort=False):
        eligible=g[g.metric_eligible==1]
        row=dict(sweep=sweep,param=param,method=method,n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=len(eligible),n_returned=int(g.success.sum()),n_timed=int(g.method_wall_s.notna().sum()),n_complete=int(g.completed.fillna(False).astype(bool).sum()),n_timeout=int(g.timed_out.sum()),n_exhausted=int(g.exhausted.fillna(False).astype(bool).sum()),source_file='|'.join(g.source_file.unique()))
        for status in ['missing_embedding','empty_requirement','budget_infeasible']:
            row['n_'+status]=int((g.query_status==status).sum())
        for status in ['complete','reference_timeout','reference_failure','reference_memory_limit']:
            row['n_'+status if status!='complete' else 'n_reference_complete']=int((g.reference_status==status).sum())
        for metric in ['recall','hv_ratio','ref_size','size','method_wall_s','total_s','evals','iterations','n_after']:
            selected=eligible if metric in ['recall','hv_ratio','ref_size','size'] else g
            for suffix,quantile in [('median',.5),('q25',.25),('q75',.75),('p90',.9)]:
                row[metric+'_'+suffix]=selected[metric].quantile(quantile)
            row[metric+'_mean']=selected[metric].mean()
        rows.append(row)
    order={m:i for i,m in enumerate(['PACS','PTS','eps-ILP g=8','PBF'])}  # legend order: PBS, PTS, then the baselines
    return pd.DataFrame(rows).sort_values(['sweep','method'],key=lambda c:c.map(order).fillna(99) if c.name=='method' else c,kind='stable').reset_index(drop=True)

def band(data,folder,stem,metric,xlabel,ylabel,logx=False,logy=False,default=None,categorical=False):
    fig,ax=plots.paper_panel()
    for method,g in data.groupby('method',sort=False):
        g=g.sort_values('param',key=lambda s:s.replace(0,np.inf)) if categorical else g.sort_values('param')
        x=np.arange(len(g)) if categorical else g.param.to_numpy()
        y=g[metric+'_median'].to_numpy();lo=g[metric+'_q25'].to_numpy();hi=g[metric+'_q75'].to_numpy()
        ax.plot(x,y,marker='o',color=plots.color(method),label=LABELS.get(method,method))
        ax.fill_between(x,lo,hi,color=plots.color(method),alpha=.18)
        if categorical:
            ax.set_xticks(x,['none' if p==0 else f'{p:g}' for p in g.param])
        else:ax.set_xticks(sorted(data.param.unique()))
    if logx:ax.set_xscale('log');ax.xaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'))
    if logy:ax.set_yscale('log')
    if metric in ['recall','hv_ratio']:ax.set_ylim(-.03,1.05)
    if default is not None:ax.axvline(default,color='#666666',ls=':',lw=2)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    if data.method.nunique()>1:ax.legend(loc='best')
    plots.save_panel(fig,str(OUT),folder,stem,data)

def beta(data,folder):
    # The appendix presents PBS and PTS in this panel; the baselines' rows stay in the summary CSV.
    data=data[data.method.isin(['PACS','PTS'])]
    # quality is undefined where no query has a complete reference (beta=0.9 on Amazon): those settings leave the axis
    defined=sorted(data[data.n_metric>0].param.unique());quality=data[data.param.isin(defined)]
    # House style must not depend on an earlier panel having set it (this function is also imported standalone).
    _f,_a=plots.paper_panel();plots.plt.close(_f)
    # Single panels in the standard size, for the standard subfigure layouts.
    for metric,ylabel,stem in [('recall','mean recall','matching_threshold_recall'),('hv_ratio','mean HV ratio','matching_threshold_hv')]:
        fig,ax=plots.paper_panel()
        for method,g in quality.groupby('method',sort=False):
            g=g.sort_values('param');ax.plot(g.param,g[metric+'_mean'],marker='o',color=plots.color(method),label=LABELS[method])
        ax.set_ylim(-.03,1.05);ax.set_ylabel(ylabel);ax.set_xlabel(r'matching threshold $\beta$');ax.set_xticks(defined);ax.legend(loc='upper left')
        plots.save_panel(fig,str(OUT),folder,stem,quality)
    fig,ax=plots.paper_panel();g=data[data.method=='PACS'].sort_values('param')
    for col,label,color in [('n_feasible','feasible','#2E5E8E'),('n_metric','complete reference','#82B366'),('n_empty_requirement','empty requirement','#B85450'),('n_budget_infeasible','budget infeasible','#9673A6')]:
        ax.plot(g.param,g[col],marker='o',label=label,color=color)
    ax.set_ylabel('queries / 100');ax.set_ylim(-3,103);ax.set_xlabel(r'matching threshold $\beta$');ax.set_xticks(sorted(data.param.unique()));ax.legend(loc='center left')
    plots.save_panel(fig,str(OUT),folder,'matching_threshold_counts',g)
    fig,axes=plots.plt.subplots(3,1,figsize=(10,15),layout='constrained')
    # short labels: at the paper font scale the long names overrun the axis height and are clipped
    for ax,metric,ylabel in zip(axes[:2],['recall','hv_ratio'],['mean recall','mean HV ratio']):
        for method,g in data.groupby('method',sort=False):
            g=g.sort_values('param');ax.plot(g.param,g[metric+'_mean'],marker='o',color=plots.color(method),label=LABELS[method])
        ax.set_ylim(-.03,1.05);ax.set_ylabel(ylabel)
    g=data[data.method=='PACS'].sort_values('param')  # query statuses are per query, identical across methods
    for col,label,color in [('n_feasible','feasible','#2E5E8E'),('n_metric','complete reference','#82B366'),('n_empty_requirement','empty requirement','#B85450'),('n_budget_infeasible','budget infeasible','#9673A6')]:
        axes[2].plot(g.param,g[col],marker='o',label=label,color=color)
    axes[2].set_ylabel('queries / 100');axes[2].set_ylim(-3,103)
    axes[0].legend(loc='upper left',ncol=2,columnspacing=.8,handlelength=1.5);axes[2].legend(loc='best')  # two columns keep the legend above the low-threshold PTS lines
    for ax in axes:ax.set_xlabel(r'matching threshold $\beta$');ax.set_xticks(sorted(data.param.unique()))
    plots.save_panel(fig,str(OUT),folder,'matching_threshold',data)

def envelope(frame,default,folder):
    selected=[]
    # the starred default points come from the validated default job (presented: its PTS rows are the depth-first measurement)
    for sweep,method,value,default_method in [('beam','PACS',100,'PACS'),('budget','PTS',500,'PTS')]:
        part=frame[(frame.sweep==sweep)&(frame.method==method)&(frame.param!=value)]
        if part.empty:continue
        center=default[default.method==default_method].copy();center['sweep']=sweep;center['param']=value;center['method']=method
        selected.extend([part,center])
    data=summary(pd.concat(selected,ignore_index=True))
    # PBS without beam truncation as a reference: its measured arm in the default job (the *_reach rerun through
    # build_default's PBS override), one diamond at (its latency, recall 1.0); no reference lines (user, 2026-10-01)
    complete=default[default.method=='PBS-unbounded'].copy();assert len(complete)==100,len(complete)
    complete['sweep']='reference';complete['param']=0   # one group; groupby drops NaN keys
    reference=summary(complete)
    for suffix,latency in [('', 'total_s_median'),('_p90','total_s_p90')]:
        fig,ax=plots.paper_panel()
        x_ref=float(reference[latency].iloc[0])*1000
        ax.scatter([x_ref],[1.0],marker='D',s=260,color=COMPLETE_COLOR,edgecolors='black',zorder=6,label=COMPLETE_LABEL)
        for method,g in data.groupby('method',sort=False):
            g=g.sort_values('param')
            x=g[latency]*1000;y=g.recall_mean
            ax.plot(x,y,marker='o',color=plots.color(method),label=LABELS[method])
            chosen=g.param.eq(100 if method=='PACS' else 500)
            ax.scatter(x[chosen],y[chosen],marker='*',s=1300,color=plots.color(method),edgecolors='black',zorder=5)
        ax.set_xscale('log');ax.set_xlabel(('median' if not suffix else 'p90')+' retrieval + method (ms)');ax.set_ylabel('mean frontier recall');ax.set_ylim(-.03,1.05);h,l=ax.get_legend_handles_labels();h,l=h[1:]+h[:1],l[1:]+l[:1]   # PBS, PTS, then the reference
        ax.legend(h,l,loc='lower center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False,fontsize='small',handlelength=1.5,columnspacing=1.0)  # above the axes, as in build_complete_search
        # explicit sparse decade-style ticks; the default minor labels collide on a narrow log range
        xs=np.append((data[latency]*1000).to_numpy(),x_ref);lo,hi=np.nanmin(xs),np.nanmax(xs)
        grid=[.5,1,2,3,5,10,20,30,50,100,200,500,1000] if hi/lo<30 else [.5,1,2,5,10,20,50,100,200,500,1000]   # no 3s on wide ranges
        ticks=[t for t in grid if lo/1.3<=t<=hi*1.3]
        ax.set_xticks(ticks);ax.xaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v,_:f'{v:g}'));ax.xaxis.set_minor_formatter(plots.plt.NullFormatter())
        plots.save_panel(fig,str(OUT),folder,'envelope'+suffix,pd.concat([data,reference],ignore_index=True))
    return data

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plots.OUT_ROOT=OUT.parent
    frames={name:read(name) for name in SOURCES};dfs_frames={name:read_dfs(name) for name in DFS_SOURCES}
    reach_frames={name:read_reach(name) for name in REACH_SOURCES}
    spec=importlib.util.spec_from_file_location('default_builder',Path(__file__).resolve().parent/'build_default.py')
    default_builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(default_builder)
    for dataset in ['amazon','airbnb']:
        frame=frames[dataset]
        if dataset=='airbnb':
            extra=frames['airbnb_msweep'];existing=set(frame[frame.sweep=='budget'].param)
            frame=pd.concat([frame,extra[~extra.param.isin(existing)]],ignore_index=True)
        for key in ([dataset]+(['airbnb_msweep'] if dataset=='airbnb' else [])):
            dfs=dfs_frames.get(key)
            if dfs is None:continue
            if key=='airbnb_msweep':dfs=dfs[~dfs.param.isin(existing)]
            frame=pd.concat([frame,dfs],ignore_index=True)
        reach=reach_frames[dataset]
        if reach is not None:
            # replace the PBS rows group for group: same sweeps, parameters and queries as the rows they replace
            base=frame[frame.method.isin(PBS_METHODS)]
            assert set(map(tuple,base[['sweep','param','method']].drop_duplicates().to_numpy()))==set(map(tuple,reach[['sweep','param','method']].drop_duplicates().to_numpy()))
            assert set(base.query_id)==set(reach.query_id)
            frame=pd.concat([frame[~frame.method.isin(PBS_METHODS)],reach],ignore_index=True)
        frame=present(frame);assert not frame.duplicated(['sweep','param','method','query_id']).any()
        data=summary(frame);data.to_csv(OUT/f'{dataset}_summary.csv',index=False)
        folder=dataset
        for sweep,methods,tag,xlabel,default in [('beam',['PACS'],'pbs',r'beam width $b$',100),('budget',['PTS'],'pts',r'iteration budget $M$',500),('kappa',['PTS'],'cap',r'candidate cap $\kappa$',None)]:
            part=data[(data.sweep==sweep)&(data.method.isin(methods))]
            for metric,suffix,ylabel,logy in [('recall','recall','frontier recall',False),('total_s','time','retrieval + method (s)',True),('method_wall_s','method_time','method time (s)',True),('evals','work','candidate evaluations',True)]:
                band(part,folder,f'search_control_{tag}_{suffix}',metric,xlabel,ylabel,sweep!='kappa',logy,default,sweep=='kappa')
        alpha=data[data.sweep=='alpha']
        band(alpha[alpha.method=='PTS'].assign(method='reference'),folder,'alpha_size','ref_size',r'budget relaxation $\alpha$','complete frontier size',default=.5)
        band(alpha,folder,'alpha_recall','recall',r'budget relaxation $\alpha$','frontier recall',default=.5)
        beta(data[data.sweep=='beta'],folder)
        default=default_builder.present(default_builder.read(dataset,'default'))
        # Default does not expose all sensitivity-only counters, which remain undefined.
        for column in ['iterations','n_after']:
            if column not in default:default[column]=np.nan
        envelope(frame,default,folder)
    code=REPO
    diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    (OUT/'generation_source.diff').write_bytes(diff)
    manifest=dict(builder_sha256=sha(__file__),presentation='PTS is the depth-first policy: PTS series come from the *_dfs jobs (runner label PTS-DFS) and the UCT rows of the base jobs are excluded from every panel; both remain in the raw records',generation_command=sys.executable+' '+str(Path(__file__).resolve()),helpers={str(p):sha(p) for p in [Path(__file__).resolve().parent/'build_default.py',code/'gquery/figures/plot_exp4.py',code/'gquery/figures/default_status.py']},source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),sources={name:[dict(path=str(p),sha256=sha(p),reconstruction_sha256=sha(p.parent/'reconstruction.json')) for p in paths] for name,paths in SOURCES.items()},reach_sources={name:dict(path=str(p),sha256=sha(p),reconstruction_sha256=sha(p.parent/'reconstruction.json')) for name,p in REACH_SOURCES.items() if (p.parent/'reconstruction.json').exists()},default_sources={dataset:dict(path=str(default_builder.SOURCES[(dataset,'default')]),sha256=sha(default_builder.SOURCES[(dataset,'default')])) for dataset in ['amazon','airbnb']},precedence='Amazon rows are the disjoint union of the interrupted full job (its whole-setting checkpoint) and the continuation job covering the remaining planned settings. Main sensitivity job for overlapping M values, separate Airbnb M32000, default job for PBS100 and PTS500 envelope points. PBS rows (PACS) of every sweep come from the *_reach jobs (reachability bound in FeasibleExtensions) and replace those of the sources above group for group. No pooling of duplicate measurements.',quality_denominator='feasible complete nonempty reference, with failed empty method output zero',time_denominator='every attempted measured method invocation, including partial outputs and failures',generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
