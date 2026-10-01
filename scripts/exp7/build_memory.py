"""Memory tables from validated isolated measurements: catalog-level host and
GPU peaks plus per-method isolated search peaks. The epsilon arm here is the
grid-4 isolated process (memory-grid harness), presented beside the deadline grids 4 and 16."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
CAMPAIGN=Path(os.environ.get('CAMPAIGN_ROOT',REPO/'results'/'campaign'))
FIGURES=Path(os.environ.get('FIGURES_ROOT',REPO/'data_figures'/'exp7'))
MEMORY_GRIDS=Path(os.environ.get('MEMORY_GRIDS_ROOT',REPO/'results'/'memory_grids'))
ROOT=Path(os.environ.get('CAMPAIGN_MEMORY_ROOT',CAMPAIGN/'memory'))
OUT=Path(os.environ.get('CAMPAIGN_MEMORY_OUT',FIGURES/'exp7_memory'))
# The presented epsilon arm is grid 4, measured in isolated processes by the manuscript owner's memory-grid harness
# (scripts/exp4/memory_grid_jobs.json) on the campaign's exact inputs; its records are re-checked here.
GRIDS=Path(os.environ.get('CAMPAIGN_MEMORY_GRIDS',MEMORY_GRIDS))
LADDER=[('e11_cat10k','110K'),('e11_cat30k','330K'),('e12_cat1m','1.1M'),('e12_cat2m','2.14M'),('airbnb_12c','Airbnb 231K')]
METHOD_LABEL={'PBS':r'\texttt{PBS}','PTS':r'\texttt{PTS}','eps-ILP g=4':r'$\epsilon$-ILP ($g{=}4$)','eps-ILP':r'$\epsilon$-ILP ($g{=}2$)','PBS-unbounded':r'\texttt{PBS} unb.','Pareto-DP':'DP'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_grid(setting,original):
    """The g=4 isolated process from the memory-grid harness, checked against its validation record and the campaign inputs."""
    path=GRIDS/setting/f'mem_{setting}.json';vpath=GRIDS/setting/'validation.json';val=json.loads(vpath.read_text())
    assert val['record_sha256']==sha(path) and val['setting']==setting and val['queries']==100
    rec=json.loads(path.read_text());child=rec['search_processes']['eps-ILP-g4']
    assert child['config']['eps_grid']==4 and child['requested']==len(child['rows'])==100 and rec['status_counts']=={'feasible':100} and child['sampling_interval_s']==.001
    assert [r['query_id'] for r in child['rows']]==[r['query_id'] for r in original['search_processes']['eps-ILP']['rows']],'grid run queries differ from the campaign'
    assert abs(child['sampled_search_peak_rss_mb']-child['candidate_resident_baseline_mb']-child['sampled_search_increase_mb'])<1e-8
    assert all(r['solver_backend']=='scipy.optimize.milp/HiGHS' and r['highs_threads_configured']==1 for r in child['rows'])
    return child,{str(path):sha(path),str(vpath):sha(vpath)}

def read_conv(setting,original):
    """The conventional Pareto DP process (R6), measured alone in the *_dpconv job on the same queries; optional until validated."""
    d=ROOT/(setting+'_dpconv');path=d/f'mem_{setting}.json'
    if not (d/'reconstruction.json').exists():return None,{}
    proof=json.loads((d/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path)
    rec=json.loads(path.read_text());child=rec['search_processes']['Pareto-DP-conv']
    assert [r['query_id'] for r in child['rows']]==[r['query_id'] for r in original['search_processes']['PBS']['rows']] and child['requested']==100
    return child,{str(path):sha(path)}

PBS_METHODS=('PBS','PBS-unbounded')

def read_reach(setting,original):
    """PBS and unbounded PBS rerun with the reachability bound in FeasibleExtensions (*_reach job, 2026-10-01) on the same
    queries; their processes replace the campaign PBS processes. Required unless PBS_REACH_OPTIONAL=1 (builder testing only)."""
    d=ROOT/(setting+'_reach');path=d/f'mem_{setting}.json'
    if not (d/'reconstruction.json').exists() and os.environ.get('PBS_REACH_OPTIONAL')=='1':return None,{}
    proof=json.loads((d/'reconstruction.json').read_text());assert proof['files'][path.name]==sha(path)
    rec=json.loads(path.read_text());assert rec['n_items']==original['n_items']
    children={m:rec['search_processes'][m] for m in PBS_METHODS}
    for c in children.values():
        assert [r['query_id'] for r in c['rows']]==[r['query_id'] for r in original['search_processes']['PBS']['rows']] and c['requested']==100
    return children,{str(path):sha(path)}

def present(children,grid4,conv=None,reach=None):
    """PTS is the depth-first policy (runner process DFS; the UCT process is not presented); the epsilon arm is the
    isolated g=4 process (the campaign's g=2 process is not presented)."""
    out={};archived='DFS' in children
    for m,c in children.items():
        if m=='eps-ILP' or (archived and m=='PTS'):continue
        out['PTS' if m=='DFS' else m]=c
    out['eps-ILP g=4']=grid4
    # Pareto DP is the conventional DP without candidate reduction: the *_dpconv process of the archived campaign, or the
    # Pareto-DP-conv process of a fresh job; the archived reduced DP process is not presented
    if conv is not None:out['Pareto-DP']=conv
    elif 'Pareto-DP-conv' in out:out['Pareto-DP']=out.pop('Pareto-DP-conv')
    if reach is not None:out.update(reach)
    return out

def read(setting):
    path=ROOT/setting/f'mem_{setting}.json';proof=json.loads((path.parent/'reconstruction.json').read_text())
    assert proof['files'][path.name]==sha(path) and proof['kind']=='memory' and proof['setting']==setting
    return json.loads(path.read_text()),path

def main():
    OUT.mkdir(parents=True,exist_ok=True);records={};sources={}
    grids={};convs={};reaches={}
    for setting,_ in LADDER:
        records[setting],sources[setting]=read(setting)
        grids[setting],extra=read_grid(setting,records[setting]);sources.update({k:v for k,v in extra.items()})
        convs[setting],extra=read_conv(setting,records[setting]);sources.update(extra)
        reaches[setting],extra=read_reach(setting,records[setting]);sources.update(extra)
    catalog=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'catalog & items & host RSS after retrieval & peak GPU & emb.\ matrix & max isolated search peak \\',r'\midrule']
    rows=[]
    for setting,label in LADDER:
        rec=records[setting];children=present(rec['search_processes'],grids[setting],convs[setting],reaches[setting])
        gpu=max(v for v in [rec['gpu_after_load_mb'],rec['gpu_peak_retrieval_mb']] if v is not None)
        # ru_maxrss of a child forked from the large parent reports the parent's high-water mark, so the isolated
        # search peak is the child's own 1 ms RSS sampling (baseline of the stripped inputs plus the search increase).
        search_peak=max(c['sampled_search_peak_rss_mb'] for c in children.values())
        catalog.append(' & '.join([label,f"{rec['n_items']:,}",f"${rec['rss_after_retrieval_mb']/1024:.1f}$\\,GB",f"${gpu/1024:.1f}$\\,GB",f"${rec['embedding_matrix_mb']/1024:.1f}$\\,GB",f"${search_peak/1024:.2f}$\\,GB"])+r' \\')
        rows.append(dict(setting=setting,label=label,n_items=rec['n_items'],n_queries=rec['n_queries'],status_counts=json.dumps(rec['status_counts']),rss_baseline_mb=rec['rss_baseline_mb'],rss_after_imports_mb=rec['rss_after_imports_mb'],rss_after_load_mb=rec['rss_after_load_mb'],rss_after_retrieval_mb=rec['rss_after_retrieval_mb'],gpu_after_load_mb=rec['gpu_after_load_mb'],gpu_peak_retrieval_mb=rec['gpu_peak_retrieval_mb'],gpu_reserved_peak_mb=rec['gpu_reserved_peak_mb'],embedding_matrix_mb=rec['embedding_matrix_mb'],embedding_fp16_mb=rec['embedding_fp16_mb'],gpu_device=rec['gpu_device'],load_s=rec['load_s'],retrieval_s_total=rec['retrieval_s_total'],max_isolated_search_peak_mb=search_peak))
    catalog+=[r'\bottomrule',r'\end{tabular}']
    (OUT/'table_memory.tex').write_text('% generated by scripts/exp7/build_memory.py from validated revision records; host RSS is the parent process high-water mark after fresh retrieval; the isolated search peak is the largest sampled resident size (1 ms sampling) of the five presented per-method processes (PTS is the depth-first policy; the UCT process was measured but is not presented) holding only the stripped candidate inputs (epsilon grid 4)\n'+'\n'.join(catalog)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'memory_catalog.csv',index=False)
    methods=['PBS','PTS','eps-ILP g=4','PBS-unbounded','Pareto-DP']
    per=[r'\begin{tabular}{l'+'r'*len(methods)+'}',r'\toprule','catalog & '+' & '.join(METHOD_LABEL.get(m,m) for m in methods)+r' \\',r'\midrule']
    mrows=[]
    for setting,label in LADDER:
        children=present(records[setting]['search_processes'],grids[setting],convs[setting],reaches[setting]);cells=[label]
        for m in methods:
            c=children[m];cells.append(f"${c['sampled_search_peak_rss_mb']:.0f}$ (+${c['sampled_search_increase_mb']:.2f}$)")
            mrows.append(dict(setting=setting,method=m,process_peak_rss_mb=c['process_peak_rss_mb'],sampled_search_peak_rss_mb=c['sampled_search_peak_rss_mb'],sampled_search_increase_mb=c['sampled_search_increase_mb'],candidate_resident_baseline_mb=c['candidate_resident_baseline_mb'],peak_before_search_mb=c['peak_before_search_mb'],rss_samples=c['rss_samples'],sampling_interval_s=c['sampling_interval_s'],method_wall_s=c['method_wall_s'],requested=c['requested'],returned=c['returned'],timeouts=c['timeouts'],config=json.dumps(c['config'],sort_keys=True),pid=c['pid']))
        per.append(' & '.join(cells)+r' \\')
    per+=[r'\bottomrule',r'\end{tabular}']
    (OUT/'table_memory_methods.tex').write_text('% isolated per-method sampled peak RSS in MB of a fresh process holding only the stripped candidate inputs, with the sampled increase over its resident baseline in parentheses; ru_maxrss is not used because a child forked from the large parent inherits the parent high-water mark; epsilon arm is grid 4; 1 ms sampling cannot establish exact allocation peaks\n'+'\n'.join(per)+'\n')
    pd.DataFrame(mrows).to_csv(OUT/'memory_methods.csv',index=False)
    code=REPO;diff=subprocess.check_output(['git','-C',str(code),'diff','--binary','HEAD'])
    manifest=dict(builder_sha256=sha(__file__),presentation='PTS is the depth-first policy (runner process DFS); the UCT process is excluded from both tables and remains only in the raw records. The epsilon arm is the isolated g=4 process measured by the memory-grid harness on the campaign inputs (scripts/exp4/memory_grid_jobs.json, validated by scripts/exp7/validate_memory_grid.py and re-checked here); the campaign g=2 process is not presented. Pareto DP is the conventional DP without candidate reduction (*_dpconv process); the reduced DP process is not presented. PBS and unbounded PBS are the *_reach processes (reachability bound in FeasibleExtensions, 2026-10-01); the campaign PBS processes remain only in the raw records',source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
        sources={k:(dict(path=str(p),sha256=sha(p),reconstruction_sha256=sha(p.parent/'reconstruction.json')) if isinstance(p,Path) else dict(path=k,sha256=p)) for k,p in sources.items()},
        semantics='host RSS values are monotonic process high-water marks (baseline, after imports, after catalog load, after fresh retrieval); GPU peak is the larger of the post-load and retrieval allocation peaks on the selected device, reported separately from other occupants of the host GPUs; isolated search peaks are the sampled resident peaks (1 ms) of one fresh process per method with the stripped candidate inputs; the recorded ru_maxrss of those children equals the parent high-water mark (inherited across fork) and is retained in the raw records but not reported; the presented epsilon memory arm is the isolated grid-4 process from the memory-grid harness; the campaign grid-2 process is retained in the raw records only',
        generated={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('manifest') and p.name not in ('table_memory_increases.tex','memory_ilp_grids.csv')})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
