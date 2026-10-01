"""Independently reconstruct scale, length-budget, and large-catalog records
from their archived candidate arrays, without importing solvers."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ablation_validator',HERE/'reconstruct_ablation.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
geo=base.geo;sha=base.sha;load_archive=base.load_archive;check_record=base.check_record;check_metrics=base.check_metrics;check_timing=base.check_timing;params_of=base.params_of

def result(directory,files,**fields):
    out=dict(**fields,solvers_invoked=False,files={str(Path(p).relative_to(directory)):sha(p) for p in files},geometry_sha256=sha(HERE/'reconstruct_default.py'),ablation_validator_sha256=sha(HERE/'reconstruct_ablation.py'),validator_sha256=sha(__file__))
    (Path(directory)/'reconstruction.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='files'}),flush=True);return out

def pairs(text,default):
    # Mirrors bench_scale._parse: an absent option means the built-in ladder, an empty string skips the sweep.
    if text is None: return default
    return [(p.split(':')[0].strip(),float(p.split(':')[1])) for p in text.split(',') if p.strip()]

def check_epsilon(row):
    if row['method'].startswith('eps') and row.get('completed') is True:
        assert row['n_solves']==row['n_expected_solves'] and row['n_nonoptimal']==0 and not row['timed_out']

def validate_scale(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'scale.csv');params=params_of(directory)
    assert params['num_queries']==expected
    default_catalogs=[('e11_cat10k',10.),('e11_cat15k',15.),('e11_cat20k',20.),('e11_cat25k',25.),('e11_cat30k',30.)]
    default_lengths=[('e11_gs2',2.),('e11_gs4',4.),('e11_gs6',6.),('e11_gs8',8.)]
    settings=[('catalog',s,p) for s,p in pairs(params['catalogs'],default_catalogs)]+[('length',s,p) for s,p in pairs(params['lengths'],default_lengths)]
    pts='PTS' if params.get('pts_traversal','uct')=='uct' else 'PTS-DFS'
    methods=['PACS',pts]+([f"eps-ILP g={params['grid']}"] if params['grid']>0 else [])
    assert set(map(tuple,frame[['sweep','setting','param','method']].drop_duplicates().to_numpy()))=={(sw,s,p,m) for sw,s,p in settings for m in methods}
    assert not frame.duplicated(['sweep','setting','method','query_id']).any() and frame.groupby(['sweep','setting','method']).size().eq(expected).all()
    state=dict(error=0.);count=0;records=0;files=[directory/'scale.csv']
    for sweep,setting,param in settings:
        path=directory/'inputs'/f'{sweep}-{setting}'/'frontiers.jsonl.gz';files.append(path);saved=load_archive(path,expected)
        subset=frame[(frame.sweep==sweep)&(frame.setting==setting)];assert set(subset.query_id.astype(str))==set(saved);records+=len(saved)
        for q,raw in saved.items():
            rows=subset[subset.query_id.astype(str)==q];assert len(rows)==len(methods)
            eligible,corner,denominator,keys,verify,counter=check_record(raw,rows,params['beta'],params['alpha'])
            for row in rows.to_dict('records'):
                config=json.loads(row['method_config'])
                expected_config={'PACS':dict(beam=params['beam']),pts:dict(expansion_budget=params['expansion_budget'],candidate_limit=params['candidate_limit'])}.get(row['method'],dict(grid=params['grid'],per_solve_budget_s=params['timeout_seconds']))
                assert {k:config[k] for k in expected_config}==expected_config and config['timeout_seconds']==params['timeout_seconds'] and config.get('pts_traversal','uct')==params.get('pts_traversal','uct')
                front=json.loads(row['frontier_json']);verify(front)
                check_metrics(row,front,eligible,corner,denominator,keys,state);check_timing(row);check_epsilon(row)
                assert row['success']==int(bool(front)) and row['ref_time_s']==row['reference_s']
            count+=counter()
    counts=[dict(sweep=sw,setting=s,method=m,n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=int(g.metric_eligible.sum()),n_returned=int(g.success.sum()),n_failure=int((g.solver_status=='solver_failure').sum()),n_timeout=int(g.timed_out.sum()),n_timed=int(g.method_wall_s.notna().sum())) for (sw,s,m),g in frame.groupby(['sweep','setting','method'],sort=False)]
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    return result(directory,files,kind='scale',rows=len(frame),input_query_records=records,assignments_checked=count,max_hv_absolute_error=state['error'])

TIERS=[(100,500),(200,1000),(400,2000),(800,4000),(1600,8000)]

def validate_length_budget(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'length_budget.csv');params=params_of(directory)
    assert params['num_queries']==expected
    ladder=pairs(params['lengths'],[('e11_gs2',2.),('e11_gs4',4.),('e11_gs6',6.),('e11_gs8',8.)])
    pts='PTS' if params.get('pts_traversal','uct')=='uct' else 'PTS-DFS'
    arms={('PACS',float(b),float('nan')) for b,_ in TIERS}|{(pts,float('nan'),float(m)) for _,m in TIERS}
    key=lambda r:(r['method'],float(r['beam']) if r['method']=='PACS' else float('nan'),float(r['expansion_budget']) if r['method']==pts else float('nan'))
    def same(a,b):return a[0]==b[0] and all((math.isnan(x) and math.isnan(y)) or x==y for x,y in zip(a[1:],b[1:]))
    state=dict(error=0.);count=0;records=0;files=[directory/'length_budget.csv']
    for setting,n in ladder:
        path=directory/'inputs'/setting/'frontiers.jsonl.gz';files.append(path);saved=load_archive(path,expected)
        subset=frame[frame.setting==setting];assert set(subset.query_id.astype(str))==set(saved) and subset.n.eq(n).all();records+=len(saved)
        assert len(subset)==expected*len(arms)
        for q,raw in saved.items():
            rows=subset[subset.query_id.astype(str)==q];assert len(rows)==len(arms)
            seen=[key(r) for r in rows.to_dict('records')]
            assert all(any(same(k,a) for a in arms) for k in seen) and len(seen)==len(arms)
            eligible,corner,denominator,keys,verify,counter=check_record(raw,rows,params['beta'],params['alpha'])
            for row in rows.to_dict('records'):
                config=json.loads(row['method_config'])
                assert config['candidate_limit']==params['candidate_limit'] and config['timeout_seconds']==params['timeout_seconds'] and config.get('pts_traversal','uct')==params.get('pts_traversal','uct')
                assert (config['beam']==row['beam']) if row['method']=='PACS' else (config['expansion_budget']==row['expansion_budget'])
                front=json.loads(row['frontier_json']);verify(front)
                check_metrics(row,front,eligible,corner,denominator,keys,state);check_timing(row)
                assert row['success']==int(bool(front))
            count+=counter()
    counts=[dict(setting=s,n=n,method=m,beam=b,expansion_budget=e,n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=int(g.metric_eligible.sum()),n_returned=int(g.success.sum()),n_failure=int((g.solver_status=='solver_failure').sum()),n_timeout=int(g.timed_out.sum()),n_exhausted=int(g.exhausted.fillna(0).sum())) for (s,n,m,b,e),g in frame.fillna({'beam':-1,'expansion_budget':-1}).groupby(['setting','n','method','beam','expansion_budget'],sort=False)]
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    return result(directory,files,kind='length_budget',rows=len(frame),input_query_records=records,assignments_checked=count,max_hv_absolute_error=state['error'])

def validate_scale_xl(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'scale_xl.csv');params=params_of(directory)
    assert params['num_queries']==expected
    lst=lambda s:[x.strip() for x in s.split(',') if x.strip()]
    scales=lst(params['scales']);ilp=set(lst(params['ilp_settings']));pbf=set(lst(params['pbf_settings']));unc=set(lst(params['uncapped_settings']));kap=set(lst(params['kappa_settings']))
    kappas=[int(k) for k in params['kappa_sweep'].split(',') if k.strip()!='']
    pts='PTS' if params.get('pts_traversal','uct')=='uct' else 'PTS-DFS'
    def methods_for(setting):
        m=['PACS unbounded','PACS',pts]
        if setting in unc:m.append(f'{pts} uncapped')
        if setting in kap:m+=[f'{pts} k={k}' for k in kappas if not (k==params['candidate_limit'] or (k==0 and setting in unc))]
        if setting in ilp:m.append(f"eps-ILP g={params['grid']}")
        if setting in pbf:m.append('PBF')
        if params.get('methods'):  # a job measuring a declared subset of arms, possibly the conventional Pareto DP
            subset=[x.strip() for x in params['methods'].split(',') if x.strip()];assert set(subset)<=set(m)|{'Pareto-DP'},(subset,m);return subset
        return m
    present=[s for s in scales if s in set(frame.setting)]
    missing=[s for s in scales if s not in set(frame.setting)]
    assert set(map(tuple,frame[['setting','method']].drop_duplicates().to_numpy()))=={(s,m) for s in present for m in methods_for(s)}
    assert not frame.duplicated(['setting','method','query_id']).any() and frame.groupby(['setting','method']).size().eq(expected).all()
    state=dict(error=0.);count=0;records=0;files=[directory/'scale_xl.csv'];exact_unbounded=0;exact_pbf=0
    for setting in present:
        path=directory/'inputs'/setting/'frontiers.jsonl.gz';files.append(path);saved=load_archive(path,expected)
        subset=frame[frame.setting==setting];assert set(subset.query_id.astype(str))==set(saved);records+=len(saved)
        for q,raw in saved.items():
            rows=subset[subset.query_id.astype(str)==q];assert len(rows)==len(methods_for(setting))
            eligible,corner,denominator,keys,verify,counter=check_record(raw,rows,params['beta'],params['alpha'])
            assert rows.ref_timeout_s.eq(params['ref_timeout']).all() and rows.ref_s.eq(rows.reference_s).all()
            assert (rows.ref_done.astype(bool)==bool(raw['reference_complete'])).all()
            for row in rows.to_dict('records'):
                config=json.loads(row['method_config']);m=row['method']
                if m=='PACS unbounded':assert config['beam']==0
                elif m=='Pareto-DP':assert config['use_reduction'] is False and config.get('label_rule','raw_cost')=='raw_cost'
                elif m=='PACS':assert config['beam']==params['beam']
                elif m==pts:assert config==dict(alpha=params['alpha'],beta=params['beta'],timeout_seconds=params['timeout_seconds'],expansion_budget=params['expansion_budget'],candidate_limit=params['candidate_limit'],pts_traversal=params.get('pts_traversal','uct'))
                elif m==f'{pts} uncapped':assert config['candidate_limit']==0 and config['expansion_budget']==params['expansion_budget']
                elif m.startswith(f'{pts} k='):assert config['candidate_limit']==int(m.split('=')[1])
                elif m.startswith('eps'):assert config['grid']==params['grid'] and config['per_solve_budget_s']==params['timeout_seconds']
                else:assert config['pbf_timeout']==params['pbf_timeout'] and config['pbf_max_combos']==params.get('pbf_max_combos',200_000_000) and config.get('pbf_max_memory_bytes')==(int(params['pbf_memory_gib']*2**30) if params.get('pbf_memory_gib') else None)
                assert config['timeout_seconds']==params['timeout_seconds'] and config.get('pts_traversal','uct')==params.get('pts_traversal','uct')
                front=json.loads(row['frontier_json']);verify(front)
                check_metrics(row,front,eligible,corner,denominator,keys,state);check_timing(row);check_epsilon(row)
                assert row['success']==int(bool(front))
                # The measured unbounded run and a completed brute force are exact, so they must recover the reference.
                if eligible and m in ('PACS unbounded','Pareto-DP') and row['solver_status']=='returned' and not row['timed_out'] and row.get('completed') is True:
                    assert row['recall']==1. and {geo.key(p) for p in front}==keys;exact_unbounded+=1
                if eligible and m=='PBF' and row.get('completed') is True:
                    assert row['recall']==1. and {geo.key(p) for p in front}==keys and row['pbf_done']==1;exact_pbf+=1
                if m=='PBF' and row.get('termination_reason') in ('combination_limit','memory_limit'):
                    assert row['solver_status']=='work_limit' and not row['timed_out'] and row['pbf_done']==0
            count+=counter()
    counts=[dict(setting=s,method=m,n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=int(g.metric_eligible.sum()),n_returned=int(g.success.sum()),n_failure=int((g.solver_status=='solver_failure').sum()),n_timeout=int(g.timed_out.sum()),n_work_limit=int(g.termination_reason.isin(['combination_limit','memory_limit']).sum()),n_memory_limit=int((g.termination_reason=='memory_limit').sum()),n_timed=int(g.method_wall_s.notna().sum()),n_reference_complete=int((g.reference_status=='complete').sum()),n_reference_timeout=int((g.reference_status=='reference_timeout').sum())) for (s,m),g in frame.groupby(['setting','method'],sort=False)]
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    return result(directory,files,kind='scale_xl',rows=len(frame),input_query_records=records,assignments_checked=count,max_hv_absolute_error=state['error'],settings=present,missing_settings=missing,exact_unbounded_rows=exact_unbounded,exact_pbf_rows=exact_pbf)

VALIDATORS={'scale_xl':validate_scale_xl,'length_budget':validate_length_budget,'scale':validate_scale}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100);args=parser.parse_args()
    directory=Path(args.directory);kind='scale_xl' if (directory/'scale_xl.csv').exists() else 'length_budget' if (directory/'length_budget.csv').exists() else 'scale'
    VALIDATORS[kind](args.directory,args.expected)
