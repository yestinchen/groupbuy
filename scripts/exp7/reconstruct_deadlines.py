"""Independently reconstruct deadline metrics from archived candidates,
references, and observed snapshots, without importing solvers or the runner."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('geometry',HERE/'reconstruct_default.py')
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
KS=(1,5,10,20,50)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_candidates(directory,expected):
    with gzip.open(directory/'inputs'/'candidates.jsonl.gz','rt') as f: saved={str(r['query_id']):r for r in map(json.loads,f)}
    assert len(saved)==expected;return saved

def verify_points(front,slots,budget,alpha,beta,state):
    assert len({geo.key(p) for p in front})==len(front);geo.assert_nondominated(front)
    for p in front:
        assert len(p['choice'])==len(slots);totals=[0.,0.,0.]
        for d,j in enumerate(p['choice']):
            assert isinstance(j,int) and 0<=j<len(slots[d]);values=slots[d][j];assert values[0]>=beta-1e-9
            totals=[a+b for a,b in zip(totals,values)]
        q,c,r=totals;assert c<=budget*(1+alpha)+1e-9
        assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip((q,c,r,max(c-budget,0.)),(p['quality'],p['cost'],p['rating'],p['cost_overage'])))
        state['assignments']+=1

def metrics(snapshots,ref,eligible,deadlines,observed_until_s,offset_s):
    keys={geo.key(p) for p in ref}
    corner=(min(p['quality'] for p in ref)-.001,-max(p['cost_overage'] for p in ref)-.001,min(p['rating'] for p in ref)-.001) if ref else None
    hv=geo.volume(ref,corner) if corner else 0.
    shifted=[(t+offset_s,ps) for t,ps in snapshots];row={}
    for k in KS:
        reached=next((t for t,ps in shifted if len({geo.key(p) for p in ps}&keys)>=k),None)
        row[f't_to_{k}']=reached if eligible and reached is not None else float('nan')
        row[f'reached_{k}']=int(reached is not None) if eligible else float('nan')
        row[f'censored_{k}']=int(reached is None) if eligible else float('nan')
    for deadline in deadlines:
        available=next((ps for t,ps in reversed(shifted) if t<=deadline),[])
        row[f'recall@{deadline}']=len({geo.key(p) for p in available}&keys)/len(keys) if eligible and keys else float('nan')
        row[f'hv@{deadline}']=geo.volume(available,corner)/hv if eligible and hv>0 else float('nan')
    row['censor_time_s']=observed_until_s+offset_s
    return row

def validate(directory,expected=100):
    directory=Path(directory).resolve();frame=pd.read_csv(directory/'anytime.csv')
    params=json.loads((directory.parent/(directory.name+'.job')/'execution_params.json').read_text());assert params['num_queries']==expected
    deadlines=[float(x) for x in params['deadlines'].split(',')];mode=params['mode'];clock=params['clock'];grid=params['grid']
    methods=['PACS','PTS',f'eps-ILP g={grid}']+(['eps-ILP g=16'] if grid!=16 else [])+['PBS-unbounded','Pareto-DP','DFS']
    if params.get('methods'):  # a job that measures a subset (one extra epsilon grid beside the campaign)
        subset=[m.strip() for m in params['methods'].split(',') if m.strip()];assert set(subset)<=set(methods)|{'Pareto-DP-conv'},(subset,methods);methods=subset
    saved=load_candidates(directory,expected);assert set(frame.query_id.astype(str))==set(saved)
    fresh=[m for m in methods if m not in ('Pareto-DP','DFS')]+['Pareto-DP-conv']   # the current runner's method list
    assert set(frame.method) in (set(methods),set(fresh)) and set(frame['mode'])=={mode} and set(frame.clock)=={clock}
    if mode=='hard':
        assert set(frame.elapsed_budget_s)==set(deadlines) and frame.groupby(['method','elapsed_budget_s']).size().eq(expected).all() and not frame.duplicated(['query_id','method','elapsed_budget_s']).any()
    else:
        assert set(frame.elapsed_budget_s)=={params['timeout_seconds']} and frame.groupby('method').size().eq(expected).all() and not frame.duplicated(['query_id','method']).any()
    state=dict(assignments=0,fields=0,snapshots=0,error=0.);references=0;files=[directory/'anytime.csv',directory/'inputs'/'candidates.jsonl.gz']
    for q,raw in saved.items():
        rows=frame[frame.query_id.astype(str)==q];slots=raw['candidates'];budget=raw['budget'];alpha=raw['alpha'];assert alpha==params['alpha']
        if slots and all(slots):
            minimum=sum(min(item[1] for item in slot) for slot in slots);assert raw['query_status']==('feasible' if minimum<=budget*(1+alpha)+1e-9 else 'budget_infeasible')
        else: assert raw['query_status']=='empty_requirement'
        assert set(rows.query_status)=={raw['query_status']} and rows.query_feasible.eq(int(raw['query_status']=='feasible')).all()
        query_dir=Path(rows.artifact_dir.iloc[0]).parent;reference=json.loads((query_dir/'reference.json').read_text());files.append(query_dir/'reference.json')
        ref=reference['points'];verify_points(ref,slots,budget,alpha,params['beta'],state)
        eligible=reference['status']=='complete' and bool(ref) and raw['query_status']=='feasible'
        assert set(rows.reference_status)=={reference['status']} and set(rows.metric_eligible)=={int(eligible)}
        if raw['query_status']!='feasible': assert reference['status']=='not_applicable' and ref==[]
        else: references+=1
        for row in rows.to_dict('records'):
            run_dir=Path(row['artifact_dir']);applicable=[row['elapsed_budget_s']] if mode=='hard' else deadlines
            offset=row['retrieval_s'] if clock=='fresh-stages' else 0.
            if clock=='fresh-stages': assert bool(row['retrieval_measured']) and math.isfinite(offset)
            if row['solver_status'] in ('not_run','retrieval_exceeds_budget'):
                assert not run_dir.exists()
                for d in applicable: assert math.isnan(row[f'recall@{d}']) and math.isnan(row[f'hv@{d}'])
                if row['solver_status']=='not_run': assert raw['query_status']!='feasible'
                else: assert mode=='hard' and row['elapsed_budget_s']-offset<=0
                continue
            observation=json.loads((run_dir/'observation.json').read_text());files.append(run_dir/'observation.json')
            assert observation['status']==row['solver_status'] and observation['mode']==mode
            snapshots=[]
            for event in observation['snapshots']:
                snap=json.loads((run_dir/event['file']).read_text());files.append(run_dir/event['file']);state['snapshots']+=1
                verify_points(snap['points'],slots,budget,alpha,params['beta'],state)
                assert event['available_s']<=observation['observed_until_s']+1e-9
                snapshots.append((event['available_s'],snap['points']))
            assert [t for t,_ in snapshots]==sorted(t for t,_ in snapshots)
            if mode=='hard': assert math.isclose(observation['budget_s'],row['elapsed_budget_s']-offset,abs_tol=1e-9)
            expected_metrics=metrics(snapshots,ref,eligible,applicable,observation['observed_until_s'],offset)
            for key,value in expected_metrics.items():
                actual=row[key]
                assert (pd.isna(value) and pd.isna(actual)) or abs(value-actual)<1e-9,(q,row['method'],key,value,actual)
                if key.startswith('hv@') and not pd.isna(value): state['error']=max(state['error'],abs(value-actual))
                state['fields']+=1
            assert row['n_snapshots']==len(snapshots) and row['ref_size']==len(ref)
            if row['solver_status']=='completed' and eligible and row['method']=='PBS-unbounded' and observation.get('completion',{}).get('timing',{}).get('exhausted'):
                assert {geo.key(p) for p in snapshots[-1][1]}=={geo.key(p) for p in ref}
    counts=[]
    for (m,l),g in frame.groupby(['method','elapsed_budget_s'],sort=False):
        c=dict(method=m,elapsed_budget_s=l,n_requested=len(g),n_feasible=int(g.query_feasible.sum()),n_metric=int(g.metric_eligible.sum()),n_completed=int((g.solver_status=='completed').sum()),n_timeout=int((g.solver_status=='timeout').sum()),n_terminated=int((g.solver_status=='terminated').sum()),n_failure=int((g.solver_status=='failed').sum()),n_not_run=int(g.solver_status.isin(['not_run','retrieval_exceeds_budget']).sum()))
        for d in ([l] if mode=='hard' else deadlines):
            e=g[g.metric_eligible==1];c[f'recall@{d}_mean']=e[f'recall@{d}'].mean();c[f'hv@{d}_mean']=e[f'hv@{d}'].mean()
        counts.append(c)
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    result=dict(kind='deadlines',mode=mode,clock=clock,rows=len(frame),input_query_records=len(saved),complete_references=references,snapshots_checked=state['snapshots'],assignments_checked=state['assignments'],metric_fields_checked=state['fields'],max_hv_absolute_error=state['error'],solvers_invoked=False,files={str(p.relative_to(directory)):sha(p) for p in files},geometry_sha256=sha(HERE/'reconstruct_default.py'),validator_sha256=sha(__file__))
    (directory/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100);args=parser.parse_args()
    validate(args.directory,args.expected)
