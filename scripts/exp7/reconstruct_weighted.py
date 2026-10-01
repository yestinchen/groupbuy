"""Recompute every weighted comparison value from archived candidates and
returned bundles, without importing solvers."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

WEIGHTS=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
METHODS=['ILP','PBS','PTS','PBS-noLP','PTS-noLP']
OPT_TOL=1e-7
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def wname(w):
    s=sum(w) or 1;return '/'.join(f'{x/s:.2f}' for x in w)

def value(slots,choice,budget,alpha,w):
    n=len(slots);assert len(choice)==n
    q=sum(slots[d][j][0] for d,j in enumerate(choice));c=sum(slots[d][j][1] for d,j in enumerate(choice));r=sum(slots[d][j][2] for d,j in enumerate(choice))
    pen=w[1]*max(0.,c-budget)/(alpha*budget) if alpha>0 and budget>0 else 0.
    return w[0]*q/n-pen+w[2]*r/(5.*n),c

def validate(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'all_results.csv')
    params=json.loads((directory.parent/(directory.name+'.job')/'execution_params.json').read_text());assert params['num_queries']==expected
    with gzip.open(directory/'inputs'/'candidates.jsonl.gz','rt') as f: archive={str(r['query_id']):r for r in map(json.loads,f)}
    assert len(archive)==expected and set(frame.query_id.astype(str))==set(archive)
    names={wname(w):w for w in WEIGHTS}
    assert set(frame.weights)==set(names) and set(frame.method)==set(METHODS)
    assert not frame.duplicated(['query_id','weights','method']).any() and frame.groupby(['weights','method']).size().eq(expected).all()
    checked=0;optimal_checked=0;beta=params['beta']
    # the PTS policy actually measured: rows carry it only when the runner exposes the option
    pts_rows=frame[frame.method.isin(['PTS','PTS-noLP'])]
    if 'traversal' in frame.columns: assert (pts_rows.traversal==params.get('pts_traversal','uct')).all()
    else: assert params.get('pts_traversal','uct')=='uct'
    for q,raw in archive.items():
        rows=frame[frame.query_id.astype(str)==q];slots=raw['candidates'];budget=raw['budget'];alpha=raw['alpha'];assert alpha==params['alpha']
        if slots and all(slots):
            minimum=sum(min(item[1] for item in slot) for slot in slots)
            assert raw['query_status']==('feasible' if minimum<=budget*(1+alpha)+1e-9 else 'budget_infeasible')
        else: assert raw['query_status']=='empty_requirement'
        assert set(rows.query_status)=={raw['query_status']}
        for w,g in rows.groupby('weights'):
            by=g.set_index('method');ilp=by.loc['ILP'];weights=names[w]
            proved=ilp['ilp_status']=='proved_optimal'
            assert set(g.metric_eligible)=={int(proved)} and set(g.reference_status)=={ilp['reference_status']}
            assert (ilp['reference_status']=='complete')==proved
            f_opt=None
            for row in g.to_dict('records'):
                if raw['query_status']!='feasible':
                    assert row['solver_status']=='not_run' and not isinstance(row['choice_json'],str) and row['feasible']==0;continue
                if isinstance(row['choice_json'],str):
                    choice=json.loads(row['choice_json']);assert all(isinstance(j,int) and 0<=j<len(slots[d]) for d,j in enumerate(choice))
                    assert all(slots[d][j][0]>=beta-1e-9 for d,j in enumerate(choice))
                    v,cost=value(slots,choice,budget,alpha,weights)
                    assert cost<=budget*(1+alpha)+1e-9 and math.isclose(v,row['value'],rel_tol=1e-12,abs_tol=1e-12),(q,w,row['method'],v,row['value'])
                    assert row['feasible']==1 and row['solver_status'] in ('returned','solver_timeout');checked+=1
                    if row['method']=='ILP': f_opt=v
                else:
                    assert row['feasible']==0 and (math.isnan(row['value']) if isinstance(row['value'],float) else row['value'] is None)
                    assert row['solver_status'] in ('solver_failure','solver_timeout','no_assignment')
            if proved:
                assert f_opt is not None and ilp['optimal']==1 and ilp['rel_gap']==0
                for row in g[g.method!='ILP'].to_dict('records'):
                    if isinstance(row['choice_json'],str):
                        v=row['value'];gap=f_opt-v;assert v<=f_opt+OPT_TOL*max(1.,abs(f_opt))
                        assert row['optimal']==int(gap<=OPT_TOL*max(1.,abs(f_opt))) and math.isclose(row['rel_gap'],gap/max(abs(f_opt),1e-9),rel_tol=1e-9,abs_tol=1e-12);optimal_checked+=1
                    else: assert row['optimal']==0 and math.isnan(row['rel_gap'])
            else:
                assert g.optimal.isna().all() and g.rel_gap.isna().all()
            for row in g.to_dict('records'):
                # The weighted solvers expose no phase split (search_s stays null); method_wall_s is the measured invocation.
                if row.get('method_wall_s') is not None and isinstance(row['method_wall_s'],float) and math.isfinite(row['method_wall_s']):
                    assert raw['query_status']=='feasible' and bool(row['retrieval_measured']) and math.isclose(row['total_s'],row['retrieval_s']+row['method_wall_s'],abs_tol=1e-9)
                    assert math.isclose(row['end_to_end_s'],row['total_s'],abs_tol=1e-12)
                else: assert row['solver_status']=='not_run'
    overall=pd.read_csv(directory/'summary_overall.csv',index_col=0)
    for method,s in overall.iterrows():
        g=frame[frame.method==method];comparable=g[g.metric_eligible==1];returned=comparable[comparable.feasible==1]
        assert s['n']==len(g) and s['n_metric']==len(comparable) and s['n_gap']==len(returned) and s['n_feasible']==g.feasible.sum()
        if len(comparable): assert math.isclose(s['optimal_fraction'],comparable.optimal.mean(),abs_tol=1e-12)
    files=[directory/'all_results.csv',directory/'summary_overall.csv',directory/'summary_by_weight.csv',directory/'inputs'/'candidates.jsonl.gz']
    result=dict(kind='weighted',rows=len(frame),input_query_records=len(archive),values_checked=checked,optimality_checked=optimal_checked,solvers_invoked=False,files={str(p.relative_to(directory)):sha(p) for p in files},validator_sha256=sha(__file__))
    (directory/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100);args=parser.parse_args()
    validate(args.directory,args.expected)
