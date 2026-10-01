"""Recompute the historical weighted beta diagnostic (weighted.csv) from
archived candidates and returned bundles, without importing solvers."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

WEIGHTS=[(1.,1.,1.),(1.,2.,1.),(2.,1.,1.),(1.,1.,2.),(1.,.5,1.)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def value(slots,choice,budget,alpha,l1,l2,l3):
    n=len(slots);assert len(choice)==n
    p=sum(l1*slots[i][choice[i]][0]/n+l3*slots[i][choice[i]][2]/(5.*n) for i in range(n))
    c=sum(slots[i][choice[i]][1] for i in range(n))
    return p-((l2*max(0.,c-budget)/(alpha*budget)) if alpha>0 and budget>0 else 0.),c

def validate(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'weighted.csv')
    params=json.loads((directory.parent/(directory.name+'.job')/'execution_params.json').read_text());assert params['num_queries']==expected
    lst=lambda k,cast:[cast(x) for x in params[k].split(',') if x.strip()]
    beams,budgets,kappas,betas=lst('beams',int),lst('budgets',int),lst('kappas',int),lst('betas',float)
    sweeps={('default',params['beta'],params['beta']):['ILP','ACS','MCTS']}
    sweeps.update({('beam',float(b),params['beta']):['ILP','ACS'] for b in beams});sweeps.update({('budget',float(m),params['beta']):['ILP','MCTS'] for m in budgets})
    sweeps.update({('kappa',float(k),params['beta']):['ILP','MCTS'] for k in kappas});sweeps.update({('weights',f'{l1}/{l2}/{l3}',params['beta']):['ILP','ACS','MCTS'] for l1,l2,l3 in WEIGHTS})
    sweeps.update({('beta',bt,bt):['ILP','ACS','MCTS'] for bt in betas if abs(bt-params['beta'])>1e-9})
    frame['param_key']=[(s,(p if s=='weights' else float(p)),b) for s,p,b in zip(frame.sweep,frame.param,frame.beta)]
    actual={(s,p,b,m) for (s,p,b),m in zip(frame.param_key,frame.method)}
    assert actual=={(s,p,b,m) for (s,p,b),ms in sweeps.items() for m in ms},(actual^{(s,p,b,m) for (s,p,b),ms in sweeps.items() for m in ms})
    assert frame.groupby(['sweep','param','beta','method']).size().eq(expected).all() and not frame.duplicated(['sweep','param','beta','method','query_id']).any()
    archives={};files=[directory/'weighted.csv']
    for bt in sorted({b for _,_,b in sweeps}):
        path=directory/'inputs'/f'beta-{bt:g}'/'candidates.jsonl.gz';files.append(path)
        with gzip.open(path,'rt') as f: archives[bt]={str(r['query_id']):r for r in map(json.loads,f)}
        assert len(archives[bt])==expected
    checked=0;ratios=0
    for (sweep,param,bt),g in frame.groupby('param_key'):
        for q,rows in g.groupby(g.query_id.astype(str)):
            raw=archives[bt][q];slots=raw['candidates'];budget=raw['budget'];alpha=raw['alpha'];assert alpha==params['alpha']
            assert set(rows.query_status)=={raw['query_status']}
            l1,l2,l3=(map(float,param.split('/')) if sweep=='weights' else (1.,1.,1.))
            by=rows.set_index('method');ilp=by.loc['ILP'];comparable=int(ilp['metric_eligible'])
            assert set(rows.metric_eligible)=={comparable}
            f_opt=None
            for row in rows.to_dict('records'):
                if raw['query_status']!='feasible':
                    # the historical diagnostic reports the query status itself for unattempted rows
                    assert row['solver_status'] in ('not_run',raw['query_status']) and row['success']==0 and not isinstance(row['choice_json'],str);continue
                if isinstance(row['choice_json'],str):
                    choice=json.loads(row['choice_json']);assert all(isinstance(j,int) and 0<=j<len(slots[d]) for d,j in enumerate(choice))
                    v,cost=value(slots,choice,budget,alpha,l1,l2,l3);assert cost<=budget*(1+alpha)+1e-9
                    assert math.isclose(v,row['value'],rel_tol=1e-12,abs_tol=1e-12) and row['success']==1;checked+=1
                    if row['method']=='ILP': f_opt=v
                else: assert row['success']==0 and math.isnan(row['value'])
            if comparable:
                assert f_opt is not None and f_opt>0 and ilp['certified']==1 and math.isclose(ilp['ratio'],1.)
                for row in rows[rows.method!='ILP'].to_dict('records'):
                    if row['success']: assert math.isclose(row['ratio'],row['value']/f_opt,rel_tol=1e-12) and row['value']<=f_opt+1e-7*max(1.,abs(f_opt));ratios+=1
                    else: assert math.isnan(row['ratio'])
            else: assert rows.ratio.isna().all()
    result=dict(kind='weighted_diagnostic',rows=len(frame),input_query_records=sum(map(len,archives.values())),values_checked=checked,ratios_checked=ratios,solvers_invoked=False,files={str(p.relative_to(directory)):sha(p) for p in files},validator_sha256=sha(__file__))
    (directory/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100);args=parser.parse_args()
    validate(args.directory,args.expected)
