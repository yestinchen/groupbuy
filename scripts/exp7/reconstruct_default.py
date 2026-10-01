"""Reconstruct default assignment validity and quality without importing solvers."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def key(p):
    return tuple(round(p[k], 9) for k in ['quality', 'cost_overage', 'rating'])


def assert_nondominated(frontier):
    vectors=sorted((key(p) for p in frontier),key=lambda v:(v[1],-v[0],-v[2]))
    ranks={r:i+1 for i,r in enumerate(sorted({v[2] for v in vectors},reverse=True))}
    tree=[-math.inf]*(len(ranks)+1)
    for q,overhead,rating in vectors:
        index=ranks[rating];best=-math.inf
        while index:
            best=max(best,tree[index]);index-=index & -index
        assert best<q, 'Saved frontier contains a dominated nine-decimal vector'
        index=ranks[rating]
        while index<len(tree):
            tree[index]=max(tree[index],q);index+=index & -index


def volume(records, corner):
    # Integrate disjoint quality slabs; each cross-section is a rectangle union.
    points = [(p['quality'], -p['cost_overage'], p['rating']) for p in records]
    points = [p for p in points if all(p[j] > corner[j] for j in range(3))]
    levels = sorted({corner[0], *(p[0] for p in points)})
    total = 0.
    for left, right in zip(levels, levels[1:]):
        yz = sorted(((p[1],p[2]) for p in points if p[0] >= right), reverse=True)
        area = 0.; highest = corner[2]
        for y,z in yz:
            if z > highest:
                area += (y-corner[1])*(z-highest)
                highest = z
        total += (right-left)*area
    return total


def validate(directory, expected=None):
    directory=Path(directory)
    csv=directory/('default.csv' if (directory/'default.csv').exists() else 'baselines.csv')
    df=pd.read_csv(csv)
    assert not df.duplicated(['query_id','method']).any()
    with gzip.open(directory/'frontiers.jsonl.gz','rt') as f:
        records=[json.loads(line) for line in f]
    assert len(records)==df.query_id.nunique()
    if expected is not None: assert len(records)==expected
    verified=0; max_error=0.
    for raw in records:
        selected=df[df.query_id.astype(str)==str(raw['query_id'])]
        assert set(selected.method)==set(raw['methods'])
        slots=raw['candidates']; budget=raw['budget']; alpha=raw['alpha']
        for front in [raw['reference'],*raw['methods'].values()]:
            assert len({key(p) for p in front})==len(front)
            assert_nondominated(front)
            for p in front:
                assert len(p['choice'])==len(slots)
                q=c=r=0.
                for d,j in enumerate(p['choice']):
                    assert isinstance(j,int) and 0<=j<len(slots[d])
                    qq,cc,rr=slots[d][j]
                    assert qq>=.8-1e-9
                    q+=qq;c+=cc;r+=rr
                assert all(math.isfinite(v) for v in [q,c,r])
                assert c<=budget*(1+alpha)+1e-9
                assert (q,c,r,max(c-budget,0.))==(p['quality'],p['cost'],p['rating'],p['cost_overage'])
                verified+=1
        ref=raw['reference']
        eligible=raw['query_status']=='feasible' and raw['reference_complete'] and raw['reference_status']=='complete' and bool(ref)
        if eligible:
            corner=(min(p['quality'] for p in ref)-.001,-max(p['cost_overage'] for p in ref)-.001,min(p['rating'] for p in ref)-.001)
            denominator=volume(ref,corner); keys={key(p) for p in ref}
        for row in selected.to_dict('records'):
            assert bool(row['metric_eligible'])==eligible
            if eligible:
                front=raw['methods'][row['method']]
                recall=len({key(p) for p in front}&keys)/len(keys)
                hv=volume(front,corner)/denominator
                assert math.isclose(recall,row['recall'],abs_tol=1e-12)
                assert math.isclose(hv,row['hv_ratio'],rel_tol=1e-9,abs_tol=1e-10),(raw['query_id'],row['method'],hv,row['hv_ratio'])
                max_error=max(max_error,abs(hv-row['hv_ratio']))
            else:
                assert math.isnan(row['recall']) and math.isnan(row['hv_ratio'])
            if math.isfinite(row['method_wall_s']):
                assert math.isclose(row['time_s'],row['method_wall_s'],abs_tol=1e-12)
                assert bool(row['retrieval_measured'])
                assert math.isclose(row['total_s'],row['retrieval_s']+row['method_wall_s'],abs_tol=1e-9)
            if row['method'].startswith('eps') and row['completed'] is True:
                assert row['solves']==row['n_expected_solves'] and row['n_nonoptimal']==0 and not row['timed_out']
            if row.get('termination_reason') in ('combination_limit','memory_limit'):
                assert not row['timed_out'] and row['solver_status']=='work_limit' and not row['completed']
            if row.get('termination_reason')=='memory_limit':
                assert row['pbf_peak_rss_delta_bytes']>row['pbf_max_memory_bytes']
    summaries=[]
    for method,s in df.groupby('method',sort=False):
        summaries.append(dict(method=method,n_requested=len(s),n_feasible=int(s.query_feasible.sum()),n_metric=int(s.metric_eligible.sum()),n_returned=int(s.success.sum()),n_completed=int(s.completed.fillna(False).astype(bool).sum()),n_timeout=int(s.timed_out.sum()),n_failure=int((s.solver_status=='solver_failure').sum()),n_work_limit=int(s.termination_reason.isin(['iteration_limit','combination_limit','memory_limit','work_limit']).sum()),n_guard=int((s.termination_reason=='combination_limit').sum()),n_memory_limit=int((s.termination_reason=='memory_limit').sum()),n_timed=int(s.method_wall_s.notna().sum()),n_invalid=int(s.invalid_assignments.sum())))
    result=dict(directory=str(directory.resolve()),queries=len(records),rows=len(df),assignments_checked=verified,max_hv_absolute_error=max_error,solvers_invoked=False,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [csv,directory/'frontiers.jsonl.gz']},counts=summaries)
    (directory/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(summaries).to_csv(directory/'denominators.csv',index=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directories',nargs='+');parser.add_argument('--expected',type=int);args=parser.parse_args()
    for directory in args.directories:
        r=validate(directory,args.expected)
        print(json.dumps({k:r[k] for k in ['directory','queries','rows','assignments_checked','max_hv_absolute_error']}),flush=True)
