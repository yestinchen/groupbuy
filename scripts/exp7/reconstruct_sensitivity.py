"""Independently reconstruct sensitivity assignments and metrics, without solvers."""
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
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def planned_settings(params):
    """Ordered (sweep, param, methods) exactly as run_sensitivity schedules them."""
    planned=[];pts='PTS' if params.get('pts_traversal','uct')=='uct' else 'PTS-DFS'
    for option,sweep,methods in [('beams','beam',['PACS']),('budgets','budget',[pts]),('kappas','kappa',[pts]),('betas','beta',['PACS',pts]+([] if params['skip_beta_baselines'] else [f"eps-ILP g={params['beta_grid']}",'PBF'])),('alphas','alpha',['PACS',pts])]:
        planned.extend((sweep,float(value),list(methods)) for value in params[option].split(',') if value.strip())
    return planned

def planned_groups(params):
    return {(sweep,param,method) for sweep,param,methods in planned_settings(params) for method in methods}

def validate(directory,expected=100,interrupted=False):
    directory=Path(directory)
    frame=pd.read_csv(directory/'sensitivity.csv')
    params=json.loads((directory.parent/(directory.name+'.job')/'execution_params.json').read_text())
    assert params['num_queries']==expected
    planned=planned_settings(params)
    groups=planned_groups(params)
    actual=set(map(tuple,frame[['sweep','param','method']].drop_duplicates().to_numpy()))
    missing=[]
    if interrupted:
        # A whole-setting checkpoint: the completed settings must be a strict prefix of the schedule with every method present.
        assert actual<=groups,(actual-groups)
        done=[(sweep,param) for sweep,param,methods in planned if all((sweep,param,method) in actual for method in methods)]
        partial=[(sweep,param) for sweep,param,methods in planned if any((sweep,param,method) in actual for method in methods) and (sweep,param) not in done]
        assert not partial,partial
        assert done==[(sweep,param) for sweep,param,_ in planned[:len(done)]] and 0<len(done)<len(planned),(done,len(planned))
        missing=[(sweep,param) for sweep,param,_ in planned[len(done):]]
        groups={(sweep,param,method) for sweep,param,methods in planned[:len(done)] for method in methods}
        assert actual==groups
    else:
        assert actual==groups,(actual-groups,groups-actual)
    inputs=pd.read_csv(directory/'sensitivity_inputs.csv')
    assert not inputs.duplicated(['sweep','param']).any()
    assert set(map(tuple,inputs[['sweep','param']].to_numpy()))=={(s,p) for s,p,m in groups}
    assert inputs.n_requested.eq(expected).all() and inputs.n_available.eq(expected).all()
    assert not frame.duplicated(['sweep','param','method','query_id']).any()
    assert frame.groupby(['sweep','param','method']).size().eq(expected).all()
    count=0;error=0.;input_count=0
    files=[directory/'sensitivity.csv',directory/'sensitivity_summary.csv',directory/'sensitivity_inputs.csv']
    for index,subset in frame.groupby('input_index'):
        path=directory/'inputs'/str(index)/'frontiers.jsonl.gz';files.append(path)
        with gzip.open(path,'rt') as f: saved={str(r['query_id']):r for r in map(json.loads,f)}
        assert len(saved)==expected
        assert set(subset.query_id.astype(str))==saved.keys()
        input_count+=len(saved)
        for q,raw in saved.items():
            rows=subset[subset.query_id.astype(str)==q]
            beta=rows.beta.iloc[0]
            slots=raw['candidates'];budget=raw['budget'];alpha=raw['alpha']
            assert rows.budget.eq(budget).all() and rows.alpha.eq(alpha).all()
            if raw['query_status']!='missing_embedding':
                if slots and all(slots):
                    minimum=sum(min(item[1] for item in slot) for slot in slots)
                    expected_status='feasible' if minimum<=budget*(1+alpha)+1e-9 else 'budget_infeasible'
                    assert all(math.isclose(v,minimum,abs_tol=1e-9) for v in rows.minimum_assignment_cost)
                    assert raw['query_status']==expected_status
                else:assert raw['query_status']=='empty_requirement'
            assert np.allclose(rows.relaxed_cap,budget*(1+alpha),rtol=0,atol=1e-9)
            fronts=[raw['reference']]+[json.loads(s) for s in rows.frontier_json]
            for front in fronts:
                assert len({geo.key(p) for p in front})==len(front)
                geo.assert_nondominated(front)
                for p in front:
                    assert len(p['choice'])==len(slots)
                    totals=[0.,0.,0.]
                    for d,j in enumerate(p['choice']):
                        assert isinstance(j,int) and 0<=j<len(slots[d])
                        values=slots[d][j];assert values[0]>=beta-1e-9
                        totals=[a+b for a,b in zip(totals,values)]
                    quality,cost,rating=totals
                    assert all(math.isfinite(v) for v in totals)
                    assert cost<=budget*(1+alpha)+1e-9
                    assert (quality,cost,rating,max(cost-budget,0.))==(p['quality'],p['cost'],p['rating'],p['cost_overage'])
                    count+=1
            ref=raw['reference']
            eligible=raw['query_status']=='feasible' and raw['reference_complete'] and raw['reference_status']=='complete' and bool(ref)
            if eligible:
                corner=(min(p['quality'] for p in ref)-.001,-max(p['cost_overage'] for p in ref)-.001,min(p['rating'] for p in ref)-.001)
                denominator=geo.volume(ref,corner);keys={geo.key(p) for p in ref}
            for row in rows.to_dict('records'):
                assert row['query_status']==raw['query_status'] and row['reference_status']==raw['reference_status']
                assert bool(row['metric_eligible'])==eligible
                actual_config=json.loads(row['method_config'])
                expected_config=dict(params)
                field={'beam':'beam','budget':'expansion_budget','kappa':'candidate_limit'}.get(row['sweep'])
                if field:expected_config[field]=int(row['param'])
                assert actual_config==expected_config
                expected_alpha=row['param'] if row['sweep']=='alpha' else params['alpha']
                expected_beta=row['param'] if row['sweep']=='beta' else params['beta']
                assert row['alpha']==expected_alpha and row['beta']==expected_beta
                front=json.loads(row['frontier_json'])
                if eligible:
                    recall=len({geo.key(p) for p in front}&keys)/len(keys)
                    hv=geo.volume(front,corner)/denominator
                    assert math.isclose(recall,row['recall'],abs_tol=1e-12)
                    assert math.isclose(hv,row['hv_ratio'],rel_tol=1e-9,abs_tol=1e-10),(q,row['method'],hv,row['hv_ratio'])
                    error=max(error,abs(hv-row['hv_ratio']))
                else:assert math.isnan(row['recall']) and math.isnan(row['hv_ratio'])
                assert row['ref_size']==len(ref)
                if math.isfinite(row['method_wall_s']):
                    assert bool(row['retrieval_measured'])
                    assert math.isclose(row['time_s'],row['method_wall_s'],abs_tol=1e-12)
                    assert math.isclose(row['total_s'],row['retrieval_s']+row['method_wall_s'],abs_tol=1e-9)
                else:assert row['solver_status']=='not_run'
                if row['method'].startswith('eps') and row['completed'] is True:
                    assert row['n_solves']==row['n_expected_solves'] and row['n_nonoptimal']==0 and not row['timed_out']
                if row['termination_reason']=='combination_limit':
                    assert not row['timed_out'] and row['solver_status']=='work_limit' and not row['completed']
    summary=pd.read_csv(directory/'sensitivity_summary.csv')
    assert not summary.duplicated(['sweep','param','method']).any()
    assert set(map(tuple,summary[['sweep','param','method']].to_numpy()))==groups
    for s in summary.to_dict('records'):
        rows=frame[(frame.sweep==s['sweep']) & (frame.param==s['param']) & (frame.method==s['method'])]
        eligible=rows[rows.metric_eligible==1]
        assert s['n_requested']==expected and s['n_rows']==len(rows) and s['n_metric']==len(eligible)
        for col in ['feasible','budget_infeasible','empty_requirement','missing_embedding']:
            assert s['n_'+col]==(rows.query_status==col).sum()
        for status in ['complete','reference_timeout','reference_failure','reference_memory_limit','not_applicable']:
            assert s['n_reference_complete' if status=='complete' else 'n_'+status]==(rows.reference_status==status).sum()
        assert s['n_returned']==rows.success.sum()
        assert s['n_method_timeout']==rows.timed_out.sum()
        assert s['n_method_complete']==rows.completed.fillna(False).astype(bool).sum()
        assert s['n_method_not_run']==(rows.solver_status=='not_run').sum()
        assert s['n_method_failure']==rows.solver_status.isin(['solver_failure','no_assignment']).sum()
        assert s['n_method_work_limit']==rows.termination_reason.isin(['iteration_limit','combination_limit']).sum()
        if len(eligible):
            assert math.isclose(s['recall'],eligible.recall.mean(),abs_tol=1e-12)
            assert math.isclose(s['hv'],eligible.hv_ratio.mean(),abs_tol=1e-10)
        else:assert math.isnan(s['recall']) and math.isnan(s['hv'])
    result=dict(rows=len(frame),input_query_records=input_count,assignments_checked=count,max_hv_absolute_error=error,solvers_invoked=False,interrupted=bool(interrupted),planned_settings=len(planned),completed_settings=len(planned)-len(missing),missing_settings=missing,files={str(p.relative_to(directory)):sha(p) for p in files},geometry_sha256=sha(HERE/'reconstruct_default.py'),validator_sha256=sha(__file__))
    (directory/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100)
    parser.add_argument('--interrupted',action='store_true',help='accept a whole-setting checkpoint of a terminated job; completed settings must be a strict schedule prefix')
    args=parser.parse_args()
    validate(args.directory,args.expected,args.interrupted)
