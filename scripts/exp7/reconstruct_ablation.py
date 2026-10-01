"""Independently reconstruct ablation, budget split, deceptive, and case study
records from their archived candidate arrays, without importing solvers."""
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

PACS_ARMS=['full','-rawcost','-scalar(crowd)','-scalar(epsgrid)','-reduction']
PTS_ARMS=['full','-bound','-prefixdom','-close','-bound-prefixdom','-cap']
PMCTS_ARMS=['full','-rollout','randrollout','-stagnation','-cap','-widening','+warmseed']

def pts_family(params):
    return 'PTS' if params.get('pts_traversal','uct')=='uct' else 'PTS-DFS'

def ablation_config(family,arm,params):
    beam,m,kappa=params['beam'],params['expansion_budget'],params['candidate_limit']
    if family=='PACS':
        return {'full':dict(beam_size=beam,prefix_rule='raw_cost',truncation='scalar',use_reduction=True),
                '-rawcost':dict(beam_size=beam,prefix_rule='overage',truncation='scalar',use_reduction=True),
                '-scalar(crowd)':dict(beam_size=beam,prefix_rule='raw_cost',truncation='crowding',use_reduction=True),
                '-scalar(epsgrid)':dict(beam_size=beam,prefix_rule='raw_cost',truncation='epsgrid',use_reduction=True),
                '-reduction':dict(beam_size=beam,prefix_rule='raw_cost',truncation='scalar',use_reduction=False)}[arm]
    if family.startswith('PTS'):
        base=dict(expansion_budget=m,candidate_limit=kappa)
        if family=='PTS-DFS':base['traversal']=params['pts_traversal']
        return {'full':dict(base),'-bound':dict(base,use_bound=False),'-prefixdom':dict(base,prefix_dominance=False),'-close':dict(base,close_explored=False),
                '-bound-prefixdom':dict(base,use_bound=False,prefix_dominance=False),'-cap':dict(base,candidate_limit=0)}[arm]
    base=dict(expansion_budget=m,candidate_limit=kappa,rollout='greedy',rho=0.1)
    return {'full':base,'-rollout':dict(base,rollout='none'),'randrollout':dict(base,rollout='random'),'-stagnation':dict(base,rho=0.0),
            '-cap':dict(base,candidate_limit=0),'-widening':dict(base,progressive_widening=False),'+warmseed':dict(base,warm_seed=10)}[arm]

def load_archive(path,expected=None):
    with gzip.open(path,'rt') as f: saved={str(r['query_id']):r for r in map(json.loads,f)}
    if expected is not None: assert len(saved)==expected,(path,len(saved),expected)
    return saved

def check_record(raw,rows,beta,alpha_expected):
    """Feasibility from candidate arrays, valid nondominated frontiers, and status agreement. Returns (eligible, corner, denominator, keys, count)."""
    slots=raw['candidates'];budget=raw['budget'];alpha=raw['alpha'];count=0
    # budgets pass through CSV (16 significant digits), so compare with a tight relative tolerance
    assert np.allclose(rows.budget,budget,rtol=1e-12,atol=0) and alpha==alpha_expected
    if raw['query_status']!='missing_embedding':
        if slots and all(slots):
            minimum=sum(min(item[1] for item in slot) for slot in slots)
            expected='feasible' if minimum<=budget*(1+alpha)+1e-9 else 'budget_infeasible'
            assert raw['query_status']==expected,(raw['query_id'],raw['query_status'],expected)
            if 'minimum_assignment_cost' in rows: assert all(math.isclose(v,minimum,abs_tol=1e-9) for v in rows.minimum_assignment_cost)
        else: assert raw['query_status']=='empty_requirement'
    if 'relaxed_cap' in rows: assert np.allclose(rows.relaxed_cap,budget*(1+alpha),rtol=0,atol=1e-9)
    def verify(front):
        nonlocal count
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
            assert all(math.isfinite(v) for v in totals) and cost<=budget*(1+alpha)+1e-9
            assert (quality,cost,rating,max(cost-budget,0.))==(p['quality'],p['cost'],p['rating'],p['cost_overage'])
            count+=1
    ref=raw['reference'];verify(ref)
    eligible=raw['query_status']=='feasible' and raw['reference_complete'] and raw['reference_status']=='complete' and bool(ref)
    corner=denominator=keys=None
    if eligible:
        corner=(min(p['quality'] for p in ref)-.001,-max(p['cost_overage'] for p in ref)-.001,min(p['rating'] for p in ref)-.001)
        denominator=geo.volume(ref,corner);keys={geo.key(p) for p in ref}
    for row in rows.to_dict('records'):
        assert row['query_status']==raw['query_status'] and row['reference_status']==raw['reference_status'],(raw['query_id'],row.get('method',row.get('arm')))
        assert bool(row['metric_eligible'])==eligible
        assert row['ref_size']==len(ref)
        if row['query_status']!='feasible':
            assert row['solver_status']=='not_run' and json.loads(row['frontier_json'])==[]
    return eligible,corner,denominator,keys,verify,lambda:count

def check_metrics(row,front,eligible,corner,denominator,keys,state):
    if eligible:
        recall=len({geo.key(p) for p in front}&keys)/len(keys);hv=geo.volume(front,corner)/denominator
        assert math.isclose(recall,row['recall'],abs_tol=1e-12),(row.get('method',row.get('arm')),recall,row['recall'])
        assert math.isclose(hv,row['hv_ratio'],rel_tol=1e-9,abs_tol=1e-10)
        state['error']=max(state['error'],abs(hv-row['hv_ratio']))
    else: assert math.isnan(row['recall']) and math.isnan(row['hv_ratio'])

def check_timing(row):
    if row.get('method_wall_s') is not None and math.isfinite(row['method_wall_s']):
        if row.get('workload')!='synthetic': assert bool(row['retrieval_measured'])
        assert math.isclose(row['time_s'],row['method_wall_s'],abs_tol=1e-12)
        if row.get('total_s') is not None and math.isfinite(row['total_s']):
            assert math.isclose(row['total_s'],row['retrieval_s']+row['method_wall_s'],abs_tol=1e-9)
    else: assert row['solver_status'] in ('not_run','reference')

def result(directory,files,**fields):
    out=dict(**fields,solvers_invoked=False,files={str(Path(p).relative_to(directory)):sha(p) for p in files},geometry_sha256=sha(HERE/'reconstruct_default.py'),validator_sha256=sha(__file__))
    (Path(directory)/'reconstruction.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='files'}),flush=True);return out

def params_of(directory):
    directory=Path(directory);return json.loads((directory.parent/(directory.name+'.job')/'execution_params.json').read_text())

def validate_ablation(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'ablation.csv');params=params_of(directory)
    assert params['num_queries']==expected
    betas=[float(b) for b in params['betas'].split(',') if b.strip()]
    pts=pts_family(params)
    if 'pts_traversal' in params:   # archived campaign job of the two-policy runner
        arms=[('PACS',a) for a in PACS_ARMS]+[(pts,a) for a in PTS_ARMS if not (pts=='PTS-DFS' and a=='-close')]+[('PMCTS',a) for a in PMCTS_ARMS]
    else:                           # the current runner: PBS full and -rawcost, PTS full, -bound, -prefixdom, -bound-prefixdom
        arms=[('PACS',a) for a in PACS_ARMS[:2]]+[('PTS',a) for a in ['full','-bound','-prefixdom','-bound-prefixdom']]
    assert set(map(tuple,frame[['beta','family','arm']].drop_duplicates().to_numpy()))=={(b,f,a) for b in betas for f,a in arms}
    assert not frame.duplicated(['beta','family','arm','query_id']).any() and frame.groupby(['beta','family','arm']).size().eq(expected).all()
    state=dict(error=0.);count=0;records=0;files=[directory/'ablation.csv']
    for beta in betas:
        path=directory/'inputs'/f'beta-{beta:g}'/'frontiers.jsonl.gz';files.append(path);saved=load_archive(path,expected)
        subset=frame[frame.beta==beta];assert set(subset.query_id.astype(str))==set(saved);records+=len(saved)
        for q,raw in saved.items():
            rows=subset[subset.query_id.astype(str)==q];assert len(rows)==len(arms)
            eligible,corner,denominator,keys,verify,counter=check_record(raw,rows,beta,params['alpha'])
            for row in rows.to_dict('records'):
                got,want=json.loads(row['method_config']),ablation_config(row['family'],row['arm'],params)
                # the current runner keeps only scalar truncation and records no truncation field
                if row['family']=='PACS' and 'truncation' not in got and want.get('truncation')=='scalar':want.pop('truncation')
                assert got==want,(got,want)
                front=json.loads(row['frontier_json']);verify(front)
                check_metrics(row,front,eligible,corner,denominator,keys,state);check_timing(row)
                assert row['success']==int(bool(front))
                if row['solver_status']=='solver_failure' and eligible: assert row['recall']==0
            count+=counter()
    counts=[]
    for (beta,family,arm),s in frame.groupby(['beta','family','arm'],sort=False):
        counts.append(dict(beta=beta,family=family,arm=arm,n_requested=len(s),n_feasible=int(s.query_feasible.sum()),n_metric=int(s.metric_eligible.sum()),n_returned=int(s.success.sum()),n_failure=int((s.solver_status=='solver_failure').sum()),n_timeout=int(s.timed_out.sum()),n_timed=int(s.method_wall_s.notna().sum()),n_invalid=int(s.invalid_assignments.sum())))
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    return result(directory,files,kind='ablation',rows=len(frame),input_query_records=records,assignments_checked=count,max_hv_absolute_error=state['error'],arms=len(arms),betas=betas)

def wvalue(slots,choice,budget,alpha):
    n=len(slots)
    p=sum(slots[i][choice[i]][0]/n+slots[i][choice[i]][2]/(5.0*n) for i in range(n))
    c=sum(slots[i][choice[i]][1] for i in range(n))
    return p-(max(0.,c-budget)/(alpha*budget) if alpha>0 and budget>0 else 0.)

def allowances(slots,budget,mode):
    n=len(slots)
    if mode=='equal': return [budget/n]*n
    w=[float(np.median([item[1] for item in s])) for s in slots] if mode=='median' else [min(item[1] for item in s) for s in slots]
    total=sum(w);return [budget/n]*n if total<=0 else [budget*x/total for x in w]

def validate_budget_split(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'budget_split.csv');params=params_of(directory)
    assert params['num_queries']==expected
    methods={'Exact (shared)','PACS','PTS','ILP','ACS','Split-equal','Split-proportional','Split-median'}
    assert set(frame.method)==methods and not frame.duplicated(['method','query_id']).any() and frame.groupby('method').size().eq(expected).all()
    path=directory/'inputs'/'frontiers.jsonl.gz';saved=load_archive(path,expected);assert set(frame.query_id.astype(str))==set(saved)
    state=dict(error=0.);count=0;scalars=0;splits=0
    for q,raw in saved.items():
        rows=frame[frame.query_id.astype(str)==q];slots=raw['candidates'];budget=raw['budget'];alpha=raw['alpha']
        eligible,corner,denominator,keys,verify,counter=check_record(raw,rows,params['beta'],params['alpha'])
        assert (rows.method_config.map(json.loads).map(lambda c:(c['beam'],c['expansion_budget'],c['candidate_limit']))==(params['beam'],params['expansion_budget'],params['candidate_limit'])).all()
        assert (rows.method_config.map(json.loads).map(lambda c:c.get('pts_traversal','uct'))==params.get('pts_traversal','uct')).all()  # the PTS policy actually measured
        by=rows.set_index('method')
        ilp=by.loc['ILP'];optimum=None
        if ilp.solver_status=='returned' and ilp.proved_optimal==1:
            choice=json.loads(ilp.choice_json);assert len(choice)==len(slots) and all(0<=j<len(slots[d]) for d,j in enumerate(choice))
            assert sum(slots[d][j][1] for d,j in enumerate(choice))<=budget*(1+alpha)+1e-9
            optimum=wvalue(slots,choice,budget,alpha);assert math.isclose(optimum,ilp.wvalue,rel_tol=1e-12) and ilp.wratio==1.
        for row in rows.to_dict('records'):
            front=json.loads(row['frontier_json']);verify(front);check_timing(row)
            if row['method']=='Exact (shared)':
                assert front==raw['reference'] and (row['solver_status']=='reference')==bool(raw['reference'])
                continue
            if row['is_scalar']:
                scalars+=1;assert front==[] and math.isnan(row['recall'])
                if row['choice_json'] is not None and isinstance(row['choice_json'],str):
                    choice=json.loads(row['choice_json']);value=wvalue(slots,choice,budget,alpha)
                    assert math.isclose(value,row['wvalue'],rel_tol=1e-12)
                    assert sum(slots[d][j][1] for d,j in enumerate(choice))<=budget*(1+alpha)+1e-9 and all(slots[d][j][0]>=params['beta']-1e-9 for d,j in enumerate(choice))
                    if row['method']=='ACS': assert (math.isclose(row['wratio'],value/optimum,rel_tol=1e-12) if optimum else math.isnan(row['wratio']))
                else: assert math.isnan(row['wratio']) or row['method']=='ILP'
                continue
            check_metrics(row,front,eligible,corner,denominator,keys,state)
            assert row['success']==int(bool(front))
            if row['method'].startswith('Split') and raw['query_status']=='feasible':
                splits+=1;mode=row['method'].split('-')[1];allow=allowances(slots,budget,mode)
                assert np.allclose(json.loads(row['allowance_json']),allow,rtol=1e-12,atol=1e-9)
                kept=[[j for j,item in enumerate(s) if item[1]<=a*(1+alpha)+1e-9] for s,a in zip(slots,allow)]
                assert json.loads(row['restricted_counts_json'])==[len(k) for k in kept]
                assert row['split_starved']==int(any(len(k)==0 for k in kept))
                if row['split_starved']: assert front==[] and row['solver_status']=='no_assignment'
                for p in front: assert all(j in kept[d] for d,j in enumerate(p['choice']))
                if optimum and front: assert math.isclose(row['wratio'],max(wvalue(slots,p['choice'],budget,alpha) for p in front)/optimum,rel_tol=1e-12)
        count+=counter()
    counts=[]
    for method,s in frame.groupby('method',sort=False):
        counts.append(dict(method=method,n_requested=len(s),n_feasible=int(s.query_feasible.sum()),n_metric=int(s.metric_eligible.sum()),n_returned=int(s.success.sum()),n_starved=int(s.split_starved.fillna(0).sum()),n_failure=int((s.solver_status=='solver_failure').sum()),n_timeout=int(s.timed_out.fillna(0).sum()),n_timed=int(s.method_wall_s.notna().sum()),n_proved_optimal=int(s.get('proved_optimal',pd.Series(dtype=float)).fillna(0).sum()) if 'proved_optimal' in s else 0))
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    return result(directory,[directory/'budget_split.csv',path],kind='budget_split',rows=len(frame),input_query_records=len(saved),assignments_checked=count,scalar_rows_checked=scalars,split_rows_checked=splits,max_hv_absolute_error=state['error'])

def validate_deceptive(directory,expected=100):
    directory=Path(directory);frame=pd.read_csv(directory/'deceptive.csv');params=params_of(directory)
    assert params['num_queries']==expected
    widths=[int(x) for x in params['widths'].split(',') if x.strip()];ks=[int(x) for x in params['ks'].split(',') if x.strip()];budgets=[int(x) for x in params['eval_budgets'].split(',') if x.strip()]
    arms={('PACS',rule,float(b)) for b in widths for rule in ('raw_cost','overage')}|{('PTS','raw_cost',float(m)) for m in budgets}
    state=dict(error=0.);count=0;files=[directory/'deceptive.csv']
    synthetic=directory/'inputs'/'synthetic'/'frontiers.jsonl.gz';catalog=directory/'inputs'/'catalog'/'frontiers.jsonl.gz';files+=[synthetic,catalog]
    syn=load_archive(synthetic,len(ks));cat=load_archive(catalog,expected)
    assert set(syn)=={f'syn_k{k}' for k in ks}
    for workload,saved in [('synthetic',syn),('catalog',cat)]:
        subset=frame[frame.workload==workload];assert set(subset.query_id.astype(str))==set(saved)
        assert set(map(tuple,subset[['method','rule','param']].drop_duplicates().to_numpy()))==arms
        assert not subset.duplicated(['method','rule','param','query_id']).any() and subset.groupby(['method','rule','param']).size().eq(len(saved)).all()
        for q,raw in saved.items():
            rows=subset[subset.query_id.astype(str)==q];slots=raw['candidates']
            beta=0. if workload=='synthetic' else params['beta']
            eligible,corner,denominator,keys,verify,counter=check_record(raw,rows,beta,params['alpha'])
            if workload=='synthetic':
                k=int(q[5:]);assert len(slots)==k+1 and [len(s) for s in slots]==[2]+[3]*k and raw['budget']==100.
                assert rows.k.eq(k).all()
            elif raw['query_status']=='feasible':
                minimum=sum(min(item[1] for item in slot) for slot in slots)
                assert math.isclose(raw['budget'],minimum*1.05,rel_tol=1e-12)
                order=json.loads(rows.requirement_order_json.iloc[0]);assert sorted(order)==list(range(len(slots)))
                best=[max(s,key=lambda item:item[0])[1] for s in slots];assert best==sorted(best,reverse=True)
            for row in rows.to_dict('records'):
                config=json.loads(row['method_config']);assert config['prefix_rule']==row['rule'] and config[('beam' if row['method']=='PACS' else 'expansion_budget')]==row['param']
                assert row['method']=='PACS' or config.get('traversal','uct')==params.get('pts_traversal','uct')  # the PTS policy actually measured
                front=json.loads(row['frontier_json']);verify(front)
                check_metrics(row,front,eligible,corner,denominator,keys,state)
                assert row['success']==int(bool(front)) and row['recovered']==int(len(front)>1)
                if row.get('method_wall_s') is not None and math.isfinite(row['method_wall_s']):
                    # deceptive rows keep the solver's internal search time; the measured wrapper wall time is separate
                    assert row['time_s']<=row['method_wall_s']+1e-6
                else: assert row['solver_status']=='not_run'
            count+=counter()
    counts=[]
    for (workload,method,rule,param),s in frame.groupby(['workload','method','rule','param'],sort=False):
        counts.append(dict(workload=workload,method=method,rule=rule,param=param,n_requested=len(s),n_feasible=int(s.query_feasible.sum()),n_metric=int(s.metric_eligible.sum()),n_returned=int(s.success.sum()),n_recovered=int(s.recovered.sum()),n_failure=int((s.solver_status=='solver_failure').sum()),n_timeout=int(s.timed_out.fillna(0).sum())))
    pd.DataFrame(counts).to_csv(directory/'denominators.csv',index=False)
    return result(directory,files,kind='deceptive',rows=len(frame),input_query_records=len(syn)+len(cat),assignments_checked=count,max_hv_absolute_error=state['error'])

def validate_case(directory,expected=100):
    directory=Path(directory);params=params_of(directory);provenance=json.loads((directory/'case_study_provenance.json').read_text())
    assert provenance['requested']==expected==params['num_queries'] and provenance['timeout_seconds']==params['timeout_seconds']
    assert provenance.get('pts_traversal','uct')==params.get('pts_traversal','uct')  # the PTS policy actually measured
    files=[directory/'case_study_provenance.json'];selected=provenance['selected']
    if selected is None:
        return result(directory,files,kind='case_study',selected=None,considered=len(provenance['considered']))
    path=directory/'inputs'/'frontiers.jsonl.gz';files+=[path,directory/'case_study.csv',directory/'case_study.tex']
    saved=load_archive(path,1);raw=saved[str(selected['query_id'])];slots=raw['candidates'];alpha=raw['alpha'];budget=raw['budget']
    assert raw['query_status']=='feasible' and raw['reference_complete'] and raw['reference_status']=='complete'
    assert params['min_front']<=len(raw['reference'])<=params['max_front'] and len(slots)<=4 and selected['reference_size']==len(raw['reference'])
    considered=[c for c in provenance['considered'] if c['reference_complete'] and params['min_front']<=c['reference_size']<=params['max_front'] and c['n_slots']<=4]
    assert any(c['query_id']==selected['query_id'] for c in considered)
    frame=pd.DataFrame(); n=len(slots); count=0
    def verify(front):
        nonlocal count
        geo.assert_nondominated(front)
        for p in front:
            totals=[0.,0.,0.]
            for d,j in enumerate(p['choice']):
                assert 0<=j<len(slots[d]);totals=[a+b for a,b in zip(totals,slots[d][j])]
            assert totals[1]<=budget*(1+alpha)+1e-9 and (totals[0],totals[1],totals[2],max(totals[1]-budget,0.))==(p['quality'],p['cost'],p['rating'],p['cost_overage']);count+=1
    verify(raw['reference'])
    for front in raw['methods'].values(): verify(front)
    keys={geo.key(p) for p in raw['reference']}
    table=pd.read_csv(directory/'case_study.csv');picked=selected['picked'];assert len(table)==len(picked)==params['n_rows'] or len(table)==len(picked)<=params['n_rows']
    traced={m:{tuple(t['key']) for t in trace} for m,trace in selected['traces'].items()}
    for row,p in zip(table.to_dict('records'),picked):
        assert tuple(p['key']) in keys
        totals=[0.,0.,0.]
        for d,j in enumerate(p['choice']): totals=[a+b for a,b in zip(totals,slots[d][j])]
        assert math.isclose(row['Q'],totals[0]/n,rel_tol=1e-9) and math.isclose(row['C'],totals[1],rel_tol=1e-9) and math.isclose(row['R'],totals[2]/n,rel_tol=1e-9) and math.isclose(row['dC'],max(totals[1]-budget,0.),abs_tol=1e-9)
        found={'PBS' if m=='PACS' else m for m,ks in traced.items() if tuple(p['key']) in ks}
        assert set(str(row['found_by']).split(', '))==(found or {'--'}),(row['found_by'],found)
    tex=(directory/'case_study.tex').read_text();assert tex.count('\\\\')==len(table)+1
    return result(directory,files,kind='case_study',selected=selected['query_id'],considered=len(provenance['considered']),qualified=len(considered),rows=len(table),assignments_checked=count)

VALIDATORS={'ablation':validate_ablation,'budget_split':validate_budget_split,'deceptive':validate_deceptive,'case':validate_case}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100);args=parser.parse_args()
    d=Path(args.directory);kind=next(k for k,f in [('ablation','ablation.csv'),('budget_split','budget_split.csv'),('deceptive','deceptive.csv'),('case','case_study_provenance.json')] if (d/f).exists())
    VALIDATORS[kind](args.directory,args.expected)
