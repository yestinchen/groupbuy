"""Check the structure, denominators, and sampling records of one isolated
memory measurement, without importing solvers."""
import argparse
import hashlib
import json
from pathlib import Path

METHODS=['PBS','PTS','eps-ILP','PBS-unbounded','Pareto-DP','DFS','Pareto-DP-conv']  # a job may measure a declared subset
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(directory,expected=100):
    directory=Path(directory);params=json.loads((directory.parent/(directory.name+'.job')/'execution_params.json').read_text())
    setting=params['query_setting'];path=directory/f'mem_{setting}.json';rec=json.loads(path.read_text())
    assert rec['setting']==setting and rec['n_queries_requested']==expected==params['num_queries']
    assert rec['n_queries']==sum(rec['status_counts'].values())==expected
    assert rec['retrieval_cached'] is False and rec['retrieval_s_total'] is not None and rec['retrieval_historical_s_total'] is None
    assert rec['cuda_available'] and str(rec['gpu_device']).startswith('cuda') and rec['gpu_peak_retrieval_mb'] is not None
    assert rec['alpha']==params['alpha'] and rec['beta']==params['beta'] and rec['expansion_budget']==params['expansion_budget'] and rec['candidate_limit']==params['candidate_limit'] and rec['timeout_seconds']==params['timeout_seconds']
    assert rec['rss_baseline_mb']<=rec['rss_after_imports_mb']<=rec['rss_after_load_mb']<=rec['rss_after_retrieval_mb']
    assert rec['embedding_matrix_mb'] is not None and rec['n_embeddings']==rec['n_items']
    children=rec['search_processes'];assert list(children)==params['methods'].split(',') and set(children)<=set(METHODS)
    feasible=rec['status_counts'].get('feasible',0);pids=set()
    for method,child in children.items():
        assert child['method']==method and child['requested']==expected and len(child['rows'])==expected and child['shared_catalog_loaded'] is False
        assert child['sampling_interval_s']==.001 and child['rss_samples']>=2 and child['sampled_search_peak_rss_mb']>=child['candidate_resident_baseline_mb']
        assert child['process_peak_rss_mb']>=child['peak_before_search_mb'] and child['pid'] not in pids;pids.add(child['pid'])
        assert child['config']['alpha']==params['alpha'] and child['config']['timeout_s']==params['timeout_seconds'] and child['config']['PTS_expansions']==params['expansion_budget'] and child['config']['PTS_candidate_limit']==params['candidate_limit']
        statuses=[r['query_status'] for r in child['rows']]
        assert sum(s=='feasible' for s in statuses)==feasible and sum(r['solver_status']=='not_run' for r in child['rows'])==expected-feasible
        assert child['returned']==sum(r['success'] for r in child['rows']) and child['timeouts']==sum(int(r.get('timed_out') or 0) or int(r['solver_status']=='solver_timeout') for r in child['rows'])  # the worker records a timeout as solver_status, not as a flag
        assert child['returned']+child['timeouts']>=0 and child['returned']<=feasible
    assert rec['n_search_solved']==sum(c['returned'] for c in children.values()) and rec['n_search_timeouts']==sum(c['timeouts'] for c in children.values())
    manifest=json.loads((directory/'manifest.json').read_text());assert manifest['requested']==expected and manifest['config']['query_setting']==setting
    result=dict(kind='memory',setting=setting,n_items=rec['n_items'],n_queries=rec['n_queries'],status_counts=rec['status_counts'],methods=list(children),isolated_processes=len(pids),solvers_invoked=False,files={p.name:sha(p) for p in [path,directory/'manifest.json']},validator_sha256=sha(__file__))
    (directory/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--expected',type=int,default=100);args=parser.parse_args()
    validate(args.directory,args.expected)
