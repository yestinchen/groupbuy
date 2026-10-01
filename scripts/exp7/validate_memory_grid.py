"""Validate isolated g=4/16 memory measurements against the campaign memory jobs: same queries, budgets, candidate counts and input provenance."""
from pathlib import Path
import hashlib
import json
import sys

import os
REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(os.environ.get('CAMPAIGN_ROOT', REPO / 'results' / 'campaign'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def validate(directory):
    directory = Path(directory)
    meta = directory.parent / (directory.name + '.job')
    params = json.loads((meta / 'execution_params.json').read_text())
    setting = params['query_setting']; count = params['num_queries']
    path = directory / f'mem_{setting}.json'; record = json.loads(path.read_text())
    old_meta = CAMPAIGN / 'memory' / (setting + '.job')
    old_inputs = json.loads((old_meta / 'load-001/inputs.json').read_text())[:count]
    inputs = json.loads((meta / 'load-001/inputs.json').read_text())
    assert inputs == old_inputs, 'Query IDs, budgets, counts, or minimum costs changed'
    old_manifest = json.loads((old_meta / 'load-001/manifest.json').read_text())
    new_manifest = json.loads((meta / 'load-001/manifest.json').read_text())
    # Includes source query/catalog file hashes and retrieval precision provenance.
    assert new_manifest['inputs'] == old_manifest['inputs'][:count], 'Input provenance changed'
    assert record['n_queries_requested'] == record['n_queries'] == count
    assert record['status_counts'] == {'feasible': count}
    assert record['retrieval_cached'] is False and record['cuda_available']
    assert record['alpha'] == .5 and record['beta'] == .8 and record['timeout_seconds'] == 60
    assert list(record['search_processes']) == ['eps-ILP-g4', 'eps-ILP-g16']
    pids=set(); report=[]
    for grid in [4,16]:
        child = record['search_processes'][f'eps-ILP-g{grid}']
        assert child['config']['eps_grid'] == grid
        assert child['config']['alpha'] == .5 and child['config']['timeout_s'] == 60
        assert child['requested'] == len(child['rows']) == count
        assert [r['query_id'] for r in child['rows']] == [i['query_id'] for i in inputs]
        assert all(r['query_status']=='feasible' for r in child['rows'])
        assert all(r['solver_status'] in ['returned', 'no_assignment', 'solver_timeout'] for r in child['rows'])
        assert all(r['solver_backend']=='scipy.optimize.milp/HiGHS' and r['highs_threads_configured']==1 for r in child['rows'])
        assert child['returned'] == sum(r['success'] for r in child['rows'])
        assert child['timeouts'] == sum(r['solver_status']=='solver_timeout' for r in child['rows'])
        assert child['shared_catalog_loaded'] is False and child['sampling_interval_s'] == .001
        assert child['rss_samples'] >= 2 and child['pid'] not in pids
        pids.add(child['pid'])
        increase=child['sampled_search_peak_rss_mb']-child['candidate_resident_baseline_mb']
        assert increase >= 0 and abs(increase-child['sampled_search_increase_mb']) < 1e-8
        report.append({'grid':grid,'memory_increase_mb':increase,'returned':child['returned'],'timeouts':child['timeouts']})
    result={'setting':setting,'queries':count,'methods':report,'input_match':'exact match to original query IDs, budgets, candidate counts, minimum costs, and file provenance','record_sha256':sha(path),'inputs_sha256':sha(meta/'load-001/instances.pkl'),'validator_sha256':sha(Path(__file__))}
    (directory/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result
if __name__=='__main__': validate(sys.argv[1])
