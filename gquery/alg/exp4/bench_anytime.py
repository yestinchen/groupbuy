"""Recall and hypervolume at common receiver-observed elapsed budgets.

Observation runs retain deadline snapshots from one invocation. Hard runs make
an independent invocation per deadline. Historical retrieval is never a clock
offset. The durable observation/reference files permit offline reconstruction.
"""
from __future__ import annotations
import json
from pathlib import Path
import click
import pandas as pd

from gquery.alg.exp4.core import load_instances, instance_status, metric_context
from gquery.alg.exp4.protocol import write_manifest
from gquery.alg.exp4.elapsed import (
    KS, SCHEMA, atomic_json, points, reconstruct, run_isolated, summarize,
)

DEADLINES = [1.1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def evaluate_instance(inst, config, directory, deadlines, methods, *, mode='observation',
                      clock='method', timeout_seconds=60., sample_interval_s=.001,
                      memory_limit_mb=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    status = instance_status(inst, config['alpha'])
    fresh = getattr(inst, 'retrieval_measured', False)
    if clock == 'fresh-stages' and not fresh and status == 'feasible':
        raise ValueError('fresh-stages clock requires retrieval measured in this run')
    if mode == 'hard' and clock != 'method':
        raise ValueError('hard interruption currently applies to the method clock only')
    offset = inst.retrieval_s if clock == 'fresh-stages' and fresh else 0.
    if mode == 'observation' and max(deadlines) > timeout_seconds + offset:
        raise ValueError('observation deadlines must fit the configured observation horizon')
    ref = []
    reference_status = 'not_applicable'
    if status == 'feasible':
        observation = run_isolated(inst, 'PBS-unbounded', config, directory/'reference',
                                   timeout_seconds, sample_interval_s=sample_interval_s,
                                   memory_limit_mb=memory_limit_mb)
        complete = observation['status'] == 'completed' and observation.get('completion',{}).get('timing',{}).get('exhausted')
        snaps = reconstruct(directory/'reference', observation)
        ref = snaps[-1][1] if complete and snaps else []
        reference_status = ('complete' if ref else 'reference_failure') if complete else (
            'reference_timeout' if observation['status']=='timeout' else
            'reference_memory_limit' if observation['status']=='memory_limit' else 'reference_failure')
    context = metric_context(inst, config['alpha'], reference_status=='complete', ref, reference_status)
    atomic_json(directory/'reference.json', dict(status=reference_status,
                points=[dict(choice=list(p.choice), quality=p.quality, cost=p.cost,
                             rating=p.rating, cost_overage=p.cost_overage) for p in ref]))
    rows = []
    for method in methods:
        limits = deadlines if mode == 'hard' else [timeout_seconds]
        for limit in limits:
            budget = limit-offset if mode == 'hard' else limit
            run_dir = directory / (method.replace(' ', '_').replace('=', '-') + f'-{mode}-{limit}')
            if status != 'feasible' or budget <= 0:
                observation = dict(status='not_run' if status!='feasible' else 'retrieval_exceeds_budget',
                                   snapshots=[], observed_until_s=0., mode=mode, budget_s=max(0.,budget))
                snapshots = []
            else:
                observation = run_isolated(inst, method, config, run_dir, budget,
                    mode=mode, sample_interval_s=sample_interval_s, memory_limit_mb=memory_limit_mb)
                snapshots = reconstruct(run_dir, observation)
            applicable = [limit] if mode=='hard' else deadlines
            # Published output persists after failure/termination. Feasible method
            # failures remain in every quality denominator under a complete reference.
            row = summarize(snapshots, ref, bool(context['metric_eligible']), applicable,
                            observed_until_s=observation['observed_until_s'], offset_s=offset)
            row.update(query_id=inst.query.id, method=method, **context,
                       solver_status=observation['status'], mode=mode, clock=clock,
                       elapsed_budget_s=limit, artifact_dir=str(run_dir), timing_schema=SCHEMA,
                       retrieval_s=inst.retrieval_s if fresh else None,
                       retrieval_historical_s=None if fresh else inst.retrieval_s,
                       retrieval_measured=fresh, encoding_status='excluded_precomputed',
                       total_status='sum_of_fresh_measured_stages_excluding_process_startup' if clock=='fresh-stages' else 'method_only',
                       publication_boundary='synchronous validated frontier and atomic file observed by parent',
                       **{k:observation.get(k) for k in ('completion_observed_s','process_end_s','overshoot_s',
                           'terminate_requested_s','rss_sample_count','rss_requested_interval_s','sampled_rss_peak_bytes')})
            rows.append(row)
    return rows


def method_list(grid, subset=None):
    """Grid-derived method list; an explicit subset must be drawn from it (used to measure one extra
    epsilon grid beside an existing campaign without repeating the other methods)."""
    full=['PACS','PTS',f'eps-ILP g={grid}']+(['eps-ILP g=16'] if grid!=16 else [])+['PBS-unbounded','Pareto-DP-conv']
    if not subset:
        return full
    chosen=[m.strip() for m in subset.split(',') if m.strip()]
    unknown=[m for m in chosen if m not in full]
    if unknown:
        raise click.BadParameter(f'unknown methods {unknown}; choose from {full}')
    return chosen


@click.command()
@click.option('--num-queries', type=int, default=100)
@click.option('--query-setting', type=str, default='e10_1k_100q')
@click.option('--alpha', type=float, default=0.5)
@click.option('--beta', type=float, default=0.8)
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option('--candidate-limit', type=int, default=50)
@click.option('--grid', type=int, default=8, help='Primary epsilon grid; g=16 is also included.')
@click.option('--timeout-seconds', type=float, default=60.0)
@click.option('--deadlines', type=str, default=None, help='Comma separated elapsed seconds.')
@click.option('--exp-name', type=str, default='exp4_r7_anytime')
@click.option('--output-dir', type=click.Path(path_type=Path), default=None)
@click.option('--mode', type=click.Choice(['observation','hard']), default='observation')
@click.option('--clock', type=click.Choice(['method','fresh-stages']), default='method')
@click.option('--fresh-retrieval', is_flag=True)
@click.option('--query-indices', default=None, help='Requested zero-based indices for a representative pilot.')
@click.option('--sample-interval-ms', type=float, default=1.)
@click.option('--memory-limit-mb', type=float, default=None, help='Parent observed absolute worker RSS limit.')
@click.option('--methods', type=str, default=None, help='Comma separated subset of the grid-derived method list, e.g. "eps-ILP g=4"; default runs every method.')
def main(num_queries, query_setting, alpha, beta, beam, expansion_budget,
         candidate_limit, grid, timeout_seconds, deadlines, exp_name, output_dir,
         mode, clock, fresh_retrieval, query_indices, sample_interval_ms, memory_limit_mb, methods):
    dls = [float(x) for x in deadlines.split(',')] if deadlines else DEADLINES
    if not dls or any(x<=0 for x in dls) or timeout_seconds<=0 or grid<1:
        raise click.BadParameter('positive deadlines, timeout and grid required')
    if mode=='hard' and clock!='method':
        raise click.BadParameter('hard mode requires --clock method')
    if clock=='fresh-stages' and not fresh_retrieval:
        raise click.BadParameter('--clock fresh-stages requires --fresh-retrieval')
    out = output_dir or Path('data_out')/exp_name
    out.mkdir(parents=True, exist_ok=False)
    insts, _ = load_instances(query_setting, num_queries, beta, alpha,
                              include_invalid=True, use_disk_cache=not fresh_retrieval)
    if query_indices:
        insts=[insts[int(i)] for i in query_indices.split(',')]
    # Candidate arrays permit offline validation of every archived reference and snapshot assignment.
    from gquery.alg.exp4.default_reporting import save_candidates
    save_candidates(out/'inputs', getattr(insts, 'requested_records', insts), alpha)
    methods=method_list(grid,methods)
    config=dict(alpha=alpha,beta=beta,beam=beam,expansion_budget=expansion_budget,
                candidate_limit=candidate_limit,grid=grid,deadlines=dls,methods=methods,
                mode=mode,clock=clock,timeout_seconds=timeout_seconds,
                sample_interval_ms=sample_interval_ms,memory_limit_mb=memory_limit_mb,
                snapshot_instrumentation='synchronous validation/filter/file publication charged',
                metric_policy='nine-decimal rounded dominance and identity; raw-float HV and complete-reference nadir',
                method_order='listed order, each in fresh spawned process, no implicit warmup')
    write_manifest(out/'manifest',config,insts)
    rows=[]
    for index,inst in enumerate(insts):
        rows+=evaluate_instance(inst,config,out/f'query-{index}-{inst.query.id}',dls,methods,
                               mode=mode,clock=clock,timeout_seconds=timeout_seconds,
                               sample_interval_s=sample_interval_ms/1000.,memory_limit_mb=memory_limit_mb)
        pd.DataFrame(rows).to_csv(out/'anytime.csv',index=False)
        click.echo(f'Completed requested query {inst.query.id}, {len(rows)} method records')
    click.echo(str(out/'anytime.csv'))


if __name__=='__main__':
    main()
