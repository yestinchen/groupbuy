"""Requested-input sensitivity reporting under the shared R4/R5 protocol.

The historical CLI body is retained in bench_sensitivity to preserve local
edits. Its callback dispatches here. Ordinary invocation timing excludes this
module's post-call assignment validation and metric computation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gquery.alg.exp4.core import (
    InstanceBatch, ReferenceResult, _min_feasible_cost, exact_frontier,
    instance_status, load_instances, metric_context, nadir, pkey, score,
    timed_for, to_points,
)
from gquery.alg.exp4.elapsed import canonical_points, points
from gquery.alg.exp4.protocol import report_row, write_manifest
from gquery.alg.candidates import PState
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.brute_force import solve_brute_force
from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp
from gquery.alg.pareto.metrics import hypervolume_3d


def reference_for(inst, alpha, seconds, frontier=None):
    # Callers pass their own module-level reference function so it stays patchable.
    try:
        return (frontier or exact_frontier)(inst, alpha, timeout_s=seconds)
    except MemoryError:
        return ReferenceResult([], False, float('nan'), 'reference_memory_limit')
    except Exception:
        return ReferenceResult([], False, float('nan'), 'reference_failure')


def validated_frontier(frontier, inst, alpha):
    """Keep valid assignments even if another returned assignment is invalid."""
    valid, invalid = [], 0
    for p in frontier:
        try:
            valid.extend(points(canonical_points(
                [PState(list(p.choice), p.quality, p.cost, p.rating)],
                inst.slots, inst.query.budget, alpha)))
        except (ValueError, TypeError, IndexError, AttributeError):
            invalid += 1
    return points(canonical_points(
        [PState(list(p.choice), p.quality, p.cost, p.rating) for p in valid],
        inst.slots, inst.query.budget, alpha)) if valid else [], invalid


def pbf_result(inst, alpha, seconds):
    # The legacy brute-force guard is a work limit, not a time limit.
    count = math.prod(map(len, inst.slots))
    frontier, done, elapsed, checked = solve_brute_force(inst, alpha, timeout_s=seconds)
    reason = 'complete' if done else 'combination_limit' if count > 200_000_000 else 'timeout'
    return SimpleNamespace(frontier=frontier, completed=done, exhausted=done,
                           termination_reason=reason, completion_scope='full_input',
                           n_evals=checked * inst.n_slots if reason != 'combination_limit' else 0)


def evaluate(inst, alpha, beta, sweep, param, method, config, reference):
    context = metric_context(inst, alpha, reference[1], reference[0], reference.status)
    # Require all three conditions, even for an inconsistent external reference.
    context['metric_eligible'] = int(context['query_feasible'] and reference[1]
                                     and bool(reference[0]) and reference.status == 'complete')
    if reference.status == 'complete' and not context['metric_eligible'] and context['query_feasible']:
        context['reference_status'] = 'reference_failure'
    if method == 'PACS':
        fn = lambda f: solve_pbs(inst.slots, inst.query.budget, alpha,
                beam_size=config['beam'], prefix_rule='raw_cost', use_reduction=True, timeout_flag=f)
    elif method.startswith('PTS'):
        fn = lambda f: solve_pts(inst.slots, inst.query.budget, alpha,
                expansion_budget=config['expansion_budget'], candidate_limit=config['candidate_limit'],
                use_reduction=True, timeout_flag=f)
    elif method.startswith('eps'):
        fn = lambda f: solve_epsilon_ilp(inst.query, inst.slots, alpha,
                config['beta_grid'], f, config['beta_ilp_budget'])
    else:
        fn = lambda f: pbf_result(inst, alpha, config['beta_bf_timeout'])
    out, elapsed, timed_out = timed_for(inst, alpha, fn, config['timeout_seconds'], name=method)
    frontier, evals, cell_failure = [], np.nan, False
    cell_counts = {}
    work_counts = {}
    if out is not None:
        if isinstance(out, tuple):
            frontier, stats = out
            evals = stats['n_solves'] * inst.n_candidates
            timed_out = timed_out or bool(stats.get('hit_time_budget', False))
            timing = inst.method_timings[method]
            cell_counts = {key: stats.get(key, 0) for key in ['n_solves','n_infeasible','n_nonoptimal']}
            r_min = sum(min(it.rating for it, _ in slot) for slot in inst.slots)
            r_max = sum(max(it.rating for it, _ in slot) for slot in inst.slots)
            expected = 1 + (1 if r_max <= r_min + 1e-9 else config['beta_grid']) * (config['beta_grid'] if alpha > 0 else 1)
            cell_counts['n_expected_solves'] = expected
            cell_failure = bool(stats.get('n_nonoptimal', 0)) or (not timed_out and stats['n_solves'] < expected)
            timing.update(completed=not timed_out and not cell_failure, exhausted=None,
                          termination_reason='timeout' if timed_out else 'grid_failure' if cell_failure else 'grid_complete',
                          completion_scope='configured_grid')
        else:
            frontier = to_points(out.frontier, inst.query.budget) if method != 'PBF' else out.frontier
            evals = out.n_evals
            work_counts = {key: getattr(out, key, np.nan) for key in [
                'n_before', 'n_after', 'iterations', 'n_action_evals', 'n_rollout_evals',
                'n_bound_evals', 'n_frontier_comparisons', 'n_frontier_gains',
                'n_created_nodes', 'n_retired_nodes', 'n_drained_nodes']}
            timed_out = timed_out or out.termination_reason in {'timeout', 'cancelled'}
    frontier, invalid = validated_frontier(frontier, inst, alpha)
    timings = getattr(inst, 'method_timings', {})
    if method in timings:
        timings[method]['solver_status'] = ('solver_failure' if out is None or invalid or cell_failure else
            'solver_timeout' if timed_out else 'work_limit' if timings[method].get('termination_reason') == 'combination_limit' else 'returned' if frontier else 'no_assignment')
    ref_points = reference[0] if context['metric_eligible'] else []
    ref_pt = nadir(ref_points)
    row = report_row(inst, alpha, {
        **context, **cell_counts, **work_counts, 'sweep': sweep, 'method': method, 'param': param,
        'alpha': alpha, 'beta': beta, 'query_id': inst.query.id,
        # Avoid report_row's legacy beam-key alias. Config has its own field.
        'method_config': json.dumps(config, sort_keys=True),
        'success': int(bool(frontier)), 'invalid_assignments': invalid,
        **score(frontier, {pkey(p) for p in ref_points}, ref_pt,
                hypervolume_3d(ref_points, ref_pt) if ref_pt else 0.,
                metric_eligible=context['metric_eligible']),
        'time_s': elapsed, 'timed_out': int(timed_out), 'evals': evals,
        'ref_size': len(reference[0]), 'reference_s': reference[2],
        'n_candidates': inst.n_candidates, 'n_slots': inst.n_slots,
        'minimum_assignment_cost': _min_feasible_cost(inst.slots),
        'budget': inst.query.budget, 'relaxed_cap': inst.query.budget * (1+alpha),
        'empty_requirement_cause': 'not_determined' if context['query_status'] == 'empty_requirement' else None,
        'validation_boundary': 'post-call validation and scoring excluded from method wall time',
        'frontier_json': json.dumps([dict(choice=p.choice, quality=p.quality, cost=p.cost,
                                        rating=p.rating, cost_overage=p.cost_overage) for p in frontier]),
    })
    if not context['query_feasible']:
        row['solver_status'] = 'not_run'
    return row


def summarize(rows, requested, available):
    df = pd.DataFrame(rows)
    status = df.query_status
    eligible = df.metric_eligible.astype(bool)
    returned = df.success.astype(bool)
    counts = dict(n_requested=requested, n_available=available,
                  n_unavailable=max(0, requested-available), n_rows=len(df),
                  n_loaded=int(status.isin(['feasible', 'budget_infeasible']).sum()),
                  n_metric=int(eligible.sum()),
                  n_successfully_evaluated=int((eligible & returned).sum()),
                  n_returned=int(returned.sum()),
                  n_method_failure=int(df.solver_status.isin(['solver_failure','no_assignment']).sum()),
                  n_method_timeout=int(df.timed_out.sum()),
                  n_method_not_run=int((df.solver_status == 'not_run').sum()),
                  n_method_complete=int(df.completed.fillna(False).astype(bool).sum()),
                  n_method_work_limit=int(df.termination_reason.isin(['iteration_limit','combination_limit']).sum()))
    for value in ['missing_embedding', 'empty_requirement', 'budget_infeasible', 'feasible']:
        counts['n_' + value] = int((status == value).sum())
    for value in ['complete', 'reference_timeout', 'reference_failure', 'reference_memory_limit', 'not_applicable']:
        counts['n_reference_complete' if value == 'complete' else 'n_' + value] = int((df.reference_status == value).sum())
    counts.update(recall=df.loc[eligible, 'recall'].mean(), hv=df.loc[eligible, 'hv_ratio'].mean(),
                  size=df.loc[eligible, 'size'].mean(),
                  success=df.loc[status == 'feasible', 'success'].mean(),
                  med_time=df.time_s.median(), evals=df.evals.median())
    return counts


def run_sensitivity(**options):
    # Local import avoids the shared validation helper's module dependency.
    from gquery.alg.exp4.default_reporting import validated_reference, save_query
    config = dict(options)
    output = Path('data_out') / config['exp_name']
    output.mkdir(parents=True, exist_ok=True)
    lists = {key: [cast(x) for x in config[key].split(',') if x.strip()]
             for key, cast in [('beams', int), ('budgets', int), ('kappas', int),
                               ('betas', float), ('alphas', float)]}
    all_rows, summaries, inputs = [], [], []
    settings = []
    pts = 'PTS'
    for key, sweep, field in [('beams','beam','beam'),('budgets','budget','expansion_budget'),
                               ('kappas','kappa','candidate_limit')]:
        for value in lists[key]:
            settings.append((sweep, value, config['alpha'], config['beta'],
                             ['PACS'] if sweep == 'beam' else [pts], {field: value}))
    for value in lists['betas']:
        methods = ['PACS', pts]
        if not config['skip_beta_baselines']:
            methods += [f"eps-ILP g={config['beta_grid']}", 'PBF']
        settings.append(('beta', value, config['alpha'], value, methods, {}))
    for value in lists['alphas']:
        settings.append(('alpha', value, value, config['beta'], ['PACS', pts], {}))
    # Adjacent beam/iteration/cap settings share input loading and references.
    previous_key, batch, references = None, None, None
    for index, (sweep, param, alpha, beta, methods, overrides) in enumerate(settings):
        key = (alpha, beta)
        if key != previous_key:
            batch, references = None, None
            batch, _ = load_instances(config['query_setting'], config['num_queries'],
                                       beta, alpha, include_invalid=True)
            records = getattr(batch, 'requested_records', batch)
            batch = InstanceBatch(records, alpha, include_invalid=True)
            previous_key = key
            write_manifest(output / 'manifests' / f'input-{index}',
                           {**config, 'actual_alpha': alpha, 'actual_beta': beta}, batch)
            references = [validated_reference(reference_for(inst, alpha, config['timeout_seconds']), inst, alpha)
                          for inst in batch]
            input_index = index
            for inst, reference in zip(batch.requested_records, references):
                save_query(output / 'inputs' / str(input_index), inst, alpha, reference, {})
        actual = {**config, **overrides}
        counts = dict(batch.status_counts)
        inputs.append(dict(sweep=sweep, param=param, alpha=alpha, beta=beta,
                           n_requested=config['num_queries'], n_available=len(batch.requested_records),
                           n_loaded=sum(counts.get(x,0) for x in ['feasible','budget_infeasible']),
                           n_unavailable=max(0,config['num_queries']-len(batch.requested_records)),
                           status_counts=json.dumps(counts, sort_keys=True)))
        rows = [evaluate(inst, alpha, beta, sweep, param, method, actual, ref)
                for inst, ref in zip(batch.requested_records, references) for method in methods]
        for row in rows:
            row['input_index'] = input_index
        all_rows.extend(rows)
        for method in methods:
            selected = [r for r in rows if r['method'] == method]
            if selected:
                summary = summarize(selected, config['num_queries'], len(batch.requested_records))
            else:
                # No available source records is distinct from loaded invalid records.
                summary = dict(n_requested=config['num_queries'], n_available=0, n_loaded=0,
                               n_unavailable=config['num_queries'], n_rows=0, n_metric=0)
            summaries.append(dict(sweep=sweep, param=param, alpha=alpha, beta=beta, method=method, **summary))
        # Checkpoints preserve completed settings if a later run is interrupted.
        pd.DataFrame(all_rows).to_csv(output / 'sensitivity.csv', index=False)
        pd.DataFrame(summaries).to_csv(output / 'sensitivity_summary.csv', index=False)
        pd.DataFrame(inputs).to_csv(output / 'sensitivity_inputs.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    return all_rows
