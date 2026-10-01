"""Validated default comparisons and independently reconstructible records."""
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

from gquery.alg.exp4.core import ReferenceResult, instance_status
from gquery.alg.exp4.sensitivity_reporting import validated_frontier
from gquery.alg.pareto.brute_force import solve_brute_force


def validated_reference(reference, inst, alpha):
    frontier, invalid = validated_frontier(reference[0], inst, alpha)
    complete = reference[1] and not invalid and bool(frontier) and reference.status == 'complete'
    status = 'reference_failure' if invalid or (reference.status == 'complete' and not complete) else reference.status
    return ReferenceResult(frontier, complete, reference[2], status)


def pbf_call(inst, alpha, seconds, max_combos, max_memory_bytes=None):
    """max_combos <= 0 disables the combination guard; max_memory_bytes bounds the RSS growth of the call."""
    stats = {}
    frontier, done, _, checked = solve_brute_force(
        inst, alpha, seconds, max_combos if max_combos and max_combos > 0 else None,
        max_memory_bytes=max_memory_bytes, stats=stats)
    reason = stats['termination_reason']
    guarded = reason == 'combination_limit'
    return SimpleNamespace(frontier=frontier, completed=done, exhausted=done,
                           termination_reason=reason, completion_scope='full_input',
                           n_evals=0 if guarded else checked * inst.n_slots,
                           n_checked=0 if guarded else checked,
                           baseline_rss_bytes=stats['baseline_rss_bytes'],
                           peak_rss_delta_bytes=stats['peak_rss_delta_bytes'])


def pbf_memory_fields(inst, result, max_combos, max_memory_bytes):
    """Per-row PBF memory record, merged into the method timing fields."""
    if result is None or 'PBF' not in getattr(inst, 'method_timings', {}):
        return
    inst.method_timings['PBF'].update(
        pbf_max_combos=max_combos if max_combos and max_combos > 0 else None,
        pbf_max_memory_bytes=max_memory_bytes, pbf_combinations_checked=result.n_checked,
        pbf_baseline_rss_bytes=result.baseline_rss_bytes,
        pbf_peak_rss_delta_bytes=result.peak_rss_delta_bytes)


def finish_method(inst, alpha, name, frontier, *, failed=False, timed_out=False, stats=None, grid=None):
    frontier, invalid = validated_frontier(frontier, inst, alpha)
    timing = getattr(inst, 'method_timings', {}).get(name, {})
    fields = {'invalid_assignments': invalid}
    if stats is not None:
        r_min = sum(min(it.rating for it, _ in s) for s in inst.slots)
        r_max = sum(max(it.rating for it, _ in s) for s in inst.slots)
        expected = 1 + (1 if r_max <= r_min + 1e-9 else grid) * (grid if alpha * inst.query.budget > 0 else 1)
        fields.update({k: stats.get(k, 0) for k in ['n_solves', 'n_nonoptimal', 'n_infeasible']})
        fields['n_expected_solves'] = expected
        timed_out = timed_out or bool(stats.get('hit_time_budget'))
        failed = failed or bool(stats.get('n_nonoptimal')) or (not timed_out and stats['n_solves'] != expected)
        timing.update(completed=not timed_out and not failed, exhausted=None,
                      termination_reason='timeout' if timed_out else 'grid_failure' if failed else 'grid_complete',
                      completion_scope='configured_grid')
    timed_out = timed_out or timing.get('termination_reason') in {'timeout', 'cancelled'}
    status = ('not_run' if instance_status(inst, alpha) != 'feasible' else
              'solver_failure' if failed or invalid else 'solver_timeout' if timed_out else
              'work_limit' if timing.get('termination_reason') in {'combination_limit', 'memory_limit'} else
              'returned' if frontier else 'no_assignment')
    timing['solver_status'] = status
    fields.update(solver_status=status, timed_out=int(timed_out))
    return frontier, fields


def save_query(directory, inst, alpha, reference, frontiers):
    """Candidate objective arrays permit offline feasibility and metric checks."""
    def records(frontier):
        return [dict(choice=list(p.choice), quality=p.quality, cost=p.cost,
                     cost_overage=p.cost_overage, rating=p.rating) for p in frontier]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(query_id=inst.query.id, budget=inst.query.budget, alpha=alpha,
                  query_status=instance_status(inst, alpha),
                  candidates=[[[float(q), float(it.price), float(it.rating)] for it,q in slot] for slot in inst.slots],
                  reference_status=reference.status, reference_complete=bool(reference[1]),
                  reference=records(reference[0]), methods={k:records(v) for k,v in frontiers.items()},
                  metric_coordinates=['sum_matching_quality', 'negative_cost_overhead', 'total_rating'],
                  recall_decimals=9, reference_margin=0.001)
    with gzip.open(directory / 'frontiers.jsonl.gz', 'at') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')


def save_candidates(directory, records, alpha):
    """Candidate objective arrays alone, for runners that store references and
    outputs elsewhere (deadline snapshots), so every archived assignment can be
    checked offline against the actual input."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / 'candidates.jsonl.gz', 'wt') as f:
        for inst in records:
            f.write(json.dumps(dict(
                query_id=inst.query.id, budget=inst.query.budget, alpha=alpha,
                query_status=instance_status(inst, alpha), n_slots=inst.n_slots,
                candidates=[[[float(q), float(it.price), float(it.rating)] for it, q in slot] for slot in inst.slots]),
                allow_nan=False) + '\n')

