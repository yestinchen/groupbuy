"""Durable, receiver-observed frontier publication and isolated elapsed limits.

Snapshots contain validated original-index assignments. Canonical objectives
round to nine decimals before weak dominance and duplicate merging. Search
comparators retain their solver-specific floating semantics. Publication costs
are charged synchronously; metric computation occurs offline.
"""
from __future__ import annotations
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from gquery.alg.candidates import PState, _non_dominated_final
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.dp import solve_dp
from gquery.alg.exp4.baselines import solve_unbounded_pbs
from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp
from gquery.alg.pareto.metrics import _Point, hypervolume_3d
from gquery.alg.exp4.core import pkey, nadir
from gquery.alg.exp4.protocol import timing_record

SCHEMA = 'r7-publication-v1'
KS = (1, 5, 10, 20, 50)


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(data, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def canonical_points(states, slots, budget, alpha):
    """Recompute objective values from complete original-index assignments."""
    validated = []
    for state in states:
        if len(state.choice) != len(slots) or not slots:
            raise ValueError('publication requires one original index per requirement')
        q = c = r = 0.
        for d, j in enumerate(state.choice):
            if not isinstance(j, int) or not 0 <= j < len(slots[d]):
                raise ValueError('publication candidate index outside original input')
            item, quality = slots[d][j]
            q += quality; c += item.price; r += item.rating
        if not all(math.isfinite(v) for v in (q, c, r)):
            raise ValueError('nonfinite assignment objectives')
        if c > budget * (1 + alpha) + 1e-9:
            raise ValueError('published assignment violates relaxed budget')
        if (q, c, r) != (state.quality, state.cost, state.rating):
            raise ValueError('published objectives differ from requirement-order reconstruction')
        validated.append(PState(list(state.choice), q, c, r))
    frontier = _non_dominated_final(validated, budget)
    return [dict(choice=list(s.choice), quality=s.quality,
                 cost=s.cost, rating=s.rating,
                 cost_overage=max(0., s.cost-budget),
                 raw_quality=s.quality, raw_rating=s.rating) for s in frontier]


def points(records):
    return [_Point(p['quality'], p['rating'], p['cost'], p['cost_overage'], tuple(p['choice']))
            for p in records]


class SnapshotSink:
    """List-compatible trace consumer; every file is a complete frontier."""
    def __init__(self, directory, inst, alpha, origin):
        self.directory = Path(directory)
        self.inst, self.alpha, self.origin = inst, alpha, origin
        self.states = []
        self.sequence = 0
        self.publication_s = 0.

    def append(self, entry):
        self.extend([entry])

    def extend(self, entries):
        entries = list(entries)
        if not entries:
            return
        start = time.perf_counter()
        self.states.extend(PState(list(s.choice), s.quality, s.cost, s.rating) for _, s in entries)
        snapshot = canonical_points(self.states, self.inst.slots, self.inst.query.budget, self.alpha)
        # Retain canonical survivors with raw objectives for subsequent validation.
        self.states = [PState(p['choice'], p['raw_quality'], p['cost'], p['raw_rating']) for p in snapshot]
        atomic_json(self.directory / f'snapshot-{self.sequence:08d}.json',
                    dict(schema=SCHEMA, sequence=self.sequence, points=snapshot,
                         preparation_completed_s=time.perf_counter()-self.origin))
        self.sequence += 1
        self.publication_s += time.perf_counter() - start


def invoke(inst, method, config, sink):
    cs, b, a = inst.slots, inst.query.budget, config['alpha']
    if method == 'PACS':
        return solve_pbs(cs, b, a, beam_size=config['beam'], trace=sink)
    if method == 'PTS':
        return solve_pts(cs, b, a, expansion_budget=config['expansion_budget'],
                         candidate_limit=config['candidate_limit'], trace=sink)
    if method == 'PBS-unbounded':
        return solve_unbounded_pbs(cs, b, a, trace=sink)
    if method == 'Pareto-DP-conv':
        return solve_dp(cs, b, a, trace=sink)
    if method.startswith('eps-ILP g='):
        return solve_epsilon_ilp(inst.query, cs, a, int(method.split('=')[1]),
                                 None, config['worker_budget_s'], trace=sink)
    raise ValueError(f'unknown method {method}')


def _worker(conn, directory, inst, method, config, target):
    directory = Path(directory)
    conn.send('ready')
    origin = conn.recv()
    conn.close()
    sink = SnapshotSink(directory, inst, config['alpha'], origin)
    try:
        result = target(inst, method, config, sink)
        elapsed = time.perf_counter() - origin
        status = getattr(result, 'termination_reason', 'completed')
        if isinstance(result, tuple):
            stats = result[1]
            status = ('solver_time_limit' if stats.get('hit_time_budget') else
                      'solver_failure' if stats.get('n_nonoptimal') else 'completed')
        record = dict(status=status, solver_return_s=elapsed,
                      publication_s=sink.publication_s,
                      timing=timing_record(inst, method, elapsed, result))
    except MemoryError:
        record = dict(status='memory_limit', traceback=traceback.format_exc())
    except Exception:
        record = dict(status='solver_failure', traceback=traceback.format_exc())
    atomic_json(directory / 'completion.json', record)


def run_isolated(inst, method, config, directory, budget_s, *, mode='observation',
                 sample_interval_s=.001, memory_limit_mb=None, target=invoke):
    """Terminate at the parent deadline, retain only already observed files.

    Imports/input loading/process startup precede the ready/go handshake. Parent
    observation and OS scheduling add measurable delay. No exact peak RSS or
    zero-overshoot guarantee is implied. Paths must be new to prevent stale data.
    """
    import psutil
    if budget_s <= 0 or sample_interval_s <= 0:
        raise ValueError('positive elapsed budget and sampling interval required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    cfg = dict(config, worker_budget_s=budget_s)
    proc = ctx.Process(target=_worker, args=(child, str(directory), inst, method, cfg, target))
    proc.start(); child.close()
    # Startup is excluded, bounded separately, and never becomes a query timeout.
    if not parent.poll(60):
        proc.kill(); proc.join(); parent.close()
        record = dict(schema=SCHEMA, status='startup_failure', snapshots=[], rss_samples=[],
                      method=method, mode=mode, budget_s=budget_s, observed_until_s=0.)
        atomic_json(directory / 'observation.json', record)
        return record
    try:
        ready = parent.recv()
    except EOFError:
        ready = None
    if ready != 'ready':
        proc.join(); parent.close()
        raise RuntimeError('worker failed before ready handshake')
    monitor = psutil.Process(proc.pid)
    origin = time.perf_counter()
    parent.send(origin); parent.close()
    seen, observations, samples = set(), [], []
    status, completion, terminated_at = 'worker_failure', None, None
    observed_until = 0.
    try:
        while True:
            now = time.perf_counter() - origin
            if now >= budget_s:
                status = 'timeout'; terminated_at = now
                break
            # Observe file names only. Deserialization is offline after enforcement.
            available = sorted(directory.glob('snapshot-*.json'))
            observed = time.perf_counter() - origin
            if observed >= budget_s:
                status = 'timeout'; terminated_at = observed
                break
            for path in available:
                if path.name not in seen:
                    seen.add(path.name)
                    observations.append(dict(file=path.name, available_s=observed))
            observed_until = observed
            try:
                rss = monitor.memory_info().rss
                samples.append([time.perf_counter()-origin, rss])
            except psutil.NoSuchProcess:
                rss = 0
            if memory_limit_mb is not None and rss > memory_limit_mb * 1024**2:
                status = 'memory_limit'; terminated_at = time.perf_counter()-origin
                break
            done_exists = (directory / 'completion.json').exists()
            done_observed = time.perf_counter()-origin
            if done_exists and done_observed < budget_s:
                # Completion can race with the first directory scan. Observe all
                # final snapshots before accepting it, still strictly by cutoff.
                final_available = sorted(directory.glob('snapshot-*.json'))
                final_observed = time.perf_counter()-origin
                if final_observed >= budget_s:
                    status = 'timeout'; terminated_at = final_observed
                    break
                for path in final_available:
                    if path.name not in seen:
                        seen.add(path.name)
                        observations.append(dict(file=path.name, available_s=final_observed))
                observed_until = final_observed
                completion = final_observed
                status = 'returned'
                break
            if not proc.is_alive():
                status = 'worker_failure'
                break
            time.sleep(min(sample_interval_s, max(0., budget_s-(time.perf_counter()-origin))))
    finally:
        if proc.is_alive() and status != 'returned':
            proc.kill()
        proc.join(timeout=1.)
        if proc.is_alive():
            proc.kill(); proc.join()
    end = time.perf_counter()-origin
    record = dict(schema=SCHEMA, method=method, mode=mode, status=status,
                  budget_s=budget_s, observed_until_s=observed_until,
                  completion_observed_s=completion, process_end_s=end,
                  terminate_requested_s=terminated_at,
                  overshoot_s=max(0., end-budget_s) if status=='timeout' else 0.,
                  snapshots=observations, rss_samples=samples,
                  rss_sample_count=len(samples), rss_requested_interval_s=sample_interval_s,
                  rss_actual_intervals_s=[b[0]-a[0] for a,b in zip(samples,samples[1:])],
                  sampled_rss_peak_bytes=max((s[1] for s in samples), default=None),
                  memory_limit_mb=memory_limit_mb, child_pid=proc.pid,
                  clock='method_from_parent_go_including_publication_and_observation',
                  startup='excluded_ready_handshake', config=cfg)
    if completion is not None:
        record['completion'] = json.loads((directory/'completion.json').read_text())
        record['status'] = record['completion']['status']
    atomic_json(directory / 'observation.json', record)
    return record


def reconstruct(directory, observation):
    directory = Path(directory)
    return [(event['available_s'], points(json.loads((directory/event['file']).read_text())['points']))
            for event in observation['snapshots']]


def summarize(snapshots, ref, eligible, deadlines, *, observed_until_s, offset_s=0.):
    """Recall uses nine-decimal keys, HV raw objectives as in shared scoring.

    Time-to-K is right censored and uses simultaneous snapshot cardinality.
    """
    ref_keys = {pkey(p) for p in ref}
    corner = nadir(ref)
    hv = hypervolume_3d(ref, corner) if corner else 0.
    shifted = [(t+offset_s, ps) for t,ps in snapshots]
    row = dict(hv_reference_point=corner, ref_size=len(ref_keys),
               t_first=next((t for t,ps in shifted if ps), float('nan')),
               n_snapshots=len(snapshots), censor_time_s=observed_until_s+offset_s)
    for k in KS:
        reached = next((t for t,ps in shifted if len({pkey(p) for p in ps}&ref_keys)>=k), None)
        row[f't_to_{k}'] = reached if eligible and reached is not None else float('nan')
        row[f'reached_{k}'] = int(reached is not None) if eligible else float('nan')
        row[f'censored_{k}'] = int(reached is None) if eligible else float('nan')
        row[f'k_possible_{k}'] = int(k <= len(ref_keys)) if eligible else float('nan')
    for deadline in deadlines:
        available = next((ps for t,ps in reversed(shifted) if t<=deadline), [])
        row[f'recall@{deadline}'] = len({pkey(p) for p in available}&ref_keys)/len(ref_keys) if eligible and ref_keys else float('nan')
        row[f'hv@{deadline}'] = hypervolume_3d(available, corner)/hv if eligible and hv>0 else float('nan')
    return row
