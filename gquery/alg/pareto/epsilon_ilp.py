'''epsilon-constraint MILP baseline for Pareto CGQ: one HiGHS solve per cell of
a g x g grid over the cost overhead and total rating bounds.'''

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from gquery.alg.common import GroupQuery, Item
from gquery.alg.pareto.metrics import _Point, _filter_non_dominated


def _build(slots, budget, alpha, eps_cost, eps_rating):
    '''
    max sum q*x  s.t. one x per slot, sum c*x <= (1+alpha)B,
    sum c*x - s <= B (so s >= DeltaC), s <= eps_cost, sum r*x >= eps_rating.
    '''
    from scipy.optimize import LinearConstraint, Bounds
    from scipy.sparse import csr_matrix

    sizes = [len(s) for s in slots]
    n_x = sum(sizes)
    penalty_active = alpha > 0 and budget > 0
    n_vars = n_x + (1 if penalty_active else 0)

    q = np.empty(n_x); c = np.empty(n_x); r = np.empty(n_x)
    k = 0
    for cands in slots:
        for item, qual in cands:
            q[k] = qual; c[k] = item.price; r[k] = item.rating
            k += 1

    # ties on Q can give weakly dominated points, filtered at the end.
    obj = np.zeros(n_vars)
    obj[:n_x] = -q

    rows: List[int] = []; cols: List[int] = []; vals: List[float] = []
    lb: List[float] = []; ub: List[float] = []

    off = 0
    for i, sz in enumerate(sizes):
        rows.extend([i] * sz); cols.extend(range(off, off + sz)); vals.extend([1.0] * sz)
        off += sz; lb.append(1.0); ub.append(1.0)

    row = len(sizes)
    max_budget = budget * (1.0 + alpha)
    rows.extend([row] * n_x); cols.extend(range(n_x)); vals.extend(c.tolist())
    lb.append(-np.inf); ub.append(max_budget)

    if penalty_active:
        row += 1
        rows.extend([row] * n_x); cols.extend(range(n_x)); vals.extend(c.tolist())
        rows.append(row); cols.append(n_x); vals.append(-1.0)
        lb.append(-np.inf); ub.append(budget)

    if eps_rating is not None:
        row += 1
        rows.extend([row] * n_x); cols.extend(range(n_x)); vals.extend(r.tolist())
        lb.append(eps_rating); ub.append(np.inf)

    A = csr_matrix((vals, (rows, cols)), shape=(len(lb), n_vars))
    var_lb = np.zeros(n_vars); var_ub = np.ones(n_vars)
    integrality = np.ones(n_vars)
    if penalty_active:
        # eps_cost enters only through the upper bound on s
        cap = alpha * budget if eps_cost is None else min(eps_cost, alpha * budget)
        var_ub[n_x] = max(cap, 0.0)
        integrality[n_x] = 0

    return (obj, LinearConstraint(A, np.asarray(lb), np.asarray(ub)),
            integrality, Bounds(var_lb, var_ub), sizes, n_x, penalty_active)


def _solve_one(slots, budget, alpha, eps_cost, eps_rating, time_limit):
    from scipy.optimize import milp
    obj, cons, integrality, bounds, sizes, n_x, _ = _build(
        slots, budget, alpha, eps_cost, eps_rating)
    options = {"disp": False, "mip_rel_gap": 0.0, "threads": 1}
    if time_limit and time_limit > 0:
        options["time_limit"] = time_limit
    # SciPy forwards backend options verbatim; suppress only this expected notice.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*threads.*", category=RuntimeWarning)
        res = milp(c=obj, constraints=cons, integrality=integrality,
                   bounds=bounds, options=options)
    # only take proven optima. a time-limited incumbent is not exact.
    if res.x is None or res.status != 0:
        return None, res.status
    choice: List[int] = []
    off = 0
    for sz in sizes:
        choice.append(int(np.argmax(res.x[off:off + sz])))
        off += sz
    return choice, res.status


def _point_from_choice(slots, choice, budget) -> _Point:
    q = c = r = 0.0
    for i, j in enumerate(choice):
        item, qual = slots[i][j]
        q += qual; c += item.price; r += item.rating
    return _Point(q, r, c, max(0.0, c - budget), choice)


def solve_epsilon_ilp(
    query: GroupQuery,
    slots: List[List[Tuple[Item, float]]],
    alpha: float,
    grid: int,
    timeout_flag: Optional[threading.Event],
    time_budget_s: float,
    trace: Optional[List] = None,
) -> Tuple[List[_Point], Dict]:
    '''sweep the eps grid, one solve per cell. returns (frontier, stats).'''
    budget = query.budget
    t0 = time.perf_counter()
    n_solves = 0
    stats = {
        "grid": grid,
        "n_solves": 0,
        "hit_time_budget": False,
        "n_infeasible": 0,
        "n_nonoptimal": 0,
    }

    # payoff table: ranges for the two constrained objectives
    base_choice, base_status = _solve_one(
        slots, budget, alpha, None, None, time_budget_s)
    n_solves += 1
    if base_choice is None:
        stats["n_solves"] = n_solves
        if base_status == 2:
            stats["n_infeasible"] += 1
        else:
            stats["n_nonoptimal"] += 1
        stats.update(reduction_time_s=0.0, search_time_s=time.perf_counter()-t0,
                     output_time_s=0.0, counts_after=[len(c) for c in slots])
        return [], stats
    points: List[_Point] = [_point_from_choice(slots, base_choice, budget)]
    if trace is not None:
        trace.append((time.perf_counter() - t0, points[0]))

    r_min = sum(min(it.rating for it, _ in cands) for cands in slots)
    r_max = sum(max(it.rating for it, _ in cands) for cands in slots)
    overrun_max = alpha * budget if alpha > 0 else 0.0

    rating_levels = ([r_min] if r_max <= r_min + 1e-9
                     else list(np.linspace(r_min, r_max, grid)))
    cost_levels = ([None] if overrun_max <= 0
                   else list(np.linspace(0.0, overrun_max, grid)))

    for eps_r in rating_levels:
        for eps_c in cost_levels:
            if timeout_flag is not None and timeout_flag.is_set():
                stats["hit_time_budget"] = True
                break
            if time_budget_s > 0 and (time.perf_counter() - t0) > time_budget_s:
                stats["hit_time_budget"] = True
                break
            remaining = (time_budget_s - (time.perf_counter() - t0)) if time_budget_s > 0 else 0
            choice, status = _solve_one(slots, budget, alpha, eps_c, eps_r, remaining)
            n_solves += 1
            if choice is None:
                if status == 2:
                    stats["n_infeasible"] += 1
                else:
                    stats["n_nonoptimal"] += 1
                continue
            points.append(_point_from_choice(slots, choice, budget))
            if trace is not None:
                trace.append((time.perf_counter() - t0, points[-1]))
        if stats["hit_time_budget"]:
            break

    stats["n_solves"] = n_solves
    stats["solve_time_s"] = time.perf_counter() - t0
    t_output = time.perf_counter()
    front = _filter_non_dominated(points)
    stats.update(reduction_time_s=0.0, search_time_s=t_output-t0,
                 output_time_s=time.perf_counter()-t_output, counts_after=[len(c) for c in slots])
    return front, stats
