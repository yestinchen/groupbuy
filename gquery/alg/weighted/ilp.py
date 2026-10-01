'''
Weighted CGQ exact solver, pure-Python branch and bound. the problem it solves:
  max  l1/n * sum(fm*x) + l3/(5n) * sum(r*x) - l2/(alpha*B) * y
  s.t. one x per slot, sum(c*x) <= (1+alpha)*B, y >= sum(c*x) - B, y >= 0
key identity: F(x) = min(g1, g2), where g1 = sum p*x and
g2 = sum p'*x + l2/alpha with p'_io = p_io - l2*c_o/(alpha B).
below budget g2 >= g1, above it g2 <= g1.
'''

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from gquery.alg.common import GroupQuery, Item


EPS = 1e-9


def _upper_hull(points: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    # upper hull of (weight, profit). off-hull items are LP-dominated.
    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    dedup: List[Tuple[float, float, int]] = []
    for w, p, i in pts:
        if dedup and abs(dedup[-1][0] - w) < EPS:
            continue
        if dedup and p <= dedup[-1][1] + EPS:
            continue
        dedup.append((w, p, i))
    hull: List[Tuple[float, float, int]] = []
    for pt in dedup:
        while len(hull) >= 2:
            (w1, p1, _), (w2, p2, _) = hull[-2], hull[-1]
            w3, p3, _ = pt
            # slope must strictly decrease along the hull
            if (p2 - p1) * (w3 - w2) <= (p3 - p2) * (w2 - w1) + EPS:
                hull.pop()
            else:
                break
        hull.append(pt)
    return hull


@dataclass
class _ClassData:
    weights: List[float]
    profits: List[float]
    hull: List[Tuple[float, float, int]]

    @property
    def base_weight(self) -> float:
        return self.hull[0][0]

    @property
    def base_profit(self) -> float:
        return self.hull[0][1]


def _prepare(classes: List[Tuple[List[float], List[float]]]) -> List[_ClassData]:
    out = []
    for weights, profits in classes:
        pts = [(w, p, i) for i, (w, p) in enumerate(zip(weights, profits))]
        out.append(_ClassData(weights, profits, _upper_hull(pts)))
    return out


def _lp_bound(classes: Sequence[_ClassData], capacity: float) -> Optional[float]:
    '''
    MCKP LP optimum: lightest item per class, then hull increments by slope.
    slopes decrease within a class, so taking them in global slope order is the LP opt.
    returns None if even the lightest choice per class is over capacity.
    '''
    base_w = 0.0
    base_p = 0.0
    # (slope, dw, dp)
    increments: List[Tuple[float, float, float]] = []
    for cd in classes:
        base_w += cd.base_weight
        base_p += cd.base_profit
        h = cd.hull
        for a in range(len(h) - 1):
            dw = h[a + 1][0] - h[a][0]
            dp = h[a + 1][1] - h[a][1]
            if dw > EPS and dp > EPS:
                increments.append((dp / dw, dw, dp))

    if base_w > capacity + EPS:
        return None

    slack = capacity - base_w
    total = base_p
    increments.sort(key=lambda t: -t[0])
    for slope, dw, dp in increments:
        if slack <= EPS:
            break
        if dw <= slack:
            total += dp
            slack -= dw
        else:
            # fractional last increment
            total += slope * slack
            slack = 0.0
    return total


def _greedy_incumbent(classes: Sequence[_ClassData], capacity: float
                      ) -> Optional[Tuple[float, List[int]]]:
    # lightest per class, then upgrade by best slope
    choice = [cd.hull[0][2] for cd in classes]
    weight = sum(cd.base_weight for cd in classes)
    if weight > capacity + EPS:
        return None
    profit = sum(cd.base_profit for cd in classes)

    pos = [0] * len(classes)
    while True:
        best = None
        for ci, cd in enumerate(classes):
            a = pos[ci]
            if a + 1 >= len(cd.hull):
                continue
            dw = cd.hull[a + 1][0] - cd.hull[a][0]
            dp = cd.hull[a + 1][1] - cd.hull[a][1]
            if dw <= capacity - weight + EPS and dw > EPS and dp > EPS:
                slope = dp / dw
                if best is None or slope > best[0]:
                    best = (slope, ci, dw, dp)
        if best is None:
            break
        _, ci, dw, dp = best
        pos[ci] += 1
        weight += dw
        profit += dp
        choice[ci] = classes[ci].hull[pos[ci]][2]
    return profit, choice


def _branch_and_bound_cgq(
    g1_classes: List[_ClassData],
    g2_classes: List[_ClassData],
    weights: List[List[float]],
    p1: List[List[float]],
    p2: List[List[float]],
    capacity: float,
    budget: float,
    g2_const: float,
    penalty_active: bool,
    timeout_flag: Optional[threading.Event],
    node_budget: int,
    trace: Optional[List[Tuple[float, float]]] = None,
    t_start: float = 0.0,
) -> Tuple[Optional[float], Optional[List[int]], int, bool]:
    '''
    BB on F = min(g1, g2). node bound = min(LP(g1), LP(g2)) - safe, since
    F <= g1 and F <= g2 pointwise. leaves are scored with the true F.
    '''
    n = len(weights)
    nodes = 0
    timed_out = False

    # incumbent: greedy on g1 and on g2, scored by the true objective
    best_value = float("-inf")
    best_choice: Optional[List[int]] = None
    for cls in (g1_classes, g2_classes) if penalty_active else (g1_classes,):
        seed = _greedy_incumbent(cls, capacity)
        if seed is None:
            continue
        _, ch = seed
        w = sum(weights[i][ch[i]] for i in range(n))
        pr = sum(p1[i][ch[i]] for i in range(n))
        v = pr - _penalty(w, budget, capacity, g2_const) if penalty_active else pr
        if v > best_value:
            best_value, best_choice = v, list(ch)
    if best_choice is None:
        return None, None, 0, True
    if trace is not None:
        trace.append((time.perf_counter() - t_start, best_value))

    def rec(depth: int, weight: float, prof1: float, prof2: float,
            choice: List[int]) -> None:
        nonlocal best_value, best_choice, nodes, timed_out
        if timed_out:
            return
        nodes += 1
        if nodes > node_budget:
            timed_out = True
            return
        if (nodes & 0x3FF) == 0 and timeout_flag is not None and timeout_flag.is_set():
            timed_out = True
            return

        if depth == n:
            v = prof1 - _penalty(weight, budget, capacity, g2_const) if penalty_active else prof1
            if v > best_value + EPS:
                best_value, best_choice = v, list(choice)
                if trace is not None:
                    trace.append((time.perf_counter() - t_start, v))
            return

        room = capacity - weight
        b1 = _lp_bound(g1_classes[depth:], room)
        if b1 is None:
            return
        ub = prof1 + b1
        if penalty_active:
            b2 = _lp_bound(g2_classes[depth:], room)
            if b2 is None:
                return
            ub = min(ub, prof2 + b2 + g2_const)
        if ub <= best_value + EPS:
            return

        cd = g1_classes[depth]
        order = sorted(range(len(cd.weights)), key=lambda j: -cd.profits[j])
        for j in order:
            w = weights[depth][j]
            if weight + w > capacity + EPS:
                continue
            choice.append(j)
            rec(depth + 1, weight + w, prof1 + p1[depth][j],
                prof2 + (p2[depth][j] if penalty_active else 0.0), choice)
            choice.pop()
            if timed_out:
                return

    rec(0, 0.0, 0.0, 0.0, [])
    return best_value, best_choice, nodes, not timed_out


def _penalty(cost: float, budget: float, capacity: float, g2_const: float) -> float:
    # l2 * max(C-B, 0) / (alpha B). here g2_const = l2/alpha.
    if budget <= 0 or g2_const <= 0:
        return 0.0
    return max(0.0, cost - budget) * (g2_const / budget)


def solve_ilp(
    query: GroupQuery,
    slot_candidates: List[List[Tuple[Item, float]]],
    alpha: float,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    timeout_flag: Optional[threading.Event] = None,
    node_budget: int = 20_000_000,
    use_dominance: bool = True,
    trace: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Optional[List[int]], Dict]:
    '''
    exact optimum of the weighted objective.
    use_dominance drops item k when a sibling j has w_j <= w_k and p_j >= p_k.
    safe: the penalty is monotone in total cost, so swapping in j never hurts.
    '''
    n = len(slot_candidates)
    budget = query.budget
    max_budget = budget * (1.0 + alpha)
    penalty_active = alpha > 0 and budget > 0

    base_profit = []
    weights = []
    for cands in slot_candidates:
        ws = [it.price for it, _ in cands]
        ps = [lambda1 * q / n + lambda3 * it.rating / (5.0 * n) for it, q in cands]
        weights.append(ws)
        base_profit.append(ps)

    kept_idx: Optional[List[List[int]]] = None
    if use_dominance:
        kept_idx = []
        rw, rp = [], []
        for ws, ps in zip(weights, base_profit):
            order = sorted(range(len(ws)), key=lambda j: (ws[j], -ps[j]))
            keep, best_p = [], float("-inf")
            for j in order:
                if ps[j] > best_p + EPS:
                    keep.append(j)
                    best_p = ps[j]
            if not keep:
                keep = [order[0]]
            kept_idx.append(keep)
            rw.append([ws[j] for j in keep])
            rp.append([ps[j] for j in keep])
        weights, base_profit = rw, rp

    # g2 folds the penalty into per-item profits, l2/alpha gets added back at the bound
    if penalty_active:
        coef = lambda2 / (alpha * budget)
        g2_profit = [[base_profit[i][j] - coef * weights[i][j]
                      for j in range(len(weights[i]))] for i in range(n)]
        g2_const = lambda2 / alpha
    else:
        g2_profit = base_profit
        g2_const = 0.0

    g1_classes = _prepare([(weights[i], base_profit[i]) for i in range(n)])
    g2_classes = _prepare([(weights[i], g2_profit[i]) for i in range(n)])

    _t0 = time.perf_counter()
    value, choice, nodes, proved = _branch_and_bound_cgq(
        g1_classes, g2_classes, weights, base_profit, g2_profit,
        max_budget, budget, g2_const, penalty_active, timeout_flag, node_budget,
        trace, _t0,
    )
    stats = {"nodes": nodes, "cases": 1, "proved_optimal": proved,
             "n_after_reduction": sum(len(w) for w in weights)}
    if choice is None:
        return None, stats
    # map back to original indices
    if kept_idx is not None:
        choice = [kept_idx[i][choice[i]] for i in range(n)]
    return choice, stats
