'''
LP-bound machinery for the weighted search: dominance reduction, suffix LP
bounds for F = min(g1, g2), greedy incumbents.
careful: hull (LP-dominance) reduction must not prune candidates. an off-hull
item can still be the integer optimum (A=(0,0), B=(10,10), C=(5,4) at capacity 5),
so it is only safe inside the LP bound itself.
'''

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from gquery.alg.common import Item
from gquery.alg.weighted.ilp import _ClassData, _prepare, EPS


def slot_profits(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    lambda1: float,
    lambda3: float,
) -> List[List[float]]:
    # p = l1*q/n + l3*r/(5n)
    n = len(slot_candidates)
    return [[lambda1 * q / n + lambda3 * it.rating / (5.0 * n) for it, q in cands]
            for cands in slot_candidates]


def dominance_filter(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    profits: Sequence[Sequence[float]],
) -> Tuple[List[List[Tuple[Item, float]]], List[List[float]], List[List[int]]]:
    '''
    drop k when a sibling j has w_j <= w_k and p_j >= p_k.
    ok to do: penalty is monotone in total cost, so j never loses to k.
    '''
    out_c: List[List[Tuple[Item, float]]] = []
    out_p: List[List[float]] = []
    out_i: List[List[int]] = []

    for cands, ps in zip(slot_candidates, profits):
        order = sorted(range(len(cands)),
                       key=lambda j: (cands[j][0].price, -ps[j]))
        keep: List[int] = []
        best_p = float("-inf")
        for j in order:
            # ascending price: survive only by strictly improving on best profit
            if ps[j] > best_p + EPS:
                keep.append(j)
                best_p = ps[j]
        if not keep:
            keep = [order[0]]
        out_c.append([cands[j] for j in keep])
        out_p.append([ps[j] for j in keep])
        out_i.append(keep)
    return out_c, out_p, out_i


class SuffixLP:
    '''
    per-depth slope-sorted increments with prefix sums. each suffix LP bound
    becomes a binary search instead of a fresh sort.
    '''

    __slots__ = ("base_w", "base_p", "cum_w", "cum_p", "slope", "owner", "hull_pos", "n")

    def __init__(self, classes: Sequence[_ClassData]):
        n = len(classes)
        self.n = n
        self.base_w: List[float] = [0.0] * (n + 1)
        self.base_p: List[float] = [0.0] * (n + 1)
        for d in range(n - 1, -1, -1):
            self.base_w[d] = self.base_w[d + 1] + classes[d].base_weight
            self.base_p[d] = self.base_p[d + 1] + classes[d].base_profit

        self.cum_w: List[List[float]] = [[] for _ in range(n + 1)]
        self.cum_p: List[List[float]] = [[] for _ in range(n + 1)]
        self.slope: List[List[float]] = [[] for _ in range(n + 1)]
        self.owner: List[List[int]] = [[] for _ in range(n + 1)]
        self.hull_pos: List[List[int]] = [[] for _ in range(n + 1)]

        for d in range(n):
            incs: List[Tuple[float, float, float, int, int]] = []
            for ci in range(d, n):
                h = classes[ci].hull
                for a in range(len(h) - 1):
                    dw = h[a + 1][0] - h[a][0]
                    dp = h[a + 1][1] - h[a][1]
                    if dw > EPS and dp > EPS:
                        incs.append((dp / dw, dw, dp, ci, a))
            incs.sort(key=lambda t: -t[0])
            cw, cp = [0.0], [0.0]
            for slope, dw, dp, ci, a in incs:
                cw.append(cw[-1] + dw)
                cp.append(cp[-1] + dp)
                self.slope[d].append(slope)
                self.owner[d].append(ci)
                self.hull_pos[d].append(a)
            self.cum_w[d] = cw
            self.cum_p[d] = cp
        self.cum_w[n] = [0.0]
        self.cum_p[n] = [0.0]

    def bound(self, depth: int, capacity: float) -> Optional[float]:
        # LP optimum over slots depth..n-1. None if infeasible.
        slack = capacity - self.base_w[depth]
        if slack < -EPS:
            return None
        cw = self.cum_w[depth]
        # largest i with cw[i] <= slack
        lo, hi = 0, len(cw) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cw[mid] <= slack:
                lo = mid
            else:
                hi = mid - 1
        total = self.base_p[depth] + self.cum_p[depth][lo]
        if lo < len(self.slope[depth]):
            total += self.slope[depth][lo] * (slack - cw[lo])
        return total

    def completion(
        self, depth: int, capacity: float, classes: Sequence[_ClassData]
    ) -> Optional[Tuple[float, List[int]]]:
        # bound plus feasible integral completion (fractional tail dropped)
        slack = capacity - self.base_w[depth]
        if slack < -EPS:
            return None
        cw = self.cum_w[depth]
        lo, hi = 0, len(cw) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cw[mid] <= slack:
                lo = mid
            else:
                hi = mid - 1
        pos = {ci: 0 for ci in range(depth, self.n)}
        for i in range(lo):
            ci = self.owner[depth][i]
            pos[ci] = self.hull_pos[depth][i] + 1
        bound = self.base_p[depth] + self.cum_p[depth][lo]
        if lo < len(self.slope[depth]):
            bound += self.slope[depth][lo] * (slack - cw[lo])
        choice = [classes[ci].hull[pos[ci]][2] for ci in range(depth, self.n)]
        return bound, choice


class BoundOracle:
    '''
    F = min(g1, g2) with g1 = sum p*x, g2 = sum p'*x + l2/alpha.
    upper bound = min of the two suffix LP bounds. F <= each pointwise, so this is safe.
    '''

    def __init__(
        self,
        weights: List[List[float]],
        profits: List[List[float]],
        budget: float,
        alpha: float,
        lambda2: float,
    ):
        self.weights = weights
        self.profits = profits
        self.budget = budget
        self.alpha = alpha
        self.lambda2 = lambda2
        self.max_budget = budget * (1.0 + alpha)
        self.penalty_active = alpha > 0 and budget > 0 and lambda2 > 0
        n = len(weights)

        self.g1 = _prepare([(weights[i], profits[i]) for i in range(n)])
        self.lp1 = SuffixLP(self.g1)
        if self.penalty_active:
            coef = lambda2 / (alpha * budget)
            self.g2_profits = [[profits[i][j] - coef * weights[i][j]
                                for j in range(len(weights[i]))] for i in range(n)]
            self.g2 = _prepare([(weights[i], self.g2_profits[i]) for i in range(n)])
            self.lp2 = SuffixLP(self.g2)
            self.g2_const = lambda2 / alpha
        else:
            self.g2_profits = profits
            self.g2 = self.g1
            self.lp2 = self.lp1
            self.g2_const = 0.0

    def penalty(self, cost: float) -> float:
        if not self.penalty_active:
            return 0.0
        return max(0.0, cost - self.budget) * (self.g2_const / self.budget)

    def value(self, prof1: float, cost: float) -> float:
        return prof1 - self.penalty(cost)

    def bound(self, depth: int, prof1: float, prof2: float, cost: float) -> Optional[float]:
        # upper bound over all completions of the prefix. None = infeasible.
        room = self.max_budget - cost
        if room < -EPS:
            return None
        b1 = self.lp1.bound(depth, room)
        if b1 is None:
            return None
        ub = prof1 + b1
        if self.penalty_active:
            b2 = self.lp2.bound(depth, room)
            if b2 is None:
                return None
            ub = min(ub, prof2 + b2 + self.g2_const)
        return ub

    def completion(self, depth: int, cost: float) -> Optional[Tuple[float, List[int]]]:
        return self.lp1.completion(depth, self.max_budget - cost, self.g1)

    def root_bound(self) -> Optional[float]:
        return self.bound(0, 0.0, 0.0, 0.0)


def lp_completion(
    classes: Sequence[_ClassData],
    capacity: float,
) -> Optional[Tuple[float, List[int]]]:
    '''
    LP bound plus a feasible integral completion. whole increments in slope
    order, fractional tail dropped. returns (bound, item indices) or None.
    '''
    base_w = sum(cd.base_weight for cd in classes)
    if base_w > capacity + EPS:
        return None
    total = sum(cd.base_profit for cd in classes)
    pos = [0] * len(classes)

    increments: List[Tuple[float, float, float, int, int]] = []
    for ci, cd in enumerate(classes):
        h = cd.hull
        for a in range(len(h) - 1):
            dw = h[a + 1][0] - h[a][0]
            dp = h[a + 1][1] - h[a][1]
            if dw > EPS and dp > EPS:
                increments.append((dp / dw, dw, dp, ci, a))
    increments.sort(key=lambda t: -t[0])

    slack = capacity - base_w
    bound = total
    for slope, dw, dp, ci, a in increments:
        if slack <= EPS:
            break
        # within-class increments have to apply in order
        if pos[ci] != a:
            continue
        if dw <= slack:
            bound += dp
            slack -= dw
            pos[ci] = a + 1
        else:
            # fractional tail only counts toward the bound
            bound += slope * slack
            slack = 0.0
            break

    choice = [classes[ci].hull[pos[ci]][2] for ci in range(len(classes))]
    return bound, choice


def greedy_incumbent(
    oracle: BoundOracle,
) -> Optional[Tuple[float, List[int]]]:
    # cheapest per slot, then best-gain upgrades
    n = len(oracle.weights)
    choice = [min(range(len(oracle.weights[i])), key=lambda j: oracle.weights[i][j])
              for i in range(n)]
    cost = sum(oracle.weights[i][choice[i]] for i in range(n))
    if cost > oracle.max_budget + EPS:
        return None
    prof = sum(oracle.profits[i][choice[i]] for i in range(n))

    improved = True
    while improved:
        improved = False
        best = None
        for i in range(n):
            cur_w = oracle.weights[i][choice[i]]
            cur_p = oracle.profits[i][choice[i]]
            for j in range(len(oracle.weights[i])):
                dw = oracle.weights[i][j] - cur_w
                dp = oracle.profits[i][j] - cur_p
                if dp <= EPS or cost + dw > oracle.max_budget + EPS:
                    continue
                gain = oracle.value(prof + dp, cost + dw) - oracle.value(prof, cost)
                if gain > EPS and (best is None or gain > best[0]):
                    best = (gain, i, j, dw, dp)
        if best is not None:
            _, i, j, dw, dp = best
            choice[i] = j
            cost += dw
            prof += dp
            improved = True
    return oracle.value(prof, cost), choice
