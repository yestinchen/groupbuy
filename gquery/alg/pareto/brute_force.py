'''PBF: Pareto brute force. enumerate every assignment over the unreduced
candidate sets, keep the non-dominated ones.'''

from __future__ import annotations

import itertools
import os
import time
from typing import List, Sequence, Tuple

from gquery.alg.candidates import _FINAL_DP as _DP
from gquery.alg.pareto.metrics import _Point


def fast_non_dominated(points: Sequence[_Point]) -> List[_Point]:
    '''non-dominated filter in O(n log n): sort by overage so dominators come
    first, then a Fenwick tree over rating ranks keeps the running max quality.
    only used off the measured path (reference, PBF pruning).'''
    if not points:
        return []
    # careful: round to _DP decimals first. summation-order ulps give equal
    # ratings distinct ranks and dominated points slip through

    qy = [round(p.quality, _DP) for p in points]
    ov = [round(p.cost_overage, _DP) for p in points]
    rt = [round(p.rating, _DP) for p in points]
    order = sorted(range(len(points)),
                   key=lambda i: (ov[i], -qy[i], -rt[i]))
    ratings = sorted(set(rt))
    rank = {r: i + 1 for i, r in enumerate(ratings)}
    m = len(ratings)
    # position i is the i-th largest rating, so a prefix query covers rating >= r
    tree = [-float("inf")] * (m + 1)

    def update(i: int, v: float) -> None:
        while i <= m:
            if v > tree[i]:
                tree[i] = v
            i += i & (-i)

    def query(i: int) -> float:
        best = -float("inf")
        while i > 0:
            if tree[i] > best:
                best = tree[i]
            i -= i & (-i)
        return best

    keep: List[_Point] = []
    seen = set()
    for i in order:
        p = points[i]
        pos = m - rank[rt[i]] + 1
        if query(pos) >= qy[i]:
            continue
        k = (qy[i], ov[i], rt[i])
        if k in seen:
            continue
        seen.add(k)
        keep.append(p)
        update(pos, qy[i])
    return keep


_PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_bytes() -> int:
    '''resident set size of this process, read from /proc (cheap enough to poll).'''
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE


def solve_brute_force(
    inst,
    alpha: float,
    timeout_s: float = 60.0,
    max_combos: int | None = 200_000_000,
    prune_every: int = 200_000,
    max_memory_bytes: int | None = None,
    stats: dict | None = None,
) -> Tuple[List[_Point], bool, float, int]:
    '''PBF: enumerate everything over the unreduced candidate sets. skips
    outright if the assignment space exceeds max_combos (None disables the
    guard). max_memory_bytes bounds the growth of the process RSS over its
    value at the call, i.e. excluding the catalog and candidates already
    loaded; on reaching it the partial frontier is returned. stats, if given,
    receives termination_reason, baseline_rss_bytes and peak_rss_delta_bytes.'''
    budget = inst.query.budget
    max_budget = budget * (1.0 + alpha)
    sizes = [len(s) for s in inst.slots]
    base = rss_bytes()
    peak = 0
    def report(reason):
        if stats is not None:
            stats.update(termination_reason=reason, baseline_rss_bytes=base,
                         peak_rss_delta_bytes=peak)
    total = 1
    for s in sizes:
        total *= s
        if max_combos is not None and total > max_combos:
            report("combination_limit")
            return [], False, 0.0, total
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    feasible: List[_Point] = []
    checked = 0

    def over_memory():
        nonlocal peak
        peak = max(peak, rss_bytes() - base)
        return max_memory_bytes is not None and peak > max_memory_bytes

    for combo in itertools.product(*[range(s) for s in sizes]):
        checked += 1
        if (checked & 0xFFFF) == 0:
            reason = ("memory_limit" if over_memory() else
                      "timeout" if time.perf_counter() > deadline else None)
            if reason:
                partial = fast_non_dominated(feasible)
                report(reason)
                return partial, False, time.perf_counter() - t0, checked
        q = c = r = 0.0
        for d, j in enumerate(combo):
            it, qq = inst.slots[d][j]
            c += it.price
            if c > max_budget + 1e-9:
                break
            q += qq
            r += it.rating
        else:
            feasible.append(_Point(q, r, c, max(0.0, c - budget), list(combo)))
            # prune periodically to bound memory. loss-free
            if len(feasible) >= prune_every:
                feasible = fast_non_dominated(feasible)
                over_memory()
    frontier = fast_non_dominated(feasible)
    over_memory()
    report("complete")
    return frontier, True, time.perf_counter() - t0, checked
