'''
Objective vectors and frontier metrics: the _Point objective record, the
(Q, -dC, R) nondominance filter, exact three dimensional hypervolume and
coverage.
'''

from __future__ import annotations

from typing import List, Sequence, Tuple

from gquery.alg.common import is_dominated


class _Point:
    __slots__ = ("quality", "rating", "cost", "cost_overage", "choice")

    def __init__(self, quality, rating, cost, cost_overage, choice):
        self.quality = quality
        self.rating = rating
        self.cost = cost
        self.cost_overage = cost_overage
        self.choice = choice

    def key(self) -> Tuple[float, float, float]:
        return (round(self.quality, 9), round(self.rating, 9), round(self.cost_overage, 9))


def _filter_non_dominated(points: Sequence[_Point]) -> List[_Point]:
    # non-dominated on (Q, -DeltaC, R)
    keep: List[_Point] = []
    for i, p in enumerate(points):
        if any(i != j and is_dominated(p, q) for j, q in enumerate(points)):
            continue
        keep.append(p)
    seen = set()
    out = []
    for p in keep:
        k = p.key()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _area_2d(pts: Sequence[Tuple[float, float]], x0: float, y0: float) -> float:
    # staircase sweep: sort by x desc, add a strip each time y improves
    area = 0.0
    best_y = y0
    for x, y in sorted(pts, key=lambda t: -t[0]):
        if x <= x0:
            continue
        if y > best_y:
            area += (x - x0) * (y - best_y)
            best_y = y
    return area


def hypervolume_3d(points: Sequence[_Point], ref: Tuple[float, float, float]) -> float:
    '''
    exact HV for maximizing (Q, -DeltaC, R). slice along distinct Q levels,
    dominated (-DeltaC, R) area times slab depth. ties share one slab.
    '''
    if not points:
        return 0.0
    rq, rc, rr = ref
    pts = [(p.quality, -p.cost_overage, p.rating) for p in points if p.quality > rq]
    if not pts:
        return 0.0

    levels = sorted({q for q, _, _ in pts}, reverse=True)
    total = 0.0
    for idx, q_level in enumerate(levels):
        lower = levels[idx + 1] if idx + 1 < len(levels) else rq
        depth = q_level - max(lower, rq)
        if depth <= 0:
            continue
        active = [(negc, r) for q, negc, r in pts if q >= q_level]
        total += _area_2d(active, rc, rr) * depth
    return total


def coverage(a: Sequence[_Point], b: Sequence[_Point]) -> float:
    # fraction of b weakly dominated by a
    if not b:
        return float("nan")
    if not a:
        return 0.0
    cnt = 0
    for q in b:
        if any((p.quality >= q.quality and p.rating >= q.rating
                and p.cost_overage <= q.cost_overage) for p in a):
            cnt += 1
    return cnt / len(b)
