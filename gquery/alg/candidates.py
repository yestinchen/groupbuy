'''
Candidate sets and partial assignments shared by every solver: the PState
partial assignment, three dimensional candidate reduction, the raw-cost and
final (cost overhead) nondominance filters with their bounded maxima
fallbacks, and the optimistic suffix bounds.
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from gquery.alg.common import Item


@dataclass
class PState:
    __slots__ = ("choice", "quality", "cost", "rating")
    choice: List[int]
    quality: float
    cost: float
    rating: float


def reduce_candidates_pairwise(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
) -> Tuple[List[List[Tuple[Item, float]]], List[List[int]]]:
    '''
    3-d candidate reduction: drop item k if some sibling has q>=, r>=, c<=
    with at least one strict. keeps the frontier intact.
    '''
    out_c: List[List[Tuple[Item, float]]] = []
    out_i: List[List[int]] = []
    for cands in slot_candidates:
        order = sorted(range(len(cands)), key=lambda j: (cands[j][0].price,
                                                         -cands[j][1],
                                                         -cands[j][0].rating))
        keep: List[int] = []
        for j in order:
            it, q = cands[j]
            dominated = False
            for k in keep:
                itk, qk = cands[k]
                if (qk >= q and itk.rating >= it.rating and itk.price <= it.price
                        and (qk > q or itk.rating > it.rating or itk.price < it.price)):
                    dominated = True
                    break
            if not dominated:
                keep.append(j)
        if not keep:
            keep = [order[0]]
        out_c.append([cands[j] for j in keep])
        out_i.append(keep)
    return out_c, out_i


class _SuffixBounds:
    '''
    max_q[d] / max_r[d] / min_c[d] over slots d..n-1. each objective is taken
    independently, so the triple is optimistic - not an attainable point.
    '''

    __slots__ = ("max_q", "max_r", "min_c")

    def __init__(self, red_c: Sequence[Sequence[Tuple[Item, float]]]) -> None:
        n = len(red_c)
        self.max_q = [0.0] * (n + 1)
        self.max_r = [0.0] * (n + 1)
        self.min_c = [0.0] * (n + 1)
        for d in range(n - 1, -1, -1):
            cands = red_c[d]
            self.max_q[d] = self.max_q[d + 1] + max((q for _, q in cands), default=0.0)
            self.max_r[d] = self.max_r[d + 1] + max((it.rating for it, _ in cands), default=0.0)
            self.min_c[d] = self.min_c[d + 1] + min((it.price for it, _ in cands), default=0.0)


def _non_dominated_raw_pairwise(states: List[PState]) -> List[PState]:
    '''
    prefix rule: non-dominated on (Q, -C, R) with raw cost.
    sort by cost first so any dominator appears before its victim.
    '''
    if not states:
        return []
    states = sorted(states, key=lambda s: (s.cost, -s.quality, -s.rating))
    keep: List[PState] = []
    for s in states:
        dominated = False
        # every t in keep has cost <= s.cost
        for t in keep:
            if t.quality >= s.quality and t.rating >= s.rating:
                if t.quality > s.quality or t.rating > s.rating or t.cost < s.cost:
                    dominated = True
                    break
        if not dominated:
            keep.append(s)
    return keep


_FINAL_DP = 9


def _non_dominated_final_pairwise(states: List[PState], budget: float) -> List[PState]:
    '''
    final frontier on (Q, -dC, R). round to _FINAL_DP decimals BEFORE the
    dominance test - raw float sums differ by an ulp depending on accumulation
    order and dominated points can survive. rounding first also keeps the
    relation transitive, a tolerance-based >= does not.
    '''
    if not states:
        return []
    ov = [round(max(0.0, s.cost - budget), _FINAL_DP) for s in states]
    qy = [round(s.quality, _FINAL_DP) for s in states]
    rt = [round(s.rating, _FINAL_DP) for s in states]
    keep: List[PState] = []
    for i, s in enumerate(states):
        dominated = False
        for j, t in enumerate(states):
            if i == j:
                continue
            better = (qy[j] > qy[i] or ov[j] < ov[i] or rt[j] > rt[i])
            nowor = (qy[j] >= qy[i] and ov[j] <= ov[i] and rt[j] >= rt[i])
            if better and nowor:
                dominated = True
                break
        if not dominated:
            keep.append(s)
    # dedup identical objective vectors, keep one representative
    seen = set()
    out = []
    for s in keep:
        k = (round(s.quality, 9), round(max(0.0, s.cost - budget), 9), round(s.rating, 9))
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _maxima_indices(points: Sequence[Tuple[float, float, float]]) -> List[int]:
    """Strict maxima of (cost ascending, quality descending, rating descending).

    Sweep cost and query a Fenwick prefix maximum over descending quality ranks.
    Insert identical triples together after querying so exact ties survive.
    Return stable sweep order. Requires finite coordinates.
    Time O(k log(k+1)), auxiliary space O(k).
    """
    order = sorted(range(len(points)), key=lambda j: (points[j][0], -points[j][1], -points[j][2]))
    ranks = {q: i + 1 for i, q in enumerate(sorted({p[1] for p in points}, reverse=True))}
    tree = [-math.inf] * (len(ranks) + 1)
    keep = []
    start = 0
    while start < len(order):
        point = points[order[start]]
        end = start + 1
        while end < len(order) and points[order[end]] == point:
            end += 1
        rank = ranks[point[1]]
        best = -math.inf
        i = rank
        while i:
            best = max(best, tree[i])
            i -= i & -i
        if best < point[2]:
            keep.extend(order[start:end])
        i = rank
        while i < len(tree):
            tree[i] = max(tree[i], point[2])
            i += i & -i
        start = end
    return keep


def _finite_points(points):
    return all(math.isfinite(x) for p in points for x in p)


def reduce_candidates(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
) -> Tuple[List[List[Tuple[Item, float]]], List[List[int]]]:
    """Strict reduction. Bound native-object scans before maxima fallback."""
    out_c, out_i = [], []
    for cands in slot_candidates:
        order = sorted(range(len(cands)), key=lambda j: (cands[j][0].price,
                                                        -cands[j][1], -cands[j][0].rating))
        remaining = len(cands) * max(1, len(cands).bit_length())
        keep = []
        for j in order:
            it, q = cands[j]
            dominated = False
            for k in keep:
                remaining -= 1
                if remaining < 0:
                    break
                itk, qk = cands[k]
                if (qk >= q and itk.rating >= it.rating and itk.price <= it.price
                        and (qk > q or itk.rating > it.rating or itk.price < it.price)):
                    dominated = True
                    break
            if remaining < 0:
                points = [(it.price, q, it.rating) for it, q in cands]
                if not _finite_points(points):
                    _, indices = reduce_candidates_pairwise([cands])
                    keep = indices[0]
                else:
                    keep = _maxima_indices(points)
                break
            if not dominated:
                keep.append(j)
        if not keep:
            # Preserve the legacy IndexError for an empty requirement.
            keep = [order[0]]
        out_c.append([cands[j] for j in keep])
        out_i.append(keep)
    return out_c, out_i


def _non_dominated_raw(states: List[PState]) -> List[PState]:
    ordered = sorted(states, key=lambda s: (s.cost, -s.quality, -s.rating))
    remaining = len(states) * max(1, len(states).bit_length())
    keep = []
    for s in ordered:
        dominated = False
        for t in keep:
            remaining -= 1
            if remaining < 0:
                points = [(v.cost, v.quality, v.rating) for v in states]
                if not _finite_points(points):
                    return _non_dominated_raw_pairwise(states)
                return [states[j] for j in _maxima_indices(points)]
            if t.quality >= s.quality and t.rating >= s.rating:
                if t.quality > s.quality or t.rating > s.rating or t.cost < s.cost:
                    dominated = True
                    break
        if not dominated:
            keep.append(s)
    return keep


def _non_dominated_final(states: List[PState], budget: float) -> List[PState]:
    points = [(round(max(0.0, s.cost - budget), _FINAL_DP),
               round(s.quality, _FINAL_DP), round(s.rating, _FINAL_DP)) for s in states]
    if not _finite_points(points):
        return _non_dominated_final_pairwise(states, budget)
    retained = set(_maxima_indices(points))
    seen = set()
    out = []
    # Preserve input order and the first representative of each rounded vector.
    for j, s in enumerate(states):
        if j in retained and points[j] not in seen:
            seen.add(points[j])
            out.append(s)
    return out
