'''
weighted PTS (depth-first tree search): the same search structure as the Pareto version, but with a
known weight vector, returning the single best-F bundle. Pareto retention is
kept as-is (F is monotone in (Q,-C,R) so the optimum survives). use_lp adds an
MCKP LP bound; the bound is admissible so it only changes time, never the
answer.
'''

from __future__ import annotations

import threading
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from gquery.alg.candidates import _SuffixBounds
from gquery.alg.common import Item
from gquery.alg.weighted.bounds import slot_profits
from gquery.alg.weighted.objective import WeightedSearchResult, _seed, _setup

EPS = 1e-9


@dataclass
class _Node:
    depth: int
    choice: List[int]
    quality: float
    cost: float
    rating: float
    prof1: float
    prof2: float
    parent: Optional["_Node"] = None
    children: List["_Node"] = field(default_factory=list)
    untried: Optional[List[int]] = None
    dead: bool = False


def solve_pts(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    budget: float,
    alpha: float,
    l1: float = 1.0,
    l2: float = 1.0,
    l3: float = 1.0,
    expansion_budget: int = 500,
    candidate_limit: int = 50,
    use_lp: bool = True,
    prefix_dominance: bool = True,
    timeout_flag: Optional[threading.Event] = None,
) -> WeightedSearchResult:
    t0 = time.perf_counter()
    n = len(slot_candidates)
    max_budget = budget * (1.0 + alpha)
    red_c, kept, oracle = _setup(slot_candidates, budget, alpha, l1, l2, l3, use_lp)
    best_v, best_c = _seed(oracle, red_c, kept, slot_candidates,
                           budget, alpha, l1, l2, l3)
    prof = slot_profits(red_c, l1, l3)
    evals = 0
    pruned = 0

    def scal(q, c, r):
        pen = (l2 * max(0.0, c - budget) / (alpha * budget)) \
            if (alpha > 0 and budget > 0) else 0.0
        return l1 * q / n - pen + l3 * r / (5.0 * n)

    '''careful: the cap ranking must not depend on the prefix, otherwise
    prefix dominance below is unsafe.'''
    capped: List[List[int]] = []
    for d in range(n):
        order = sorted(range(len(red_c[d])), key=lambda j: -prof[d][j])
        capped.append(order[:candidate_limit] if candidate_limit > 0 else order)

    sb = _SuffixBounds([[red_c[d][j] for j in capped[d]] for d in range(n)])
    min_c = sb.min_c

    pref_q: List[List[float]] = [[] for _ in range(n + 1)]
    pref_c: List[List[float]] = [[] for _ in range(n + 1)]
    pref_r: List[List[float]] = [[] for _ in range(n + 1)]

    def _dominated(depth, q, c, r) -> bool:
        qs, cs, rs = pref_q[depth], pref_c[depth], pref_r[depth]
        hi = bisect_right(cs, c)
        for i in range(hi):
            if qs[i] >= q and rs[i] >= r:
                return True
        lo = bisect_left(cs, c)
        drop = [i for i in range(lo, len(cs)) if q >= qs[i] and r >= rs[i]]
        if drop:
            ds = set(drop)
            pref_q[depth] = qs = [qs[i] for i in range(len(qs)) if i not in ds]
            pref_c[depth] = cs = [cs[i] for i in range(len(cs)) if i not in ds]
            pref_r[depth] = rs = [rs[i] for i in range(len(rs)) if i not in ds]
            hi = bisect_right(cs, c)
        qs.insert(hi, q); cs.insert(hi, c); rs.insert(hi, r)
        return False

    def greedy_suffix(depth, q, c, r, cheapest=False):
        ch: List[int] = []
        for d in range(depth, n):
            best_j, best_k = -1, None
            for j in capped[d]:
                item, qq = red_c[d][j]
                if c + item.price > max_budget + EPS:
                    continue
                k = -item.price if cheapest else scal(q + qq, c + item.price,
                                                      r + item.rating)
                if best_k is None or k > best_k:
                    best_k, best_j = k, j
            if best_j < 0:
                return None
            it, qq = red_c[d][best_j]
            ch.append(best_j); q += qq; c += it.price; r += it.rating
        return ch, q, c, r

    root = _Node(0, [], 0.0, 0.0, 0.0, 0.0, 0.0)
    it_done = 0

    def actions(node: _Node) -> List[int]:
        nonlocal evals, pruned
        acts = []
        rest = min_c[node.depth + 1]
        for j in capped[node.depth]:
            evals += 1
            item, q = red_c[node.depth][j]
            cost = node.cost + item.price
            if cost + rest > max_budget + EPS:
                continue
            if oracle is not None:
                a1 = node.prof1 + prof[node.depth][j]
                a2 = node.prof2 + oracle.g2_profits[node.depth][j]
                b = oracle.bound(node.depth + 1, a1, a2, cost)
                if b is None or b <= best_v + EPS:
                    pruned += 1
                    continue
            acts.append(j)
        acts.sort(key=lambda j: -scal(node.quality + red_c[node.depth][j][1],
                                      node.cost + red_c[node.depth][j][0].price,
                                      node.rating + red_c[node.depth][j][0].rating))
        return acts

    for it_done in range(1, expansion_budget + 1):
        if timeout_flag is not None and (it_done & 0x3F) == 0 and timeout_flag.is_set():
            break
        node = root
        while node.depth < n and not node.dead:
            if node.untried is None:
                node.untried = actions(node)
            live = next((c for c in node.children if not c.dead), None)
            # depth first: finish the first live child's subtree before creating
            # its next sibling; a node without live children expands or is exhausted
            if live is not None:
                node = live
                continue
            if not node.untried:
                node.dead = True
            break

        if node.dead:
            if node is root:
                break
            continue

        leaf = node
        if node.depth < n and node.untried:
            j = -1
            while node.untried:
                cj = node.untried.pop(0)
                item, q = red_c[node.depth][cj]
                cq, cc, cr = node.quality + q, node.cost + item.price, \
                    node.rating + item.rating
                if prefix_dominance and _dominated(node.depth + 1, cq, cc, cr):
                    continue
                j = cj
                break
            if j < 0:
                if not any(not ch.dead for ch in node.children):
                    node.dead = True
                    if node is root:
                        break
                continue
            item, q = red_c[node.depth][j]
            leaf = _Node(node.depth + 1, node.choice + [j], node.quality + q,
                         node.cost + item.price, node.rating + item.rating,
                         node.prof1 + prof[node.depth][j],
                         node.prof2 + (oracle.g2_profits[node.depth][j]
                                       if oracle is not None else 0.0),
                         parent=node)
            node.children.append(leaf)

        if leaf.depth == n:
            sol = (list(leaf.choice), leaf.quality, leaf.cost, leaf.rating)
            leaf.dead = True
        else:
            sol = greedy_suffix(leaf.depth, leaf.quality, leaf.cost, leaf.rating)
            if sol is None and leaf.cost + min_c[leaf.depth] <= max_budget + EPS:
                sol = greedy_suffix(leaf.depth, leaf.quality, leaf.cost,
                                    leaf.rating, cheapest=True)
            if sol is not None:
                ch, q, c, r = sol
                sol = (list(leaf.choice) + list(ch), q, c, r)

        if sol is None:
            leaf.dead = True
            val = 0.0
        else:
            ch, q, c, r = sol
            val = scal(q, c, r)
            if val > best_v:
                best_v = val
                best_c = [kept[i][ch[i]] for i in range(n)]

        if leaf.dead:
            cur = leaf.parent
            while cur is not None and not cur.dead:
                if cur.untried or any(not ch.dead for ch in cur.children):
                    break
                cur.dead = True
                cur = cur.parent

    return WeightedSearchResult(best_c, best_v, time.perf_counter() - t0,
                                root.dead, evals, pruned, it_done)
