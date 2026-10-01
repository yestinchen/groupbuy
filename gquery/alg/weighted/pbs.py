'''
weighted PBS (beam search): the same search structure as the Pareto version, but with a
known weight vector, returning the single best-F bundle. Pareto retention is
kept as-is (F is monotone in (Q,-C,R) so the optimum survives). use_lp adds an
MCKP LP bound; the bound is admissible so it only changes time, never the
answer.
'''

from __future__ import annotations

import threading
import time

from typing import List, Optional, Sequence, Tuple

from gquery.alg.candidates import PState, _non_dominated_raw
from gquery.alg.common import Item
from gquery.alg.weighted.bounds import slot_profits
from gquery.alg.weighted.objective import WeightedSearchResult, _seed, _setup

EPS = 1e-9


def solve_pbs(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    budget: float,
    alpha: float,
    l1: float = 1.0,
    l2: float = 1.0,
    l3: float = 1.0,
    beam_size: int = 100,
    use_lp: bool = True,
    timeout_flag: Optional[threading.Event] = None,
) -> WeightedSearchResult:
    t0 = time.perf_counter()
    n = len(slot_candidates)
    max_budget = budget * (1.0 + alpha)
    red_c, kept, oracle = _setup(slot_candidates, budget, alpha, l1, l2, l3, use_lp)
    best_v, best_c = _seed(oracle, red_c, kept, slot_candidates,
                           budget, alpha, l1, l2, l3)
    evals = 0
    pruned = 0

    def scal(q, c, r):
        pen = (l2 * max(0.0, c - budget) / (alpha * budget)) \
            if (alpha > 0 and budget > 0) else 0.0
        return l1 * q / n - pen + l3 * r / (5.0 * n)

    prof = slot_profits(red_c, l1, l3) if oracle is not None else None
    beam: List[PState] = [PState([], 0.0, 0.0, 0.0)]
    p1s: List[float] = [0.0]
    p2s: List[float] = [0.0]

    for depth in range(n):
        if timeout_flag is not None and timeout_flag.is_set():
            break
        nxt: List[PState] = []
        n1: List[float] = []
        n2: List[float] = []
        for idx, st in enumerate(beam):
            for j, (item, q) in enumerate(red_c[depth]):
                evals += 1
                cost = st.cost + item.price
                if cost > max_budget + EPS:
                    continue
                if oracle is not None:
                    a1 = p1s[idx] + prof[depth][j]
                    a2 = p2s[idx] + oracle.g2_profits[depth][j]
                    b = oracle.bound(depth + 1, a1, a2, cost)
                    if b is None or b <= best_v + EPS:
                        pruned += 1
                        continue
                else:
                    a1 = a2 = 0.0
                nxt.append(PState(st.choice + [j], st.quality + q, cost,
                                  st.rating + item.rating))
                n1.append(a1); n2.append(a2)
        if not nxt:
            beam = []
            break

        # keep by raw cost, not overage. safe: F is monotone in (Q,-C,R)
        keep_idx = {id(s): i for i, s in enumerate(nxt)}
        front = _non_dominated_raw(nxt)
        f1 = [n1[keep_idx[id(s)]] for s in front]
        f2 = [n2[keep_idx[id(s)]] for s in front]

        if depth == n - 1:
            for s in front:
                v = scal(s.quality, s.cost, s.rating)
                if v > best_v:
                    best_v = v
                    best_c = [kept[i][s.choice[i]] for i in range(n)]

        if beam_size > 0 and len(front) > beam_size:
            order = sorted(range(len(front)),
                           key=lambda i: -scal(front[i].quality, front[i].cost,
                                               front[i].rating))[:beam_size]
            front = [front[i] for i in order]
            f1 = [f1[i] for i in order]
            f2 = [f2[i] for i in order]
        beam, p1s, p2s = front, f1, f2

    return WeightedSearchResult(best_c, best_v, time.perf_counter() - t0,
                                False, evals, pruned)
