'''PBS with a configurable prefix dominance rule, for the ablation bench only.
prefix_rule "raw_cost" is PBS; "overage" replaces raw cost with cost overhead
in the partial assignment dominance (the ΔC ablation arm).'''

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from gquery.alg.common import Item
from gquery.alg.candidates import (
    PState, _non_dominated_final, _non_dominated_raw, reduce_candidates,
)


@dataclass
class VPACSResult:
    frontier: List[PState]
    n_evals: int
    n_before: int
    n_after: int
    solve_time_s: float
    reduction_time_s: float


def solve_pacs_variant(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    budget: float,
    alpha: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    beam_size: int = 100,
    prefix_rule: str = "raw_cost",
    use_reduction: bool = True,
    timeout_flag: Optional[threading.Event] = None,
) -> VPACSResult:
    n = len(slot_candidates)
    max_budget = budget * (1.0 + alpha)

    t_red = time.perf_counter()
    n_before = sum(len(c) for c in slot_candidates)
    if use_reduction:
        red_c, kept = reduce_candidates(slot_candidates)
    else:
        red_c = [list(c) for c in slot_candidates]
        kept = [list(range(len(c))) for c in slot_candidates]
    n_after = sum(len(c) for c in red_c)
    reduction_s = time.perf_counter() - t_red

    def scal(s: PState) -> float:
        pen = (lambda2 * max(0.0, s.cost - budget) / (alpha * budget)) \
            if (alpha > 0 and budget > 0) else 0.0
        return lambda1 * s.quality / n - pen + lambda3 * s.rating / (5.0 * n)

    t0 = time.perf_counter()
    # reachability bound, as in gquery.alg.pareto.pbs.solve_pbs
    min_suffix = [0.0] * (n + 1)
    for d in range(n - 1, -1, -1):
        min_suffix[d] = min_suffix[d + 1] + (min(it.price for it, _ in red_c[d])
                                             if red_c[d] else float("inf"))
    evals = 0
    beam: List[PState] = [PState([], 0.0, 0.0, 0.0)]
    for depth in range(n):
        if timeout_flag is not None and timeout_flag.is_set():
            break
        nxt: List[PState] = []
        for st in beam:
            for j, (item, q) in enumerate(red_c[depth]):
                evals += 1
                cost = st.cost + item.price
                if cost + min_suffix[depth + 1] > max_budget + 1e-9:
                    continue
                nxt.append(PState(st.choice + [j], st.quality + q, cost,
                                  st.rating + item.rating))
        if not nxt:
            beam = []
            break
        front = (_non_dominated_raw(nxt) if prefix_rule == "raw_cost"
                 else _non_dominated_final(nxt, budget))
        if beam_size > 0 and len(front) > beam_size:
            front = sorted(front, key=scal, reverse=True)[:beam_size]
        beam = front

    frontier = _non_dominated_final(beam, budget) if beam else []
    for s in frontier:
        s.choice = [kept[i][s.choice[i]] for i in range(len(s.choice))]
    return VPACSResult(frontier, evals, n_before, n_after,
                       time.perf_counter() - t0, reduction_s)
