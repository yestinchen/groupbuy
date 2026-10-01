'''
PBS: pareto beam search.
prefix dominance on (Q, -C, R) with raw cost, final frontier on (Q, -dC, R).
extensions must pass the reachability bound, so every retained partial
assignment has a feasible completion.
'''

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from gquery.alg.candidates import (
    PState, _non_dominated_final, _non_dominated_raw, reduce_candidates,
)
from gquery.alg.common import Item


@dataclass
class PBSResult:
    frontier: List[PState]
    n_evals: int
    n_before: int
    n_after: int
    solve_time_s: float
    reduction_time_s: float
    prefix_rule: str
    level_widths: Tuple[int, ...] = ()


def solve_pbs(
    slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
    budget: float,
    alpha: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    beam_size: int = 50,
    # "raw_cost" | "overage" (overage only for the ablation)
    prefix_rule: str = "raw_cost",
    use_reduction: bool = True,
    timeout_flag: Optional[threading.Event] = None,
    trace: Optional[List[Tuple[float, 'PState']]] = None,
) -> PBSResult:
    n = len(slot_candidates)
    max_budget = budget * (1.0 + alpha)

    t_red = time.perf_counter()
    n_before = sum(len(c) for c in slot_candidates)
    if use_reduction and n and all(slot_candidates):
        red_c, kept = reduce_candidates(slot_candidates)
    else:
        red_c = [list(c) for c in slot_candidates]
        kept = [list(range(len(c))) for c in slot_candidates]
    n_after = sum(len(c) for c in red_c)
    reduction_s = time.perf_counter() - t_red

    t0 = time.perf_counter()
    # reachability bound: min_suffix[d] is the least total price of one candidate
    # for each of requirements d..n-1, so an extension at depth d is kept only if
    # its cost plus min_suffix[d+1] fits the relaxed cap
    min_suffix = [0.0] * (n + 1)
    for d in range(n - 1, -1, -1):
        min_suffix[d] = min_suffix[d + 1] + (min(it.price for it, _ in red_c[d])
                                             if red_c[d] else float("inf"))
    evals = 0
    level_widths: List[int] = []
    beam: List[PState] = [PState([], 0.0, 0.0, 0.0)]

    completed = False
    cancelled = False
    truncated = False
    for depth in range(n):
        if timeout_flag is not None and timeout_flag.is_set():
            break
        nxt: List[PState] = []
        for st in beam:
            if timeout_flag is not None and timeout_flag.is_set():
                cancelled = True
                break
            for j, (item, q) in enumerate(red_c[depth]):
                evals += 1
                cost = st.cost + item.price
                if cost + min_suffix[depth + 1] > max_budget + 1e-9:
                    continue
                nxt.append(PState(st.choice + [j], st.quality + q, cost,
                                  st.rating + item.rating))
        if cancelled:
            break
        if not nxt:
            beam = []
            completed = True
            break

        if prefix_rule == "raw_cost":
            front = _non_dominated_raw(nxt)
        else:
            front = _non_dominated_final(nxt, budget)

        level_widths.append(len(front))

        if beam_size > 0 and len(front) > beam_size:
            truncated = True
            # truncate by F' = l1*Q/n - l2*dC/(alpha*B) + l3*R/(5n)
            def scal(s: PState) -> float:
                pen = (lambda2 * max(0.0, s.cost - budget) / (alpha * budget)) \
                    if (alpha > 0 and budget > 0) else 0.0
                return lambda1 * s.quality / n - pen + lambda3 * s.rating / (5.0 * n)
            front = sorted(front, key=scal, reverse=True)[:beam_size]
        beam = front
    else:
        completed = True

    t_output = time.perf_counter()
    final_states = [s for s in beam if len(s.choice) == n and n > 0]
    frontier = _non_dominated_final(final_states, budget) if final_states else []
    # map choice indices back to the original candidate lists
    for s in frontier:
        s.choice = [kept[i][s.choice[i]] for i in range(len(s.choice))]
    if trace is not None:
        published = [PState(list(s.choice), s.quality, s.cost, s.rating) for s in frontier]
        elapsed = time.perf_counter() - t_red
        trace.extend((elapsed, s) for s in published)

    result = PBSResult(frontier, evals, n_before, n_after,
                       time.perf_counter() - t0, reduction_s, prefix_rule,
                       level_widths=tuple(level_widths))
    result.output_time_s = time.perf_counter() - t_output
    result.search_time_s = t_output - (t_red + reduction_s)
    result.counts_after = [len(c) for c in red_c]
    result.counts_searched = result.counts_after
    result.completed = completed
    result.truncated = truncated
    result.exhausted = completed and not truncated and prefix_rule == "raw_cost"
    result.completion_scope = "all_reduced_candidates" if result.exhausted else "beam_search"
    result.termination_reason = "completed" if completed else "cancelled"
    return result
