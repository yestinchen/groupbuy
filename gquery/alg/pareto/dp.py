"""Conventional requirement-stage Pareto DP on the retrieved candidates.

Stage labels are compared on the objective values and the consumed resource
(quality, raw cost, rating), the standard dominance for a resource-constrained
problem, which is exact. No candidate reduction is applied. Labels are inserted
incrementally, exact triples are merged, and paths are reconstructed from
parent pointers only for the final output. With raw-cost labels this is the
same mathematical recurrence as PBS without beam truncation, minus the
candidate reduction.
"""
from dataclasses import dataclass
import time

from gquery.alg.candidates import PState, _non_dominated_final


@dataclass(slots=True)
class _Label:
    quality: float
    cost: float
    rating: float
    parent: object = None
    index: int = -1


@dataclass
class DPResult:
    frontier: list
    n_evals: int
    n_before: int
    n_after: int
    reduction_time_s: float
    search_time_s: float
    output_time_s: float
    solve_time_s: float
    completed: bool
    exhausted: bool
    termination_reason: str
    counts_after: list
    counts_searched: list
    level_widths: tuple
    n_label_comparisons: int
    completion_scope: str = "all_retrieved_candidates"


def solve_dp(slot_candidates, budget, alpha, timeout_flag=None, trace=None):
    start = time.perf_counter()
    counts = [len(c) for c in slot_candidates]
    n = len(counts)
    candidates = [list(c) for c in slot_candidates]
    reduced = time.perf_counter()
    labels = [_Label(0., 0., 0.)] if n and all(counts) else []
    evals = comparisons = 0
    widths = []
    completed = not labels
    cap = budget * (1 + alpha)
    stop = lambda: timeout_flag is not None and timeout_flag.is_set()
    final = []
    for depth, group in enumerate(candidates):
        if not labels:
            completed = True
            break
        if stop():
            break
        next_labels = []
        cancelled = False
        for parent in labels:
            for j, (item, quality) in enumerate(group):
                if stop():
                    cancelled = True
                    break
                evals += 1
                cost = parent.cost + item.price
                if cost > cap + 1e-9:
                    continue
                new = _Label(parent.quality + quality, cost,
                             parent.rating + item.rating, parent, j)
                dominated = False
                remove = []
                for k, old in enumerate(next_labels):
                    if (k & 255) == 0 and stop():
                        cancelled = True
                        break
                    comparisons += 1
                    # Weak dominance on (quality, raw cost, rating) merges identical labels.
                    if old.cost <= new.cost and old.quality >= new.quality and old.rating >= new.rating:
                        dominated = True
                        break
                    if new.cost <= old.cost and new.quality >= old.quality and new.rating >= old.rating:
                        remove.append(k)
                if cancelled:
                    break
                if not dominated:
                    for k in reversed(remove):
                        del next_labels[k]
                    next_labels.append(new)
            if cancelled:
                break
        if cancelled:
            break
        labels = next_labels
        widths.append(len(labels))
        if depth == n - 1:
            final = labels
            completed = True
    output_start = time.perf_counter()
    states = []
    for label in final:
        indices = []
        cur = label
        while cur.parent is not None:
            indices.append(cur.index)
            cur = cur.parent
        states.append(PState(list(reversed(indices)), label.quality, label.cost, label.rating))
    frontier = _non_dominated_final(states, budget)
    # Full frontier becomes available only after output filtering/reconstruction.
    if trace is not None:
        published = [PState(list(s.choice), s.quality, s.cost, s.rating) for s in frontier]
        elapsed = time.perf_counter() - start
        trace.extend((elapsed, s) for s in published)
    end = time.perf_counter()
    return DPResult(frontier, evals, sum(counts), sum(counts), reduced-start,
                    output_start-reduced, end-output_start, end-reduced,
                    completed, completed, "completed" if completed else "cancelled",
                    list(counts), list(counts), tuple(widths), comparisons)
