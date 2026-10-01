'''
The weighted objective F = l1*Q/n - l2*max(C-B,0)/(alpha*B) + l3*R/(5n), the
result record shared by weighted PBS and PTS, and their common setup
(candidate reduction and the optional MCKP LP bound).
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from gquery.alg.candidates import reduce_candidates
from gquery.alg.common import Item
from gquery.alg.weighted.bounds import BoundOracle, greedy_incumbent, slot_profits


@dataclass
class WeightedSearchResult:
    choice: Optional[List[int]]
    value: float
    search_time_s: float
    # PTS only; always False for PBS
    exhausted: bool
    n_evals: int
    n_pruned_lp: int = 0
    iterations: int = 0


def weighted_value(slot_candidates: Sequence[Sequence[Tuple[Item, float]]],
                   choice: Sequence[int], budget: float, alpha: float,
                   l1: float, l2: float, l3: float) -> float:
    '''all reported values go through this, so a solver can't report a score
    its bundle doesn't have.'''
    n = len(slot_candidates)
    if n == 0 or choice is None or len(choice) != n:
        return float("-inf")
    q = sum(slot_candidates[d][j][1] for d, j in enumerate(choice))
    c = sum(slot_candidates[d][j][0].price for d, j in enumerate(choice))
    r = sum(slot_candidates[d][j][0].rating for d, j in enumerate(choice))
    pen = (l2 * max(0.0, c - budget) / (alpha * budget)) \
        if (alpha > 0 and budget > 0) else 0.0
    return l1 * q / n - pen + l3 * r / (5.0 * n)


def _setup(slot_candidates, budget, alpha, l1, l2, l3, use_lp):
    red_c, kept = reduce_candidates(slot_candidates)
    oracle = None
    if use_lp:
        red_p = slot_profits(red_c, l1, l3)
        weights = [[it.price for it, _ in cands] for cands in red_c]
        oracle = BoundOracle(weights, red_p, budget, alpha, l2)
    return red_c, kept, oracle


def _seed(oracle, red_c, kept, slot_candidates, budget, alpha, l1, l2, l3):
    # greedy incumbent so the bound has something to prune against

    if oracle is None:
        return float("-inf"), None
    inc = greedy_incumbent(oracle)
    if not inc:
        return float("-inf"), None
    orig = [kept[i][inc[1][i]] for i in range(len(inc[1]))]
    return weighted_value(slot_candidates, orig, budget, alpha, l1, l2, l3), orig
