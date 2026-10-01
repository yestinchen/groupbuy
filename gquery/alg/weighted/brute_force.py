'''weighted brute force. try every assignment, keep the best F.'''

from __future__ import annotations

import itertools

from gquery.alg.weighted.objective import weighted_value


def solve_brute_force(slots, budget, alpha, l1, l2, l3):
    max_b = budget * (1.0 + alpha)
    best_v, best_c = float("-inf"), None
    for combo in itertools.product(*[range(len(c)) for c in slots]):
        cost = sum(slots[d][j][0].price for d, j in enumerate(combo))
        if cost > max_b + 1e-9:
            continue
        v = weighted_value(slots, list(combo), budget, alpha, l1, l2, l3)
        if v > best_v:
            best_v, best_c = v, list(combo)
    return best_v, best_c
