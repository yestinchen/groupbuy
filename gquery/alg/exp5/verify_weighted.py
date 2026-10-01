'''
sanity check before the full run: enumeration, the exact ILP, and weighted
PBS/PTS must agree on random instances. a value above the optimum = bug.
run: python -m gquery.alg.exp5.verify_weighted
'''

from __future__ import annotations

import random


from gquery.alg.common import GroupQuery, Item
from gquery.alg.weighted.objective import weighted_value
from gquery.alg.weighted.pbs import solve_pbs as solve_weighted_pbs
from gquery.alg.weighted.pts import solve_pts as solve_weighted_pts
from gquery.alg.weighted.brute_force import solve_brute_force as solve_weighted_bf
from gquery.alg.weighted.ilp import solve_ilp as solve_weighted_ilp

TOL = 1e-7
WEIGHTS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def _rand_slots(rng: random.Random, n: int, m: int):
    out = []
    for d in range(n):
        cands = []
        for k in range(m):
            it = Item(f"i{d}_{k}", [], round(rng.uniform(0, 40), 2),
                      rng.choice([1.0, 2.0, 3.0, 4.0, 5.0]), None, f"i{d}_{k}")
            cands.append((it, round(rng.uniform(0.5, 1.0), 3)))
        out.append(cands)
    return out


def main() -> None:
    rng = random.Random(20260809)
    checked = failures = 0
    for trial in range(25):
        n, m = rng.randint(2, 4), rng.randint(2, 4)
        slots = _rand_slots(rng, n, m)
        floor = sum(min(it.price for it, _ in s) for s in slots)
        for alpha in (0.0, 0.5):
            budget = floor * rng.uniform(1.0, 1.8) if alpha else floor * 1.5
            q = GroupQuery(f"syn{trial}", None, budget, None, None)
            for (l1, l2, l3) in WEIGHTS:
                bf_v, _ = solve_weighted_bf(slots, budget, alpha, l1, l2, l3)
                if bf_v == float("-inf"):
                    continue
                checked += 1

                out = solve_weighted_ilp(q, slots, alpha, float(l1), float(l2), float(l3))
                if out is not None and out[0] is not None:
                    ilp_v = weighted_value(slots, out[0], budget, alpha, l1, l2, l3)
                    if abs(ilp_v - bf_v) > TOL * max(1.0, abs(bf_v)):
                        print(f"  [FAIL] ILP vs enumeration  w={(l1,l2,l3)} "
                              f"alpha={alpha}: {ilp_v:.9f} vs {bf_v:.9f}")
                        failures += 1

                for name, fn in (("PBS", solve_weighted_pbs),
                                 ("PTS", solve_weighted_pts)):
                    r = fn(slots, budget, alpha, float(l1), float(l2), float(l3))
                    if r.choice is None:
                        print(f"  [FAIL] {name} returned nothing where a bundle exists"
                              f"  w={(l1,l2,l3)} alpha={alpha}")
                        failures += 1
                        continue
                    cost = sum(slots[d][j][0].price for d, j in enumerate(r.choice))
                    if cost > budget * (1 + alpha) + 1e-9:
                        print(f"  [FAIL] {name} infeasible bundle, cost {cost:.2f}")
                        failures += 1
                    fresh = weighted_value(slots, r.choice, budget, alpha, l1, l2, l3)
                    if abs(fresh - r.value) > 1e-9:
                        print(f"  [FAIL] {name} reported {r.value:.9f} but its bundle "
                              f"scores {fresh:.9f}")
                        failures += 1
                    if fresh > bf_v + TOL * max(1.0, abs(bf_v)):
                        print(f"  [FAIL] {name} exceeds the optimum: {fresh:.9f} "
                              f"> {bf_v:.9f}  w={(l1,l2,l3)}")
                        failures += 1

    print(f"\nchecked {checked} (instance, weight) pairs across alpha in (0, 0.5)")
    print("PASS: enumeration, ILP and both adapters agree" if not failures
          else f"FAIL: {failures} violations")


if __name__ == "__main__":
    main()
