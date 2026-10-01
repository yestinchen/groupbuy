"""Complete-frontier baseline arms shared by the runners, independent of references."""
import time
from gquery.alg.candidates import PState
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.dp import solve_dp


# PBS-unbounded: PBS without beam truncation (candidate reduction, raw-cost prefix
# dominance). Pareto-DP-conv: the conventional requirement-stage DP on the retrieved
# candidates, no reduction. Both are exact.
BASELINE_NAMES = ("PBS-unbounded", "Pareto-DP-conv")


def solve_unbounded_pbs(slots, budget, alpha, timeout_flag=None, trace=None):
    start = time.perf_counter()
    result = solve_pbs(slots, budget, alpha, beam_size=0, prefix_rule="raw_cost",
                       timeout_flag=timeout_flag)
    if trace is not None:
        points = [PState(list(s.choice), s.quality, s.cost, s.rating) for s in result.frontier]
        elapsed = time.perf_counter()-start
        trace.extend((elapsed, p) for p in points)
    return result


def baseline_arms(inst, alpha, expansion_budget=500, candidate_limit=50,
                  lambda1=1., lambda2=1., lambda3=1.):
    slots, budget = inst.slots, inst.query.budget
    return [
        ("PBS-unbounded", lambda f: solve_unbounded_pbs(slots, budget, alpha, timeout_flag=f)),
        ("Pareto-DP-conv", lambda f: solve_dp(slots, budget, alpha, timeout_flag=f)),
    ]
