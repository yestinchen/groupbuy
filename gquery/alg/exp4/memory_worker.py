"""One search method per fresh process. No catalog tensors are loaded."""
import json
import resource
import sys
import time
from pathlib import Path
from gquery.alg.exp4.core import InstanceUnpickler, timed_for, instance_status
from gquery.alg.exp4.protocol import report_row
from gquery.alg.exp4.baselines import BASELINE_NAMES, baseline_arms
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp


def current_rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def main():
    payload, target, method, alpha, expansions, cap, timeout = sys.argv[1:]
    alpha, timeout = float(alpha), float(timeout)
    with open(payload, "rb") as fh:
        instances = InstanceUnpickler(fh).load()
    # eps-ILP is the g=2 grid of the campaign; eps-ILP-g<N> selects another grid (memory grid jobs)
    grid = int(method.rsplit("g", 1)[1]) if method.startswith("eps-ILP-g") else 2
    baseline = current_rss_mb()
    # ru_maxrss is deliberately reported as an absolute high water mark.
    before_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rows = []
    timeouts = returned = 0
    import threading
    stop = threading.Event()
    samples = [baseline]
    def sample():
        while not stop.wait(.001):
            samples.append(current_rss_mb())
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    for inst in instances:
        if method in BASELINE_NAMES:
            fn = dict(baseline_arms(inst, alpha, int(expansions), int(cap)))[method]
        elif method == "PBS":
            fn = lambda f: solve_pbs(inst.slots, inst.query.budget, alpha, beam_size=100, timeout_flag=f)
        elif method == "PTS":
            fn = lambda f: solve_pts(inst.slots, inst.query.budget, alpha,
                expansion_budget=int(expansions), candidate_limit=int(cap), timeout_flag=f)
        elif method == "eps-ILP" or method.startswith("eps-ILP-g"):
            fn = lambda f: solve_epsilon_ilp(inst.query, inst.slots, alpha, grid, f, timeout)
        else:
            raise ValueError(method)
        result, elapsed, hit = timed_for(inst, alpha, fn, timeout, name=method)
        has_output = bool(result[0] if isinstance(result, tuple) else getattr(result, "frontier", []))
        timeouts += int(hit)
        returned += int(has_output)
        rows.append(report_row(inst, alpha, {"query_id": inst.query.id,
                    "method": method, "time_s": elapsed, "success": int(has_output)}))
    samples.append(current_rss_mb())
    stop.set()
    monitor.join()
    rec = {"sampling_interval_s": .001, "rss_samples": len(samples),
           "sampled_search_peak_rss_mb": max(samples),
           "sampled_search_increase_mb": max(0.0, max(samples)-baseline),
           "sampling_limit": "sampling may miss allocations shorter than 1ms; includes interpreter and input baseline",
           "config": {"PBS_unbounded_beam": 0, "Pareto_DP_label_limit": None,
                      "Pareto_DP_conv": "no candidate reduction, raw cost label dominance",
                      "PBS_beam": 100, "eps_grid": grid, "PTS_expansions": int(expansions),
                      "PTS_traversal": "depth_first",
                      "PTS_candidate_limit": int(cap), "alpha": alpha, "timeout_s": timeout},
           "method": method, "pid": __import__("os").getpid(),
           "candidate_resident_baseline_mb": baseline, "peak_before_search_mb": before_peak,
           "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "rss_after_search_mb": current_rss_mb(), "shared_catalog_loaded": False,
           "method_wall_s": time.perf_counter()-start, "requested": len(instances),
           "timeouts": timeouts, "returned": returned, "rows": rows,
           "memory_semantics": "isolated method process peak includes imports and stripped inputs; not a subtraction of monotonic peaks"}
    Path(target).write_text(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
