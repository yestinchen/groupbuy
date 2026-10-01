'''
weighted PBS/PTS vs the exact ILP, 7 weight vectors per query.
retrieval is shared across solvers. every value is recomputed by
weighted_value from the returned bundle, so no solver can misreport.
'''

from __future__ import annotations

import gzip
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from gquery.alg.exp4.core import load_instances, instance_status, solver_status, timed
from gquery.alg.exp4.protocol import timing_record
from gquery.alg.weighted.objective import weighted_value
from gquery.alg.weighted.pbs import solve_pbs as solve_weighted_pbs
from gquery.alg.weighted.pts import solve_pts as solve_weighted_pts
from gquery.alg.weighted.ilp import solve_ilp as solve_weighted_ilp
from gquery.utils import setup_logging

logger = setup_logging("exp5_weighted_vs_ilp")

WEIGHTS = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
           (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

OPT_TOL = 1e-7


def _wname(w) -> str:
    s = sum(w) or 1
    return "/".join(f"{x/s:.2f}" for x in w)


def evaluate_instance(inst, w, alpha, beam, expansion_budget, candidate_limit,
                      timeout_seconds):
    """Emit every method even when the input or scalar reference fails."""
    b = inst.query.budget
    l1, l2, l3 = map(float, w)
    qstatus = instance_status(inst, alpha)
    args = (inst.slots, b, alpha, l1, l2, l3)
    arms = (("PBS", solve_weighted_pbs, dict(beam_size=beam, use_lp=True)),
            ("PTS", solve_weighted_pts, dict(expansion_budget=expansion_budget,
                candidate_limit=candidate_limit, use_lp=True)),
            ("PBS-noLP", solve_weighted_pbs, dict(beam_size=beam, use_lp=False)),
            ("PTS-noLP", solve_weighted_pts, dict(expansion_budget=expansion_budget,
                candidate_limit=candidate_limit, use_lp=False)))
    out, t_ilp, timeout = (None, np.nan, False)
    if qstatus == "feasible":
        out, t_ilp, timeout = timed(lambda flag: solve_weighted_ilp(
            inst.query, inst.slots, alpha, l1, l2, l3, timeout_flag=flag),
            timeout_seconds, logger, "ILP")
    has_opt = out is not None and out[0] is not None
    proved = has_opt and bool(out[1].get("proved_optimal")) and not timeout
    f_opt = weighted_value(*args[:1], out[0], *args[1:]) if has_opt else None
    reference_status = ("not_applicable" if qstatus != "feasible" else
                        "reference_timeout" if timeout else
                        "complete" if proved else "reference_failure")
    rows = []

    def emit(name, val, seconds, to=False, failed=False, **kw):
        gap = f_opt - val if proved and val is not None else None
        rows.append({"query_id": inst.query.id, "weights": _wname(w),
                     "method": name, "query_status": qstatus,
                     "query_feasible": int(qstatus == "feasible"),
                     "reference_status": reference_status,
                     "metric_eligible": int(proved),
                     "solver_status": solver_status(qstatus, val is not None, to, failed),
                     "feasible": int(val is not None), "value": val,
                     "search_s": seconds,
                     "optimal": (int(gap <= OPT_TOL * max(1.0, abs(f_opt)))
                                 if gap is not None else 0 if proved else np.nan),
                     "rel_gap": gap / max(abs(f_opt), 1e-9) if gap is not None else np.nan,
                     "timeout": int(to), **timing_record(inst, "weighted-ILP" if name == "ILP" else "weighted-" + name, seconds), **kw})

    emit("ILP", f_opt, t_ilp, timeout, out is None,
         ilp_status="proved_optimal" if proved else reference_status,
         ilp_nodes=out[1].get("nodes", np.nan) if out else np.nan,
         choice_json=json.dumps(list(out[0])) if has_opt else None)
    for name, fn, kw in arms:
        if qstatus != "feasible":
            emit(name, None, np.nan)
            continue
        r, secs, to = timed(lambda flag: fn(*args, **kw, timeout_flag=flag),
                            timeout_seconds, logger, name)
        val = weighted_value(inst.slots, r.choice, b, alpha, l1, l2, l3) \
              if r is not None and r.choice is not None else None
        emit(name, val, secs, to, r is None,
             n_evals=r.n_evals if r else np.nan,
             n_pruned_lp=r.n_pruned_lp if r else np.nan,
             exhausted=int(r.exhausted) if r else 0,
             choice_json=json.dumps(list(r.choice)) if r is not None and r.choice is not None else None)
        if proved and val is not None and val > f_opt + OPT_TOL * max(1.0, abs(f_opt)):
            raise ValueError(f"{name} exceeds scalar reference on {inst.query.id}")
    return rows


def aggregate_status(g):
    if "method_wall_s" in g:
        g = g.assign(search_s=g.method_wall_s)
    comparable = g[g.metric_eligible == 1]
    returned = comparable[comparable.feasible == 1]
    # Feasible failures remain in optimal_fraction's denominator. Undefined
    # gaps have their own explicit denominator, never an implicit dropna count.
    return pd.Series({"n": len(g), "n_query_feasible": g.query_feasible.sum(),
        "n_feasible": g.feasible.sum(), "n_metric": len(comparable),
        "n_gap": len(returned), "n_no_assignment": (g.solver_status == "no_assignment").sum(),
        "n_solver_failure": (g.solver_status == "solver_failure").sum(),
        "n_reference_timeout": (g.reference_status == "reference_timeout").sum(),
        "optimal_fraction": comparable.optimal.mean(),
        "median_gap": returned.rel_gap.median(), "worst_gap": returned.rel_gap.max(),
        "median_ms": g.end_to_end_s.median() * 1000,
        "p90_ms": g.end_to_end_s.quantile(0.9) * 1000})


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--query-setting", type=str, default="e10_1k_100q")
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--beam", type=int, default=50)
@click.option("--expansion-budget", type=int, default=2000)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--exp-name", type=str, default="exp5_weighted_search_vs_ilp")
def main(num_queries, query_setting, alpha, beta, beam, expansion_budget,
         candidate_limit, timeout_seconds, exp_name):
    insts, _ = load_instances(query_setting, num_queries, beta, alpha, include_invalid=True)
    logger.info(f"{len(insts)} instances x {len(WEIGHTS)} weight vectors")
    out_dir = Path("data_out") / exp_name
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    # Candidate objective arrays let every reported value and returned bundle be
    # recomputed offline, without any solver.
    with gzip.open(out_dir / "inputs" / "candidates.jsonl.gz", "wt") as f:
        for inst in insts:
            f.write(json.dumps(dict(
                query_id=inst.query.id, budget=inst.query.budget, alpha=alpha,
                query_status=instance_status(inst, alpha), n_slots=inst.n_slots,
                candidates=[[[float(q), float(it.price), float(it.rating)] for it, q in slot] for slot in inst.slots],
                value_definition="l1*sum(q)/n - l2*max(0,cost-budget)/(alpha*budget) + l3*sum(rating)/(5n)"),
                allow_nan=False) + "\n")

    rows: List[Dict] = []
    for inst in insts:
        for w in WEIGHTS:
            rows.extend(evaluate_instance(inst, w, alpha, beam, expansion_budget,
                                          candidate_limit, timeout_seconds))
        logger.info(f"query {inst.query.id} done")

    df = pd.DataFrame(rows)
    df["end_to_end_s"] = df.total_s
    df.to_csv(out_dir / "all_results.csv", index=False)

    overall = df.groupby("method").apply(aggregate_status, include_groups=False)
    by_weight = df.groupby(["weights", "method"]).apply(aggregate_status, include_groups=False)
    overall.to_csv(out_dir / "summary_overall.csv")
    by_weight.to_csv(out_dir / "summary_by_weight.csv")

    meta = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "working_tree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True).stdout.strip()),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query_setting": query_setting, "num_queries": num_queries,
        "alpha": alpha, "beta": beta, "beam": beam,
        "expansion_budget": expansion_budget, "candidate_limit": candidate_limit,
        "timeout_seconds": timeout_seconds,
        "weight_vectors": [list(w) for w in WEIGHTS],
        "exact_reference": "gquery.alg.weighted.ilp.solve_ilp",
        "rows": len(df),
        "expected_rows_per_method": len(insts) * len(WEIGHTS),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n=== {len(insts)} queries x {len(WEIGHTS)} weight vectors ===")
    with pd.option_context("display.width", 200,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(overall.to_string())
    _emit_table(overall, out_dir)
    print(f"\nwrote {out_dir}/")


def _emit_table(overall: pd.DataFrame, out_dir: Path) -> None:
    disp = {"ILP": r"\texttt{ILP} (exact)", "PBS": r"\texttt{PBS}",
            "PTS": r"\texttt{PTS}"}
    # gap columns are all zero on this workload, keep them in the CSV only
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"method & optimal & med.\ time & p90 \\",
             r"\midrule"]
    for m in ("ILP", "PBS", "PTS"):
        if m not in overall.index:
            continue
        r = overall.loc[m]
        lines.append(f"{disp[m]} & ${r.optimal_fraction:.3f}$ & "
                     f"${r.median_ms:.2f}$\\,ms & ${r.p90_ms:.2f}$\\,ms \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_weighted_search_vs_ilp.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
