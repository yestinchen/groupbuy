'''
does scaling (b, M) with query length recover the long-query recall?
the shared defaults (b=100, M=500) give median recall ~0.16/0.01 at n=8 on
amazon (0.05/0.001 on airbnb). sweep fixed vs scaled budgets over the
fixed-group-size workloads gs2..gs8 (--lengths picks the airbnb sets).
the figure wants recall and time vs n for each budget tier - the "scaled"
curve in the paper picks the tier matched to the length.
'''

from __future__ import annotations

from gquery.alg.exp4.protocol import report_row

from gquery.alg.exp4.core import timed_for, instance_status, metric_context
from gquery.alg.exp4.protocol import timing_record, write_manifest
from gquery.alg.exp4.default_reporting import finish_method, save_query, validated_reference
from gquery.alg.exp4.sensitivity_reporting import reference_for

import json
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from gquery.alg.common import save_as_paramed_file
from gquery.alg.exp4.core import (
    exact_frontier, load_instances, nadir, pkey, score, timed, to_points,
)
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.metrics import hypervolume_3d
from gquery.utils import setup_logging

logger = setup_logging("exp4_length_budget")

LENGTHS = [("e11_gs2", 2), ("e11_gs4", 4), ("e11_gs6", 6), ("e11_gs8", 8)]
# (b, M) tiers; kappa stays 50 throughout. lowest tier = the shared default
# operating point (fair-defaults campaign); each step doubles both knobs
TIERS = [(100, 500), (200, 1000), (400, 2000), (800, 4000), (1600, 8000)]


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--exp-name", type=str, default="exp4_pplus_length_budget")
@click.option("--lengths", type=str, default="",
              help="setting:n pairs overriding the amazon LENGTHS, e.g. "
                   "airbnb_gs2:2,airbnb_gs4:4,airbnb_gs6:6,airbnb_gs8:8")
def main(num_queries, alpha, beta, candidate_limit, timeout_seconds,
         exp_name, lengths):
    rows: List[Dict] = []
    out = Path("data_out") / exp_name
    out.mkdir(parents=True, exist_ok=True)
    shared = dict(alpha=alpha, beta=beta, candidate_limit=candidate_limit, timeout_seconds=timeout_seconds)
    pts_label = "PTS"

    def frontier_json(pts):
        return json.dumps([dict(choice=list(p.choice), quality=p.quality, cost=p.cost,
                                rating=p.rating, cost_overage=p.cost_overage) for p in pts])

    ladder = LENGTHS
    if lengths.strip():
        ladder = [(p.split(":")[0].strip(), int(p.split(":")[1]))
                  for p in lengths.split(",") if p.strip()]
    for setting, n in ladder:
        insts, _ = load_instances(setting, num_queries, beta, alpha, include_invalid=True)
        logger.info(f"{setting} (n={n}): {len(insts)} instances")
        for inst in insts:
            b0 = inst.query.budget
            reference = validated_reference(reference_for(inst, alpha, timeout_seconds, exact_frontier), inst, alpha)
            (exact, done, ref_s) = reference
            context = metric_context(inst, alpha, reference[1], reference[0], reference.status)
            context["metric_eligible"] = int(context["query_feasible"] and reference[1]
                                             and bool(exact) and reference.status == "complete")
            if reference.status == "complete" and not context["metric_eligible"] and context["query_feasible"]:
                context["reference_status"] = "reference_failure"
            save_query(out / "inputs" / setting, inst, alpha, reference, {})
            ref_keys = {pkey(p) for p in exact}
            ref_pt = nadir(exact)
            hv_ref = hypervolume_3d(exact, ref_pt) if ref_pt else 0.0
            common = {**context, "setting": setting, "n": n, "query_id": inst.query.id,
                      "budget": b0, "relaxed_cap": b0 * (1.0 + alpha),
                      "minimum_assignment_cost": inst.min_feasible_cost,
                      "n_candidates": inst.n_candidates, "n_slots": inst.n_slots,
                      "ref_size": len(exact), "reference_s": ref_s}

            for beam, m in TIERS:
                name = f"PACS b={beam}"
                r, t, to = timed_for(inst, alpha, lambda f: solve_pbs(
                    inst.slots, b0, alpha, beam_size=beam,
                    prefix_rule="raw_cost", use_reduction=True,
                    timeout_flag=f), timeout_seconds, logger, name)
                pts, fields = finish_method(inst, alpha, name, to_points(r.frontier, b0) if r else [],
                                            failed=r is None, timed_out=to)
                rows.append(report_row(inst, alpha, {
                    **common, **fields, "method": "PACS",
                    "beam": beam, "expansion_budget": np.nan,
                    "method_config": json.dumps(dict(shared, beam=beam), sort_keys=True),
                    "frontier_json": frontier_json(pts), "success": int(len(pts) > 0),
                    **score(pts, ref_keys, ref_pt, hv_ref, metric_eligible=context["metric_eligible"]),
                    "time_s": (r.solve_time_s + r.reduction_time_s) if r else t,
                    "evals": r.n_evals if r else np.nan,
                }))
                name = f"{pts_label} M={m}"
                r, t, to = timed_for(inst, alpha, lambda f: solve_pts(
                    inst.slots, b0, alpha, expansion_budget=m,
                    candidate_limit=candidate_limit, use_reduction=True,
                    timeout_flag=f), timeout_seconds, logger, name)
                pts, fields = finish_method(inst, alpha, name, to_points(r.frontier, b0) if r else [],
                                            failed=r is None, timed_out=to)
                rows.append(report_row(inst, alpha, {
                    **common, **fields, "method": pts_label,
                    "beam": np.nan, "expansion_budget": m,
                    "method_config": json.dumps(dict(shared, expansion_budget=m), sort_keys=True),
                    "frontier_json": frontier_json(pts), "success": int(len(pts) > 0),
                    **score(pts, ref_keys, ref_pt, hv_ref, metric_eligible=context["metric_eligible"]),
                    "time_s": (r.solve_time_s + r.reduction_time_s) if r else t,
                    "evals": r.n_evals if r else np.nan,
                    "exhausted": int(r.exhausted) if r else 0,
                }))

    df = pd.DataFrame(rows)
    df.to_csv(out / "length_budget.csv", index=False)
    tag = ladder[0][0].rsplit("_", 1)[0] if ladder else "e11"   # e11 / airbnb
    save_as_paramed_file(df, exp_name, "length_budget", f"{tag}_gs", num_queries,
                         {"alpha": alpha, "beta": beta})

    for method, col in (("PACS", "beam"), ("PTS", "expansion_budget")):
        s = df[df.method == method]
        if s.empty:
            continue
        print(f"\n=== {method}: recall / median time by length and {col} ===")
        agg = s.groupby(["n", col]).agg(
            recall=("recall", "median"), hv=("hv_ratio", "median"),
            med_time_ms=("time_s", lambda v: v.median() * 1000),
            nq=("query_id", "nunique"))
        with pd.option_context("display.width", 200,
                               "display.float_format", lambda v: f"{v:,.3f}"):
            print(agg.to_string())


if __name__ == "__main__":
    main()
