'''one-factor ablations -> ablation.csv (fig ablation). run at beta=0.8 and
0.6 - at 0.8 the reduced sets are small enough to exhaust, the pruning rules
only show up at the lower threshold.

PBS arms: full, and -rawcost (cost overhead instead of raw cost in the partial
assignment dominance). PTS arms: full, -bound (no objective upper bound),
-prefixdom (no partial assignment dominance), -bound-prefixdom (neither).'''

from __future__ import annotations

from gquery.alg.exp4.protocol import report_row

from gquery.alg.exp4.core import timed_for, metric_context
from gquery.alg.exp4.default_reporting import finish_method, save_query, validated_reference
from gquery.alg.exp4.sensitivity_reporting import reference_for

import json
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd

from gquery.alg.common import save_as_paramed_file
from gquery.alg.exp4.core import (
    exact_frontier, load_instances, nadir, pkey, score, to_points,
)
from gquery.alg.exp4.variants import solve_pacs_variant
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.metrics import hypervolume_3d
from gquery.utils import setup_logging

logger = setup_logging("exp4_ablation")


def _pacs_arms(beam: int) -> List[Tuple[str, str, Dict]]:
    return [
        ("PACS", "full", dict(beam_size=beam, prefix_rule="raw_cost", use_reduction=True)),
        ("PACS", "-rawcost", dict(beam_size=beam, prefix_rule="overage", use_reduction=True)),
    ]


def _pts_arms(m: int, kappa: int) -> List[Tuple[str, str, Dict]]:
    base = dict(expansion_budget=m, candidate_limit=kappa)
    return [
        ("PTS", "full", dict(base)),
        ("PTS", "-bound", dict(base, use_bound=False)),
        ("PTS", "-prefixdom", dict(base, prefix_dominance=False)),
        ("PTS", "-bound-prefixdom", dict(base, use_bound=False, prefix_dominance=False)),
    ]


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--query-setting", type=str, default="e10_1k_100q")
@click.option("--alpha", type=float, default=0.5)
@click.option("--betas", type=str, default="0.8,0.6")
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--exp-name", type=str, default="exp4_pplus_ablation")
def main(num_queries, query_setting, alpha, betas, beam, expansion_budget,
         candidate_limit, timeout_seconds, exp_name):
    beta_list = [float(b) for b in betas.split(",") if b.strip()]
    rows: List[Dict] = []
    out = Path("data_out") / exp_name
    out.mkdir(parents=True, exist_ok=True)

    for beta in beta_list:
        insts, _ = load_instances(query_setting, num_queries, beta, alpha, include_invalid=True)
        logger.info(f"beta={beta}: {len(insts)} instances")
        for inst in insts:
            # Validated complete reference; candidate arrays and reference choices are
            # archived per beta so recall, HV, and assignment validity reconstruct offline.
            reference = validated_reference(reference_for(inst, alpha, timeout_seconds, exact_frontier), inst, alpha)
            (exact, done, _) = reference
            context = metric_context(inst, alpha, reference[1], reference[0], reference.status)
            context["metric_eligible"] = int(context["query_feasible"] and reference[1]
                                             and bool(exact) and reference.status == "complete")
            if reference.status == "complete" and not context["metric_eligible"] and context["query_feasible"]:
                context["reference_status"] = "reference_failure"
            save_query(out / "inputs" / f"beta-{beta:g}", inst, alpha, reference, {})
            ref_keys = {pkey(p) for p in exact}
            ref_pt = nadir(exact)
            hv_ref = hypervolume_3d(exact, ref_pt) if ref_pt else 0.0

            arms = _pacs_arms(beam) + _pts_arms(expansion_budget, candidate_limit)
            for family, arm, kw in arms:
                solver = solve_pacs_variant if family == "PACS" else solve_pts
                r, t, to = timed_for(inst, alpha,
                    lambda f, s=solver, k=kw: s(inst.slots, inst.query.budget,
                                                alpha, timeout_flag=f, **k),
                    timeout_seconds, logger, f"{family}/{arm}")
                pts = to_points(r.frontier, inst.query.budget) if r else []
                pts, fields = finish_method(inst, alpha, f"{family}/{arm}", pts,
                                            failed=r is None, timed_out=to)
                rows.append(report_row(inst, alpha, {
                    **context, **fields,
                    "beta": beta, "family": family, "arm": arm,
                    "method_config": json.dumps(kw, sort_keys=True),
                    "exhausted": (int(r.exhausted) if r is not None and hasattr(r, "exhausted") else np.nan),
                    "frontier_json": json.dumps([dict(choice=list(p.choice), quality=p.quality, cost=p.cost,
                                                      rating=p.rating, cost_overage=p.cost_overage) for p in pts]),
                    "query_id": inst.query.id, "success": int(len(pts) > 0),
                    **score(pts, ref_keys, ref_pt, hv_ref, metric_eligible=context["metric_eligible"]),
                    "time_s": ((r.solve_time_s + r.reduction_time_s) if r else t),
                    "retrieval_s": inst.retrieval_s if getattr(inst, "retrieval_measured", False) else np.nan,
                    "retrieval_historical_s": inst.retrieval_s if not getattr(inst, "retrieval_measured", False) else np.nan,
                    "end_to_end_s": (((r.solve_time_s + r.reduction_time_s) if r else t) + inst.retrieval_s) if getattr(inst, "retrieval_measured", False) else np.nan,
                    "evals": r.n_evals if r else np.nan,
                    "ref_size": len(ref_keys), "reference_s": reference[2],
                    "n_candidates": inst.n_candidates, "n_slots": inst.n_slots,
                    "budget": inst.query.budget, "relaxed_cap": inst.query.budget * (1 + alpha),
                    "minimum_assignment_cost": inst.min_feasible_cost,
                }))

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)
    save_as_paramed_file(df, exp_name, "ablation", query_setting, num_queries,
                         {"alpha": alpha, "beta": beta_list[0], "beam": beam})

    for beta in beta_list:
        s = df[df.beta == beta]
        if s.empty:
            continue
        print(f"\n=== ablation, beta={beta} "
              f"({s.query_id.nunique()} queries, exact reference) ===")
        agg = s.groupby(["family", "arm"]).agg(
            requested=("query_id", "size"), feasible=("query_feasible", "sum"),
            metric_n=("metric_eligible", "sum"), success=("success", "mean"), size=("size", "mean"),
            recall=("recall", "mean"), hv=("hv_ratio", "mean"),
            evals=("evals", "median"), med_time=("time_s", "median"),
        )
        with pd.option_context("display.width", 200,
                               "display.float_format", lambda v: f"{v:,.4f}"):
            print(agg.to_string())


if __name__ == "__main__":
    main()
