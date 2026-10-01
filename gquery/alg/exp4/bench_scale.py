'''catalog-size and query-length sweeps, plus the retrieval vs search latency
breakdown -> scale.csv. the 2018-replication and ANN panels are not produced
here (dataset / index not present), the gap is recorded in the output.'''

from __future__ import annotations

from gquery.alg.exp4.protocol import report_row

from gquery.alg.exp4.core import timed_for, instance_status, metric_context
from gquery.alg.exp4.protocol import timing_record, write_manifest
from gquery.alg.exp4.default_reporting import finish_method, save_query, validated_reference
from gquery.alg.exp4.sensitivity_reporting import reference_for

import json
import time
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
from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp
from gquery.alg.pareto.metrics import hypervolume_3d
from gquery.utils import setup_logging

logger = setup_logging("exp4_scale")

CATALOGS = [("e11_cat10k", 10), ("e11_cat15k", 15), ("e11_cat20k", 20),
            ("e11_cat25k", 25), ("e11_cat30k", 30)]
LENGTHS = [("e11_gs2", 2), ("e11_gs4", 4), ("e11_gs6", 6), ("e11_gs8", 8)]


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--grid", type=int, default=8)
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--exp-name", type=str, default="exp4_pplus_scale")
@click.option("--catalogs", type=str, default=None,
              help="comma list of setting:param; empty string skips the sweep")
@click.option("--lengths", type=str, default=None,
              help="comma list of setting:param; empty string skips the sweep")
def main(num_queries, alpha, beta, beam, expansion_budget, candidate_limit,
         grid, timeout_seconds, exp_name, catalogs, lengths):
    def _parse(arg, default):
        if arg is None:
            return default
        return [(s.split(":")[0], float(s.split(":")[1]))
                for s in arg.split(",") if s.strip()]

    catalog_list = _parse(catalogs, CATALOGS)
    length_list = _parse(lengths, LENGTHS)
    rows: List[Dict] = []
    out_dir = Path("data_out") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shared = dict(alpha=alpha, beta=beta, timeout_seconds=timeout_seconds)
    pts_label = "PTS"

    def run_sweep(sweep: str, setting: str, param: float):
        try:
            insts, _ = load_instances(setting, num_queries, beta, alpha, include_invalid=True)
        except Exception as exc:
            logger.error(f"{setting}: {exc!r}")
            return
        logger.info(f"{sweep} {setting} (param={param}): {len(insts)} instances")
        for inst in insts:
            b = inst.query.budget
            # keep the reference's own wall clock, shows if bounding buys anything
            reference = validated_reference(reference_for(inst, alpha, timeout_seconds, exact_frontier), inst, alpha)
            (exact, done, ref_s) = reference
            context = metric_context(inst, alpha, reference[1], reference[0], reference.status)
            context["metric_eligible"] = int(context["query_feasible"] and reference[1]
                                             and bool(exact) and reference.status == "complete")
            if reference.status == "complete" and not context["metric_eligible"] and context["query_feasible"]:
                context["reference_status"] = "reference_failure"
            save_query(out_dir / "inputs" / f"{sweep}-{setting}", inst, alpha, reference, {})
            ref_keys = {pkey(p) for p in exact}
            ref_pt = nadir(exact)
            hv_ref = hypervolume_3d(exact, ref_pt) if ref_pt else 0.0

            def emit(method, pts, secs, evals, solves=1, failed=False, timed_out=False,
                     stats=None, config=None, **kw):
                pts, fields = finish_method(inst, alpha, method, pts, failed=failed, timed_out=timed_out,
                                            stats=stats, grid=grid if stats is not None else None)
                rows.append(report_row(inst, alpha, {
                    **context, **fields,
                    "sweep": sweep, "setting": setting, "param": param,
                    "query_id": inst.query.id, "method": method,
                    "method_config": json.dumps(config or {}, sort_keys=True),
                    "frontier_json": json.dumps([dict(choice=list(p.choice), quality=p.quality, cost=p.cost,
                                                      rating=p.rating, cost_overage=p.cost_overage) for p in pts]),
                    "success": int(len(pts) > 0),
                    **score(pts, ref_keys, ref_pt, hv_ref, metric_eligible=context["metric_eligible"]),
                    "search_s": secs, "retrieval_s": inst.retrieval_s if getattr(inst, "retrieval_measured", False) else np.nan,
                    "retrieval_historical_s": inst.retrieval_s if not getattr(inst, "retrieval_measured", False) else np.nan,
                    "end_to_end_s": (secs + inst.retrieval_s) if getattr(inst, "retrieval_measured", False) else np.nan,
                    "evals": evals, "solves": solves,
                    "n_candidates": inst.n_candidates, "n_slots": inst.n_slots,
                    "budget": b, "relaxed_cap": b * (1.0 + alpha),
                    "minimum_assignment_cost": inst.min_feasible_cost,
                    "ref_size": len(exact), "ref_time_s": ref_s, "reference_s": ref_s, **kw,
                }))

            r, t, to = timed_for(inst, alpha, lambda f: solve_pbs(
                inst.slots, b, alpha, beam_size=beam, prefix_rule="raw_cost",
                use_reduction=True, timeout_flag=f),
                timeout_seconds, logger, "PACS")
            emit("PACS", to_points(r.frontier, b) if r else [],
                 (r.solve_time_s + r.reduction_time_s) if r else t,
                 r.n_evals if r else np.nan, failed=r is None, timed_out=to,
                 config=dict(shared, beam=beam))

            r, t, to = timed_for(inst, alpha, lambda f: solve_pts(
                inst.slots, b, alpha, expansion_budget=expansion_budget,
                candidate_limit=candidate_limit, use_reduction=True,
                timeout_flag=f), timeout_seconds, logger, pts_label)
            emit(pts_label, to_points(r.frontier, b) if r else [],
                 (r.solve_time_s + r.reduction_time_s) if r else t,
                 r.n_evals if r else np.nan,
                 exhausted=int(r.exhausted) if r else 0, failed=r is None, timed_out=to,
                 config=dict(shared, expansion_budget=expansion_budget, candidate_limit=candidate_limit))

            if grid <= 0:
                continue  # grid 0 skips the epsilon-ILP arm (policy comparison runs)
            out, t, to = timed_for(inst, alpha,
                lambda f: solve_epsilon_ilp(inst.query, inst.slots, alpha, grid,
                                            f, timeout_seconds),
                timeout_seconds, logger, f"eps-ILP g={grid}")
            if out is None:
                emit(f"eps-ILP g={grid}", [], t, np.nan, 0, failed=True, timed_out=to,
                     config=dict(shared, grid=grid, per_solve_budget_s=timeout_seconds))
            else:
                front, stats = out
                emit(f"eps-ILP g={grid}", front, t,
                     stats["n_solves"] * inst.n_candidates, stats["n_solves"], timed_out=to,
                     stats=stats, config=dict(shared, grid=grid, per_solve_budget_s=timeout_seconds))

    for setting, size in catalog_list:
        run_sweep("catalog", setting, size)
    for setting, n in length_list:
        run_sweep("length", setting, n)

    df = pd.DataFrame(rows)
    # record what could not be run, so the omission is in the data too
    df.attrs["not_run"] = "amazon_reviews_2018 absent; no ANN index in repo"
    out = out_dir
    df.to_csv(out / "scale.csv", index=False)
    pd.DataFrame([
        {"panel": "reviews2018_replication", "status": "not run",
         "reason": "dataset absent from this machine"},
        {"panel": "ann_retrieval", "status": "not run",
         "reason": "no ANN index implemented; retrieval is exact GPU brute force"},
    ]).to_csv(out / "omitted_panels.csv", index=False)
    save_as_paramed_file(df, exp_name, "scale", "e11", num_queries,
                         {"alpha": alpha, "beta": beta, "beam": beam})

    for sweep in ("catalog", "length"):
        s = df[df.sweep == sweep]
        if s.empty:
            continue
        print(f"\n=== {sweep} sweep ===")
        agg = s.groupby(["param", "method"]).agg(
            requested=("query_id", "size"), feasible=("query_feasible", "sum"),
            metric_n=("metric_eligible", "sum"), success=("success", "mean"), recall=("recall", "mean"),
            hv=("hv_ratio", "mean"), cands=("n_candidates", "mean"),
            search_ms=("search_s", lambda v: v.median() * 1000),
            retr_ms=("retrieval_s", lambda v: v.median() * 1000),
        )
        with pd.option_context("display.width", 200,
                               "display.float_format", lambda v: f"{v:,.3f}"):
            print(agg.to_string())

    print("\n=== latency decomposition (median ms, PACS) ===")
    p = df[df.method == "PACS"]
    if not p.empty:
        dec = p.groupby(["sweep", "param"]).agg(
            retrieval_ms=("retrieval_s", lambda v: v.median() * 1000),
            search_ms=("search_s", lambda v: v.median() * 1000),
            end_to_end_ms=("end_to_end_s", lambda v: v.median() * 1000),
        )
        dec["search_pct"] = dec.search_ms / dec.end_to_end_ms * 100
        with pd.option_context("display.width", 200,
                               "display.float_format", lambda v: f"{v:,.3f}"):
            print(dec.to_string())
    print("\nNot run: Reviews-2018 replication (dataset absent), "
          "ANN retrieval (no ANN index in this repository).")


if __name__ == "__main__":
    main()
