'''default-setting comparison: PBS / PTS / eps-ILP / PBF -> default.csv.
recall is scored against the exact frontier. eval counts only comparable
within the same method family.'''

from __future__ import annotations

from gquery.alg.exp4.protocol import report_row
from gquery.alg.exp4.default_reporting import validated_reference, pbf_call, pbf_memory_fields, finish_method, save_query

import time
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from gquery.alg.common import save_as_paramed_file
from gquery.alg.pareto.brute_force import solve_brute_force
from gquery.alg.exp4.core import (
    exact_frontier, load_instances, nadir, pkey, score,
    timed, timed_for, to_points, instance_status, metric_context, solver_status,
)
from gquery.alg.exp4.baselines import BASELINE_NAMES, baseline_arms
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp
from gquery.alg.pareto.metrics import _filter_non_dominated, hypervolume_3d
from gquery.utils import setup_logging

logger = setup_logging("exp4_default")


def _weakly_dominated_by(p, front) -> bool:
    return any(t.quality >= p.quality - 1e-9
               and t.cost_overage <= p.cost_overage + 1e-9
               and t.rating >= p.rating - 1e-9
               for t in front)


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--query-setting", type=str, default="e10_1k_100q")
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--lambda1", type=float, default=1.0)
@click.option("--lambda2", type=float, default=1.0)
@click.option("--lambda3", type=float, default=1.0)
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--grids", type=str, default="2,4,8,16")
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--pbf-timeout", type=float, default=60.0)
@click.option("--pbf-max-combos", type=int, default=200_000_000,
              help="Combination guard for PBF; 0 disables it")
@click.option("--pbf-memory-gib", type=float, default=None,
              help="PBF stops with its partial frontier once its RSS growth exceeds this many GiB")
@click.option("--skip-pbf", is_flag=True, default=False)
@click.option("--methods", type=str, default=None,
              help="Comma separated subset of the method names to run (e.g. Pareto-DP-conv); default runs every method")
@click.option("--exp-name", type=str, default="exp4_pplus_default")
def main(num_queries, query_setting, alpha, beta, lambda1, lambda2, lambda3,
         beam, expansion_budget, candidate_limit, grids, timeout_seconds,
         pbf_timeout, pbf_max_combos, pbf_memory_gib, skip_pbf, methods, exp_name):
    pbf_memory = int(pbf_memory_gib * 2**30) if pbf_memory_gib else None
    grid_list = [int(g) for g in grids.split(",") if g.strip()]
    wanted = {m.strip() for m in methods.split(",") if m.strip()} if methods else None
    want = lambda name: wanted is None or name in wanted
    insts, _ = load_instances(query_setting, num_queries, beta, alpha, include_invalid=True)
    logger.info(f"{len(insts)} instances | grids={grid_list} | beam={beam}")

    rows: List[Dict] = []
    for inst in insts:
        q, budget = inst.query, inst.query.budget
        fr: Dict[str, List] = {}
        meta: Dict[str, Dict] = {}

        def record(name, points, seconds, evals, solves, **kw):
            key = "PACS+" if name == "PACS" else name
            points, fields = finish_method(inst, alpha, key, points,
                failed=kw.get("failed", False), timed_out=kw.get("timed_out", False),
                stats=kw.pop("stats", None), grid=kw.pop("grid", None))
            kw.update(fields)
            fr[name] = points
            meta[name] = {"time_s": seconds, "evals": evals, "solves": solves, **kw}

        query_status = instance_status(inst, alpha)
        if query_status == "feasible":
            # --- PACS+ ---
            if want("PACS"):
              r, t, to = timed_for(inst, alpha, lambda f: solve_pbs(
                inst.slots, budget, alpha, lambda1, lambda2, lambda3,
                beam_size=beam, prefix_rule="raw_cost", use_reduction=True,
                timeout_flag=f), timeout_seconds, logger, "PACS+")
              record("PACS", to_points(r.frontier, budget) if r else [],
                   (r.solve_time_s + r.reduction_time_s) if r else t,
                   r.n_evals if r else np.nan, 1, failed=r is None, timed_out=to,
                   n_before=r.n_before if r else np.nan,
                   n_after=r.n_after if r else np.nan)

            # --- PACS with the extension-unsafe prefix rule ---
            if want("PACS_dC"):
              r, t, to = timed_for(inst, alpha, lambda f: solve_pbs(
                inst.slots, budget, alpha, lambda1, lambda2, lambda3,
                beam_size=beam, prefix_rule="overage", use_reduction=True,
                timeout_flag=f), timeout_seconds, logger, "PACS_dC")
              record("PACS_dC", to_points(r.frontier, budget) if r else [],
                   (r.solve_time_s + r.reduction_time_s) if r else t,
                   r.n_evals if r else np.nan, 1, failed=r is None, timed_out=to)

            # --- PTS ---
            if want("PTS"):
              r, t, to = timed_for(inst, alpha, lambda f: solve_pts(
                inst.slots, budget, alpha, lambda1, lambda2, lambda3,
                expansion_budget=expansion_budget, candidate_limit=candidate_limit,
                use_reduction=True,
                timeout_flag=f), timeout_seconds, logger, "PTS")
              record("PTS", to_points(r.frontier, budget) if r else [],
                   (r.solve_time_s + r.reduction_time_s) if r else t,
                   r.n_evals if r else np.nan, 1, failed=r is None, timed_out=to,
                   exhausted=int(r.exhausted) if r else 0,
                   n_action_evals=r.n_action_evals if r else np.nan,
                   n_rollout_evals=r.n_rollout_evals if r else np.nan,
                   n_bound_evals=r.n_bound_evals if r else np.nan,
                   n_frontier_comparisons=(r.n_frontier_comparisons if r else np.nan),
                   n_frontier_gains=r.n_frontier_gains if r else np.nan,
                   iterations=r.iterations if r else np.nan,
                   expansion_budget=expansion_budget, candidate_limit=candidate_limit)

            # Independent measured calls, never reuse reference timing or output.
            for name, fn in baseline_arms(inst, alpha, expansion_budget, candidate_limit,
                                          lambda1, lambda2, lambda3):
                if not want(name):
                    continue
                r, t, to = timed_for(inst, alpha, fn, timeout_seconds, logger, name)
                record(name, to_points(r.frontier, budget) if r else [], t,
                       r.n_evals if r else np.nan, 1, failed=r is None, timed_out=to,
                       n_before=r.n_before if r else np.nan,
                       n_after=r.n_after if r else np.nan)

            # --- epsilon-constraint ILP family ---
            for g in grid_list:
                name = f"eps-ILP g={g}"
                if not want(name):
                    continue
                out, t, to = timed_for(inst, alpha,
                    lambda f, g=g: solve_epsilon_ilp(q, inst.slots, alpha, g, f,
                                                     timeout_seconds),
                    timeout_seconds, logger, name)
                if out is None:
                    record(name, [], t, np.nan, 0, failed=True, timed_out=to)
                else:
                    front, stats = out
                    # timeout / internal time budget / non-optimal cells recorded
                    # separately. 'unreliable' below is the OR of them
                    record(name, front, t, stats["n_solves"] * inst.n_candidates,
                           stats["n_solves"],
                           timed_out=to, stats=stats, grid=g,
                           hit_time_budget=bool(stats["hit_time_budget"]),
                           n_nonoptimal=int(stats["n_nonoptimal"]),
                           n_infeasible=int(stats["n_infeasible"]))

            # --- PBF, on the queries where it finishes ---
            if not skip_pbf and want("PBF"):
                result, elapsed, hit = timed_for(inst, alpha,
                    lambda f: pbf_call(inst, alpha, pbf_timeout, pbf_max_combos, pbf_memory),
                    pbf_timeout, logger, "PBF")
                pbf_memory_fields(inst, result, pbf_max_combos, pbf_memory)
                record("PBF", result.frontier if result else [], elapsed,
                       result.n_evals if result else np.nan, 1,
                       failed=result is None, timed_out=hit)

        else:
            names = ["PACS", "PACS_dC", "PTS", *BASELINE_NAMES] + [f"eps-ILP g={g}" for g in grid_list]
            if not skip_pbf:
                names.append("PBF")
            for name in [n for n in names if want(n)]:
                record(name, [], np.nan, np.nan, 0)

        # --- references ---
        reference = exact_frontier(
            inst, alpha, lambda1, lambda2, lambda3, timeout_s=timeout_seconds)
        reference = validated_reference(reference, inst, alpha)
        exact, ex_done, ex_s = reference
        union = _filter_non_dominated([p for f in fr.values() for p in f])
        ref = exact if ex_done else []
        context = metric_context(inst, alpha, ex_done, ref, reference.status)
        ref_keys = {pkey(p) for p in ref}
        ref_pt = nadir(ref)
        hv_ref = hypervolume_3d(ref, ref_pt) if ref_pt else 0.0

        # sanity: every found point should be weakly dominated by the exact
        # reference. membership is the wrong test - dominated points are fine
        union_keys = {pkey(p) for p in union}
        outside = (sum(1 for p in union if not _weakly_dominated_by(p, ref))
                   if ex_done else np.nan)

        for name, pts in fr.items():
            m = meta[name]
            rows.append(report_row(inst, alpha, {
                "query_id": q.id, "method": name,
                "success": int(len(pts) > 0),
                **context,
                "solver_status": m["solver_status"],
                "invalid_assignments": m["invalid_assignments"],
                "n_expected_solves": m.get("n_expected_solves", np.nan),
                **score(pts, ref_keys, ref_pt, hv_ref,
                        metric_eligible=context["metric_eligible"]),
                "recall_vs_union": (len({pkey(p) for p in pts} & union_keys)
                                    / len(union_keys)) if union_keys else np.nan,
                "time_s": m["time_s"], "end_to_end_s": m["time_s"] + inst.retrieval_s,
                "evals": m["evals"], "solves": m["solves"],
                "timed_out": int(m.get("timed_out", False)),
                "hit_time_budget": int(m.get("hit_time_budget", False)),
                "n_nonoptimal": int(m.get("n_nonoptimal", 0)),
                "n_infeasible": int(m.get("n_infeasible", 0)),
                # incomplete for any reason. filter on this, not timed_out
                "unreliable": int(bool(m.get("timed_out", False))
                                  or bool(m.get("hit_time_budget", False))
                                  or int(m.get("n_nonoptimal", 0)) > 0),
                "n_before": m.get("n_before", np.nan),
                "n_after": m.get("n_after", np.nan),
                "ref_size": len(ref), "ref_is_exact": int(ex_done),
                "ref_time_s": ex_s, "union_outside_exact": outside,
                "n_candidates": inst.n_candidates, "n_slots": inst.n_slots,
                "within_budget": int(inst.within_budget),
                "retrieval_s": inst.retrieval_s,
                # PTS-only diagnostics, NaN for the rest
                "exhausted": m.get("exhausted", np.nan),
                "n_action_evals": m.get("n_action_evals", np.nan),
                "n_rollout_evals": m.get("n_rollout_evals", np.nan),
                "n_bound_evals": m.get("n_bound_evals", np.nan),
                "n_frontier_comparisons": m.get("n_frontier_comparisons", np.nan),
                "n_frontier_gains": m.get("n_frontier_gains", np.nan),
                "iterations": m.get("iterations", np.nan),
                "expansion_budget": m.get("expansion_budget", np.nan),
                "candidate_limit": m.get("candidate_limit", np.nan),
            }))
        save_query(Path("data_out") / exp_name, inst, alpha, reference, fr)
        pd.DataFrame(rows).to_csv(Path("data_out") / exp_name / "default.csv", index=False)
        logger.info(f"query {q.id}: ref={len(ref)}{'' if ex_done else ' (reference incomplete)'}  "
                    + "  ".join(f"{k}={len(v)}" for k, v in fr.items()))

    df = pd.DataFrame(rows)
    out = Path("data_out") / exp_name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "default.csv", index=False)
    save_as_paramed_file(df, exp_name, "default", query_setting, num_queries,
                         {"alpha": alpha, "beta": beta, "beam": beam})

    bad = df[(df.ref_is_exact == 1) & (df.union_outside_exact > 0)]
    if not bad.empty:
        logger.error(f"{bad.query_id.nunique()} queries found points outside the "
                     f"'exact' reference -- the reference is not exact there.")

    print(f"\n=== {df.query_id.nunique()} queries | reference = exact frontier "
          f"({df.ref_is_exact.mean()*100:.0f}% of queries) ===\n")
    agg = df.groupby("method").agg(
        requested=("query_id", "size"), feasible=("query_feasible", "sum"),
        metric_n=("metric_eligible", "sum"),
        success=("success", "mean"), size=("size", "mean"),
        recall=("recall", "mean"), hv=("hv_ratio", "mean"),
        evals=("evals", "median"), solves=("solves", "mean"),
        med_time=("time_s", "median"),
    ).sort_values("recall", ascending=False)
    with pd.option_context("display.width", 200,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(agg.to_string())


if __name__ == "__main__":
    main()
