
'''sensitivity sweeps -> sensitivity.csv: beam width b, PTS budget M,
candidate cap kappa (0 = uncapped), and beta. careful: beta changes the
instance itself, so that sweep re-retrieves. queries with an empty requirement
at high beta go into n_infeasible, they are not scored as recall 0.'''

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from gquery.alg.common import save_as_paramed_file
from gquery.alg.exp4.core import (
    exact_frontier, load_instances, nadir, pkey, score, timed, to_points,
)
from gquery.alg.pareto.brute_force import solve_brute_force
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp
from gquery.alg.pareto.metrics import hypervolume_3d
from gquery.utils import setup_logging

logger = setup_logging("exp4_sensitivity")


def _reference(inst, alpha, timeout_s):
    exact, done, _ = exact_frontier(inst, alpha, timeout_s=timeout_s)
    if not done or not exact:
        return None
    ref_pt = nadir(exact)
    return exact, {pkey(p) for p in exact}, ref_pt, hypervolume_3d(exact, ref_pt)


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--query-setting", type=str, default="e10_1k_100q")
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--betas", type=str, default="0.4,0.5,0.6,0.7,0.8,0.9")
@click.option("--alphas", type=str, default="0.1,0.25,0.5,0.75,1.0")
@click.option("--beams", type=str, default="1,5,20,50,100,200,500,1000")
@click.option("--budgets", type=str, default="200,500,1000,2000,4000,8000")
@click.option("--kappas", type=str, default="1,5,20,50,100,200,0")
@click.option("--beta-grid", type=int, default=8,
              help="epsilon-ILP grid used in the beta sweep")
@click.option("--beta-ilp-budget", type=float, default=10.0,
              help="per-query time budget for the epsilon-ILP grid in the beta sweep")
@click.option("--beta-bf-timeout", type=float, default=10.0,
              help="per-query brute force limit in the beta sweep")
@click.option("--skip-beta-baselines", is_flag=True, default=False)
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--exp-name", type=str, default="exp4_pplus_sensitivity")
def main(num_queries, query_setting, alpha, beta, betas, alphas, beams, budgets,
         kappas, beta_grid, beta_ilp_budget, beta_bf_timeout, skip_beta_baselines,
         beam, expansion_budget, candidate_limit, timeout_seconds, exp_name):
    beam_list = [int(x) for x in beams.split(",") if x.strip()]
    budget_list = [int(x) for x in budgets.split(",") if x.strip()]
    kappa_list = [int(x) for x in kappas.split(",") if x.strip()]
    beta_list = [float(x) for x in betas.split(",") if x.strip()]
    alpha_list = [float(x) for x in alphas.split(",") if x.strip()]

    rows: List[Dict] = []

    def run_pacs(inst, b, ref):
        r, t, to = timed(lambda f: solve_pbs(
            inst.slots, inst.query.budget, alpha, beam_size=b,
            prefix_rule="raw_cost", use_reduction=True, timeout_flag=f),
            timeout_seconds, logger, f"PACS b={b}")
        pts = to_points(r.frontier, inst.query.budget) if r else []
        return pts, ((r.solve_time_s + r.reduction_time_s) if r else t), \
            (r.n_evals if r else np.nan), to, None

    def run_pmcts(inst, m, k, ref):
        r, t, to = timed(lambda f: solve_pts(
            inst.slots, inst.query.budget, alpha, expansion_budget=m,
            candidate_limit=k, use_reduction=True, timeout_flag=f),
            timeout_seconds, logger, f"PTS M={m} k={k}")
        pts = to_points(r.frontier, inst.query.budget) if r else []
        return pts, ((r.solve_time_s + r.reduction_time_s) if r else t), \
            (r.n_evals if r else np.nan), to, (int(r.exhausted) if r else 0)

    def emit(sweep, method, param, inst, pts, secs, evals, to, exhausted, ref):
        _, ref_keys, ref_pt, hv_ref = ref
        rows.append({
            "sweep": sweep, "method": method, "param": param,
            "exhausted": exhausted,
            "query_id": inst.query.id, "success": int(len(pts) > 0),
            **score(pts, ref_keys, ref_pt, hv_ref),
            "time_s": secs, "retrieval_s": inst.retrieval_s,
            "end_to_end_s": secs + inst.retrieval_s,
            "evals": evals, "timed_out": int(to),
            "ref_size": len(ref_keys), "n_candidates": inst.n_candidates,
            "n_slots": inst.n_slots,
        })

    # --- sweeps that share one candidate set -------------------------------
    insts, _ = load_instances(query_setting, num_queries, beta, alpha)
    logger.info(f"default beta={beta}: {len(insts)} instances")
    for inst in insts:
        ref = _reference(inst, alpha, timeout_seconds)
        if ref is None:
            logger.warning(f"query {inst.query.id}: no exact reference, skipped")
            continue
        for b in beam_list:
            emit("beam", "PACS", b, inst, *run_pacs(inst, b, ref), ref)
        for m in budget_list:
            emit("budget", "PTS", m, inst,
                 *run_pmcts(inst, m, candidate_limit, ref), ref)
        for k in kappa_list:
            emit("kappa", "PTS", k, inst,
                 *run_pmcts(inst, expansion_budget, k, ref), ref)

    # --- beta sweep: the instance itself changes ---------------------------
    for bt in beta_list:
        binsts, _ = load_instances(query_setting, num_queries, bt, alpha)
        n_infeasible = num_queries - len(binsts)
        logger.info(f"beta={bt}: {len(binsts)} instances "
                    f"({n_infeasible} with an empty requirement)")
        for inst in binsts:
            ref = _reference(inst, alpha, timeout_seconds)
            if ref is None:
                continue
            for name, fn in (("PACS", lambda i: run_pacs(i, beam, ref)),
                             ("PTS", lambda i: run_pmcts(i, expansion_budget,
                                                         candidate_limit, ref))):
                pts, secs, evals, to, exh = fn(inst)
                emit("beta", name, bt, inst, pts, secs, evals, to, exh, ref)

            # timed-out baseline runs are kept as success=0, not dropped
            if skip_beta_baselines:
                continue

            name = f"eps-ILP g={beta_grid}"
            out, t, to = timed(
                lambda f: solve_epsilon_ilp(inst.query, inst.slots, alpha,
                                            beta_grid, f, beta_ilp_budget),
                timeout_seconds, logger, name)
            if out is None:
                emit("beta", name, bt, inst, [], t, np.nan, int(to), None, ref)
            else:
                front, stats = out
                emit("beta", name, bt, inst, front, t,
                     stats["n_solves"] * inst.n_candidates,
                     int(to or stats.get("hit_time_budget", False)), None, ref)

            bf, done, bf_s, checked = solve_brute_force(
                inst, alpha, timeout_s=beta_bf_timeout)
            emit("beta", "PBF", bt, inst, bf if done else [],
                 bf_s if done else beta_bf_timeout,
                 checked * inst.n_slots if done else np.nan,
                 int(not done), None, ref)
        # write down the retrieval-induced infeasibility, once per beta
        rows.append({"sweep": "beta_infeasible", "method": "retrieval",
                     "param": bt, "query_id": "-", "n_infeasible": n_infeasible,
                     "n_total": num_queries,
                     "n_candidates": (np.mean([i.n_candidates for i in binsts])
                                      if binsts else 0.0)})

    # --- alpha sweep: budget relaxation. retrieval is beta-keyed and unchanged,
    # but alpha sets the feasible cap B*(1+alpha) and the dC normalization, so
    # both the reference frontier and candidate reduction depend on it. we reload
    # instances and recompute the exact reference per alpha. queries with no
    # assignment under the relaxed cap are counted as budget-infeasible. ---
    for al in alpha_list:
        ainsts, _ = load_instances(query_setting, num_queries, beta, al)
        n_feasible = 0
        for inst in ainsts:
            exact, done, _ = exact_frontier(inst, al, timeout_s=timeout_seconds)
            if not done:
                continue  # a timeout is not an alpha effect; drop, do not mislabel
            if not exact:
                continue  # no assignment fits the relaxed cap at this alpha
            n_feasible += 1
            ref_pt = nadir(exact)
            ref = (exact, {pkey(p) for p in exact}, ref_pt,
                   hypervolume_3d(exact, ref_pt))
            rb, tb, tob = timed(lambda f: solve_pbs(
                inst.slots, inst.query.budget, al, beam_size=beam,
                prefix_rule="raw_cost", use_reduction=True, timeout_flag=f),
                timeout_seconds, logger, f"PACS alpha={al}")
            emit("alpha", "PACS", al, inst,
                 to_points(rb.frontier, inst.query.budget) if rb else [],
                 (rb.solve_time_s + rb.reduction_time_s) if rb else tb,
                 rb.n_evals if rb else np.nan, tob, None, ref)
            rm, tm, tom = timed(lambda f: solve_pts(
                inst.slots, inst.query.budget, al, expansion_budget=expansion_budget,
                candidate_limit=candidate_limit, use_reduction=True, timeout_flag=f),
                timeout_seconds, logger, f"PTS alpha={al}")
            emit("alpha", "PTS", al, inst,
                 to_points(rm.frontier, inst.query.budget) if rm else [],
                 (rm.solve_time_s + rm.reduction_time_s) if rm else tm,
                 rm.n_evals if rm else np.nan, tom,
                 (int(rm.exhausted) if rm else 0), ref)
        logger.info(f"alpha={al}: {n_feasible}/{len(ainsts)} feasible")
        rows.append({"sweep": "alpha_infeasible", "method": "budget",
                     "param": al, "query_id": "-",
                     "n_infeasible": len(ainsts) - n_feasible,
                     "n_total": len(ainsts)})

    df = pd.DataFrame(rows)
    out = Path("data_out") / exp_name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sensitivity.csv", index=False)
    save_as_paramed_file(df, exp_name, "sensitivity", query_setting, num_queries,
                         {"alpha": alpha, "beta": beta, "beam": beam})

    for sweep in ("beam", "budget", "kappa", "beta", "alpha"):
        s = df[df.sweep == sweep]
        if s.empty:
            continue
        print(f"\n=== {sweep} sweep ===")
        agg = s.groupby(["method", "param"]).agg(
            success=("success", "mean"), size=("size", "mean"),
            recall=("recall", "mean"), hv=("hv_ratio", "mean"),
            evals=("evals", "median"), med_time=("time_s", "median"),
            n=("query_id", "nunique"),
        )
        with pd.option_context("display.width", 200,
                               "display.float_format", lambda v: f"{v:,.4f}"):
            print(agg.to_string())
    inf = df[df.sweep == "beta_infeasible"]
    if not inf.empty:
        print("\n=== queries with an empty requirement (excluded above) ===")
        print(inf[["param", "n_infeasible", "n_total", "n_candidates"]].to_string(index=False))
    ainf = df[df.sweep == "alpha_infeasible"]
    if not ainf.empty:
        print("\n=== queries with no assignment under the relaxed cap (by alpha) ===")
        print(ainf[["param", "n_infeasible", "n_total"]].to_string(index=False))


# Retain the historical callback for reviewing the existing local edits.
# All CLI invocations use the requested-input reporting implementation below.
legacy_callback = main.callback
from gquery.alg.exp4.sensitivity_reporting import run_sensitivity
main.callback = run_sensitivity


if __name__ == "__main__":
    main()
