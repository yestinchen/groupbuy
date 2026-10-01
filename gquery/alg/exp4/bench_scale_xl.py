'''
end-to-end scale ladder: 110k -> 330k -> 1.1m -> 2.2m items, same 100
queries, plus the recalibrated-budget arm at the top scale. per query we
report retrieval, search, and end-to-end time, candidate growth, and
recall/hv against the reference.

Recall and hypervolume require a nonempty complete reference. Incomplete
references retain explicit status and undefined quality metrics. Cached
retrieval durations do not define a measured total.
'''

from __future__ import annotations

from gquery.alg.exp4.protocol import report_row

from gquery.alg.exp4.core import timed_for, instance_status, metric_context
from gquery.alg.exp4.protocol import timing_record, write_manifest
from gquery.alg.exp4.default_reporting import finish_method, pbf_call, pbf_memory_fields, save_query, validated_reference
from gquery.alg.pareto.dp import solve_dp
from gquery.alg.exp4.sensitivity_reporting import reference_for

import json
import time
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd

from gquery.alg.pareto.epsilon_ilp import solve_epsilon_ilp
from gquery.alg.pareto.metrics import _filter_non_dominated, hypervolume_3d
from gquery.alg.exp4.core import (
    exact_frontier, load_instances, nadir, pkey, score, timed, to_points,
)
from gquery.alg.pareto.brute_force import solve_brute_force
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.utils import setup_logging

logger = setup_logging("exp6_scale")


@click.command()
@click.option("--scales", type=str,
              default="e11_cat10k,e11_cat30k,e12_cat1m,e12_cat2m,e12_cat2m_recal")
@click.option("--num-queries", type=int, default=100)
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--timeout-seconds", type=float, default=60.0)
@click.option("--ref-timeout", type=float, default=600.0)
@click.option("--grid", type=int, default=8)
@click.option("--ilp-settings", type=str, default="e11_cat10k,e11_cat30k",
              help="eps-ILP runtime grows with candidates; past 330k it is "
                   "cut rather than silently timing out")
@click.option("--pbf-settings", type=str, default="e11_cat10k",
              help="PBF only where the 200m-combination bound admits it")
@click.option("--pbf-timeout", type=float, default=30.0)
@click.option("--pbf-max-combos", type=int, default=200_000_000,
              help="Combination guard for PBF; 0 disables it")
@click.option("--pbf-memory-gib", type=float, default=None,
              help="PBF stops with its partial frontier once its RSS growth exceeds this many GiB")
@click.option("--uncapped-settings", type=str,
              default="e11_cat10k,e11_cat30k,e12_cat1m,e12_cat2m,e12_cat2m_recal",
              help="where uncapped PTS runs (pilot says it is sane everywhere)")
@click.option("--kappa-sweep", type=str, default="10,25,50,100,0")
@click.option("--kappa-settings", type=str, default="e12_cat2m",
              help="candidate-cap sweep at the top scale: is kappa=50 still "
                   "lossless at 2.2m")
@click.option("--methods", type=str, default=None,
              help="Comma separated subset of the arm names to measure (e.g. Pareto-DP); default measures every arm")
@click.option("--exp-name", type=str, default="exp6_scale")
def main(scales, num_queries, alpha, beta, beam, expansion_budget,
         candidate_limit, timeout_seconds, ref_timeout, grid, ilp_settings,
         pbf_settings, pbf_timeout, pbf_max_combos, pbf_memory_gib, uncapped_settings, kappa_sweep,
         kappa_settings, methods, exp_name):
    wanted = {m.strip() for m in methods.split(",") if m.strip()} if methods else None
    want = lambda name: wanted is None or name in wanted
    ilp_set = set(s.strip() for s in ilp_settings.split(",") if s.strip())
    pbf_set = set(s.strip() for s in pbf_settings.split(",") if s.strip())
    unc_set = set(s.strip() for s in uncapped_settings.split(",") if s.strip())
    kap_set = set(s.strip() for s in kappa_settings.split(",") if s.strip())
    kappas = [int(k) for k in kappa_sweep.split(",") if k.strip() != ""]

    rows: List[Dict] = []
    items_cache = None
    out = Path("data_out") / exp_name
    out.mkdir(parents=True, exist_ok=True)
    shared = dict(alpha=alpha, beta=beta, timeout_seconds=timeout_seconds)
    pts_label = "PTS"
    pbf_memory = int(pbf_memory_gib * 2**30) if pbf_memory_gib else None

    def frontier_json(pts):
        return json.dumps([dict(choice=list(p.choice), quality=p.quality, cost=p.cost,
                                rating=p.rating, cost_overage=p.cost_overage) for p in pts])

    for setting in [s.strip() for s in scales.split(",") if s.strip()]:
        t0 = time.perf_counter()
        try:
            insts, items_cache = load_instances(setting, num_queries, beta,
                                                alpha, items_cache=items_cache, include_invalid=True)
        except Exception as exc:
            logger.error(f"{setting}: {exc!r}")
            continue
        catalog_items = len(items_cache[1]) if items_cache is not None else np.nan
        logger.info(f"{setting}: {len(insts)} instances, catalog={catalog_items}, "
                    f"load {time.perf_counter() - t0:.0f}s")

        for inst in insts:
            b = inst.query.budget
            # The reference is computed once under its own 600 s limit and archived; it
            # is never reported as a measured method.
            reference = validated_reference(reference_for(inst, alpha, ref_timeout, exact_frontier), inst, alpha)
            (exact, ref_done, ref_s) = reference
            context = metric_context(inst, alpha, reference[1], reference[0], reference.status)
            context["metric_eligible"] = int(context["query_feasible"] and reference[1]
                                             and bool(exact) and reference.status == "complete")
            if reference.status == "complete" and not context["metric_eligible"] and context["query_feasible"]:
                context["reference_status"] = "reference_failure"
            save_query(out / "inputs" / setting, inst, alpha, reference, {})

            # Buffer measured arms before scoring against a complete reference.
            # (method, points, internal seconds, extra fields, failed, timed_out, stats, config)
            arms = []

            def timed_for_(inst_, alpha_, fn, limit, logger_, name):
                # an arm outside the requested subset is neither measured nor recorded
                return timed_for(inst_, alpha_, fn, limit, logger_, name) if want(name) else (None, np.nan, False)

            def add(arm):
                if want(arm[0]):
                    arms.append(arm)

            # Separate measured unbounded PBS invocation under the ordinary method limit.
            r, t, to = timed_for_(inst, alpha, lambda f: solve_pbs(
                inst.slots, b, alpha, beam_size=0, prefix_rule="raw_cost",
                use_reduction=True, timeout_flag=f),
                timeout_seconds, logger, "PACS unbounded")
            add(("PACS unbounded", to_points(r.frontier, b) if r else [],
                         (r.solve_time_s + r.reduction_time_s) if r else t,
                         {"evals": r.n_evals if r else np.nan,
                          "solve_s": r.solve_time_s if r else np.nan,
                          "reduction_s": r.reduction_time_s if r else np.nan,
                          "n_before": r.n_before if r else np.nan,
                          "n_after": r.n_after if r else np.nan},
                         r is None, to, None, dict(shared, beam=0)))

            # Conventional Pareto DP: standard label dominance on (quality, raw cost, rating), no candidate reduction.
            r, t, to = timed_for_(inst, alpha, lambda f: solve_dp(
                inst.slots, b, alpha, timeout_flag=f), timeout_seconds, logger, "Pareto-DP")
            add(("Pareto-DP", to_points(r.frontier, b) if r else [],
                 (r.solve_time_s + r.reduction_time_s) if r else t,
                 {"evals": r.n_evals if r else np.nan,
                  "solve_s": r.solve_time_s if r else np.nan,
                  "reduction_s": r.reduction_time_s if r else np.nan,
                  "n_before": r.n_before if r else np.nan,
                  "n_after": r.n_after if r else np.nan},
                 r is None, to, None, dict(shared, use_reduction=False)))

            r, t, to = timed_for_(inst, alpha, lambda f: solve_pbs(
                inst.slots, b, alpha, beam_size=beam, prefix_rule="raw_cost",
                use_reduction=True, timeout_flag=f),
                timeout_seconds, logger, "PACS")
            add(("PACS", to_points(r.frontier, b) if r else [],
                         (r.solve_time_s + r.reduction_time_s) if r else t,
                         {"evals": r.n_evals if r else np.nan,
                          "solve_s": r.solve_time_s if r else np.nan,
                          "reduction_s": r.reduction_time_s if r else np.nan,
                          "n_before": r.n_before if r else np.nan,
                          "n_after": r.n_after if r else np.nan},
                         r is None, to, None, dict(shared, beam=beam)))

            r, t, to = timed_for_(inst, alpha, lambda f: solve_pts(
                inst.slots, b, alpha, expansion_budget=expansion_budget,
                candidate_limit=candidate_limit, use_reduction=True,
                timeout_flag=f), timeout_seconds, logger, pts_label)
            add((pts_label, to_points(r.frontier, b) if r else [],
                         (r.solve_time_s + r.reduction_time_s) if r else t,
                         {"evals": r.n_evals if r else np.nan,
                          "exhausted": int(r.exhausted) if r else 0,
                          "solve_s": r.solve_time_s if r else np.nan,
                          "reduction_s": r.reduction_time_s if r else np.nan,
                          "n_before": r.n_before if r else np.nan,
                          "n_after": r.n_after if r else np.nan},
                         r is None, to, None,
                         dict(shared, expansion_budget=expansion_budget, candidate_limit=candidate_limit)))

            if setting in unc_set:
                r, t, to = timed_for_(inst, alpha, lambda f: solve_pts(
                    inst.slots, b, alpha, expansion_budget=expansion_budget,
                    candidate_limit=0, use_reduction=True, timeout_flag=f),
                    timeout_seconds, logger, f"{pts_label} uncapped")
                add((f"{pts_label} uncapped",
                             to_points(r.frontier, b) if r else [],
                             (r.solve_time_s + r.reduction_time_s) if r else t,
                             {"evals": r.n_evals if r else np.nan,
                              "exhausted": int(r.exhausted) if r else 0,
                              "solve_s": r.solve_time_s if r else np.nan,
                              "reduction_s": r.reduction_time_s if r else np.nan},
                             r is None, to, None,
                             dict(shared, expansion_budget=expansion_budget, candidate_limit=0)))

            if setting in kap_set:
                for k in kappas:
                    if k == candidate_limit or (k == 0 and setting in unc_set):
                        continue
                    r, t, to = timed_for_(inst, alpha, lambda f: solve_pts(
                        inst.slots, b, alpha,
                        expansion_budget=expansion_budget,
                        candidate_limit=k, use_reduction=True,
                        timeout_flag=f), timeout_seconds, logger, f"{pts_label} k={k}")
                    add((f"{pts_label} k={k}",
                                 to_points(r.frontier, b) if r else [],
                                 (r.solve_time_s + r.reduction_time_s) if r else t,
                                 {"evals": r.n_evals if r else np.nan,
                                  "exhausted": int(r.exhausted) if r else 0,
                                  "solve_s": r.solve_time_s if r else np.nan,
                                  "reduction_s": r.reduction_time_s if r else np.nan},
                                 r is None, to, None,
                                 dict(shared, expansion_budget=expansion_budget, candidate_limit=k)))

            if setting in ilp_set:
                out_ilp, t, to = timed_for_(inst, alpha,
                    lambda f: solve_epsilon_ilp(inst.query, inst.slots, alpha,
                                                grid, f, timeout_seconds),
                    timeout_seconds, logger, f"eps-ILP g={grid}")
                if out_ilp is None:
                    add((f"eps-ILP g={grid}", [], t, {"solves": 0}, True, to, None,
                                 dict(shared, grid=grid, per_solve_budget_s=timeout_seconds)))
                else:
                    front, stats = out_ilp
                    add((f"eps-ILP g={grid}", front, t,
                                 {"solves": stats["n_solves"],
                                  "evals": stats["n_solves"] * inst.n_candidates},
                                 False, to, stats, dict(shared, grid=grid, per_solve_budget_s=timeout_seconds)))

            if setting in pbf_set:
                # Measured brute force call with its work limit kept distinct from timeout.
                bf, t, to = timed_for_(inst, alpha,
                    lambda f: pbf_call(inst, alpha, pbf_timeout, pbf_max_combos, pbf_memory),
                    pbf_timeout + timeout_seconds, logger, "PBF")
                pbf_memory_fields(inst, bf, pbf_max_combos, pbf_memory)
                add(("PBF", bf.frontier if bf else [], t,
                             {"pbf_done": int(bf.completed) if bf else 0,
                              "evals": bf.n_evals if bf else np.nan},
                             bf is None, to, None,
                             dict(shared, pbf_timeout=pbf_timeout, pbf_max_combos=pbf_max_combos,
                                  pbf_max_memory_bytes=pbf_memory)))

            ref_points = exact if context["metric_eligible"] else []
            ref_mode = "exact" if context["metric_eligible"] else reference.status
            ref_keys = {pkey(p) for p in ref_points}
            ref_pt = nadir(ref_points)
            hv_ref = hypervolume_3d(ref_points, ref_pt) if ref_pt else 0.0

            base = {**context,
                "setting": setting, "catalog_items": catalog_items,
                "query_id": inst.query.id,
                "n_candidates": inst.n_candidates, "n_slots": inst.n_slots,
                "retrieval_s": inst.retrieval_s if getattr(inst, "retrieval_measured", False) else np.nan,
                    "retrieval_historical_s": inst.retrieval_s if not getattr(inst, "retrieval_measured", False) else np.nan,
                "budget": b, "min_feasible_cost": inst.min_feasible_cost,
                "minimum_assignment_cost": inst.min_feasible_cost,
                "relaxed_cap": b * (1.0 + alpha),
                "within_budget": int(inst.within_budget),
                "tight80": int(inst.min_feasible_cost > 0.8 * b),
                "ref_mode": ref_mode, "ref_done": int(ref_done),
                "ref_s": ref_s, "reference_s": ref_s, "ref_timeout_s": ref_timeout,
                "ref_size": len(exact),
            }
            for method, pts, secs, extra, failed, to, stats, config in arms:
                pts, fields = finish_method(inst, alpha, method, pts, failed=failed, timed_out=to,
                                            stats=stats, grid=grid if stats is not None else None)
                rows.append(report_row(inst, alpha, {**base, **fields, "method": method,
                             "method_config": json.dumps(config, sort_keys=True),
                             "frontier_json": frontier_json(pts),
                             "success": int(len(pts) > 0),
                             **score(pts, ref_keys, ref_pt, hv_ref, metric_eligible=context["metric_eligible"]),
                             "search_s": secs,
                             "end_to_end_s": (secs + inst.retrieval_s) if getattr(inst, "retrieval_measured", False) else np.nan, **extra}))
            logger.info(f"{setting} q{inst.query.id}: cand={inst.n_candidates} "
                        f"ref={ref_mode} {ref_s:.1f}s size={len(ref_points)}")

        df = pd.DataFrame(rows)
        df.to_csv(out / "scale_xl.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== per-scale medians ===")
    agg = df.groupby(["setting", "method"]).agg(
        requested=("query_id", "size"), feasible=("query_feasible", "sum"),
        metric_n=("metric_eligible", "sum"),
        recall=("recall", "median"), hv=("hv_ratio", "median"),
        search_ms=("search_s", lambda v: v.median() * 1000),
        retr_ms=("retrieval_s", lambda v: v.median() * 1000),
        e2e_ms=("end_to_end_s", lambda v: v.median() * 1000),
        cands=("n_candidates", "median"),
    )
    with pd.option_context("display.width", 220,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(agg.to_string())

    print("\n=== budget tightness by scale ===")
    per_q = df.drop_duplicates(["setting", "query_id"])
    tight = per_q.groupby("setting").agg(
        within_budget=("within_budget", "mean"), tight80=("tight80", "mean"),
        mfc_median=("min_feasible_cost", "median"))
    print(tight.to_string())


if __name__ == "__main__":
    main()
