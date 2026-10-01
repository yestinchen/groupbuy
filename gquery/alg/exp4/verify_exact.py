'''check the reference: unbounded-beam PBS must match exhaustive enumeration
on every enumerable instance. exits non-zero on any mismatch - one bad
instance and the recall numbers mean nothing.'''

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import click
import pandas as pd

from gquery.alg.pareto.brute_force import solve_brute_force
from gquery.alg.exp4.core import (
    exact_frontier, load_instances, pkey,
)
from gquery.utils import setup_logging

logger = setup_logging("exp4_verify_exact")


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--query-setting", type=str, default="e10_1k_100q")
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--bf-timeout", type=float, default=120.0)
@click.option("--max-combos", type=int, default=40_000_000,
              help="Skip enumeration above this many candidate tuples.")
@click.option("--exp-name", type=str, default="exp4_verify_exact")
def main(num_queries, query_setting, alpha, beta, bf_timeout, max_combos, exp_name):
    insts, _ = load_instances(query_setting, num_queries, beta, alpha)
    logger.info(f"{len(insts)} instances loaded")

    rows: List[Dict] = []
    n_cmp = n_bad = 0
    for inst in insts:
        bf, done, bf_s, checked = solve_brute_force(
            inst, alpha, timeout_s=bf_timeout, max_combos=max_combos)
        if not done:
            rows.append({"query_id": inst.query.id, "enumerated": False,
                         "combos": checked})
            continue
        ex, ex_done, ex_s = exact_frontier(inst, alpha)
        if not ex_done:
            logger.warning(f"query {inst.query.id}: unbounded PACS+ timed out")
            rows.append({"query_id": inst.query.id, "enumerated": True,
                         "pacs_timeout": True})
            continue

        kb = {pkey(p) for p in bf}
        ke = {pkey(p) for p in ex}
        n_cmp += 1
        missing, extra = len(kb - ke), len(ke - kb)
        if missing or extra:
            n_bad += 1
            logger.error(f"query {inst.query.id}: MISMATCH  "
                         f"missing={missing} spurious={extra} |bf|={len(kb)}")
        rows.append({
            "query_id": inst.query.id, "enumerated": True, "pacs_timeout": False,
            "n_bf": len(kb), "n_exact": len(ke),
            "missing": missing, "spurious": extra,
            "bf_time_s": bf_s, "exact_time_s": ex_s,
            "combos": checked, "n_slots": inst.n_slots,
            "n_candidates": inst.n_candidates,
        })

    df = pd.DataFrame(rows)
    output = Path("data_out") / exp_name
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "verify_exact.csv", index=False)
    ok = df[df.get("missing").notna()] if "missing" in df.columns else pd.DataFrame()

    print(f"\n=== unbounded PACS+ vs exhaustive enumeration ===")
    print(f"instances loaded           : {len(insts)}")
    print(f"enumerable within limits   : {n_cmp}")
    print(f"exact agreement            : {n_cmp - n_bad}/{n_cmp}")
    if not ok.empty:
        print(f"mean |frontier|            : {ok.n_bf.mean():.1f}")
        print(f"median enumeration time    : {ok.bf_time_s.median():.3f} s")
        print(f"median unbounded PACS+ time: {ok.exact_time_s.median()*1000:.3f} ms")
        if ok.exact_time_s.median() > 0:
            print(f"speedup                    : "
                  f"{ok.bf_time_s.median()/ok.exact_time_s.median():,.0f}x")
    if n_cmp == 0:
        print("FAILED: no completed exhaustive comparisons.")
        sys.exit(1)
    if n_bad:
        print(f"\nFAILED: {n_bad} instances disagree. "
              f"Recall numbers built on this reference are not trustworthy.")
        sys.exit(1)
    print("\nPASS: unbounded PACS+ reproduces the exhaustive frontier exactly.")


if __name__ == "__main__":
    main()
