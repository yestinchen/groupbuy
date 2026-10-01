'''
recalibrated budgets for the top scale. budgets were generated as
budget = m * min_feasible_cost on the 10k catalog (60pct easy at m >= 1.1,
40pct tight at m < 1.0). a bigger catalog holds cheaper matches, mfc drops,
and tight queries silently go easy. recover each query's multiplier and
re-anchor it:

  m_q       = budget_q / mfc_10k(q)
  budget'_q = m_q * mfc_cat2m(q)

writes queries_recal_cat2m.parquet next to the original query file, same
rows, only the budget column changed.
'''

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from pyarrow import parquet as pq

from gquery.alg.exp4.core import load_instances
from gquery.ioutils import file_name_path_dict

SRC_KEY = "sample_gqueries_e10_1k_100q"
DST_KEY = "sample_gqueries_e10_1k_100q_recal_cat2m"


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--base-setting", type=str, default="e11_cat10k")
@click.option("--target-setting", type=str, default="e12_cat2m")
@click.option("--dst-key", type=str, default=DST_KEY,
              help="file_name_path_dict key the parquet is written to; must "
                   "match the target catalog (recal ladder extension)")
@click.option("--src-key", type=str, default=SRC_KEY,
              help="file_name_path_dict key of the base-rung query parquet "
                   "(airbnb ladder: sample_gqueries_airbnb_ladder)")
def main(num_queries, alpha, beta, base_setting, target_setting, dst_key,
         src_key):
    base_insts, _ = load_instances(base_setting, num_queries, beta, alpha)
    tgt_insts, _ = load_instances(target_setting, num_queries, beta, alpha)
    mfc_base = {i.query.id: i.min_feasible_cost for i in base_insts}
    mfc_tgt = {i.query.id: i.min_feasible_cost for i in tgt_insts}

    df = pq.read_table(file_name_path_dict[src_key]).to_pandas()
    out_path = Path(file_name_path_dict[dst_key])

    new_budgets = []
    mults = []
    drift = []
    for idx, row in df.head(num_queries).iterrows():
        qid = str(idx)
        b = float(row["budget"])
        if qid not in mfc_base or qid not in mfc_tgt:
            new_budgets.append(b)
            continue
        m = b / mfc_base[qid]
        new_budgets.append(m * mfc_tgt[qid])
        mults.append(m)
        drift.append(mfc_tgt[qid] / mfc_base[qid])

    df = df.head(num_queries).copy()
    old = df["budget"].astype(float).values
    df["budget"] = new_budgets
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    mults = np.array(mults)
    drift = np.array(drift)
    print(f"wrote {out_path} ({len(df)} queries)")
    print(f"multipliers: tight (m<1.0) {np.mean(mults < 1.0):.2f}, "
          f"easy (m>=1.1) {np.mean(mults >= 1.1):.2f}")
    print(f"mfc drift {target_setting}/{base_setting}: median {np.median(drift):.3f}, "
          f"p10 {np.percentile(drift, 10):.3f}, p90 {np.percentile(drift, 90):.3f}")
    print(f"budget change: median {np.median(np.array(new_budgets)/old):.3f}x")


if __name__ == "__main__":
    main()
