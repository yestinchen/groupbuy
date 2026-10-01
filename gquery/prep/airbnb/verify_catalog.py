#!/usr/bin/env python3
"""Verify the Inside Airbnb catalog against the Amazon catalog schema contract."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

AMZ = Path(os.path.expanduser(
    "~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_10k_each/All_Beauty/batch_000000.parquet"))
CAT = Path(os.path.expanduser("~/dataset/inside_airbnb/catalog"))

CONTRACT = [
    ("average_rating", pa.float64()),
    ("price", pa.float64()),
    ("main_category", pa.large_string()),
    ("store", pa.large_string()),
    ("rating_number", pa.int64()),
    ("categories", pa.list_(pa.string())),
    ("title_feats", pa.list_(pa.string())),
    ("title_for_embedding", pa.large_string()),
    ("keywords", pa.list_(pa.string())),
]

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


# --- 1. contract vs the Amazon reference schema -------------------------
amz = pq.ParquetFile(AMZ).schema_arrow
amz_names = list(amz.names)
print("Amazon reference schema:", AMZ)
print("  columns:", amz_names)
contract_names = [n for n, _ in CONTRACT]
check(amz_names[:len(contract_names)] == contract_names,
      f"Amazon column order {amz_names[:9]} != contract {contract_names}")
for n, t in CONTRACT:
    at = amz.field(n).type
    if at == t:
        note = "match"
    elif n == "categories" and pa.types.is_list(at):
        note = f"ALLOWED diff (Amazon {at}; column is always empty)"
    else:
        note = "MISMATCH"
        failures.append(f"Amazon {n}: {at} vs contract {t}")
    print(f"  {n:22s} amazon={str(at):22s} contract={str(t):22s} {note}")
print(f"  Amazon-only column not in contract (added by a later stage): "
      f"{[n for n in amz_names if n not in contract_names]}")

# --- 2. per-city checks --------------------------------------------------
rows = []
stats = json.load(open(CAT / "ingest_stats.json"))
raw_by_name = {s["name"]: s for s in stats}

for d in sorted(p for p in CAT.iterdir() if p.is_dir()):
    f = d / "batch_000000.parquet"
    check(f.exists(), f"{d.name}: missing batch_000000.parquet")
    if not f.exists():
        continue
    sch = pq.ParquetFile(f).schema_arrow
    check(list(sch.names) == contract_names,
          f"{d.name}: columns {list(sch.names)} != {contract_names}")
    for n, t in CONTRACT:
        check(sch.field(n).type == t, f"{d.name}: {n} type {sch.field(n).type} != {t}")

    df = pq.read_table(f).to_pandas()
    n = len(df)
    check(df["price"].notna().all() and (df["price"] > 0).all(),
          f"{d.name}: non-positive or null price")
    check(df["average_rating"].between(0, 5).all(),
          f"{d.name}: rating outside [0,5]")
    check(df["main_category"].eq(d.name).all(),
          f"{d.name}: main_category != dir name")
    check(df["rating_number"].ge(3).all(), f"{d.name}: rating_number < 3")
    check(df["categories"].map(len).eq(0).all(), f"{d.name}: non-empty categories")
    kw_frac = float((df["keywords"].map(len) > 0).mean())
    check(kw_frac >= 0.95, f"{d.name}: keywords non-empty on only {kw_frac:.3%}")
    tfe_frac = float((df["title_for_embedding"].str.strip().str.len() > 0).mean())
    check(tfe_frac == 1.0, f"{d.name}: title_for_embedding non-empty on {tfe_frac:.4%}")
    check(not df["title_for_embedding"].str.contains("<br", case=False).any(),
          f"{d.name}: <br> survived cleaning")
    check(df["store"].notna().all(), f"{d.name}: null store")

    s = raw_by_name[d.name]
    rows.append({
        "city": d.name,
        "raw": s["raw"],
        "out": n,
        "yield": n / s["raw"],
        "med_usd": float(df["price"].median()),
        "p90_usd": float(df["price"].quantile(0.90)),
        "med_rating": float(df["average_rating"].median()),
        "mean_kw": float(df["keywords"].map(len).mean()),
        "kw_ok": kw_frac,
        "tfe_ok": tfe_frac,
        "rating_step_drop": s["rating_drop_frac"],
    })

tab = pd.DataFrame(rows).sort_values("out", ascending=False)
pd.set_option("display.width", 200)
print("\nPer-city summary")
print(tab.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\nTOTAL raw={tab['raw'].sum():,} out={tab['out'].sum():,} "
      f"({tab['out'].sum()/tab['raw'].sum():.1%})")

big_rating_loss = tab[tab["rating_step_drop"] > 0.5]
print("\nCities losing >50% at the rating-composite step: "
      + (", ".join(big_rating_loss["city"]) if len(big_rating_loss) else "NONE"))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES:"))
for f_ in failures:
    print("  FAIL:", f_)
tab.to_json(CAT / "verify_table.json", orient="records", indent=2)
sys.exit(1 if failures else 0)
