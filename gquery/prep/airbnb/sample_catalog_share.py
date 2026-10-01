#!/usr/bin/env python3
"""
Nested per-city subsamples of the Airbnb catalog for the downward scale
ladder (PLAN_AIRBNB_LADDER.md).

One deterministic permutation per city (seed pinned); the rung with share s
takes the first ceil(s * n_city) rows of that permutation, so every smaller
rung is a subset of every larger one and the full pool is the top rung.
Shares keep the relative city mix (5K Amsterdam .. 40K London) at every
rung instead of flattening it.

Output layout mirrors the source catalog so `gquery.ioutils.load_items` and
`gquery.prep.airbnb.qgen` read it unchanged:

  <out-root>_lad<label>/<City>/batch_000000.parquet      (full schema)

Usage:
  python -m gquery.prep.airbnb.sample_catalog_share build
  python -m gquery.prep.airbnb.sample_catalog_share verify
"""

from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import click
import numpy as np
import pandas as pd
from pyarrow import parquet as pq

CATALOG = os.path.expanduser("~/dataset/inside_airbnb/catalog_titled_embed")
# (share, dir suffix) -- labels are the rounded total row counts
RUNGS = [(0.125, "lad29k"), (0.25, "lad58k"), (0.5, "lad115k")]
KEY_COLS = ["title_for_embedding", "price", "rating_number"]
NONNULL_COLS = ["price", "average_rating", "title_embedding"]


def _rung_dir(catalog: str, suffix: str) -> Path:
    return Path(f"{catalog}_{suffix}")


def _city_dirs(catalog: str):
    for d in sorted(Path(catalog).iterdir()):
        f = d / "batch_000000.parquet"
        if f.exists():
            yield d.name, f


@click.group()
def cli():
    pass


@cli.command("build")
@click.option("--catalog", type=str, default=CATALOG, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def build(catalog, seed):
    totals = Counter()
    for city, f in _city_dirs(catalog):
        df = pq.read_table(f).to_pandas()
        n = len(df)
        perm = np.random.default_rng(seed).permutation(n)
        line = [f"{city:>16} {n:>6}"]
        for share, suffix in RUNGS:
            k = int(math.ceil(share * n))
            sub = df.iloc[np.sort(perm[:k])].reset_index(drop=True)
            out = _rung_dir(catalog, suffix) / city
            out.mkdir(parents=True, exist_ok=True)
            sub.to_parquet(out / "batch_000000.parquet", index=False)
            totals[suffix] += k
            line.append(f"{suffix}={k}")
        totals["full"] += n
        print("  ".join(line))
    print("totals: " + ", ".join(f"{k}={v}" for k, v in totals.items()))


@cli.command("verify")
@click.option("--catalog", type=str, default=CATALOG, show_default=True)
def verify(catalog):
    '''hard gates: nestedness (multiset of KEY_COLS) along the ladder, no
    nulls in the columns the solvers and retrieval read, same schema.'''
    ladder = [_rung_dir(catalog, s) for _, s in RUNGS] + [Path(catalog)]
    failures = 0
    totals = Counter()
    for city, _ in _city_dirs(catalog):
        prev_keys, prev_schema = None, None
        for d in ladder:
            f = d / city / "batch_000000.parquet"
            if not f.exists():
                print(f"MISSING {f}")
                failures += 1
                continue
            t = pq.read_table(f)
            totals[d.name] += t.num_rows
            for c in NONNULL_COLS:
                if t.column(c).null_count:
                    print(f"NULLS {f} {c}")
                    failures += 1
            df = t.select(KEY_COLS).to_pandas()
            keys = Counter(map(tuple, df.itertuples(index=False, name=None)))
            if prev_keys is not None:
                if any(keys[k] < v for k, v in prev_keys.items()):
                    print(f"NOT NESTED {city}: {d.name} does not contain the "
                          f"previous rung")
                    failures += 1
                if t.schema.names != prev_schema:
                    print(f"SCHEMA {city}: {d.name} differs")
                    failures += 1
            prev_keys, prev_schema = keys, t.schema.names
    print("rows: " + ", ".join(f"{k}={v}" for k, v in totals.items()))
    print("PASS" if failures == 0 else f"FAIL ({failures})")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    cli()
