#!/usr/bin/env python3
"""
Sample up to 10k items from each category in the merged keyword+embedding catalog
and write the result to a new catalog directory (SAMPLE_10k_EACH).

Output layout:
  {output_dir}/{Category}/batch_000000.parquet  (one file per category)

Usage:
  python -m gquery.prep.amazon.sample_catalog_10k
"""

import os
import sys
import random
from pathlib import Path

import pandas as pd
from pyarrow import parquet as pq

from gquery.prep.amazon.constants import INTERESTING_CATEGORIES

DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
BASE_DIR = os.path.join(DATASET_ROOT, "amazon_review_2023")

INPUT_DIR = os.path.join(BASE_DIR, "meta_parquet_clean_attributes_keywords_embed_merged")
OUTPUT_DIR = os.path.join(BASE_DIR, "meta_parquet_clean_attributes_keywords_embed_merged_10k_each")

ITEMS_PER_CATEGORY = 10_000
SEED = 42


def sample_category(category: str, n: int, seed: int) -> pd.DataFrame:
    cat_dir = Path(INPUT_DIR) / category
    files = sorted(cat_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {cat_dir}")

    dfs = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if len(df) <= n:
        print(f"  {category}: {len(df)} items (taking all — below cap of {n})")
        return df
    else:
        sampled = df.sample(n=n, random_state=seed)
        print(f"  {category}: sampled {n} / {len(df)} items")
        return sampled.reset_index(drop=True)


def main():
    rng = random.Random(SEED)
    out_base = Path(OUTPUT_DIR)

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Items per category: {ITEMS_PER_CATEGORY}, seed: {SEED}")
    print()

    total = 0
    for category in INTERESTING_CATEGORIES:
        out_dir = out_base / category
        out_file = out_dir / "batch_000000.parquet"

        if out_file.exists():
            n = pq.ParquetFile(out_file).metadata.num_rows
            print(f"  {category}: already exists ({n} rows), skipping")
            total += n
            continue

        df = sample_category(category, ITEMS_PER_CATEGORY, SEED)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_file, engine="pyarrow", index=False, compression="snappy")
        total += len(df)

    print(f"\nDone. Total items written: {total}")
    print(f"Catalog directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
