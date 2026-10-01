#!/usr/bin/env python3
"""
Sample N items per category from the full merged keyword+embedding catalog,
ensuring the result is a strict superset of an existing baseline catalog
(SAMPLE_10k_EACH by default).

For each category:
  1. Load the existing baseline items.
  2. Load the full catalog.
  3. Keep all baseline items, then sample (N - n_baseline) additional items from
     the remainder of the full catalog.

This guarantees nested catalogs: 10k ⊂ 15k ⊂ 20k ⊂ 25k ⊂ 30k, so the same
queries remain valid across all catalog sizes.

Two sizing modes (exactly one must be given):
  --items-per-category N   flat target: every category gets N items (capped by
                           the pool size).
  --share FLOAT            proportional target: category c gets
                           n_c = max(n_baseline_c, ceil(share * pool_c)),
                           i.e. a fixed share of the full pool, never dropping
                           below the baseline. This keeps the relative category
                           mix of the full catalog instead of flattening it.

Output layout:
  {output_dir}/{Category}/batch_000000.parquet

Usage:
  python -m gquery.prep.amazon.sample_catalog_nk --items-per-category 15000
  python -m gquery.prep.amazon.sample_catalog_nk --items-per-category 20000
  python -m gquery.prep.amazon.sample_catalog_nk --share 0.5 \
      --baseline-dir ~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_30k_each
"""

import math
import os
import sys
from pathlib import Path

import click
import pandas as pd
from pyarrow import parquet as pq

from gquery.prep.amazon.constants import INTERESTING_CATEGORIES

DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
BASE_DIR = os.path.join(DATASET_ROOT, "amazon_review_2023")
INPUT_DIR = os.path.join(BASE_DIR, "meta_parquet_clean_attributes_keywords_embed_merged")
BASELINE_DIR = os.path.join(BASE_DIR, "meta_parquet_clean_attributes_keywords_embed_merged_10k_each")

DEFAULT_SEED = 42


def _load_parquets(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    dfs = [pq.read_table(f).to_pandas() for f in files]
    return pd.concat(dfs, ignore_index=True)


def _count_rows(directory: Path) -> int:
    """Row count of a category directory, from parquet metadata only."""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}")
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def sample_category(category: str, n: int, seed: int, baseline_dir: Path) -> pd.DataFrame:
    """Sample n items for a category, keeping all baseline items."""
    # Load baseline items
    baseline_cat_dir = Path(baseline_dir) / category
    baseline_df = _load_parquets(baseline_cat_dir)
    n_baseline = len(baseline_df)

    if n <= n_baseline:
        # Target is smaller than or equal to baseline; just take first n from baseline
        print(f"  {category}: taking {n} / {n_baseline} baseline items (target <= baseline)")
        return baseline_df.iloc[:n].reset_index(drop=True)

    # Load full catalog
    full_dir = Path(INPUT_DIR) / category
    full_df = _load_parquets(full_dir)

    if len(full_df) <= n:
        print(f"  {category}: {len(full_df)} items (taking all -- below cap of {n})")
        return full_df.reset_index(drop=True)

    # Find items in full catalog that are NOT in the baseline.
    # Use title_for_embedding + price as a composite key.
    key_col = "title_for_embedding"
    baseline_keys = set(
        zip(baseline_df[key_col].fillna(""), baseline_df["price"].fillna(0))
    )
    remainder_mask = ~pd.Series(
        list(zip(full_df[key_col].fillna(""), full_df["price"].fillna(0)))
    ).apply(lambda x: x in baseline_keys)

    remainder_df = full_df[remainder_mask].reset_index(drop=True)
    n_extra = n - n_baseline

    if len(remainder_df) <= n_extra:
        extra = remainder_df
    else:
        extra = remainder_df.sample(n=n_extra, random_state=seed)

    result = pd.concat([baseline_df, extra], ignore_index=True)
    print(f"  {category}: {n_baseline} baseline + {len(extra)} extra = {len(result)} items (of {len(full_df)} total)")
    return result


def share_targets(share: float, baseline_dir: Path) -> dict[str, int]:
    """Per-category target sizes for proportional (--share) mode.

    n_c = max(n_baseline_c, ceil(share * pool_c))
    """
    targets = {}
    for category in INTERESTING_CATEGORIES:
        pool_n = _count_rows(Path(INPUT_DIR) / category)
        baseline_n = _count_rows(Path(baseline_dir) / category)
        target = max(baseline_n, math.ceil(share * pool_n))
        targets[category] = target
        print(f"  {category}: pool {pool_n}, baseline {baseline_n} -> target {target}")
    return targets


def _share_tag(share: float) -> str:
    """Directory suffix for a share value, e.g. 0.5 -> '0p5'."""
    return f"{share:g}".replace(".", "p")


@click.command()
@click.option("--items-per-category", type=int, default=None,
              help="Flat mode: number of items to sample per category.")
@click.option("--share", type=float, default=None,
              help="Proportional mode: fraction of each category's pool to sample "
                   "(floored at the baseline size).")
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option("--baseline-dir", type=click.Path(path_type=Path), default=Path(BASELINE_DIR),
              show_default=True,
              help="Baseline catalog that the output must be a superset of.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Override output directory. Default: ...merged_{N}k_each/ (flat mode) "
                   "or ...merged_cat{share_tag}/ (share mode)")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing category files.")
def main(items_per_category: int | None, share: float | None, seed: int,
         baseline_dir: Path, output_dir: Path | None, force: bool):
    if (items_per_category is None) == (share is None):
        raise click.BadParameter(
            "give exactly one of --items-per-category or --share")

    if items_per_category is not None and items_per_category <= 0:
        raise click.BadParameter("--items-per-category must be positive")
    if share is not None and not (0 < share <= 1):
        raise click.BadParameter("--share must be in (0, 1]")

    baseline_dir = Path(os.path.expanduser(str(baseline_dir)))
    if output_dir is not None:
        output_dir = Path(os.path.expanduser(str(output_dir)))

    if share is not None:
        mode = f"share={share}"
        if output_dir is None:
            output_dir = Path(BASE_DIR) / (
                f"meta_parquet_clean_attributes_keywords_embed_merged_cat{_share_tag(share)}")
    else:
        mode = f"items-per-category={items_per_category}"
        nk = items_per_category // 1000
        if output_dir is None:
            output_dir = Path(BASE_DIR) / f"meta_parquet_clean_attributes_keywords_embed_merged_{nk}k_each"

    print(f"Input (full):     {INPUT_DIR}")
    print(f"Input (baseline): {baseline_dir}")
    print(f"Output:           {output_dir}")
    print(f"Mode: {mode}, seed: {seed}")
    print()

    if share is not None:
        print("Per-category targets:")
        targets = share_targets(share, baseline_dir)
        print(f"  planned total: {sum(targets.values())}")
        print()
    else:
        targets = {c: items_per_category for c in INTERESTING_CATEGORIES}

    total = 0
    for category in INTERESTING_CATEGORIES:
        out_cat_dir = output_dir / category
        out_file = out_cat_dir / "batch_000000.parquet"

        if out_file.exists() and not force:
            n = pq.ParquetFile(out_file).metadata.num_rows
            print(f"  {category}: already exists ({n} rows), skipping")
            total += n
            continue

        df = sample_category(category, targets[category], seed, baseline_dir)
        out_cat_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_file, engine="pyarrow", index=False, compression="snappy")
        total += len(df)

    print(f"\nDone. Total items: {total}")
    print(f"Catalog directory: {output_dir}")


if __name__ == "__main__":
    main()
