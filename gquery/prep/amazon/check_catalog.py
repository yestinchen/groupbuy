#!/usr/bin/env python3
"""
Sanity gates for a sampled / symlinked item catalog.

Walks every INTERESTING_CATEGORIES directory of --catalog-dir (symlinks are
followed, so a catalog may be a directory of links into the full pool) and
checks, reading parquet column-by-column so that memory stays flat:

  (a) row count                                            [informational]
  (b) zero nulls in price / average_rating / title_embedding      [HARD GATE]
  (c) title_embedding values have length 1024 (min/max per file)   [HARD GATE]
  (d) duplicate item ids within and across categories      [informational]
  (e) with --baseline-dir: nestedness, i.e. every
      (title_for_embedding, price) key of the baseline category
      also appears in the catalog category                       [HARD GATE]

Item id for (d) is `parent_asin` when the schema has it, otherwise the
(title_for_embedding, price) composite key used by sample_catalog_nk.py.

Exit code is nonzero if any hard gate fails.

Usage:
  python -m gquery.prep.amazon.check_catalog \
      --catalog-dir ~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_cat1m \
      --baseline-dir ~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_30k_each
  python -m gquery.prep.amazon.check_catalog \
      --catalog-dir ~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_cat2m
"""

import os
import sys
from collections import Counter
from pathlib import Path

import click
import pyarrow.compute as pc
from pyarrow import parquet as pq

from gquery.prep.amazon.constants import INTERESTING_CATEGORIES

DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
BASE_DIR = os.path.join(DATASET_ROOT, "amazon_review_2023")

EMBED_COL = "title_embedding"
EMBED_DIM = 1024
NULL_CHECK_COLS = ["price", "average_rating", EMBED_COL]
KEY_COLS = ["title_for_embedding", "price"]


def _parquet_files(directory: Path) -> list[Path]:
    """Parquet files of a category directory (symlinked dirs are followed)."""
    return sorted(Path(directory).resolve().glob("*.parquet"))


def _composite_keys(files: list[Path]) -> set:
    """(title_for_embedding, price) keys of a set of parquet files."""
    keys = set()
    for f in files:
        table = pq.read_table(f, columns=KEY_COLS)
        titles = table.column("title_for_embedding").to_pylist()
        prices = table.column("price").to_pylist()
        keys.update(
            zip(("" if t is None else t for t in titles),
                (0 if p is None else p for p in prices))
        )
        del table, titles, prices
    return keys


def check_category(category: str, cat_dir: Path):
    """Run the per-category gates. Returns (stats dict, list of failures)."""
    failures = []
    files = _parquet_files(cat_dir)
    if not files:
        failures.append(f"{category}: no parquet files in {cat_dir}")
        return {"rows": 0, "ids": Counter()}, failures

    n_rows = 0
    nulls = {c: 0 for c in NULL_CHECK_COLS}
    ids = Counter()
    id_col = None

    for f in files:
        pf = pq.ParquetFile(f)
        n_rows += pf.metadata.num_rows
        names = pf.schema_arrow.names
        if id_col is None:
            id_col = "parent_asin" if "parent_asin" in names else None

        # (b) nulls, one column at a time
        for col in NULL_CHECK_COLS:
            if col not in names:
                failures.append(f"{category}: {f.name} is missing column '{col}'")
                continue
            arr = pq.read_table(f, columns=[col]).column(col)
            nulls[col] += arr.null_count

            # (c) embedding dimension: vectorized min/max over the file's
            # non-null values (cheap, and stronger than a single spot check)
            if col == EMBED_COL:
                lengths = pc.min_max(pc.list_value_length(arr)).as_py()
                lo, hi = lengths["min"], lengths["max"]
                if lo is None:
                    failures.append(f"{category}: {f.name} has no non-null {EMBED_COL}")
                elif lo != EMBED_DIM or hi != EMBED_DIM:
                    failures.append(
                        f"{category}: {f.name} {EMBED_COL} lengths in [{lo}, {hi}], "
                        f"expected {EMBED_DIM}")
                del lengths
            del arr

        # (d) item ids
        if id_col is not None:
            ids.update(pq.read_table(f, columns=[id_col]).column(id_col).to_pylist())
        else:
            table = pq.read_table(f, columns=KEY_COLS)
            titles = table.column("title_for_embedding").to_pylist()
            prices = table.column("price").to_pylist()
            ids.update(
                zip(("" if t is None else t for t in titles),
                    (0 if p is None else p for p in prices))
            )
            del table, titles, prices

    for col, n in nulls.items():
        if n:
            failures.append(f"{category}: {n} null values in '{col}'")

    dup_within = sum(c - 1 for c in ids.values() if c > 1)
    print(f"  {category}: {n_rows} rows, {len(files)} files, "
          f"nulls {nulls}, dup ids within category: {dup_within}")

    return {"rows": n_rows, "ids": ids, "dup_within": dup_within,
            "id_col": id_col or "(title_for_embedding, price)"}, failures


@click.command()
@click.option("--catalog-dir", type=click.Path(path_type=Path), required=True,
              help="Catalog directory to check (one subdirectory per category).")
@click.option("--baseline-dir", type=click.Path(path_type=Path), default=None,
              help="If given, verify the catalog is a superset of this baseline.")
def main(catalog_dir: Path, baseline_dir: Path | None):
    catalog_dir = Path(os.path.expanduser(str(catalog_dir)))
    if baseline_dir is not None:
        baseline_dir = Path(os.path.expanduser(str(baseline_dir)))

    print(f"Catalog:  {catalog_dir}")
    print(f"Baseline: {baseline_dir}")
    print()

    all_failures = []
    total_rows = 0
    per_category = {}
    global_ids = Counter()
    id_col_used = None
    dup_within_total = 0

    print("Per-category checks:")
    for category in INTERESTING_CATEGORIES:
        cat_dir = catalog_dir / category
        if not cat_dir.exists():
            all_failures.append(f"{category}: missing directory {cat_dir}")
            print(f"  {category}: MISSING")
            continue
        stats, failures = check_category(category, cat_dir)
        all_failures.extend(failures)
        total_rows += stats["rows"]
        per_category[category] = stats["rows"]
        dup_within_total += stats.get("dup_within", 0)
        id_col_used = stats.get("id_col", id_col_used)
        global_ids.update(stats["ids"])
        del stats

    print()
    print(f"Total rows: {total_rows}")
    print(f"Categories found: {len(per_category)} / {len(INTERESTING_CATEGORIES)}")

    # (d) duplicates across the whole catalog
    dup_global = sum(c - 1 for c in global_ids.values() if c > 1)
    dup_keys = sum(1 for c in global_ids.values() if c > 1)
    print(f"Duplicate item ids (key: {id_col_used}): "
          f"{dup_global} extra rows over {dup_keys} repeated ids "
          f"({dup_within_total} within categories, "
          f"{dup_global - dup_within_total} across categories) (informational)")

    # (e) nestedness
    if baseline_dir is not None:
        print()
        print("Nestedness (baseline subset of catalog):")
        for category in INTERESTING_CATEGORIES:
            base_files = _parquet_files(baseline_dir / category)
            cat_files = _parquet_files(catalog_dir / category)
            if not base_files:
                all_failures.append(f"{category}: no baseline parquet files")
                continue
            base_keys = _composite_keys(base_files)
            cat_keys = _composite_keys(cat_files)
            missing = base_keys - cat_keys
            status = "OK" if not missing else f"FAIL ({len(missing)} missing)"
            print(f"  {category}: {len(base_keys)} baseline keys, "
                  f"{len(cat_keys)} catalog keys -> {status}")
            if missing:
                all_failures.append(
                    f"{category}: {len(missing)} baseline keys missing from catalog")
            del base_keys, cat_keys, missing

    print()
    if all_failures:
        print(f"FAILED ({len(all_failures)} hard-gate violations):")
        for msg in all_failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("All hard gates passed.")


if __name__ == "__main__":
    main()
