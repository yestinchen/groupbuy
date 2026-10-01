"""
Merge keyword-enriched parquet batches with their embedding batches.

This is the compact end product for the q6b item catalog:
- read the keyword parquet batches
- read the matching embedding parquet batches
- keep only the columns needed for retrieval and inspection
- write one parquet file per input parquet file

To keep memory usage low, we only load:
- from the keyword file: `average_rating`, `price`, `main_category`, `store`,
  `rating_number`, `categories`, `title_feats`, `title_for_embedding`,
  `keywords`
- from the embedding file: `title_embedding`

All other wide source columns such as `details`, `features`, and `title`
stay out of the merged output.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from gquery.prep.amazon.constants import INTERESTING_CATEGORIES
except ImportError:
    try:
        from constants import INTERESTING_CATEGORIES
    except ImportError:
        INTERESTING_CATEGORIES = []


def get_batch_number(filename: str) -> int:
    try:
        if "_b_" in filename:
            parts = filename.split("_b_")
            if len(parts) > 1:
                return int(parts[1].split(".")[0])
        if filename.startswith("batch_"):
            return int(filename.split("_")[1].split(".")[0])
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return int(numbers[0])
        return 0
    except Exception:
        return 0


def _existing_columns(file: Path) -> set[str]:
    return set(pq.read_schema(file).names)


def _read_keyword_table(file: Path, max_rows_per_file: int | None = None) -> pa.Table:
    wanted = [
        "average_rating",
        "price",
        "main_category",
        "store",
        "rating_number",
        "categories",
        "title_feats",
        "title_for_embedding",
        "keywords",
    ]
    existing = _existing_columns(file)
    missing = [col for col in wanted if col not in existing]
    if missing:
        raise ValueError(f"{file} is missing required columns: {missing}")
    table = pq.read_table(file, columns=wanted)
    if max_rows_per_file is not None and max_rows_per_file > 0:
        table = table.slice(0, max_rows_per_file)
    return table


def _read_embedding_table(file: Path, max_rows_per_file: int | None = None) -> pa.Table:
    existing = _existing_columns(file)
    if "title_embedding" not in existing:
        raise ValueError(f"{file} is missing required column: title_embedding")
    table = pq.read_table(file, columns=["title_embedding"])
    if max_rows_per_file is not None and max_rows_per_file > 0:
        table = table.slice(0, max_rows_per_file)
    return table


def _valid_row_mask(table: pa.Table) -> pa.Array:
    price = table["price"]
    average_rating = table["average_rating"]
    valid_price = pc.fill_null(pc.greater(price, 0), False)
    valid_rating = pc.fill_null(pc.greater(average_rating, 0), False)
    return pc.and_(valid_price, valid_rating)


def _merge_tables(
    keyword_table: pa.Table,
    embedding_table: pa.Table,
) -> pa.Table:
    if len(keyword_table) != len(embedding_table):
        raise ValueError(
            "Keyword and embedding batch sizes differ: "
            f"{len(keyword_table)} vs {len(embedding_table)}"
        )

    arrays = [
        keyword_table["average_rating"],
        keyword_table["price"],
        keyword_table["main_category"],
        keyword_table["store"],
        keyword_table["rating_number"],
        keyword_table["categories"],
        keyword_table["title_feats"],
        keyword_table["title_for_embedding"],
        keyword_table["keywords"],
        embedding_table["title_embedding"],
    ]
    names = [
        "average_rating",
        "price",
        "main_category",
        "store",
        "rating_number",
        "categories",
        "title_feats",
        "title_for_embedding",
        "keywords",
        "title_embedding",
    ]
    return pa.Table.from_arrays(arrays, names=names)


def process_category(
    input_root: Path,
    embed_root: Path,
    output_root: Path,
    category: str,
    max_rows_per_file: int | None = None,
) -> int:
    input_dir = input_root / category
    embed_dir = embed_root / category
    if not input_dir.exists():
        print(f"  Skipping missing input category: {input_dir}")
        return 0
    if not embed_dir.exists():
        print(f"  Skipping missing embed category: {embed_dir}")
        return 0

    input_files = sorted(input_dir.glob("*.parquet"), key=lambda f: get_batch_number(f.name))
    if not input_files:
        print(f"  Warning: no parquet files found in {input_dir}")
        return 0

    total_rows = 0

    for input_file in tqdm(input_files, desc=f"Category {category}"):
        embed_file = embed_dir / input_file.name
        if not embed_file.exists():
            raise FileNotFoundError(f"Missing embedding file for {input_file.name}: {embed_file}")

        output_dir = output_root / category
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / input_file.name
        if output_file.exists():
            print(f"  Skipping {input_file.name}: output file already exists: {output_file}")
            continue

        keyword_table = _read_keyword_table(input_file, max_rows_per_file=max_rows_per_file)
        embed_table = _read_embedding_table(embed_file, max_rows_per_file=max_rows_per_file)

        if len(keyword_table) != len(embed_table):
            raise ValueError(
                f"Row count mismatch for {input_file.name}: "
                f"{len(keyword_table)} keyword rows vs {len(embed_table)} embedding rows"
            )

        mask = _valid_row_mask(keyword_table)
        keyword_table = keyword_table.filter(mask)
        embed_table = embed_table.filter(mask)
        merged_table = _merge_tables(keyword_table, embed_table)
        merged_table = merged_table.select([
            "average_rating",
            "price",
            "main_category",
            "store",
            "rating_number",
            "categories",
            "title_feats",
            "title_for_embedding",
            "keywords",
            "title_embedding",
        ])
        pq.write_table(merged_table, output_file, compression="snappy")
        total_rows += len(merged_table)
        print(f"  Wrote {len(merged_table)} rows to {output_file}")

    return total_rows


def _default_root() -> str:
    root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
    return os.path.join(root, "amazon_review_2023")


@click.command()
@click.option(
    "--input-root",
    "--titled-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root folder containing category subfolders from meta_parquet_clean_attributes_keywords",
)
@click.option(
    "--embed-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root folder containing category subfolders from meta_parquet_clean_attributes_titled_embed",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Output root; one parquet file per category will be written here",
)
@click.option(
    "--category",
    type=str,
    default=None,
    help="Single category name (e.g. Baby_Products), or None for all INTERESTING_CATEGORIES",
)
@click.option(
    "--max-rows-per-file",
    type=int,
    default=None,
    help="Optional cap for rows per parquet batch, useful for quick validation",
)
def main(
    input_root: Path | None,
    embed_root: Path | None,
    output_root: Path | None,
    category: str | None,
    max_rows_per_file: int | None,
) -> None:
    if input_root is None:
        input_root = Path(_default_root()) / "meta_parquet_clean_attributes_keywords"
    if embed_root is None:
        embed_root = Path(_default_root()) / "meta_parquet_clean_attributes_titled_embed"
    if output_root is None:
        output_root = Path(_default_root()) / "meta_parquet_clean_attributes_keywords_embed_merged"

    if category is None or category == "None":
        categories = INTERESTING_CATEGORIES
    else:
        categories = [category]

    print(f"Input root:  {input_root}")
    print(f"Embed root:  {embed_root}")
    print(f"Output root: {output_root}")
    if max_rows_per_file is not None and max_rows_per_file > 0:
        print(f"Max rows per file: {max_rows_per_file}")
    print()

    total = 0
    for cat in categories:
        total += process_category(
            input_root=input_root,
            embed_root=embed_root,
            output_root=output_root,
            category=cat,
            max_rows_per_file=max_rows_per_file,
        )

    print()
    print(f"Done. Total rows written: {total}")


if __name__ == "__main__":
    main()
