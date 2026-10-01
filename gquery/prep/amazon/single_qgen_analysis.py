from __future__ import annotations

"""
Compute embedding-based price/rating statistics for generated single queries.

Input:
- query parquets from ~/dataset/amazon_review_2023/data_out/qgen/keyword_w_title/
- same-category item catalog parquets from
  ~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged/

For each query, compute cosine similarity against items in the same category,
then summarize matched items above several similarity thresholds.

The analysis writes one parquet per category per threshold:
  ~/dataset/amazon_review_2023/analysis/embedding_price_distribution/
    threshold_<threshold>/<category>.parquet
"""

import re
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
from pyarrow import parquet as pq
from tqdm import tqdm


DEFAULT_QUERY_ROOT = Path("~/dataset/amazon_review_2023/data_out/qgen/keyword_w_title")
DEFAULT_ITEM_ROOT = Path("~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged")
DEFAULT_OUTPUT_ROOT = Path("~/dataset/amazon_review_2023/analysis/embedding_price_distribution")
DEFAULT_THRESHOLDS = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def _get_batch_number(filename: str) -> int:
    try:
        if "_b_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                return int(parts[1].split(".")[0])
        if filename.startswith("batch_"):
            return int(filename.split("_")[1].split(".")[0])
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return int(numbers[0])
        return 0
    except Exception:
        return 0


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _to_1d_array(value) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _format_threshold(threshold: float) -> str:
    return format(threshold, "g")


def _read_query_table(query_file: Path, max_queries: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(query_file, engine="pyarrow")
    required = {"keyword", "query_desc", "query_desc_embedding"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{query_file} is missing required column(s): {sorted(missing)}")
    df = df.loc[:, ["keyword", "query_desc", "query_desc_embedding"]].copy()
    df["keyword"] = df["keyword"].map(_normalize_text)
    df["query_desc"] = df["query_desc"].map(_normalize_text)
    if max_queries is not None and max_queries > 0 and len(df) > max_queries:
        df = df.head(max_queries).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


@dataclass(frozen=True)
class ItemCatalog:
    embeddings: np.ndarray
    prices: np.ndarray
    ratings: np.ndarray
    item_count: int


def _load_category_catalog(category_dir: Path) -> ItemCatalog:
    files = sorted(category_dir.glob("*.parquet"), key=lambda p: _get_batch_number(p.name))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {category_dir}")

    emb_rows: list[np.ndarray] = []
    price_rows: list[np.ndarray] = []
    rating_rows: list[np.ndarray] = []

    for file in files:
        table = pq.read_table(file, columns=["price", "average_rating", "title_embedding"])
        df = table.to_pandas()
        if df.empty:
            continue

        prices = pd.to_numeric(df["price"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        ratings = pd.to_numeric(df["average_rating"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        embeddings = [_to_1d_array(value) for value in df["title_embedding"].tolist()]

        valid_rows = [idx for idx, emb in enumerate(embeddings) if emb is not None]
        if not valid_rows:
            continue

        emb_rows.extend(embeddings[idx] for idx in valid_rows if embeddings[idx] is not None)
        price_rows.append(prices[valid_rows])
        rating_rows.append(ratings[valid_rows])

    if not emb_rows:
        raise ValueError(f"No valid item embeddings found in {category_dir}")

    item_embeddings = np.vstack(emb_rows).astype(np.float32, copy=False)
    item_embeddings = _normalize_rows(item_embeddings)
    item_prices = np.concatenate(price_rows).astype(np.float32, copy=False)
    item_ratings = np.concatenate(rating_rows).astype(np.float32, copy=False)
    return ItemCatalog(
        embeddings=item_embeddings,
        prices=item_prices,
        ratings=item_ratings,
        item_count=int(len(item_prices)),
    )


def _extract_query_embeddings(df: pd.DataFrame) -> np.ndarray:
    embeddings = []
    for value in df["query_desc_embedding"].tolist():
        arr = _to_1d_array(value)
        if arr is None:
            raise ValueError("Query file contains an invalid query_desc_embedding")
        embeddings.append(arr)
    query_embeddings = np.vstack(embeddings).astype(np.float32, copy=False)
    return _normalize_rows(query_embeddings)


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "item_count": 0,
            "price_min": np.nan,
            "price_max": np.nan,
            "price_mean": np.nan,
            "price_median": np.nan,
            "price_std": np.nan,
            "price_q25": np.nan,
            "price_q75": np.nan,
        }

    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            "item_count": 0,
            "price_min": np.nan,
            "price_max": np.nan,
            "price_mean": np.nan,
            "price_median": np.nan,
            "price_std": np.nan,
            "price_q25": np.nan,
            "price_q75": np.nan,
        }

    return {
        "item_count": int(clean.size),
        "price_min": float(np.min(clean)),
        "price_max": float(np.max(clean)),
        "price_mean": float(np.mean(clean)),
        "price_median": float(np.median(clean)),
        "price_std": float(np.std(clean, ddof=0)),
        "price_q25": float(np.quantile(clean, 0.25)),
        "price_q75": float(np.quantile(clean, 0.75)),
    }


def _summarize_matches(mask: np.ndarray, prices: np.ndarray, ratings: np.ndarray) -> dict[str, float]:
    matched_prices = prices[mask]
    matched_ratings = ratings[mask]
    price_stats = _summary_stats(matched_prices)
    rating_stats = _summary_stats(matched_ratings)
    return {
        **price_stats,
        "rating_min": rating_stats["price_min"],
        "rating_max": rating_stats["price_max"],
        "rating_mean": rating_stats["price_mean"],
        "rating_median": rating_stats["price_median"],
        "rating_std": rating_stats["price_std"],
        "rating_q25": rating_stats["price_q25"],
        "rating_q75": rating_stats["price_q75"],
    }


def _build_output_rows(
    category: str,
    query_df: pd.DataFrame,
    similarities: np.ndarray,
    catalog: ItemCatalog,
    thresholds: tuple[float, ...],
) -> dict[float, pd.DataFrame]:
    rows_by_threshold: dict[float, list[dict]] = {threshold: [] for threshold in thresholds}

    for row_idx, row in query_df.iterrows():
        row_sims = similarities[row_idx]
        for threshold in thresholds:
            mask = row_sims >= threshold
            stats = _summarize_matches(mask, catalog.prices, catalog.ratings)
            rows_by_threshold[threshold].append(
                {
                    "category": category,
                    "keyword": row["keyword"],
                    "query_desc": row["query_desc"],
                    "threshold": float(threshold),
                    **stats,
                }
            )

    return {threshold: pd.DataFrame(rows) for threshold, rows in rows_by_threshold.items()}


def process_category(
    category: str,
    query_root: Path,
    item_root: Path,
    output_root: Path,
    thresholds: tuple[float, ...],
    max_queries: int | None = None,
) -> None:
    query_file = query_root / f"{category}.parquet"
    category_item_dir = item_root / category
    if not query_file.exists():
        print(f"Skipping missing query file: {query_file}")
        return
    if not category_item_dir.exists():
        print(f"Skipping missing item folder: {category_item_dir}")
        return

    print(f"Loading queries for {category}...")
    query_df = _read_query_table(query_file, max_queries=max_queries)
    if query_df.empty:
        print(f"Skipping {category}: no query rows loaded")
        return

    print(f"Loading catalog for {category}...")
    catalog = _load_category_catalog(category_item_dir)

    query_embeddings = _extract_query_embeddings(query_df)
    if query_embeddings.shape[1] != catalog.embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch for {category}: "
            f"queries={query_embeddings.shape[1]}, items={catalog.embeddings.shape[1]}"
        )

    print(
        f"Computing cosine similarity for {category}: "
        f"{len(query_df)} queries x {catalog.item_count} items"
    )
    similarities = np.matmul(query_embeddings, catalog.embeddings.T)

    rows_by_threshold = _build_output_rows(category, query_df, similarities, catalog, thresholds)

    for threshold, out_df in rows_by_threshold.items():
        threshold_dir = output_root / f"threshold_{_format_threshold(threshold)}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        output_file = threshold_dir / f"{category}.parquet"
        out_df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"  Wrote {len(out_df)} rows to {output_file}")


def _parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = []
    for token in value.split(","):
        text = token.strip()
        if not text:
            continue
        thresholds.append(float(text))
    if not thresholds:
        raise ValueError("No thresholds provided")
    return tuple(thresholds)


def _discover_categories(query_root: Path) -> list[str]:
    files = sorted(query_root.glob("*.parquet"))
    return [file.stem for file in files]


@click.command()
@click.option(
    "--query-root",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_QUERY_ROOT,
    show_default=True,
    help="Folder containing category-level query parquet files.",
)
@click.option(
    "--item-root",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_ITEM_ROOT,
    show_default=True,
    help="Folder containing same-category item parquet batches.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_ROOT,
    show_default=True,
    help="Analysis output root.",
)
@click.option(
    "--thresholds",
    type=str,
    default="0.7,0.75,0.8,0.85,0.9,0.95",
    show_default=True,
    help="Comma-separated similarity thresholds.",
)
@click.option(
    "--category",
    type=str,
    default=None,
    help="Single category to process. Default: process all query files in query-root.",
)
@click.option(
    "--max-queries",
    type=int,
    default=None,
    help="Optional cap on the number of query rows per category (useful for fast sanity checks).",
)
def main(
    query_root: Path,
    item_root: Path,
    output_root: Path,
    thresholds: str,
    category: str | None,
    max_queries: int | None,
) -> None:
    threshold_values = _parse_thresholds(thresholds)
    categories = [category] if category else _discover_categories(query_root)
    if not categories:
        raise click.ClickException(f"No query parquet files found in {query_root}")

    print(f"Query root: {query_root}")
    print(f"Item root:  {item_root}")
    print(f"Output root: {output_root}")
    print(f"Thresholds: {', '.join(_format_threshold(t) for t in threshold_values)}")
    if max_queries is not None:
        print(f"Max queries: {max_queries}")
    print(f"Categories:  {' '.join(categories)}")
    print()

    for cat in tqdm(categories, desc="Categories"):
        process_category(
            category=cat,
            query_root=query_root,
            item_root=item_root,
            output_root=output_root,
            thresholds=threshold_values,
            max_queries=max_queries,
        )


if __name__ == "__main__":
    main()
