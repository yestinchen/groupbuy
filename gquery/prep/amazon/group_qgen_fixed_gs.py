"""
Generate e10-style grouped queries with a fixed group_size (number of requirements).

Fork of group_qgen_e10.py with one key change: --group-size sets a fixed size
instead of sampling uniformly from [3, 8]. This produces query sets where every
query has exactly N requirements, for scalability studies over query length.

Usage:
  python -m gquery.prep.amazon.group_qgen_fixed_gs --group-size 4
  python -m gquery.prep.amazon.group_qgen_fixed_gs --group-size 10
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Iterable
import sys

import click
import numpy as np
import pandas as pd

from gquery.prep.amazon.constants import INTERESTING_CATEGORIES


DATASET_ROOT = Path("~/dataset/amazon_review_2023").expanduser()

DEFAULT_QUERY_ROOT    = DATASET_ROOT / "data_out/qgen/keyword_w_title"
DEFAULT_ANALYSIS_ROOT = DATASET_ROOT / "analysis/embedding_price_distribution_10k_each"
DEFAULT_OUTPUT_ROOT   = DATASET_ROOT / "data_out/qgen/group_keyword_w_title_qgen11_fixed_gs"
DEFAULT_THRESHOLD       = 0.8
DEFAULT_TOTAL_NUM       = 1000
DEFAULT_SEED            = 42
DEFAULT_MULTIPLIER_LOW       = 0.8
DEFAULT_MULTIPLIER_HIGH      = 0.99
DEFAULT_EASY_MULTIPLIER_LOW  = 1.1
DEFAULT_EASY_MULTIPLIER_HIGH = 1.5
DEFAULT_WITHIN_BUDGET_FRACTION = 0.6


CATEGORY_ABBR = {
    "All_Beauty": "ab",
    "Amazon_Fashion": "af",
    "Appliances": "ap",
    "Arts_Crafts_and_Sewing": "acs",
    "Baby_Products": "bp",
    "Cell_Phones_and_Accessories": "cpa",
    "Electronics": "el",
    "Health_and_Personal_Care": "hpc",
    "Pet_Supplies": "ps",
    "Sports_and_Outdoors": "so",
    "Toys_and_Games": "tg",
}


# ---------------------------------------------------------------------------
# Small helpers (unchanged from group_qgen_e10.py)
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _normalize_text(value) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _ensure_list(value):
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


def _normalize_embedding(value) -> list[float]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.tolist()
    return [float(value)]


def _fmt_threshold(value: float) -> str:
    return f"{value:g}"


def _category_abbr(category: str) -> str:
    if category in CATEGORY_ABBR:
        return CATEGORY_ABBR[category]
    parts = [p for p in category.split("_") if p]
    return "".join(p[0].lower() for p in parts) if parts else category.lower()


def _canonical_categories(categories: Iterable[str]) -> list[str]:
    category_set = {str(c) for c in categories}
    seen: set[str] = set()
    ordered: list[str] = []
    for cat in INTERESTING_CATEGORIES:
        if cat in category_set:
            ordered.append(cat)
            seen.add(cat)
    for cat in categories:
        cat = str(cat)
        if cat not in seen and cat in category_set:
            ordered.append(cat)
            seen.add(cat)
    return ordered


def _parse_categories(raw: str | None) -> list[str]:
    if not raw or not raw.strip():
        return list(INTERESTING_CATEGORIES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise click.BadParameter("--categories must contain at least one category")
    return _canonical_categories(parts)


def _mix_stem(categories: list[str]) -> str:
    return "_".join(_category_abbr(c) for c in _canonical_categories(categories))


# ---------------------------------------------------------------------------
# Weighted sampling without replacement
# ---------------------------------------------------------------------------

def _weighted_without_replacement(
    items: list[str],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[str]:
    if k <= 0 or not items:
        return []
    pool = [(item, max(0.0, float(w))) for item, w in zip(items, weights)]
    chosen: list[str] = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        if total <= 0:
            idx = rng.randrange(len(pool))
        else:
            target = rng.random() * total
            acc = 0.0
            idx = len(pool) - 1
            for pos, (_, w) in enumerate(pool):
                acc += w
                if acc >= target:
                    idx = pos
                    break
        chosen.append(pool.pop(idx)[0])
    return chosen


# ---------------------------------------------------------------------------
# Data loading (unchanged)
# ---------------------------------------------------------------------------

def _load_category_tables(
    query_root: Path,
    analysis_root: Path,
    categories: list[str],
    threshold: float,
) -> pd.DataFrame:
    threshold_dir = analysis_root / f"threshold_{_fmt_threshold(threshold)}"
    if not threshold_dir.exists():
        raise FileNotFoundError(
            f"Analysis threshold directory not found: {threshold_dir}\n"
            "Run run_analysis_e10.sh first to generate price stats for SAMPLE_10k_EACH."
        )

    frames: list[pd.DataFrame] = []
    required_query_cols = {"keyword", "query_desc", "query_desc_embedding"}
    required_analysis_cols = {
        "category", "keyword", "query_desc", "item_count",
        "price_min", "price_max", "price_mean",
        "price_median", "price_q25", "price_q75",
    }

    for category in categories:
        query_file    = query_root / f"{category}.parquet"
        analysis_file = threshold_dir / f"{category}.parquet"
        if not query_file.exists():
            raise FileNotFoundError(f"Missing query parquet: {query_file}")
        if not analysis_file.exists():
            raise FileNotFoundError(f"Missing analysis parquet: {analysis_file}")

        query_df = pd.read_parquet(query_file)
        if missing := required_query_cols - set(query_df.columns):
            raise ValueError(f"{query_file} missing columns: {sorted(missing)}")
        query_df = query_df.copy()
        query_df["category"]             = category
        query_df["keyword"]              = query_df["keyword"].map(_normalize_text)
        query_df["query_desc"]           = query_df["query_desc"].map(_normalize_text)
        query_df["query_desc_embedding"] = query_df["query_desc_embedding"].map(_normalize_embedding)

        analysis_df = pd.read_parquet(analysis_file)
        if missing := required_analysis_cols - set(analysis_df.columns):
            raise ValueError(f"{analysis_file} missing columns: {sorted(missing)}")
        analysis_df = analysis_df.copy()
        analysis_df["category"]   = analysis_df["category"].map(_normalize_text)
        analysis_df["keyword"]    = analysis_df["keyword"].map(_normalize_text)
        analysis_df["query_desc"] = analysis_df["query_desc"].map(_normalize_text)
        analysis_df["item_count"] = analysis_df["item_count"].fillna(0).astype(int)

        merged = query_df.merge(
            analysis_df,
            on=["category", "keyword", "query_desc"],
            how="inner",
            validate="one_to_one",
        )
        merged = merged[merged["item_count"] > 0].reset_index(drop=True)
        if not merged.empty:
            frames.append(merged)

    if not frames:
        raise ValueError("No usable rows found after loading the selected categories")

    pool = pd.concat(frames, ignore_index=True)
    pool["query_desc_embedding"] = pool["query_desc_embedding"].map(_normalize_embedding)
    pool["query_len"] = pool["query_desc"].map(lambda x: len(_normalize_text(x).split()))
    return pool


# ---------------------------------------------------------------------------
# Price floor
# ---------------------------------------------------------------------------

def _get_min_price(row: pd.Series) -> float:
    candidates = [row.get("price_min"), row.get("price_q25"), row.get("price_median")]
    val = next((float(v) for v in candidates if not _is_missing(v) and math.isfinite(float(v))), 1.0)
    return max(0.01, val)


# ---------------------------------------------------------------------------
# Group construction -- fixed group_size
# ---------------------------------------------------------------------------

def _sample_rows_from_category(
    category_df: pd.DataFrame,
    count: int,
    rng: random.Random,
) -> pd.DataFrame:
    if count <= 0 or category_df.empty:
        return category_df.iloc[0:0].copy()
    replace = count > len(category_df)
    weights = category_df["item_count"].astype(float).clip(lower=0.0)
    if float(weights.sum()) <= 0:
        weights = None
    return category_df.sample(
        n=count, replace=replace, weights=weights,
        random_state=rng.randint(0, 2**32 - 1),
    ).reset_index(drop=True)


def _build_group_rows(
    pool: pd.DataFrame,
    categories: list[str],
    group_size: int,
    total_num: int,
    seed: int,
    threshold: float,
    multiplier_low: float,
    multiplier_high: float,
    easy_multiplier_low: float,
    easy_multiplier_high: float,
    within_budget_fraction: float,
) -> pd.DataFrame:
    rng = random.Random(seed)
    cat_to_df  = {cat: pool[pool["category"] == cat].reset_index(drop=True) for cat in categories}
    cat_weights = [
        float(math.log1p(max(1.0, pool.loc[pool["category"] == cat, "item_count"].astype(float).sum())))
        for cat in categories
    ]
    mix_name = _mix_stem(categories)

    rows: list[dict] = []
    for group_id in range(total_num):
        if len(categories) == 1:
            chosen_categories = [categories[0]]
        else:
            min_cats = 2 if group_size <= 4 else 3
            max_cats = min(group_size, len(categories))
            if min_cats > max_cats:
                min_cats = max_cats
            num_cats = rng.randint(min_cats, max_cats)
            chosen_categories = _weighted_without_replacement(categories, cat_weights, num_cats, rng)
            if not chosen_categories:
                chosen_categories = [rng.choice(categories)]

        ordered = chosen_categories[:]
        rng.shuffle(ordered)
        if ordered:
            rotate = rng.randrange(len(ordered))
            ordered = ordered[rotate:] + ordered[:rotate]

        sequence: list[str] = []
        while len(sequence) < group_size:
            sequence.extend(ordered)
        sequence = sequence[:group_size]

        cat_counts: dict[str, int] = {}
        for cat in sequence:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        sampled_by_cat = {cat: _sample_rows_from_category(cat_to_df[cat], cnt, rng)
                          for cat, cnt in cat_counts.items()}
        cat_offsets = {cat: 0 for cat in cat_counts}

        query_descs:   list[str]         = []
        query_embs:    list[list[float]] = []
        min_prices:    list[float]       = []
        cats_in_group: list[str]         = []

        for cat in sequence:
            df     = sampled_by_cat[cat]
            offset = cat_offsets[cat]
            row    = df.iloc[min(offset, len(df) - 1)]
            cat_offsets[cat] = offset + 1

            query_descs.append(_normalize_text(row["query_desc"]))
            query_embs.append(_normalize_embedding(row["query_desc_embedding"]))
            min_prices.append(_get_min_price(row))
            cats_in_group.append(cat)

        min_feasible_cost = sum(min_prices)
        gt_budget = int(round(min_feasible_cost))

        easy = rng.random() < within_budget_fraction
        if easy:
            multiplier = rng.uniform(easy_multiplier_low, easy_multiplier_high)
        else:
            multiplier = rng.uniform(multiplier_low, multiplier_high)
        budget = max(1, int(round(min_feasible_cost * multiplier)))

        rows.append({
            "group_id":             group_id,
            "category_mix":         mix_name,
            "categories":           cats_in_group,
            "query_desc":           query_descs,
            "query_desc_embedding": query_embs,
            "prices":               [max(1, int(round(p))) for p in min_prices],
            "budget":               budget,
            "gt_budget":            gt_budget,
            "group_size":           group_size,
            "threshold":            float(threshold),
            "multiplier":           round(multiplier, 4),
            "within_budget":        easy,
            "seed":                 seed,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _quality_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows to report."
    avg_gs    = df["group_size"].mean()
    unique_ct = df["categories"].map(lambda c: len(set(_ensure_list(c))))
    pct_within = df["within_budget"].mean() * 100.0 if "within_budget" in df.columns else float("nan")
    pct_tight  = 100.0 - pct_within

    easy_df  = df[df["within_budget"]] if "within_budget" in df.columns else df.iloc[0:0]
    tight_df = df[~df["within_budget"]] if "within_budget" in df.columns else df.iloc[0:0]

    easy_mult  = easy_df["multiplier"].mean()  if not easy_df.empty  else float("nan")
    tight_mult = tight_df["multiplier"].mean() if not tight_df.empty else float("nan")

    return (
        f"Rows: {len(df)}\n"
        f"Fixed group size: {int(avg_gs)}\n"
        f"Avg unique categories/group: {unique_ct.mean():.2f}\n"
        f"Mixed-category rate: {(unique_ct >= 2).mean() * 100:.1f}%\n"
        f"Avg min-feasible cost (gt_budget): {df['gt_budget'].mean():.2f}\n"
        f"Avg budget: {df['budget'].mean():.2f}\n"
        f"\n"
        f"Within-budget queries (budget >= min_feasible): {pct_within:.1f}%\n"
        f"  avg multiplier: {easy_mult:.2f}  (hard-budget solvable, alpha=0)\n"
        f"Tight-budget queries (budget < min_feasible):  {pct_tight:.1f}%\n"
        f"  avg multiplier: {tight_mult:.2f}  (solution exceeds hard budget; reachable with alpha > 0)\n"
    )


def _print_examples(df: pd.DataFrame, n: int) -> None:
    if n <= 0 or df.empty:
        return
    print("\nSample outputs:")
    for _, row in df.head(n).iterrows():
        cats   = ", ".join(_ensure_list(row["categories"]))
        descs  = " | ".join(_ensure_list(row["query_desc"]))
        pstr   = ", ".join(str(p) for p in _ensure_list(row["prices"]))
        mult   = row.get("multiplier", float("nan"))
        print(f"- [{cats}] {descs}")
        print(f"  min_prices: [{pstr}]  budget: {row['budget']}  gt_budget: {row['gt_budget']}  multiplier: {mult:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--group-size", type=int, required=True,
              help="Fixed number of requirements per query.")
@click.option("--query-root",    type=click.Path(path_type=Path, exists=True),
              default=DEFAULT_QUERY_ROOT,    show_default=True)
@click.option("--analysis-root", type=click.Path(path_type=Path),
              default=DEFAULT_ANALYSIS_ROOT, show_default=True)
@click.option("--output-root",   type=click.Path(path_type=Path),
              default=DEFAULT_OUTPUT_ROOT,   show_default=True)
@click.option("--categories",    type=str, default=None, show_default=True)
@click.option("--threshold",     type=float, default=DEFAULT_THRESHOLD, show_default=True)
@click.option("--total-num",     type=int,   default=DEFAULT_TOTAL_NUM,  show_default=True)
@click.option("--seed",          type=int,   default=DEFAULT_SEED,       show_default=True)
@click.option("--show-samples",  type=int,   default=10,                 show_default=True)
@click.option("--multiplier-low",  type=float, default=DEFAULT_MULTIPLIER_LOW,  show_default=True)
@click.option("--multiplier-high", type=float, default=DEFAULT_MULTIPLIER_HIGH, show_default=True)
@click.option("--easy-multiplier-low",  type=float, default=DEFAULT_EASY_MULTIPLIER_LOW,  show_default=True)
@click.option("--easy-multiplier-high", type=float, default=DEFAULT_EASY_MULTIPLIER_HIGH, show_default=True)
@click.option("--within-budget-fraction", type=float, default=DEFAULT_WITHIN_BUDGET_FRACTION, show_default=True)
def main(
    group_size: int,
    query_root: Path,
    analysis_root: Path,
    output_root: Path,
    categories: str | None,
    threshold: float,
    total_num: int,
    seed: int,
    show_samples: int,
    multiplier_low: float,
    multiplier_high: float,
    easy_multiplier_low: float,
    easy_multiplier_high: float,
    within_budget_fraction: float,
) -> None:
    if group_size < 2:
        raise click.BadParameter("--group-size must be >= 2")
    selected = _parse_categories(categories)
    if total_num <= 0:
        raise click.BadParameter("--total-num must be positive")

    print(f"Generating {total_num} queries with fixed group_size={group_size}")
    print(f"Categories: {', '.join(selected)}")

    pool = _load_category_tables(query_root, analysis_root, selected, threshold)
    out_df = _build_group_rows(
        pool, selected, group_size, total_num, seed, threshold,
        multiplier_low, multiplier_high,
        easy_multiplier_low, easy_multiplier_high,
        within_budget_fraction,
    )

    threshold_dir = output_root / f"threshold_{_fmt_threshold(threshold)}"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    output_file = threshold_dir / f"{_mix_stem(selected)}_gs{group_size}.parquet"

    out_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(_quality_report(out_df))
    _print_examples(out_df, show_samples)
    print(f"\nWrote {len(out_df)} rows to {output_file}")


if __name__ == "__main__":
    main()
