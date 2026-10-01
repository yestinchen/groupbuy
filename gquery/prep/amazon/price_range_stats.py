from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

import pandas as pd


def normalize_keyword(token) -> str:
    text = str(token).strip()
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def iter_keywords(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return iter_keywords(value.tolist())
        except Exception:
            pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                return iter_keywords(parsed)
            except Exception:
                pass
        parts = re.split(r"[,\n;]+", text)
        out = []
        for part in parts:
            token = normalize_keyword(part)
            if token:
                out.append(token)
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for part in value:
            out.extend(iter_keywords(part))
        return out
    token = normalize_keyword(value)
    return [token] if token else []


def read_kept_keywords(keyword_file: Path) -> list[str]:
    if not keyword_file.exists():
        raise FileNotFoundError(f"Missing filtered keyword file: {keyword_file}")
    df = pd.read_csv(keyword_file)
    if "keyword" not in df.columns:
        raise ValueError(f"{keyword_file} is missing required column: keyword")
    keywords = []
    seen = set()
    for value in df["keyword"].astype(str).tolist():
        keyword = normalize_keyword(value)
        if keyword and keyword not in seen:
            seen.add(keyword)
            keywords.append(keyword)
    return keywords


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def analyze_category(
    category: str,
    input_root: Path,
    keyword_root: Path,
    output_root: Path,
    max_files: int | None = None,
) -> None:
    category_input = input_root / category
    keyword_file = keyword_root / f"{category}.csv"
    output_file = output_root / f"{category}.csv"

    if not category_input.exists():
        print(f"Skipping missing catalog folder: {category_input}")
        return
    if not keyword_file.exists():
        print(f"Skipping missing keyword file: {keyword_file}")
        return

    kept_keywords = read_kept_keywords(keyword_file)
    kept_set = set(kept_keywords)

    parquet_files = sorted(category_input.glob("*.parquet"))
    if max_files is not None and max_files > 0:
        parquet_files = parquet_files[:max_files]

    if not parquet_files:
        print(f"Skipping {category}: no parquet files found in {category_input}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "keyword",
            "item_count",
            "price_min",
            "price_max",
            "price_mean",
            "price_median",
            "price_std",
            "price_q25",
            "price_q75",
            "rating_min",
            "rating_max",
            "rating_mean",
            "rating_median",
            "rating_std",
            "rating_q25",
            "rating_q75",
        ]).to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
        return

    rows = []
    total_items = 0
    matched_items = 0

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, columns=["keywords", "price", "average_rating"], engine="pyarrow")
        total_items += len(df)
        df["price"] = _numeric_series(df["price"])
        df["average_rating"] = _numeric_series(df["average_rating"])

        for _, row in df.iterrows():
            keywords = iter_keywords(row["keywords"])
            matched = []
            seen = set()
            for keyword in keywords:
                if keyword in kept_set and keyword not in seen:
                    seen.add(keyword)
                    matched.append(keyword)
            if not matched:
                continue
            matched_items += 1
            for keyword in matched:
                rows.append(
                    {
                        "keyword": keyword,
                        "price": row["price"],
                        "average_rating": row["average_rating"],
                    }
                )

    if rows:
        exploded = pd.DataFrame(rows)
        grouped = exploded.groupby("keyword", sort=True)
        summary = grouped.agg(
            item_count=("keyword", "size"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            price_mean=("price", "mean"),
            price_median=("price", "median"),
            price_std=("price", lambda s: s.std(ddof=0)),
            price_q25=("price", lambda s: s.quantile(0.25)),
            price_q75=("price", lambda s: s.quantile(0.75)),
            rating_min=("average_rating", "min"),
            rating_max=("average_rating", "max"),
            rating_mean=("average_rating", "mean"),
            rating_median=("average_rating", "median"),
            rating_std=("average_rating", lambda s: s.std(ddof=0)),
            rating_q25=("average_rating", lambda s: s.quantile(0.25)),
            rating_q75=("average_rating", lambda s: s.quantile(0.75)),
        ).reset_index()
        summary = summary.sort_values(["item_count", "keyword"], ascending=[False, True]).reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=[
            "keyword",
            "item_count",
            "price_min",
            "price_max",
            "price_mean",
            "price_median",
            "price_std",
            "price_q25",
            "price_q75",
            "rating_min",
            "rating_max",
            "rating_mean",
            "rating_median",
            "rating_std",
            "rating_q25",
            "rating_q75",
        ])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"{category}: analyzed {total_items:,} items across {len(parquet_files)} file(s)")
    print(f"{category}: matched {matched_items:,} items across {len(summary):,} kept keyword(s)")
    print(f"{category}: wrote {output_file}")


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit("Usage: python - INPUT_ROOT KEYWORD_ROOT OUTPUT_ROOT MAX_FILES [CATEGORIES...]")

    input_root = Path(sys.argv[1])
    keyword_root = Path(sys.argv[2])
    output_root = Path(sys.argv[3])
    max_files_raw = sys.argv[4].strip()
    max_files = int(max_files_raw) if max_files_raw else None
    categories = sys.argv[5:]

    for category in categories:
        analyze_category(
            category=category,
            input_root=input_root,
            keyword_root=keyword_root,
            output_root=output_root,
            max_files=max_files,
        )


if __name__ == "__main__":
    main()
