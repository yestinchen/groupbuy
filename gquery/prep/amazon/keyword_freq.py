from __future__ import annotations

import ast
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def normalize_token(token) -> str:
    text = str(token).strip()
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"\s+", " ", text)
    return text


def iter_keywords(value):
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
        return [normalize_token(part) for part in parts if normalize_token(part)]
    if isinstance(value, (list, tuple, set)):
        out = []
        for part in value:
            if isinstance(part, (list, tuple, set)):
                out.extend(iter_keywords(part))
            else:
                token = normalize_token(part)
                if token:
                    out.append(token)
        return out
    token = normalize_token(value)
    return [token] if token else []


def read_keyword_column(path: Path) -> pd.Series:
    try:
        df = pd.read_parquet(path, columns=["keywords"], engine="pyarrow")
    except Exception:
        df = pd.read_parquet(path, engine="pyarrow")
        if "keywords" not in df.columns:
            raise ValueError(f"{path} does not contain a 'keywords' column")
        df = df[["keywords"]]
    return df["keywords"]


def main(input_path: str, output_file: str, top_n: int) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")

    counter = Counter()
    total_items = 0
    items_with_keywords = 0

    for parquet_file in files:
        keywords_series = read_keyword_column(parquet_file)
        total_items += len(keywords_series)
        for value in keywords_series:
            keywords = iter_keywords(value)
            if not keywords:
                continue
            items_with_keywords += 1
            seen = set()
            for keyword in keywords:
                lowered = keyword.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                counter[lowered] += 1

    rows = [
        {
            "keyword": keyword,
            "item_count": count,
            "item_share": count / total_items if total_items else 0.0,
        }
        for keyword, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].lower(), kv[0]))
    ]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Analyzed {total_items:,} items from {len(files)} file(s)")
    print(f"Items with keywords: {items_with_keywords:,}")
    print(f"Unique keywords: {len(out_df):,}")
    print(f"Wrote sorted frequency table to: {output_path}")
    print()
    print(f"Top {min(top_n, len(out_df))} keywords by item frequency:")
    for idx, row in out_df.head(top_n).iterrows():
        rank = idx + 1
        print(f"{rank:>4}. {row['keyword']}  ({int(row['item_count']):,} items, {row['item_share']:.4f})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python - INPUT_PATH OUTPUT_FILE TOP_N")
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
