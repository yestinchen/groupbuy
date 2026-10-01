"""
Generate representative product keywords for every item in a parquet catalog.

This is the item-level companion to e7_gen_product_single_query.py:
- use the product title plus extracted title keywords
- ask a local chat model to select 1 or 2 concise product keywords
- write the result back to each parquet file as a `keywords` column

The script processes an input folder of parquet batches and mirrors the folder
structure into an output folder, so it fits the Amazon preparation workflow.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import backoff
import click
import pandas as pd
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from tqdm import tqdm

import openai

try:
    from constants import INTERESTING_CATEGORIES
except ImportError:
    INTERESTING_CATEGORIES = []


PROMPT_KEYWORDS = """You are selecting search keywords for an Amazon product.

From the product title and extracted feature keywords, choose 1 or 2 concise keywords or short noun phrases that best represent the product type or model.
The keywords should be the terms a shopper would type to find this product.

Rules:
- Return only the keywords, separated by commas.
- Use 1 keyword if one term is enough, otherwise use 2.
- Prefer the core product name or model name, not generic adjectives.
- Do not add explanation or punctuation beyond commas.
- Keep each term short.

Example:
Title: Apple iPhone 14 Pro Max Case with MagSafe
Features: phone case, magsafe, iphone 14
Keywords: iphone case, magsafe

Title: {title}
Features: {features}
Keywords:"""


def create_chat_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="token-somethinghere",
        base_url=base_url,
    )


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _feats_to_string(feats) -> str:
    if feats is None or (isinstance(feats, float) and pd.isna(feats)):
        return ""
    if isinstance(feats, (list, tuple)) or (hasattr(feats, "__iter__") and not isinstance(feats, str)):
        return ", ".join(_normalize_text(k) for k in feats if _normalize_text(k))
    return _normalize_text(feats)


def _clean_keyword_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"^[\-\*\d\.\)\(]+", "", token).strip()
    token = token.strip(" \t\r\n\"'`")
    token = re.sub(r"\s+", " ", token)
    return token


def _parse_keywords(raw_text: str, fallback_title: str) -> list[str]:
    text = _normalize_text(raw_text)
    if not text:
        text = fallback_title
    parts = []
    for token in re.split(r"[,\n;]+", text):
        cleaned = _clean_keyword_token(token)
        if cleaned:
            parts.append(cleaned)
    if not parts and fallback_title:
        fallback_tokens = [tok for tok in re.split(r"\s+", fallback_title) if tok]
        if fallback_tokens:
            parts = [" ".join(fallback_tokens[:2])]
    deduped = []
    seen = set()
    for token in parts:
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            deduped.append(token)
    return deduped[:2]


def _source_title(row) -> str:
    title_for_embedding = _normalize_text(row.get("title_for_embedding", ""))
    if title_for_embedding:
        return title_for_embedding
    return _normalize_text(row.get("title", ""))


@backoff.on_exception(backoff.expo, (openai.OpenAIError,), max_time=3)
async def _generate_keywords(client, title: str, features: str, model_name: str) -> list[str]:
    prompt = PROMPT_KEYWORDS.format(title=title, features=features)
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stop=["\n\n"],
    )
    raw = (response.choices[0].message.content or "").strip()
    keywords = _parse_keywords(raw, title)
    if not keywords:
        keywords = _parse_keywords(features, title)
    return keywords[:2]


def get_batch_number(filename) -> int:
    try:
        if "_b_" in str(filename):
            parts = str(filename).split("_")
            if len(parts) >= 2:
                return int(parts[1].split(".")[0])
        if str(filename).startswith("batch_"):
            return int(str(filename).split("_")[1].split(".")[0])
        numbers = re.findall(r"\d+", str(filename))
        if numbers:
            return int(numbers[0])
        return 0
    except Exception:
        return 0


def process_file(
    file: Path,
    output_folder: Path,
    model_name: str,
    base_url: str,
    max_concurrent: int = 64,
    max_rows_per_file: int | None = None,
) -> int:
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / file.name

    if output_file.exists():
        print(f"  Skipping {file.name}: output exists")
        return 0

    df = pq.read_table(file).to_pandas()
    if max_rows_per_file is not None and max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
        print(f"  Limiting to first {len(df)} rows (--max-rows-per-file)")

    if "title" not in df.columns and "title_for_embedding" not in df.columns and "title_feats" not in df.columns:
        print(f"  Skipping {file.name}: no usable title columns")
        return 0

    titles = [_source_title(row) for _, row in df.iterrows()]
    feats_list = df["title_feats"].tolist() if "title_feats" in df.columns else [""] * len(df)
    n_rows = len(df)

    async def run_all():
        client = create_chat_client(base_url)
        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        results = [None] * n_rows
        pbar = tqdm(total=n_rows, desc=file.name, unit="row")

        async def wrap(i: int, title: str, feats):
            features = _feats_to_string(feats)
            async with semaphore:
                try:
                    keywords = await _generate_keywords(client, title, features, model_name)
                except Exception:
                    keywords = _parse_keywords(features, title)
            results[i] = keywords
            pbar.update(1)

        try:
            await asyncio.gather(*(wrap(i, titles[i], feats_list[i]) for i in range(n_rows)))
            return results
        finally:
            pbar.close()
            await client.close()

    keywords = asyncio.run(run_all())
    out_df = df.copy()
    out_df["keywords"] = keywords
    out_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"  Saved {len(out_df)} rows to {output_file.name}")
    return len(out_df)


def process_category(
    input_folder: Path,
    output_folder: Path,
    category: str,
    model_name: str,
    base_url: str,
    max_concurrent: int = 64,
    max_rows_per_file: int | None = None,
) -> None:
    in_dir = input_folder / category if category else input_folder
    out_dir = output_folder / category if category else output_folder
    if not in_dir.exists():
        print(f"  Warning: input folder not found: {in_dir}")
        return
    files = sorted(in_dir.glob("*.parquet"), key=lambda f: get_batch_number(f.name))
    if not files:
        print(f"  Warning: no parquet files in {in_dir}")
        return
    print(f"  Found {len(files)} file(s) in {in_dir}")
    for file in tqdm(files, desc=f"Processing category {category}"):
        process_file(
            file=file,
            output_folder=out_dir,
            model_name=model_name,
            base_url=base_url,
            max_concurrent=max_concurrent,
            max_rows_per_file=max_rows_per_file,
        )


@click.command()
@click.option("--input-folder", type=click.Path(path_type=Path), required=True, help="Input folder with category parquet batches.")
@click.option("--output-folder", type=click.Path(path_type=Path), required=True, help="Output folder for keyword-enriched parquet batches.")
@click.option("--category", type=str, default="All_Beauty", help="Category to process. Use an empty string to process the input folder directly.")
@click.option("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Local chat model used to choose representative product keywords.")
@click.option("--base-url", type=str, default="http://localhost:8123/v1", help="OpenAI-compatible base URL for the local chat server.")
@click.option("--max-concurrent", type=int, default=128, help="Maximum concurrent rows in flight.")
@click.option("--max-rows-per-file", type=int, default=None, help="Optional cap for rows per parquet file, useful for quick validation.")
def main(
    input_folder: Path,
    output_folder: Path,
    category: str,
    model_name: str,
    base_url: str,
    max_concurrent: int,
    max_rows_per_file: int | None,
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if category is None or category == "":
        categories = INTERESTING_CATEGORIES if INTERESTING_CATEGORIES else [""]
    else:
        categories = [category]

    for cat in categories:
        if cat:
            print(f"Processing category {cat}")
        else:
            print(f"Processing input folder {input_folder}")
        process_category(
            input_folder=input_folder,
            output_folder=output_folder,
            category=cat,
            model_name=model_name,
            base_url=base_url,
            max_concurrent=max_concurrent,
            max_rows_per_file=max_rows_per_file,
        )


if __name__ == "__main__":
    main()
