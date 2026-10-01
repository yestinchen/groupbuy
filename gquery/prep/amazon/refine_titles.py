"""
Generate a title description from extracted keywords and the original title for the embedding approach.

Reads parquet(s) with title_feats and title, adds column title_for_embedding using a chat model to
generate a short product title from both sources. Sends one request per item; use
--max-concurrent so many requests are in flight at once and vLLM can continuous-batch them.

Usage:
  Single category:
    python -m gquery.prep.amazon.refine_titles --input-folder meta_parquet_clean_attributes/Baby_Products \\
      --output-folder meta_parquet_clean_attributes_titled/Baby_Products --model-name qwen/Qwen2.5-3B-Instruct
  All INTERESTING_CATEGORIES:
    python -m gquery.prep.amazon.refine_titles --input-folder meta_parquet_clean_attributes \\
      --output-folder meta_parquet_clean_attributes_titled --category None --model-name qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from pyarrow import parquet as pq
from tqdm import tqdm

try:
    from constants import INTERESTING_CATEGORIES
except ImportError:
    INTERESTING_CATEGORIES = []


PROMPT_TITLE_FROM_TITLE_AND_KEYWORDS = """You are rewriting an Amazon catalog title for retrieval.

Turn the following original product title and extracted keywords into a short, natural product title.

Rules:
- Use the original title as the primary source of truth and the keywords as secondary evidence.
- Preserve reliable brand names, product names, model names, quantities, sizes, counts, colors, and variant markers.
- Do not invent claims, ingredients, benefits, or product types that are not supported by the title or keywords.
- If the original title is already concise and correct, lightly clean it instead of rewriting it from scratch.
- Keep the core product type first, then the most important attributes.
- Keep it fairly compact, but allow enough detail for identity: usually 6 to 18 words.
- If the title needs extra detail to stay faithful, prefer that over over-compressing it.
- Prefer a clean title-like phrase over a keyword list.
- Do not use bullet points or extra explanation.
- Avoid repeating the same idea twice.
- If the source is noisy or ambiguous, choose the safest accurate title rather than an overconfident guess.

Examples:
Original title: Lurrose 100Pcs Full Cover Fake Toenails Artificial Transparent Nail Tips Nail Art for DIY
Keywords: Lurrose, 100Pcs, Full Cover, Fake, Toenails, Artificial, Transparent, Nail Tips, Nail Art, DIY
Title: Lurrose 100Pcs Transparent Full Cover Fake Toenails

Original title: Garnier Fructis Color Sealer, Instant, Lightweight Leave-In, Color Shield, For Color-Treated Hair, 6 oz.
Keywords: Garnier, Fructis, Color Sealer, Instant, Lightweight, Leave-In, Color Shield, Color-Treated Hair, 6 oz
Title: Garnier Fructis Instant Color Shield Leave-In Conditioner

Original title: {title}
Keywords: {keywords}

Short product title:"""

# Base URL for local vLLM OpenAI-compatible server (chat).
VLLM_CHAT_BASE_URL = "http://localhost:8123/v1"

# Max tokens per generated title. Query descriptions (query_desc in sb3a_gen_query_desc) have no explicit limit (stop at double newline).
DEFAULT_MAX_TITLE_TOKENS = 1024
DEFAULT_REQUEST_TIMEOUT = 10.0
DEFAULT_TEMPERATURE = 0.0


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


def _title_simple(feats) -> str:
    """Join keywords with space. Handles list, array, or missing."""
    if feats is None or (isinstance(feats, float) and pd.isna(feats)):
        return ""
    if isinstance(feats, (list, tuple)) or hasattr(feats, "__iter__") and not isinstance(feats, str):
        return " ".join(str(k).strip() for k in feats if k)
    return str(feats).strip()


def _feats_to_keywords_str(feats) -> str:
    """Convert title_feats to a single comma-separated string. Handles list or array."""
    if feats is None or (isinstance(feats, float) and pd.isna(feats)):
        return ""
    if isinstance(feats, (list, tuple)) or (hasattr(feats, "__iter__") and not isinstance(feats, str)):
        return ", ".join(str(k).strip() for k in feats if k)
    return str(feats).strip()


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _is_empty_feats(feats) -> bool:
    """Empty check that avoids truthiness on numpy arrays."""
    if feats is None:
        return True
    try:
        if len(feats) == 0:
            return True
    except (TypeError, AttributeError):
        pass
    try:
        if isinstance(feats, (float, int)) and pd.isna(feats):
            return True
    except (TypeError, ValueError):
        pass
    return False


async def _title_llm_async(
    client,
    original_title,
    feats,
    model_name: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """One request per item; semaphore limits concurrency so vLLM can batch in flight."""
    if _is_empty_feats(feats):
        return None
    original_title_str = _normalize_text(original_title)
    keywords_str = _feats_to_keywords_str(feats)
    if not original_title_str and not keywords_str:
        return None
    prompt = PROMPT_TITLE_FROM_TITLE_AND_KEYWORDS.format(
        title=original_title_str or "",
        keywords=keywords_str or "",
    )
    async with semaphore:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n\n"],
                ),
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            title = (response.choices[0].message.content or "").strip().split("\n")[0].strip()
            return title or None
        except Exception:
            return None


def process_file(
    file: Path,
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    base_url: str,
    skip_existing: bool = True,
    max_concurrent: int = 32,
    llm_max_title_tokens: int = DEFAULT_MAX_TITLE_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_rows_per_file: int | None = None,
) -> int:
    """
    Process a single parquet: add title_for_embedding from title_feats via LLM.
    Sends one request per row with up to max_concurrent in flight so vLLM can continuous-batch.
    Returns number of rows written.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    file_pure_name = file.name.split(".")[0]
    output_file = output_folder / f"{file_pure_name}.parquet"

    if skip_existing and output_file.exists():
        print(f"  Skipping {file.name}: output exists")
        return 0

    df = pq.read_table(file).to_pandas()
    if max_rows_per_file is not None and max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
        print(f"  Limiting to first {len(df)} rows (--max-rows-per-file)")
    if "title_feats" not in df.columns:
        print(f"  Skipping {file.name}: no title_feats column")
        return 0

    titles = df["title"].tolist() if "title" in df.columns else [""] * len(df)
    feats_list = df["title_feats"].tolist()
    n_rows = len(feats_list)

    async def run_all():
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="token-somethinghere", base_url=base_url)
        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        results = [None] * n_rows
        pbar = tqdm(total=n_rows, desc=file.name, unit="row")

        async def wrap(i, title, feats):
            r = await _title_llm_async(client, title, feats, model_name, llm_max_title_tokens, temperature, semaphore)
            results[i] = r
            pbar.update(1)
            return r

        tasks = [wrap(i, titles[i], feats_list[i]) for i in range(n_rows)]
        try:
            await asyncio.gather(*tasks)
            return results
        finally:
            pbar.close()
            await client.close()

    titles = asyncio.run(run_all())
    keep_mask = [title is not None and str(title).strip() != "" for title in titles]
    kept_df = df.loc[keep_mask].reset_index(drop=True)
    kept_df["title_for_embedding"] = [title for title in titles if title is not None and str(title).strip() != ""]

    dropped = len(df) - len(kept_df)
    kept_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"  Saved {len(kept_df)} rows to {output_file.name} (dropped {dropped})")
    return len(kept_df)


def process_category(
    input_folder: Path,
    output_folder: Path,
    category: str,
    model_name: str,
    base_url: str,
    skip_existing: bool = True,
    max_concurrent: int = 32,
    llm_max_title_tokens: int = DEFAULT_MAX_TITLE_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
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
    for f in tqdm(files, desc=f"Category {category or 'all'}"):
        process_file(
            f, in_dir, out_dir, model_name, base_url,
            skip_existing, max_concurrent, llm_max_title_tokens, temperature, max_rows_per_file,
        )


@click.command()
@click.option(
    "--input-folder",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Root folder containing category subfolders with parquet files (with title_feats)",
)
@click.option(
    "--output-folder",
    type=click.Path(path_type=Path),
    required=True,
    help="Output root folder; category subfolders will be created",
)
@click.option(
    "--category",
    type=str,
    default=None,
    help="Single category name (e.g. Baby_Products), or None for all INTERESTING_CATEGORIES",
)
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="Chat model for generating titles (e.g. qwen/Qwen2.5-3B-Instruct)",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=32,
    help="Max concurrent requests in flight. vLLM continuous-batches these (try 32–64).",
)
@click.option(
    "--max-title-tokens",
    type=int,
    default=DEFAULT_MAX_TITLE_TOKENS,
    help=f"Max tokens per generated title (default {DEFAULT_MAX_TITLE_TOKENS}). Query descriptions (query_desc) use no explicit limit.",
)
@click.option(
    "--temperature",
    type=float,
    default=DEFAULT_TEMPERATURE,
    help=f"Sampling temperature for title generation (default {DEFAULT_TEMPERATURE}).",
)
@click.option(
    "--no-skip-existing",
    is_flag=True,
    help="Overwrite existing output files",
)
@click.option(
    "--base-url",
    type=str,
    default=VLLM_CHAT_BASE_URL,
    help="OpenAI-compatible API base URL (default: vLLM chat server)",
)
@click.option(
    "--max-rows-per-file",
    type=int,
    default=None,
    help="If set, only process this many rows per parquet (for sampling/debug).",
)
def main(
    input_folder: Path,
    output_folder: Path,
    category: str | None,
    model_name: str,
    max_concurrent: int,
    max_title_tokens: int,
    temperature: float,
    no_skip_existing: bool,
    base_url: str,
    max_rows_per_file: int | None,
) -> None:
    """Generate title_for_embedding from title_feats using a chat model."""
    skip_existing = not no_skip_existing
    if category is None or category == "None":
        for cat in INTERESTING_CATEGORIES:
            process_category(
                input_folder, output_folder, cat, model_name, base_url,
                skip_existing, max_concurrent, max_title_tokens, temperature, max_rows_per_file,
            )
    else:
        process_category(
            input_folder, output_folder, category, model_name, base_url,
            skip_existing, max_concurrent, max_title_tokens, temperature, max_rows_per_file,
        )


if __name__ == "__main__":
    main()
