from __future__ import annotations

"""
Generate a single shopper-style query description for each keyword stat row.

The single query generation:
1) load keyword statistics from ~/dataset/amazon_review_2023/analysis/price_distribute/
2) prompt a local vLLM chat model to produce one search query description per keyword
3) embed the generated description with the local embedding vLLM
4) inspect a small sample of outputs for quality

This script keeps the workflow simple:
- load every CSV in the stats directory
- sample rows if requested
- call the local vLLM chat server one row at a time with async batching
- embed the generated query descriptions with the local embedding server
- write only keyword, query_desc, and query_desc_embedding to parquet or CSV
- print a compact quality report and a few sampled examples
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import backoff
import click
import pandas as pd
import openai
from openai import AsyncOpenAI
from tqdm import tqdm


DEFAULT_STATS_DIR = Path("~/dataset/amazon_review_2023/analysis/price_distribute")
DEFAULT_BASE_URL = "http://localhost:8123/v1"
DEFAULT_EMBED_BASE_URL = "http://localhost:8134/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


PROMPT_TEMPLATE = """You write shopper search queries for Amazon-like product discovery.

Given the keyword below, write exactly ONE short, title-like natural language search query description a shopper would type to find that product.

Rules:
- Output only the query text.
- Keep it short and natural, usually 6 to 12 words.
- Make it sound like a shopper request or product title, not a keyword list.
- You may add a small amount of useful descriptive wording if it is strongly suggested by the keyword.
- Descriptive additions should be light and common, such as a form factor or obvious use case.
- Do not invent unsupported brands, materials, colors, or product types.
- If a safe expansion is not obvious, stay very close to the keyword.
- When possible, include a few additional descriptive words so the query feels more like a natural shopper search.
- Keep it phrase-like rather than a full sentence.
- Do not add explanation, bullets, quotes, or a trailing period.
- If the keyword already looks like a good search query, keep it close in meaning.

Examples:
Keyword: photo frame
Query: photo wall frame for pictures

Keyword: kayaking stand
Query: kayak storage stand for outdoor use

Keyword: blush
Query: natural blush makeup palette for everyday wear

Keyword: {keyword}
Query:"""


@dataclass(frozen=True)
class QueryGenConfig:
    stats_dir: Path
    output_file: Path | None
    model_name: str
    base_url: str
    embed_model_name: str
    embed_base_url: str
    max_concurrent: int
    max_tokens: int
    temperature: float
    sample_size: int | None
    seed: int
    show_samples: int


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_keyword(value: str) -> str:
    text = _normalize_text(value)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`")
    return text


def _clean_model_query(raw_text: str, fallback_keyword: str) -> str:
    text = _normalize_text(raw_text)
    if not text:
        return fallback_keyword

    text = text.split("\n")[0].strip()
    text = re.sub(r"^(query|search query|search|result)\s*:\s*", "", text, flags=re.I)
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"\s+", " ", text)

    # Remove obvious sentence punctuation while keeping the phrase intact.
    text = text.rstrip(".;:!?")
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    return text or fallback_keyword


def _load_keyword_stats(stats_dir: Path) -> pd.DataFrame:
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    files = sorted(stats_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {stats_dir}")

    frames = []
    for file in files:
        df = pd.read_csv(file)
        if "keyword" not in df.columns:
            raise ValueError(f"Missing 'keyword' column in {file}")
        df = df.copy()
        df["category"] = file.stem
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["keyword"] = out["keyword"].map(_normalize_keyword)
    out = out[out["keyword"] != ""].reset_index(drop=True)
    return out


def _format_stats(row: pd.Series) -> dict[str, str]:
    return {
        "keyword": _normalize_text(row.get("keyword", "")),
    }


def _build_prompt(row: pd.Series) -> str:
    stats = _format_stats(row)
    return PROMPT_TEMPLATE.format(**stats)


@backoff.on_exception(
    backoff.expo,
    (openai.OpenAIError,),
    max_time=60,
)
async def _generate_one(
    client: AsyncOpenAI,
    embed_client: AsyncOpenAI,
    row: pd.Series,
    model_name: str,
    embed_model_name: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict:
    keyword = _normalize_keyword(row.get("keyword", ""))
    prompt = _build_prompt(row)

    async with semaphore:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n"],
        )

    raw_query = (response.choices[0].message.content or "").strip()
    query = _clean_model_query(raw_query, keyword)
    embed_response = await embed_client.embeddings.create(
        input=query or " ",
        model=embed_model_name,
    )
    query_embedding = embed_response.data[0].embedding
    return {
        "keyword": keyword,
        "query_desc": query,
        "query_desc_embedding": query_embedding,
    }


def _sample_dataframe(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(df) <= sample_size:
        return df.reset_index(drop=True)
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def _quality_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows to report."

    avg_words = df["query_desc"].map(lambda x: len(_normalize_text(x).split())).mean()
    exact_rate = (df["query_desc"].map(_normalize_text).str.lower() == df["keyword"].map(_normalize_text).str.lower()).mean() * 100.0
    punct_rate = df["query_desc"].map(lambda x: bool(re.search(r"[.!?]", _normalize_text(x)))).mean() * 100.0
    long_rate = (df["query_desc"].map(lambda x: len(_normalize_text(x).split())) > 8).mean() * 100.0
    empty_rate = (df["query_desc"].map(_normalize_text) == "").mean() * 100.0

    return (
        f"Rows: {len(df)}\n"
        f"Avg words/query: {avg_words:.2f}\n"
        f"Exact keyword match rate: {exact_rate:.1f}%\n"
        f"Sentence punctuation rate: {punct_rate:.1f}%\n"
        f"Long query rate (>8 words): {long_rate:.1f}%\n"
        f"Empty query rate: {empty_rate:.1f}%"
    )


def _print_examples(df: pd.DataFrame, n: int) -> None:
    if n <= 0 or df.empty:
        return
    print("\nSample outputs:")
    cols = ["keyword", "query_desc"]
    sample_df = df[cols].head(n)
    for _, row in sample_df.iterrows():
        print(f"- {row['keyword']} -> {row['query_desc']}")


async def _run_generation(config: QueryGenConfig) -> pd.DataFrame:
    df = _load_keyword_stats(config.stats_dir)
    df = _sample_dataframe(df, config.sample_size, config.seed)

    client = AsyncOpenAI(api_key="token-somethinghere", base_url=config.base_url)
    embed_client = AsyncOpenAI(api_key="token-somethinghere", base_url=config.embed_base_url)
    semaphore = asyncio.Semaphore(max(1, config.max_concurrent))
    results: list[dict] = [None] * len(df)  # type: ignore[assignment]

    pbar = tqdm(total=len(df), desc="Generating single queries", unit="row")
    try:
        async def wrap(idx: int, row: pd.Series) -> None:
            result = await _generate_one(
                client,
                embed_client,
                row,
                config.model_name,
                config.embed_model_name,
                config.max_tokens,
                config.temperature,
                semaphore,
            )
            results[idx] = result
            pbar.update(1)

        await asyncio.gather(*(wrap(i, row) for i, (_, row) in enumerate(df.iterrows())))
    finally:
        pbar.close()
        await client.close()
        await embed_client.close()

    return pd.DataFrame(results)


@click.command()
@click.option(
    "--stats-dir",
    type=click.Path(path_type=Path, exists=True),
    default=DEFAULT_STATS_DIR,
    show_default=True,
    help="Directory containing keyword-stat CSVs.",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output parquet/csv path. If omitted, results are only printed.",
)
@click.option(
    "--model-name",
    type=str,
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Local vLLM chat model name.",
)
@click.option(
    "--base-url",
    type=str,
    default=DEFAULT_BASE_URL,
    show_default=True,
    help="OpenAI-compatible vLLM chat base URL.",
)
@click.option(
    "--embed-base-url",
    type=str,
    default=DEFAULT_EMBED_BASE_URL,
    show_default=True,
    help="OpenAI-compatible embedding vLLM base URL.",
)
@click.option(
    "--embed-model-name",
    type=str,
    default=DEFAULT_EMBED_MODEL_NAME,
    show_default=True,
    help="Local vLLM embedding model name.",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=16,
    show_default=True,
    help="Number of in-flight requests to batch on vLLM.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=24,
    show_default=True,
    help="Max generated tokens per query.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    show_default=True,
    help="Sampling temperature for query generation.",
)
@click.option(
    "--sample-size",
    type=int,
    default=25,
    show_default=True,
    help="How many keyword rows to sample before generation. Use 0 or a negative value for all rows.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed used when sampling rows.",
)
@click.option(
    "--show-samples",
    type=int,
    default=10,
    show_default=True,
    help="How many generated rows to print for a quick quality check.",
)
def main(
    stats_dir: Path,
    output_file: Path | None,
    model_name: str,
    base_url: str,
    embed_base_url: str,
    embed_model_name: str,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    sample_size: int,
    seed: int,
    show_samples: int,
) -> None:
    sample_size_opt = None if sample_size <= 0 else sample_size
    config = QueryGenConfig(
        stats_dir=stats_dir,
        output_file=output_file,
        model_name=model_name,
        base_url=base_url,
        embed_model_name=embed_model_name,
        embed_base_url=embed_base_url,
        max_concurrent=max_concurrent,
        max_tokens=max_tokens,
        temperature=temperature,
        sample_size=sample_size_opt,
        seed=seed,
        show_samples=show_samples,
    )

    try:
        out_df = asyncio.run(_run_generation(config))
    except Exception as exc:
        raise click.ClickException(
            f"Generation failed. Check that the local vLLM server is running at {base_url} "
            f"and that model '{model_name}' is available. Underlying error: {exc}"
        ) from exc

    print(_quality_report(out_df))
    _print_examples(out_df, config.show_samples)

    if config.output_file is not None:
        config.output_file.parent.mkdir(parents=True, exist_ok=True)
        if config.output_file.suffix.lower() == ".csv":
            out_df.to_csv(config.output_file, index=False)
        else:
            out_df.to_parquet(config.output_file, engine="pyarrow", index=False)
        print(f"\nWrote {len(out_df)} rows to {config.output_file}")


if __name__ == "__main__":
    main()
