"""
Compute embeddings for all refined titles and store them in separate parquet files.

This is the storage-light companion to sb2_compute_embedding.py:
- read refined-title parquet batches
- batch the embedding calls so vLLM can continuous-batch them
- write only the item id and embedding vector to a new parquet file

The output is intentionally compact, so the refined-title text itself stays in the
source parquet and the embeddings live in a separate folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import backoff
import click
import openai
import pandas as pd
import pyarrow.parquet as pq
from openai import OpenAI
from tqdm import tqdm

try:
    from constants import INTERESTING_CATEGORIES
except ImportError:
    INTERESTING_CATEGORIES = []


def create_local_client(base_url: str) -> OpenAI:
    return OpenAI(
        api_key="token-somethinghere",
        base_url=base_url,
    )


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, str)):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


@backoff.on_exception(
    backoff.expo,
    (openai.OpenAIError,),
    max_time=60,
)
def compute_embeddings(client, texts, model_name):
    response = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
    return [item.embedding for item in data]


def get_batch_number(filename):
    try:
        if "_b_" in filename:
            parts = filename.split("_b_")
            if len(parts) > 1:
                return int(parts[1].split(".")[0])
        elif filename.startswith("batch_"):
            return int(filename.split("_")[1].split(".")[0])
        numbers = re.findall(r"\d+", filename)
        if numbers:
            return int(numbers[0])
        return 0
    except Exception:
        return 0


def _source_text(row, title_column: str) -> str:
    if title_column in row:
        text = _normalize_text(row.get(title_column))
        if text:
            return text
    return _normalize_text(row.get("title", ""))


def process_file(
    file: Path,
    model_name: str,
    input_folder: Path,
    output_folder: Path,
    base_url: str = "http://localhost:8124/v1",
    title_column: str = "title_for_embedding",
    id_column: str = "id",
    embedding_column: str = "title_embedding",
    batch_size: int = 64,
    max_rows_per_file: int | None = None,
):
    """
    Compute embeddings for one parquet file and write a compact output parquet.
    The output keeps only the item id and embedding column to reduce storage.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{file.name.split('.')[0]}.parquet"

    if output_file.exists():
        print(f"  Skipping {file.name}: output file {output_file.name} already exists")
        return

    df = pq.read_table(file).to_pandas()
    if max_rows_per_file is not None and max_rows_per_file > 0:
        df = df.head(max_rows_per_file)
        print(f"  Limiting to first {len(df)} rows (--max-rows-per-file)")

    if title_column not in df.columns and "title" not in df.columns:
        print(f"  Skipping {file.name}: missing refined title column")
        return

    if id_column not in df.columns:
        df[id_column] = df.index

    if title_column not in df.columns:
        title_column = "title"

    texts = [_source_text(row, title_column) for _, row in df.iterrows()]
    ids = df[id_column].tolist()

    client = create_local_client(base_url)
    embeddings = []

    batch_size = max(1, int(batch_size))
    pbar = tqdm(total=len(df), desc=f"Processing file {file.name}", unit="row")
    for start in range(0, len(df), batch_size):
        batch_texts = []
        for text in texts[start:start + batch_size]:
            batch_texts.append(_normalize_text(text) or " ")
        batch_embeddings = compute_embeddings(client, batch_texts, model_name)
        embeddings.extend(batch_embeddings)
        pbar.update(len(batch_texts))
    pbar.close()

    if embeddings:
        out_df = pd.DataFrame({
            id_column: ids,
            embedding_column: embeddings,
        })
        out_df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"  Saved {len(out_df)} compact embedding rows to {output_file.name}")


def process_category(
    model_name,
    input_folder,
    output_folder,
    category,
    base_url: str = "http://localhost:8124/v1",
    title_column: str = "title_for_embedding",
    id_column: str = "id",
    embedding_column: str = "title_embedding",
    batch_size: int = 64,
    max_rows_per_file: int | None = None,
):
    files = list(Path(input_folder).glob("*.parquet"))
    files.sort(key=lambda f: get_batch_number(f.name))

    if not files:
        print(f"  Warning: No parquet files found in {input_folder}")
        return

    print(f"  Found {len(files)} batch file(s), processing in order...")

    for file in tqdm(files, desc=f"Processing category {category}"):
        if file.name.endswith(".parquet"):
            process_file(
                file,
                model_name,
                input_folder,
                output_folder,
                base_url,
                title_column,
                id_column,
                embedding_column,
                batch_size,
                max_rows_per_file,
            )


def _default_input_folder():
    root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
    return os.path.join(root, "amazon_review_2023", "meta_parquet_clean_attributes_titled")


def _default_output_folder():
    root = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
    return os.path.join(root, "amazon_review_2023", "meta_parquet_clean_attributes_titled_embed")


@click.command()
@click.option(
    "--model-name",
    type=str,
    required=True,
    default="Qwen/Qwen3-Embedding-0.6B",
    help="The name of the embedding model to use",
)
@click.option(
    "--input-folder",
    type=str,
    default=None,
    help="Input folder (default: $DATASET_ROOT/amazon_review_2023/meta_parquet_clean_attributes_titled)",
)
@click.option(
    "--base-url",
    type=str,
    default="http://localhost:8124/v1",
    help="OpenAI-compatible base URL for the local embedding server.",
)
@click.option(
    "--output-folder",
    type=str,
    default=None,
    help="Output folder (default: $DATASET_ROOT/amazon_review_2023/meta_parquet_clean_attributes_titled_embed)",
)
@click.option(
    "--category",
    type=str,
    required=False,
    help="The category to use",
    default="Baby_Products",
)
@click.option(
    "--title-column",
    type=str,
    default="title_for_embedding",
    help="Column to use as text for embedding. Falls back to title if missing.",
)
@click.option(
    "--id-column",
    type=str,
    default="id",
    help="Column to preserve in the compact embedding parquet.",
)
@click.option(
    "--embedding-column",
    type=str,
    default="title_embedding",
    help="Column name to write embeddings into.",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="Number of texts to send per embedding request. Larger values better utilize vLLM batching.",
)
@click.option(
    "--max-rows-per-file",
    type=int,
    default=None,
    help="Optional cap for rows per parquet file, useful for quick validation.",
)
def main(
    model_name,
    input_folder,
    base_url,
    output_folder,
    category,
    title_column,
    id_column,
    embedding_column,
    batch_size,
    max_rows_per_file,
):
    if input_folder is None:
        input_folder = _default_input_folder()
    if output_folder is None:
        output_folder = _default_output_folder()
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if category is None or category == "None":
        for cat in INTERESTING_CATEGORIES:
            input_folder_category = Path(input_folder) / cat
            output_folder_category = Path(output_folder) / cat
            process_category(
                model_name,
                input_folder_category,
                output_folder_category,
                cat,
                base_url,
                title_column,
                id_column,
                embedding_column,
                batch_size,
                max_rows_per_file,
            )
    else:
        input_folder_category = Path(input_folder) / category
        output_folder_category = Path(output_folder) / category
        process_category(
            model_name,
            input_folder_category,
            output_folder_category,
            category,
            base_url,
            title_column,
            id_column,
            embedding_column,
            batch_size,
            max_rows_per_file,
        )


if __name__ == "__main__":
    main()
