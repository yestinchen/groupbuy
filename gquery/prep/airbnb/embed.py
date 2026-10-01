"""
Embed the Airbnb catalog titles with Qwen3-Embedding-0.6B.

Client for the same vLLM embedding server the Amazon pipeline used
(vllm/embed_vllm_serve.sh, port 8124), so the model and serving stack match
the Amazon catalog exactly. Start the server first:

  conda run -n vllmenv bash vllm/embed_vllm_serve.sh   # (or the vllm env)

Reads the per-city parquets produced by ingest.py, embeds title_for_embedding
with the listing's city appended, and writes the same rows plus a
title_embedding column to a sibling catalog_embed/ tree -- the inline-embedding
layout the Amazon merged catalog uses, so ioutils can point at the directory
unchanged.

Retrieval is whole-catalog embedding similarity with no category mask, so the
city must live in the embedding itself: requirements name their city and
listings carry theirs via the appended sentence. This keeps "one stay per
city" semantics without touching the search code.

Usage:
  python -m gquery.prep.airbnb.embed
"""

from __future__ import annotations

import os
from pathlib import Path

import backoff
import click
import openai
import pandas as pd
from openai import OpenAI

DEFAULT_IN = os.path.expanduser("~/dataset/inside_airbnb/catalog")
DEFAULT_OUT = os.path.expanduser("~/dataset/inside_airbnb/catalog_embed")
MODEL = "Qwen/Qwen3-Embedding-0.6B"
BASE_URL = "http://localhost:8124/v1"
BATCH = 256


@backoff.on_exception(backoff.expo, (openai.OpenAIError,), max_tries=6)
def _embed_batch(client: OpenAI, model: str, texts):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def embed_texts(client: OpenAI, model: str, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        out.extend(_embed_batch(client, model, texts[i:i + BATCH]))
    return out


@click.command()
@click.option("--in-dir", type=str, default=DEFAULT_IN, show_default=True)
@click.option("--out-dir", type=str, default=DEFAULT_OUT, show_default=True)
@click.option("--model", type=str, default=MODEL, show_default=True)
@click.option("--base-url", type=str, default=BASE_URL, show_default=True)
def main(in_dir, out_dir, model, base_url):
    client = OpenAI(api_key="token-somethinghere", base_url=base_url)
    cities = sorted(p.name for p in Path(in_dir).iterdir() if p.is_dir())
    print(f"{len(cities)} cities from {in_dir}")

    for city in cities:
        src = Path(in_dir) / city / "batch_000000.parquet"
        dst_dir = Path(out_dir) / city
        dst = dst_dir / "batch_000000.parquet"
        if dst.exists():
            print(f"  {city}: exists, skipping")
            continue
        df = pd.read_parquet(src)
        city_text = city.replace("_", " ")
        texts = [(t if isinstance(t, str) and t.strip() else " ")
                 + f" Located in {city_text}."
                 for t in df.title_for_embedding]
        df["title_embedding"] = embed_texts(client, model, texts)
        dst_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        dim = len(df.title_embedding.iloc[0])
        print(f"  {city}: {len(df)} rows embedded (dim={dim}) -> {dst}",
              flush=True)

    print("done")


if __name__ == "__main__":
    main()
