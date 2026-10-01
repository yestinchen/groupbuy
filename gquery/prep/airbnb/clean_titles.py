"""
Clean Airbnb listing titles with Qwen2.5-7B-Instruct, mirroring the Amazon
pipeline (gquery/prep/amazon/refine_titles.py): the raw name is noisy
marketing text (emoji, separators, station-distance shorthand), so a chat
model rewrites it into a short natural stay title that becomes
title_for_embedding. The city is NOT put into the title here -- embed.py
appends "Located in <city>." at embedding time, same as the raw-title path.

Start the generate server first (same one the Amazon pipeline used):
  conda run -n vllmenv bash vllm/tools_vllm_serve.sh    # port 8123

Usage:
  python -m gquery.prep.airbnb.clean_titles
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import click
import pandas as pd
from openai import AsyncOpenAI

DEFAULT_IN = os.path.expanduser("~/dataset/inside_airbnb/catalog")
DEFAULT_OUT = os.path.expanduser("~/dataset/inside_airbnb/catalog_titled")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_URL = "http://localhost:8123/v1"

PROMPT = """You are rewriting an Airbnb listing title for retrieval.

Turn the following original listing name, description snippet, and extracted attributes into a short, natural stay title.

Rules:
- Use the original name and description as the primary source of truth and the attributes as secondary evidence.
- Preserve reliable facts: the kind of place, number of guests or bedrooms, notable amenities, and the neighbourhood if stated.
- Remove emoji, decorative symbols, separators, promotional phrases, and station-distance shorthand.
- Do not invent amenities, views, or qualities that are not supported by the source.
- Keep the kind of place first, then the most important attributes.
- Keep it fairly compact, but allow enough detail for identity: usually 6 to 18 words.
- Prefer a clean title-like phrase over a keyword list.
- Do not use bullet points or extra explanation.
- Do not include the city name.

Examples:
Name: *NEW* Cozy flat|5min->Shinjuku Sta.|MAX 4ppl|WiFi
Description: Our apartment is a 5 minute walk from the station...
Attributes: entire_home, rental_unit, sleeps_3_4, wifi, kitchen
Title: Cozy apartment for 4 near the station with wifi and kitchen

Name: Chic & Sunny Le Marais Hideaway - AC + Balcony!
Description: Beautiful one-bedroom in the heart of Le Marais with air conditioning...
Attributes: entire_home, rental_unit, sleeps_1_2, air_conditioning, balcony
Title: Sunny one-bedroom apartment in Le Marais with air conditioning and balcony

Now rewrite this listing.
Name: {name}
Description: {desc}
Attributes: {attrs}
Title:"""


def _split_name(text: str) -> tuple:
    '''title_for_embedding from ingest is "<name>. <description>"; recover the
    two halves (the name never contains ". " in practice is not guaranteed, so
    split on the first occurrence and cap the description snippet).'''
    if ". " in text:
        name, desc = text.split(". ", 1)
    else:
        name, desc = text, ""
    return name.strip(), desc.strip()[:300]


def _clean(out: str) -> str:
    t = re.sub(r"\s+", " ", str(out)).strip().strip('"')
    return t.split("\n")[0][:200]


async def _one(client, model, sem, name, desc, attrs, temperature, max_tokens):
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": PROMPT.format(
                        name=name, desc=desc, attrs=attrs)}],
                    temperature=temperature, max_tokens=max_tokens)
                t = _clean(r.choices[0].message.content)
                if t:
                    return t
            except Exception:
                await asyncio.sleep(2 ** attempt)
        return name  # keep the raw name rather than fail the row


async def _run_city(client, model, df, max_concurrent, temperature, max_tokens):
    sem = asyncio.Semaphore(max_concurrent)
    tasks = []
    for t, kw in zip(df.title_for_embedding, df.keywords):
        name, desc = _split_name(t)
        attrs = ", ".join(list(kw)[:10])
        tasks.append(_one(client, model, sem, name, desc, attrs,
                          temperature, max_tokens))
    return await asyncio.gather(*tasks)


@click.command()
@click.option("--in-dir", type=str, default=DEFAULT_IN, show_default=True)
@click.option("--out-dir", type=str, default=DEFAULT_OUT, show_default=True)
@click.option("--model", type=str, default=MODEL, show_default=True)
@click.option("--base-url", type=str, default=BASE_URL, show_default=True)
@click.option("--max-concurrent", type=int, default=64, show_default=True)
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--max-tokens", type=int, default=64, show_default=True)
def main(in_dir, out_dir, model, base_url, max_concurrent, temperature,
         max_tokens):
    client = AsyncOpenAI(api_key="token-somethinghere", base_url=base_url)
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
        titles = asyncio.run(_run_city(client, model, df, max_concurrent,
                                       temperature, max_tokens))
        df["title_raw"] = df.title_for_embedding
        df["title_for_embedding"] = titles
        dst_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        print(f"  {city}: {len(df)} titles cleaned -> {dst}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
