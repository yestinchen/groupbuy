from __future__ import annotations

import asyncio
import csv
import re
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


PROMPT = """You are filtering Amazon search keywords.

Decide whether the keyword names a PRODUCT TYPE.

Keep it only if the keyword clearly names the thing being sold, a product category, or a specific item type.
Reject it if it is mainly an attribute, material, color, size, quantity, style, brand, model, feature, or a generic shopping word.

Examples:
Keyword: screen protector
Answer: YES | names a product type

Keyword: dog collar
Answer: YES | names a product type

Keyword: waterproof
Answer: NO | attribute, not a product type

Keyword: pack
Answer: NO | generic quantity word, not a product type

Keyword: {keyword}
Answer:"""


def create_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key="token-somethinghere", base_url=base_url)


def parse_decision(text: str) -> tuple[bool, str]:
    cleaned = (text or "").strip()
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    m = re.match(r"^(yes|no)\s*(?:[|:\-]\s*(.*))?$", first_line, flags=re.I)
    if m:
        keep = m.group(1).lower() == "yes"
        reason = (m.group(2) or "").strip()
        return keep, reason
    if first_line.lower().startswith("yes"):
        return True, first_line[3:].lstrip(" |:-")
    if first_line.lower().startswith("no"):
        return False, first_line[2:].lstrip(" |:-")
    return False, first_line


async def judge_keyword(client: AsyncOpenAI, keyword: str, model_name: str, semaphore: asyncio.Semaphore) -> tuple[bool, str]:
    prompt = PROMPT.format(keyword=keyword)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                stop=["\n\n"],
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return False, f"model_error: {type(exc).__name__}"
    return parse_decision(raw)


def main(input_file: str, output_file: str, model_name: str, base_url: str, max_concurrent: int, min_item_count: int) -> None:
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    required = {"keyword", "item_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")

    df = df[df["item_count"] >= min_item_count].copy()
    if df.empty:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"No keywords met MIN_ITEM_COUNT={min_item_count} for {input_path.name}")
        print(f"Wrote empty output to {output_path}")
        return

    keywords = df["keyword"].astype(str).tolist()
    total = len(keywords)
    client = create_client(base_url)
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    results: list[tuple[bool, str] | None] = [None] * total
    pbar = tqdm(total=total, desc=input_path.name, unit="keyword")

    async def wrap(i: int, keyword: str):
        keep, reason = await judge_keyword(client, keyword, model_name, semaphore)
        results[i] = (keep, reason)
        pbar.update(1)

    async def run_all():
        try:
            await asyncio.gather(*(wrap(i, keywords[i]) for i in range(total)))
        finally:
            pbar.close()
            await client.close()

    asyncio.run(run_all())

    keep_mask = [bool(item and item[0]) for item in results]
    kept_df = df.loc[keep_mask].copy()
    kept_df["reason"] = [item[1] if item else "" for item in results if item and item[0]]
    kept_df = kept_df.sort_values(["item_count", "keyword"], ascending=[False, True]).reset_index(drop=True)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    judged = len(df)
    kept = len(kept_df)
    print(f"Judged {judged:,} keywords from {input_path.name}")
    print(f"Kept {kept:,} product-type keywords")
    print(f"Wrote filtered keywords to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise SystemExit("Usage: python - INPUT_FILE OUTPUT_FILE MODEL_NAME BASE_URL MAX_CONCURRENT MIN_ITEM_COUNT")
    main(
        input_file=sys.argv[1],
        output_file=sys.argv[2],
        model_name=sys.argv[3],
        base_url=sys.argv[4],
        max_concurrent=int(sys.argv[5]),
        min_item_count=int(sys.argv[6]),
    )
