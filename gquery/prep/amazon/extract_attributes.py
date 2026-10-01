import asyncio

from gquery.prep.amazon.constants import INTERESTING_CATEGORIES
from openai import AsyncOpenAI, OpenAI
import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import backoff
import openai
import re

def create_local_client():
    return OpenAI(
        api_key='token-somethinghere', 
        base_url="http://localhost:8123/v1",
        timeout=10.0,
    )

def create_local_async_client():
    return AsyncOpenAI(
        api_key='token-somethinghere',
        base_url="http://localhost:8123/v1",
        timeout=10.0,
    )

PROMPT_TEMPLATE = """Below is the title or the description of an Amazon product. Please extract the representative requirements of keywords that this product matches, separated by commas. Please include all the relavent keywords as much as possible.
{content}
"""
@backoff.on_exception(
    backoff.expo,                      # exponential backoff: 1s, 2s, 4s, 8s...
    (openai.OpenAIError,),  # retry for all requests errors
    max_time=3                        # stop after 60 seconds total
)
def extract_feature(client, text, model_name):
    prompt = PROMPT_TEMPLATE.format(content=text)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stop=["\n"]
    )
    res = response.choices[0].message.content
    # print('raw:', text)
    # print('extracted:', res)
    # map it to a list.
    res = res.split(',')
    res = [r.strip() for r in res]
    return res

@backoff.on_exception(
    backoff.expo,
    (openai.OpenAIError,),
    max_time=3
)
async def extract_feature_async(client, text, model_name):
    prompt = PROMPT_TEMPLATE.format(content=text)
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stop=["\n"]
    )
    res = response.choices[0].message.content
    res = res.split(',')
    return [r.strip() for r in res]

def _normalize_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()

def _feats_to_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, str)):
        return ", ".join(_normalize_text(v) for v in value if _normalize_text(v))
    return _normalize_text(value)

def process_file(file, model_name, input_folder, output_folder, skip_features=False, max_concurrent=64):
    """
    Process a single parquet file and extract features.
    Produces one output file per input file.
    Skips processing if output file already exists.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    file_pure_name = file.name.split('.')[0]
    output_file = Path(output_folder) / f'{file_pure_name}.parquet'
    
    # Check if output file already exists
    if output_file.exists():
        print(f"  Skipping {file.name}: output file {output_file.name} already exists")
        return
    
    df = pq.read_table(file).to_pandas()
    titles = df["title"].tolist() if "title" in df.columns else [""] * len(df)
    feats = df["features"].tolist() if "features" in df.columns else [None] * len(df)

    async def run_all():
        client = create_local_async_client()
        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        row_results = [None] * len(df)
        pbar = tqdm(total=len(df), desc=f"Processing file {file.name}", unit="row")

        async def wrap(i, title, feature_text):
            async with semaphore:
                title_text = _normalize_text(title)
                feature_text_str = _feats_to_text(feature_text)
                try:
                    title_feats = await asyncio.wait_for(
                        extract_feature_async(client, title_text, model_name),
                        timeout=10.0,
                    )
                    if skip_features:
                        features_feats = []
                    else:
                        features_feats = await asyncio.wait_for(
                            extract_feature_async(client, feature_text_str, model_name),
                            timeout=10.0,
                        )
                    row_results[i] = (True, title_feats, features_feats)
                except Exception as exc:
                    print(f"  Warning: row {i} timed out/failed ({exc}); dropping row")
                    row_results[i] = (False, None, None)
            pbar.update(1)

        try:
            await asyncio.gather(*(wrap(i, titles[i], feats[i]) for i in range(len(df))))
            return row_results
        finally:
            pbar.close()
            await client.close()

    row_results = asyncio.run(run_all())
    keep_mask = [result is not None and result[0] for result in row_results]
    kept_df = df.loc[keep_mask].reset_index(drop=True)
    kept_df["title_feats"] = [result[1] for result in row_results if result is not None and result[0]]
    kept_df["features_feats"] = [result[2] for result in row_results if result is not None and result[0]]
    dropped = len(df) - len(kept_df)
    kept_df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"  Saved {len(kept_df)} rows to {output_file.name} (dropped {dropped})")
    

def get_batch_number(filename):
    """
    Extract batch number from filename for sorting.
    Handles formats like:
    - batch_000000.parquet -> 0
    - batch_000001.parquet -> 1
    - raw_items_b_000000.parquet -> 0 (if _b_ format)
    """
    try:
        if '_b_' in filename:
            # Format: name_b_000000.parquet
            parts = filename.split('_b_')
            if len(parts) > 1:
                return int(parts[1].split('.')[0])
        elif filename.startswith('batch_'):
            # Format: batch_000000.parquet
            return int(filename.split('_')[1].split('.')[0])
        # Fallback: try to extract any number from filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])
        return 0
    except:
        return 0

def process_category(model_name, input_folder, output_folder, category, skip_features=False, max_concurrent=64):
    """
    Process all parquet files in a category folder, in order by batch number.
    """
    # Get all parquet files
    files = list(Path(input_folder).glob('*.parquet'))
    
    # Sort files by batch number to process in order
    files.sort(key=lambda f: get_batch_number(f.name))
    
    if not files:
        print(f"  Warning: No parquet files found in {input_folder}")
        return
    
    print(f"  Found {len(files)} batch file(s), processing in order...")
    
    for file in tqdm(files, desc=f"Processing category {category}"):
        if file.name.endswith('.parquet'):
            process_file(file, model_name, input_folder, output_folder, skip_features, max_concurrent)

@click.command()
@click.option(
    '--model-name', type=str, required=True, 
    help='The name of the model to use', 
    default='qwen/Qwen2.5-3B-Instruct'
)
@click.option(
    '--input-folder', type=str, required=True, 
    help='The input folder to use', 
    default='~/dataset/amazon_review_2023/meta_parquet_clean'
)
@click.option(
    '--output-folder', type=str, required=True, 
    help='The output folder to use', 
    default='~/dataset/amazon_review_2023/meta_parquet_clean_attributes'
)
@click.option(
    '--category', type=str, required=False, 
    help='The category to use', 
    # default='All_Beauty'
    default='Baby_Products'
)
@click.option(
    '--skip-features', type=bool, required=False, 
    help='Whether to skip features extraction', 
    default=True
)
@click.option(
    '--max-concurrent', type=int, required=False,
    help='Maximum number of rows processed concurrently per file. This is the main vLLM batching knob.',
    default=128
)
def main(model_name, input_folder, output_folder, category, skip_features, max_concurrent):
    if category is None:
        for category in INTERESTING_CATEGORIES:
            input_folder_category = Path(input_folder) / category
            output_folder_category = Path(output_folder) / category
            process_category(model_name, input_folder_category, output_folder_category, category, skip_features, max_concurrent)
    else:
        input_folder_category = Path(input_folder) / category
        output_folder_category = Path(output_folder) / category
        process_category(model_name, input_folder_category, output_folder_category, category, skip_features, max_concurrent)

if __name__ == "__main__":
    main()
