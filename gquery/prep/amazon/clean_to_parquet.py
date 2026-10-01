import gzip
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from gquery.prep.amazon.constants import INTERESTING_CATEGORIES
import click

def clean_to_parquet(input_file_path, output_file_path):
    df = []
    columns = set()
    with gzip.open(input_file_path, 'rb') as fp:
        for idx, line in tqdm(enumerate(fp)):
            obj = json.loads(line.strip())
            # things to check:
            #  price must not be None.
            if obj['price'] is None:
                continue
            try:
                obj['price'] = float(obj['price'])
            except:
                # ignore the row.
                continue
            del obj['images']
            if 'videos' in obj:
                del obj['videos']
            del obj['bought_together']
            # for details, some have empty string as key, make it _default_
            new_details = {}
            if 'details' in obj:
                for key, value in obj['details'].items():
                    if value == None or value == '':
                        continue
                    if key == '':
                        new_details['_default_'] = value
                    else:
                        new_details[key] = str(value)
            obj['details'] = new_details
            df.append(obj)
            # get first level key as columns.
            columns = columns.union(set(obj.keys()))

    # don't expand object columns, only keep it one level deep.
    # split into batches of 100000 rows, and save to parquet.

    output_path = Path(output_file_path)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = 10000
    
    for i in range(0, len(df), batch_size):
        df_batch = df[i:i+batch_size]
        df_batch = pd.DataFrame(df_batch, columns=list(columns))
        batch_name = f'batch_{i//batch_size:06d}.parquet'
        print(f"Writing {batch_name}, rows: {len(df_batch)}")
        df_batch.to_parquet(f'{output_path}/{batch_name}')
    
    print(f"Saving to {output_path}, rows: {len(df)}")


# @click.command()
# @click.option('--idx', type=int, required=True, help='The index of the category to clean')
@click.command()
@click.option(
    "--input-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("~/dataset/amazon_review_2023/meta_jsonl").expanduser(),
    show_default=True,
    help="Folder containing the downloaded meta JSONL.GZ files.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("~/dataset/amazon_review_2023/meta_parquet_clean").expanduser(),
    show_default=True,
    help="Folder to write cleaned parquet batches into.",
)
def main(input_root, output_root):
    for category in INTERESTING_CATEGORIES:
        print(f"Cleaning {category}")
        input_file_path = input_root / f"{category}.jsonl.gz"
        output_file_path = output_root / category
        clean_to_parquet(str(input_file_path), str(output_file_path))

if __name__ == "__main__":
    main()
