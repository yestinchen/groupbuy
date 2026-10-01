# Data preparation

Both pipelines write under `DATASET_ROOT` (default `~/dataset`) at the paths
`gquery/ioutils.py` registers. The stages that call a language model or an
embedding model talk to OpenAI-compatible servers; `scripts/prep/serve/` starts
them with vLLM (Qwen2.5-7B-Instruct on port 8125, Qwen3-Embedding-0.6B on
port 8124). Every sampling step is seeded (42) and every rerun of a stage is
deterministic given its inputs.

## Amazon Reviews 2023

Raw input: the `meta_*.jsonl` files of the 11 categories under
`$DATASET_ROOT/amazon_review_2023/meta_jsonl/` (All_Beauty, Amazon_Fashion,
Appliances, Arts_Crafts_and_Sewing, Baby_Products, Cell_Phones_and_Accessories,
Electronics, Health_and_Personal_Care, Pet_Supplies, Sports_and_Outdoors,
Toys_and_Games).

```bash
bash scripts/prep/amazon.sh clean attributes titles keywords embed merge
bash scripts/prep/amazon.sh keyword_stats single_queries sample_10k single_query_analysis
bash scripts/prep/amazon.sh group_queries fixed_gs catalogs check
bash scripts/exp4/prewarm.sh            # then, with the retrieval caches in place:
bash scripts/prep/amazon.sh recal
```

| stage | module | output under `amazon_review_2023/` |
|---|---|---|
| `clean` | `gquery.prep.amazon.clean_to_parquet` | `meta_parquet_clean/<category>/` |
| `attributes` | `extract_attributes` (LLM) | `meta_parquet_clean_attributes/` |
| `titles` | `refine_titles` (LLM: cleaned title from the keywords of the original title) | `meta_parquet_clean_attributes_titled/` |
| `keywords` | `item_keywords` (LLM: product keywords per item) | `meta_parquet_clean_attributes_keywords/` |
| `embed` | `title_embeddings` (Qwen3-Embedding-0.6B on the cleaned titles) | `meta_parquet_clean_attributes_titled_embed/` |
| `merge` | `merge_embeddings` | `meta_parquet_clean_attributes_keywords_embed_merged/` (the full 2.14M item pool) |
| `keyword_stats` | `keyword_freq`, `keyword_filter` (LLM), `price_range_stats` | `analysis/keyword_freq/`, `analysis/product_keywords/`, `analysis/price_distribute/` |
| `single_queries` | `single_qgen` (LLM: one natural language requirement per product keyword, embedded) | `data_out/qgen/keyword_w_title/<category>.parquet` |
| `sample_10k` | `sample_catalog_10k` | `..._merged_10k_each/` (the 110K default catalog) |
| `single_query_analysis` | `single_qgen_analysis` (matches and eligible prices of every requirement on the 110K catalog at beta=0.8) | `analysis/embedding_price_distribution_10k_each/` |
| `group_queries` | `group_qgen` (3 to 8 requirements from distinct categories; the budget is the sum of the minimum eligible prices times a factor from [1.1, 1.5] with probability 0.6 and from [0.8, 0.99] otherwise, rounded to an integer) | `data_out/qgen/group_keyword_w_title_qgen10_v2/threshold_0.8/` (1,000 queries) |
| `fixed_gs` | `group_qgen_fixed_gs` | `data_out/qgen/group_keyword_w_title_qgen11_fixed_gs/threshold_0.8/*_gs{2,4,6,8}.parquet` |
| `catalogs` | `sample_catalog_nk` (nested 15K to 30K per category, the 0.5 share catalog of 1.1M items, and the full pool as `cat2m`) | `..._merged_{15k,20k,25k,30k}_each/`, `..._merged_cat1m/`, `..._merged_cat2m/` |
| `check` | `check_catalog` (row counts and nestedness gates) | |
| `recal` | `gquery.prep.recal_budgets` (each query's budget multiplier re-applied to the minimum feasible cost on the larger catalog) | `data_out/qgen/e10_1k_100q/queries_recal_{cat30k,cat1m,cat2m}.parquet` |

The 100-query benchmark `e10_1k_100q` is the first 100 queries of the
1,000-query pool that have a feasible assignment at alpha=0, verified by
exhaustive enumeration with a 60 s limit; the selected file
`data_out/qgen/e10_1k_100q/queries.parquet` ships with the data release.

## Inside Airbnb

Raw input: the detailed listings CSV of each of the 12 cities
(amsterdam, barcelona, berlin, london, los-angeles, mexico-city,
new-york-city, paris, rome, sydney, tokyo, toronto) under
`$DATASET_ROOT/inside_airbnb/raw/<city>.csv.gz`. `gquery/prep/airbnb/README.md`
documents the schema mapping, the pinned exchange rates, the rating
composite and the listing filters.

```bash
bash scripts/prep/airbnb.sh ingest titles embed pool calibrate queries fixed_gs
bash scripts/exp4/prewarm.sh            # then, with the retrieval caches in place:
bash scripts/prep/airbnb.sh ladder
```

| stage | module | output under `inside_airbnb/` |
|---|---|---|
| `ingest` | `gquery.prep.airbnb.ingest` (Amazon-schema catalogs, prices in USD, rating = mean of the six review subscores) | `catalog/<City>/` |
| `titles` | `clean_titles` (LLM) | `catalog_titled/` |
| `embed` | `embed` | `catalog_titled_embed/` (the 231K catalog) |
| `pool` | `qgen build-pool`, `qgen embed-pool` (templated stay requirements per city) | `data_out/qgen/airbnb_12c/pool*` |
| `calibrate` | `qgen calibrate` (candidate counts across thresholds, informational) | |
| `queries` | `qgen generate` (100 queries, 3 to 8 distinct cities, same budget rule as Amazon with per-requirement minimum prices rounded to integers) | `data_out/qgen/airbnb_12c/queries.parquet` |
| `fixed_gs` | `qgen generate --group-size-min N --group-size-max N` | `data_out/qgen/airbnb_12c/queries_gs{2,4,6,8}.parquet` |
| `ladder` | `sample_catalog_share` (nested per-city shares 1/8, 1/4, 1/2), `qgen generate` on the 29K rung, `recal_budgets` per rung | `catalog_titled_embed_lad{29k,58k,115k}/`, `data_out/qgen/airbnb_ladder/` |

## Checks before running experiments

```bash
ls $DATASET_ROOT/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_10k_each/
ls $DATASET_ROOT/amazon_review_2023/data_out/qgen/e10_1k_100q/queries.parquet
ls $DATASET_ROOT/amazon_review_2023/data_out/qgen/group_keyword_w_title_qgen11_fixed_gs/threshold_0.8/
ls $DATASET_ROOT/inside_airbnb/catalog_titled_embed/ $DATASET_ROOT/inside_airbnb/data_out/qgen/airbnb_12c/
```
