#!/bin/bash
# Amazon Reviews 2023 preparation, one stage per argument, in pipeline order.
#
#   bash scripts/prep/amazon.sh clean attributes titles keywords embed merge \
#        keyword_stats single_queries sample_10k single_query_analysis \
#        group_queries fixed_gs catalogs recal check
#
# Inputs: the raw meta_*.jsonl files under $DATASET_ROOT/amazon_review_2023/meta_jsonl/.
# The LLM stages (attributes, titles, keywords, keyword_stats, single_queries)
# need the Qwen2.5-7B-Instruct server and the embed stages need the
# Qwen3-Embedding-0.6B server from scripts/prep/serve. Outputs land under
# $DATASET_ROOT/amazon_review_2023/ at the paths gquery/ioutils.py expects
# (see DATAPREP.md).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

AR="$DATASET_ROOT/amazon_review_2023"
CATEGORIES=(All_Beauty Amazon_Fashion Appliances Arts_Crafts_and_Sewing Baby_Products
            Cell_Phones_and_Accessories Electronics Health_and_Personal_Care Pet_Supplies
            Sports_and_Outdoors Toys_and_Games)
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"

stage_clean() {        # raw jsonl -> meta_parquet_clean
  python -m gquery.prep.amazon.clean_to_parquet \
    --input-root "$AR/meta_jsonl" --output-root "$AR/meta_parquet_clean"
}

stage_attributes() {   # LLM attribute extraction -> meta_parquet_clean_attributes
  for c in "${CATEGORIES[@]}"; do
    python -m gquery.prep.amazon.extract_attributes \
      --input-folder "$AR/meta_parquet_clean" --output-folder "$AR/meta_parquet_clean_attributes" \
      --category "$c" --model-name "$LLM_MODEL" --base-url "$LLM_BASE_URL" \
      --max-concurrent "${MAX_CONCURRENT_ATTRIBUTES:-128}"
  done
}

stage_titles() {       # cleaned titles from keywords -> meta_parquet_clean_attributes_titled
  for c in "${CATEGORIES[@]}"; do
    python -m gquery.prep.amazon.refine_titles \
      --input-folder "$AR/meta_parquet_clean_attributes" --output-folder "$AR/meta_parquet_clean_attributes_titled" \
      --category "$c" --model-name "$LLM_MODEL" --base-url "$LLM_BASE_URL" \
      --max-concurrent "$MAX_CONCURRENT" --max-title-tokens 1024 --temperature 0.0
  done
}

stage_keywords() {     # per-item product keywords -> meta_parquet_clean_attributes_keywords
  for c in "${CATEGORIES[@]}"; do
    python -m gquery.prep.amazon.item_keywords \
      --input-folder "$AR/meta_parquet_clean_attributes_titled" --output-folder "$AR/meta_parquet_clean_attributes_keywords" \
      --category "$c" --model-name "$LLM_MODEL" --base-url "$LLM_BASE_URL" \
      --max-concurrent "${MAX_CONCURRENT_KEYWORDS:-128}"
  done
}

stage_embed() {        # title embeddings -> meta_parquet_clean_attributes_titled_embed
  python -m gquery.prep.amazon.title_embeddings \
    --input-folder "$AR/meta_parquet_clean_attributes_titled" --output-folder "$AR/meta_parquet_clean_attributes_titled_embed" \
    --category None --model-name "$EMBED_MODEL" --batch-size 64
}

stage_merge() {        # keywords + embeddings -> meta_parquet_clean_attributes_keywords_embed_merged (the full catalog)
  python -m gquery.prep.amazon.merge_embeddings \
    --input-root "$AR/meta_parquet_clean_attributes_keywords" --embed-root "$AR/meta_parquet_clean_attributes_titled_embed" \
    --output-root "$AR/meta_parquet_clean_attributes_keywords_embed_merged" --category None
}

stage_keyword_stats() {  # keyword frequencies, LLM keyword filter, price ranges per keyword -> analysis/
  mkdir -p "$AR/analysis/keyword_freq" "$AR/analysis/product_keywords" "$AR/analysis/price_distribute"
  for c in "${CATEGORIES[@]}"; do
    python -m gquery.prep.amazon.keyword_freq "$AR/meta_parquet_clean_attributes_keywords/$c" "$AR/analysis/keyword_freq/$c.csv" 50
    python -m gquery.prep.amazon.keyword_filter "$AR/analysis/keyword_freq/$c.csv" "$AR/analysis/product_keywords/$c.csv" "$LLM_MODEL" "$LLM_BASE_URL"
  done
  python -m gquery.prep.amazon.price_range_stats \
    "$AR/meta_parquet_clean_attributes_keywords_embed_merged" "$AR/analysis/product_keywords" "$AR/analysis/price_distribute" "" "${CATEGORIES[@]}"
}

stage_single_queries() {  # one natural-language requirement per product keyword -> data_out/qgen/keyword_w_title
  mkdir -p "$AR/data_out/qgen/keyword_w_title"
  for c in "${CATEGORIES[@]}"; do
    tmp="$(mktemp -d)"; ln -s "$AR/analysis/price_distribute/$c.csv" "$tmp/$c.csv"
    python -m gquery.prep.amazon.single_qgen --stats-dir "$tmp" --output-file "$AR/data_out/qgen/keyword_w_title/$c.parquet" \
      --model-name "$LLM_MODEL" --base-url "$LLM_BASE_URL" --embed-model-name "$EMBED_MODEL" --embed-base-url "$EMBED_BASE_URL" \
      --max-concurrent 16 --max-tokens 24 --temperature 0.0 --sample-size 0 --show-samples 0
    rm -rf "$tmp"
  done
}

stage_sample_10k() {   # 10,000 items per category -> the 110K default catalog (SAMPLE_10k_EACH)
  python -m gquery.prep.amazon.sample_catalog_10k
}

stage_single_query_analysis() {  # matching and price distributions of the single queries on the 110K catalog
  python -m gquery.prep.amazon.single_qgen_analysis \
    --query-root "$AR/data_out/qgen/keyword_w_title" --item-root "$AR/meta_parquet_clean_attributes_keywords_embed_merged_10k_each" \
    --output-root "$AR/analysis/embedding_price_distribution_10k_each" --thresholds 0.8
}

stage_group_queries() {  # the 1,000-query pool with the two-mode budget rule -> e10_1k_v2
  python -m gquery.prep.amazon.group_qgen \
    --query-root "$AR/data_out/qgen/keyword_w_title" --analysis-root "$AR/analysis/embedding_price_distribution_10k_each" \
    --output-root "$AR/data_out/qgen/group_keyword_w_title_qgen10_v2" \
    --total-num 1000 --threshold 0.8 --seed 42 --within-budget-fraction 0.6 \
    --easy-multiplier-low 1.1 --easy-multiplier-high 1.5 --multiplier-low 0.8 --multiplier-high 0.99
  echo "e10_1k_100q is the first 100 queries of this pool with a feasible assignment at alpha=0 (verified by"
  echo "exhaustive enumeration, 60 s limit); the selected parquet ships with the data release at"
  echo "  $AR/data_out/qgen/e10_1k_100q/queries.parquet"
}

stage_fixed_gs() {     # fixed group size sets for the query length sweep (e11_gs2..gs8)
  for gs in 2 4 6 8; do
    python -m gquery.prep.amazon.group_qgen_fixed_gs --group-size "$gs" --total-num 100 --show-samples 0
  done
}

stage_catalogs() {     # nested larger catalogs: 15k..30k per category, cat1m (share 0.5), cat2m (full pool)
  for nk in 15000 20000 25000 30000; do
    python -m gquery.prep.amazon.sample_catalog_nk --items-per-category "$nk" --force
  done
  python -m gquery.prep.amazon.sample_catalog_nk --share 0.5 --force \
    --baseline-dir "$AR/meta_parquet_clean_attributes_keywords_embed_merged_30k_each" \
    --output-dir "$AR/meta_parquet_clean_attributes_keywords_embed_merged_cat1m"
  # cat2m is the full 11-category pool: one symlink per category, no sampling
  full="$AR/meta_parquet_clean_attributes_keywords_embed_merged_cat2m"; mkdir -p "$full"
  for c in "${CATEGORIES[@]}"; do ln -sfn "$AR/meta_parquet_clean_attributes_keywords_embed_merged/$c" "$full/$c"; done
}

stage_recal() {        # budgets re-anchored to each larger catalog's minimum feasible cost (needs the retrieval caches, GPU)
  for target in e11_cat30k:cat30k e12_cat1m:cat1m e12_cat2m:cat2m; do
    python -m gquery.prep.recal_budgets --num-queries "$NUM_QUERIES" --alpha "$ALPHA" --beta "$BETA" \
      --base-setting e11_cat10k --target-setting "${target%%:*}" \
      --dst-key "sample_gqueries_e10_1k_100q_recal_${target##*:}"
  done
}

stage_check() {        # nestedness and row-count gates on the ladder catalogs
  for d in 15k_each 20k_each 25k_each 30k_each cat1m cat2m; do
    python -m gquery.prep.amazon.check_catalog --catalog-dir "$AR/meta_parquet_clean_attributes_keywords_embed_merged_$d"
  done
}

for stage in "$@"; do
  echo ">>> amazon: $stage"
  "stage_$stage" 2>&1 | tee "$LOG_DIR/prep_amazon_$stage.log"
done
