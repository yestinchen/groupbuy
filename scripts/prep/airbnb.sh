#!/bin/bash
# Inside Airbnb preparation, one stage per argument, in pipeline order.
#
#   bash scripts/prep/airbnb.sh ingest titles embed pool calibrate queries fixed_gs ladder
#
# Inputs: the detailed listings CSVs of the 12 cities under
# $DATASET_ROOT/inside_airbnb/raw/<city>.csv.gz (see gquery/prep/airbnb/README.md).
# The titles stage needs the Qwen2.5-7B-Instruct server and the embed stages
# the Qwen3-Embedding-0.6B server from scripts/prep/serve. Outputs land under
# $DATASET_ROOT/inside_airbnb/ at the paths gquery/ioutils.py expects.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

AB="$DATASET_ROOT/inside_airbnb"
CAT="$AB/catalog_titled_embed"
AIRBNB_BETA="${AIRBNB_BETA:-0.8}"

stage_ingest() {      # raw CSVs -> per-city parquet catalogs in the Amazon schema (USD prices, rating composite, keywords)
  python -m gquery.prep.airbnb.ingest --raw-dir "$AB/raw" --out-dir "$AB/catalog" --seed 42
}

stage_titles() {      # listing names cleaned by the LLM -> catalog_titled
  python -m gquery.prep.airbnb.clean_titles --in-dir "$AB/catalog" --out-dir "$AB/catalog_titled" \
    --model "$LLM_MODEL" --base-url "$LLM_BASE_URL"
}

stage_embed() {       # title embeddings -> catalog_titled_embed (the 231K catalog, SAMPLE_AIRBNB_12C)
  python -m gquery.prep.airbnb.embed --in-dir "$AB/catalog_titled" --out-dir "$CAT" \
    --model "$EMBED_MODEL" --base-url "$EMBED_BASE_URL"
}

stage_pool() {        # templated stay requirements per city, embedded once
  python -m gquery.prep.airbnb.qgen build-pool --seed 42
  python -m gquery.prep.airbnb.qgen embed-pool --base-url "$EMBED_BASE_URL"
}

stage_calibrate() {   # candidate set sizes across matching thresholds (informational)
  python -m gquery.prep.airbnb.qgen calibrate
}

stage_queries() {     # the 100 default queries, 3 to 8 cities each -> airbnb_12c
  python -m gquery.prep.airbnb.qgen generate --beta "$AIRBNB_BETA" --num-queries "$NUM_QUERIES" --seed 42
}

stage_fixed_gs() {    # fixed group size sets for the query length sweep (airbnb_gs2..gs8)
  for n in 2 4 6 8; do
    python -m gquery.prep.airbnb.qgen generate --beta "$AIRBNB_BETA" --num-queries "$NUM_QUERIES" --seed 42 \
      --group-size-min "$n" --group-size-max "$n" --out-name "queries_gs$n.parquet"
  done
}

stage_ladder() {      # nested per-city shares 1/8, 1/4, 1/2 (29K, 58K, 115K), ladder queries on the 29K rung, per-rung budgets
  python -m gquery.prep.airbnb.sample_catalog_share build --catalog "$CAT"
  python -m gquery.prep.airbnb.sample_catalog_share verify --catalog "$CAT"
  python -m gquery.prep.airbnb.qgen generate --beta "$AIRBNB_BETA" --num-queries "$NUM_QUERIES" --seed 42 \
    --catalog "${CAT}_lad29k" --out-dir "$AB/data_out/qgen/airbnb_ladder"
  for rung in lad58k lad115k lad231k; do   # needs the retrieval caches (GPU)
    python -m gquery.prep.recal_budgets --num-queries "$NUM_QUERIES" --alpha "$ALPHA" --beta "$BETA" \
      --src-key sample_gqueries_airbnb_ladder --base-setting airbnb_lad29k --target-setting "airbnb_$rung" \
      --dst-key "sample_gqueries_airbnb_ladder_recal_$rung"
  done
}

for stage in "$@"; do
  echo ">>> airbnb: $stage"
  "stage_$stage" 2>&1 | tee "$LOG_DIR/prep_airbnb_$stage.log"
done
