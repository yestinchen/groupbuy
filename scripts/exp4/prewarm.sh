#!/bin/bash
# Retrieve and cache the candidate sets of every query setting the job matrix
# uses (GPU). The campaign jobs retrieve afresh anyway (scripts/exp4/run_job.py
# bypasses this cache); the cache is what makes the gates and the unit-level
# reruns CPU-only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

# Amazon default workload at every threshold and relaxation of the sweeps
python -m gquery.alg.exp4.prewarm --settings e10_1k_100q --betas "$BETA_SWEEP" \
  --num-queries "$NUM_QUERIES" --alpha "$ALPHA" 2>&1 | tee "$LOG_DIR/prewarm_amazon.log"
for a in 0.1 0.25 0.75 1.0; do
  python -m gquery.alg.exp4.prewarm --settings e10_1k_100q --betas "$BETA" --num-queries "$NUM_QUERIES" --alpha "$a"
done 2>&1 | tee -a "$LOG_DIR/prewarm_amazon.log"
# Amazon catalog and length ladders
python -m gquery.alg.exp4.prewarm \
  --settings e11_cat10k,e11_cat15k,e11_cat20k,e11_cat25k,e11_cat30k,e11_gs2,e11_gs4,e11_gs6,e11_gs8,e12_cat1m,e12_cat2m,e11_cat30k_recal,e12_cat1m_recal,e12_cat2m_recal \
  --betas "$BETA" --num-queries "$NUM_QUERIES" --alpha "$ALPHA" 2>&1 | tee "$LOG_DIR/prewarm_amazon_scale.log"
# Airbnb
python -m gquery.alg.exp4.prewarm --settings airbnb_12c --betas 0.7,0.75,0.8,0.85,0.9 \
  --num-queries "$NUM_QUERIES" --alpha "$ALPHA" 2>&1 | tee "$LOG_DIR/prewarm_airbnb.log"
for a in 0.1 0.25 0.75 1.0; do
  python -m gquery.alg.exp4.prewarm --settings airbnb_12c --betas "$BETA" --num-queries "$NUM_QUERIES" --alpha "$a"
done 2>&1 | tee -a "$LOG_DIR/prewarm_airbnb.log"
python -m gquery.alg.exp4.prewarm \
  --settings airbnb_gs2,airbnb_gs4,airbnb_gs6,airbnb_gs8,airbnb_lad29k,airbnb_lad58k,airbnb_lad58k_recal,airbnb_lad115k,airbnb_lad115k_recal,airbnb_lad231k,airbnb_lad231k_recal \
  --betas "$BETA" --num-queries "$NUM_QUERIES" --alpha "$ALPHA" 2>&1 | tee "$LOG_DIR/prewarm_airbnb_scale.log"
