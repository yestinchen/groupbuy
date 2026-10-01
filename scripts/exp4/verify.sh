#!/bin/bash
# The two gates every measurement depends on. Both must print PASS.
#
#  verify_exact     PBS without beam truncation (the reference frontier of every
#                   recall and HV number) equals exhaustive enumeration on every
#                   enumerable query of the default workloads.
#  verify_weighted  the exact weighted ILP, weighted PBS and weighted PTS agree
#                   with enumeration on random instances.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

for setting in e10_1k_100q airbnb_12c; do
  python -m gquery.alg.exp4.verify_exact --num-queries "$NUM_QUERIES" --query-setting "$setting" \
    --alpha "$ALPHA" --beta "$BETA" --max-combos 50000000 --bf-timeout 60 \
    --exp-name "verify_exact_$setting" 2>&1 | tee "$LOG_DIR/verify_exact_$setting.log"
done
python -m gquery.alg.exp5.verify_weighted 2>&1 | tee "$LOG_DIR/verify_weighted.log"
grep -q '^PASS' "$LOG_DIR/verify_weighted.log" || { echo "verify_weighted did not PASS" >&2; exit 1; }
