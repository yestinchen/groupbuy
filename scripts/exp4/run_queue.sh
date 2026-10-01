#!/bin/bash
# Run campaign jobs one at a time: launch the adapter, wait, then validate the
# finished output with the family's reconstruction script (which recomputes
# every recall and hypervolume from the archived candidate sets without
# invoking a solver). Stops at the first failed job or failed validation.
#
#   bash scripts/exp4/run_queue.sh [--matrix scripts/exp4/jobs.json] [--expected 100] JOB_ID...
#   bash scripts/exp4/run_queue.sh --all            # every job of jobs.json, in matrix order
#   bash scripts/exp4/run_queue.sh --memory-grids   # the isolated epsilon-ILP g=4/16 memory jobs
#
# One GPU must be visible (CUDA_VISIBLE_DEVICES); the adapter refuses otherwise.
# Launch detached (setsid nohup ... &) for the full matrix: it takes days.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$REPO_ROOT/scripts/exp4"
EXP7="$REPO_ROOT/scripts/exp7"
MATRIX="$HERE/jobs.json"
EXPECTED=100
QUEUE_LOG="$LOG_DIR/queue.log"
while [ $# -gt 0 ]; do
  case "$1" in
    --matrix) MATRIX=$2; shift 2;;
    --expected) EXPECTED=$2; shift 2;;
    --log) QUEUE_LOG=$2; shift 2;;
    --all) set -- $(python -c "import json;print(' '.join(j['id'] for j in json.load(open('$MATRIX'))['jobs']))"); break;;
    --memory-grids) MATRIX="$HERE/memory_grid_jobs.json"; set -- $(python -c "import json;print(' '.join(j['id'] for j in json.load(open('$MATRIX'))['jobs'] if not j['id'].startswith('pilot')))"); break;;
    *) break;;
  esac
done
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
validator_for() {
  case "$1" in
    *ablation*|*budget_split*|*deceptive*|*_case*) echo reconstruct_ablation;;
    sensitivity.*) echo reconstruct_sensitivity;;
    scalability.*) echo reconstruct_scalability;;
    deadlines.*) echo reconstruct_deadlines;;
    memory.*) echo reconstruct_memory;;
    default.*) echo reconstruct_default;;
    weighted.amazon_beta_diagnostic) echo reconstruct_weighted_diagnostic;;
    weighted.*) echo reconstruct_weighted;;
    e1*|airbnb_12c) echo validate_memory_grid;;   # memory grid matrix
    *) echo none;;
  esac
}
for job in "$@"; do
  output=$(python "$HERE/run_job.py" --matrix "$MATRIX" --job "$job" --print-output)
  if [ -f "$output/reconstruction.json" ] || [ -f "$output/validation.json" ]; then echo "$(stamp) skip $job (already validated at $output)" | tee -a "$QUEUE_LOG"; continue; fi
  echo "$(stamp) launch $job -> $output" | tee -a "$QUEUE_LOG"
  python "$HERE/run_job.py" --matrix "$MATRIX" --job "$job" >> "$QUEUE_LOG" 2>&1
  code=$?
  echo "$(stamp) $job exit=$code" | tee -a "$QUEUE_LOG"
  if [ "$code" -ne 0 ]; then echo "$(stamp) STOP: $job failed" | tee -a "$QUEUE_LOG"; exit 1; fi
  family=$(validator_for "$job")
  if [ "$family" = none ]; then echo "$(stamp) no validator for $job" | tee -a "$QUEUE_LOG"; continue; fi
  echo "$(stamp) validate $job with $family" | tee -a "$QUEUE_LOG"
  if [ "$family" = validate_memory_grid ]; then
    python "$EXP7/validate_memory_grid.py" "$output" >> "$QUEUE_LOG" 2>&1
  else
    python "$EXP7/$family.py" --expected "$EXPECTED" "$output" >> "$QUEUE_LOG" 2>&1
  fi
  code=$?
  echo "$(stamp) validate $job exit=$code" | tee -a "$QUEUE_LOG"
  if [ "$code" -ne 0 ]; then echo "$(stamp) STOP: validation of $job failed" | tee -a "$QUEUE_LOG"; exit 1; fi
done
echo "$(stamp) QUEUE COMPLETE" | tee -a "$QUEUE_LOG"
