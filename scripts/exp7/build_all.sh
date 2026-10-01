#!/bin/bash
# Build every exp7 figure panel and table from validated campaign measurements.
#
# Reads  $CAMPAIGN_ROOT/<family>/<job>/  (the run_queue outputs, validated)
#        $MEMORY_GRIDS_ROOT/<setting>/    (the memory grid jobs, validated)
# Writes $FIGURES_ROOT/exp7_<topic>/       (PDF + PNG + the CSV behind each panel,
#                                          the .tex tables, and a manifest.json
#                                          with the hash of every generated file)
#
# Order matters: sensitivity imports build_default, timing and the two-method
# panel import build_scalability, the memory increase table reads build_memory's
# table, and the elapsed-time panels reread the deadlines CSVs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$REPO_ROOT/scripts/exp7"
mkdir -p "$FIGURES_ROOT"
for step in build_default build_sensitivity build_ablation build_deadlines build_scalability build_timing \
            build_two_method_scaling build_memory build_memory_increase_table build_weighted; do
  echo ">>> $step"
  python "$HERE/$step.py" > "$LOG_DIR/exp7_$step.log" 2>&1 || { echo "$step failed, see $LOG_DIR/exp7_$step.log" >&2; exit 1; }
done
echo ">>> relabel_elapsed_time"
python "$HERE/relabel_elapsed_time.py" --dataset both --grid 16 > "$LOG_DIR/exp7_relabel_elapsed_time.log" 2>&1
echo "figures and tables under $FIGURES_ROOT"
