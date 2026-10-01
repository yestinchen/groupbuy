#!/bin/bash
# Shared settings for every driver under scripts/. Source it, do not run it.
#
# Every value can be overridden from the environment. The defaults are the
# paper's: alpha=0.5, beta=0.8, b=100 (PBS beam width), M=500 (PTS iteration
# budget), kappa=50 (PTS candidate cap), 100 queries, 60 s per solver call.

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "${CONDA_ENV:-llmsq}" ]; then
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "${CONDA_ENV:-llmsq}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_ROOT="${DATASET_ROOT:-$HOME/dataset}"

# where the campaign runners write and the figure builders read
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$REPO_ROOT/results/campaign}"
export MEMORY_GRIDS_ROOT="${MEMORY_GRIDS_ROOT:-$REPO_ROOT/results/memory_grids}"
export FIGURES_ROOT="${FIGURES_ROOT:-$REPO_ROOT/data_figures/exp7}"

# single-threaded numerics for every timed measurement
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cgq-mpl}"

QUERY_SETTING="${QUERY_SETTING:-e10_1k_100q}"
NUM_QUERIES="${NUM_QUERIES:-100}"
ALPHA="${ALPHA:-0.5}"
BETA="${BETA:-0.8}"
TIMEOUT="${TIMEOUT:-60.0}"
REF_TIMEOUT="${REF_TIMEOUT:-600.0}"
DEFAULT_BEAM_SIZE="${DEFAULT_BEAM_SIZE:-100}"
DEFAULT_EXPANSION_BUDGET="${DEFAULT_EXPANSION_BUDGET:-500}"
DEFAULT_CANDIDATE_LIMIT="${DEFAULT_CANDIDATE_LIMIT:-50}"
BETA_SWEEP="${BETA_SWEEP:-0.4,0.5,0.6,0.7,0.8,0.9}"

# LLM and embedding servers used by the data preparation (scripts/prep/serve)
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8125/v1}"
EMBED_MODEL="${EMBED_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
EMBED_BASE_URL="${EMBED_BASE_URL:-http://localhost:8124/v1}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"
