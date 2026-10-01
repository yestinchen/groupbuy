# Reproducing the Experiments

This repository contains the CGQ solvers (PBS, PTS, the conventional Pareto
DP, the epsilon-constraint ILP, brute force, the exact weighted ILP and the
weighted PBS and PTS) and the experiment suite that produces every figure and
table of the paper. The whole suite is driven from `scripts/`.

If you need to build the datasets first, see [DATAPREP.md](DATAPREP.md).

## 1. Environment Setup

The shell scripts assume a Conda environment named `llmsq`.

```bash
conda env create -f environment.yml
conda activate llmsq
pip install -e .
```

Notes:

- `environment.yml` installs Python 3.11 and `requirements.txt`.
- the scripts activate `llmsq` themselves (`scripts/env.sh`), but test the
  environment once manually first.
- run everything from the repository root.
- one GPU is needed for retrieval. The solvers, the gates and the figures are
  CPU-only.
- the data preparation needs vLLM servers for Qwen2.5-7B-Instruct and
  Qwen3-Embedding-0.6B (`scripts/prep/serve/`).

## 2. Dataset Layout

The code loads datasets from `~/dataset` by default; override with:

```bash
export DATASET_ROOT=/path/to/your/dataset/root
```

The query settings used by the suite (resolved in `gquery/ioutils.py`):

- `e10_1k_100q`: the fixed 100-query Amazon benchmark on the 110K catalog
  (10,000 items per category, 11 categories)
- `e11_gs2` ... `e11_gs8`: fixed group-size sets for the query-length sweep
- `e11_cat10k` ... `e11_cat30k`: same queries, growing catalog
- `e12_cat1m`, `e12_cat2m`: same queries, 1.1M and 2.14M items
- `e11_cat30k_recal`, `e12_cat1m_recal`, `e12_cat2m_recal`: budgets
  re-anchored to each catalog's minimum feasible cost
- `airbnb_12c`: the 100-query Airbnb benchmark on the 231K catalog (12 cities)
- `airbnb_gs2` ... `airbnb_gs8`: fixed group-size sets
- `airbnb_lad29k` ... `airbnb_lad231k` and their `_recal` variants: the Airbnb
  catalog ladder

## 3. Main Run

```bash
bash scripts/exp4/prewarm.sh
bash scripts/exp4/verify.sh
CUDA_VISIBLE_DEVICES=<one gpu> bash scripts/exp4/run_queue.sh --all
CUDA_VISIBLE_DEVICES=<one gpu> bash scripts/exp4/run_queue.sh --memory-grids
bash scripts/exp7/build_all.sh
```

This runs, in order:

1. `prewarm`: retrieve and cache candidates for every setting (GPU). The
   gates and any direct runner invocation read the cache.
2. `verify`: check that PBS without beam truncation matches exhaustive
   enumeration on both default workloads, and that the weighted ILP, weighted
   PBS and weighted PTS agree with enumeration. Both must print PASS. If this
   fails, stop - every recall number is measured against that reference.
3. `run_queue --all`: the job matrix `scripts/exp4/jobs.json`, one runner
   invocation per job, each in a fresh process with fresh GPU retrieval,
   single-threaded numerics and a provenance record (`run_job.py`). After each
   job the queue validates its output with the family's
   `scripts/exp7/reconstruct_*.py`, which recomputes every recall and
   hypervolume from the archived candidate sets without invoking a solver.
   The full matrix takes days; launch it detached.
4. `run_queue --memory-grids`: the isolated epsilon-ILP g=4 and g=16 memory
   processes (`scripts/exp4/memory_grid_jobs.json`).
5. `build_all`: every figure panel and table.

A runner can also be invoked directly, e.g.
`python -m gquery.alg.exp4.bench_default --help`; it then reads the prewarm
cache and writes under `data_out/<exp-name>/`.

## 4. Outputs

- per-job CSVs and archived candidate sets: `results/campaign/<family>/<job>/`
  (`CAMPAIGN_ROOT`), plus a `<job>.job/` directory with the parameters,
  environment and loader manifests
- figures (PDF + PNG + the CSV behind each panel) and tables:
  `data_figures/exp7/exp7_<topic>/` (`FIGURES_ROOT`), each with a
  `manifest.json` listing its sources and the hash of every generated file
- the paper's own copies of the panel CSVs, tables and manifests: `results/exp7/`

Which jobs feed which figure group:

| figure group | jobs | runner |
|---|---|---|
| default comparison, `table_pareto_*` | `default.*` | `bench_default` |
| search budgets, beam width and iteration sweeps, budget relaxation, matching threshold | `sensitivity.amazon*`, `sensitivity.airbnb*` | `bench_sensitivity` |
| component ablation, case study table | `sensitivity.*_ablation`, `sensitivity.*_case_gap` | `bench_ablation`, `case_study` |
| recall and HV over elapsed time | `deadlines.*` | `bench_anytime` |
| catalog size, query length, length-scaled budgets, million-item ladder, timing | `scalability.*` | `bench_scale`, `bench_length_budget`, `bench_scale_xl` |
| memory increase table | `memory.*` and the memory grid jobs | `bench_memory` |
| weighted CGQ vs the exact ILP, `table_weighted` | `weighted.*` | `exp5.bench_weighted_vs_ilp` |

Labels in the CSVs are the runners' historical ones: `PACS` is PBS,
`PACS_dC` is the PBS ablation arm with cost overhead in the prefix dominance,
`PBS-unbounded` is PBS without beam truncation, `Pareto-DP-conv` (or
`Pareto-DP` in the ladder) is DP, `PBF` is brute force, `eps-ILP g=N` is the
epsilon-constraint ILP with an N by N grid. The archived measurements in
`results/exp7` also carry `PTS-DFS` and `DFS` rows: the same depth-first PTS,
measured while an exploration-policy switch still existed. The builders map
them to PTS and accept fresh runs, where PTS rows are simply `PTS`.

Measured but not presented: epsilon-ILP grids g=2 and g=8 in the default
comparison, the epsilon-ILP arm of the per-category catalog sweep, the
epsilon-ILP and brute force arms of the matching-threshold sweep, the
tightness and candidate-count panels of the ladder, and the Airbnb case
study. The builders leave them in the summary CSVs.

To redraw figures from existing measurements without rerunning anything:

```bash
bash scripts/exp7/build_all.sh
```

or a single group:

```bash
python scripts/exp7/build_default.py
```

## 5. Common Overrides

Defaults live in `scripts/env.sh`. The usual ones:

```bash
NUM_QUERIES=100
ALPHA=0.5
BETA=0.8
DEFAULT_BEAM_SIZE=100
DEFAULT_EXPANSION_BUDGET=500
DEFAULT_CANDIDATE_LIMIT=50
TIMEOUT=60.0
CAMPAIGN_ROOT=results/campaign
FIGURES_ROOT=data_figures/exp7
```

Example:

```bash
FIGURES_ROOT=/tmp/figs bash scripts/exp7/build_all.sh
```

The job matrix carries its own arguments; overriding the defaults changes
direct runner invocations only.

## 6. Code Layout

| path | contents |
|---|---|
| `gquery/alg/common.py` | items, queries, retrieval |
| `gquery/alg/candidates.py` | partial assignments, candidate reduction, nondominance filters, suffix bounds |
| `gquery/alg/pareto/` | `pbs.solve_pbs`, `pts.solve_pts`, `dp.solve_dp`, `epsilon_ilp.solve_epsilon_ilp`, `brute_force.solve_brute_force`, `metrics` |
| `gquery/alg/weighted/` | `pbs.solve_pbs`, `pts.solve_pts`, `ilp.solve_ilp`, `brute_force.solve_brute_force`, `objective`, `bounds` |
| `gquery/alg/exp4/`, `gquery/alg/exp5/` | the runners and their loading, timing and reporting code |
| `gquery/figures/` | the panel helpers the builders use |
| `gquery/prep/` | data preparation (`amazon/`, `airbnb/`, `recal_budgets`) |
| `scripts/prep/`, `scripts/exp4/`, `scripts/exp7/` | drivers |
