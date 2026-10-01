# Published measurements

One directory per topic, as the manuscript includes them. Each holds the CSV
behind every figure panel (`<panel>.csv` beside where `<panel>.pdf` is drawn),
the generated `.tex` tables, per-dataset summary CSVs with explicit
denominators, and a `manifest.json` written by the builder in
`scripts/exp7/` that records the validated source jobs, the presentation rules
(which arms are shown), and the SHA-256 of every generated file.

| directory | paper items |
|---|---|
| `exp7_default` | default comparison (`default_dist_*`), `table_pareto_amazon.tex`, `table_pareto_airbnb.tex`; the epsilon grid sweep and the extra baselines are measured but not shown |
| `exp7_sensitivity` | `envelope`, `search_control_*`, `alpha_*`, `matching_threshold_*` |
| `exp7_ablation` | `ablation_dist_*`, `table_case_study_amazon.tex` |
| `exp7_deadlines` | `elapsed_recall`, `elapsed_hv` (rendered from `anytime_deadline` and `anytime_hv_deadline`) |
| `exp7_scale` | `scale_dist_*`, `length_budget_*`, `scale_xl_*`, `timing_*` |
| `exp7_memory` | `table_memory_increases.tex` |
| `exp7_weighted` | `table_weighted.tex` |

The files are the ones the paper was built from. Path-bearing columns and
manifest fields (`source_file`, `path`, `generation_command`) had the
authors' home directory replaced by `<paper>` and `<code>`, so the manifest
hashes refer to the originals for CSVs that carry such a column
(`*_source.csv`, `*_presented.csv`, the summary CSVs). Every other file is
byte-identical to what `scripts/exp7/build_all.sh` regenerates from the
archived job outputs.
