"""Render observation panels with elapsed time axes and shared legends."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gquery.figures import plot_exp4 as plots

import os
REPO = Path(__file__).resolve().parents[2]
FIGURES = Path(os.environ.get('FIGURES_ROOT', REPO / 'data_figures' / 'exp7'))
ROOT = FIGURES / 'exp7_deadlines'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plots.LABELS.update({'PTS': 'PTS', 'PBS-unbounded': 'PBS unbounded', 'Pareto-DP': 'Pareto DP'})
plots.COLORS.update({'PBS-unbounded': '#2E5E8E', 'Pareto-DP': '#9673A6'})
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['amazon', 'airbnb', 'both'], default='both')
parser.add_argument('--grid', type=int, choices=[4, 16], default=16)
args = parser.parse_args()
manifest = json.loads((ROOT / 'manifest.json').read_text())
legacy = manifest.pop('observation_axis_revision', None)
if legacy:
    for dataset in ['amazon', 'airbnb']:
        manifest.setdefault('observation_axis_revisions', {}).setdefault(dataset, legacy)
for folder, prefix in [('amazon', 'anytime'), ('airbnb', 'anytime')]:
    if args.dataset != 'both' and folder != args.dataset:
        continue
    for suffix, ylabel, out_stem in [('deadline', 'frontier recall', 'elapsed_recall'), ('hv_deadline', 'hypervolume ratio', 'elapsed_hv')]:
        stem = f'{prefix}_{suffix}'
        frame = pd.read_csv(ROOT / folder / f'{stem}.csv')
        arms = ['PACS', 'PTS', f'eps-ILP g={args.grid}']
        assert set(arms).issubset(set(frame.method))
        frame = pd.concat([frame[frame.method == m] for m in arms])
        series = [(g.deadline_ms, g['median'], g.q25, g.q75,
                   plots.color(m), plots.LABELS.get(m, m))
                  for m, g in frame.groupby('method', sort=False)]
        plots._band_panel(str(ROOT), folder, out_stem, series, 'elapsed time (ms)', ylabel,
                          logx=True, ylim=(0, 1.05) if suffix == 'deadline' else None,
                          legend=False)
        for ext in ['pdf', 'png']:
            rel = f'{folder}/{out_stem}.{ext}'
            manifest['generated'][rel] = sha(ROOT / rel)
    handles = [Line2D([], [], color=plots.color(m), marker='o', linewidth=3,
                      markersize=8, label=plots.LABELS.get(m, m))
               for m in frame.method.drop_duplicates()]
    fig = plt.figure(figsize=(10, 0.6))
    fig.legend(handles=handles, loc='center', ncol=3, fontsize=24,
               frameon=False, handlelength=1.3, columnspacing=1.3,
               handletextpad=0.5)
    for ext in ['pdf', 'png']:
        rel = f'{folder}/elapsed_legend.{ext}'
        fig.savefig(ROOT / rel, bbox_inches='tight', pad_inches=0.04)
        manifest['generated'][rel] = sha(ROOT / rel)
    plt.close(fig)
manifest.setdefault('observation_axis_revisions', {})[args.dataset] = dict(script=str(Path(__file__).resolve()),
    sha256=sha(Path(__file__)), xlabel='elapsed time (ms)', legend=f'PBS, PTS, and epsilon-ILP g={args.grid} shared across each recall/HV pair', data='Unchanged summary CSVs')
(ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
