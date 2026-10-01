"""Render PBS/PTS median latency and IQR from validated per-query timings."""
from pathlib import Path
import hashlib
import importlib.util
import math
import json
import shutil
import pandas as pd
from gquery.figures import plot_exp4 as plots

import os
REPO = Path(__file__).resolve().parents[2]
FIGURES = Path(os.environ.get('FIGURES_ROOT', REPO / 'data_figures' / 'exp7'))
SOURCE = FIGURES / 'exp7_scale'
OUT = SOURCE
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
loader_path = Path(__file__).resolve().parent / 'build_scalability.py'
spec = importlib.util.spec_from_file_location('scalability_source', loader_path)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
sources = {str(loader_path): sha(loader_path)}
outputs = []
for folder, prefix, labels in [
    ('amazon', 'scale_xl', ['110K', '330K', '1.1M', '2.14M']),
    ('airbnb', 'scale_xl', ['29K', '58K', '115K', '231K']),
]:
    latency = SOURCE / folder / f'{prefix}_latency.csv'
    recall = SOURCE / folder / f'{prefix}_recall.csv'
    frame = pd.read_csv(latency)
    frame = frame[frame.method.isin(['PACS', 'PTS'])].copy()
    raw = loader.read(folder, 'xl')
    for source in raw.source_file.unique():
        path = Path(source)
        sources[str(path)] = sha(path)
        proof = path.parent / 'reconstruction.json'
        sources[str(proof)] = sha(proof)
    for index, row in frame.iterrows():
        values = raw[(raw.setting == row.setting) & (raw.method == row.method)].total_s * 1000
        assert len(values) == 100 and values.notna().all()
        assert math.isclose(values.median(), row.e2e_median_ms, rel_tol=1e-10)
        frame.loc[index, 'q25_ms'] = values.quantile(0.25)
        frame.loc[index, 'q75_ms'] = values.quantile(0.75)
        frame.loc[index, 'n_evaluated'] = len(values)
    frame['n_evaluated'] = frame.n_evaluated.astype(int)
    quality = pd.read_csv(recall)
    sizes = quality[quality.method == 'PACS'].catalog_items.to_numpy()
    fig, ax = plots.paper_panel()
    for method, label in [('PACS', 'PBS'), ('PTS', 'PTS')]:
        group = frame[frame.method == method]
        assert len(group) == len(sizes) == len(labels)
        plots._band(ax, sizes, group.e2e_median_ms.to_numpy(),
                    group.q25_ms.to_numpy(), group.q75_ms.to_numpy(), plots.color(method), label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_formatter(plots.plt.NullFormatter())
    lo, hi = frame.q25_ms.min(), frame.q75_ms.max()
    ax.set_yticks([t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200) if lo / 1.6 <= t <= hi * 1.6])
    ax.yaxis.set_major_formatter(plots.plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(plots.plt.NullFormatter())
    ax.set_xlabel('catalog items')
    ax.set_ylabel('median time (ms)')
    ax.legend(loc='lower right')
    plots.save_panel(fig, str(OUT), folder, f'{prefix}_latency_pbs_pts', frame)
    for path in [latency, recall]:
        sources[str(path)] = sha(path)
    outputs += [f'{folder}/{prefix}_latency_pbs_pts.{ext}' for ext in ['pdf', 'png', 'csv']]
manifest = dict(builder_sha256=sha(Path(__file__)), plot_helper_sha256=sha(Path(plots.__file__)), sources=sources,
                change='Main latency panels show only PBS and PTS. Median values unchanged; bands show the 25th and 75th percentiles of total_s over all 100 measured queries per setting.',
                generated={name: sha(OUT / name) for name in outputs})
(OUT / 'manifest_two_method.json').write_text(json.dumps(manifest, indent=2) + '\n')
publication = None
