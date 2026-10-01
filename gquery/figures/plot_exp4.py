
'''
figures for the exp4 suite. one --figure key per paper figure ("all" for
everything). each panel writes PDF + PNG plus the CSV behind it.
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.transforms import Bbox

DATA = Path("data_out")
OUT_ROOT = Path("data_figures/papers")

# one colour per method, used by every panel
COLORS = {
    "PACS": "#6C8EBF",
    "PACS unbounded": "#2E5E8E",
    "PTS": "#D79B00",
    "PBF": "#82B366",
    "PACS_dC": "#B85450",
    "eps-ILP g=2": "#C7CBD1",
    "eps-ILP g=4": "#A5ABB4",
    "eps-ILP g=8": "#7F8894",
    "eps-ILP g=16": "#5A6472",
    "ILP": "#5A6472",
    "ACS": "#6C8EBF",
    "MCTS": "#D79B00",
    "Exact (shared)": "#5A6472",
    "Split-equal": "#B85450",
    "Split-median": "#9673A6",
    "Split-proportional": "#C17D11",
}
'''
benches write the historical labels (PACS etc.), mapped to paper names here so
old CSVs stay readable. careful: a key that stops matching silently drops the
series.
'''
LABELS = {
    "PACS": "PBS",
    "PACS_dC": r"PBS$_{\Delta C}$",
    "Exact-PACS": "Exact-PBS",
    "PBF": "BF",
    "eps-ILP g=2": r"$\epsilon$-ILP $g$=2",
    "eps-ILP g=4": r"$\epsilon$-ILP $g$=4",
    "eps-ILP g=8": r"$\epsilon$-ILP $g$=8",
    "eps-ILP g=16": r"$\epsilon$-ILP $g$=16",
}


# ======= layout =======
'''
panels keep the 10/6 aspect of the old single-panel figures. LaTeX scales the
strip to textwidth, so font/line/marker sizes get divided by that scale factor
to land at the page targets below.
'''
PANEL_ASPECT = 10.0 / 6.0
PANEL_W_IN = 6.0
TEXTWIDTH_IN = 7.0
COLUMNWIDTH_IN = 3.33
_SNS_PAPER_BASE_PT = 9.6

# page-rendered targets (text a bit above the old 5.4pt, which was borderline)
TARGET_FONT_PT = 6.5
TARGET_LW_PT = 0.55
TARGET_MS_PT = 2.0


def grid(ncols: int, nrows: int = 1,
         target_width_in: float = TEXTWIDTH_IN,
         panel_h_in: Optional[float] = None):
    '''target_width_in = width LaTeX will place the figure at; one-column
    figures pass COLUMNWIDTH_IN so fonts stay legible.'''
    w = PANEL_W_IN * ncols
    panel_h = panel_h_in or (PANEL_W_IN / PANEL_ASPECT)
    h = panel_h * nrows
    # what LaTeX will apply
    scale = target_width_in / w
    plt.clf()
    sns.set_style("whitegrid")
    sns.set_context("paper",
                    font_scale=TARGET_FONT_PT / (_SNS_PAPER_BASE_PT * scale))
    plt.rcParams["lines.linewidth"] = TARGET_LW_PT / scale
    plt.rcParams["lines.markersize"] = TARGET_MS_PT / scale
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    return fig, np.atleast_1d(np.asarray(axes)).ravel()


# panels inherit rcParams now, explicit sizes removed


# point size labels should have *after* scaling to the column width
TEXTWIDTH_IN = 7.0
_SNS_PAPER_BASE_PT = 9.6


def style(scale: float = 1.9) -> None:
    plt.clf()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=scale)


def label(m: str) -> str:
    return LABELS.get(m, m)


def color(m: str) -> str:
    return COLORS.get(m, "#888888")


def save(fig: plt.Figure, exp_name: str, folder: str, stem: str,
         df: Optional[pd.DataFrame] = None) -> None:
    base = OUT_ROOT / exp_name / folder
    base.mkdir(parents=True, exist_ok=True)
    box = Bbox.from_bounds(0, 0, *fig.get_size_inches())
    for fmt in ("png", "pdf"):
        fig.savefig(base / f"{stem}.{fmt}", format=fmt, bbox_inches=box)
    plt.close(fig)
    if df is not None:
        df.to_csv(base / f"{stem}.csv", index=False)
    print(f"  wrote {base}/{stem}.{{png,pdf,csv}}")


def read(exp: str, name: str) -> Optional[pd.DataFrame]:
    p = DATA / exp / f"{name}.csv"
    if not p.exists():
        print(f"  [skip] missing {p}")
        return None
    return pd.read_csv(p)


def e2e(df):
    '''end-to-end time series (search + retrieval); falls back to the search
    column for older csvs without the retrieval breakdown.'''
    if "end_to_end_s" in df.columns:
        return df["end_to_end_s"]
    base = df["time_s"] if "time_s" in df.columns else df["search_s"]
    if "retrieval_s" in df.columns:
        return base + df["retrieval_s"].fillna(0.0)
    return base


# ======= default comparison =======


# ======= anytime: time to k points =======


# ======= beam width and iteration sweeps =======


# ======= matching threshold sweep =======


# ======= ablation =======

# only the arms we report on. the rest stay in the CSV
DOCUMENTED_ARMS = {
    "PACS": ["full", "-rawcost"],
    "PTS": ["full", "-bound", "-prefixdom", "-bound-prefixdom", "-close"],
}
'''
ablation is reported at the default threshold. fig_beta covers the threshold
axis, repeating every arm at a second beta buys nothing.
'''
ABLATION_BETA = 0.8


# ======= budget split =======


# ======= deceptive workloads =======


# ======= catalog and length scaling =======


# ======= weighted diagnostics =======


# ======= per-query distributions =======


# ======= epsilon grid =======


# ======= per-query distribution figures =======
'''
conventions: a method returning nothing scores zero (never a gap in an
average). runtime restricted to completed queries is labelled conditional.
evals compare within one family only.
'''

'''
display order for the default comparison. eps-ILP g=16 is the strongest grid
reported, it stands for the repeated-ILP baseline.
'''
DEFAULT_ARMS = [("PACS", "PBS"), ("PTS", "PTS"),
                ("eps-ILP g=16", r"$\epsilon$-ILP"), ("PBF", "BF")]

IQR_ALPHA = 0.18

'''
tick labels for the ablation panels. mechanism names do not fit at this font
size and colour already carries the family, so the ticks are mnemonics and
the caption spells them out.
'''
ABLATION_TICK = {
    ("PACS", "-rawcost"): r"$\Delta C$",
    ("PTS", "-bound"): r"$-$B",
    ("PTS", "-prefixdom"): r"$-$P",
    ("PTS", "-bound-prefixdom"): r"$-$BP",
    ("PTS", "-close"): r"$-$X",
}


def _band(ax, x, med, lo, hi, colour, label_, marker="o", ls="-"):
    '''median with an IQR band, used by every sweep panel.'''
    ax.plot(x, med, marker=marker, ls=ls, color=colour, label=label_)
    ax.fill_between(x, lo, hi, color=colour, alpha=IQR_ALPHA, linewidth=0)


def _quartiles(g: pd.DataFrame, col: str):
    q = g[col].quantile([0.25, 0.5, 0.75]).values
    return q[1], q[0], q[2]


def _box_panel(exp_name, folder, stem, data, labs, ylabel, colours,
               logy=False, hline=None, ylim=None, yticks=None, df=None, tick_rotation=None):
    '''one box plot per file, old single-panel style.'''
    fig, ax = paper_panel()
    bp = ax.boxplot(data, tick_labels=labs, showfliers=True, patch_artist=True,
                    widths=0.6,
                    flierprops=dict(marker=".", markersize=6,
                                    markerfacecolor="#666666",
                                    markeredgecolor="none"),
                    medianprops=dict(color="black", linewidth=PAPER_LINEWIDTH))
    for patch, l in zip(bp["boxes"], labs):
        patch.set_facecolor(colours[l])
        patch.set_alpha(0.85)
    if tick_rotation is not None:
        ax.set_xticks(range(1, len(labs)+1), labs, rotation=tick_rotation,
                      ha="right", fontsize=14)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="#B85450", lw=PAPER_LINEWIDTH, ls="--")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        '''a decade-labelled log axis is nearly blank over one order of
        magnitude, ratio panels name their own ticks.'''
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:g}" for t in yticks])
        ax.minorticks_off()
    save_panel(fig, exp_name, folder, stem, df)


def _band_panel(exp_name, folder, stem, series, xlabel, ylabel,
                logx=True, logy=False, ylim=None, vline=None,
                legend="best", xticks=None, df=None):
    '''series: list of (x, median, q25, q75, colour, label).'''
    fig, ax = paper_panel()
    for x, med, lo, hi, colour, lbl in series:
        _band(ax, x, med, lo, hi, colour, lbl)
    if vline is not None:
        ax.axvline(vline, color="#555555", lw=PAPER_LINEWIDTH, ls=":")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        '''swept controls rarely land on decade boundaries; without this the
        log x axis labels one tick and the sweep is unreadable.'''
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:g}" for t in xticks])
        ax.minorticks_off()
    if legend and len(series) > 1:
        ax.legend(**legend) if isinstance(legend, dict) else ax.legend(loc=legend)
    save_panel(fig, exp_name, folder, stem, df)


def _sparse_ticks(xs, keep=None, n=4):
    '''at most n ticks, always including keep, labels collide otherwise.'''
    xs = list(dict.fromkeys(sorted(xs)))
    if len(xs) <= n:
        return xs
    if keep is not None and keep in xs:
        return sorted({xs[0], keep, xs[-1]})
    return sorted({xs[0], xs[len(xs) // 2], xs[-1]})


def fig_anytime_dist(exp_name: str, src: str = "exp4_pplus_anytime",
                     folder: str = "pareto",
                     prefix: str = "anytime_dist", clock: str = "method",
                     mode: str = "observation") -> None:
    '''time-to-K is conditional on reaching K, so reach rates are annotated
    on the bars. src/folder/prefix select the bench dir and the output stems,
    so a second dataset reuses the panels unchanged.'''
    a = read(src, "anytime")
    if a is None:
        return
    if "clock" in a.columns:
        a = a[(a.clock == clock) & (a["mode"] == mode)]
    if a.empty:
        return
    dls = sorted({float(c.split("@")[1]) for c in a.columns if c.startswith("recall@")})
    ks = sorted({int(c.split("_")[-1]) for c in a.columns if c.startswith("t_to_")})
    arms = [(m, LABELS.get(m, m)) for m in dict.fromkeys(a.method)]
    COLORS.update({"PBS-unbounded": "#2E5E8E", "Pareto-DP": "#9673A6", "DFS": "#82B366"})

    for metric, ylabel, suffix in (("recall", "frontier recall", "deadline"),
                                   ("hv", "hypervolume ratio", "hv_deadline")):
        if not any(c.startswith(metric + "@") for c in a.columns):
            continue
        series, rows = [], []
        for m, lbl in arms:
            subset = a[a.method == m]
            med = [subset[f"{metric}@{d}"].median() for d in dls]
            lo = [subset[f"{metric}@{d}"].quantile(0.25) for d in dls]
            hi = [subset[f"{metric}@{d}"].quantile(0.75) for d in dls]
            series.append(([d * 1000 for d in dls], med, lo, hi, color(m), lbl))
            rows += [{"method": m, "deadline_ms": d * 1000, "median": v,
                      "q25": l_, "q75": h_, "metric": metric,
                      "n_evaluated": int(subset[f"{metric}@{d}"].notna().sum()),
                      "clock": clock, "mode": mode}
                     for d, v, l_, h_ in zip(dls, med, lo, hi)]
        _band_panel(exp_name, folder, f"{prefix}_{suffix}", series,
                    "elapsed budget (ms)", ylabel, logx=True,
                    ylim=(0, 1.05) if metric == "recall" else None,
                    legend={"loc": "lower center", "bbox_to_anchor": (.5, 1.01),
                            "ncol": 3, "fontsize": 14}, df=pd.DataFrame(rows))
    if mode == "hard":
        return  # Time-to-K summaries must not pool different censoring horizons.

    data, labs, colours, frames = [], [], {}, []
    for m, lbl in arms:
        v = a[a.method == m].t_first.dropna()
        v = v[v > 0]
        if len(v):
            data.append(v.values); labs.append(lbl); colours[lbl] = color(m)
            frames.append(pd.DataFrame({"method": m, "value": v.values}))
    if data:
        _box_panel(exp_name, folder, f"{prefix}_tfirst", data, labs,
                   "time to first point (s)", colours, logy=True,
                   df=pd.concat(frames, ignore_index=True), tick_rotation=30)

    if not ks:
        return
    fig, ax = paper_panel()
    width = 0.8 / max(len(arms), 1)
    xs = np.arange(len(ks))
    rows = []
    for i, (m, lbl) in enumerate(arms):
        s = a[a.method == m]
        med = [s[f"t_to_{k}"].median() for k in ks]
        reach = [s[f"reached_{k}"].mean() for k in ks]
        pos = xs + (i - (len(arms) - 1) / 2) * width
        ax.bar(pos, med, width, color=color(m), edgecolor="black",
               linewidth=1.0, label=lbl)
        for x, mv, r in zip(pos, med, reach):
            if np.isfinite(mv):
                '''anchor-mode rotation grows the label upward from the bar
                top; the default mode rotates the bounding box and clips the
                digits.'''
                ax.annotate(f"{r:.0%}", (x, mv), textcoords="offset points",
                            xytext=(0, 4), ha="left", va="center",
                            rotation=90, rotation_mode="anchor",
                            fontsize=12)
        rows += [{"method": m, "k": k, "median": mv, "reached": r}
                 for k, mv, r in zip(ks, med, reach)]
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k}" for k in ks])
    ax.set_xlabel("$K$ frontier points")
    ax.set_ylabel("time to $K$ (s)")
    # headroom for the rotated reach labels and the legend
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 60)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=14)
    save_panel(fig, exp_name, folder, f"{prefix}_ttok", pd.DataFrame(rows))


def fig_scale_dist(exp_name: str, src: str = "exp4_pplus_scale",
                   folder: str = "scale", prefix: str = "scale_dist",
                   sweeps: Sequence[str] = ("catalog", "length")) -> None:
    '''sweeps selects which ladders to draw; a sweep with no rows in the csv is
    skipped (the airbnb catalog is the whole pool, so that run is length only).
    with a single sweep requested the stem drops the sweep tag, it is already
    in the prefix.'''
    df = read(src, "scale")
    if df is None:
        return
    df = df.assign(e2e_s=e2e(df))
    arms = [(m, lbl) for m, lbl in DEFAULT_ARMS if m in set(df.method)]
    specs = [(s, xl, tag) for s, xl, tag in
             (("catalog", "items per category (K)", "catalog"),
              ("length", "requirements per query", "length"))
             if s in sweeps]
    tagged = len(specs) > 1

    for sweep, xl, tag in specs:
        sub = df[df.sweep == sweep]
        if sub.empty:
            print(f"  [skip] no {sweep} rows in {src}")
            continue
        for col, ylab, logy, kind in (("recall", "frontier recall", False, "recall"),
                                      ("e2e_s", "end-to-end time (s)", True, "time")):
            series, rows = [], []
            for m, lbl in arms:
                s = sub[sub.method == m]
                if s.empty:
                    continue
                xs, med, lo, hi = [], [], [], []
                for prm, g in s.groupby("param"):
                    m_, l_, h_ = _quartiles(g, col)
                    xs.append(prm); med.append(m_); lo.append(l_); hi.append(h_)
                o = np.argsort(xs)
                xs, med = np.array(xs)[o], np.array(med)[o]
                lo, hi = np.array(lo)[o], np.array(hi)[o]
                series.append((xs, med, lo, hi, color(m), lbl))
                rows += [{"sweep": sweep, "method": m, "param": p, "metric": col,
                          "median": v, "q25": l_, "q75": h_}
                         for p, v, l_, h_ in zip(xs, med, lo, hi)]
            if series:
                stem = (f"{prefix}_{tag}_{kind}" if tagged
                        else f"{prefix}_{kind}")
                _band_panel(exp_name, folder, stem, series,
                            xl, ylab, logx=(sweep == "catalog"), logy=logy,
                            ylim=None if logy else (0, 1.05),
                            # Both recall curves fall to the right in the length
                            # sweep, so the free corner moves with the sweep.
                            legend=("upper right" if sweep == "length" else
                                    "lower left") if kind == "recall"
                                   else "upper left",
                            xticks=_sparse_ticks(sorted(set(sub.param))),
                            df=pd.DataFrame(rows))


# registered after their definitions, FIGS above is built earlier


# ======= one-plot-per-file helpers =======
'''
composed by LaTeX as 0.23 textwidth subfigures. authored large since each
panel is reduced ~4x on the page.
'''

PAPER_FONT_SCALE = 3.5
PAPER_FIGSIZE = (10.0, 6.0)
PAPER_LINEWIDTH = 3
PAPER_MARKERSIZE = 12


def paper_panel(figsize: Tuple[float, float] = PAPER_FIGSIZE):
    '''one axes in the old house style, sized for a 0.23-width subfigure.'''
    plt.clf()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=PAPER_FONT_SCALE)
    plt.rcParams["lines.linewidth"] = PAPER_LINEWIDTH
    plt.rcParams["lines.markersize"] = PAPER_MARKERSIZE
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_panel(fig, exp_name: str, folder: str, stem: str,
               df: Optional[pd.DataFrame] = None) -> None:
    '''fixed-bbox save, same as the old save_figure_paper.'''
    fig.tight_layout()
    save(fig, exp_name, folder, stem, df)


# ======= appendix figures: only what the main paper leaves out =======


def _bar_panel(exp_name, folder, stem, labels, values, ylabel, colours,
               ylim=None, annotate=True, df=None):
    '''one bar panel, for rates with no per-query distribution.'''
    fig, ax = paper_panel()
    xs = np.arange(len(labels))
    ax.bar(xs, values, 0.65, color=[colours[l] for l in labels],
           edgecolor="black", linewidth=1.0)
    if annotate:
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize="xx-small")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    save_panel(fig, exp_name, folder, stem, df)


# allocation strategies for the shared-budget study, display order
BUDGET_ARMS = [("Exact (shared)", "shared"), ("PACS", "PBS"), ("PTS", "PTS"),
               ("Split-equal", "equal"), ("Split-median", "med."),
               ("Split-proportional", "min.")]


'''catalog sizes stated here because a cached slot load never touches the
item files, so the csv's catalog_items column can be nan.

the ladder protocol: budgets are anchored to each catalog's min-feasible
costs with the query's fixed multiplier, so budget tightness is identical
across scales and catalog size is the only moving variable. the _recal
setting names are historical; they ARE the ladder. the fixed-110K-budget
arms are kept only for the supplement's budget-drift check.'''
XL_LADDER = [("e11_cat10k", "110K", 110000),
             ("e11_cat30k_recal", "330K", 298233),
             ("e12_cat1m_recal", "1.1M", 1095576),
             ("e12_cat2m_recal", "2.14M", 2139883)]
XL_FIXED = [("e11_cat10k", "110K", 110000), ("e11_cat30k", "330K", 298233),
            ("e12_cat1m", "1.1M", 1095576), ("e12_cat2m", "2.14M", 2139883)]

'''airbnb downward ladder (PLAN_AIRBNB_LADDER.md): nested per-city shares
1/8, 1/4, 1/2 of the 231K pool, queries generated on the 29K rung, budgets
re-anchored per rung exactly as on amazon. sizes here are fallbacks; the
csv's catalog_items column wins when the run loaded the items.'''
AIRBNB_LADDER = [("airbnb_lad29k", "29K", 28872),
                 ("airbnb_lad58k_recal", "58K", 57737),
                 ("airbnb_lad115k_recal", "115K", 115466),
                 ("airbnb_lad231k_recal", "231K", 230926)]
AIRBNB_FIXED = [("airbnb_lad29k", "29K", 28872),
                ("airbnb_lad58k", "58K", 57737),
                ("airbnb_lad115k", "115K", 115466),
                ("airbnb_lad231k", "231K", 230926)]


def fig_scale_xl(exp_name: str, src: str = "exp6_scale",
                 ladder=XL_LADDER, fixed=XL_FIXED,
                 top_setting: str = "e12_cat2m_recal",
                 folder: str = "scale", prefix: str = "scale_xl",
                 recall_legend: str = "lower left",
                 tightness_legend_above: bool = False) -> None:
    '''the million-item ladder. reads exp6_scale, not the exp4 prefix.
    the airbnb variant passes its own source / ladder / output names.'''
    df = read(src, "scale_xl")
    if df is None:
        return

    def _n(s, fallback):
        v = df[df.setting == s].catalog_items.dropna() \
            if "catalog_items" in df.columns else pd.Series(dtype=float)
        return int(v.iloc[0]) if len(v) else fallback

    order = [(s, lbl, _n(s, n)) for s, lbl, n in ladder if s in set(df.setting)]
    fixed = [(s, lbl, _n(s, n)) for s, lbl, n in fixed]

    # end-to-end latency for every ladder method, with the shared retrieval
    # stage as the floor. eps-ILP and PBF are cut from the ladder by
    # protocol: they already take seconds per query at 110K and the ILP
    # grid's work grows with the candidate sets
    labs = [lbl for _, lbl, _ in order]
    ns = [n for _, _, n in order]
    fig, ax = paper_panel()
    rows = []
    for meth, lbl, line_style in (
            ("PACS", "PBS", "-"),
            ("PTS", "PTS", "-"),
            ("PACS unbounded", "PBS unbounded", "--")):
        vals = [df[(df.setting == s) & (df.method == meth)]
                .end_to_end_s.median() * 1000 for s, _, _ in order]
        ax.plot(ns, vals, marker="o", linestyle=line_style,
                color=color(meth), label=lbl)
        rows += [{"setting": s, "method": meth, "e2e_median_ms": float(v)}
                 for (s, _, _), v in zip(order, vals)]
    ret = [df[(df.setting == s) & (df.method == "PTS")]
           .retrieval_s.median() * 1000 for s, _, _ in order]
    ax.plot(ns, ret, ":", marker="s", color="#888888", label="retrieval")
    rows += [{"setting": s, "method": "retrieval", "e2e_median_ms": float(v)}
             for (s, _, _), v in zip(order, ret)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(labs)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    allv = [v for v in ret if np.isfinite(v)] + \
        [r["e2e_median_ms"] for r in rows if np.isfinite(r["e2e_median_ms"])]
    lo_v, hi_v = (min(allv), max(allv)) if allv else (1, 20)
    ax.set_yticks([t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200)
                   if lo_v / 1.6 <= t <= hi_v * 1.6])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("catalog items")
    ax.set_ylabel("median time (ms)")
    ax.legend(loc="lower right", fontsize="x-small")
    save_panel(fig, exp_name, folder, f"{prefix}_latency",
               pd.DataFrame(rows))

    # recall over the same ladder, the headline pair to the latency panel,
    # in the house sweep style: median lines with IQR bands. the bands are
    # wide because per-query recall at scale is bimodal (near 0 or near 1),
    # so the prose also reports the means
    fig, ax = paper_panel()
    rows = []
    for meth, lbl in (("PACS", "PBS"), ("PTS", "PTS")):
        med, lo, hi = [], [], []
        for s, _, n in order:
            g = df[(df.setting == s) & (df.method == meth)]
            m_, l_, h_ = _quartiles(g, "recall")
            med.append(m_); lo.append(l_); hi.append(h_)
            rows.append({"method": meth, "catalog_items": n,
                         "recall_median": float(m_), "q25": float(l_),
                         "q75": float(h_),
                         "recall_mean": float(g.recall.mean())})
        _band(ax, np.array(ns), np.array(med), np.array(lo), np.array(hi),
              color(meth), lbl)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(labs)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("catalog items")
    ax.set_ylabel("frontier recall")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc=recall_legend)
    save_panel(fig, exp_name, folder, f"{prefix}_recall", pd.DataFrame(rows))

    # candidate growth over the ladder
    xs2, med, lo, hi = [], [], [], []
    for s, _, n_items in order:
        g = df[(df.setting == s) & (df.method == "PTS")]
        m_, l_, h_ = _quartiles(g, "n_candidates")
        xs2.append(n_items); med.append(m_)
        lo.append(l_); hi.append(h_)
    _band_panel(exp_name, folder, f"{prefix}_candidates",
                [(np.array(xs2), np.array(med), np.array(lo), np.array(hi),
                  color("PTS"), "candidates")],
                "catalog items", "candidates per query",
                logx=True, legend=None,
                df=pd.DataFrame({"catalog_items": xs2, "median": med,
                                 "q25": lo, "q75": hi}))

    # candidate-cap sweep at the top scale
    top = df[df.setting == top_setting]
    kap = [("PTS k=10", "10"), ("PTS k=25", "25"), ("PTS", "50"),
           ("PTS k=100", "100"), ("PTS uncapped", "unc.")]
    kap = [(m, l) for m, l in kap if m in set(top.method)]
    if kap:
        # means for the same reason as the recall panel above
        vals = [top[top.method == m].recall.mean() for m, _ in kap]
        _bar_panel(exp_name, folder, f"{prefix}_kappa",
                   [l for _, l in kap], vals, "mean frontier recall",
                   {l: color("PTS") for _, l in kap}, ylim=(0, 1.1),
                   df=pd.DataFrame({"kappa": [l for _, l in kap],
                                    "recall_mean": vals}))

    # budget tightness, the supplement's protocol check: the ladder holds
    # the tight-budget share flat, while budgets held fixed at their 110K
    # values would drift loose
    per_q = df.drop_duplicates(["setting", "query_id"])
    fig, ax = paper_panel()
    width = 0.38
    rows = []
    tlabs = [l for _, l, _ in fixed]
    for off, lad, lbl, col in (
            (-width / 2, ladder, "per-catalog budgets", color("PTS")),
            (width / 2, fixed, f"fixed {ladder[0][1]} budgets", "#888888")):
        vals = [per_q[per_q.setting == s].tight80.mean() for s, _, _ in lad
                if s in set(per_q.setting)]
        xs3 = np.arange(len(vals))
        ax.bar(xs3 + off, vals, width, label=lbl, color=col,
               edgecolor="black", linewidth=1.0)
        for x, v in zip(xs3 + off, vals):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize="xx-small")
        rows += [{"budgets": lbl, "setting": s, "tight80_share": float(v)}
                 for (s, _, _), v in zip(lad, vals)]
    ax.set_xticks(np.arange(len(tlabs)))
    ax.set_xticklabels(tlabs)
    ax.set_xlabel("catalog items")
    ax.set_ylabel("tight-budget share")
    # 0.62 fits amazon's 0.41 flat share; airbnb's is 0.71, so grow with it
    tmax = max(r["tight80_share"] for r in rows) if rows else 0.5
    ax.set_ylim(0, max(0.62, 1.25 * tmax))
    if tightness_legend_above:   # tall bars leave no free corner inside
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  fontsize="x-small", frameon=False)
    else:
        ax.legend(loc="upper right", fontsize="x-small")
    save_panel(fig, exp_name, folder, f"{prefix}_tightness",
               pd.DataFrame(rows))


def fig_length_budget(exp_name: str, src: str = "exp4_pplus_length_budget",
                      folder: str = "scale",
                      prefix: str = "length_budget") -> None:
    '''do budgets scaled with query length recover the long-query recall.
    one curve per (b, M) tier; the fixed default is the lowest tier.'''
    df = read(src, "length_budget")
    if df is None:
        return
    tiers = [(100, 500), (200, 1000), (400, 2000), (800, 4000), (1600, 8000)]
    shades = ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
    for method, col, tag in (("PACS", "beam", "pbs"),
                             ("PTS", "expansion_budget", "pts")):
        sub = df[df.method == method]
        if sub.empty:
            continue
        fig, ax = paper_panel()
        rows = []
        for (b_, m_), shade in zip(tiers, shades):
            prm = b_ if col == "beam" else m_
            g = sub[sub[col] == prm]
            med = g.groupby("n").recall.median()
            lbl = (f"$b$={b_}" if col == "beam" else f"$M$={m_}")
            if prm == tiers[0][0 if col == "beam" else 1]:
                lbl += " (default)"
            ax.plot(med.index.values, med.values, marker="o", color=shade,
                    label=lbl)
            rows += [{"method": method, "param": prm, "n": int(n),
                      "recall_median": float(v)} for n, v in med.items()]
        ax.set_xlabel("requirements per query $n$")
        ax.set_ylabel("frontier recall")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(sorted(sub.n.unique()))
        ax.legend(loc="lower left", fontsize="x-small")
        save_panel(fig, exp_name, folder, f"{prefix}_{tag}",
                   pd.DataFrame(rows))


'''the default operating points are set by one shared rule (smallest
parameter whose median recall saturates on the amazon tuning workload), but
no (b, M) pair equalizes time or work -- the methods occupy different time
ranges. the envelope panels are therefore the comparison that is fair: each
method's own recall-time curve, traced by its control parameter.'''


'''
entry point stays last: every FIGS.update above must run before the command
line resolves a figure name.
'''


FIGS = {
    "anytime_dist": fig_anytime_dist,
    "scale_dist": fig_scale_dist,
    "scale_xl": fig_scale_xl,
    "length_budget": fig_length_budget,
}


@click.command()
@click.option("--figure", type=str, default="all")
@click.option("--exp-name", type=str, default="exp4_pplus")
def main(figure, exp_name):
    names = list(FIGS) if figure == "all" else [f.strip() for f in figure.split(",")]
    for n in names:
        if n not in FIGS:
            print(f"unknown figure '{n}'; known: {', '.join(FIGS)}")
            continue
        print(f"[{n}]")
        FIGS[n](exp_name)


if __name__ == "__main__":
    main()
