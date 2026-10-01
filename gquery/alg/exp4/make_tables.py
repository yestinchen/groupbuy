'''emit the latex table bodies (bare tabular, for \\input) from the bench csvs.'''

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd

OUT = Path("data_out/exp4_pplus_tables")


def _e2e(df: pd.DataFrame) -> pd.Series:
    '''end-to-end time = search + retrieval. falls back to search time for
    old csvs that predate the retrieval column.'''
    if "end_to_end_s" in df.columns:
        return df.end_to_end_s
    if "retrieval_s" in df.columns:
        return df.time_s + df.retrieval_s.fillna(0.0)
    return df.time_s


def _fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--"
    ms = seconds * 1000
    if ms < 1:
        return f"${ms:.2f}$\\,ms"
    if ms < 1000:
        return f"${ms:.1f}$\\,ms"
    return f"${ms/1000:.2f}$\\,s"


def _bold_max(vals: List[float], i: int, s: str) -> str:
    finite = [v for v in vals if np.isfinite(v)]
    if finite and np.isfinite(vals[i]) and vals[i] == max(finite):
        return r"\textbf{" + s + "}"
    return s


def table_pareto(beam: int = 100,
                 src: str = "data_out/exp4_pplus_default/default.csv"
                 ) -> Optional[str]:
    '''the csv carries no beam column, so the label falls back to this
    parameter -- keep it on the shared default from scripts14/common_env.sh'''
    '''Reference-eligible quality and actual attempted fresh total latency.'''
    p = Path(src)
    if not p.exists():
        print(f"  [skip] {p} missing")
        return None
    df = pd.read_csv(p)
    present = set(df.method)

    m_pts = None
    if "expansion_budget" in df.columns:
        v = df[df.method == "PTS"].expansion_budget.dropna()
        m_pts = int(v.iloc[0]) if len(v) else None
    if "beam" in df.columns:
        v = df[df.method == "PACS"].beam.dropna()
        if len(v):
            beam = int(v.iloc[0])

    order = ["PACS", "PTS", "PBS-unbounded", "Pareto-DP", "Pareto-DP-conv", "DFS",
             "eps-ILP g=2", "eps-ILP g=4", "eps-ILP g=8", "eps-ILP g=16", "PBF"]
    disp = {
        "PACS": r"\texttt{PBS} ($b{=}" + str(beam) + "$)",
        "PTS": r"\texttt{PTS} ($M{=}" + (str(m_pts) if m_pts else "500") + "$)",
        "PBF": r"\texttt{BF}",
        "PBS-unbounded": r"PBS unbounded",
        "Pareto-DP": r"Pareto DP",
        "Pareto-DP-conv": r"Pareto DP (conv.)",
        "DFS": r"DFS",
    }
    for g in (2, 4, 8, 16):
        disp[f"eps-ILP g={g}"] = r"\texttt{$\epsilon$-ILP} $g{=}" + str(g) + "$"
    missing = [m for m in order if m not in present]
    if missing:
        print(f"  [warn] absent from default.csv, rows omitted: {', '.join(missing)}")
        print(f"         methods present: {', '.join(sorted(map(str, present)))}")
    order = [m for m in order if m in present]

    required = {"metric_eligible", "method_wall_s", "total_s", "completed", "termination_reason"}
    if not required.issubset(df.columns):
        raise ValueError("Pareto tables require current measured reporting fields")
    rows = []
    for m in order:
        s_ = df[df.method == m]
        done = s_[s_.metric_eligible == 1]
        e2e = s_.total_s.dropna()
        rows.append({
            "m": m,
            "recall": done.recall.mean() if len(done) else np.nan,
            "hv": done.hv_ratio.mean() if len(done) else np.nan,
            "runs": s_.solves.mean(),
            # fraction of requested queries with at least one returned assignment
            "success": s_.success.mean(),
            "median": e2e.median(),
            "p90": e2e.quantile(0.9),
            "n": len(done), "total": len(s_),
        })

    recs = [r["recall"] for r in rows]
    hvs = [r["hv"] for r in rows]

    out = [r"\setlength{\tabcolsep}{3.5pt}%",
           r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"method & recall & HV & success & runs & median & p90 \\", r"\midrule"]
    for i, r in enumerate(rows):
        if r["m"] in ("eps-ILP g=16", "PBF") and i > 0:
            out.append(r"\midrule")
        out.append(" & ".join([
            disp.get(r["m"], r["m"]),
            _bold_max(recs, i, f"{r['recall']:.3f}" if np.isfinite(r["recall"]) else "--"),
            _bold_max(hvs, i, f"{r['hv']:.3f}" if np.isfinite(r["hv"]) else "--"),
            f"{r['success']:.2f}" if np.isfinite(r["success"]) else "--",
            f"{r['runs']:.0f}" if np.isfinite(r["runs"]) else "--",
            _fmt_time(r["median"]), _fmt_time(r["p90"]),
        ]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]

    for r in rows:
        if r["n"] != r["total"]:
            print(f"  for the caption -- {r['m']} scored on {r['n']}/{r['total']} "
                  "queries with complete references")
    print(f"  reference is complete on {df.ref_is_exact.mean()*100:.0f}% of queries")
    return "\n".join(out)


'''deadlines reported on amazon. other datasets keep whichever of these their
grid contains and fill up from the ladder below.'''


@click.command()
@click.option("--which", type=str, default="pareto")
@click.option("--src", type=str, default=None,
              help="default.csv to summarize; the default is the Amazon exp4_pplus_default run")
def main(which, src):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {"pareto": lambda: table_pareto(src=src) if src else table_pareto(),
            "pareto_airbnb": lambda: table_pareto(
                src=src or "data_out/exp4_airbnb_default/default.csv")}
    names = list(jobs) if which == "all" else [w.strip() for w in which.split(",")]
    for n in names:
        print(f"[{n}]")
        tex = jobs[n]()
        if tex is None:
            continue
        (OUT / f"table_{n}.tex").write_text(tex + "\n")
        print(f"  wrote {OUT}/table_{n}.tex")
        print(tex)
        print()


if __name__ == "__main__":
    main()
