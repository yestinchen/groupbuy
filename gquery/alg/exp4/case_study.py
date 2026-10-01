'''qualitative case study -> case_study.tex. selection is automatic, not
hand-picked: among queries with a mid-sized exact frontier and <= 4 slots,
take the one whose frontier spans the widest normalized range in all 3
objectives.'''

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

from gquery.alg.exp4.core import (
    exact_frontier, instance_status, load_instances, pkey, to_points,
)
from gquery.alg.exp4.default_reporting import save_query, validated_reference
from gquery.alg.exp4.sensitivity_reporting import reference_for
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.pts import solve_pts
from gquery.ioutils import get_query_setting, load_items
from gquery.utils import setup_logging

logger = setup_logging("exp4_case_study")

# internal label -> paper name, same map as make_tables / plot_exp4
_DISPLAY = {"PACS": "PBS"}


def _title_map(query_setting: str) -> Dict[str, str]:
    # Item.title is None for this catalog - titles live in title_for_embedding.
    # join readable titles back here
    _, item_file = get_query_setting(query_setting)
    df = load_items(item_file)
    col = next((c for c in ("title", "title_for_embedding", "title_feats")
                if c in df.columns), None)
    if col is None:
        return {}
    return {str(i): str(t) for i, t in zip(df["id"], df[col])}


def _spread(front) -> float:
    # geometric mean of the normalized range in each objective
    if len(front) < 3:
        return 0.0
    qs = [p.quality for p in front]
    cs = [p.cost for p in front]
    rs = [p.rating for p in front]
    out = 1.0
    for v in (qs, cs, rs):
        lo, hi = min(v), max(v)
        out *= (hi - lo) / hi if hi > 0 else 0.0
    return out ** (1 / 3)


def _tex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


@click.command()
@click.option("--num-queries", type=int, default=100)
@click.option("--query-setting", type=str, default="e10_1k_100q")
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--beam", type=int, default=100)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--min-front", type=int, default=6)
@click.option("--max-front", type=int, default=60)
@click.option("--n-rows", type=int, default=6)
@click.option("--timeout-seconds", type=float, default=60.0,
              help="reference computation limit per query")
@click.option("--selection", type=click.Choice(["spread", "method_gap"]), default="spread",
              help="spread: widest normalized objective range; method_gap: the query on which PTS recovers the most reference points that PBS misses, net of the reverse")
@click.option("--exp-name", type=str, default="exp4_pplus_case")
def main(num_queries, query_setting, alpha, beta, beam, expansion_budget,
         candidate_limit, min_front, max_front, n_rows, timeout_seconds,
         selection, exp_name):
    batch, _ = load_instances(query_setting, num_queries, beta, alpha, include_invalid=True)
    records = list(getattr(batch, "requested_records", batch))
    out = Path("data_out") / exp_name
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(requested=num_queries, available=len(records),
                      status_counts=dict(getattr(batch, "status_counts", {})),
                      selection_rule="feasible, validated complete reference, "
                                     f"{min_front} <= |P| <= {max_front}, <= 4 requirements, "
                                     + ("largest count of reference points recovered by PTS but not PBS, net of the reverse, then widest range"
                                        if selection == "method_gap" else "widest geometric-mean normalized objective range"),
                      selection=selection,
                      timeout_seconds=timeout_seconds, alpha=alpha, beta=beta, beam=beam,
                      expansion_budget=expansion_budget, candidate_limit=candidate_limit,
                      considered=[])

    best = None
    for inst in records:
        if instance_status(inst, alpha) != "feasible":
            continue
        reference = validated_reference(reference_for(inst, alpha, timeout_seconds, exact_frontier), inst, alpha)
        front = reference[0]
        complete = bool(reference[1] and reference.status == "complete" and front)
        provenance["considered"].append(dict(
            query_id=inst.query.id, n_slots=inst.n_slots, reference_status=reference.status,
            reference_complete=complete, reference_size=len(front), reference_s=reference[2]))
        if not complete or not (min_front <= len(front) <= max_front):
            continue
        # a short query fits in a table, a 8-slot bundle does not
        if inst.n_slots > 4:
            continue
        s = _spread(front)
        score: Tuple = (s,)
        if selection == "method_gap":
            # one untraced run per method to rank the candidate queries by the discovery gap
            keys_ref = {pkey(p) for p in front}
            got = {}
            for name, run in (("PACS", lambda: solve_pbs(inst.slots, inst.query.budget, alpha, beam_size=beam,
                                                         prefix_rule="raw_cost", use_reduction=True)),
                              ("PTS", lambda: solve_pts(inst.slots, inst.query.budget, alpha, expansion_budget=expansion_budget,
                                                        candidate_limit=candidate_limit, use_reduction=True))):
                got[name] = {pkey(p) for p in to_points(run().frontier, inst.query.budget)} & keys_ref
            pts_only, pbs_only = len(got["PTS"] - got["PACS"]), len(got["PACS"] - got["PTS"])
            provenance["considered"][-1].update(pts_only=pts_only, pbs_only=pbs_only)
            score = (pts_only - pbs_only, s)
        if best is None or score > best[0]:
            best = (score, inst, front, reference)
    if best is None:
        provenance["selected"] = None
        (out / "case_study_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
        print("no suitable query found; relax --min-front/--max-front")
        return
    score, inst, front, reference = best
    spread = score[-1]
    b = inst.query.budget
    logger.info(f"selected query {inst.query.id}: |P|={len(front)} "
                f"slots={inst.n_slots} spread={spread:.3f}")

    # discovery traces - who found each point, and when the first one came
    prov: Dict[Tuple, List[str]] = {}
    first: Dict[str, Tuple] = {}
    finals: Dict[str, List] = {}
    traces: Dict[str, List] = {}
    for name, fn in (
        ("PACS", lambda tr: solve_pbs(
            inst.slots, b, alpha, beam_size=beam, prefix_rule="raw_cost",
            use_reduction=True, trace=tr)),
        ("PTS", lambda tr: solve_pts(
            inst.slots, b, alpha, expansion_budget=expansion_budget,
            candidate_limit=candidate_limit, use_reduction=True, trace=tr)),
    ):
        tr: List = []
        result = fn(tr)
        finals[name] = to_points(result.frontier, b)
        pts = to_points([s for _, s in tr], b)
        times = [t for t, _ in tr]
        traces[name] = [dict(t_s=t, key=list(pkey(p)), choice=list(p.choice))
                        for t, p in zip(times, pts)]
        for t, p in sorted(zip(times, pts), key=lambda x: x[0]):
            prov.setdefault(pkey(p), []).append(name)
            if name not in first:
                first[name] = (pkey(p), t)

    # sort by cost so adjacent rows differ by one trade
    rows_all = sorted(front, key=lambda p: p.cost)
    picked: List = []
    if len(rows_all) <= n_rows:
        picked = rows_all
    elif selection == "method_gap":
        # show the trade-offs that only PTS recovers beside shared ones, each evenly spaced by cost
        found = {m: {pkey(p) for p in finals[m]} for m in finals}
        only_keys = {pkey(p) for p in rows_all if pkey(p) in found["PTS"] and pkey(p) not in found["PACS"]}
        only = [p for p in rows_all if pkey(p) in only_keys]
        rest = [p for p in rows_all if pkey(p) not in only_keys]

        def spaced(seq, k):
            if k <= 0 or not seq:
                return []
            idx = np.linspace(0, len(seq) - 1, min(k, len(seq))).round().astype(int)
            return [seq[i] for i in sorted(set(idx.tolist()))]
        k = min(len(only), (n_rows + 1) // 2)
        picked = sorted(spaced(only, k) + spaced(rest, n_rows - k), key=lambda p: p.cost)
    else:
        # endpoints + evenly spaced interior points, keep the extremes
        idx = np.linspace(0, len(rows_all) - 1, n_rows).round().astype(int)
        picked = [rows_all[i] for i in sorted(set(idx.tolist()))]

    titles = _title_map(query_setting)
    recs = []
    for p in picked:
        k = pkey(p)
        items = [inst.slots[d][p.choice[d]][0] for d in range(inst.n_slots)]
        who = sorted(set(prov.get(k, [])))
        recs.append({
            "titles": [(it.title or titles.get(it.id) or it.id) for it in items],
            "prices": [it.price for it in items],
            "ratings": [it.rating for it in items],
            "Q": p.quality / inst.n_slots,
            "C": p.cost,
            "dC": p.cost_overage,
            "R": p.rating / inst.n_slots,
            "found_by": ", ".join(_DISPLAY.get(w, w) for w in who) if who else "--",
            "t_first_ms": min([t for m, (kk, t) in first.items() if kk == k],
                              default=np.nan) * 1000,
        })
    df = pd.DataFrame(recs)
    df.to_csv(out / "case_study.csv", index=False)
    # candidate arrays, the validated reference, and both final frontiers permit
    # an offline check of every displayed row and its discovery attribution
    save_query(out / "inputs", inst, alpha, reference, finals)
    provenance["selected"] = dict(
        query_id=inst.query.id, spread=spread, budget=b, relaxed_cap=b * (1.0 + alpha),
        reference_size=len(front), n_slots=inst.n_slots, reference_s=reference[2], selection=selection,
        pts_only_points=len({pkey(p) for p in finals['PTS']} - {pkey(p) for p in finals['PACS']}),
        pbs_only_points=len({pkey(p) for p in finals['PACS']} - {pkey(p) for p in finals['PTS']}),
        picked=[dict(choice=list(p.choice), key=list(pkey(p))) for p in picked],
        traces=traces, first_discovery={m: dict(key=list(k), t_s=t) for m, (k, t) in first.items()})
    (out / "case_study_provenance.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")

    # sanity check: nothing displayed can exceed the relaxed budget
    cap = b * (1.0 + alpha)
    bad = df[df.C > cap + 1e-9]
    if not bad.empty:
        logger.error(f"{len(bad)} displayed bundles exceed the relaxed budget")

    n = inst.n_slots
    lines = [
        r"% generated by gquery.alg.exp4.case_study -- do not edit by hand",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "l" * n + r"rrrrl}",
        r"\toprule",
        (" & ".join([r"\#"] + [f"req.\\ {i+1}" for i in range(n)]
                    + [r"$Q$", r"$C$", r"$\Delta C$", r"$R$", r"found by"])
         + r" \\"),
        r"\midrule",
    ]
    for i, r in df.iterrows():
        cells = [str(i + 1)]
        for t, pr in zip(r["titles"], r["prices"]):
            short = t if len(t) <= 14 else t[:13].rstrip() + "."
            cells.append(f"{_tex_escape(short)}\\,(\\${pr:.0f})")
        cells += [f"{r['Q']:.3f}", f"\\${r['C']:.2f}", f"\\${r['dC']:.2f}",
                  f"{r['R']:.2f}", r["found_by"]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    (out / "case_study.tex").write_text(tex + "\n")

    print(f"\nquery {inst.query.id}: budget \\${b:.2f}, "
          f"relaxed cap \\${cap:.2f}, |exact frontier| = {len(front)}, "
          f"{inst.n_slots} requirements")
    print(df[["Q", "C", "dC", "R", "found_by"]].to_string(index=False))
    print(f"\nwrote {out}/case_study.tex")


if __name__ == "__main__":
    main()
