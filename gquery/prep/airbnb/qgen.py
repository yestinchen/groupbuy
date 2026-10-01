"""
Airbnb CGQ query generation: one stay per city under a shared trip budget.

Mirrors the Amazon qgen11 scheme (same output parquet contract, same budget
multiplier scheme) with one documented deviation: requirement descriptions are
templated from listing attributes instead of LLM-generated.

Stages (run in order; each writes its artifact next to the catalog):
  build-pool   sample anchor listings per city, template requirement texts
  embed-pool   embed the pool texts with Qwen3-Embedding-0.6B (GPU)
  calibrate    report per-beta candidate counts + cross-city leakage
  generate     emit queries.parquet (e10 contract) at a chosen beta

A requirement's text names its city and the catalog embeddings carry an
appended "Located in <city>." sentence (gquery/prep/airbnb/embed.py), so retrieval
concentrates in-city without any category mask in the search code. The
calibrate stage measures how well that holds (leakage = matching items from
other cities).

Usage:
  python -m gquery.prep.airbnb.qgen build-pool
  python -m gquery.prep.airbnb.qgen embed-pool
  python -m gquery.prep.airbnb.qgen calibrate
  python -m gquery.prep.airbnb.qgen generate --beta 0.55
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

# cleaned-title embeddings (clean_titles.py -> embed.py), aligning with the
# Amazon pipeline's LLM-cleaned titles; the raw-title tree (catalog_embed)
# is kept for comparison
CATALOG = os.path.expanduser("~/dataset/inside_airbnb/catalog_titled_embed")
# keywords/prices are identical in the pre-embedding catalog (embed.py copies
# rows in order), so the CPU-only pool build does not wait on the GPU stage
CATALOG_RAW = os.path.expanduser("~/dataset/inside_airbnb/catalog")
OUT_ROOT = os.path.expanduser("~/dataset/inside_airbnb/data_out/qgen/airbnb_12c")

ROOM_TOKENS = {"entire_home", "private_room", "shared_room", "hotel_room"}
AMENITY_TOKENS = {
    "wifi", "kitchen", "washer", "dryer", "air_conditioning", "heating",
    "pool", "hot_tub", "free_parking", "gym", "elevator", "balcony",
    "dishwasher", "bathtub", "crib", "workspace",
}
SLEEPS_GUESTS = {"sleeps_1_2": 2, "sleeps_3_4": 4, "sleeps_5_6": 6,
                 "sleeps_7_plus": 8}

MULT_TIGHT = (0.8, 0.99)
MULT_EASY = (1.1, 1.5)
WITHIN_BUDGET_FRACTION = 0.6


def _parse_keywords(kw: list) -> dict:
    '''positional parse anchored on the sleeps_* token (always present):
    [room?, property, sleeps, neighbourhood?, amenities...].'''
    kw = list(kw)
    si = next((i for i, t in enumerate(kw) if t.startswith("sleeps_")), None)
    if si is None or si == 0:
        return {}
    room = kw[0] if kw[0] in ROOM_TOKENS else None
    prop = kw[si - 1]
    rest = kw[si + 1:]
    neigh = next((t for t in rest if t not in AMENITY_TOKENS), None)
    amen = [t for t in rest if t in AMENITY_TOKENS]
    return {"room": room, "prop": prop, "sleeps": kw[si],
            "neigh": neigh, "amen": amen}


def _template(parsed: dict, city_disp: str, rng: random.Random) -> str:
    prop = parsed["prop"].replace("_", " ")
    guests = SLEEPS_GUESTS.get(parsed["sleeps"], 2)
    room = parsed["room"]
    if room == "private_room":
        head = f"Private room in a {prop}"
    elif room == "shared_room":
        head = f"Shared room in a {prop}"
    elif room == "hotel_room":
        head = "Hotel room"
    else:
        head = f"Entire {prop}"
    amen = rng.sample(parsed["amen"], k=min(2, len(parsed["amen"])))
    with_part = f" with {' and '.join(a.replace('_', ' ') for a in amen)}" \
        if amen else ""
    neigh = parsed["neigh"]
    where = f"in {neigh.replace('_', ' ').title()}, {city_disp}" if neigh \
        else f"in {city_disp}"
    return f"{head} for {guests} guests {where}{with_part}."


def _load_catalog(catalog: str = CATALOG):
    frames = []
    for d in sorted(Path(catalog).iterdir()):
        f = d / "batch_000000.parquet"
        if f.exists():
            frames.append(pd.read_parquet(
                f, columns=["main_category", "price", "title_embedding"]))
    df = pd.concat(frames, ignore_index=True)
    emb = np.stack(df.title_embedding.to_numpy()).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return df.main_category.to_numpy(), df.price.to_numpy(), emb


@click.group()
def cli():
    pass


@cli.command("build-pool")
@click.option("--pool-per-city", type=int, default=60, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def build_pool(pool_per_city, seed):
    rng = random.Random(seed)
    rows = []
    for d in sorted(Path(CATALOG_RAW).iterdir()):
        f = d / "batch_000000.parquet"
        if not f.exists():
            continue
        city = d.name
        city_disp = city.replace("_", " ")
        df = pd.read_parquet(f, columns=["keywords", "price"])
        idx = rng.sample(range(len(df)), min(pool_per_city * 3, len(df)))
        taken = 0
        for i in idx:
            parsed = _parse_keywords(list(df.keywords.iloc[i]))
            if not parsed:
                continue
            rows.append({"city": city, "anchor_row": i,
                         "query_desc": _template(parsed, city_disp, rng)})
            taken += 1
            if taken >= pool_per_city:
                break
        print(f"  {city}: {taken} requirements")
    out = Path(OUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "pool.parquet", index=False)
    print(f"wrote {out}/pool.parquet ({len(rows)} requirements)")


@cli.command("embed-pool")
@click.option("--base-url", type=str, default="http://localhost:8124/v1",
              show_default=True)
def embed_pool(base_url):
    '''embeds via the same vLLM server as embed.py (port 8124).'''
    from openai import OpenAI
    from gquery.prep.airbnb.embed import MODEL, embed_texts
    out = Path(OUT_ROOT)
    df = pd.read_parquet(out / "pool.parquet")
    client = OpenAI(api_key="token-somethinghere", base_url=base_url)
    df["query_desc_embedding"] = embed_texts(client, MODEL,
                                             list(df.query_desc))
    df.to_parquet(out / "pool_embedded.parquet", index=False)
    print(f"wrote {out}/pool_embedded.parquet")


def _match_stats(cities, prices, emb, pool, betas):
    '''per requirement and beta: matching count, in-city count, min price.'''
    q = np.stack(pool.query_desc_embedding.to_numpy()).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims = q @ emb.T                                   # (n_req, n_items)
    recs = []
    for bi, beta in enumerate(betas):
        m = sims >= beta
        for i in range(len(pool)):
            sel = m[i]
            n = int(sel.sum())
            incity = int((cities[sel] == pool.city.iloc[i]).sum())
            minp = float(prices[sel].min()) if n else np.nan
            recs.append({"beta": beta, "req": i, "city": pool.city.iloc[i],
                         "n_match": n, "n_incity": incity, "min_price": minp})
    return pd.DataFrame(recs)


@cli.command("calibrate")
@click.option("--betas", type=str, default="0.4,0.45,0.5,0.55,0.6,0.65,0.7")
def calibrate(betas):
    out = Path(OUT_ROOT)
    pool = pd.read_parquet(out / "pool_embedded.parquet")
    cities, prices, emb = _load_catalog()
    bl = [float(x) for x in betas.split(",")]
    df = _match_stats(cities, prices, emb, pool, bl)
    df.to_csv(out / "calibration.csv", index=False)
    print(f"{len(pool)} requirements x {len(bl)} betas "
          f"(amazon default: median 160 candidates, p90 527)")
    print(f"{'beta':>6} {'med_match':>10} {'p90_match':>10} {'zero%':>6} "
          f"{'in-city%':>9}")
    for beta in bl:
        s = df[df.beta == beta]
        leak = 1 - s.n_incity.sum() / max(s.n_match.sum(), 1)
        print(f"{beta:>6} {s.n_match.median():>10.0f} "
              f"{s.n_match.quantile(0.9):>10.0f} "
              f"{(s.n_match == 0).mean() * 100:>5.1f}% "
              f"{(1 - leak) * 100:>8.1f}%")


@cli.command("generate")
@click.option("--beta", type=float, required=True)
@click.option("--num-queries", type=int, default=100, show_default=True)
@click.option("--group-size-min", type=int, default=3, show_default=True)
@click.option("--group-size-max", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--out-name", type=str, default="queries.parquet",
              show_default=True,
              help="output file name, e.g. queries_gs4.parquet for the "
                   "fixed-length scalability sets")
@click.option("--catalog", type=str, default=CATALOG, show_default=True,
              help="catalog the requirements are matched against; the "
                   "downward ladder generates on its smallest rung "
                   "(sample_catalog_share.py) so nested rungs stay feasible")
@click.option("--out-dir", type=str, default=OUT_ROOT, show_default=True,
              help="where queries + metadata go; the embedded pool is "
                   "always read from the airbnb_12c dir")
def generate(beta, num_queries, group_size_min, group_size_max, seed,
             out_name, catalog, out_dir):
    pool = pd.read_parquet(Path(OUT_ROOT) / "pool_embedded.parquet")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cities, prices, emb = _load_catalog(catalog)
    stats = _match_stats(cities, prices, emb, pool, [beta])
    ok = stats[(stats.beta == beta) & (stats.n_match > 0)
               & stats.min_price.notna()]
    by_city: dict = {}
    for _, r in ok.iterrows():
        by_city.setdefault(r.city, []).append(int(r.req))
    usable_cities = sorted(c for c, v in by_city.items() if len(v) >= 3)
    print(f"beta={beta}: {len(ok)}/{len(stats)} requirements usable, "
          f"{len(usable_cities)} cities")

    rng = random.Random(seed)
    minp = {int(r.req): float(r.min_price) for _, r in ok.iterrows()}
    rows = []
    for gid in range(num_queries):
        gs = rng.randint(group_size_min,
                         min(group_size_max, len(usable_cities)))
        chosen = rng.sample(usable_cities, gs)   # distinct cities: a trip
        reqs = [rng.choice(by_city[c]) for c in chosen]
        req_prices = [max(1, int(round(minp[r]))) for r in reqs]
        gt_budget = int(round(sum(req_prices)))
        easy = rng.random() < WITHIN_BUDGET_FRACTION
        lo, hi = MULT_EASY if easy else MULT_TIGHT
        mult = rng.uniform(lo, hi)
        rows.append({
            "group_id": gid,
            "category_mix": "airbnb_12c",
            "categories": [pool.city.iloc[r] for r in reqs],
            "query_desc": [pool.query_desc.iloc[r] for r in reqs],
            "query_desc_embedding": [np.asarray(
                pool.query_desc_embedding.iloc[r], dtype=np.float32)
                for r in reqs],
            "prices": req_prices,
            "budget": max(1, int(round(gt_budget * mult))),
            "gt_budget": gt_budget,
            "group_size": gs,
            "threshold": beta,
            "multiplier": mult,
            "within_budget": bool(easy),
            "seed": seed,
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out / out_name, index=False)
    meta = {"beta": beta, "num_queries": num_queries, "seed": seed,
            "pool": str(Path(OUT_ROOT) / "pool_embedded.parquet"),
            "catalog": catalog, "catalog_items": int(len(prices)),
            "distinct_cities_per_query": True,
            "within_budget_share": float(df.within_budget.mean()),
            "median_group_size": float(df.group_size.median())}
    stem = out_name.rsplit(".", 1)[0]
    (out / f"generate_metadata_{stem}.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}/{out_name} "
          f"({len(df)} queries, median size {df.group_size.median():.0f}, "
          f"{df.within_budget.mean() * 100:.0f}% within budget)")


if __name__ == "__main__":
    cli()
