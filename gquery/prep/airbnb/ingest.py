#!/usr/bin/env python3
"""
Convert raw Inside Airbnb detailed-listings CSVs into per-city parquet catalogs
that match the Amazon-review-2023 catalog schema used everywhere else in this
repo.

Why: the CGQ experiments treat one Amazon *category* as one group. Inside Airbnb
gives us a second, structurally different benchmark where one *city* plays the
role of a category ("one listing per city" multi-city trip queries). To reuse the
existing query-generation / solver / evaluation code unchanged, the Airbnb data
has to land in exactly the same column layout as
  ~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_10k_each/
      <Category>/batch_000000.parquet
minus the `title_embedding` column, which a later stage adds.

Two dataset-specific traps are handled here (both documented in
~/dataset/inside_airbnb/INVESTIGATION.md):

  1. Currency. The `price` column is a string with a cosmetic "$" prefix, but the
     actual currency is the CITY's currency (Tokyo's "$19,746.00" is JPY 19,746).
     Every price is converted to USD with a pinned FX table before anything else,
     otherwise a shared budget constraint across cities is meaningless.

  2. Saturated rating. `review_scores_rating` has median 4.8-4.9 with 5.0 as the
     modal value, which is nearly degenerate as a CGQ utility. We instead build a
     composite: the mean of the six `review_scores_*` sub-dimensions (accuracy,
     cleanliness, checkin, communication, location, value), requiring at least 4
     of the 6 to be present.

Output layout:
  {out_dir}/{City_Name}/batch_000000.parquet

Usage:
  python -m gquery.prep.airbnb.ingest
  python -m gquery.prep.airbnb.ingest --raw-dir ~/dataset/inside_airbnb/raw \
      --out-dir ~/dataset/inside_airbnb/catalog
"""

import html
import json
import os
import random
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
BASE_DIR = os.path.join(DATASET_ROOT, "inside_airbnb")
DEFAULT_RAW_DIR = os.path.join(BASE_DIR, "raw")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "catalog")
DEFAULT_SEED = 42

# The 12 cities downloaded in the investigation. Order is by raw size, largest
# first, so the slow files fail fast if something is wrong.
CITIES = [
    "london",
    "paris",
    "los-angeles",
    "tokyo",
    "rome",
    "mexico-city",
    "new-york-city",
    "toronto",
    "sydney",
    "barcelona",
    "berlin",
    "amsterdam",
]

# City -> local currency. From INVESTIGATION.md section 4: each city's listings
# are 100% one currency, read off `price_quote_raw`'s "currency" field.
CITY_CURRENCY = {
    "london": "GBP",
    "paris": "EUR",
    "rome": "EUR",
    "barcelona": "EUR",
    "berlin": "EUR",
    "amsterdam": "EUR",
    "tokyo": "JPY",
    "mexico-city": "MXN",
    "sydney": "AUD",
    "toronto": "CAD",
    "los-angeles": "USD",
    "new-york-city": "USD",
}

# Pinned FX rates, approximate June 2026 mid-market, one unit of local currency
# in USD. Pinned (not fetched) so the catalog is reproducible: re-running this
# script a year from now must produce byte-identical prices. Refresh these
# deliberately and regenerate the whole catalog if you ever do.
FX_TO_USD = {
    "GBP": 1.27,
    "EUR": 1.08,
    "JPY": 0.0064,
    "MXN": 0.054,
    "AUD": 0.66,
    "CAD": 0.73,
    "USD": 1.0,
}

# Price cap. INVESTIGATION.md section 7 flags extreme outliers (London 75,000
# GBP, Mexico City 1.14M MXN). We winsorize rather than drop: the listing is
# real, only its quote is absurd, and dropping would bias the group sizes.
WINSOR_Q = 0.995

# The six review sub-dimensions averaged into the composite rating.
SUBSCORE_COLS = [
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]
MIN_SUBSCORES = 4

MIN_REVIEWS = 3
MAX_TITLE_FEATS = 12
MAX_AMENITY_TOKENS = 8

USECOLS = [
    "name",
    "description",
    "host_id",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "amenities",
    "price",
    "number_of_reviews",
] + SUBSCORE_COLS

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "into", "is", "it", "its", "near", "of", "on", "or", "our", "that", "the",
    "this", "to", "was", "were", "will", "with", "your", "you", "we",
}

ROOM_TYPE_TOKEN = {
    "entire home/apt": "entire_home",
    "private room": "private_room",
    "shared room": "shared_room",
    "hotel room": "hotel_room",
}

# Amenity whitelist. Inside Airbnb's amenity vocabulary has thousands of long
# tail strings ("Sonos Bluetooth sound system"); only these coarse, query-able
# facilities are kept. Each entry is (token, predicate over the lowercased
# amenity string). Substring matching, with three guards where a naive substring
# would fire on the wrong amenity: "Dishwasher" contains "washer", "Hair dryer"
# contains "dryer" and is near-universal (so useless as a discriminator), and
# "Pool table" contains "pool". "free parking" is matched as free+parking so
# that "Free street parking" and "Free residential garage" both count.
AMENITY_RULES = [
    ("wifi", lambda s: "wifi" in s or "wi-fi" in s),
    ("kitchen", lambda s: "kitchen" in s),
    ("washer", lambda s: "washer" in s and "dishwasher" not in s),
    ("dryer", lambda s: "dryer" in s and "hair dryer" not in s and "hairdryer" not in s),
    ("air_conditioning", lambda s: "air conditioning" in s),
    ("heating", lambda s: "heating" in s),
    ("pool", lambda s: "pool" in s and "pool table" not in s),
    ("hot_tub", lambda s: "hot tub" in s),
    ("free_parking", lambda s: "parking" in s and "free" in s),
    ("gym", lambda s: "gym" in s),
    ("elevator", lambda s: "elevator" in s),
    ("balcony", lambda s: "balcony" in s),
    ("dishwasher", lambda s: "dishwasher" in s),
    ("bathtub", lambda s: "bathtub" in s),
    ("crib", lambda s: "crib" in s),
    ("workspace", lambda s: "workspace" in s),
]

# Target arrow schema, mirroring the Amazon catalog minus `title_embedding`.
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("average_rating", pa.float64()),
        pa.field("price", pa.float64()),
        pa.field("main_category", pa.large_string()),
        pa.field("store", pa.large_string()),
        pa.field("rating_number", pa.int64()),
        pa.field("categories", pa.list_(pa.string())),
        pa.field("title_feats", pa.list_(pa.string())),
        pa.field("title_for_embedding", pa.large_string()),
        pa.field("keywords", pa.list_(pa.string())),
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]{1,40}>")
_DANGLING_TAG_RE = re.compile(r"<[^>]{0,40}$")
_WS_RE = re.compile(r"\s+")
_NONWORD_RE = re.compile(r"[^a-z0-9]+")


def city_to_name(city: str) -> str:
    """'new-york-city' -> 'New_York_City'. Plays the role of the Amazon category."""
    return "_".join(part.capitalize() for part in city.split("-"))


def parse_price(series: pd.Series) -> pd.Series:
    """'$1,234.00' -> 1234.0. The $ is cosmetic; the currency is the city's."""
    cleaned = series.astype(str).str.replace(r"[^0-9.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_text(value) -> str:
    """Decode HTML entities, strip <br /> and other tags, collapse whitespace.

    Entities are decoded *first* so that an escaped "&lt;br /&gt;" is also
    removed. The final substitution kills a dangling unterminated tag: Airbnb
    hard-caps `description` at 1000 characters and the cut sometimes lands
    mid-tag, leaving a trailing "<br /".
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = _BR_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _DANGLING_TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def title_tokens(name: str) -> list:
    """Lowercase alphanumeric tokens of the listing name, stopwords removed."""
    toks = [t for t in _TOKEN_RE.findall(name.lower()) if t not in STOPWORDS]
    out = []
    for t in toks:
        if t not in out:
            out.append(t)
        if len(out) >= MAX_TITLE_FEATS:
            break
    return out


def slugify(value: str) -> str:
    """'Centrum-West' -> 'centrum_west'."""
    return _NONWORD_RE.sub("_", str(value).lower()).strip("_")


def property_token(value) -> str:
    """'Entire rental unit' -> 'rental_unit'; leading entire/private/shared dropped."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    words = str(value).lower().split()
    if words and words[0] in ("entire", "private", "shared"):
        words = words[1:]
    return slugify(" ".join(words))


def sleeps_bucket(accommodates) -> str:
    try:
        n = int(float(accommodates))
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n <= 2:
        return "sleeps_1_2"
    if n <= 4:
        return "sleeps_3_4"
    if n <= 6:
        return "sleeps_5_6"
    return "sleeps_7_plus"


def amenity_tokens(raw) -> list:
    """Parse the JSON-ish amenity list and map it onto the whitelist."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    try:
        items = json.loads(raw)
    except (ValueError, TypeError):
        items = re.findall(r'"([^"]*)"', str(raw))
    if not isinstance(items, list):
        return []
    lowered = [str(a).lower() for a in items]
    out = []
    for token, pred in AMENITY_RULES:
        if any(pred(a) for a in lowered):
            out.append(token)
            if len(out) >= MAX_AMENITY_TOKENS:
                break
    return out


def build_keywords(room_type, prop_type, accommodates, neighbourhood, amenities) -> list:
    """Structured keyword list: room type, property type, size, area, amenities."""
    toks = []
    rt = ROOM_TYPE_TOKEN.get(str(room_type).strip().lower())
    if rt:
        toks.append(rt)
    toks.append(property_token(prop_type))
    toks.append(sleeps_bucket(accommodates))
    if neighbourhood is not None and not (isinstance(neighbourhood, float) and np.isnan(neighbourhood)):
        toks.append(slugify(neighbourhood))
    toks.extend(amenities)
    out = []
    for t in toks:
        if t and t not in out:
            out.append(t)
    return out


def ingest_city(city: str, raw_dir: Path, out_dir: Path) -> dict:
    """Read one raw city CSV, apply the filters, write the parquet catalog."""
    src = raw_dir / f"{city}.csv.gz"
    name = city_to_name(city)
    currency = CITY_CURRENCY[city]
    fx = FX_TO_USD[currency]
    print(f"[{name}] reading {src}")

    df = pd.read_csv(src, usecols=USECOLS, dtype={"host_id": "string"}, low_memory=False)
    n_raw = len(df)

    # --- filter 1: price parseable and > 0 -------------------------------
    price_local = parse_price(df["price"])
    keep = price_local.notna() & (price_local > 0)
    n_drop_price = int((~keep).sum())
    df = df.loc[keep].copy()
    price_local = price_local.loc[keep]

    # --- filter 2: at least MIN_REVIEWS reviews --------------------------
    n_reviews = pd.to_numeric(df["number_of_reviews"], errors="coerce")
    keep = n_reviews.notna() & (n_reviews >= MIN_REVIEWS)
    n_drop_reviews = int((~keep).sum())
    df = df.loc[keep].copy()
    price_local = price_local.loc[keep]
    n_reviews = n_reviews.loc[keep]

    # --- filter 3: composite rating computable ---------------------------
    subs = df[SUBSCORE_COLS].apply(pd.to_numeric, errors="coerce")
    n_present = subs.notna().sum(axis=1)
    composite = subs.mean(axis=1, skipna=True)
    keep = (n_present >= MIN_SUBSCORES) & composite.notna()
    n_drop_rating = int((~keep).sum())
    n_before_rating = len(df)
    df = df.loc[keep].copy()
    price_local = price_local.loc[keep]
    n_reviews = n_reviews.loc[keep]
    composite = composite.loc[keep]

    # --- filter 4: name non-null (expected ~0 per INVESTIGATION.md) ------
    keep = df["name"].notna() & (df["name"].astype(str).str.strip() != "")
    n_drop_name = int((~keep).sum())
    df = df.loc[keep].copy()
    price_local = price_local.loc[keep]
    n_reviews = n_reviews.loc[keep]
    composite = composite.loc[keep]

    n_out = len(df)
    if n_out == 0:
        raise RuntimeError(f"{city}: no rows survived the filters")

    # --- FX + winsorize --------------------------------------------------
    price_usd = price_local.astype(float) * fx
    cap = float(price_usd.quantile(WINSOR_Q))
    n_capped = int((price_usd > cap).sum())
    price_usd = price_usd.clip(upper=cap)

    composite = composite.clip(lower=0.0, upper=5.0)

    # --- text / keyword columns ------------------------------------------
    names = df["name"].map(clean_text)
    descs = df["description"].map(clean_text)
    title_for_embedding = [
        f"{nm}. {ds}" if ds else nm for nm, ds in zip(names, descs)
    ]
    feats = [title_tokens(nm) for nm in names]
    amenities = [amenity_tokens(a) for a in df["amenities"]]
    keywords = [
        build_keywords(rt, pt, ac, nb, am)
        for rt, pt, ac, nb, am in zip(
            df["room_type"],
            df["property_type"],
            df["accommodates"],
            df["neighbourhood_cleansed"],
            amenities,
        )
    ]

    store = ["" if pd.isna(v) else str(v).strip() for v in df["host_id"]]

    table = pa.table(
        {
            "average_rating": pa.array(composite.to_numpy(dtype=float), type=pa.float64()),
            "price": pa.array(price_usd.to_numpy(dtype=float), type=pa.float64()),
            "main_category": pa.array([name] * n_out, type=pa.large_string()),
            "store": pa.array(store, type=pa.large_string()),
            "rating_number": pa.array(n_reviews.to_numpy(dtype="int64"), type=pa.int64()),
            "categories": pa.array([[] for _ in range(n_out)], type=pa.list_(pa.string())),
            "title_feats": pa.array(feats, type=pa.list_(pa.string())),
            "title_for_embedding": pa.array(title_for_embedding, type=pa.large_string()),
            "keywords": pa.array(keywords, type=pa.list_(pa.string())),
        },
        schema=OUTPUT_SCHEMA,
    )

    city_dir = out_dir / name
    city_dir.mkdir(parents=True, exist_ok=True)
    dest = city_dir / "batch_000000.parquet"
    pq.write_table(table, dest)

    kw_mean = float(np.mean([len(k) for k in keywords]))
    stats = {
        "city": city,
        "name": name,
        "currency": currency,
        "fx": fx,
        "raw": n_raw,
        "drop_price": n_drop_price,
        "drop_reviews": n_drop_reviews,
        "drop_rating": n_drop_rating,
        "rating_drop_frac": n_drop_rating / n_before_rating if n_before_rating else 0.0,
        "drop_name": n_drop_name,
        "out": n_out,
        "cap_usd": cap,
        "n_capped": n_capped,
        "median_price": float(np.median(price_usd)),
        "p90_price": float(np.percentile(price_usd, 90)),
        "median_rating": float(np.median(composite)),
        "mean_keywords": kw_mean,
        "path": str(dest),
    }
    print(
        f"[{name}] raw={n_raw} -price={n_drop_price} -reviews={n_drop_reviews} "
        f"-rating={n_drop_rating} -name={n_drop_name} -> out={n_out} "
        f"({n_out / n_raw:.1%}) | {currency}x{fx} cap=${cap:,.0f} capped={n_capped} "
        f"| med=${stats['median_price']:,.0f} p90=${stats['p90_price']:,.0f} "
        f"rating={stats['median_rating']:.3f} kw={kw_mean:.1f}"
    )
    return stats


@click.command()
@click.option("--raw-dir", default=DEFAULT_RAW_DIR, show_default=True,
              help="Directory holding <city>.csv.gz Inside Airbnb detailed listings.")
@click.option("--out-dir", default=DEFAULT_OUT_DIR, show_default=True,
              help="Output catalog root; one <City_Name>/batch_000000.parquet per city.")
@click.option("--seed", default=DEFAULT_SEED, show_default=True,
              help="RNG seed. The pipeline is deterministic; the seed is pinned "
                   "only so any future sampling stage stays reproducible.")
@click.option("--cities", default=",".join(CITIES), show_default=False,
              help="Comma-separated city slugs to ingest (default: all 12).")
def main(raw_dir, out_dir, seed, cities):
    """Ingest Inside Airbnb city CSVs into Amazon-schema parquet catalogs."""
    random.seed(seed)
    np.random.seed(seed)
    raw_dir = Path(os.path.expanduser(raw_dir))
    out_dir = Path(os.path.expanduser(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    city_list = [c.strip() for c in cities.split(",") if c.strip()]

    all_stats = []
    for city in city_list:
        all_stats.append(ingest_city(city, raw_dir, out_dir))

    total_raw = sum(s["raw"] for s in all_stats)
    total_out = sum(s["out"] for s in all_stats)
    print(f"\nTotal: {total_raw:,} raw rows -> {total_out:,} catalog rows "
          f"({total_out / total_raw:.1%}) across {len(all_stats)} cities")
    print(f"Catalog written under {out_dir}")

    stats_path = out_dir / "ingest_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"Per-city stats: {stats_path}")


if __name__ == "__main__":
    main()
