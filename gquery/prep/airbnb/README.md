# gquery.prep.airbnb — Inside Airbnb → Amazon-schema catalog

Ingestion stage for the second CGQ benchmark. Converts the raw Inside Airbnb
*detailed listings* CSVs into per-city parquet catalogs that are drop-in
compatible with the Amazon-review-2023 catalogs the rest of the repo reads.

One **city** plays the role of one Amazon **category**: it is the group in the
"one item per group" grouped-query construction (here, one listing per city for
a multi-city trip).

```
stage 0  download          ~/dataset/inside_airbnb/raw/<city>.csv.gz      (see INVESTIGATION.md)
stage 1  ingest  <-- THIS  ~/dataset/inside_airbnb/catalog/<City_Name>/batch_000000.parquet
stage 2  embed             adds `title_embedding` (not done here)
```

Run:

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate llmsq
python -m gquery.prep.airbnb.ingest \
    --raw-dir ~/dataset/inside_airbnb/raw \
    --out-dir ~/dataset/inside_airbnb/catalog \
    --seed 42
```

`--cities` takes a comma-separated list of city slugs if you only want a subset.
The pipeline is fully deterministic; `--seed` is pinned only so a future
sampling stage stays reproducible.

## Output schema

Identical to
`~/dataset/amazon_review_2023/meta_parquet_clean_attributes_keywords_embed_merged_10k_each/<Category>/batch_000000.parquet`
minus `title_embedding`, which stage 2 adds.

| column | type | content |
|---|---|---|
| `average_rating` | double | composite of the six review sub-scores, in [0, 5] |
| `price` | double | nightly price in **USD**, winsorized, always > 0 |
| `main_category` | large_string | the `City_Name` (e.g. `New_York_City`) |
| `store` | large_string | `host_id` as a string (`""` if null) |
| `rating_number` | int64 | `number_of_reviews` |
| `categories` | list\<string\> | always empty (matches Amazon, where it is also empty) |
| `title_feats` | list\<string\> | ≤12 lowercase alphanumeric tokens of the listing name, stopwords removed |
| `title_for_embedding` | large_string | `"<name>. <description>"`, HTML-cleaned |
| `keywords` | list\<string\> | structured facility/type/size/area tokens |

The only schema difference from Amazon is the element type of `categories`:
Amazon stores `list<null>` because the column is empty there too. Everything
else matches by name, order and arrow type.

## FX table

`price` in the raw CSV is a string with a **cosmetic `$` prefix**; the real
currency is the city's (Tokyo's `"$19,746.00"` is JPY 19,746). Rates are
**pinned**, not fetched, so re-running the ingest reproduces the catalog
byte-for-byte. Approximate mid-market June 2026, one local unit in USD:

| currency | → USD | cities |
|---|---|---|
| GBP | 1.27 | london |
| EUR | 1.08 | paris, rome, barcelona, berlin, amsterdam |
| JPY | 0.0064 | tokyo |
| MXN | 0.054 | mexico-city |
| AUD | 0.66 | sydney |
| CAD | 0.73 | toronto |
| USD | 1.00 | los-angeles, new-york-city |

After conversion the price is **winsorized per city at the 99.5th percentile**
(capped, never dropped — the listing is real, only its spot quote is absurd;
dropping would bias group sizes). Between 26 and 201 listings per city are
capped.

## Rating composite

`review_scores_rating` is **not** used: it is saturated (median 4.8–4.9, 5.0 is
the modal value, ~25% of listings sit exactly at 5.0). Instead

```
average_rating = mean(review_scores_{accuracy, cleanliness, checkin,
                                    communication, location, value})
```

skipping nulls, requiring **≥4 of the 6** sub-scores to be present. This
de-saturates the utility considerably: only **3.7%** of catalog rows land
exactly at 5.0, versus ~25% for the raw `review_scores_rating`.

Empirically the six sub-scores are **all-or-nothing** in these snapshots: every
listing has either 0 or all 6, so the "≥4 of 6" rule never binds beyond the
review-count filter (see the `-rating` column below, which is 0 everywhere).
The rule is kept because it is the correct guard if a future snapshot is
partial.

## Filters

Applied per city, in this order:

1. `price` non-null, parseable after stripping `$` and thousands separators, and > 0
2. `number_of_reviews >= 3`
3. rating composite computable (≥4 of 6 sub-scores)
4. `name` non-null and non-blank

## Keywords

Per listing, deduplicated, order-stable:

- **room type** — `entire_home` / `private_room` / `shared_room` / `hotel_room`
- **property type** — lowercased, leading `entire`/`private`/`shared` dropped,
  non-alphanumerics → `_` (`"Entire rental unit"` → `rental_unit`,
  `"Private room in rental unit"` → `room_in_rental_unit`)
- **size bucket** — `sleeps_1_2` / `sleeps_3_4` / `sleeps_5_6` / `sleeps_7_plus`
- **neighbourhood slug** — `neighbourhood_cleansed` slugified (`"Centrum-West"` → `centrum_west`)
- **up to 8 amenities** from a fixed whitelist: `wifi, kitchen, washer, dryer,
  air_conditioning, heating, pool, hot_tub, free_parking, gym, elevator,
  balcony, dishwasher, bathtub, crib, workspace`

Amenities are matched case-insensitively by substring, with four guards where a
naive substring fires on the wrong thing: `"Dishwasher"` must not count as
`washer`; `"Hair dryer"` (near-universal, so useless as a discriminator) must
not count as `dryer`; `"Pool table"` must not count as `pool`; and
`free_parking` requires both *free* and *parking* so that `"Free street
parking"` counts.

Resulting list length is 4–12 tokens, mean 9.4–11.3 per city.

## Per-city output counts

Counts dropped at each filter step, on the 2026-06/07/08 snapshots:

| City | Cur | ×USD | Raw | −price | −reviews | −rating | −name | **Out** | Yield | Cap (USD) | Capped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| London | GBP | 1.27 | 92638 | 30398 | 21910 | 0 | 0 | **40330** | 44% | 1842 | 201 |
| Paris | EUR | 1.08 | 77679 | 29277 | 13357 | 0 | 0 | **35045** | 45% | 1897 | 176 |
| Rome | EUR | 1.08 | 37084 | 2760 | 6768 | 0 | 0 | **27556** | 74% | 1425 | 138 |
| Tokyo | JPY | 0.0064 | 34419 | 2058 | 7007 | 0 | 1 | **25353** | 74% | 1210 | 120 |
| Los_Angeles | USD | 1.00 | 43751 | 5565 | 14493 | 0 | 0 | **23693** | 54% | 2760 | 119 |
| Mexico_City | MXN | 0.054 | 31430 | 1793 | 7659 | 0 | 0 | **21978** | 70% | 1190 | 110 |
| Sydney | AUD | 0.66 | 20573 | 2787 | 4898 | 0 | 0 | **12888** | 63% | 1547 | 65 |
| New_York_City | USD | 1.00 | 30234 | 9903 | 8057 | 0 | 0 | **12274** | 41% | 1502 | 62 |
| Toronto | CAD | 0.73 | 22212 | 4655 | 5852 | 0 | 0 | **11705** | 53% | 1088 | 59 |
| Barcelona | EUR | 1.08 | 15293 | 1938 | 4646 | 0 | 0 | **8709** | 57% | 1212 | 44 |
| Berlin | EUR | 1.08 | 12776 | 4335 | 2102 | 0 | 0 | **6339** | 50% | 809 | 32 |
| Amsterdam | EUR | 1.08 | 10369 | 3992 | 1321 | 0 | 0 | **5056** | 49% | 1440 | 26 |
| **Total** | | | **428458** | | | | | **230926** | **54%** | | 1152 |

Distributions of the ingested catalog (USD):

| City | Median | p90 | Median rating | Mean keywords |
|---|---:|---:|---:|---:|
| London | 226 | 579 | 4.815 | 10.3 |
| Paris | 222 | 559 | 4.833 | 9.9 |
| Rome | 173 | 376 | 4.848 | 10.5 |
| Tokyo | 127 | 290 | 4.817 | 10.6 |
| Los_Angeles | 234 | 696 | 4.878 | 11.3 |
| Mexico_City | 93 | 242 | 4.858 | 9.4 |
| Sydney | 196 | 438 | 4.833 | 11.1 |
| New_York_City | 171 | 417 | 4.820 | 10.3 |
| Toronto | 145 | 384 | 4.875 | 11.3 |
| Barcelona | 240 | 449 | 4.747 | 10.9 |
| Berlin | 160 | 346 | 4.820 | 10.2 |
| Amsterdam | 309 | 617 | 4.857 | 9.9 |

Ten of twelve cities clear 10k listings. Berlin (6.3k) and Amsterdam (5.1k) are
genuinely small markets under strict short-term-rental regulation, not a data
defect — keep them as smaller groups or swap in Madrid/Lisbon/Istanbul.

The run also writes `~/dataset/inside_airbnb/catalog/ingest_stats.json` with the
same numbers in machine-readable form, and
`~/dataset/inside_airbnb/catalog/INGEST_SUMMARY.md` records the verified summary
plus the exact command used.

## Caveats inherited from the raw data

- Snapshot dates differ per city (2026-06-15 to 2026-08-10); NYC is ~8 weeks
  after the rest.
- `price` is a dated spot quote for a per-listing-varying 1–4 night window with
  discounts applied, not a stable list price.
- `description` is hard-capped at 1000 characters, so the cut sometimes lands
  mid-tag; the cleaner strips dangling `<br` fragments as well as complete tags.
- Tokyo has heavy text duplication (boilerplate multi-unit operator listings);
  de-duplicate on `title_for_embedding` there if embedding diversity matters.

## License

Inside Airbnb data is CC BY 4.0 — attribute Inside Airbnb / Murray Cox.

## Downstream stages (embed, qgen, calibration)

- `embed.py` — client for the same vLLM embedding server the Amazon pipeline
  used (`vllm/embed_vllm_serve.sh`, port 8124, Qwen3-Embedding-0.6B). Appends
  "Located in <city>." to every listing text: retrieval is whole-catalog
  embedding similarity with no category mask, so city membership must live in
  the embedding itself.
- `qgen_airbnb.py` — build-pool / embed-pool / calibrate / generate. One stay
  per city, distinct cities per query, budgets via the qgen11 multiplier
  scheme (0.8-0.99 tight / 1.1-1.5 easy, 60% within budget). Deviation from
  Amazon: requirement texts are templated from listing attributes, not
  LLM-generated.
- `clean_titles.py` — Qwen2.5-7B-Instruct rewrite of every listing name into
  a short natural stay title (mirrors gquery/prep/amazon/refine_titles.py;
  same generate server, port 8123). Raw titles kept in `title_raw`. This is
  the step that aligns the Airbnb text pipeline with Amazon's LLM-cleaned
  titles; the embedding tree for cleaned titles is `catalog_titled_embed/`.
- Calibration (720 requirements): with RAW titles no round beta matched the
  Amazon profile (0.7 gave median 347 candidates; 0.75/0.8 left 5%/61.5% of
  requirements empty). With CLEANED titles the similarity scale matches
  Amazon's and **beta=0.8 — the Amazon default itself — is the chosen
  setting**: median 218 candidates (Amazon: 160), 99.8% in-city, 706/720
  requirements usable (the generator excludes empty ones, so every generated
  query is feasible after retrieval). See log/airbnb/calibrate_titled.log.
- Generated workload at beta=0.8: 100 queries, 3-8 cities each (median 6),
  61% within budget. Exactness gate (verify_exact) PASSES: unbounded PBS
  reproduces the exhaustive frontier.
