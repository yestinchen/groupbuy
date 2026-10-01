"""Dataset registry: query files, catalogs, and the named query settings the runners take.

Every path is relative to DATASET_ROOT (default ~/dataset). The files are
produced by the preparation pipelines (DATAPREP.md, scripts/prep/).
"""
import os
from pathlib import Path

import pandas as pd
from pyarrow import parquet as pq

_DATASET_ROOT = os.environ.get("DATASET_ROOT", os.path.expanduser("~/dataset"))
BASE_DIR = os.path.join(_DATASET_ROOT, "amazon_review_2023") + os.sep
AIRBNB_DIR = os.path.join(_DATASET_ROOT, "inside_airbnb") + os.sep

file_name_path_dict = {
    # ---- Amazon catalogs
    # 10,000 items per category, 11 categories: the 110K default catalog
    "SAMPLE_10k_EACH": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_10k_each/",
    # nested larger catalogs for the catalog-size sweep
    "SAMPLE_15k_EACH": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_15k_each/",
    "SAMPLE_20k_EACH": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_20k_each/",
    "SAMPLE_25k_EACH": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_25k_each/",
    "SAMPLE_30k_EACH": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_30k_each/",
    # the million-item ladder: cat1m = 0.5 share of the pool on top of the 30k base (nested),
    # cat2m = the full 11-category pool (2.14M items)
    "SAMPLE_CAT1M": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_cat1m/",
    "SAMPLE_CAT2M": f"{BASE_DIR}/meta_parquet_clean_attributes_keywords_embed_merged_cat2m/",
    # ---- Amazon queries
    # the 1,000-query pool (two-mode budget rule)
    "sample_gqueries_e10_1k_10k_each_v2": f"{BASE_DIR}/data_out/qgen/group_keyword_w_title_qgen10_v2/threshold_0.8/ab_af_ap_acs_bp_cpa_el_hpc_ps_so_tg.parquet",
    # the 100-query benchmark: the first 100 pool queries with a feasible assignment at alpha=0
    "sample_gqueries_e10_1k_100q": f"{BASE_DIR}/data_out/qgen/e10_1k_100q/queries.parquet",
    # the same 100 queries with budgets re-anchored to each larger catalog's minimum feasible cost
    "sample_gqueries_e10_1k_100q_recal_cat30k": f"{BASE_DIR}/data_out/qgen/e10_1k_100q/queries_recal_cat30k.parquet",
    "sample_gqueries_e10_1k_100q_recal_cat1m": f"{BASE_DIR}/data_out/qgen/e10_1k_100q/queries_recal_cat1m.parquet",
    "sample_gqueries_e10_1k_100q_recal_cat2m": f"{BASE_DIR}/data_out/qgen/e10_1k_100q/queries_recal_cat2m.parquet",
    # fixed group size sets for the query-length sweep
    "sample_gqueries_e11_gs2": f"{BASE_DIR}/data_out/qgen/group_keyword_w_title_qgen11_fixed_gs/threshold_0.8/ab_af_ap_acs_bp_cpa_el_hpc_ps_so_tg_gs2.parquet",
    "sample_gqueries_e11_gs4": f"{BASE_DIR}/data_out/qgen/group_keyword_w_title_qgen11_fixed_gs/threshold_0.8/ab_af_ap_acs_bp_cpa_el_hpc_ps_so_tg_gs4.parquet",
    "sample_gqueries_e11_gs6": f"{BASE_DIR}/data_out/qgen/group_keyword_w_title_qgen11_fixed_gs/threshold_0.8/ab_af_ap_acs_bp_cpa_el_hpc_ps_so_tg_gs6.parquet",
    "sample_gqueries_e11_gs8": f"{BASE_DIR}/data_out/qgen/group_keyword_w_title_qgen11_fixed_gs/threshold_0.8/ab_af_ap_acs_bp_cpa_el_hpc_ps_so_tg_gs8.parquet",
    # ---- Inside Airbnb catalogs: 12 cities, one listing per city per requirement
    "SAMPLE_AIRBNB_12C": f"{AIRBNB_DIR}/catalog_titled_embed/",
    # nested per-city shares 1/8, 1/4, 1/2 of the pool (the full pool is the top rung)
    "SAMPLE_AIRBNB_LAD29K": f"{AIRBNB_DIR}/catalog_titled_embed_lad29k/",
    "SAMPLE_AIRBNB_LAD58K": f"{AIRBNB_DIR}/catalog_titled_embed_lad58k/",
    "SAMPLE_AIRBNB_LAD115K": f"{AIRBNB_DIR}/catalog_titled_embed_lad115k/",
    # ---- Inside Airbnb queries
    "sample_gqueries_airbnb_12c": f"{AIRBNB_DIR}/data_out/qgen/airbnb_12c/queries.parquet",
    "sample_gqueries_airbnb_gs2": f"{AIRBNB_DIR}/data_out/qgen/airbnb_12c/queries_gs2.parquet",
    "sample_gqueries_airbnb_gs4": f"{AIRBNB_DIR}/data_out/qgen/airbnb_12c/queries_gs4.parquet",
    "sample_gqueries_airbnb_gs6": f"{AIRBNB_DIR}/data_out/qgen/airbnb_12c/queries_gs6.parquet",
    "sample_gqueries_airbnb_gs8": f"{AIRBNB_DIR}/data_out/qgen/airbnb_12c/queries_gs8.parquet",
    # ladder queries generated on the 29K rung, and their budgets re-anchored per rung
    "sample_gqueries_airbnb_ladder": f"{AIRBNB_DIR}/data_out/qgen/airbnb_ladder/queries.parquet",
    "sample_gqueries_airbnb_ladder_recal_lad58k": f"{AIRBNB_DIR}/data_out/qgen/airbnb_ladder/queries_recal_lad58k.parquet",
    "sample_gqueries_airbnb_ladder_recal_lad115k": f"{AIRBNB_DIR}/data_out/qgen/airbnb_ladder/queries_recal_lad115k.parquet",
    "sample_gqueries_airbnb_ladder_recal_lad231k": f"{AIRBNB_DIR}/data_out/qgen/airbnb_ladder/queries_recal_lad231k.parquet",
}


def load_app_queries(file_code):
    """Query table: one row per query with the requirement keywords, the requirement
    embeddings, the budget, and the source categories; `id` is the row index."""
    file_path = file_name_path_dict[file_code]
    frame = pq.read_table(file_path).to_pandas()
    frame['id'] = frame.index
    return frame


def load_items(file_code):
    """Item catalog: every parquet file under the registered directory (nested
    category/batch layouts included), concatenated; `id` is the row index."""
    file_path = Path(file_name_path_dict[file_code])
    if file_path.is_dir():
        files = sorted(p for p in file_path.rglob('*.parquet') if p.is_file())
        if not files:
            raise FileNotFoundError(f"No parquet files found under directory: {file_path}")
        dfs = []
        for file in files:
            df = pq.read_table(file).to_pandas()
            if 'features_feats' in df.columns:
                df = df.drop(columns=['features_feats'])
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
    else:
        data = pq.read_table(file_path).to_pandas()
        if 'features_feats' in data.columns:
            data = data.drop(columns=['features_feats'])
    data['id'] = data.index
    return data


def get_query_setting(query_code: str):
    """(query file key, catalog key) of a named setting."""
    setting_dict = {
        # Amazon default workload: the 100 queries on the 110K catalog
        'e10_1k_100q': ['sample_gqueries_e10_1k_100q', 'SAMPLE_10k_EACH'],
        # the 1,000-query pool the benchmark was selected from
        'e10_1k_v2': ['sample_gqueries_e10_1k_10k_each_v2', 'SAMPLE_10k_EACH'],
        # query-length sweep: fixed group sizes on the 110K catalog
        'e11_gs2': ['sample_gqueries_e11_gs2', 'SAMPLE_10k_EACH'],
        'e11_gs4': ['sample_gqueries_e11_gs4', 'SAMPLE_10k_EACH'],
        'e11_gs6': ['sample_gqueries_e11_gs6', 'SAMPLE_10k_EACH'],
        'e11_gs8': ['sample_gqueries_e11_gs8', 'SAMPLE_10k_EACH'],
        # catalog-size sweep: the same 100 queries, growing catalogs
        'e11_cat10k': ['sample_gqueries_e10_1k_100q', 'SAMPLE_10k_EACH'],
        'e11_cat15k': ['sample_gqueries_e10_1k_100q', 'SAMPLE_15k_EACH'],
        'e11_cat20k': ['sample_gqueries_e10_1k_100q', 'SAMPLE_20k_EACH'],
        'e11_cat25k': ['sample_gqueries_e10_1k_100q', 'SAMPLE_25k_EACH'],
        'e11_cat30k': ['sample_gqueries_e10_1k_100q', 'SAMPLE_30k_EACH'],
        # the million-item ladder (nested, so feasibility only improves with the catalog)
        'e12_cat1m': ['sample_gqueries_e10_1k_100q', 'SAMPLE_CAT1M'],
        'e12_cat2m': ['sample_gqueries_e10_1k_100q', 'SAMPLE_CAT2M'],
        # budgets re-anchored per rung: budget'_q = m_q * mfc_target(q) with m_q = budget_q / mfc_10k(q),
        # which keeps the designed tight/easy mix at every scale (110K is the anchor)
        'e11_cat30k_recal': ['sample_gqueries_e10_1k_100q_recal_cat30k', 'SAMPLE_30k_EACH'],
        'e12_cat1m_recal': ['sample_gqueries_e10_1k_100q_recal_cat1m', 'SAMPLE_CAT1M'],
        'e12_cat2m_recal': ['sample_gqueries_e10_1k_100q_recal_cat2m', 'SAMPLE_CAT2M'],
        # Airbnb default workload: the 100 queries on the 231K catalog
        'airbnb_12c': ['sample_gqueries_airbnb_12c', 'SAMPLE_AIRBNB_12C'],
        'airbnb_gs2': ['sample_gqueries_airbnb_gs2', 'SAMPLE_AIRBNB_12C'],
        'airbnb_gs4': ['sample_gqueries_airbnb_gs4', 'SAMPLE_AIRBNB_12C'],
        'airbnb_gs6': ['sample_gqueries_airbnb_gs6', 'SAMPLE_AIRBNB_12C'],
        'airbnb_gs8': ['sample_gqueries_airbnb_gs8', 'SAMPLE_AIRBNB_12C'],
        # Airbnb catalog ladder: the ladder queries with fixed 29K budgets ...
        'airbnb_lad29k': ['sample_gqueries_airbnb_ladder', 'SAMPLE_AIRBNB_LAD29K'],
        'airbnb_lad58k': ['sample_gqueries_airbnb_ladder', 'SAMPLE_AIRBNB_LAD58K'],
        'airbnb_lad115k': ['sample_gqueries_airbnb_ladder', 'SAMPLE_AIRBNB_LAD115K'],
        'airbnb_lad231k': ['sample_gqueries_airbnb_ladder', 'SAMPLE_AIRBNB_12C'],
        # ... and with budgets re-anchored to each rung (the controlled ladder)
        'airbnb_lad58k_recal': ['sample_gqueries_airbnb_ladder_recal_lad58k', 'SAMPLE_AIRBNB_LAD58K'],
        'airbnb_lad115k_recal': ['sample_gqueries_airbnb_ladder_recal_lad115k', 'SAMPLE_AIRBNB_LAD115K'],
        'airbnb_lad231k_recal': ['sample_gqueries_airbnb_ladder_recal_lad231k', 'SAMPLE_AIRBNB_12C'],
    }
    return setting_dict[query_code]
