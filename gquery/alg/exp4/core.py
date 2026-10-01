'''shared stuff for the exp4 suite: load workload, score frontiers, and
build the reference frontier (unbounded-beam PBS, checked by verify_exact).'''

from __future__ import annotations

import itertools
import pickle
import threading
import time
import warnings
import hashlib
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from gquery.alg.exp4.protocol import synchronize, sha256

from gquery.alg.common import (
    GroupQuery, Item, create_embedding_cache, item_from_row, query_from_row,
    retrieve_candidates_with_quality,
)
from gquery.alg.candidates import _FINAL_DP as _DP, PState
from gquery.alg.pareto.pbs import solve_pbs
from gquery.alg.pareto.metrics import _Point, _filter_non_dominated, hypervolume_3d
from gquery.ioutils import get_query_setting, load_app_queries, load_items, file_name_path_dict


# ---------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    query: GroupQuery
    slots: List[List[Tuple[Item, float]]]
    retrieval_s: float
    n_candidates: int
    n_slots: int
    # min feasible cost <= budget, i.e. no relaxation needed
    within_budget: bool
    min_feasible_cost: float
    load_status: str = "ready"


def _min_feasible_cost(slots) -> float:
    return (sum(min(it.price for it, _ in s) for s in slots)
            if slots and all(slots) else float("inf"))


def _slot_cache_path(query_setting: str, num_queries: int, beta: float,
                     alpha: float) -> Path:
    return (Path("data_out") / "exp4_slots" /
            f"{query_setting}_{num_queries}_b{beta}_a{alpha}.pkl")


class InstanceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "gquery.alg.exp4.core4" and name == "Instance":
            return Instance
        return super().find_class(module, name)


def load_instances(
    query_setting: str,
    num_queries: int,
    beta: float,
    alpha: float,
    items_cache: Optional[Tuple] = None,
    use_disk_cache: bool = True,
    include_invalid: bool = False,
) -> Tuple[List[Instance], Tuple]:
    """Return one record per requested source row, including invalid inputs.

    Historical caches are read only when IDs, budgets and requirement counts
    match the current source. New caches use a separate status-v1 suffix so
    older artifacts are never overwritten. Historical catalog identity remains
    unverified; audit_workloads records that provenance separately.
    """
    legacy_path = _slot_cache_path(query_setting, num_queries, beta, alpha)
    cache_path = legacy_path.with_suffix(".status-v1.pkl")
    query_file, item_file = get_query_setting(query_setting)
    queries_df = load_app_queries(query_file).head(num_queries)
    queries = [query_from_row(row) for _, row in queries_df.iterrows()]
    if use_disk_cache:
        for candidate in (cache_path, legacy_path):
            if not candidate.exists():
                continue
            with open(candidate, "rb") as fh:
                saved = InstanceUnpickler(fh).load()
            if isinstance(saved, dict):
                if saved.get("fingerprint") != workload_fingerprint(query_file, item_file):
                    continue
                saved = saved["instances"]
            by_id = {i.query.id: i for i in saved}
            if len(by_id) != len(saved) or set(by_id) != {q.id for q in queries}:
                continue
            if all(by_id[q.id].query.budget == q.budget
                   and by_id[q.id].n_slots == requirement_count(q)
                   and embedding_status(q) == "ready" for q in queries):
                records = [by_id[q.id] for q in queries]
                cache_hash = sha256(candidate)
                fingerprint = workload_fingerprint(query_file, item_file)
                for inst in records:
                    inst.retrieval_measured = False
                    inst.cache_path = str(candidate.resolve())
                    inst.cache_sha256 = cache_hash
                    inst.retrieval_source = "disk_cache_historical"
                    inst.provenance = {"cache": inst.cache_path, "cache_sha256": inst.cache_sha256,
                                       "current_inputs": fingerprint,
                                       "historical_catalog_verified": candidate == cache_path}
                return _reported_batch(records, alpha, include_invalid), items_cache

    if items_cache is not None and items_cache[0] == item_file:
        _, items, cache = items_cache
    else:
        items = [item_from_row(row) for _, row in load_items(item_file).iterrows()]
        cache = create_embedding_cache(items)
        items_cache = (item_file, items, cache)

    out: List[Instance] = []
    for query in queries:
        embs = query.query_desc_embeddings
        n = requirement_count(query)
        status = embedding_status(query)
        if status != "ready":
            out.append(Instance(_strip_query(query), [[] for _ in range(n)],
                                0.0, 0, n, False, float("inf"), status))
            continue
        maxb = query.budget * (1.0 + alpha)
        kws = query.keywords if query.keywords is not None else None
        synchronize(cache)
        t0 = time.perf_counter()
        if cache is not None:
            # all requirements in one matmul + one device sync
            slots = cache.retrieve_candidates_gpu_batch(list(embs), beta, maxb)
        else:
            slots = [retrieve_candidates_with_quality(
                items, kws[r] if kws is not None else [], beta, maxb, True, embs[r], cache)
                for r in range(n)]
        synchronize(cache)
        retrieval_s = time.perf_counter() - t0
        if len(slots) != n:
            slots = [[] for _ in range(n)]
        mfc = _min_feasible_cost(slots)
        slots = [[(_strip(it), qq) for it, qq in s] for s in slots]
        out.append(Instance(_strip_query(query), slots, retrieval_s,
                            sum(len(s) for s in slots), n,
                            mfc <= query.budget + 1e-9, mfc,
                            "ready" if slots and all(slots) else "empty_requirement"))

    fingerprint = workload_fingerprint(query_file, item_file)
    for inst in out:
        inst.retrieval_measured = inst.load_status != "missing_embedding"
        inst.retrieval_source = "fresh_retrieval"
        inst.provenance = {"current_inputs": fingerprint,
                           "device": str(getattr(cache, "device", "cpu")),
                           "fp16_prefilter": bool(getattr(cache, "fp16_prefilter", False)),
                           "boundaries": "query transfer through CPU candidate materialization; synchronized before and after"}
    if use_disk_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"schema": 1, "instances": out,
                         "fingerprint": workload_fingerprint(query_file, item_file)},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)
    return _reported_batch(out, alpha, include_invalid), items_cache


@lru_cache(maxsize=256)
def _file_hash(path, size, mtime_ns):
    # Stat fields invalidate the process-local optimization. Stored identity is
    # the content hash, not a timestamp or path alone.
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def workload_fingerprint(query_file, item_file):
    paths = [Path(file_name_path_dict[query_file])]
    catalog = Path(file_name_path_dict[item_file])
    paths += sorted(catalog.rglob("*.parquet")) if catalog.is_dir() else [catalog]
    return {str(p): _file_hash(str(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in paths}


class InstanceBatch(list):
    """Legacy iterable plus explicit requested membership and input statuses.

    New reporting runners must set include_invalid=True. Legacy solvers retain
    nonempty inputs until their reporting loops are migrated. Omissions warn
    and remain available in requested_records, never silently disappear.
    """
    def __init__(self, records, alpha, include_invalid):
        self.requested_records = records
        self.status_counts = {}
        usable = []
        for inst in records:
            status = instance_status(inst, alpha)
            self.status_counts[status] = self.status_counts.get(status, 0) + 1
            if status in {"feasible", "budget_infeasible"}:
                usable.append(inst)
        if not include_invalid and len(usable) != len(records):
            warnings.warn("Invalid requested queries retained in requested_records; "
                          "set include_invalid=True to emit all status rows", RuntimeWarning)
        super().__init__(records if include_invalid else usable)


def requirement_count(query):
    for values in (query.query_desc, query.keywords, query.query_desc_embeddings):
        if values is not None:
            return len(values)
    return 0


def embedding_status(query):
    n = requirement_count(query)
    if n == 0:
        return "empty_requirement"
    embs = query.query_desc_embeddings
    if embs is None or len(embs) != n:
        return "missing_embedding"
    if any(e is None or np.asarray(e).size == 0
           or not np.isfinite(np.asarray(e, dtype=float)).all() for e in embs):
        return "missing_embedding"
    return "ready"


def instance_status(inst, alpha):
    status = getattr(inst, "load_status", "ready")
    if status != "ready":
        return status
    if not inst.slots or not all(inst.slots):
        return "empty_requirement"
    if _min_feasible_cost(inst.slots) > inst.query.budget * (1 + alpha) + 1e-9:
        return "budget_infeasible"
    return "feasible"


def metric_context(inst, alpha, reference_complete, reference_points, reference_status=None):
    query_status = instance_status(inst, alpha)
    reference_status = reference_status or ("not_applicable" if query_status != "feasible" else
                        "reference_timeout" if not reference_complete else
                        "complete" if reference_points else "reference_failure")
    return {"query_status": query_status, "reference_status": reference_status,
            "metric_eligible": int(reference_status == "complete"),
            "query_feasible": int(query_status == "feasible")}


def solver_status(query_status, has_output, timed_out=False, failed=False):
    if query_status != "feasible":
        return "not_run"
    if failed:
        return "solver_failure"
    if timed_out:
        return "solver_timeout"
    return "returned" if has_output else "no_assignment"


def _strip(it: Item) -> Item:
    return Item(it.id, it.keywords, it.price, it.rating, None, it.title)


def _strip_query(q: GroupQuery) -> GroupQuery:
    return GroupQuery(q.id, q.keywords, q.budget, None, q.query_desc)


# ---------------------------------------------------------------------------
# frontier representation and scoring
# ---------------------------------------------------------------------------

def to_points(states: Sequence[PState], budget: float) -> List[_Point]:
    return [_Point(s.quality, s.rating, s.cost, max(0.0, s.cost - budget), s.choice)
            for s in states]


def pkey(p: _Point) -> Tuple[float, float, float]:
    return (round(p.quality, 9), round(p.cost_overage, 9), round(p.rating, 9))


def nadir(all_points: Sequence[_Point]) -> Optional[Tuple[float, float, float]]:
    # hv reference corner, a bit below the worst value seen
    if not all_points:
        return None
    return (min(p.quality for p in all_points) - 1e-3,
            -max(p.cost_overage for p in all_points) - 1e-3,
            min(p.rating for p in all_points) - 1e-3)


def score(points: Sequence[_Point], ref_keys, ref_pt, hv_ref: float,
          *, metric_eligible: bool = True) -> Dict:
    got = {pkey(p) for p in points}
    eligible = bool(metric_eligible and ref_keys and ref_pt is not None and hv_ref > 0)
    return {
        "size": len(points),
        "recall": (len(got & ref_keys) / len(ref_keys)) if eligible else np.nan,
        "hv_ratio": (hypervolume_3d(points, ref_pt) / hv_ref) if eligible else np.nan,
    }


# ---------------------------------------------------------------------------
# reference frontiers
# ---------------------------------------------------------------------------

class ReferenceResult(tuple):
    """Three-field unpacking remains compatible; status distinguishes failure."""
    def __new__(cls, points, complete, seconds, status):
        result = super().__new__(cls, (points, complete, seconds))
        result.status = status
        return result


def exact_frontier(
    inst: Instance,
    alpha: float,
    l1: float = 1.0,
    l2: float = 1.0,
    l3: float = 1.0,
    timeout_s: float = 60.0,
) -> Tuple[List[_Point], bool, float]:
    '''exact frontier via unbounded-beam PBS. if the returned flag is False
    the timeout fired and the frontier is partial - do not use as reference.'''
    status = instance_status(inst, alpha)
    if status != "feasible":
        return ReferenceResult([], status in {"budget_infeasible", "empty_requirement"},
                               0.0, "not_applicable")
    flag = threading.Event()
    timer = threading.Timer(timeout_s, flag.set)
    timer.start()
    t0 = time.perf_counter()
    try:
        r = solve_pbs(inst.slots, inst.query.budget, alpha, l1, l2, l3,
                            beam_size=0, prefix_rule="raw_cost",
                            use_reduction=True, timeout_flag=flag)
    except MemoryError:
        return ReferenceResult([], False, time.perf_counter() - t0, "reference_memory_limit")
    except Exception:
        return ReferenceResult([], False, time.perf_counter() - t0, "reference_failure")
    finally:
        timer.cancel()
    points = to_points(r.frontier, inst.query.budget)
    status = "reference_timeout" if flag.is_set() else "complete" if points else "reference_failure"
    elapsed = time.perf_counter() - t0
    from gquery.alg.exp4.protocol import timing_record
    if not hasattr(inst, "method_timings"):
        inst.method_timings = {}
    inst.method_timings["PACS unbounded"] = timing_record(inst, "PBS unbounded", elapsed, r)
    return ReferenceResult(points, not flag.is_set(), elapsed, status)


# ---------------------------------------------------------------------------
# scalar guidance score, shared by every Pareto method
# ---------------------------------------------------------------------------

def pof(q: float, cost: float, r: float, n: int, budget: float, alpha: float,
        l1: float = 1.0, l2: float = 1.0, l3: float = 1.0) -> float:
    pen = (l2 * max(0.0, cost - budget) / (alpha * budget)) \
        if (alpha > 0 and budget > 0) else 0.0
    return l1 * q / n - pen + l3 * r / (5.0 * n)


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------

def timed(fn, timeout_s: float, logger=None, label: str = ""):
    '''run fn(flag) under a timeout event. returns (result, seconds, timed_out).'''
    flag = threading.Event()
    timer = threading.Timer(timeout_s, flag.set)
    timer.start()
    t0 = time.perf_counter()
    try:
        out = fn(flag)
    except Exception as exc:
        if logger is not None:
            logger.error(f"{label} raised {exc!r}")
        out = None
    finally:
        timer.cancel()
    return out, time.perf_counter() - t0, flag.is_set()


def timed_for(inst, alpha, fn, seconds, logger=None, name="solver"):
    """Do not invoke a solver on invalid or infeasible requested inputs."""
    if instance_status(inst, alpha) != "feasible":
        return None, float("nan"), False
    result, elapsed, hit = timed(fn, seconds, logger, name)
    from gquery.alg.exp4.protocol import timing_record
    if not hasattr(inst, "method_timings"):
        inst.method_timings = {}
    inst.method_timings[name] = timing_record(inst, name, elapsed, result)
    output = result[0] if isinstance(result, tuple) else getattr(result, "frontier", getattr(result, "choice", None))
    inst.method_timings[name]["solver_status"] = solver_status("feasible", bool(output), hit, result is None)
    return result, elapsed, hit


def _reported_batch(records, alpha, include_invalid):
    """CLI experiments with exp_name persist input/source manifests per load."""
    import click
    from gquery.alg.exp4.protocol import runner_manifest
    batch = InstanceBatch(records, alpha, include_invalid)
    context = click.get_current_context(silent=True)
    if context is not None:
        usable = sum(v for k,v in batch.status_counts.items() if k in {"feasible", "budget_infeasible"})
        click.echo(f"Requested={len(records)} usable={usable} statuses={batch.status_counts}", err=True)
    if context is not None and context.params.get("exp_name"):
        runner_manifest(batch, Path("data_out") / context.params["exp_name"])
    return batch
