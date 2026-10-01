'''
memory footprint of the retrieval+search pipeline, one catalog tier per
process.

peak RSS (ru_maxrss) is monotone for the life of a process, so two tiers in
one process would report the larger tier's peak for both. this module
therefore measures exactly ONE --query-setting per invocation and writes
<out-dir>/mem_<setting>.json; scripts16/03_memory.sh loops the tiers and
merges the json files.

the loading path mirrors gquery/alg/exp4/bench_scale_xl.py: the catalog is
built with the same get_query_setting -> load_items -> item_from_row ->
create_embedding_cache chain that core.load_instances uses internally, and
retrieval is then run by handing that catalog back to load_instances as
items_cache, exactly like the ladder does. the difference is that the catalog
build is hoisted out of load_instances so the "after load" and "after
retrieval" phases can be sampled separately.
'''

from __future__ import annotations

import json
import platform
import resource
import subprocess
import time
from pathlib import Path

'''baseline is taken before click / numpy / torch / gquery are imported: at
this point only the stdlib is resident. ru_maxrss is KB on linux.'''
_RSS_BASELINE_KB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

import click  # noqa: E402

SETTINGS = ["e11_cat10k", "e11_cat30k", "e12_cat1m", "e12_cat2m"]


def _rss_mb() -> float:
    # peak resident set size of this process so far; ru_maxrss is KB on linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _reset_peak(torch_mod, device) -> None:
    '''reset_peak_memory_stats raises "Invalid device argument" on a device
    whose cuda context has not been created yet; nothing is allocated there in
    that case, so there is no peak to reset.'''
    try:
        torch_mod.cuda.reset_peak_memory_stats(device)
    except RuntimeError:
        pass


class _Gpu:
    '''torch.cuda accounting scoped to the device the embedding cache picked.
    every method is a no-op returning None when cuda is unavailable, so the
    bench still records RSS on a cpu-only box.'''

    def __init__(self, torch_mod, device):
        self.torch = torch_mod
        self.device = device
        self.on = (torch_mod is not None and device is not None
                   and str(device).startswith("cuda")
                   and torch_mod.cuda.is_available())

    def reset(self) -> None:
        if self.on:
            self.torch.cuda.synchronize(self.device)
            _reset_peak(self.torch, self.device)

    def peak_mb(self):
        if not self.on:
            return None
        self.torch.cuda.synchronize(self.device)
        return self.torch.cuda.max_memory_allocated(self.device) / 2 ** 20

    def allocated_mb(self):
        if not self.on:
            return None
        return self.torch.cuda.memory_allocated(self.device) / 2 ** 20

    def reserved_mb(self):
        if not self.on:
            return None
        return self.torch.cuda.memory_reserved(self.device) / 2 ** 20

    def max_reserved_mb(self):
        if not self.on:
            return None
        return self.torch.cuda.max_memory_reserved(self.device) / 2 ** 20


def _tensor_mb(t) -> float:
    return t.element_size() * t.nelement() / 2 ** 20


@click.command()
@click.option("--query-setting", type=str,
              default="e11_cat10k",
              help="one tier per process, see module doc; any ioutils "
                   "setting is accepted (SETTINGS lists the amazon ladder)")
@click.option("--num-queries", type=int, default=100)
@click.option("--alpha", type=float, default=0.5)
@click.option("--beta", type=float, default=0.8)
@click.option("--expansion-budget", type=int, default=500)
@click.option("--candidate-limit", type=int, default=50)
@click.option("--timeout-seconds", type=float, default=60.0,
              help="per-instance guard for solve_pts, as in bench_scale_xl")
@click.option("--force-retrieval/--allow-cached-retrieval", default=True,
              help="bypass the on-disk slot cache (load_instances "
                   "use_disk_cache=False) so retrieval really runs on the GPU. "
                   "with --allow-cached-retrieval a cache hit is reported as "
                   "retrieval_cached=true and the retrieval GPU fields are null")
@click.option("--methods", default="PBS,PTS,eps-ILP,PBS-unbounded,Pareto-DP-conv", help="Each search method runs in a fresh CPU process")
@click.option("--out-dir", type=str, default="data_out/exp4_memory")
def main(query_setting, num_queries, alpha, beta, expansion_budget,
         candidate_limit, timeout_seconds, force_retrieval, methods, out_dir):
    rss_after_imports_mb = None

    '''heavy imports are deferred so _RSS_BASELINE_KB above is a real
    interpreter-only baseline.'''
    from gquery.alg.common import create_embedding_cache, item_from_row
    from gquery.alg.exp4.core import _slot_cache_path, load_instances, timed
    from gquery.alg.pareto.pts import solve_pts
    from gquery.ioutils import get_query_setting, load_items
    from gquery.utils import setup_logging

    try:
        import torch
    except ImportError:
        torch = None

    logger = setup_logging("exp4_memory")
    rss_after_imports_mb = _rss_mb()

    slot_cache = _slot_cache_path(query_setting, num_queries, beta, alpha)
    retrieval_cached = (not force_retrieval) and slot_cache.exists()

    # ---------------- phase 1: catalog + embedding cache ----------------
    '''same chain as core.load_instances (lines "query_file, item_file =
    get_query_setting(...)" through "cache = create_embedding_cache(items)"),
    lifted out so the load phase can be timed and sampled on its own.'''
    # the cache picks its own device (most free memory), so clear every device
    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            _reset_peak(torch, f"cuda:{i}")

    t0 = time.perf_counter()
    query_file, item_file = get_query_setting(query_setting)
    items = [item_from_row(row) for _, row in load_items(item_file).iterrows()]
    cache = create_embedding_cache(items)
    load_s = time.perf_counter() - t0
    items_cache = (item_file, items, cache)

    device = getattr(cache, "device", None) if cache is not None else None
    gpu = _Gpu(torch, device)
    rss_after_load_mb = _rss_mb()
    gpu_after_load_mb = gpu.peak_mb()
    gpu_reserved_after_load_mb = gpu.reserved_mb()

    # embedding matrix size straight off the live tensor(s)
    embedding_matrix_mb = None
    embedding_fp16_mb = None
    embedding_dim = None
    n_embeddings = None
    if cache is not None:
        n_embeddings = len(cache.item_indices)
        if getattr(cache, "embeddings_tensor", None) is not None:
            embedding_matrix_mb = _tensor_mb(cache.embeddings_tensor)
            embedding_dim = int(cache.embeddings_tensor.shape[1])
            if getattr(cache, "fp16_prefilter", False) and \
                    getattr(cache, "embeddings_fp16", None) is not None:
                embedding_fp16_mb = _tensor_mb(cache.embeddings_fp16)
        elif getattr(cache, "embeddings_array", None) is not None:
            embedding_matrix_mb = cache.embeddings_array.nbytes / 2 ** 20
            embedding_dim = int(cache.embeddings_array.shape[1])

    logger.info(f"{query_setting}: {len(items)} items, load {load_s:.0f}s, "
                f"device={device}, rss={rss_after_load_mb:,.0f}MB, "
                f"gpu={gpu_after_load_mb}")

    # ---------------- phase 2: retrieval ----------------
    '''retrieval is not callable on its own here: load_instances runs it
    internally (cache.retrieve_candidates_gpu_batch per query), so the phase
    is instrumented *around* that single call. --force-retrieval passes
    use_disk_cache=False so the pickled slots are neither read nor written and
    the GPU pass really happens.'''
    gpu.reset()
    t0 = time.perf_counter()
    insts, _ = load_instances(query_setting, num_queries, beta, alpha,
                              items_cache=items_cache,
                              use_disk_cache=not force_retrieval, include_invalid=True)
    retrieval_cached = not any(getattr(i, "retrieval_measured", False) for i in insts)
    retrieval_wall_s = time.perf_counter() - t0
    rss_after_retrieval_mb = _rss_mb()
    gpu_peak_retrieval_mb = None if retrieval_cached else gpu.peak_mb()
    retrieval_s_total = float(sum(i.retrieval_s for i in insts))
    n_candidates_total = int(sum(i.n_candidates for i in insts))

    logger.info(f"{query_setting}: {len(insts)} instances, "
                f"retrieval {retrieval_s_total:.2f}s "
                f"(cached={retrieval_cached}), rss={rss_after_retrieval_mb:,.0f}MB, "
                f"gpu_peak={gpu_peak_retrieval_mb}")

    # Each child loads stripped candidates only, without the shared catalog.
    import pickle
    import tempfile
    import sys
    from gquery.alg.exp4.protocol import write_manifest
    children = {}
    with tempfile.TemporaryDirectory(prefix="r5-memory-") as tmp:
        payload = Path(tmp) / "instances.pkl"
        payload.write_bytes(pickle.dumps(list(insts)))
        for method in methods.split(","):
            target = Path(tmp) / (method + ".json")
            subprocess.run([sys.executable, "-m", "gquery.alg.exp4.memory_worker",
                            str(payload), str(target), method, str(alpha),
                            str(expansion_budget), str(candidate_limit), str(timeout_seconds)],
                           check=True, cwd=tmp)
            children[method] = json.loads(target.read_text())
    search_s_total = sum(v["method_wall_s"] for v in children.values())
    n_timeouts = sum(v["timeouts"] for v in children.values())
    n_solved = sum(v["returned"] for v in children.values())
    rss_after_search_mb = None
    gpu_peak_search_mb = None

    rec = {
        "search_processes": children,
        "shared_rss_semantics": "monotonic process high water marks, not phase increments",
        "status_counts": insts.status_counts,
        "setting": query_setting,
        "n_items": len(items),
        "n_queries": len(insts),
        "n_queries_requested": num_queries,
        "n_candidates_total": n_candidates_total,

        "rss_baseline_mb": _RSS_BASELINE_KB / 1024.0,
        "rss_after_imports_mb": rss_after_imports_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "rss_after_retrieval_mb": rss_after_retrieval_mb,
        "rss_after_search_mb": rss_after_search_mb,

        "gpu_after_load_mb": gpu_after_load_mb,
        "gpu_reserved_after_load_mb": gpu_reserved_after_load_mb,
        "gpu_peak_retrieval_mb": gpu_peak_retrieval_mb,
        "gpu_reserved_peak_mb": gpu.max_reserved_mb(),
        "gpu_peak_search_mb": gpu_peak_search_mb,
        "gpu_device": str(device) if device is not None else None,
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),

        # measured off the live tensors, not estimated
        "embedding_matrix_mb": embedding_matrix_mb,
        "embedding_fp16_mb": embedding_fp16_mb,
        "embedding_dim": embedding_dim,
        "n_embeddings": n_embeddings,

        "load_s": load_s,
        "retrieval_s_total": retrieval_s_total if not retrieval_cached else None,
        "retrieval_historical_s_total": retrieval_s_total if retrieval_cached else None,
        "retrieval_wall_s": retrieval_wall_s,
        "search_s_total": search_s_total,
        "retrieval_cached": bool(retrieval_cached),
        "slot_cache_path": str(slot_cache),
        "n_search_timeouts": n_timeouts,
        "n_search_solved": n_solved,

        "alpha": alpha, "beta": beta,
        "expansion_budget": expansion_budget,
        "candidate_limit": candidate_limit,
        "timeout_seconds": timeout_seconds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "torch_version": getattr(torch, "__version__", None),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, {"query_setting": query_setting, "methods": methods,
                         "alpha": alpha, "beta": beta, "expansion_budget": expansion_budget,
                         "candidate_limit": candidate_limit, "timeout_seconds": timeout_seconds}, insts)
    path = out / f"mem_{query_setting}.json"
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)

    print(f"\n=== memory footprint: {query_setting} ===")
    for k, v in rec.items():
        print(f"  {k:28s} {v}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
