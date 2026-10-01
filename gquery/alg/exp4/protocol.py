"""Revision timing and provenance. Durations are wall seconds, never cache estimates."""
from __future__ import annotations
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "r5-v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def synchronize(cache):
    device = getattr(cache, "device", None)
    if device is not None and str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


def solver_identity(method):
    import scipy
    if "eps" in method.lower():
        try:
            from scipy.optimize._highspy._core import _Highs
            h = _Highs()
            version = f"{h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}"
        except ImportError:
            version = "unknown (bundled SciPy build)"
        return {"solver_backend": "scipy.optimize.milp/HiGHS",
                "solver_version": scipy.__version__, "highs_version": version, "highs_threads_configured": 1}
    return {"solver_backend": ("specialized Python branch and bound" if
            ("weighted" in method.lower() and "ilp" in method.lower()) or method == "ILP" or method.startswith("WS-") else "Python/" + method),
            "solver_version": platform.python_version(), "highs_version": None}


def timing_record(inst, method, elapsed, result=None, output_s=None):
    elapsed = elapsed if elapsed is not None and math.isfinite(elapsed) else None
    measured = getattr(inst, "retrieval_measured", False)
    retrieval = inst.retrieval_s if measured else None
    reduction = getattr(result, "reduction_time_s", None)
    output = getattr(result, "output_time_s", None)
    if output_s is not None:
        output = (output or 0.0) + output_s
    search = getattr(result, "search_time_s", None)
    counts_after = getattr(result, "counts_after", None)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        stats = result[1]
        reduction = stats.get("reduction_time_s")
        search = stats.get("search_time_s")
        output = stats.get("output_time_s")
        counts_after = stats.get("counts_after")
    accounted = sum(v for v in (reduction, search, output) if v is not None)
    return {"completed": getattr(result, "completed", None),
            "exhausted": getattr(result, "exhausted", None),
            "termination_reason": getattr(result, "termination_reason", None),
            "completion_scope": getattr(result, "completion_scope", None),
            "candidates_searched_by_requirement": json.dumps(getattr(result, "counts_searched", None)),
            "unattributed_method_s": max(0.0, elapsed - accounted) if elapsed is not None else None,
            "output_boundary": "solver final filtering and index remapping; caller metric conversion excluded",
            "timing_schema": SCHEMA, "encoding_s": None,
            "encoding_status": "excluded_precomputed", "retrieval_s": retrieval,
            "retrieval_historical_s": None if measured else inst.retrieval_s,
            "retrieval_measured": measured,
            "cache_path": getattr(inst, "cache_path", None),
            "cache_sha256": getattr(inst, "cache_sha256", None),
            "retrieval_source": getattr(inst, "retrieval_source", "unverified_historical"),
            "reduction_s": reduction, "search_s": search, "output_s": output,
            "method_wall_s": elapsed,
            "total_s": elapsed + retrieval if retrieval is not None and elapsed is not None else None,
            "total_status": "unattempted" if elapsed is None else "measured_excluding_encoding" if measured else "unmeasured_cached_retrieval",
            "candidates_before_by_requirement": json.dumps([len(s) for s in inst.slots]),
            "candidates_after_by_requirement": json.dumps(counts_after),
            **solver_identity(method)}


def write_manifest(directory, config, instances=(), seeds=None):
    import numpy
    import scipy
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    diff = git("diff", "--binary", "HEAD")
    (directory / "source.diff").write_bytes(diff)
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    sources = {p: sha256(repo / p) for p in paths if p and (repo / p).is_file()
               and Path(p).suffix in {".py", ".sh", ".toml", ".yaml", ".json"}}
    records = getattr(instances, "requested_records", instances)
    manifest = {"schema": SCHEMA, "command": sys.argv, "config": config,
        "seeds": seeds or {}, "source_commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(), "source_hashes": sources,
        "python": sys.version, "numpy": numpy.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "host": platform.node(), "platform": platform.platform(),
        "warmup_policy": "no implicit warmup; first retrieval can include startup costs",
        "query_order": [i.query.id for i in records],
        "precision": {"torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cuda_matmul_allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                      "cpu_retrieval": "float32 normalized dot products", "gpu_retrieval":
                      "optional float16 prefilter, float32 rescore; actual cache provenance per input",
                      "solver": "Python float64"},
        "requested": len(records), "status_counts": getattr(instances, "status_counts", {}),
        "inputs": [getattr(i, "provenance", {}) for i in records],
        "solvers": {m: solver_identity(m) for m in ["PBS", "PTS", "eps-ILP", "weighted-ILP"]}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def report_row(inst, alpha, row):
    """Attach measured phases; never synthesize a cached total."""
    from gquery.alg.exp4.core import instance_status
    method = row.get("method", row.get("family", "unknown"))
    key = method + "/" + row["arm"] if "arm" in row else method
    elapsed = row.get("time_s", row.get("search_s"))
    if method == "PACS" and "beam" in row:
        key = f"PACS b={row['beam']}"
    elif method == "PTS" and "expansion_budget" in row and "setting" in row:
        key = f"PTS M={row['expansion_budget']}"
    if "workload" in row and "param" in row:
        key = f"PACS b={row['param']}" if method == "PACS" else f"PTS M={row['param']}"
        if method == "PACS" and row["workload"] == "synthetic":
            key += " " + row["rule"]
    timings = getattr(inst, "method_timings", {})
    if method == "PACS" and key not in timings and "PACS+" in timings:
        key = "PACS+"
    fields = timings.get(key, timing_record(inst, method, elapsed))
    status = instance_status(inst, alpha)
    row.update(fields)
    row["query_status"] = status
    row["query_feasible"] = int(status == "feasible")
    row.setdefault("solver_status", status if status != "feasible" else
                   "returned" if row.get("success", row.get("size", 0)) else "no_assignment")
    row["time_s"] = fields["method_wall_s"]
    row["end_to_end_s"] = fields["total_s"]
    if not row.get("metric_eligible", False):
        for metric in ("recall", "hv_ratio"):
            if metric in row:
                row[metric] = float("nan")
    return row


def runner_manifest(instances, directory):
    import click
    context = click.get_current_context(silent=True)
    config = context.params if context else {}
    return write_manifest(Path(directory) / "manifests" / f"load-{time.time_ns()}", config,
                          instances, {k:v for k,v in config.items() if "seed" in k})
