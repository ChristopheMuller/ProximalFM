#!/usr/bin/env python3
"""Evaluate three CATE(X) methods on the nonlinear proxy sweep."""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy

from benchmarks.compute import (
    ComputeConfig,
    WorkerManager,
    apply_compute,
    resolve_device,
)
from benchmarks.methods import build_method


HERE = Path(__file__).resolve().parent
METHOD_PROXIMALFM = "proximalfm"
METHOD_P_RF = "p_kernel_rf"
METHOD_TABICL = "s_tabicl_wz_mc64"
METHODS = (METHOD_PROXIMALFM, METHOD_P_RF, METHOD_TABICL)
CACHE_VERSION = 1
RESULT_FIELDS = [
    "dimension",
    "joint_r2_target",
    "level_id",
    "method",
    "context_size",
    "status",
    "error",
    "n_train",
    "n_test",
    "pehe",
    "npehe",
    "mse_cate",
    "ate_bias",
    "predicted_ate",
    "true_test_ate",
    "true_cate_sd",
    "fit_seconds",
    "tuning_seconds",
    "prediction_seconds",
    "prediction_max_abs",
    "diagnostics_json",
    "elapsed_seconds",
    "cached",
    "observed_path",
    "truth_path",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    selected = path
    if path.is_dir():
        best = path / "best_model.ckpt"
        if best.is_file():
            selected = best
        else:
            candidates = list(path.glob("*.ckpt"))
            if not candidates:
                return {"path": str(path), "missing": True}

            def step(candidate: Path) -> tuple[int, str]:
                match = re.search(r"(\d+)", candidate.name)
                return (int(match.group(1)) if match else -1, candidate.name)

            selected = max(candidates, key=step)
    if not selected.is_file():
        return {"path": str(selected), "missing": True}
    stat = selected.stat()
    return {
        "path": str(selected),
        "sha256": sha256(selected),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def method_configs(
    checkpoint: Path, *, tabicl_predict_batch_size: int = 16384
) -> dict[str, dict]:
    return {
        METHOD_PROXIMALFM: {
            "name": "ProximalFM",
            "cls": "ProximalFM",
            "checkpoint": str(Path(checkpoint).expanduser().resolve()),
            "ensemble": 1,
        },
        METHOD_P_RF: {
            "name": "P-Kernel-RF",
            "cls": "PLearner",
            "base": "rf",
            "bridge": "kernel",
            "n_splits": 4,
            "random_state": 42,
            "q_clip": 20.0,
            "bridge_params": {
                "rank": 256,
                "lam_grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            },
            "base_cv": {
                "folds": 4,
                "seed": 1234,
                "grids": {
                    "rf": {
                        "n_estimators": [100],
                        "max_depth": [1, 3, 5],
                        "min_samples_leaf": [1, 5, 10],
                        "max_features": [1.0],
                    }
                },
            },
        },
        METHOD_TABICL: {
            "name": "S-TabICL-WZ-MC64",
            "cls": "ProxyMarginalizer",
            "estimator": {
                "cls": "S_Learner",
                "base": "tabicl",
                "use_W": True,
                "use_Z": True,
                "tabicl": {"kv_cache": True, "n_estimators": 1},
            },
            "integration": {"method": "empirical", "n_points": 64, "seed": 0},
            "predict_batch_size": tabicl_predict_batch_size,
        },
    }


def cache_signature(args, index_path: Path, configs: dict[str, dict]) -> dict:
    checksums = args.data_dir / "SHA256SUMS.txt"
    manifest = args.data_dir / "manifest.json"
    return {
        "version": CACHE_VERSION,
        "runner_sha256": sha256(Path(__file__)),
        "index_sha256": sha256(index_path),
        "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
        "checksums_sha256": sha256(checksums) if checksums.is_file() else None,
        "checkpoint": checkpoint_identity(args.checkpoint),
        "method_configs": configs,
        "device": args.device,
        "query_batch_size": args.query_batch_size,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": package_version("scikit-learn"),
            "torch": package_version("torch"),
            "tabicl": package_version("tabicl"),
        },
    }


def cache_key(row: dict) -> tuple[str, str, str, str, int]:
    return (
        row["observed_path"],
        row["truth_path"],
        str(row["level_id"]),
        row["method"],
        int(row["context_size"]),
    )


def read_cached_results(path: Path) -> dict[tuple, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(RESULT_FIELDS).issubset(
            reader.fieldnames
        ):
            return {}
        return {cache_key(row): row for row in reader if row.get("status") != "error"}


def prepare_cache(args, signature: dict) -> tuple[dict[tuple, dict], Path]:
    cache_dir = args.output_dir / ".fit_cache"
    cache_path = cache_dir / "method_results.csv"
    metadata_path = cache_dir / "metadata.json"
    valid = False
    if not args.force and cache_path.is_file() and metadata_path.is_file():
        try:
            valid = json.loads(metadata_path.read_text()).get("signature") == signature
        except (OSError, ValueError, TypeError):
            pass
    if valid:
        return read_cached_results(cache_path), cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=RESULT_FIELDS).writeheader()
    temporary.replace(cache_path)
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    temporary_metadata.write_text(json.dumps({"signature": signature}, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
    return {}, cache_path


def resolve_data_path(data_dir: Path, spelling: str) -> Path:
    path = Path(spelling)
    return path if path.is_absolute() else data_dir / path


def load_setting(data_dir: Path, row: dict) -> tuple[dict, dict]:
    observed_path = resolve_data_path(data_dir, row["observed_path"])
    truth_path = resolve_data_path(data_dir, row["truth_path"])
    with np.load(observed_path, allow_pickle=False) as data:
        required = {"X", "W", "Z", "A", "Y", "row_id", "split"}
        if not required.issubset(data.files):
            raise ValueError(
                f"{observed_path} is missing {sorted(required - set(data.files))}"
            )
        observed = {name: data[name] for name in required}
    with np.load(truth_path, allow_pickle=False) as data:
        required = {"CATE", "row_id"}
        if not required.issubset(data.files):
            raise ValueError(
                f"{truth_path} is missing {sorted(required - set(data.files))}"
            )
        truth = {name: data[name] for name in required}
    if not np.array_equal(observed["row_id"], truth["row_id"]):
        raise ValueError("observed and truth row IDs are not aligned")
    n = len(observed["row_id"])
    if any(len(value) != n for value in (*observed.values(), *truth.values())):
        raise ValueError("dataset arrays have inconsistent row counts")
    if observed["X"].shape[1] != int(row["dimension"]):
        raise ValueError("X dimension differs from index.csv")
    return observed, truth


def restrict_context(observed: dict, context_size: int) -> dict:
    split = np.asarray(observed["split"]).reshape(-1)
    if not np.isin(split, (0, 1)).all():
        raise ValueError("split must contain only 0=train and 1=query")
    context = np.flatnonzero(split == 0)
    query = np.flatnonzero(split == 1)
    if context_size < 1 or context_size > len(context):
        raise ValueError(f"context size {context_size} must be in [1, {len(context)}]")
    if not len(query):
        raise ValueError("at least one query row is required")
    indices = np.concatenate([context[:context_size], query])
    return {name: np.asarray(value)[indices] for name, value in observed.items()}


def score_cate(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("finite predictions must align with held-out CATE")
    mse = float(np.mean((prediction - truth) ** 2))
    pehe = float(np.sqrt(mse))
    scale = float(np.std(truth))
    return {
        "pehe": pehe,
        "npehe": pehe / scale if scale > 1e-12 else "",
        "mse_cate": mse,
        "ate_bias": float(np.mean(prediction - truth)),
        "predicted_ate": float(np.mean(prediction)),
        "true_test_ate": float(np.mean(truth)),
        "true_cate_sd": scale,
    }


def clean_diagnostics(value: dict) -> dict[str, float | None]:
    cleaned = {}
    for key, item in sorted((value or {}).items()):
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        cleaned[key] = number if np.isfinite(number) else None
    return cleaned


def fit_and_predict(
    method: str,
    config: dict,
    observed: dict,
    *,
    device: str,
    query_batch_size: int,
    compute: ComputeConfig | None = None,
) -> tuple[np.ndarray, dict]:
    split = np.asarray(observed["split"]).reshape(-1)
    context, query = split == 0, split == 1
    if compute is None:
        compute = ComputeConfig(
            device=device if method != METHOD_P_RF else "cpu", execution="inline"
        )
    _, estimator = build_method(config, compute)
    began = time.perf_counter()
    estimator.fit(
        observed["X"][context],
        observed["A"][context],
        observed["Y"][context],
        W=observed["W"][context],
        Z=observed["Z"][context],
    )
    fit_seconds = time.perf_counter() - began
    tuning_seconds = float(estimator.tuning_time())

    began = time.perf_counter()
    if method == METHOD_PROXIMALFM:
        parts = []
        for start in range(0, int(query.sum()), query_batch_size):
            selected = np.flatnonzero(query)[start : start + query_batch_size]
            parts.append(
                estimator.predict(
                    observed["X"][selected],
                    W=observed["W"][selected],
                    Z=observed["Z"][selected],
                )
            )
        prediction = np.concatenate(parts)
    else:
        prediction = estimator.predict(
            observed["X"][query], W=observed["W"][query], Z=observed["Z"][query]
        )
    prediction_seconds = time.perf_counter() - began
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    return prediction, {
        "status": "ok",
        "n_train": int(context.sum()),
        "n_test": int(query.sum()),
        "fit_seconds": fit_seconds,
        "tuning_seconds": tuning_seconds,
        "prediction_max_abs": float(np.max(np.abs(prediction))),
        "diagnostics_json": json.dumps(
            clean_diagnostics(estimator.diagnostics()), sort_keys=True
        ),
        "prediction_seconds": prediction_seconds,
    }


@dataclass(frozen=True)
class FitTask:
    """One cached fit, sent to a worker or run inline.

    Paths cross the process boundary instead of arrays so that a setting is
    loaded by the worker that will fit it. This keeps the parent small and
    avoids pickling the same setting once per context and method.
    """

    data_dir: Path
    row: dict
    method: str
    config: dict
    context_size: int
    compute: ComputeConfig
    query_batch_size: int


def _compute_for_method(method: str, args) -> ComputeConfig:
    """Resolve the runner budget, keeping non-CPU devices process-local."""
    device = "cpu" if method == METHOD_P_RF else args.device
    if device != "cpu":
        return ComputeConfig(device=device, execution="inline")

    n_workers = args.workers
    return ComputeConfig(
        device="cpu",
        n_workers=n_workers,
        torch_threads=args.torch_threads,
        blas_threads=args.blas_threads,
        n_jobs=1,
        execution="pool" if n_workers > 1 else "inline",
    )


def _blas_limits(threads: int):
    """Limit already-loaded BLAS runtimes inside a spawned worker when possible."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=int(threads))


def _run_fit_task(task: FitTask) -> tuple[tuple, dict]:
    """Load, fit, score, and return one result row.

    This is module-level so it can be submitted to the repository's spawn pool.
    Errors are returned as result rows to preserve the sweep's existing behavior.
    """
    result = {
        "dimension": task.row["dimension"],
        "joint_r2_target": task.row["joint_r2_target"],
        "level_id": task.row["level_id"],
        "method": task.method,
        "context_size": task.context_size,
        "status": "error",
        "error": "",
        "observed_path": task.row["observed_path"],
        "truth_path": task.row["truth_path"],
    }
    key = cache_key(result)
    began = time.perf_counter()
    try:
        apply_compute(task.compute)
        with _blas_limits(task.compute.blas_threads):
            observed, truth = load_setting(task.data_dir, task.row)
            query_truth = truth["CATE"][observed["split"] == 1]
            restricted = restrict_context(observed, task.context_size)
            prediction, details = fit_and_predict(
                task.method,
                task.config,
                restricted,
                device=task.compute.device,
                query_batch_size=task.query_batch_size,
                compute=task.compute,
            )
        result.update(details)
        result.update(score_cate(prediction, query_truth))
    except Exception as error:  # preserve the long sweep unless requested
        result.update(status="error", error=f"{type(error).__name__}: {error}")
    result["elapsed_seconds"] = time.perf_counter() - began
    result["cached"] = "0"
    return key, result


def run(args) -> int:
    args.data_dir = Path(args.data_dir).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()
    args.device = resolve_device(args.device, what="nonlinear sweep")
    # Keep the small programmatic runner API used by the study tests and notebooks
    # compatible with older SimpleNamespace callers that predate these CLI knobs.
    if not hasattr(args, "workers"):
        args.workers = 1
    elif args.workers is None:
        args.workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    args.torch_threads = getattr(args, "torch_threads", 1)
    args.blas_threads = getattr(args, "blas_threads", 1)
    args.tabicl_predict_batch_size = getattr(args, "tabicl_predict_batch_size", 16384)
    index_path = args.data_dir / "index.csv"
    with index_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "dimension",
        "joint_r2_target",
        "level_id",
        "observed_path",
        "truth_path",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"index.csv must contain {sorted(required)}")
    if args.dimensions:
        rows = [row for row in rows if int(row["dimension"]) in args.dimensions]
    if args.levels:
        rows = [row for row in rows if int(row["level_id"]) in args.levels]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no settings match the requested filters")

    configs = method_configs(
        args.checkpoint,
        tabicl_predict_batch_size=args.tabicl_predict_batch_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(args, index_path, configs)
    cached, cache_path = prepare_cache(args, signature)
    output = args.output_dir / "method_results.csv"
    temporary = output.with_suffix(".tmp.csv")
    statuses: dict[str, int] = {}
    cache_hits = computed = 0
    started = time.perf_counter()
    results_by_key: dict[tuple, dict] = {}
    ordered_keys: list[tuple] = []
    tasks_by_method: dict[str, list[FitTask]] = {method: [] for method in METHODS}

    # Resolve cache hits in the parent, then submit only missing fits. Results are
    # written in the original row/context/method order below, so parallel completion
    # order never changes the final CSV.
    for row in rows:
        for context_size in args.context_sizes:
            for method in METHODS:
                base = {
                    "dimension": row["dimension"],
                    "joint_r2_target": row["joint_r2_target"],
                    "level_id": row["level_id"],
                    "method": method,
                    "context_size": context_size,
                    "status": "error",
                    "error": "",
                    "observed_path": row["observed_path"],
                    "truth_path": row["truth_path"],
                }
                key = cache_key(base)
                ordered_keys.append(key)
                if key in cached:
                    results_by_key[key] = {**cached[key], "cached": "1"}
                    cache_hits += 1
                    continue
                compute = _compute_for_method(method, args)
                tasks_by_method[method].append(
                    FitTask(
                        data_dir=args.data_dir,
                        row=row,
                        method=method,
                        config=configs[method],
                        context_size=context_size,
                        compute=compute,
                        query_batch_size=args.query_batch_size,
                    )
                )

    total_work = len(ordered_keys)
    pending = total_work - cache_hits
    print(
        f"Running {total_work} fits ({cache_hits} cached, {pending} pending); "
        f"CPU workers={args.workers}, device={args.device}",
        flush=True,
    )
    report_every = max(1, pending // 20) if pending else 1
    completed = 0
    with cache_path.open("a", newline="") as cache_stream, WorkerManager(
        mp_context=multiprocessing.get_context("spawn")
    ) as manager:
        cache_writer = csv.DictWriter(cache_stream, fieldnames=RESULT_FIELDS)
        for method in METHODS:
            tasks = tasks_by_method[method]
            if not tasks:
                continue
            compute = tasks[0].compute
            for key, result in manager.map(_run_fit_task, tasks, compute):
                results_by_key[key] = result
                computed += 1
                completed += 1
                if result["status"] != "error":
                    cache_writer.writerow(result)
                    cache_stream.flush()
                if args.fail_fast and result["status"] == "error":
                    raise RuntimeError(
                        f"{result['method']} failed for {result['observed_path']} "
                        f"at context_size={result['context_size']}: {result['error']}"
                    )
                if completed % report_every == 0 or completed == pending:
                    print(
                        f"[{completed}/{pending}] {method} complete; "
                        f"cached={cache_hits}",
                        flush=True,
                    )

    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for key in ordered_keys:
            result = results_by_key[key]
            statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            writer.writerow(result)
    temporary.replace(output)
    metadata = {
        "settings": len(rows),
        "methods": list(METHODS),
        "method_configs": configs,
        "context_sizes": args.context_sizes,
        "result_status_counts": statuses,
        "cache": {
            "path": str(cache_path),
            "hits": cache_hits,
            "computed": computed,
            "signature": signature,
        },
        "results_sha256": sha256(output),
        "elapsed_seconds": time.perf_counter() - started,
        "target": "Held-out CATE(X)",
        "normalization": (
            "nPEHE = sqrt(mean((predicted_CATE - CATE)^2)) / std(test_CATE)"
        ),
        "proxy_adjustment_caveat": (
            "S-TabICL-WZ-MC64 is marginalized proxy adjustment, not an identified "
            "backdoor estimator under the proximal DAG."
        ),
        "proxy_integration": (
            "A seed-0 sample of 64 aligned global context (W,Z) atoms, shared "
            "across all queries; "
            "valid as an integration law here because (W,Z) is independent of X."
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved {output}; statuses={statuses}")
    return 1 if statuses.get("error", 0) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--context-sizes", type=int, nargs="+", default=[128, 512, 2048, 4096]
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=HERE.parent / "whole_1" / "best_model.ckpt"
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="CPU worker processes (default: min(8, CPU count - 2)); 1 disables pooling.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-process threads for pooled fits (default: 1).",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS threads per worker (default: 1).",
    )
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument(
        "--tabicl-predict-batch-size",
        type=int,
        default=16384,
        help="Expanded query rows per TabICL prediction call (default: 16384).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if any(value < 1 for value in args.context_sizes):
        parser.error("--context-sizes values must be positive")
    if len(set(args.context_sizes)) != len(args.context_sizes):
        parser.error("--context-sizes values must be unique")
    if args.query_batch_size < 1:
        parser.error("--query-batch-size must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.blas_threads < 1:
        parser.error("--blas-threads must be positive")
    if args.tabicl_predict_batch_size < 1:
        parser.error("--tabicl-predict-batch-size must be positive")
    args.context_sizes = sorted(args.context_sizes)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
