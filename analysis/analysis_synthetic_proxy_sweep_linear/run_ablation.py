#!/usr/bin/env python3
"""Compare proximal, backdoor, and in-context methods on the proxy-strength sweep.

Usage: python run_ablation.py --data-dir data --output-dir results
Requires NumPy, SciPy, and the project's PyTorch dependencies for ProximalFM.
Index paths are relative to --data-dir. Each observed
NPZ contains X,W,Z,A,Y,row_id; truth contains CATE,row_id
and optionally U for --oracle. Fits use the first context-size rows and score
against the last --n-query rows, averaging effect contrasts over
the corresponding context proxy distribution, which is appropriate for the
generator's X-independent proxies.

One paired realization per setting produces exploratory curves. The script
reports numerical conditioning and first-stage diagnostics, with no confidence
intervals or inferred optimal proxy-strength range. Successful fits are cached
under the output directory and reused across overlapping context-size runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy
from scipy import linalg
from tqdm import tqdm

METHODS = ("proximal_2sls", "backdoor_ols_x", "backdoor_ols_xwz")
PROXIMALFM = "proximalfm"
ORACLE = "oracle_ols_xu"
BACKDOOR_METHODS = {
    "tabicl": "backdoor_tabicl",
    "causalpfn": "backdoor_causalpfn",
}
ILL_CONDITIONED = 1e8
FIT_CACHE_VERSION = 2
CAUSALPFN_RUNTIME_SENTINEL = "PROXIMALFM_CAUSALPFN_SAFE_RUNTIME"
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
    "fit_rank",
    "fit_columns",
    "condition_number",
    "instrument_rank",
    "instrument_columns",
    "instrument_condition_number",
    "first_stage_partial_r2_min",
    "first_stage_partial_r2_mean",
    "prediction_max_abs",
    "elapsed_seconds",
    "cached",
    "observed_path",
    "truth_path",
]


def multiply(left, right):
    """Explicit reductions avoid macOS Accelerate's spurious matmul warnings."""
    left, right = np.asarray(left), np.asarray(right)
    if right.ndim == 1:
        return np.einsum("...i,i->...", left, right, optimize=False)
    return np.einsum("...i,ij->...j", left, right, optimize=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_identity(path):
    """Identify the checkpoint selected by the estimator without loading it."""
    if path is None:
        return None
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

            def step(candidate):
                match = re.search(r"(\d+)", candidate.name)
                return (int(match.group(1)) if match else -1, candidate.name)

            selected = max(candidates, key=step)
    if not selected.is_file():
        return {"path": str(selected), "missing": True}
    stat = selected.stat()
    return {
        "path": str(selected),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def fit_cache_signature(args, index_path):
    """Inputs that must agree before persisted fits can safely be reused."""
    checksums = args.data_dir / "SHA256SUMS.txt"
    return {
        "version": FIT_CACHE_VERSION,
        "index_sha256": sha256(index_path),
        "data_checksums_sha256": sha256(checksums) if checksums.is_file() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "proximalfm_checkpoint": checkpoint_identity(args.proximalfm_checkpoint),
        "proximalfm_device": args.proximalfm_device,
        "proximalfm_query_batch_size": args.proximalfm_query_batch_size,
        "proximalfm_ensemble": args.proximalfm_ensemble,
        "n_query": args.n_query,
        "backdoor": {
            "methods": sorted(getattr(args, "backdoor_methods", [])),
            "device": getattr(args, "backdoor_device", None),
            "proxy_samples": getattr(args, "backdoor_proxy_samples", None),
            "seed": getattr(args, "backdoor_seed", None),
            "predict_batch_size": getattr(args, "backdoor_predict_batch_size", None),
            "tabicl_estimators": getattr(args, "tabicl_estimators", None),
        },
    }


def fit_cache_key(row):
    """Stable identity of one dataset/method/context fit."""
    return (
        row["observed_path"],
        row["truth_path"],
        str(row["dimension"]),
        str(row["level_id"]),
        row["method"],
        int(row["context_size"]),
    )


def read_cached_results(path):
    if not path.is_file():
        return {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(RESULT_FIELDS) - {"cached"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            return {}
        cached = {}
        for row in reader:
            try:
                if row["status"] != "error":
                    cached[fit_cache_key(row)] = row
            except (KeyError, TypeError, ValueError):
                continue
    return cached


def previous_results_are_compatible(metadata, signature):
    """Allow the pre-cache result file to seed the first cache."""
    prior_signature = (metadata.get("cache") or {}).get("signature")
    if prior_signature is not None:
        if prior_signature == signature:
            return True
        added = set(signature) - set(prior_signature)
        old_methods = set(metadata.get("methods", []))
        if (
            added == {"backdoor"}
            and not old_methods.intersection(BACKDOOR_METHODS.values())
            and all(
                signature.get(key) == value for key, value in prior_signature.items()
            )
        ):
            return True
        return False
    proximal = metadata.get("proximalfm") or {}
    checkpoint = signature["proximalfm_checkpoint"] or {}
    return (
        metadata.get("index_sha256") == signature["index_sha256"]
        and metadata.get("n_query") == signature["n_query"]
        and metadata.get("numpy") == signature["numpy"]
        and metadata.get("scipy") == signature["scipy"]
        and str(Path(proximal.get("checkpoint", "")).expanduser().resolve())
        == checkpoint.get("path")
        and proximal.get("device") == signature["proximalfm_device"]
        and proximal.get("query_batch_size") == signature["proximalfm_query_batch_size"]
        and proximal.get("ensemble") == signature["proximalfm_ensemble"]
    )


def write_cache_files(cache_path, metadata_path, signature, rows):
    """Atomically initialize or replace the cache and its signature."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "cached": row.get("cached", "0")})
    temporary.replace(cache_path)
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    temporary_metadata.write_text(json.dumps({"signature": signature}, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)


def prepare_fit_cache(args, index_path):
    """Load a compatible cache, migrating the last completed result when possible."""
    cache_dir = args.output_dir / ".fit_cache"
    cache_path = cache_dir / "method_results.csv"
    metadata_path = cache_dir / "metadata.json"
    signature = fit_cache_signature(args, index_path)
    cache_valid = False
    if not args.force and cache_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            cache_valid = metadata.get("signature") == signature
        except (OSError, ValueError, TypeError):
            pass
    if cache_valid:
        cached = read_cached_results(cache_path)
        print(f"Loaded {len(cached):,} compatible cached fits from {cache_path}")
        return cached, cache_path, signature

    seed_rows = []
    output = args.output_dir / "method_results.csv"
    run_metadata = args.output_dir / "run_metadata.json"
    if not args.force and output.is_file() and run_metadata.is_file():
        try:
            metadata = json.loads(run_metadata.read_text())
            if previous_results_are_compatible(metadata, signature):
                seed_rows = list(read_cached_results(output).values())
        except (OSError, ValueError, TypeError):
            pass
    write_cache_files(cache_path, metadata_path, signature, seed_rows)
    cached = {fit_cache_key(row): row for row in seed_rows}
    if seed_rows:
        print(f"Seeded fit cache with {len(seed_rows):,} rows from {output}")
    elif args.force:
        print(f"Reset fit cache at {cache_path}")
    else:
        print(f"Created fit cache at {cache_path}")
    return cached, cache_path, signature


def outcome_design(x, a, covariates=None):
    """[1, X, A, A*X, C, A*C], with each optional C column used once."""
    x, a = np.asarray(x, dtype=float), np.asarray(a, dtype=float).reshape(-1)
    parts = [np.ones((len(x), 1)), x, a[:, None], a[:, None] * x]
    if covariates is not None:
        c = np.asarray(covariates, dtype=float)
        parts.extend([c, a[:, None] * c])
    return np.column_stack(parts)


def averaged_effect(beta, x_test, covariate_mean=None):
    """Integrate model/bridge contrasts using training covariate means."""
    d_x = x_test.shape[1]
    value = beta[1 + d_x] + multiply(x_test, beta[2 + d_x : 2 + 2 * d_x])
    if covariate_mean is not None:
        d_c = len(covariate_mean)
        start = 2 + 2 * d_x + d_c
        value = value + multiply(covariate_mean, beta[start : start + d_c])
    return value


def spectrum_details(singular_values, shape):
    singular_values = np.asarray(singular_values)
    maximum = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = np.finfo(float).eps * max(shape) * maximum
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == shape[1] and len(singular_values)
        else float("inf")
    )
    return rank, condition, tolerance


def fit_ols(design, y):
    beta, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    _, condition, _ = spectrum_details(singular, design.shape)
    return beta, {
        "fit_rank": int(rank),
        "fit_columns": design.shape[1],
        "condition_number": condition,
    }


def fit_2sls(design, instruments, y, included_columns):
    """QR/SVD instrument projection, followed by least squares; no normal inverse.

    Only n-by-p factors are formed. The n-by-n projection matrix is never built.
    SVD of the small R factor selects the numerical instrument column space and
    exposes rank deficiency instead of silently assuming invertibility.
    """
    q, r = linalg.qr(instruments, mode="economic", check_finite=False)
    left, singular, _ = linalg.svd(r, full_matrices=False, check_finite=False)
    rank, instrument_condition, _ = spectrum_details(singular, instruments.shape)
    if rank == 0:
        raise ValueError("Instrument matrix has zero numerical rank")
    rotation = left[:, :rank].T
    projected_design = multiply(rotation, multiply(q.T, design))
    projected_y = multiply(rotation, multiply(q.T, y))
    beta, details = fit_ols(projected_design, projected_y)
    details.update(
        instrument_rank=rank,
        instrument_columns=instruments.shape[1],
        instrument_condition_number=instrument_condition,
    )

    # Descriptive partial first-stage R² for W and A*W after removing the
    # included 1,X,A,A*X columns. This is not a weak-IV significance test.
    endogenous = design[:, included_columns:]
    included = design[:, :included_columns]
    included_beta = np.linalg.lstsq(included, endogenous, rcond=None)[0]
    residual_ss = np.sum((endogenous - multiply(included, included_beta)) ** 2, axis=0)
    full_residual_ss = np.maximum(
        np.sum(endogenous**2, axis=0)
        - np.sum(projected_design[:, included_columns:] ** 2, axis=0),
        0.0,
    )
    partial_r2 = np.divide(
        residual_ss - full_residual_ss,
        residual_ss,
        out=np.full_like(residual_ss, np.nan),
        where=residual_ss > 1e-12,
    )
    finite_r2 = partial_r2[np.isfinite(partial_r2)]
    details["first_stage_partial_r2_min"] = (
        float(finite_r2.min()) if len(finite_r2) else ""
    )
    details["first_stage_partial_r2_mean"] = (
        float(finite_r2.mean()) if len(finite_r2) else ""
    )
    return beta, details


def fit_method(observed, method, oracle_u=None):
    """Fit on split=0 and return CATE predictions for split=1 plus diagnostics."""
    split = np.asarray(observed["split"]).reshape(-1)
    if not np.isin(split, [0, 1]).all():
        raise ValueError("split must contain only 0=train and 1=test")
    train, test = split == 0, split == 1
    if min(int(train.sum()), int(test.sum())) == 0:
        raise ValueError("Both training and test rows are required")
    x = np.asarray(observed["X"], float)
    if x.ndim != 2 or len(x) != len(split):
        raise ValueError("X must be a row-aligned matrix")
    x_train, x_test = x[train], x[test]
    a = np.asarray(observed["A"], float).reshape(-1)[train]
    y = np.asarray(observed["Y"], float).reshape(-1)[train]
    if not np.isin(a, [0, 1]).all():
        raise ValueError("Treatment must be binary")
    c_train = None
    if method == "proximal_2sls":
        c_train = np.asarray(observed["W"], float)[train]
        z_train = np.asarray(observed["Z"], float)[train]
    elif method == "backdoor_ols_xwz":
        c_train = np.column_stack([observed["W"][train], observed["Z"][train]])
    elif method == ORACLE:
        if oracle_u is None:
            raise ValueError("Oracle diagnostic requires truth U")
        c_train = np.asarray(oracle_u, float)[train]
    elif method != "backdoor_ols_x":
        raise ValueError(f"Unknown method: {method}")
    design = outcome_design(x_train, a, c_train)
    if (
        not np.isfinite(design).all()
        or not np.isfinite(y).all()
        or not np.isfinite(x_test).all()
    ):
        raise ValueError("Nonfinite observed training data or test X")
    if len(y) <= design.shape[1]:
        raise ValueError("Training rows must exceed fitted coefficients")
    if method == "proximal_2sls":
        instruments = outcome_design(x_train, a, z_train)
        if not np.isfinite(instruments).all():
            raise ValueError("Nonfinite training instruments")
        beta, details = fit_2sls(design, instruments, y, 2 + 2 * x_train.shape[1])
    else:
        beta, details = fit_ols(design, y)
    predictions = averaged_effect(
        beta, x_test, None if c_train is None else c_train.mean(axis=0)
    )
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite fitted CATE predictions")
    deficient = details["fit_rank"] < details["fit_columns"] or details.get(
        "instrument_rank", 0
    ) < details.get("instrument_columns", 0)
    details["status"] = (
        "rank_deficient"
        if deficient
        else "ill_conditioned"
        if details["condition_number"] >= ILL_CONDITIONED
        else "ok"
    )
    details.update(
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        prediction_max_abs=float(np.abs(predictions).max()),
    )
    return predictions, details


def selected_indices(n_rows, n_context, n_query):
    """Use a context prefix and one shared query suffix for every context size."""
    if n_context < 1 or n_query < 1:
        raise ValueError("Context and query sizes must be positive")
    if n_context + n_query > n_rows:
        raise ValueError(
            f"Context size {n_context} + query size {n_query} exceeds "
            f"{n_rows} total rows"
        )
    return np.concatenate([np.arange(n_context), np.arange(n_rows - n_query, n_rows)])


def restrict_context(observed, n_context, n_query):
    """Keep a context prefix and the fixed held-out query suffix.

    A deterministic prefix gives every method exactly the same observations at
    a given context size while preserving the held-out scoring population.
    Ordering context first is also required by ProximalFM, which infers the
    split from the length of ``y_train``.
    """
    indices = selected_indices(len(observed["row_id"]), n_context, n_query)
    result = {key: np.asarray(value)[indices] for key, value in observed.items()}
    result["split"] = np.r_[
        np.zeros(n_context, dtype=np.uint8), np.ones(n_query, dtype=np.uint8)
    ]
    return result


def restrict_aligned(values, n_context, n_query):
    """Apply the same context-prefix/query selection to another row-aligned array."""
    indices = selected_indices(len(values), n_context, n_query)
    return np.asarray(values)[indices]


def fit_proximalfm(
    observed,
    *,
    checkpoint,
    device="cpu",
    query_batch_size=512,
    ensemble=1,
):
    """Score held-out rows with an in-context ProximalFM checkpoint.

    Query rows do not attend to one another and the adapter standardizes from
    the context alone.  They can therefore be predicted in chunks without
    changing the result, keeping a large sweep's peak memory bounded.
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        # The construction notes also support running this file after `cd`ing
        # into the analysis directory, where the repository package root is
        # otherwise absent from Python's import path.
        sys.path.insert(0, repo_root)
    from benchmarks.methods.proximalfm import ProximalFM_Estimator

    query_batch_size = int(query_batch_size)
    if query_batch_size < 1:
        raise ValueError("ProximalFM query batch size must be positive")
    ensemble = int(ensemble)
    if ensemble < 1:
        raise ValueError("ProximalFM ensemble size must be positive")
    split = np.asarray(observed["split"]).reshape(-1)
    train, query = split == 0, split == 1
    if min(int(train.sum()), int(query.sum())) == 0:
        raise ValueError("ProximalFM requires context and held-out query rows")

    estimator = ProximalFM_Estimator(str(checkpoint), device=device, ensemble=ensemble)
    estimator.fit(
        observed["X"][train],
        observed["A"][train],
        observed["Y"][train],
        W=observed["W"][train],
        Z=observed["Z"][train],
    )
    x, w, z = (observed[key][query] for key in ("X", "W", "Z"))
    predictions = np.concatenate(
        [
            estimator.predict(
                x[start : start + query_batch_size],
                W=w[start : start + query_batch_size],
                Z=z[start : start + query_batch_size],
            )
            for start in range(0, len(x), query_batch_size)
        ]
    )
    if not np.isfinite(predictions).all():
        raise ValueError("ProximalFM returned nonfinite CATE predictions")
    return predictions, {
        "status": "ok",
        "n_train": int(train.sum()),
        "n_test": int(query.sum()),
        "prediction_max_abs": float(np.abs(predictions).max()),
    }


def fit_backdoor_foundation_model(
    observed,
    method,
    *,
    device="cpu",
    proxy_samples=16,
    seed=0,
    predict_batch_size=4096,
    tabicl_estimators=1,
):
    """Fit a proxy-adjusted foundation model and integrate out joint (W,Z).

    The same seeded sample of aligned context proxy pairs is used for every
    query. In this sweep W,Z are independent of X, so their unconditional
    empirical law is the appropriate standardization distribution.
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from benchmarks.compute import ComputeConfig
    from benchmarks.methods import build_method

    if method not in BACKDOOR_METHODS.values():
        raise ValueError(f"Unknown backdoor foundation model: {method}")
    nested = {
        "cls": "CausalPFN",
        "use_W": True,
        "use_Z": True,
    }
    if method == BACKDOOR_METHODS["tabicl"]:
        nested = {
            "cls": "S_Learner",
            "base": "tabicl",
            "use_W": True,
            "use_Z": True,
            "tabicl": {
                "kv_cache": True,
                "n_estimators": int(tabicl_estimators),
            },
        }
    config = {
        "name": method,
        "cls": "ProxyMarginalizer",
        "estimator": nested,
        "integration": {
            "method": "empirical",
            "n_points": int(proxy_samples),
            "seed": int(seed),
        },
        "predict_batch_size": int(predict_batch_size),
    }
    _, estimator = build_method(
        config, ComputeConfig(device=device, execution="inline")
    )
    split = np.asarray(observed["split"]).reshape(-1)
    train, query = split == 0, split == 1
    if min(int(train.sum()), int(query.sum())) == 0:
        raise ValueError("Backdoor methods require context and held-out query rows")
    estimator.fit(
        observed["X"][train],
        observed["A"][train],
        observed["Y"][train],
        W=observed["W"][train],
        Z=observed["Z"][train],
    )
    predictions = np.asarray(estimator.predict(observed["X"][query])).reshape(-1)
    if len(predictions) != int(query.sum()) or not np.isfinite(predictions).all():
        raise ValueError("Backdoor foundation model returned invalid CATE predictions")
    return predictions, {
        "status": "ok",
        "n_train": int(train.sum()),
        "n_test": int(query.sum()),
        "prediction_max_abs": float(np.abs(predictions).max()),
    }


def ensure_causalpfn_safe_runtime(backdoor_methods):
    """Restart once with a one-thread FAISS/OpenMP runtime when CausalPFN is used.

    On macOS, CausalPFN's FAISS search can segfault after NumPy/PyTorch have
    initialized their own threaded runtimes. These variables must exist at
    process startup, so setting them in the already-running interpreter is too
    late. TabICL does not need this path.
    """
    if (
        "causalpfn" not in backdoor_methods
        or os.environ.get(CAUSALPFN_RUNTIME_SENTINEL) == "1"
    ):
        return
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    environment["KMP_DUPLICATE_OK"] = "True"
    environment["KMP_DUPLICATE_LIB_OK"] = "True"
    environment[CAUSALPFN_RUNTIME_SENTINEL] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def score_cate(predictions, true_cate):
    true_cate = np.asarray(true_cate, float).reshape(-1)
    if predictions.shape != true_cate.shape or not np.isfinite(true_cate).all():
        raise ValueError("True CATE must be finite and aligned with predictions")
    mse = float(np.mean((predictions - true_cate) ** 2))
    pehe, scale = float(np.sqrt(mse)), float(np.std(true_cate))
    return {
        "pehe": pehe,
        "npehe": pehe / scale if scale > 1e-12 else "",
        "mse_cate": mse,
        "ate_bias": float(np.mean(predictions - true_cate)),
        "predicted_ate": float(predictions.mean()),
        "true_test_ate": float(true_cate.mean()),
        "true_cate_sd": scale,
    }


def load_setting(data_dir, row, oracle=False):
    observed_path = Path(row["observed_path"])
    truth_path = Path(row["truth_path"])
    if not observed_path.is_absolute():
        observed_path = data_dir / observed_path
    if not truth_path.is_absolute():
        truth_path = data_dir / truth_path
    with np.load(observed_path, allow_pickle=False) as data:
        observed = {key: data[key] for key in ["X", "W", "Z", "A", "Y", "row_id"]}
    with np.load(truth_path, allow_pickle=False) as data:
        truth = {
            key: data[key]
            for key in (["CATE", "row_id", "U"] if oracle else ["CATE", "row_id"])
        }
    ids = np.asarray(observed["row_id"]).reshape(-1)
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, truth["row_id"]):
        raise ValueError("Observed/truth row IDs must be aligned and unique")
    if any(len(value) != len(ids) for value in observed.values()) or len(
        truth["CATE"]
    ) != len(ids):
        raise ValueError("Dataset arrays have inconsistent row counts")
    if observed["X"].shape[1] != int(row["dimension"]):
        raise ValueError("Dimension differs from index.csv")
    if row.get("n_rows") and int(row["n_rows"]) != len(ids):
        raise ValueError("Total row count differs from index.csv")
    return observed, truth


def _worker_init():
    """Keep CPU model and BLAS threads from multiplying across processes."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)
    import torch

    torch.set_num_threads(1)


def _compute_setting(args, row, methods, cached_results):
    required = (
        "dimension",
        "joint_r2_target",
        "level_id",
        "observed_path",
        "truth_path",
    )
    common = {name: row[name] for name in required}
    requested = [
        (context_size, method)
        for context_size in args.context_sizes
        for method in methods
    ]
    if all(
        fit_cache_key({**common, "context_size": context_size, "method": method})
        in cached_results
        for context_size, method in requested
    ):
        return [
            {
                **cached_results[
                    fit_cache_key(
                        {**common, "context_size": context_size, "method": method}
                    )
                ],
                "cached": "1",
            }
            for context_size, method in requested
        ]
    try:
        observed, truth = load_setting(args.data_dir, row, args.oracle)
        load_error = None
    except (OSError, ValueError, KeyError, IndexError) as error:
        load_error = error
    results = []
    for context_size in args.context_sizes:
        context_observed = context_u = None
        context_error = load_error
        context_prepared = False
        for method in methods:
            method_started = time.perf_counter()
            result = {
                **common,
                "method": method,
                "context_size": context_size,
                "status": "error",
                "error": "",
            }
            key = fit_cache_key(result)
            if key in cached_results:
                results.append({**cached_results[key], "cached": "1"})
                continue
            if not context_prepared:
                context_prepared = True
                if context_error is None:
                    try:
                        context_observed = restrict_context(
                            observed, context_size, args.n_query
                        )
                        context_u = (
                            None
                            if "U" not in truth
                            else restrict_aligned(
                                truth["U"], context_size, args.n_query
                            )
                        )
                    except (OSError, ValueError, KeyError, IndexError) as error:
                        context_error = error
            try:
                if context_error is not None:
                    raise context_error
                if method == PROXIMALFM:
                    predicted, fit = fit_proximalfm(
                        context_observed,
                        checkpoint=args.proximalfm_checkpoint,
                        device=args.proximalfm_device,
                        query_batch_size=args.proximalfm_query_batch_size,
                        ensemble=args.proximalfm_ensemble,
                    )
                elif method in BACKDOOR_METHODS.values():
                    predicted, fit = fit_backdoor_foundation_model(
                        context_observed,
                        method,
                        device=args.backdoor_device,
                        proxy_samples=args.backdoor_proxy_samples,
                        seed=args.backdoor_seed,
                        predict_batch_size=args.backdoor_predict_batch_size,
                        tabicl_estimators=args.tabicl_estimators,
                    )
                else:
                    predicted, fit = fit_method(context_observed, method, context_u)
                result.update(fit)
                result.update(score_cate(predicted, truth["CATE"][-args.n_query :]))
            except (
                OSError,
                ValueError,
                KeyError,
                IndexError,
                np.linalg.LinAlgError,
                RuntimeError,
                ImportError,
            ) as error:
                result.update(status="error", error=f"{type(error).__name__}: {error}")
                if args.fail_fast:
                    raise
            result["elapsed_seconds"] = time.perf_counter() - method_started
            result["cached"] = "0"
            results.append(result)
    return results


def _compute_setting_star(task):
    return _compute_setting(*task)


def run(args):
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
        raise ValueError(f"index.csv must contain settings with {sorted(required)}")
    if args.dimensions:
        rows = [row for row in rows if int(row["dimension"]) in args.dimensions]
    if args.levels:
        rows = [row for row in rows if row["level_id"] in args.levels]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No settings match the filters")
    missing_sizes = [row for row in rows if not row.get("n_rows")]
    if missing_sizes:
        raise ValueError("index.csv must contain n_rows for every dataset")
    smallest_total = min(int(row["n_rows"]) for row in rows)
    if max(args.context_sizes) + args.n_query > smallest_total:
        raise ValueError(
            f"Largest context size {max(args.context_sizes)} + query size "
            f"{args.n_query} exceeds {smallest_total} total rows"
        )
    methods = list(METHODS)
    methods.extend(
        BACKDOOR_METHODS[name] for name in getattr(args, "backdoor_methods", [])
    )
    if args.proximalfm_checkpoint is not None:
        methods.append(PROXIMALFM)
    if args.oracle:
        methods.append(ORACLE)
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and (
        (PROXIMALFM in methods and not args.proximalfm_device.startswith("cpu"))
        or (
            any(method in BACKDOOR_METHODS.values() for method in methods)
            and not args.backdoor_device.startswith("cpu")
        )
    ):
        raise ValueError("--workers > 1 requires CPU devices for all selected models")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached_results, cache_path, cache_signature = prepare_fit_cache(args, index_path)
    output = args.output_dir / "method_results.csv"
    temporary = output.with_suffix(".tmp.csv")
    started, statuses = time.perf_counter(), {}
    cache_hits = computed_fits = 0

    def tasks():
        for row in rows:
            relevant_cache = {}
            for context_size in args.context_sizes:
                for method in methods:
                    key = fit_cache_key(
                        {**row, "method": method, "context_size": context_size}
                    )
                    if key in cached_results:
                        relevant_cache[key] = cached_results[key]
            yield args, row, methods, relevant_cache

    with temporary.open("w", newline="") as stream, cache_path.open(
        "a", newline=""
    ) as cache_stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        cache_writer = csv.DictWriter(cache_stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        if workers == 1:
            results = (
                (number, _compute_setting(*task)) for number, task in enumerate(tasks())
            )
        else:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
            )
            pending = {
                pool.submit(_compute_setting_star, task): number
                for number, task in enumerate(tasks())
            }
            results = (
                (pending[future], future.result()) for future in as_completed(pending)
            )
        try:
            waiting = {}
            next_to_write = 0
            with tqdm(
                results, total=len(rows), unit="setting", desc="Sweep"
            ) as progress:
                for number, setting_results in progress:
                    for result in setting_results:
                        statuses[result["status"]] = (
                            statuses.get(result["status"], 0) + 1
                        )
                        if result["cached"] == "1":
                            cache_hits += 1
                        else:
                            computed_fits += 1
                            if result["status"] != "error":
                                cache_writer.writerow(result)
                    waiting[number] = setting_results
                    while next_to_write in waiting:
                        writer.writerows(waiting.pop(next_to_write))
                        next_to_write += 1
                    stream.flush()
                    cache_stream.flush()
                    progress.set_postfix(cached=cache_hits, computed=computed_fits)
        finally:
            if workers > 1:
                pool.shutdown(cancel_futures=True)
    temporary.replace(output)
    metadata = {
        "settings": len(rows),
        "methods": methods,
        "context_sizes": args.context_sizes,
        "n_query": args.n_query,
        "workers": workers,
        "result_status_counts": statuses,
        "cache": {
            "path": str(cache_path),
            "hits": cache_hits,
            "computed": computed_fits,
            "signature": cache_signature,
        },
        "index_sha256": sha256(index_path),
        "results_sha256": sha256(output),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "target": "Held-out CATE(X); nuisance proxies averaged over context rows",
        "normalization": "nPEHE = sqrt(mean((predicted_CATE - CATE)^2)) / std(test_CATE)",
        "conditioning_warning_threshold": ILL_CONDITIONED,
        "proximalfm": (
            None
            if args.proximalfm_checkpoint is None
            else {
                "checkpoint": str(args.proximalfm_checkpoint),
                "device": args.proximalfm_device,
                "context_sizes": args.context_sizes,
                "n_query": args.n_query,
                "query_batch_size": args.proximalfm_query_batch_size,
                "ensemble": args.proximalfm_ensemble,
                "query_proxies": "masked by ProximalFM after context-only standardization",
            }
        ),
        "backdoor_foundation_models": {
            "methods": getattr(args, "backdoor_methods", []),
            "device": getattr(args, "backdoor_device", None),
            "proxy_integration": "seeded empirical sample of aligned context (W,Z) rows",
            "proxy_samples": getattr(args, "backdoor_proxy_samples", None),
            "seed": getattr(args, "backdoor_seed", None),
            "predict_batch_size": getattr(args, "backdoor_predict_batch_size", None),
            "tabicl_estimators": getattr(args, "tabicl_estimators", None),
            "causalpfn_safe_runtime": "causalpfn"
            not in getattr(args, "backdoor_methods", [])
            or os.environ.get(CAUSALPFN_RUNTIME_SENTINEL) == "1",
        },
        "interpretation": "Exploratory curves from one paired realization per setting; no confidence intervals or claimed optimal range. First-stage partial R2 is descriptive, not a weak-IV significance test. Oracle results, when requested, use hidden U and are diagnostic.",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"Saved {output}; statuses={statuses}; "
        f"cache hits={cache_hits}, computed={computed_fits}"
    )
    return 1 if statuses.get("error", 0) else 0


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--output-dir", type=Path, default=here / "results")
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--levels", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel dataset workers (default: 1; multiple workers require CPU models).",
    )
    parser.add_argument(
        "--proximalfm-checkpoint",
        type=Path,
        default=here.parents[1] / "whole_1" / "best_model.ckpt",
        help=(
            "ProximalFM checkpoint file or directory "
            "(default: whole_1/best_model.ckpt at the repository root)."
        ),
    )
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048],
        help=(
            "Training-prefix sizes used for every method "
            "(default: 128 512 1024 2048)."
        ),
    )
    parser.add_argument(
        "--n-query",
        type=int,
        required=True,
        help="Number of shared held-out query rows per dataset.",
    )
    parser.add_argument(
        "--proximalfm-query-batch-size",
        type=int,
        default=512,
        help="Held-out queries per ProximalFM forward pass (default: 512).",
    )
    parser.add_argument(
        "--proximalfm-device",
        default="cpu",
        help="ProximalFM torch device (default: cpu).",
    )
    parser.add_argument(
        "--proximalfm-ensemble",
        type=int,
        default=1,
        help="ProximalFM feature-permutation ensemble size (default: 1).",
    )
    parser.add_argument(
        "--backdoor-methods",
        nargs="*",
        choices=sorted(BACKDOOR_METHODS),
        default=["tabicl"],
        help=(
            "Proxy-adjusted foundation models to include; pass the flag with no "
            "values to disable them (default: tabicl; CausalPFN is opt-in)."
        ),
    )
    parser.add_argument(
        "--backdoor-device",
        help="Torch device for backdoor foundation models (default: ProximalFM device).",
    )
    parser.add_argument(
        "--backdoor-proxy-samples",
        type=int,
        default=16,
        help="Aligned context (W,Z) pairs used for empirical integration (default: 16).",
    )
    parser.add_argument(
        "--backdoor-seed",
        type=int,
        default=0,
        help="Seed selecting shared empirical (W,Z) pairs (default: 0).",
    )
    parser.add_argument(
        "--backdoor-predict-batch-size",
        type=int,
        default=4096,
        help="Expanded query rows per prediction chunk (default: 4096).",
    )
    parser.add_argument(
        "--tabicl-estimators",
        type=int,
        default=1,
        help="TabICL ensemble views (default: 1).",
    )
    parser.add_argument(
        "--oracle", action="store_true", help="Include oracle X,U OLS as a diagnostic"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear the compatible fit cache and recompute every requested fit.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.context_sizes or any(size < 1 for size in args.context_sizes):
        parser.error("--context-sizes values must be positive")
    if len(set(args.context_sizes)) != len(args.context_sizes):
        parser.error("--context-sizes values must be unique")
    if args.n_query < 1:
        parser.error("--n-query must be positive")
    args.context_sizes = sorted(args.context_sizes)
    if args.proximalfm_query_batch_size < 1:
        parser.error("--proximalfm-query-batch-size must be positive")
    if args.proximalfm_ensemble < 1:
        parser.error("--proximalfm-ensemble must be positive")
    if args.backdoor_device is None:
        args.backdoor_device = args.proximalfm_device
    if args.backdoor_proxy_samples < 1:
        parser.error("--backdoor-proxy-samples must be positive")
    if args.backdoor_seed < 0:
        parser.error("--backdoor-seed must be nonnegative")
    if args.backdoor_predict_batch_size < 1:
        parser.error("--backdoor-predict-batch-size must be positive")
    if args.tabicl_estimators < 1:
        parser.error("--tabicl-estimators must be positive")
    ensure_causalpfn_safe_runtime(args.backdoor_methods)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
