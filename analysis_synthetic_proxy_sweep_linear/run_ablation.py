#!/usr/bin/env python3
"""Compare proximal, backdoor, and in-context methods on the proxy-strength sweep.

Usage: python run_ablation.py --data-dir data --output-dir results
Requires NumPy, SciPy, and the project's PyTorch dependencies for ProximalFM.
Index paths are relative to --data-dir. Each observed
NPZ contains X,W,Z,A,Y,row_id,split (0=train,1=test); truth contains CATE,row_id
and optionally U for --oracle. Fits use observed training rows only. Scores use
held-out CATE(X) at every requested context size, averaging effect contrasts over
the corresponding context proxy distribution, which is appropriate for the
generator's X-independent proxies.

One paired realization per setting produces exploratory curves. The script
reports numerical conditioning and first-stage diagnostics, with no confidence
intervals or inferred optimal proxy-strength range. Successful fits are cached
under the output directory and reused across overlapping context-size runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import scipy
from scipy import linalg

METHODS = ("proximal_2sls", "backdoor_ols_x", "backdoor_ols_xwz")
PROXIMALFM = "proximalfm"
ORACLE = "oracle_ols_xu"
ILL_CONDITIONED = 1e8
FIT_CACHE_VERSION = 1
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
        return prior_signature == signature
    proximal = metadata.get("proximalfm") or {}
    checkpoint = signature["proximalfm_checkpoint"] or {}
    return (
        metadata.get("index_sha256") == signature["index_sha256"]
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


def restrict_context(observed, n_context):
    """Keep a training prefix as context and retain every held-out query row.

    A deterministic prefix gives every method exactly the same observations at
    a given context size while preserving the held-out scoring population.
    Ordering context first is also required by ProximalFM, which infers the
    split from the length of ``y_train``.
    """
    n_context = int(n_context)
    split = np.asarray(observed["split"]).reshape(-1)
    train = np.flatnonzero(split == 0)
    query = np.flatnonzero(split == 1)
    if n_context < 1:
        raise ValueError("Context size must be positive")
    if n_context > len(train):
        raise ValueError(f"Context size {n_context} exceeds {len(train)} training rows")
    if not len(query):
        raise ValueError("At least one held-out query row is required")
    indices = np.concatenate([train[:n_context], query])
    return {key: np.asarray(value)[indices] for key, value in observed.items()}


def restrict_aligned(values, split, n_context):
    """Apply the same context-prefix/query selection to another row-aligned array."""
    split = np.asarray(split).reshape(-1)
    train = np.flatnonzero(split == 0)
    query = np.flatnonzero(split == 1)
    indices = np.concatenate([train[: int(n_context)], query])
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
    repo_root = str(Path(__file__).resolve().parents[1])
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
        observed = {
            key: data[key] for key in ["X", "W", "Z", "A", "Y", "row_id", "split"]
        }
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
    if row.get("n_train") and int(row["n_train"]) != int(
        (observed["split"] == 0).sum()
    ):
        raise ValueError("Training row count differs from index.csv")
    return observed, truth


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
    methods = list(METHODS)
    if args.proximalfm_checkpoint is not None:
        methods.append(PROXIMALFM)
    if args.oracle:
        methods.append(ORACLE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached_results, cache_path, cache_signature = prepare_fit_cache(args, index_path)
    output = args.output_dir / "method_results.csv"
    temporary = output.with_suffix(".tmp.csv")
    started, statuses = time.perf_counter(), {}
    cache_hits = computed_fits = 0
    with temporary.open("w", newline="") as stream, cache_path.open(
        "a", newline=""
    ) as cache_stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        cache_writer = csv.DictWriter(cache_stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            common = {name: row[name] for name in required}
            try:
                observed, truth = load_setting(args.data_dir, row, args.oracle)
                load_error = None
            except (OSError, ValueError, KeyError, IndexError) as error:
                load_error = error
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
                        result = {**cached_results[key], "cached": "1"}
                        cache_hits += 1
                        statuses[result["status"]] = (
                            statuses.get(result["status"], 0) + 1
                        )
                        writer.writerow(result)
                        continue
                    if not context_prepared:
                        context_prepared = True
                        if context_error is None:
                            try:
                                context_observed = restrict_context(
                                    observed, context_size
                                )
                                context_u = (
                                    None
                                    if "U" not in truth
                                    else restrict_aligned(
                                        truth["U"], observed["split"], context_size
                                    )
                                )
                            except (
                                OSError,
                                ValueError,
                                KeyError,
                                IndexError,
                            ) as error:
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
                        else:
                            predicted, fit = fit_method(
                                context_observed, method, context_u
                            )
                        result.update(fit)
                        result.update(
                            score_cate(predicted, truth["CATE"][observed["split"] == 1])
                        )
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        IndexError,
                        np.linalg.LinAlgError,
                        RuntimeError,
                    ) as error:
                        result.update(
                            status="error", error=f"{type(error).__name__}: {error}"
                        )
                        if args.fail_fast:
                            raise
                    result["elapsed_seconds"] = time.perf_counter() - method_started
                    result["cached"] = "0"
                    computed_fits += 1
                    statuses[result["status"]] = statuses.get(result["status"], 0) + 1
                    writer.writerow(result)
                    if result["status"] != "error":
                        cache_writer.writerow(result)
                        cache_stream.flush()
            stream.flush()
            print(
                f"[{number}/{len(rows)}] dimension={row['dimension']} "
                f"level={row['level_id']} (cached={cache_hits}, computed={computed_fits})",
                flush=True,
            )
    temporary.replace(output)
    metadata = {
        "settings": len(rows),
        "methods": methods,
        "context_sizes": args.context_sizes,
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
                "query_batch_size": args.proximalfm_query_batch_size,
                "ensemble": args.proximalfm_ensemble,
                "query_proxies": "masked by ProximalFM after context-only standardization",
            }
        ),
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
        "--proximalfm-checkpoint",
        type=Path,
        default=here.parent / "whole_1" / "best_model.ckpt",
        help=(
            "ProximalFM checkpoint file or directory "
            "(default: ../whole_1/best_model.ckpt)."
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
    if any(size < 1 for size in args.context_sizes):
        parser.error("--context-sizes values must be positive")
    if len(set(args.context_sizes)) != len(args.context_sizes):
        parser.error("--context-sizes values must be unique")
    args.context_sizes = sorted(args.context_sizes)
    if args.proximalfm_query_batch_size < 1:
        parser.error("--proximalfm-query-batch-size must be positive")
    if args.proximalfm_ensemble < 1:
        parser.error("--proximalfm-ensemble must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
