#!/usr/bin/env python3
"""Compare proximal, backdoor, and in-context methods on the proxy-strength sweep.

Usage: python run_ablation.py --data-dir data --output-dir results
Requires NumPy and SciPy. ``--proximalfm-checkpoint`` additionally requires the
project's PyTorch dependencies. Index paths are relative to --data-dir. Each observed
NPZ contains X,W,Z,A,Y,row_id,split (0=train,1=test); truth contains CATE,row_id
and optionally U for --oracle. Fits use observed training rows only. Scores use
held-out CATE(X), averaging effect contrasts over the empirical training proxy
distribution, which is appropriate for the generator's X-independent proxies.

One paired realization per setting produces exploratory curves. The script
reports numerical conditioning and first-stage diagnostics, with no confidence
intervals or inferred optimal proxy-strength range.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy import linalg

METHODS = ("proximal_2sls", "backdoor_ols_x", "backdoor_ols_xwz")
PROXIMALFM = "proximalfm"
ORACLE = "oracle_ols_xu"
ILL_CONDITIONED = 1e8
RESULT_FIELDS = [
    "dimension",
    "joint_r2_target",
    "level_id",
    "method",
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

    ProximalFM consumes one context-then-query sequence.  The sweep has a much
    larger training split than the model was trained to use as a context, so a
    deterministic prefix makes the context size explicit while preserving the
    held-out scoring population.  Ordering context first is required because
    the model infers the split from the length of ``y_train``.
    """
    n_context = int(n_context)
    split = np.asarray(observed["split"]).reshape(-1)
    train = np.flatnonzero(split == 0)
    query = np.flatnonzero(split == 1)
    if n_context < 1:
        raise ValueError("ProximalFM context size must be positive")
    if n_context > len(train):
        raise ValueError(
            f"ProximalFM context size {n_context} exceeds {len(train)} training rows"
        )
    if not len(query):
        raise ValueError("ProximalFM requires at least one held-out query row")
    indices = np.concatenate([train[:n_context], query])
    return {key: np.asarray(value)[indices] for key, value in observed.items()}


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
    output = args.output_dir / "method_results.csv"
    temporary = output.with_suffix(".tmp.csv")
    started, statuses = time.perf_counter(), {}
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            common = {name: row[name] for name in required}
            try:
                observed, truth = load_setting(args.data_dir, row, args.oracle)
                load_error = None
            except (OSError, ValueError, KeyError, IndexError) as error:
                load_error = error
            for method in methods:
                method_started = time.perf_counter()
                result = {**common, "method": method, "status": "error", "error": ""}
                try:
                    if load_error is not None:
                        raise load_error
                    if method == PROXIMALFM:
                        proximal_observed = restrict_context(
                            observed, args.proximalfm_context_size
                        )
                        predicted, fit = fit_proximalfm(
                            proximal_observed,
                            checkpoint=args.proximalfm_checkpoint,
                            device=args.proximalfm_device,
                            query_batch_size=args.proximalfm_query_batch_size,
                            ensemble=args.proximalfm_ensemble,
                        )
                    else:
                        predicted, fit = fit_method(observed, method, truth.get("U"))
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
                statuses[result["status"]] = statuses.get(result["status"], 0) + 1
                writer.writerow(result)
            stream.flush()
            print(
                f"[{number}/{len(rows)}] dimension={row['dimension']} level={row['level_id']}",
                flush=True,
            )
    temporary.replace(output)
    metadata = {
        "settings": len(rows),
        "methods": methods,
        "result_status_counts": statuses,
        "index_sha256": sha256(index_path),
        "results_sha256": sha256(output),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "target": "Held-out CATE(X); nuisance proxies averaged over training rows",
        "normalization": "nPEHE = sqrt(mean((predicted_CATE - CATE)^2)) / std(test_CATE)",
        "conditioning_warning_threshold": ILL_CONDITIONED,
        "proximalfm": (
            None
            if args.proximalfm_checkpoint is None
            else {
                "checkpoint": str(args.proximalfm_checkpoint),
                "device": args.proximalfm_device,
                "context_size": args.proximalfm_context_size,
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
    print(f"Saved {output}; statuses={statuses}")
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
        help="Checkpoint file or directory to include ProximalFM in the comparison.",
    )
    parser.add_argument(
        "--proximalfm-context-size",
        type=int,
        default=2048,
        help="Training-prefix rows used as ProximalFM context (default: 2048).",
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
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.proximalfm_context_size < 1:
        parser.error("--proximalfm-context-size must be positive")
    if args.proximalfm_query_batch_size < 1:
        parser.error("--proximalfm-query-batch-size must be positive")
    if args.proximalfm_ensemble < 1:
        parser.error("--proximalfm-ensemble must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
