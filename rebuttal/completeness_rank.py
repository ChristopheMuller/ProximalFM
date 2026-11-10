"""Test rank of binned P(U | Z, A) in the no-X scalar prior and toy controls."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import beta

from analysis.analysis_prior.config import PRESETS, PriorSpec
from analysis.analysis_prior.generate import generate

BINS = (2, 4, 8)
ALPHA = 0.05
PILOT_FRACTION = 0.2
TOY_SEED = 1925
RESULTS_PATH = Path(__file__).with_name("results") / "completeness_rank.csv"


def scalar_prior_spec(n_scms: int, rows: int) -> PriorSpec:
    """Use the usual MLP mechanisms in the no-X, scalar-proxy case."""
    return replace(
        PRESETS["general_ite"],
        name="rebuttal_completeness_rank",
        n_datasets=n_scms,
        seq_len=rows,
        min_dim_X=0,
        max_dim_X=0,
        max_dim_U=1,
        max_dim_W=1,
        max_dim_Z=1,
    )


def quantile_cells(
    u_pilot: np.ndarray,
    z_pilot: np.ndarray,
    u_eval: np.ndarray,
    z_eval: np.ndarray,
    bins: int,
) -> np.ndarray:
    """Assign evaluation rows to cells using the pilot quantiles."""
    quantiles = np.arange(1, bins) / bins
    u_bins = np.searchsorted(np.quantile(u_pilot, quantiles), u_eval)
    z_bins = np.searchsorted(np.quantile(z_pilot, quantiles), z_eval)
    return z_bins * bins + u_bins


def rank_test(counts: np.ndarray, alpha: float) -> tuple[float, float]:
    """Return the smallest singular value and a simultaneous error bound.

    Rows are Z bins and columns are U bins. Condition on the observed Z-bin
    totals. Each cell is then binomial, and Bonferroni works even though cells
    within a row are dependent.
    """
    row_totals = counts.sum(axis=1, keepdims=True)
    probabilities = counts / row_totals
    tail = alpha / (2 * counts.size)
    lower = np.where(counts == 0, 0.0, beta.ppf(tail, counts, row_totals - counts + 1))
    upper = np.where(
        counts == row_totals, 1.0, beta.isf(tail, counts + 1, row_totals - counts)
    )
    cell_error = np.maximum(probabilities - lower, upper - probabilities)
    # The Frobenius norm bounds the operator norm, which bounds the change in
    # any singular value. Under rank deficiency the true smallest one is zero.
    error_bound = np.linalg.norm(cell_error)
    sigma_min = np.linalg.svd(probabilities, compute_uv=False)[-1]
    return float(sigma_min), float(error_bound)


def evaluate_case(
    case: str, seed: int, u: np.ndarray, z: np.ndarray, treatment: np.ndarray
) -> list[dict]:
    """Use a single pilot/evaluation split for all resolutions and both arms."""
    split = int(PILOT_FRACTION * len(treatment))
    results = []
    for arm in (0, 1):
        pilot = treatment[:split] == arm
        evaluation = treatment[split:] == arm
        u_pilot, z_pilot = u[:split][pilot], z[:split][pilot]
        u_eval, z_eval = u[split:][evaluation], z[split:][evaluation]

        for bins in BINS:
            result = {
                "case": case,
                "seed": seed,
                "arm": arm,
                "bins": bins,
                "n_pilot": len(u_pilot),
                "n_eval": len(u_eval),
                "min_row_count": None,
                "sigma_min": None,
                "error_bound": None,
                "lower_bound": None,
                "reject": None,
                "status": "empty_arm",
                "counts_z_by_u": None,
            }
            results.append(result)
            if not len(u_pilot) or not len(u_eval):
                continue

            cells = quantile_cells(u_pilot, z_pilot, u_eval, z_eval, bins)
            counts = np.bincount(cells, minlength=bins**2).reshape(bins, bins)
            result["counts_z_by_u"] = json.dumps(counts.tolist())
            result["min_row_count"] = int(counts.sum(axis=1).min())
            if not counts.sum(axis=1).all() or not counts.sum(axis=0).all():
                result["status"] = "empty_bin"
                continue

            sigma_min, error_bound = rank_test(counts, ALPHA)
            result.update(
                sigma_min=sigma_min,
                error_bound=error_bound,
                lower_bound=max(0.0, sigma_min - error_bound),
                reject=sigma_min > error_bound,
                status="tested",
            )
    return results


def toy_controls(rows: int) -> list[dict]:
    rng = np.random.default_rng(TOY_SEED)
    u = rng.standard_normal(rows)
    noise = rng.standard_normal(rows)
    assignment = rng.random(rows)
    treatment = assignment < 0.1 + 0.8 * expit(0.8 * u)
    results = []
    for case, z in (
        ("Gaussian strong", u + 0.2 * noise),
        ("Gaussian weak", 0.2 * u + noise),
        ("Independent", noise),
    ):
        results.extend(evaluate_case(case, TOY_SEED, u, z, treatment))

    # U and Z are independent conditional on the binary mixture component.
    # A depends only on U, so P(U | Z, A) still factors through two components.
    component_mean = rng.choice([-1.5, 1.5], size=rows)
    u = component_mean + rng.standard_normal(rows)
    z = component_mean + rng.standard_normal(rows)
    treatment = assignment < 0.1 + 0.8 * expit(0.8 * u)
    results.extend(evaluate_case("Rank-two mixture", TOY_SEED, u, z, treatment))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scms", type=int, default=12)
    parser.add_argument("--rows", type=int, default=200_000, help="Rows per prior SCM")
    parser.add_argument(
        "--toy-rows", type=int, default=1_000_000, help="Rows per control"
    )
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    if args.n_scms < 1 or min(args.rows, args.toy_rows) < 5:
        parser.error("Use at least one SCM and five rows per dataset")

    spec = scalar_prior_spec(args.n_scms, args.rows)
    results = toy_controls(args.toy_rows)
    for record in generate(spec, progress_bar=True):
        # The prior may deliver an empty U block after removing constants.
        u = record.U[:, 0] if record.U.shape[1] else np.zeros(record.n_rows)
        results.extend(
            evaluate_case(
                f"MLP {record.index}", record.seed, u, record.Z[:, 0], record.A
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
