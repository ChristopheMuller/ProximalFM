"""Audit linear latent directions in the synthetic prior."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.analysis_prior.config import PRESETS
from analysis.analysis_prior.generate import load_or_generate
from analysis.analysis_prior.plotting import plt, save_figure

# Nested subsets of one table per SCM, counted before splitting by treatment.
ROW_COUNTS = (2_000, 10_000, 20_000)
TREATMENT_ARMS = (0, 1)
TOY_SEED = 0
TOY_STRONG_R2 = 0.95
TOY_FAINT_R2 = 0.001
RESULTS_DIR = Path(__file__).with_name("results")

PRIOR_SPEC = replace(
    PRESETS["general_ite"],
    name="rebuttal_spectral_decay",
    n_datasets=30,
    seq_len=max(ROW_COUNTS),
    min_dim_X=0,
    max_dim_X=0,
    mc_cate=1,
)
MAX_PROXY_CATEGORIES = int(PRIOR_SPEC.resolved_hp()[0]["max_categories"])


def whitened_basis(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal columns and the map from centered input to them."""
    n_rows, n_columns = block.shape
    if n_rows < 2 or n_columns == 0:
        return np.empty((n_rows, 0)), np.empty((n_columns, 0))

    centered = block.astype(np.float64)
    # Shift first so constant columns center to exact zeros.
    centered -= centered[0].copy()
    centered -= centered.mean(axis=0)
    basis, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(centered.dtype).eps * max(centered.shape) * singular_values[0]
    keep = singular_values > tolerance
    return basis[:, keep], directions[keep].T / singular_values[keep]


def linear_r2_spectrum(u: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return the proxy-explained R² values for orthogonal latent directions."""
    u_basis, _ = whitened_basis(u)
    z_basis, _ = whitened_basis(z)
    # The sample-size factors from whitening and cross-covariance cancel.
    singular_values = np.linalg.svd(z_basis.T @ u_basis, compute_uv=False)
    r2 = np.minimum(singular_values**2, 1.0)
    # A narrower proxy basis leaves latent directions with zero explained R².
    return np.pad(r2, (0, u_basis.shape[1] - r2.size))


def rich_proxy_features(z: np.ndarray) -> np.ndarray:
    """Add quadratic terms and indicators for levels seen on fitting rows."""
    z = z.astype(np.float64)
    left, right = np.triu_indices(z.shape[1])
    features = [z, z[:, left] * z[:, right]]

    # Cached records lack categorical metadata. Use the prior audit's
    # distinct-value rule on the fitting rows, where levels are learned.
    z_fit = z[::2]
    if len(z_fit) > MAX_PROXY_CATEGORIES:
        for index, column in enumerate(z.T):
            levels = np.unique(z_fit[:, index])
            if 2 < levels.size <= MAX_PROXY_CATEGORIES:
                features.append(column[:, None] == levels)
    return np.column_stack(features)


def weakest_direction_r2(u: np.ndarray, z: np.ndarray) -> tuple[float, float, float]:
    """Select a direction with raw Z, then score both proxy fits on held-out rows."""
    u_fit, u_test = u[::2], u[1::2]
    z_fit, z_test = z[::2], z[1::2]
    u_basis, u_transform = whitened_basis(u_fit)
    z_basis, z_transform = whitened_basis(z_fit)
    if u_basis.shape[1] == 0:
        return np.nan, np.nan, np.nan

    cross_moment = z_basis.T @ u_basis
    # Include the right null space when the proxy basis is narrower.
    n_proxy, n_latent = cross_moment.shape
    _, _, directions = np.linalg.svd(cross_moment, full_matrices=n_proxy < n_latent)
    direction = directions[-1]
    projected = cross_moment @ direction
    fit_r2 = min(float(projected @ projected), 1.0)

    target = (u_test - u_fit.mean(axis=0, dtype=np.float64)) @ (u_transform @ direction)
    predicted = (z_test - z_fit.mean(axis=0, dtype=np.float64)) @ (
        z_transform @ projected
    )
    total = np.sum((target - target.mean()) ** 2)
    if total == 0:
        return fit_r2, np.nan, np.nan
    held_out_r2 = 1 - np.sum((target - predicted) ** 2) / total

    rich_z = rich_proxy_features(z)
    rich_fit, rich_test = rich_z[::2], rich_z[1::2]
    rich_mean = rich_fit.mean(axis=0)
    rich_weights = np.linalg.lstsq(
        rich_fit - rich_mean, u_basis @ direction, rcond=None
    )[0]
    rich_prediction = (rich_test - rich_mean) @ rich_weights
    rich_r2 = 1 - np.sum((target - rich_prediction) ** 2) / total
    return fit_r2, float(held_out_r2), float(rich_r2)


def sample_size_scores(
    case: str,
    u: np.ndarray,
    z: np.ndarray,
    treatment: np.ndarray,
    population_weakest_r2: float = np.nan,
) -> list[dict[str, str | int | float]]:
    scores = []
    for n_rows in ROW_COUNTS:
        for arm in TREATMENT_ARMS:
            mask = treatment[:n_rows] == arm
            u_arm, z_arm = u[:n_rows][mask], z[:n_rows][mask]
            fit_r2, held_out_r2, rich_r2 = weakest_direction_r2(u_arm, z_arm)
            n_test = len(u_arm) // 2
            scores.append(
                {
                    "case": case,
                    "rows": n_rows,
                    "arm": arm,
                    "n_fit": len(u_arm) - n_test,
                    "n_test": n_test,
                    "raw_fit_r2": fit_r2,
                    "raw_test_r2": held_out_r2,
                    "rich_test_r2": rich_r2,
                    "population_weakest_r2": population_weakest_r2,
                }
            )
    return scores


def run_toy_controls(n_rows: int) -> list[dict[str, str | int | float]]:
    """Check the spectrum against known signals in two paired Gaussian toys."""
    rng = np.random.default_rng(TOY_SEED)
    u = rng.standard_normal((n_rows, 2))
    noise = rng.standard_normal(u.shape)
    # Random assignment gives both arms the same population spectrum.
    treatment = rng.integers(0, 2, size=n_rows)

    scores = []
    for label, weakest_r2 in (("Faint toy", TOY_FAINT_R2), ("Missing toy", 0.0)):
        population = np.array([TOY_STRONG_R2, weakest_r2])
        z = np.sqrt(population) * u + np.sqrt(1 - population) * noise
        expected = ", ".join(f"{value:.6f}" for value in population)
        print(f"\n{label} (population spectrum [{expected}])")
        print(f"{'A':>2} {'Rows':>7}  Full sample spectrum")
        for arm in TREATMENT_ARMS:
            mask = treatment == arm
            r2 = linear_r2_spectrum(u[mask], z[mask])
            observed = ", ".join(f"{value:.6f}" for value in r2)
            print(f"{arm:>2} {mask.sum():>7,}  [{observed}]")
        scores.extend(sample_size_scores(label, u, z, treatment, weakest_r2))
    return scores


def plot_distribution(scores: pd.DataFrame) -> list[Path]:
    """Plot the largest-subset scores with one point per SCM in each row."""
    prior = scores[scores["population_weakest_r2"].isna()]
    n_draws = prior["case"].nunique()
    n_rows = int(scores["rows"].iloc[0])
    proxy_fits = [
        ("rich_test_r2", "Richer Z", "#BA571C"),
        ("raw_test_r2", "Raw Z", "#146D9D"),
    ]
    toys = [("Faint toy", "^", 0.25), ("Missing toy", "s", 0.40)]

    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 3.8), sharex=True, sharey=True, layout="constrained"
    )
    for arm, ax in zip(TREATMENT_ARMS, axes, strict=True):
        arm_prior = prior[prior["arm"] == arm]
        arm_scores = scores[scores["arm"] == arm].set_index("case")
        offsets = np.linspace(-0.15, 0.15, len(arm_prior))
        for row, (column, _, color) in enumerate(proxy_fits):
            ax.scatter(arm_prior[column], row + offsets, s=28, color=color, alpha=0.75)
            for label, marker, offset in toys:
                ax.scatter(
                    arm_scores.at[label, column],
                    row + offset,
                    marker=marker,
                    s=33,
                    facecolor="white",
                    edgecolor="#3F4752",
                    linewidth=1.1,
                    zorder=3,
                )
        ax.axvline(0, color="#575F69", linewidth=0.9, linestyle="--")
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_yticks(
            range(len(proxy_fits)), labels=[label for _, label, _ in proxy_fits]
        )
        ax.set_ylim(-0.28, 1.55)
        ax.set_title(f"Treatment arm $A={arm}$", fontsize=11)
        ax.set_xlabel(r"Held-out $R^2$ (symmetric log scale)", fontsize=10)
        ax.grid(axis="x", color="#E5E8EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(
        f"Held-out recovery of the weakest latent direction\n"
        f"{n_draws} prior draws with {n_rows:,} rows each",
        fontsize=13,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#3F4752",
                label=label,
            )
            for label, marker, _ in toys
        ],
        loc="outside lower center",
        ncols=2,
        frameon=False,
        fontsize=9,
    )
    return save_figure(fig, RESULTS_DIR, "spectral_decay_holdout")


def main() -> None:
    datasets = load_or_generate(PRIOR_SPEC, progress_bar=True)
    scores = []
    print(
        f"{'SCM':>3} {'A':>2} {'Rows':>7} {'d_U':>4} {'d_Z':>4}  Weakest R²  R² spectrum"
    )
    for record in datasets:
        for arm in TREATMENT_ARMS:
            mask = record.A == arm
            u, z = record.U[mask], record.Z[mask]
            r2 = linear_r2_spectrum(u, z)
            weakest = f"{r2[-1]:.6f}" if r2.size else "n/a"
            spectrum = ", ".join(f"{value:.6f}" for value in r2)
            print(
                f"{record.index:>3} {arm:>2} {len(u):>7,} "
                f"{u.shape[1]:>4} {z.shape[1]:>4}  {weakest:>10}  [{spectrum}]"
            )
        scores.extend(
            sample_size_scores(f"SCM {record.index}", record.U, record.Z, record.A)
        )
    scores.extend(run_toy_controls(PRIOR_SPEC.seq_len))

    frame = pd.DataFrame(scores)
    largest = frame[frame["rows"] == PRIOR_SPEC.seq_len]
    figure_paths = plot_distribution(largest)
    scores_path = RESULTS_DIR / "spectral_decay_scores.csv"
    frame.to_csv(scores_path, index=False)

    prior = largest[largest["population_weakest_r2"].isna()]
    print(f"\nMedian held-out R² at {PRIOR_SPEC.seq_len:,} rows per SCM")
    print(prior.groupby("arm")[["raw_test_r2", "rich_test_r2"]].median().to_string())
    print(f"\nWrote {scores_path} and {', '.join(str(path) for path in figure_paths)}")


if __name__ == "__main__":
    main()
