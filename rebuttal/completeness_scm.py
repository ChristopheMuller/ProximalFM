# %%
# IMPORTS
####

import matplotlib.pyplot as plt
import numpy as np

from analysis.analysis_prior.config import PriorSpec
from analysis.analysis_prior.generate import generate

# %%
# CONFIG
####

dim_X = 0
dim_U = 1
dim_W = 1
dim_Z = 1
n_samples = 50_000  # total across both treatment arms
n_scms = 100
k = 4  # quantile bins per column
seed = 0


# %%
# FUNCTIONS
####


def discretize(block, k):
    """Use distinct values when possible, then encode the occupied joint bins."""
    bins = np.empty(block.shape, dtype=np.intp)
    quantiles = np.arange(1, k) / k
    for index, column in enumerate(block.T):
        values, codes = np.unique(column, return_inverse=True)
        if len(values) > k:
            codes = np.searchsorted(np.quantile(column, quantiles), column)
        bins[:, index] = codes
    _, codes = np.unique(bins, axis=0, return_inverse=True)
    return codes.ravel()


def probability_matrix(rows, cols):
    """Estimate P(column category | row category) by counting rows."""
    n_rows, n_cols = rows.max() + 1, cols.max() + 1
    counts = np.bincount(rows * n_cols + cols, minlength=n_rows * n_cols)
    counts = counts.reshape(n_rows, n_cols)
    return counts / counts.sum(axis=1, keepdims=True)


# %%
# RUN
####


def run():
    """Estimate P(U | Z, A=a) and P(W | U, A=a) using bins fitted in each arm."""
    if dim_X != 0:
        raise ValueError("This experiment requires dim_X = 0")
    if min(dim_W, dim_Z) < dim_U:
        raise ValueError("The prior requires dim_W and dim_Z to be at least dim_U")
    if k < 2:
        raise ValueError("k must be at least 2")

    spec = PriorSpec(
        name="rebuttal_completeness_scm",
        n_datasets=n_scms,
        seq_len=n_samples,
        mc_cate=1,
        seed=seed,
        min_dim_X=dim_X,
        max_dim_X=dim_X,
        min_dim_U=dim_U,
        max_dim_U=dim_U,
        min_dim_W=dim_W,
        max_dim_W=dim_W,
        min_dim_Z=dim_Z,
        max_dim_Z=dim_Z,
    )

    results = []
    for index in range(n_scms):
        sample = generate(spec, indices=[index], progress=False)[0]

        if (sample.U.shape[1], sample.W.shape[1], sample.Z.shape[1]) != (
            dim_U,
            dim_W,
            dim_Z,
        ):
            print(f"SCM {index}: skipped after constant-column filtering")
            continue

        for arm in (0, 1):
            mask = sample.A == arm
            if not mask.any():
                print(f"SCM {index}, A={arm}: no rows")
                continue

            u = discretize(sample.U[mask], k)
            w = discretize(sample.W[mask], k)
            z = discretize(sample.Z[mask], k)
            p_u_given_z = probability_matrix(z, u)
            p_w_given_u = probability_matrix(u, w)
            sv_z = np.linalg.svd(p_u_given_z, compute_uv=False)
            sv_w = np.linalg.svd(p_w_given_u, compute_uv=False)
            results.append(
                {
                    "scm": index,
                    "arm": arm,
                    "n_rows": int(mask.sum()),
                    "p_u_given_z": p_u_given_z,
                    "p_w_given_u": p_w_given_u,
                    "sv_z": sv_z,
                    "sv_w": sv_w,
                }
            )

            print(
                f"SCM {index}, A={arm}: min singular values Z={sv_z[-1]:.3g}, W={sv_w[-1]:.3g}"
            )

    return results


# %%
# RUN THE EXPERIMENT
#####

results = run()

# %%
# PLOT THE RESULTS
#####


def plot_results(results, n_scms):
    fig, axes = plt.subplots(
        2, 2, figsize=(8, 6), sharex=True, sharey=True, layout="constrained"
    )
    for proxy_axes, field, label in zip(
        axes, ("sv_z", "sv_w"), (r"U\mid Z", r"W\mid U"), strict=True
    ):
        for arm, ax in enumerate(proxy_axes):
            smallest = [
                row[field][-1]
                for row in results
                if row["arm"] == arm and row[field] is not None
            ]
            if smallest:
                ax.ecdf(smallest, compress=True)
            ax.set_title(f"$P({label}, A={arm})$\n{len(smallest)}/{n_scms} SCMs")
            ax.grid(alpha=0.25)

    axes[0, 0].set_xscale("symlog", linthresh=1e-8)
    axes[0, 0].set_xlim(left=0)
    axes[0, 0].set_ylim(0, 1.02)
    fig.supylabel("Cumulative fraction of plotted SCMs")
    fig.supxlabel("Smallest singular value")
    return fig


fig = plot_results(results, n_scms)
plt.show()
