# %%
"""
Visualize the proximal SCM prior after adding Monte Carlo CATE outputs.

Expected batch output order from PriorDataset.get_batch():
    U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, MU0, MU1,
    d_U, d_X, d_W, d_Z, seq_lens, train_sizes

Definitions used throughout this file:
    ITE  = Y1 - Y0          # one sampled latent-world treatment effect
    CATE = MU1 - MU0        # Monte Carlo estimate of E[Y(1)-Y(0) | X]
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

from proximalfm.prior import PriorDataset
from proximalfm.prior._prior_config import DEFAULT_FIXED_HP


# %% ---------------------------------------------------------------------
# 1. Instantiate Prior Dataset
# ----------------------------------------------------------------------

n_batches = 2
batch_size = 3
mc_cate = 200

# If your generator is already unconditional CATE-mode, this config is harmless.
# If your generator checks params["prior_cate"], this turns the new mode on.
scm_fixed_hp = {
    **DEFAULT_FIXED_HP,
    "prior_cate": True,
}

dataset = PriorDataset(
    batch_size=batch_size,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=2,
    min_dim_W=1,
    max_dim_W=2,
    min_dim_Z=1,
    max_dim_Z=2,
    min_dim_X=1,
    max_dim_X=1,
    min_seq_len=None,
    max_seq_len=100,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.5,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    mc_cate=mc_cate,
    scm_fixed_hp=scm_fixed_hp,
    device="cpu",
)


# %% ---------------------------------------------------------------------
# 2. Generate and Normalize Datasets
# ----------------------------------------------------------------------


def safe_standardize(x: torch.Tensor) -> torch.Tensor:
    """Standardize along the sequence dimension within each dataset."""
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (x - mu) / std


def active_matrix(tensor_2d: np.ndarray, active_dim: int) -> np.ndarray:
    """Keep only active feature columns, avoiding padded all-zero dimensions."""
    if tensor_2d.ndim == 1:
        return tensor_2d.reshape(-1, 1)
    active_dim = max(1, min(int(active_dim), tensor_2d.shape[1]))
    return tensor_2d[:, :active_dim]


def add_feature_columns(sample_data: dict, var_name: str, data: np.ndarray) -> None:
    """Add one or more feature columns to the dataframe dictionary."""
    if data.ndim == 1 or data.shape[1] == 1:
        sample_data[var_name] = data.reshape(-1)
    else:
        for col in range(data.shape[1]):
            sample_data[f"{var_name}_{col}"] = data[:, col]


iterator = iter(dataset)

print(f"Generating {n_batches * dataset.batch_size} datasets...")

all_datasets = []
printed_shapes = False

for _ in tqdm(range(n_batches), desc="Sampling Batches"):
    batch = next(iterator)

    if len(batch) < 17:
        raise RuntimeError(
            "Expected batch to contain MU0 and MU1. Your PriorDataset.get_batch() "
            "should return 17 objects: U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, "
            "MU0, MU1, d_U, d_X, d_W, d_Z, seq_lens, train_sizes."
        )

    (
        U,
        X,
        W,
        Z,
        A,
        A_probs,
        y_obs,
        y0,
        y1,
        mu0,
        mu1,
        d_U,
        d_X,
        d_W,
        d_Z,
        seq_lens,
        train_sizes,
    ) = batch

    if not printed_shapes:
        print("Batch shapes:")
        print("  U    :", tuple(U.shape))
        print("  X    :", tuple(X.shape))
        print("  W    :", tuple(W.shape))
        print("  Z    :", tuple(Z.shape))
        print("  y0   :", tuple(y0.shape))
        print("  y1   :", tuple(y1.shape))
        print("  mu0  :", tuple(mu0.shape))
        print("  mu1  :", tuple(mu1.shape))
        print("  ITE  :", tuple((y1 - y0).shape))
        print("  CATE :", tuple((mu1 - mu0).shape))
        printed_shapes = True

    U = safe_standardize(U)
    X = safe_standardize(X)
    W = safe_standardize(W)
    Z = safe_standardize(Z)

    assert (
        len(np.unique(train_sizes.cpu().numpy())) == 1
    ), "Expected all train sizes in the batch to be the same for this code."

    y_obs_ctx = y_obs[:, : train_sizes[0]]
    y_obs_ctx_mean = y_obs_ctx.mean(dim=1, keepdim=True)
    y_obs_ctx_std = y_obs_ctx.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

    y_obs_std = (y_obs - y_obs_ctx_mean) / y_obs_ctx_std
    y0_std = (y0 - y_obs_ctx_mean) / y_obs_ctx_std
    y1_std = (y1 - y_obs_ctx_mean) / y_obs_ctx_std
    mu0_std = (mu0 - y_obs_ctx_mean) / y_obs_ctx_std
    mu1_std = (mu1 - y_obs_ctx_mean) / y_obs_ctx_std

    for i in range(dataset.batch_size):
        seq_len = int(seq_lens[i].item())
        train_size = int(train_sizes[i].item())
        train_size = max(1, min(train_size, seq_len))

        u_raw = active_matrix(U[i, :seq_len].cpu().numpy(), int(d_U[i].item()))
        x_raw = active_matrix(X[i, :seq_len].cpu().numpy(), int(d_X[i].item()))
        w_raw = active_matrix(W[i, :seq_len].cpu().numpy(), int(d_W[i].item()))
        z_raw = active_matrix(Z[i, :seq_len].cpu().numpy(), int(d_Z[i].item()))

        a_raw = A[i, :seq_len].cpu().numpy()
        a_probs_raw = A_probs[i, :seq_len].cpu().numpy()
        y_obs_raw = y_obs_std[i, :seq_len].cpu().numpy()
        y0_raw = y0_std[i, :seq_len].cpu().numpy()
        y1_raw = y1_std[i, :seq_len].cpu().numpy()
        mu0_raw = mu0_std[i, :seq_len].cpu().numpy()
        mu1_raw = mu1_std[i, :seq_len].cpu().numpy()

        ite_raw = y1_raw - y0_raw
        cate_raw = mu1_raw - mu0_raw
        mc_residual_raw = ite_raw - cate_raw

        sample_data = {
            "is_train": np.arange(seq_len) < train_size,
            "A": a_raw.flatten(),
            "A_probs": a_probs_raw.flatten(),
            "Y_obs": y_obs_raw.flatten(),
            "Y0": y0_raw.flatten(),
            "Y1": y1_raw.flatten(),
            "MU0": mu0_raw.flatten(),
            "MU1": mu1_raw.flatten(),
            "ITE": ite_raw.flatten(),
            "CATE": cate_raw.flatten(),
            "ITE_minus_CATE": mc_residual_raw.flatten(),
        }

        add_feature_columns(sample_data, "U", u_raw)
        add_feature_columns(sample_data, "X", x_raw)
        add_feature_columns(sample_data, "W", w_raw)
        add_feature_columns(sample_data, "Z", z_raw)

        df_sample = pd.DataFrame(sample_data)
        df_sample.attrs["train_size"] = train_size
        df_sample.attrs["seq_len"] = seq_len
        df_sample.attrs["mc_cate"] = mc_cate

        all_datasets.append(df_sample)

print(f"Collected {len(all_datasets)} datasets.")


# %% ---------------------------------------------------------------------
# Helper functions for plotting
# ----------------------------------------------------------------------


def columns_for_var(df: pd.DataFrame, var: str) -> list[str]:
    """Return columns for scalar or vector-valued variables."""
    if var in df.columns:
        return [var]
    return sorted([c for c in df.columns if c.startswith(f"{var}_")])


def safe_kde_or_hist(series, ax, label=None, color=None, fill=True, alpha=0.2):
    """KDE when possible, histogram fallback for nearly constant or tiny data."""
    values = pd.Series(series).dropna().values
    if len(values) < 3 or len(np.unique(np.round(values, 8))) < 3:
        sns.histplot(values, ax=ax, bins=20, color=color, alpha=alpha, label=label)
    else:
        sns.kdeplot(values, ax=ax, label=label, fill=fill, alpha=alpha, color=color)


# %% ---------------------------------------------------------------------
# 3. Marginal Distributions for a Few Datasets
# ----------------------------------------------------------------------

k = 5
n_show = min(k, len(all_datasets))
vars_to_plot = ["U", "Z", "X", "W", "A_probs", "Y_obs", "ITE", "CATE"]

fig, axes = plt.subplots(
    n_show,
    len(vars_to_plot),
    figsize=(4 * len(vars_to_plot), 3 * n_show),
    sharex=False,
    sharey=False,
    squeeze=False,
)

a_prob_bins = np.linspace(0, 1, 21)
palette = sns.color_palette("viridis", n_show)

for i in range(n_show):
    df = all_datasets[i]

    for j, var in enumerate(vars_to_plot):
        ax = axes[i, j]
        cols = columns_for_var(df, var)

        if var == "A_probs":
            for col in cols:
                sns.histplot(
                    df[col].values,
                    bins=a_prob_bins,
                    ax=ax,
                    color=palette[i],
                    kde=False,
                    element="step",
                )
            ax.set_xlim(0, 1)
        else:
            dim_palette = sns.color_palette("tab10", max(1, len(cols)))
            for idx, col in enumerate(cols):
                safe_kde_or_hist(
                    df[col].values,
                    ax=ax,
                    color=dim_palette[idx] if len(cols) > 1 else palette[i],
                    label=col if len(cols) > 1 else None,
                    fill=True,
                    alpha=0.15,
                )
            if len(cols) > 1:
                ax.legend(fontsize=8, loc="upper right")
            if var not in ["ITE", "CATE"]:
                ax.set_xlim(-4, 4)

        if i == 0:
            ax.set_title(f"Marginal {var}", fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"Dataset {i}", fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------
# 4. Global Statistics Distribution across all Datasets
# ----------------------------------------------------------------------

vars_to_analyze = [
    "U",
    "Z",
    "X",
    "W",
    "A_probs",
    "Y_obs",
    "ITE",
    "CATE",
    "ITE_minus_CATE",
]
stats_to_compute = ["mean", "median", "std", "min", "max"]

stats_records = []

for i, df in enumerate(all_datasets):
    for var in vars_to_analyze:
        cols = columns_for_var(df, var)

        for col in cols:
            series = df[col]
            stats_records.append(
                {
                    "dataset_idx": i,
                    "variable": var,
                    "column": col,
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                }
            )

stats_df = pd.DataFrame(stats_records)

fig, axes = plt.subplots(
    len(stats_to_compute),
    len(vars_to_analyze),
    figsize=(4 * len(vars_to_analyze), 3 * len(stats_to_compute)),
    sharex=False,
    sharey=False,
    squeeze=False,
)

for row_idx, stat in enumerate(stats_to_compute):
    for col_idx, var in enumerate(vars_to_analyze):
        ax = axes[row_idx, col_idx]

        data_subset = stats_df[stats_df["variable"] == var]
        cols = data_subset["column"].unique()

        if len(data_subset) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        else:
            all_constant = True
            constant_vals = []
            for col in cols:
                col_vals = data_subset[data_subset["column"] == col][stat].values
                if (
                    len(col_vals) == 0
                    or (np.nanmax(col_vals) - np.nanmin(col_vals)) >= 1e-3
                ):
                    all_constant = False
                    break
                constant_vals.append(f"{col}: {col_vals[0]:.2f}")

            if all_constant:
                ax.text(
                    0.5,
                    0.5,
                    "\n".join(constant_vals),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, 1)
                ax.grid(False)
            else:
                use_kde = False
                if len(cols) > 1:
                    sns.histplot(
                        data=data_subset,
                        x=stat,
                        hue="column",
                        palette="tab10",
                        ax=ax,
                        bins=20,
                        kde=use_kde,
                        alpha=0.4,
                        element="step",
                        legend=False,
                    )
                    from matplotlib.patches import Patch

                    colors = sns.color_palette("tab10", len(cols))
                    handles = [
                        Patch(color=colors[idx], label=col)
                        for idx, col in enumerate(cols)
                    ]
                    ax.legend(handles=handles, loc="upper right", fontsize=8)
                else:
                    sns.histplot(
                        data=data_subset,
                        x=stat,
                        ax=ax,
                        bins=20,
                        kde=use_kde,
                        color="royalblue",
                        alpha=0.6,
                    )

                if var == "A_probs" and stat in ["mean", "median", "min", "max"]:
                    ax.set_xlim(0, 1)

        if row_idx == 0:
            ax.set_title(f"Variable: {var}", fontweight="bold", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel(f"Stat: {stat.upper()}", fontweight="bold", fontsize=12)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------
# 5. Potential Outcomes, Sampled ITE, and Monte Carlo CATE
# ----------------------------------------------------------------------

k = 5
n_show = min(k, len(all_datasets))

fig, axes = plt.subplots(n_show, 6, figsize=(30, 4 * n_show), squeeze=False)

for i in range(n_show):
    df = all_datasets[i]

    ite = df["ITE"]
    cate = df["CATE"]

    x_cols = columns_for_var(df, "X")
    x_axis = df[x_cols[0]] if x_cols else pd.Series(np.arange(len(df)))

    # Column 1: sampled Y0 vs Y1
    ax0 = axes[i, 0]
    sns.scatterplot(data=df, x="Y0", y="Y1", ax=ax0, alpha=0.4, color="purple", s=15)
    lims = [
        min(ax0.get_xlim()[0], ax0.get_ylim()[0]),
        max(ax0.get_xlim()[1], ax0.get_ylim()[1]),
    ]
    ax0.plot(lims, lims, "k--", alpha=0.5, zorder=0)
    ax0.set_title(f"DS {i}: sampled Y0 vs Y1")

    # Column 2: MC means MU0 vs MU1
    ax1 = axes[i, 1]
    sns.scatterplot(
        data=df, x="MU0", y="MU1", ax=ax1, alpha=0.5, color="darkgreen", s=15
    )
    lims = [
        min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
        max(ax1.get_xlim()[1], ax1.get_ylim()[1]),
    ]
    ax1.plot(lims, lims, "k--", alpha=0.5, zorder=0)
    ax1.set_title("MU0(X) vs MU1(X)")

    # Column 3: sampled ITE vs MC CATE
    ax2 = axes[i, 2]
    sns.scatterplot(x=ite, y=cate, ax=ax2, alpha=0.5, color="black", s=15)
    lims = [
        min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
        max(ax2.get_xlim()[1], ax2.get_ylim()[1]),
    ]
    ax2.plot(lims, lims, "k--", alpha=0.5, zorder=0)
    ax2.set_xlabel("Sampled ITE = Y1 - Y0")
    ax2.set_ylabel("MC CATE = MU1 - MU0")
    ax2.set_title("ITE vs CATE")

    # Column 4: ITE and CATE distributions
    ax3 = axes[i, 3]
    safe_kde_or_hist(
        ite, ax=ax3, label="Sampled ITE", fill=True, alpha=0.2, color="gray"
    )
    safe_kde_or_hist(cate, ax=ax3, label="MC CATE", fill=True, alpha=0.2, color="green")
    ax3.axvline(0, color="red", linestyle="--", linewidth=1)
    ax3.set_title(f"Effects: ATE={cate.mean():.2f}, ITE mean={ite.mean():.2f}")
    ax3.legend()

    # Column 5: CATE as a function of X_0
    ax4 = axes[i, 4]
    plot_df = pd.DataFrame({"X0": x_axis, "CATE": cate, "ITE": ite}).sort_values("X0")
    sns.scatterplot(
        data=plot_df,
        x="X0",
        y="ITE",
        ax=ax4,
        alpha=0.25,
        color="gray",
        s=12,
        label="ITE",
    )
    sns.lineplot(
        data=plot_df,
        x="X0",
        y="CATE",
        ax=ax4,
        color="green",
        linewidth=2,
        label="CATE",
    )
    ax4.set_title("CATE smoothing over sampled ITE")
    ax4.set_xlabel(x_cols[0] if x_cols else "row index")

    # Column 6: MC residual ITE - CATE
    ax5 = axes[i, 5]
    sns.histplot(
        df["ITE_minus_CATE"],
        ax=ax5,
        kde=True,
        color="orange",
        alpha=0.4,
        element="step",
    )
    ax5.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax5.set_title(
        "ITE - CATE residual\n"
        f"mean={df['ITE_minus_CATE'].mean():.2f}, std={df['ITE_minus_CATE'].std():.2f}"
    )

for ax in axes.flatten():
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------
# 6. ATE Thresholds Based on True Monte Carlo CATE
# ----------------------------------------------------------------------

target_proportions = [0.1, 0.3, 0.5]

abs_ates = [abs(df["CATE"].mean()) for df in all_datasets]
ate_thresholds = np.quantile(abs_ates, target_proportions)

for prop, t in zip(target_proportions, ate_thresholds):
    print(f"Threshold for the bottom {prop:.0%}: |ATE| < {t:.4f}")


# %% ---------------------------------------------------------------------
# 7. Global CATE Statistics across all Datasets
# ----------------------------------------------------------------------

cate_stats_records = []

for i, df in enumerate(all_datasets):
    ite = df["ITE"]
    cate = df["CATE"]

    cate_stats_records.append(
        {
            "dataset_idx": i,
            "ATE": cate.mean(),
            "sampled_ITE_mean": ite.mean(),
            "ATE_minus_ITE_mean": cate.mean() - ite.mean(),
            "CATE_std": cate.std(),
            "ITE_std": ite.std(),
            "CATE_median": cate.median(),
            "CATE_min": cate.min(),
            "CATE_max": cate.max(),
            "MC_residual_std": df["ITE_minus_CATE"].std(),
        }
    )

cate_summary_df = pd.DataFrame(cate_stats_records)

stats_to_plot = [
    "ATE",
    "sampled_ITE_mean",
    "ATE_minus_ITE_mean",
    "CATE_std",
    "ITE_std",
    "MC_residual_std",
]

fig, axes = plt.subplots(
    1, len(stats_to_plot), figsize=(4.5 * len(stats_to_plot), 5), squeeze=False
)
axes = axes.flatten()
colors = sns.color_palette("tab10", len(stats_to_plot))

for j, stat in enumerate(stats_to_plot):
    ax = axes[j]
    data = cate_summary_df[stat]

    sns.histplot(data, kde=True, ax=ax, color=colors[j], alpha=0.6)

    if stat in ["ATE", "sampled_ITE_mean", "ATE_minus_ITE_mean"]:
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)

    ax.set_title(f"Dist. of {stat}", fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency" if j == 0 else "")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------
# 8. ATE Estimation Diagnostics
# ----------------------------------------------------------------------


def estimate_ate_t_learner(df: pd.DataFrame, cov_cols: list[str]) -> float:
    """Simple cross-fitted T-learner ATE estimate from observed data."""
    if not cov_cols:
        return np.nan

    a_values = df["A"].values
    y_values = df["Y_obs"].values
    x_values = df[cov_cols].values
    n_samples = len(df)

    n_0 = np.sum(a_values == 0)
    n_1 = np.sum(a_values == 1)
    if n_0 < 5 or n_1 < 5:
        return np.nan

    n_splits = min(5, n_0, n_1)
    if n_splits < 2:
        return np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred_0 = np.zeros(n_samples)
    pred_1 = np.zeros(n_samples)

    for train_idx, val_idx in skf.split(x_values, a_values):
        x_train = x_values[train_idx]
        a_train = a_values[train_idx]
        y_train = y_values[train_idx]
        x_val = x_values[val_idx]

        idx_0 = a_train == 0
        idx_1 = a_train == 1
        if np.sum(idx_0) < 2 or np.sum(idx_1) < 2:
            return np.nan

        m0 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        m1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        m0.fit(x_train[idx_0], y_train[idx_0])
        m1.fit(x_train[idx_1], y_train[idx_1])
        pred_0[val_idx] = m0.predict(x_val)
        pred_1[val_idx] = m1.predict(x_val)

    return float((pred_1 - pred_0).mean())


def process_single_dataset(df: pd.DataFrame):
    true_ate = df["CATE"].mean()
    sampled_ite_ate = df["ITE"].mean()

    a1 = df[df["A"] == 1]
    a0 = df[df["A"] == 0]
    if len(a1) > 0 and len(a0) > 0:
        naive_ate = a1["Y_obs"].mean() - a0["Y_obs"].mean()
    else:
        naive_ate = np.nan

    x_cols = columns_for_var(df, "X")
    w_cols = columns_for_var(df, "W")
    z_cols = columns_for_var(df, "Z")
    u_cols = columns_for_var(df, "U")

    # A is not included as a covariate for the T-learner because it is used to split arms.
    covs_x = x_cols
    covs_xw = x_cols + w_cols
    covs_xwz = x_cols + w_cols + z_cols
    covs_xwzu = x_cols + w_cols + z_cols + u_cols

    x_ate = estimate_ate_t_learner(df, covs_x)
    xw_ate = estimate_ate_t_learner(df, covs_xw)
    xwz_ate = estimate_ate_t_learner(df, covs_xwz)
    xwzu_ate = estimate_ate_t_learner(df, covs_xwzu)

    return true_ate, sampled_ite_ate, naive_ate, x_ate, xw_ate, xwz_ate, xwzu_ate


results = Parallel(n_jobs=-1)(
    delayed(process_single_dataset)(df)
    for df in tqdm(all_datasets, desc="Calculating ATEs")
)

true_ates = np.array([r[0] for r in results], dtype=float)
sampled_ite_ates = np.array([r[1] for r in results], dtype=float)
naive_ates = np.array([r[2] for r in results], dtype=float)
pred_x_ates = np.array([r[3] for r in results], dtype=float)
pred_xw_ates = np.array([r[4] for r in results], dtype=float)
pred_xwz_ates = np.array([r[5] for r in results], dtype=float)
pred_xwzu_ates = np.array([r[6] for r in results], dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(23, 6))

# Estimated vs true ATE
ax = axes[0]
scatter_specs = [
    (sampled_ite_ates, "Sampled ITE ATE", "gray"),
    (naive_ates, "Naive ATE", "#2c7bb6"),
    (pred_x_ates, "T-learner X", "#fdae61"),
    (pred_xw_ates, "T-learner X,W", "#d7191c"),
    (pred_xwz_ates, "T-learner X,W,Z", "#9467bd"),
    (pred_xwzu_ates, "T-learner X,W,Z,U oracle", "#2ca02c"),
]

for values, label, color in scatter_specs:
    sns.scatterplot(
        x=true_ates,
        y=values,
        color=color,
        label=label,
        alpha=0.8,
        s=50,
        ax=ax,
    )

all_values = [true_ates] + [values for values, _, _ in scatter_specs]
finite_values = np.concatenate(
    [v[np.isfinite(v)] for v in all_values if np.isfinite(v).any()]
)
if len(finite_values) > 0:
    min_val = finite_values.min()
    max_val = finite_values.max()
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)

ax.set_title("Estimated vs True ATE", fontweight="bold", fontsize=14)
ax.set_xlabel("True ATE = mean(MU1 - MU0)", fontsize=12)
ax.set_ylabel("Estimated ATE", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Absolute error distribution
err_data = pd.DataFrame(
    {
        "Sampled ITE": np.abs(sampled_ite_ates - true_ates),
        "Naive": np.abs(naive_ates - true_ates),
        "X": np.abs(pred_x_ates - true_ates),
        "X,W": np.abs(pred_xw_ates - true_ates),
        "X,W,Z": np.abs(pred_xwz_ates - true_ates),
        "X,W,Z,U": np.abs(pred_xwzu_ates - true_ates),
    }
)
ax = axes[1]
sns.boxplot(data=err_data, palette="tab10", ax=ax)
ax.set_title("Absolute Error Distribution", fontweight="bold", fontsize=14)
ax.set_ylabel("Absolute Error", fontsize=12)
ax.tick_params(axis="x", rotation=30)
ax.grid(True, alpha=0.3)

# Signed bias distribution
bias_data = pd.DataFrame(
    {
        "Sampled ITE": sampled_ite_ates - true_ates,
        "Naive": naive_ates - true_ates,
        "X": pred_x_ates - true_ates,
        "X,W": pred_xw_ates - true_ates,
        "X,W,Z": pred_xwz_ates - true_ates,
        "X,W,Z,U": pred_xwzu_ates - true_ates,
    }
)
ax = axes[2]
sns.boxplot(data=bias_data, palette="tab10", ax=ax)
ax.axhline(0, color="black", linestyle="--", alpha=0.6)
ax.set_title("Signed Bias Distribution", fontweight="bold", fontsize=14)
ax.set_ylabel("Signed Bias", fontsize=12)
ax.tick_params(axis="x", rotation=30)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% ---------------------------------------------------------------------
# 9. Print Summary Metrics
# ----------------------------------------------------------------------


def mean_abs_error(values: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(target)
    if not mask.any():
        return np.nan
    return float(np.abs(values[mask] - target[mask]).mean())


print(f"Average True ATE: {np.nanmean(true_ates):.3f}")
print(
    f"Average Sampled ITE ATE: {np.nanmean(sampled_ite_ates):.3f} "
    f"(Error: {mean_abs_error(sampled_ite_ates, true_ates):.3f})"
)
print(
    f"Average Naive ATE: {np.nanmean(naive_ates):.3f} "
    f"(Error: {mean_abs_error(naive_ates, true_ates):.3f})"
)
print(
    f"Average Predicted ATE (X): {np.nanmean(pred_x_ates):.3f} "
    f"(Error: {mean_abs_error(pred_x_ates, true_ates):.3f})"
)
print(
    f"Average Predicted ATE (X,W): {np.nanmean(pred_xw_ates):.3f} "
    f"(Error: {mean_abs_error(pred_xw_ates, true_ates):.3f})"
)
print(
    f"Average Predicted ATE (X,W,Z): {np.nanmean(pred_xwz_ates):.3f} "
    f"(Error: {mean_abs_error(pred_xwz_ates, true_ates):.3f})"
)
print(
    f"Average Predicted ATE (X,W,Z,U oracle): {np.nanmean(pred_xwzu_ates):.3f} "
    f"(Error: {mean_abs_error(pred_xwzu_ates, true_ates):.3f})"
)

# %%
