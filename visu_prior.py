#%%

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from proximalfm.prior import PriorDataset


#%% ---------------------------------------------------------------------
# 1. Instantiate Prior Dataset
# ----------------------------------------------------------------------
n_batches = 2
batch_size = 60

dataset = PriorDataset(
    batch_size=batch_size,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=10,
    min_dim_W=1,
    max_dim_W=10,
    min_dim_Z=1,
    max_dim_Z=10,
    min_dim_X=1,
    max_dim_X=10,
    min_seq_len=None,
    max_seq_len=1024,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.5,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    device="cpu",
)

#%% ---------------------------------------------------------------------
# 2. Generate and Normalize Datasets
# ----------------------------------------------------------------------


def safe_standardize(x):
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    std = torch.where(std < 1e-5, torch.ones_like(std), std)
    return (x - mu) / std


iterator = iter(dataset)

print(f"Generating {n_batches * dataset.batch_size} datasets...")

all_datasets = []

for _ in tqdm(range(n_batches), desc="Sampling Batches"):
    batch = next(iterator)

    U, X, W, Z, A, A_probs, y_obs, y0, y1 = batch[:9]
    seq_lens, train_sizes = batch[-2:]

    U = safe_standardize(U)
    X = safe_standardize(X)
    W = safe_standardize(W)
    Z = safe_standardize(Z)

    assert (
        len(np.unique(train_sizes.cpu().numpy())) == 1
    ), "Expected all train sizes in the batch to be the same for this code."
    y_obs_ctx = y_obs[:, : train_sizes[0]]
    y_obs_ctx_mean = y_obs_ctx.mean(dim=1, keepdim=True)
    y_obs_ctx_std = y_obs_ctx.std(dim=1, keepdim=True)
    y_obs_ctx_std = y_obs_ctx.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

    y_obs_std = (y_obs - y_obs_ctx_mean) / y_obs_ctx_std
    y0_std = (y0 - y_obs_ctx_mean) / y_obs_ctx_std
    y1_std = (y1 - y_obs_ctx_mean) / y_obs_ctx_std

    for i in range(dataset.batch_size):
        seq_len = int(seq_lens[i].item())
        train_size = int(train_sizes[i].item())
        train_size = max(1, min(train_size, seq_len))

        u_raw = U[i, :seq_len].cpu().numpy()
        x_raw = X[i, :seq_len].cpu().numpy()
        w_raw = W[i, :seq_len].cpu().numpy()
        z_raw = Z[i, :seq_len].cpu().numpy()
        a_raw = A[i, :seq_len].cpu().numpy()
        a_probs_raw = A_probs[i, :seq_len].cpu().numpy()
        y_obs_raw = y_obs_std[i, :seq_len].cpu().numpy()
        y0_raw = y0_std[i, :seq_len].cpu().numpy()
        y1_raw = y1_std[i, :seq_len].cpu().numpy()

        sample_data = {
            "is_train": np.arange(seq_len) < train_size,
            "A": a_raw.flatten(),
            "A_probs": a_probs_raw.flatten(),
            "Y_obs": y_obs_raw.flatten(),
            "Y0": y0_raw.flatten(),
            "Y1": y1_raw.flatten(),
        }

        for var_name, data in [("U", u_raw), ("X", x_raw), ("W", w_raw), ("Z", z_raw)]:
            if data.ndim == 1:
                sample_data[var_name] = data
            else:
                for col in range(data.shape[1]):
                    sample_data[f"{var_name}_{col}"] = data[:, col]

        df_sample = pd.DataFrame(sample_data)

        df_sample.attrs["train_size"] = train_size
        df_sample.attrs["seq_len"] = seq_len

        all_datasets.append(df_sample)

#%% ---------------------------------------------------------------------
# 3. Marginal Unique Datasets
# ----------------------------------------------------------------------

k = 5
vars_to_plot = ["U", "Z", "X", "W", "A_probs", "Y_obs"]

fig, axes = plt.subplots(k, 6, figsize=(20, 3 * k), sharex=False, sharey=False)

a_prob_bins = np.linspace(0, 1, 21)
palette = sns.color_palette("viridis", k)

for i in range(min(k, len(all_datasets))):
    df = all_datasets[i]

    for j, var in enumerate(vars_to_plot):
        ax = axes[i, j]

        if var in df.columns:
            cols = [var]
        else:
            cols = sorted([c for c in df.columns if c.startswith(f"{var}_")])

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
            dim_palette = sns.color_palette("tab10", len(cols))
            for idx, col in enumerate(cols):
                sns.kdeplot(
                    df[col].values,
                    ax=ax,
                    color=dim_palette[idx] if len(cols) > 1 else palette[i],
                    label=col if len(cols) > 1 else None,
                    fill=True,
                    alpha=0.15,
                )
            if len(cols) > 1:
                ax.legend(fontsize=8, loc="upper right")
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

#%% ---------------------------------------------------------------------
# 4. Global Statistics Distribution across all Datasets (Fixed)
# ---------------------------------------------------------------------

vars_to_analyze = ["U", "Z", "X", "W", "A_probs", "Y_obs"]
stats_to_compute = ["mean", "median", "std", "min", "max"]

stats_records = []

for i, df in enumerate(all_datasets):
    for var in vars_to_analyze:
        if var in df.columns:
            cols = [var]
        else:
            cols = [c for c in df.columns if c.startswith(f"{var}_")]

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
    figsize=(20, 3 * len(stats_to_compute)),
    sharex=False,
    sharey=False,
)

for row_idx, stat in enumerate(stats_to_compute):
    for col_idx, var in enumerate(vars_to_analyze):
        ax = axes[row_idx, col_idx]

        data_subset = stats_df[stats_df["variable"] == var]
        cols = data_subset["column"].unique()

        all_constant = True
        constant_vals = []
        for col in cols:
            col_vals = data_subset[data_subset["column"] == col][stat].values
            if len(col_vals) == 0 or (np.max(col_vals) - np.min(col_vals)) >= 1e-3:
                all_constant = False
                break
            else:
                if len(col_vals) > 0:
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
            for col in cols:
                col_vals = data_subset[data_subset["column"] == col][stat].values
                if np.std(col_vals) < 1e-2 or len(np.unique(col_vals)) <= 5:
                    use_kde = False
                    break

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
                    Patch(color=colors[idx], label=col) for idx, col in enumerate(cols)
                ]
                ax.legend(handles=handles, loc="upper right", fontsize=8)
            else:
                sns.histplot(
                    data=data_subset,
                    x=stat,
                    ax=ax,
                    bins=20,  # Hardcoded bins avoids zero-variance division crashes
                    kde=use_kde,
                    color="royalblue",
                    alpha=0.6,
                )

            if var == "A_probs" and stat in ["mean", "median", "min", "max"]:
                ax.set_xlim(0, 1)

        if row_idx == 0:
            ax.set_title(f"Variable: {var}", fontweight="bold", fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(f"Stat: {stat.upper()}", fontweight="bold", fontsize=12)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------------
# 5. Potential Outcomes & CATE (One Row per Dataset)
# ---------------------------------------------------------------------
k = 15
fig, axes = plt.subplots(k, 4, figsize=(20, 4 * k))

for i in range(min(k, len(all_datasets))):
    df = all_datasets[i]

    # Calculate Individual Treatment Effect (ITE/CATE)
    # Using Y1 - Y0 (Standard convention)
    cate = df["Y1"] - df["Y0"]

    # --- Column 1: Y0 vs Y1 Scatter ---
    ax0 = axes[i, 0]
    sns.scatterplot(data=df, x="Y0", y="Y1", ax=ax0, alpha=0.4, color="purple", s=15)
    # Add identity line to see if Y1 is generally higher/lower than Y0
    lims = [
        min(ax0.get_xlim()[0], ax0.get_ylim()[0]),
        max(ax0.get_xlim()[1], ax0.get_ylim()[1]),
    ]
    ax0.plot(lims, lims, "k--", alpha=0.5, zorder=0)
    ax0.set_title(f"DS {i}: $Y_0$ vs $Y_1$")

    # --- Column 2: Distribution of Y0 and Y1 ---
    ax1 = axes[i, 1]
    sns.kdeplot(df["Y0"], ax=ax1, label="$Y_0$", fill=True, alpha=0.2, color="blue")
    sns.kdeplot(df["Y1"], ax=ax1, label="$Y_1$", fill=True, alpha=0.2, color="orange")
    ax1.set_title("Potential Outcomes Dist.")
    ax1.legend()

    # --- Column 3: Y_obs for A=0 and A=1 ---
    ax2 = axes[i, 2]
    # Check if both groups exist to avoid KDE errors
    if len(df[df["A"] == 0]) > 1:
        sns.kdeplot(
            df[df["A"] == 0]["Y_obs"],
            ax=ax2,
            label="$Y_{obs} | A=0$",
            fill=True,
            alpha=0.2,
            color="blue",
        )
    if len(df[df["A"] == 1]) > 1:
        sns.kdeplot(
            df[df["A"] == 1]["Y_obs"],
            ax=ax2,
            label="$Y_{obs} | A=1$",
            fill=True,
            alpha=0.2,
            color="orange",
        )
    ax2.set_title("Observed Outcomes by Group")
    ax2.legend()

    # --- Column 4: Distribution of CATE (Y1 - Y0) ---
    ax3 = axes[i, 3]
    sns.histplot(cate, ax=ax3, kde=True, color="green", alpha=0.3, element="step")
    ax3.axvline(0, color="red", linestyle="--", linewidth=2)
    ax3.set_title(f"CATE Distribution ($\\mu={cate.mean():.2f}$)")
    ax3.set_xlabel("$Y1 - Y0$")

# Global formatting
for ax in axes.flatten():
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

#%%

target_proportions = [0.1, 0.3, 0.5]

abs_ates = [abs((df["Y1"] - df["Y0"]).mean()) for df in all_datasets]
ate_thresholds = np.quantile(abs_ates, target_proportions)

for prop, t in zip(target_proportions, ate_thresholds):
    print(f"Threshold for the bottom {prop:.0%}: |ATE| < {t:.4f}")


#%% ---------------------------------------------------------------------
# 6. Global CATE Statistics across all Datasets
# ---------------------------------------------------------------------

cate_stats_records = []

for i, df in enumerate(all_datasets):
    # Calculate Individual Treatment Effect
    cate = df["Y1"] - df["Y0"]

    cate_stats_records.append(
        {
            "dataset_idx": i,
            "ATE": cate.mean(),
            "CATE_std": cate.std(),
            "CATE_median": cate.median(),
            "CATE_min": cate.min(),
            "CATE_max": cate.max(),
        }
    )

cate_summary_df = pd.DataFrame(cate_stats_records)

# We'll plot 1 row x 5 columns for the different CATE summary stats
stats_to_plot = ["ATE", "CATE_std", "CATE_median", "CATE_min", "CATE_max"]
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

colors = ["crimson", "darkslateblue", "forestgreen", "darkorange", "darkcyan"]

for j, stat in enumerate(stats_to_plot):
    ax = axes[j]
    data = cate_summary_df[stat]

    sns.histplot(data, kde=True, ax=ax, color=colors[j], alpha=0.6)

    # Add a vertical line at 0 for positional context (especially for ATE/Median)
    if stat in ["ATE", "CATE_median", "CATE_min", "CATE_max"]:
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)

    ax.set_title(f"Dist. of {stat}", fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency" if j == 0 else "")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

#%%
def estimate_ate_t_learner(df, cov_cols):
    from sklearn.model_selection import StratifiedKFold

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
        x_train, a_train, y_train = (
            x_values[train_idx],
            a_values[train_idx],
            y_values[train_idx],
        )
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
    return (pred_1 - pred_0).mean()


from joblib import Parallel, delayed


def process_single_dataset(df):
    t_ate = (df["Y1"] - df["Y0"]).mean()
    a1 = df[df["A"] == 1]
    a0 = df[df["A"] == 0]
    if len(a1) > 0 and len(a0) > 0:
        n_ate = a1["Y_obs"].mean() - a0["Y_obs"].mean()
    else:
        n_ate = np.nan
    x_cols = [c for c in df.columns if c.startswith("X_")]
    if not x_cols and "X" in df.columns:
        x_cols = ["X"]
    w_cols = [c for c in df.columns if c.startswith("W_")]
    if not w_cols and "W" in df.columns:
        w_cols = ["W"]
    z_cols = [c for c in df.columns if c.startswith("Z_")]
    if not z_cols and "Z" in df.columns:
        z_cols = ["Z"]
    u_cols = [c for c in df.columns if c.startswith("U_")]
    if not u_cols and "U" in df.columns:
        u_cols = ["U"]
    covs_xa = x_cols
    covs_xaw = x_cols + w_cols
    covs_xawu = x_cols + w_cols + u_cols
    xa_ate = estimate_ate_t_learner(df, covs_xa)
    xaw_ate = estimate_ate_t_learner(df, covs_xaw)
    xawu_ate = estimate_ate_t_learner(df, covs_xawu)
    return t_ate, n_ate, xa_ate, xaw_ate, xawu_ate


results = Parallel(n_jobs=-1)(
    delayed(process_single_dataset)(df)
    for df in tqdm(all_datasets, desc="Calculating ATEs")
)

true_ates = np.array([r[0] for r in results])
naive_ates = np.array([r[1] for r in results])
pred_xa_ates = np.array([r[2] for r in results])
pred_xaw_ates = np.array([r[3] for r in results])
pred_xawu_ates = np.array([r[4] for r in results])

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

ax = axes[0]
sns.scatterplot(
    x=true_ates,
    y=naive_ates,
    color="#2c7bb6",
    label="Naive ATE",
    alpha=0.8,
    s=50,
    ax=ax,
)
sns.scatterplot(
    x=true_ates,
    y=pred_xa_ates,
    color="#fdae61",
    label="Predicted from X, A",
    alpha=0.8,
    s=50,
    ax=ax,
)
sns.scatterplot(
    x=true_ates,
    y=pred_xaw_ates,
    color="#d7191c",
    label="Predicted from X, A, W",
    alpha=0.8,
    s=50,
    ax=ax,
)
sns.scatterplot(
    x=true_ates,
    y=pred_xawu_ates,
    color="#2ca02c",
    label="Predicted from X, A, W, U (Oracle)",
    alpha=0.8,
    s=50,
    ax=ax,
)

min_val = min(
    true_ates.min(),
    naive_ates.min(),
    pred_xa_ates.min(),
    pred_xaw_ates.min(),
    pred_xawu_ates.min(),
)
max_val = max(
    true_ates.max(),
    naive_ates.max(),
    pred_xa_ates.max(),
    pred_xaw_ates.max(),
    pred_xawu_ates.max(),
)
ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.6)
ax.set_title("Estimated vs True ATE", fontweight="bold", fontsize=14)
ax.set_xlabel("True ATE", fontsize=12)
ax.set_ylabel("Estimated ATE", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

err_data = pd.DataFrame(
    {
        "Naive": np.abs(naive_ates - true_ates),
        "X, A": np.abs(pred_xa_ates - true_ates),
        "X, A, W": np.abs(pred_xaw_ates - true_ates),
        "X, A, W, U": np.abs(pred_xawu_ates - true_ates),
    }
)
ax = axes[1]
sns.boxplot(data=err_data, palette=["#2c7bb6", "#fdae61", "#d7191c", "#2ca02c"], ax=ax)
ax.set_title("Absolute Error Distribution", fontweight="bold", fontsize=14)
ax.set_ylabel("Absolute Error", fontsize=12)
ax.grid(True, alpha=0.3)

bias_data = pd.DataFrame(
    {
        "Naive": naive_ates - true_ates,
        "X, A": pred_xa_ates - true_ates,
        "X, A, W": pred_xaw_ates - true_ates,
        "X, A, W, U": pred_xawu_ates - true_ates,
    }
)
ax = axes[2]
sns.boxplot(data=bias_data, palette=["#2c7bb6", "#fdae61", "#d7191c", "#2ca02c"], ax=ax)
ax.axhline(0, color="black", linestyle="--", alpha=0.6)
ax.set_title("Signed Bias Distribution", fontweight="bold", fontsize=14)
ax.set_ylabel("Signed Bias", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# print average results:
print(f"Average True ATE: {true_ates.mean():.3f}")
print(
    f"Average Naive ATE: {naive_ates.mean():.3f} (Error: {np.abs(naive_ates - true_ates).mean():.3f})"
)
print(
    f"Average Predicted ATE (X, A): {pred_xa_ates.mean():.3f} (Error: {np.abs(pred_xa_ates - true_ates).mean():.3f})"
)
print(
    f"Average Predicted ATE (X, A, W): {pred_xaw_ates.mean():.3f} (Error: {np.abs(pred_xaw_ates - true_ates).mean():.3f})"
)
print(
    f"Average Predicted ATE (X, A, W, U): {pred_xawu_ates.mean():.3f} (Error: {np.abs(pred_xawu_ates - true_ates).mean():.3f})"
)

#%%
