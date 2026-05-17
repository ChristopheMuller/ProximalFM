#%%

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from proximalfm.prior import PriorDataset


#%% ---------------------------------------------------------------------
# 1. Instantiate Prior Dataset
# ----------------------------------------------------------------------
n_batches = 10
batch_size = 5

dataset = PriorDataset(
    batch_size=batch_size,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_W=1,
    max_dim_W=1,
    min_dim_Z=1,
    max_dim_Z=1,
    min_dim_X=2,
    max_dim_X=2,
    min_seq_len=None,
    max_seq_len=2048,
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
    # Prevent division by zero for zero-variance or zero-padded features
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

# Create a k x 6 grid
fig, axes = plt.subplots(k, 6, figsize=(20, 3 * k), sharex=False, sharey=False)

a_prob_bins = np.linspace(0, 1, 21)
palette = sns.color_palette("viridis", k)

for i in range(min(k, len(all_datasets))):
    df = all_datasets[i]

    for j, var in enumerate(vars_to_plot):
        ax = axes[i, j]

        # 1. Extract data (handling multi-dimensional columns like X_0, X_1)
        if var in df.columns:
            data = df[var].values
        else:
            # Flatten dynamic columns (e.g., X_0, X_1 -> one distribution)
            data = df.filter(like=f"{var}_").values.flatten()

        # 2. Plotting Logic
        if var == "A_probs":
            # Histogram for treatment probabilities
            sns.histplot(
                data,
                bins=a_prob_bins,
                ax=ax,
                color=palette[i],
                kde=False,
                element="step",
            )
            ax.set_xlim(0, 1)
        else:
            # KDE for continuous features
            sns.kdeplot(data, ax=ax, color=palette[i], fill=True, alpha=0.2)
            # Center the view around zero for standardized features
            ax.set_xlim(-4, 4)

        # 3. Labeling
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
# 4. Global Statistics Distribution across all Datasets
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

# sharex=False allows each subplot to find its own optimal range
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

        data_subset = stats_df[stats_df["variable"] == var][stat]

        if np.std(data_subset) < 1e-4:
            ax.text(
                0.5,
                0.5,
                f"Constant: {data_subset.iloc[0]:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            ax.grid(False)

        else:

            sns.histplot(data_subset, ax=ax, kde=True, color="royalblue", alpha=0.6)

            # Row-specific logic for A_probs range
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
k = 10
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

#%% ---------------------------------------------------------------------
# 7. Signal Propagation & DAG Verification (Non-linear R2)
# ---------------------------------------------------------------------


def get_nl_r2(df, target_cols, predictor_cols):
    """Calculates non-linear R2 using a Random Forest."""
    X = df[predictor_cols]
    # Handle multi-dimensional targets by averaging R2 or flattening
    y = df[target_cols]
    if y.shape[1] == 1:
        y = y.values.flatten()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


dag_stats = []

for i, df in tqdm(
    enumerate(all_datasets), total=len(all_datasets), desc="Analyzing DAG Flow"
):

    cols_u = [c for c in df.columns if c.startswith("U")]
    cols_x = [c for c in df.columns if c.startswith("X")]
    cols_w = [c for c in df.columns if c.startswith("W")]
    cols_z = [c for c in df.columns if c.startswith("Z")]

    r2_u_to_z = get_nl_r2(df, cols_z, cols_u)
    r2_u_to_w = get_nl_r2(df, cols_w, cols_u)
    r2_u_to_x = get_nl_r2(df, cols_x, cols_u)
    r2_uxz_to_a = get_nl_r2(df, ["A_probs"], cols_u + cols_x + cols_z)
    r2_uaxw_to_y = get_nl_r2(df, ["Y_obs"], cols_u + ["A_probs"] + cols_w + cols_x)

    dag_stats.append(
        {
            "U -> Z": r2_u_to_z,
            "U -> W": r2_u_to_w,
            "U -> X": r2_u_to_x,
            "U+X+Z -> A-probs": r2_uxz_to_a,
            "U+A+W+X -> Y": r2_uaxw_to_y,
        }
    )

dag_df = pd.DataFrame(dag_stats)

#%%

plt.figure(figsize=(14, 7))
dag_df_long = dag_df.melt(var_name="DAG Path", value_name="Non-linear R2")

sns.violinplot(
    data=dag_df_long,
    x="DAG Path",
    y="Non-linear R2",
    palette="Spectral",
    inner=None,  # Remove the inner boxplot to keep it clean
    alpha=0.4,
)

sns.stripplot(
    data=dag_df_long,
    x="DAG Path",
    y="Non-linear R2",
    color="black",
    size=3,
    alpha=0.3,
    jitter=True,
)

plt.title(
    "Distribution of Signal Propagation ($R^2$) Across 50 Datasets",
    fontweight="bold",
    fontsize=15,
)
plt.ylabel("Non-linear $R^2$ (Variance Explained)", fontsize=12)
plt.xlabel("Causal Relationship Path", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Moderate Signal")
plt.axhline(0.1, color="red", linestyle=":", alpha=0.5, label="Noise Floor")

plt.tight_layout()
plt.show()

#%% ---------------------------------------------------------------------
# 8. Bias Analysis: How much of the selection bias can be explained by adjusting for X, W, Z (Proximal) vs. X,U (Oracle)?
# ---------------------------------------------------------------------

# Y1

bias_records = []

for i, df in tqdm(
    enumerate(all_datasets), total=len(all_datasets), desc="Analyzing Confounding"
):
    # Identify column groups
    cols_u = [c for c in df.columns if c.startswith("U")]
    cols_x = [c for c in df.columns if c.startswith("X")]
    cols_proxies = [c for c in df.columns if c.startswith(("W", "Z", "X"))]
    cols_oracle = cols_x + cols_u

    # 1. Ground Truth: True ATE and Potential Outcome Means
    true_mu1 = df["Y1"].mean()

    # 2. Raw Observed Mean (The biased estimate)
    raw_mu1 = df[df["A"] == 1]["Y_obs"].mean()
    raw_bias = abs(raw_mu1 - true_mu1)

    def get_adjusted_bias(features):
        # Train on observed data
        train_df = df[df["A"] == 1]
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(train_df[features], train_df["Y_obs"])

        # Predict for the whole population (Counterfactual)
        pred_y1 = model.predict(df[features])
        return abs(pred_y1.mean() - true_mu1)

    bias_records.append(
        {
            "Raw Selection Bias": raw_bias,
            "Adjusted by X": get_adjusted_bias(cols_x),
            "Adjusted by X,W,Z (Proximal)": get_adjusted_bias(cols_proxies),
            "Adjusted by X,U (Oracle)": get_adjusted_bias(cols_oracle),
        }
    )

bias_df = pd.DataFrame(bias_records)

plt.figure(figsize=(12, 6))
bias_df_long = bias_df.melt(
    var_name="Adjustment Set", value_name="Absolute Bias (Error)"
)

sns.violinplot(
    data=bias_df_long,
    x="Adjustment Set",
    y="Absolute Bias (Error)",
    palette="Reds",
    inner="quartile",
    alpha=0.6,
)
sns.stripplot(
    data=bias_df_long,
    x="Adjustment Set",
    y="Absolute Bias (Error)",
    color="black",
    size=3,
    alpha=0.3,
    jitter=True,
)

plt.title(
    "How Covariates Explain Selection Bias ($|\\hat{\\mu}_1 - \\mu_1|$)",
    fontweight="bold",
    fontsize=15,
)
plt.ylabel("Absolute Error in Mean Outcome", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.show()

# Define the metrics we want to see
metrics = ["mean", "median", "std", "min", "max"]

# Calculate statistics for each adjustment scenario
bias_summary = bias_df.agg(metrics).T

# Rename columns for a professional look in the output
bias_summary.columns = [m.capitalize() for m in metrics]

# Display the table
print("### Confounding & Bias Reduction Metrics ###")
print(bias_summary.to_markdown(floatfmt=".4f"))

# Optional: Percentage of Bias Removed (relative to Raw)
# This helps quantify the "Value Add" of Proxies vs X alone
avg_raw_bias = bias_summary.loc["Raw Selection Bias", "Mean"]
proximal_improvement = (
    1 - (bias_summary.loc["Adjusted by X,W,Z (Proximal)", "Mean"] / avg_raw_bias)
) * 100

print(
    f"\n> **Key Insight:** The Proximal set (X, W, Z) reduced average bias by **{proximal_improvement:.2f}%** compared to the Raw data."
)

# Y0

bias_records = []

for i, df in tqdm(
    enumerate(all_datasets), total=len(all_datasets), desc="Analyzing Confounding"
):
    # Identify column groups
    cols_u = [c for c in df.columns if c.startswith("U")]
    cols_x = [c for c in df.columns if c.startswith("X")]
    cols_proxies = [c for c in df.columns if c.startswith(("W", "Z", "X"))]
    cols_oracle = cols_x + cols_u

    # 1. Ground Truth: True ATE and Potential Outcome Means
    true_mu1 = df["Y0"].mean()

    # 2. Raw Observed Mean (The biased estimate)
    raw_mu1 = df[df["A"] == 0]["Y_obs"].mean()
    raw_bias = abs(raw_mu1 - true_mu1)

    def get_adjusted_bias(features):
        # Train on observed data
        train_df = df[df["A"] == 0]
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(train_df[features], train_df["Y_obs"])

        # Predict for the whole population (Counterfactual)
        pred_y1 = model.predict(df[features])
        return abs(pred_y1.mean() - true_mu1)

    bias_records.append(
        {
            "Raw Selection Bias": raw_bias,
            "Adjusted by X": get_adjusted_bias(cols_x),
            "Adjusted by X,W,Z (Proximal)": get_adjusted_bias(cols_proxies),
            "Adjusted by X,U (Oracle)": get_adjusted_bias(cols_oracle),
        }
    )

bias_df = pd.DataFrame(bias_records)

plt.figure(figsize=(12, 6))
bias_df_long = bias_df.melt(
    var_name="Adjustment Set", value_name="Absolute Bias (Error)"
)

sns.violinplot(
    data=bias_df_long,
    x="Adjustment Set",
    y="Absolute Bias (Error)",
    palette="Reds",
    inner="quartile",
    alpha=0.6,
)
sns.stripplot(
    data=bias_df_long,
    x="Adjustment Set",
    y="Absolute Bias (Error)",
    color="black",
    size=3,
    alpha=0.3,
    jitter=True,
)

plt.title(
    "How Covariates Explain Selection Bias ($|\\hat{\\mu}_0 - \\mu_0|$)",
    fontweight="bold",
    fontsize=15,
)
plt.ylabel("Absolute Error in Mean Outcome", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.show()

# Define the metrics we want to see
metrics = ["mean", "median", "std", "min", "max"]

# Calculate statistics for each adjustment scenario
bias_summary = bias_df.agg(metrics).T

# Rename columns for a professional look in the output
bias_summary.columns = [m.capitalize() for m in metrics]

# Display the table
print("### Confounding & Bias Reduction Metrics ###")
print(bias_summary.to_markdown(floatfmt=".4f"))

# Optional: Percentage of Bias Removed (relative to Raw)
# This helps quantify the "Value Add" of Proxies vs X alone
avg_raw_bias = bias_summary.loc["Raw Selection Bias", "Mean"]
proximal_improvement = (
    1 - (bias_summary.loc["Adjusted by X,W,Z (Proximal)", "Mean"] / avg_raw_bias)
) * 100

print(
    f"\n> **Key Insight:** The Proximal set (X, W, Z) reduced average bias by **{proximal_improvement:.2f}%** compared to the Raw data."
)
