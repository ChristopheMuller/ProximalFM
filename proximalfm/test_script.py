# %% ── 0. Imports & Style ──────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import warnings
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.simplefilter(action="ignore", category=FutureWarning)

# Import your new Proximal Prior Dataset
from prior._dataset import PriorDataset

# ── Aesthetics ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("tab10")
C_X = PALETTE[0]
C_Z = PALETTE[1]
C_W = PALETTE[2]
C_A0 = PALETTE[7]
C_A1 = PALETTE[3]
C_ITE = PALETTE[5]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)

print("✓ Imports OK")


# %% ── 1. Configuration & Generation ─────────────────────────────────────────

print("Initializing Proximal PriorDataset...")
dataset = PriorDataset(
    batch_size=8,
    batch_size_per_gp=4,
    min_dim_U=3,
    max_dim_U=5,
    min_dim_W=4,
    max_dim_W=6,
    min_dim_Z=4,
    max_dim_Z=6,
    min_dim_X=10,
    max_dim_X=15,
    min_seq_len=1000,
    max_seq_len=2000,
    prior_type="mlp_scm",
    device="cpu",
)

print("\nDataset configured:")
print(dataset)

print("\nGenerating a batch of data...")
X, W, Z, A, Y_obs, Y0, Y1, d_X, d_W, d_Z, seq_lens, train_sizes = dataset.get_batch()
print("✓ Batch generated.")

# Extract the first dataset in the batch and remove padding for analysis
b_idx = 0
slen = seq_lens[b_idx].item()

X_act = X[b_idx, :slen, : d_X[b_idx]].numpy()
W_act = W[b_idx, :slen, : d_W[b_idx]].numpy()
Z_act = Z[b_idx, :slen, : d_Z[b_idx]].numpy()
A_act = A[b_idx, :slen].numpy()
Y_obs_act = Y_obs[b_idx, :slen].numpy()
Y0_act = Y0[b_idx, :slen].numpy()
Y1_act = Y1[b_idx, :slen].numpy()
ITE_act = Y1_act - Y0_act

print(
    f"Extracted Dataset 0: {slen} samples. Dims -> X:{d_X[b_idx]}, W:{d_W[b_idx]}, Z:{d_Z[b_idx]}"
)


# %% ── 2. Tensor-Level Inspection (Padding & NaNs) ───────────────────────────


def describe_tensor(name: str, t: np.ndarray) -> pd.Series:
    arr = t.ravel()
    return pd.Series(
        {
            "shape": str(tuple(t.shape)),
            "NaN": int(np.isnan(arr).sum()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        },
        name=name,
    )


tensors = {
    "X_act": X_act,
    "W_act": W_act,
    "Z_act": Z_act,
    "A_act": A_act,
    "Y_obs": Y_obs_act,
    "ITE": ITE_act,
}
summary = pd.DataFrame([describe_tensor(k, v) for k, v in tensors.items()])
print("\n" + summary.to_string())

# Sanity Check Counterfactuals
consistency_error = np.max(np.abs(Y_obs_act - np.where(A_act == 1, Y1_act, Y0_act)))
print(f"\nCounterfactual Consistency Error (should be 0.0): {consistency_error}")
assert consistency_error < 1e-5, "CRITICAL: Y_obs does not match Counterfactuals!"


# %% ── 3. Univariate Distributions & Mixed Noise ──────────────────────────────


def plot_group_kde(ax, arr: np.ndarray, label: str, color):
    for d in range(min(arr.shape[1], 5)):  # Plot up to 5 dims to avoid clutter
        sns.kdeplot(arr[:, d], ax=ax, color=color, alpha=0.5, linewidth=1.6)
    ax.set_title(f"{label} (First 5 Dims)", fontweight="bold")
    ax.set_ylabel("density")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
groups = [
    ("X Covariates", X_act, C_X),
    ("W (Outcome Proxy)", W_act, C_W),
    ("Z (Treatment Proxy)", Z_act, C_Z),
]

for ax, (name, arr, color) in zip(axes, groups):
    plot_group_kde(ax, arr, name, color)

fig.suptitle(
    "Univariate KDE — Check for exploding variances or collapsed features",
    y=1.02,
    fontweight="bold",
)
plt.tight_layout()
plt.show()


# %% ── 4. MLP Functional Diversity Across the Batch ───────────────────────────
#
#   Because generate_dataset instantiates a NEW neural network for every dataset,
#   the manifold mapping should look entirely different across the batch!
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False, sharex=False)

for i in range(4):
    slen_i = seq_lens[i].item()
    w_vals = W[i, :slen_i, 0].numpy()  # First dimension of W
    z_vals = Z[i, :slen_i, 0].numpy()  # First dimension of Z

    axes[i].scatter(w_vals, z_vals, alpha=0.4, s=5, color=C_Z)
    axes[i].set_title(f"Dataset {i} in Batch", fontweight="bold")
    axes[i].set_xlabel("Proxy W_0")
    if i == 0:
        axes[i].set_ylabel("Proxy Z_0")

fig.suptitle(
    "MLP Functional Diversity: $W_0$ vs $Z_0$ manifold changes per dataset",
    y=1.05,
    fontweight="bold",
)
plt.tight_layout()
plt.show()


# %% ── 5. Empirical Treatment Overlap (Positivity) ────────────────────────────
#
#   Since we don't return the true propensity, we fit an empirical propensity
#   score using logistic regression on the observed proxies to check overlap.
# ─────────────────────────────────────────────────────────────────────────────

# Fit empirical propensity score model
XWZ_concat = np.column_stack((X_act, W_act, Z_act))
ps_model = LogisticRegression(max_iter=1000).fit(XWZ_concat, A_act)
ps_hat = ps_model.predict_proba(XWZ_concat)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Treatment Assignment
treat_counts = pd.Series(A_act).value_counts().sort_index()
axes[0].bar(["Control (A=0)", "Treated (A=1)"], treat_counts.values, color=[C_A0, C_A1])
axes[0].set_title("Treatment Assignment", fontweight="bold")

# Estimated Propensity Histogram
axes[1].hist(ps_hat, bins=30, color=C_A1, edgecolor="white", alpha=0.8)
axes[1].set_title("Empirical Propensity $\hat{e}(X,W,Z)$", fontweight="bold")

# Overlap by Arm
for a_val, label, color in [(0, "A=0", C_A0), (1, "A=1", C_A1)]:
    mask = A_act == a_val
    sns.kdeplot(
        ps_hat[mask], ax=axes[2], label=label, color=color, fill=True, alpha=0.3
    )
axes[2].set_title("Propensity Overlap by Arm", fontweight="bold")
axes[2].legend(frameon=False)

fig.suptitle(
    "Positivity Analysis via Empirical Propensity Scores", y=1.02, fontweight="bold"
)
plt.tight_layout()
plt.show()


# %% ── 6. Outcome Distributions & Heterogeneity ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for arr, label, color in [(Y0_act, "Y(0)", C_A0), (Y1_act, "Y(1)", C_A1)]:
    sns.kdeplot(arr, ax=axes[0], label=label, color=color, fill=True, alpha=0.3)
axes[0].set_title("Potential Outcomes Y(0) and Y(1)", fontweight="bold")
axes[0].legend(frameon=False)

sns.kdeplot(ITE_act, ax=axes[1], color=C_ITE, fill=True, alpha=0.5)
axes[1].axvline(ITE_act.mean(), color="black", linestyle="--")
axes[1].set_title("True Individual Treatment Effect (CATE)", fontweight="bold")

ite_sign = (ITE_act >= 0).astype(int)
axes[2].scatter(
    Y0_act, Y1_act, c=ite_sign, cmap="bwr_r", alpha=0.3, s=8, rasterized=True
)
lims = [min(Y0_act.min(), Y1_act.min()), max(Y0_act.max(), Y1_act.max())]
axes[2].plot(lims, lims, "k--", linewidth=1, alpha=0.5)
axes[2].set_title("Y(0) vs Y(1)", fontweight="bold")

fig.suptitle(
    "Outcome Analysis (Generated by Structural MLPs)", y=1.02, fontweight="bold"
)
plt.tight_layout()
plt.show()


# %% ── 7. Predictability of CATE (Information Progression) ────────────────────
#
#   How much does the model benefit from observing the proxies?
#   We evaluate predictability of the True CATE by progressively adding features.
# ─────────────────────────────────────────────────────────────────────────────

print("Training Oracles for Information Progression...")

# 1. Naive S-Learner RF (Predict Y from X and A, ignoring W and Z)
XA_np = np.column_stack((X_act, A_act))
s_learner = RandomForestRegressor(
    n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
).fit(XA_np, Y_obs_act)
pred_slearner = s_learner.predict(
    np.column_stack((X_act, np.ones_like(A_act)))
) - s_learner.predict(np.column_stack((X_act, np.zeros_like(A_act))))
r2_slearner = r2_score(ITE_act, pred_slearner)

# 2. Oracle RF on Baseline X (Predict True CATE directly)
rf_x = RandomForestRegressor(
    n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
).fit(X_act, ITE_act)
pred_x = rf_x.predict(X_act)
r2_x = r2_score(ITE_act, pred_x)

# 3. Oracle RF on X + Treatment Proxy Z
XZ_np = np.column_stack((X_act, Z_act))
rf_xz = RandomForestRegressor(
    n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
).fit(XZ_np, ITE_act)
pred_xz = rf_xz.predict(XZ_np)
r2_xz = r2_score(ITE_act, pred_xz)

# 4. Oracle RF on X + Outcome Proxy W
XW_np = np.column_stack((X_act, W_act))
rf_xw = RandomForestRegressor(
    n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
).fit(XW_np, ITE_act)
pred_xw = rf_xw.predict(XW_np)
r2_xw = r2_score(ITE_act, pred_xw)

# 5. Oracle RF on ALL Observables (X, W, Z)
XWZ_np = np.column_stack((X_act, W_act, Z_act))
rf_xwz = RandomForestRegressor(
    n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
).fit(XWZ_np, ITE_act)
pred_xwz = rf_xwz.predict(XWZ_np)
r2_xwz = r2_score(ITE_act, pred_xwz)

# ── Plotting ──
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True, sharex=True)

plots = [
    (pred_slearner, r2_slearner, "1. Naive S-Learner\n(X, A → Y_obs)", C_A1),
    (pred_x, r2_x, "2. Oracle RF\n(X → True CATE)", C_ITE),
    (pred_xz, r2_xz, "3. Oracle RF\n(X, Z → True CATE)", "#1D9E75"),
    (pred_xw, r2_xw, "4. Oracle RF\n(X, W → True CATE)", "#7F77DD"),
    (pred_xwz, r2_xwz, "5. Oracle RF All\n(X, W, Z → True CATE)", "#D85A30"),
]

lims = [ITE_act.min(), ITE_act.max()]

for ax, (preds, r2, title, color) in zip(axes, plots):
    ax.scatter(ITE_act, preds, alpha=0.3, s=10, color=color, rasterized=True)
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect Prediction")
    ax.set_title(f"{title}\nCATE R² = {r2:.3f}", fontweight="bold", fontsize=11)
    ax.set_xlabel("True CATE")
    if ax == axes[0]:
        ax.set_ylabel("Predicted CATE")
    ax.legend(fontsize=9, frameon=False, loc="upper left")

fig.suptitle(
    "Information Progression: Value of Proxies in Learning the CATE Manifold",
    fontweight="bold",
    fontsize=14,
    y=1.08,
)
plt.tight_layout()
plt.show()

print("\n── CATE Predictability Progression (R²) ──")
print(
    f"1. Naive S-Learner (X, A)     : {r2_slearner:.3f}  <- Baseline biased by latent U"
)
print(f"2. Oracle RF (X)              : {r2_x:.3f}  <- Fit without proxies")
print(f"3. Oracle RF (X, Z)           : {r2_xz:.3f}  <- Adding treatment proxy")
print(f"4. Oracle RF (X, W)           : {r2_xw:.3f}  <- Adding outcome proxy")
print(
    f"5. Oracle RF (X, W, Z)        : {r2_xwz:.3f}  <- Best possible fit using all observables"
)
