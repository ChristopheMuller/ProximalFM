# =============================================================================
#  Proximal Prior — Exploratory Visualization Script
#  Run cell-by-cell in VSCode (Python Interactive / Jupyter mode)
# =============================================================================
# Each #%% block is a self-contained "chapter".
# Run them in order: later chapters build on objects created earlier.
# =============================================================================

#%% ── 0. Imports & Style ──────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from scipy import stats

# ── Aesthetics ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE   = sns.color_palette("tab10")
C_X       = PALETTE[0]   # blue  — covariates X
C_Z       = PALETTE[1]   # orange— treatment proxy Z
C_W       = PALETTE[2]   # green — outcome proxy W
C_U       = PALETTE[4]   # purple— latent confounder U
C_A0      = PALETTE[7]   # grey  — control
C_A1      = PALETTE[3]   # red   — treated
C_ITE     = PALETTE[5]   # brown — ITE / causal estimands

plt.rcParams.update({
    "figure.dpi"        : 120,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.facecolor"  : "white",
})

print("✓ Imports OK")


#%% ── 1. Configuration & Single-Episode Generation ───────────────────────────
#
#   We instantiate ProximalPriorConfig with explicit, moderate dimensions so
#   every plot is readable.  Adjust freely.
# ─────────────────────────────────────────────────────────────────────────────

from proximalfm.Structure   import ProximalPriorConfig   # ← adjust to your package path
from proximalfm.PriorLinear import ProximalPrior
# from proximalfm.Prior import ProximalPrior

from proximalfm.Structure import NetworkSpec

cfg = ProximalPriorConfig(
    d_x            = 5,
    d_u            = 1,
    d_z            = 2,
    d_w            = 2,
    noise_scale_x  = 0.10,
    noise_scale_z  = 0.15,
    noise_scale_w  = 0.05,
    noise_scale_y  = 0.10,
    normalize_episode = False,
    temperature=3,
)

prior = ProximalPrior(config=cfg, seed=0)

N_OBS = 2_000   # observations per episode
episode = prior.generate_episode(batch_size=N_OBS, episode_seed=42)

print("Episode generated successfully.")
print(f"  X  : {tuple(episode.X.shape)}")
print(f"  Z  : {tuple(episode.Z.shape)}")
print(f"  W  : {tuple(episode.W.shape)}")
print(f"  U  : {tuple(episode.U.shape)}")
print(f"  A  : {tuple(episode.A.shape)}")
print(f"  Y  : {tuple(episode.Y.shape)}")
print(f"  Y0 : {tuple(episode.Y0.shape)}")
print(f"  Y1 : {tuple(episode.Y1.shape)}")
print(f"  propensity: {tuple(episode.propensity.shape)}")


#%% ── 2. Tensor-Level Inspection (dtypes, ranges, NaNs) ──────────────────────

def describe_tensor(name: str, t: torch.Tensor) -> pd.Series:
    arr = t.cpu().float().numpy().ravel()
    return pd.Series({
        "shape"  : str(tuple(t.shape)),
        "dtype"  : str(t.dtype),
        "NaN"    : int(np.isnan(arr).sum()),
        "Inf"    : int(np.isinf(arr).sum()),
        "min"    : float(arr.min()),
        "max"    : float(arr.max()),
        "mean"   : float(arr.mean()),
        "std"    : float(arr.std()),
    }, name=name)

tensors = {
    "X" : episode.X,  "Z" : episode.Z,  "W" : episode.W,
    "U" : episode.U,  "A" : episode.A,
    "Y" : episode.Y,  "Y0": episode.Y0, "Y1": episode.Y1,
    "propensity": episode.propensity,
    "ITE": episode.Y1 - episode.Y0,
}

summary = pd.DataFrame([describe_tensor(k, v) for k, v in tensors.items()])
print(summary.to_string())


#%% ── 3. Univariate Distributions — Continuous Variables ─────────────────────
#
#   One subplot per variable-group (X, Z, W, U).
#   Each dimension of the group is overlaid on the same axis.
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_kde(ax, tensor: torch.Tensor, label: str, color, alpha: float = 0.6):
    """Overlay KDE for every dimension of a 2-D tensor on a single axis."""
    arr = tensor.cpu().float().numpy()
    for d in range(arr.shape[1]):
        sns.kdeplot(arr[:, d], ax=ax, color=color, alpha=alpha, linewidth=1.6,
                    label=f"{label}_{d}")
    ax.set_title(label, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("value")
    ax.set_ylabel("density")

groups   = [("X", episode.X, C_X), ("Z", episode.Z, C_Z),
            ("W", episode.W, C_W), ("U", episode.U, C_U)]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, tensor, color) in zip(axes, groups):
    plot_group_kde(ax, tensor, name, color)

fig.suptitle("Univariate KDE — Continuous Variables (one episode)", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()


#%% ── 4. Treatment Distribution — Propensity & Assignment ────────────────────

A_np   = episode.A.cpu().numpy()
ps_np  = episode.propensity.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 4a. Propensity score histogram
axes[0].hist(ps_np, bins=40, color=C_A1, edgecolor="white", alpha=0.8)
axes[0].axvline(ps_np.mean(), color="black", linestyle="--", label=f"mean={ps_np.mean():.3f}")
axes[0].set_title("Propensity Score e(X, U, Z)", fontweight="bold")
axes[0].set_xlabel("P(A=1 | X, U, Z)")
axes[0].set_ylabel("count")
axes[0].legend(frameon=False)

# 4b. Treatment bar
treat_counts = pd.Series(A_np).value_counts().sort_index()
axes[1].bar(["Control (A=0)", "Treated (A=1)"],
            treat_counts.values,
            color=[C_A0, C_A1], edgecolor="white")
for bar, val in zip(axes[1].patches, treat_counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"{val} ({val/N_OBS*100:.1f}%)", ha="center", va="bottom", fontsize=10)
axes[1].set_title("Treatment Assignment", fontweight="bold")
axes[1].set_ylabel("count")

# 4c. Propensity by treatment arm — overlap check
for a_val, label, color in [(0, "A=0", C_A0), (1, "A=1", C_A1)]:
    mask = A_np == a_val
    sns.kdeplot(ps_np[mask], ax=axes[2], label=label, color=color, fill=True, alpha=0.3)
axes[2].set_title("Propensity Overlap by Arm", fontweight="bold")
axes[2].set_xlabel("P(A=1 | X, U, Z)")
axes[2].legend(frameon=False)

fig.suptitle("Treatment & Propensity", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"Treatment rate : {A_np.mean():.4f}")
print(f"Propensity — min: {ps_np.min():.4f}  max: {ps_np.max():.4f}  "
      f"mean: {ps_np.mean():.4f}  std: {ps_np.std():.4f}")
print(f"Temperature (propensity scaling) : {cfg.temperature}")


#%% ── 5. Outcome Distributions — Y, Y0, Y1, ITE ─────────────────────────────

Y_np   = episode.Y.cpu().numpy()
Y0_np  = episode.Y0.cpu().numpy()
Y1_np  = episode.Y1.cpu().numpy()
ITE_np = (episode.Y1 - episode.Y0).cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# 5a. Observed outcome by arm
for a_val, label, color in [(0, "A=0", C_A0), (1, "A=1", C_A1)]:
    mask = A_np == a_val
    sns.kdeplot(Y_np[mask], ax=axes[0], label=label, color=color, fill=True, alpha=0.3)
axes[0].set_title("Observed Y by Treatment Arm", fontweight="bold")
axes[0].set_xlabel("Y")
axes[0].legend(frameon=False)

# 5b. Potential outcomes
for arr, label, color in [(Y0_np, "Y(0)", C_A0), (Y1_np, "Y(1)", C_A1)]:
    sns.kdeplot(arr, ax=axes[1], label=label, color=color, fill=True, alpha=0.3)
axes[1].set_title("Potential Outcomes Y(0) and Y(1)", fontweight="bold")
axes[1].set_xlabel("value")
axes[1].legend(frameon=False)

# 5c. ITE distribution
try:
    axes[2].hist(ITE_np, bins=50, color=C_ITE, edgecolor="white", alpha=0.85)
except ValueError:
    print("Warning: histogram failed (possibly due to NaNs or Infs in ITE). "
          "Falling back to KDE plot.")
    sns.kdeplot(ITE_np, ax=axes[2], color=C_ITE, fill=True, alpha=0.3)
axes[2].axvline(ITE_np.mean(), color="black", linestyle="--",
                label=f"ATE={ITE_np.mean():.3f}")
axes[2].axvline(0, color="red", linestyle=":", alpha=0.5, label="zero")
axes[2].set_title("Individual Treatment Effect (ITE = Y1−Y0)", fontweight="bold")
axes[2].set_xlabel("ITE")
axes[2].set_ylabel("count")
axes[2].legend(frameon=False)

# 5d. Y0 vs Y1 scatter (colour = ITE sign)
ite_sign = (ITE_np >= 0).astype(int)
axes[3].scatter(Y0_np, Y1_np, c=ite_sign, cmap="bwr_r",
                alpha=0.3, s=8, rasterized=True)
lims = [min(Y0_np.min(), Y1_np.min()), max(Y0_np.max(), Y1_np.max())]
axes[3].plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Y0=Y1 (ATE=0)")
axes[3].set_title("Y(0) vs Y(1)  (red = negative ITE)", fontweight="bold")
axes[3].set_xlabel("Y(0)")
axes[3].set_ylabel("Y(1)")
axes[3].legend(fontsize=8, frameon=False)

fig.suptitle("Outcome Analysis", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"ATE (mean ITE)   : {ITE_np.mean():.4f}")
print(f"ITE std          : {ITE_np.std():.4f}")
print(f"Fraction ITE > 0 : {(ITE_np > 0).mean():.4f}")


#%% ── 6. Pairwise Correlations — Full Variable Block ─────────────────────────
#
#   Build a flat DataFrame with all observed variables (X, Z, W, A, Y).
#   U is latent in practice but included here for oracle analysis.
# ─────────────────────────────────────────────────────────────────────────────

df = prior.episode_to_dataframe(episode, include_latents=True)
print("DataFrame columns:", df.columns.tolist())
print(df.describe().round(3).to_string())


# ── 6b. Correlation Heatmap ────────────────────────────────────────────────

# Separate variable groups for block-ordering
x_cols  = [c for c in df.columns if c.startswith("x_")]
z_cols  = [c for c in df.columns if c.startswith("z_")]
w_cols  = [c for c in df.columns if c.startswith("w_")]
u_cols  = [c for c in df.columns if c.startswith("u_")]
other   = ["treatment", "outcome", "y0", "y1", "propensity"]
ordered = x_cols + z_cols + w_cols + u_cols + other

corr = df[ordered].corr()

# Build a colour-band mask to separate groups visually
n = len(ordered)
group_sizes = [len(x_cols), len(z_cols), len(w_cols), len(u_cols), len(other)]
group_labels = ["X", "Z", "W", "U", "outcomes"]

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.zeros_like(corr, dtype=bool)   # show everything

sns.heatmap(
    corr, ax=ax, mask=mask,
    cmap="RdBu_r", vmin=-1, vmax=1, center=0,
    linewidths=0.3, linecolor="white",
    annot=True, fmt=".2f", annot_kws={"size": 7},
    square=True, cbar_kws={"shrink": 0.6},
)

# Draw group separator lines
cumulative = 0
for size in group_sizes[:-1]:
    cumulative += size
    ax.axhline(cumulative, color="black", linewidth=1.5)
    ax.axvline(cumulative, color="black", linewidth=1.5)

ax.set_title("Pairwise Pearson Correlations  (block-ordered by variable group)",
             fontweight="bold", pad=14)
plt.tight_layout()
plt.show()


#%% ── 7. Proxy Quality — Z as Treatment Proxy, W as Outcome Proxy ────────────
#
#   In proximal causal inference:
#     • Z must be associated with U  (and thus with A) but independent of Y | U, X, A
#     • W must be associated with U  (and thus with Y) but independent of A | U, X
#
#   Here we check empirical correlations and scatter plots against each U dimension.
# ─────────────────────────────────────────────────────────────────────────────

U_np = episode.U.cpu().numpy()
Z_np = episode.Z.cpu().numpy()
W_np = episode.W.cpu().numpy()

def corr_matrix(A: np.ndarray, B: np.ndarray,
                row_labels: list, col_labels: list) -> pd.DataFrame:
    """Compute Pearson r between every pair of columns from A and B."""
    return pd.DataFrame(
        [[float(np.corrcoef(A[:, i], B[:, j])[0, 1])
          for j in range(B.shape[1])]
         for i in range(A.shape[1])],
        index=row_labels, columns=col_labels,
    )

u_labels = [f"U_{i}" for i in range(U_np.shape[1])]
z_labels = [f"Z_{i}" for i in range(Z_np.shape[1])]
w_labels = [f"W_{i}" for i in range(W_np.shape[1])]

corr_UZ = corr_matrix(U_np, Z_np, u_labels, z_labels)
corr_UW = corr_matrix(U_np, W_np, u_labels, w_labels)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, data, title in [
    (axes[0], corr_UZ, "U ↔ Z  (treatment proxy should correlate with U)"),
    (axes[1], corr_UW, "U ↔ W  (outcome proxy should correlate with U)"),
]:
    sns.heatmap(data, ax=ax, cmap="PiYG", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontweight="bold", pad=10)

fig.suptitle("Proxy–Confounder Correlation Matrices", y=1.04, fontweight="bold")
plt.tight_layout()
plt.show()


#%% ── 7b. U → Z / U → W Scatter Panels ──────────────────────────────────────

def scatter_panel(fig, gs_row, tensor_a, tensor_b,
                  name_a, name_b, color, alpha=0.25, s=6):
    """Scatter every dimension of tensor_a against every dim of tensor_b."""
    d_a, d_b = tensor_a.shape[1], tensor_b.shape[1]
    ax_arr = []
    for i in range(d_a):
        for j in range(d_b):
            ax = fig.add_subplot(gs_row[i, j])
            ax.scatter(tensor_a[:, i], tensor_b[:, j],
                       alpha=alpha, s=s, color=color, rasterized=True)
            r, p = stats.pearsonr(tensor_a[:, i], tensor_b[:, j])
            ax.set_title(f"r={r:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{name_a}_{i}", fontsize=8)
            if i == d_a - 1:
                ax.set_xlabel(f"{name_b}_{j}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax_arr.append(ax)
    return ax_arr

d_u = U_np.shape[1]
d_z = Z_np.shape[1]
d_w = W_np.shape[1]

fig = plt.figure(figsize=(5 * d_z + 1, 4 * d_u * 2 + 1))
gs_top = gridspec.GridSpec(d_u, d_z, figure=fig,
                            top=0.95, bottom=0.55, hspace=0.5, wspace=0.4)
gs_bot = gridspec.GridSpec(d_u, d_w, figure=fig,
                            top=0.45, bottom=0.05, hspace=0.5, wspace=0.4)

scatter_panel(fig, gs_top, U_np, Z_np, "U", "Z", C_Z)
scatter_panel(fig, gs_bot, U_np, W_np, "U", "W", C_W)

fig.text(0.5, 0.97, "Latent U  vs  Treatment Proxy Z",
         ha="center", fontweight="bold", fontsize=13)
fig.text(0.5, 0.47, "Latent U  vs  Outcome Proxy W",
         ha="center", fontweight="bold", fontsize=13)
plt.show()


#%% ── 8. Causal Graph Empirical Checks ───────────────────────────────────────
#
#   Verify key d-separation / marginal independence claims:
#
#   Expected STRONG correlations
#     Z ↔ A   (Z is a treatment-side proxy → influences / is influenced by U → A)
#     W ↔ Y   (W is an outcome-side proxy  → associated with Y through U)
#     U ↔ A   (direct confounding path)
#     U ↔ Y
#
#   Expected WEAKER / NO direct edge
#     Z ↔ Y | A, X, U   (Z excluded from outcome equation)
#     W ↔ A | X, U      (W excluded from treatment equation)
# ─────────────────────────────────────────────────────────────────────────────

# Flatten multi-dim variables to their first principal component for 1-D plots
def first_pc(arr: np.ndarray) -> np.ndarray:
    centered = arr - arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[0]

z_pc = first_pc(Z_np)
w_pc = first_pc(W_np)
u_pc = first_pc(U_np)

pairs = [
    (z_pc,  A_np, "Z (PC1)", "A",        "Z ↔ A  [expected: strong]", C_Z),
    (w_pc,  Y_np, "W (PC1)", "Y",        "W ↔ Y  [expected: strong]", C_W),
    (u_pc,  A_np, "U (PC1)", "A",        "U ↔ A  [expected: strong]", C_U),
    (u_pc,  Y_np, "U (PC1)", "Y",        "U ↔ Y  [expected: strong]", C_U),
    (z_pc,  Y_np, "Z (PC1)", "Y",        "Z ↔ Y  [expected: weaker]", C_Z),
    (w_pc,  A_np, "W (PC1)", "A",        "W ↔ A  [expected: weaker]", C_W),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (x, y, xlab, ylab, title, color) in zip(axes.ravel(), pairs):
    ax.scatter(x, y, alpha=0.2, s=5, color=color, rasterized=True)
    r, p = stats.pearsonr(x, y.ravel())
    ax.set_title(f"{title}\nr={r:.3f}, p={p:.2e}", fontsize=9, fontweight="bold")
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    m, b = np.polyfit(x, y.ravel(), 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, "k--", linewidth=1.5, alpha=0.7)

fig.suptitle("Causal Graph Empirical Checks (first PC per group)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


#%% ── 9. Conditional Treatment Effect (CATE) vs Covariates ──────────────────
#
#   Bin observations by each X dimension and compute the average ITE per bin.
#   This gives a non-parametric view of how heterogeneous the treatment effect is.
# ─────────────────────────────────────────────────────────────────────────────

X_np = episode.X.cpu().numpy()

fig, axes = plt.subplots(1, cfg.d_x, figsize=(5 * cfg.d_x, 4))
for i, ax in enumerate(axes):
    xi      = X_np[:, i]
    bins    = np.quantile(xi, np.linspace(0, 1, 21))   # 20 equal-frequency bins
    bin_idx = np.digitize(xi, bins[1:-1])
    cate_by_bin = [ITE_np[bin_idx == b].mean() for b in range(20)]
    bin_centers = [(bins[b] + bins[b + 1]) / 2 for b in range(20)]

    ax.bar(bin_centers, cate_by_bin,
           width=(bins[-1] - bins[0]) / 22,
           color=C_ITE, edgecolor="white", alpha=0.8)
    ax.axhline(ITE_np.mean(), color="black", linestyle="--",
               linewidth=1.2, label=f"ATE={ITE_np.mean():.3f}")
    ax.axhline(0, color="red", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(f"X_{i}", fontsize=10)
    ax.set_ylabel("E[ITE | X bin]", fontsize=10)
    ax.set_title(f"CATE vs X_{i}", fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

fig.suptitle("Conditional Average Treatment Effect (CATE) by Covariate Bin",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


#%% ── 10. Cross-Episode Stability ────────────────────────────────────────────
#
#   Generate K episodes and track key statistics across them.
#   This reveals how much the prior varies from episode to episode.
# ─────────────────────────────────────────────────────────────────────────────

K       = 50
N_SMALL = 500    # smaller n to keep it fast

records = []
for k in range(K):
    ep = prior.generate_episode(batch_size=N_SMALL, episode_seed=k)
    ite_k = (ep.Y1 - ep.Y0).cpu().numpy()
    ps_k  = ep.propensity.cpu().numpy()
    records.append({
        "episode"       : k,
        "ATE"           : float(ite_k.mean()),
        "ITE_std"       : float(ite_k.std()),
        "treat_rate"    : float(ep.A.mean().item()),
        "ps_mean"       : float(ps_k.mean()),
        "ps_std"        : float(ps_k.std()),
        "Y_std"         : float(ep.Y.std().item()),
    })

edf = pd.DataFrame(records)

metrics = ["ATE", "ITE_std", "treat_rate", "ps_mean", "ps_std", "Y_std"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

for ax, metric in zip(axes.ravel(), metrics):
    ax.plot(edf["episode"], edf[metric], linewidth=1.0, alpha=0.7, color=C_ITE)
    ax.axhline(edf[metric].mean(), color="black", linestyle="--",
               linewidth=1.2, label=f"μ={edf[metric].mean():.3f}")
    ax.fill_between(
        edf["episode"],
        edf[metric].mean() - edf[metric].std(),
        edf[metric].mean() + edf[metric].std(),
        alpha=0.12, color=C_ITE,
    )
    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel("episode index")
    ax.legend(fontsize=8, frameon=False)

fig.suptitle(f"Cross-Episode Stability  ({K} episodes, N={N_SMALL} each)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print(edf.describe().round(4).to_string())


#%% ── 11. DataLoader Batch Inspection ────────────────────────────────────────
#
#   Use the full PyTorch DataLoader pipeline (custom_collate, seed isolation)
#   and inspect the shape and statistics of one real training batch.
# ─────────────────────────────────────────────────────────────────────────────

from proximalfm.PriorDataLoader import create_causal_data_loaders    # ← adjust path

loaders = create_causal_data_loaders(
    prior_config         = cfg,
    batch_size           = 8,
    samples_per_episode  = 200,
    train_episodes       = 40,
    val_episodes         = 10,
    num_workers          = 0,      # 0 = debug-friendly (no multiprocessing)
    seed                 = 0,
)

batch = next(iter(loaders["train"]))

print("── Batch shape report (N=samples, B=batch_size, D=dim) ──────────────")
for key, tensor in batch.items():
    print(f"  {key:6s} : {str(tuple(tensor.shape)):20s}  "
          f"dtype={tensor.dtype}  "
          f"min={tensor.min():.3f}  max={tensor.max():.3f}")


#%% ── 11b. Batch Heatmap — N×B for one variable ─────────────────────────────
#
#   Visualise the (N, B) matrix for Y and ITE to confirm each column
#   (episode) has its own statistical fingerprint.
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, cmap in [(axes[0], "Y", "coolwarm"), (axes[1], "ITE", "PiYG")]:
    data = batch[key].squeeze(-1).numpy()   # (N, B)
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(f"{key}  —  shape {data.shape}  (rows=obs, cols=episodes)",
                 fontweight="bold")
    ax.set_xlabel("Episode in batch")
    ax.set_ylabel("Observation index")

fig.suptitle("DataLoader Batch Heatmap", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


#%% ── 12. Train / Val Distribution Comparison ────────────────────────────────
#
#   Sanity check: train and val batches should look statistically equivalent
#   (they come from non-overlapping episode seeds but the same prior).
# ─────────────────────────────────────────────────────────────────────────────

train_batch = next(iter(loaders["train"]))
val_batch   = next(iter(loaders["val"]))

keys_to_check = ["Y", "ITE", "A"]
fig, axes = plt.subplots(1, len(keys_to_check), figsize=(14, 4))

for ax, key in zip(axes, keys_to_check):
    t_arr = train_batch[key].numpy().ravel()
    v_arr = val_batch[key].numpy().ravel()
    bins = np.linspace(min(t_arr.min(), v_arr.min()),
                       max(t_arr.max(), v_arr.max()), 40)
    ax.hist(t_arr, bins=bins, alpha=0.5, label="train", color=C_A1,  density=True)
    ax.hist(v_arr, bins=bins, alpha=0.5, label="val",   color=C_A0, density=True)
    ks_stat, ks_p = stats.ks_2samp(t_arr, v_arr)
    ax.set_title(f"{key}  (KS stat={ks_stat:.3f}, p={ks_p:.3f})", fontweight="bold")
    ax.set_xlabel(key)
    ax.set_ylabel("density")
    ax.legend(frameon=False)

fig.suptitle("Train vs Val Batch Distributions  (should be similar, p>0.05 OK)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


#%% ── 13. Summary Dashboard ──────────────────────────────────────────────────
#
#   A single-figure summary meant to be saved / shared.
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# ── Row 0: per-group KDE ──────────────────────────────────────────────────
for col, (name, tensor, color) in enumerate(
    [("X", episode.X, C_X), ("Z", episode.Z, C_Z),
     ("W", episode.W, C_W), ("U", episode.U, C_U)]
):
    ax = fig.add_subplot(gs[0, col])
    arr = tensor.cpu().float().numpy()
    for d in range(arr.shape[1]):
        sns.kdeplot(arr[:, d], ax=ax, color=color, alpha=0.7, linewidth=1.5)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("value"); ax.set_ylabel("density")

# ── Row 1: propensity / ITE / potential outcomes / CATE X_0 ──────────────
ax = fig.add_subplot(gs[1, 0])
ax.hist(ps_np, bins=30, color=C_A1, edgecolor="white", alpha=0.8)
ax.set_title("Propensity Score", fontweight="bold"); ax.set_xlabel("P(A=1)")

ax = fig.add_subplot(gs[1, 1])
ax.hist(ITE_np, bins=40, color=C_ITE, edgecolor="white", alpha=0.85)
ax.axvline(ITE_np.mean(), color="k", linestyle="--",
           label=f"ATE={ITE_np.mean():.3f}")
ax.set_title("ITE Distribution", fontweight="bold")
ax.set_xlabel("Y(1)−Y(0)"); ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[1, 2])
for arr, label, color in [(Y0_np, "Y(0)", C_A0), (Y1_np, "Y(1)", C_A1)]:
    sns.kdeplot(arr, ax=ax, label=label, color=color, fill=True, alpha=0.3)
ax.set_title("Potential Outcomes", fontweight="bold"); ax.legend(frameon=False)

ax = fig.add_subplot(gs[1, 3])
xi      = X_np[:, 0]
bins    = np.quantile(xi, np.linspace(0, 1, 16))
bin_idx = np.digitize(xi, bins[1:-1])
cate_b  = [ITE_np[bin_idx == b].mean() for b in range(15)]
bc      = [(bins[b] + bins[b + 1]) / 2 for b in range(15)]
ax.bar(bc, cate_b, width=(bins[-1]-bins[0])/17, color=C_ITE, edgecolor="white", alpha=0.8)
ax.axhline(ITE_np.mean(), color="k", linestyle="--")
ax.set_title("CATE vs X_0", fontweight="bold"); ax.set_xlabel("X_0")

# ── Row 2: cross-episode ATE / treat_rate / proxy correlation heatmaps ───
ax = fig.add_subplot(gs[2, 0])
ax.plot(edf["episode"], edf["ATE"],        color=C_ITE, linewidth=1.2)
ax.axhline(edf["ATE"].mean(), color="k", linestyle="--",
           label=f"μ={edf['ATE'].mean():.3f}")
ax.set_title("ATE across Episodes", fontweight="bold")
ax.set_xlabel("episode"); ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[2, 1])
ax.plot(edf["episode"], edf["treat_rate"], color=C_A1, linewidth=1.2)
ax.axhline(edf["treat_rate"].mean(), color="k", linestyle="--",
           label=f"μ={edf['treat_rate'].mean():.3f}")
ax.set_title("Treatment Rate across Episodes", fontweight="bold")
ax.set_xlabel("episode"); ax.legend(fontsize=8, frameon=False)

ax = fig.add_subplot(gs[2, 2])
sns.heatmap(corr_UZ, ax=ax, cmap="PiYG", vmin=-1, vmax=1,
            annot=True, fmt=".2f", cbar=False, linewidths=0.5)
ax.set_title("U ↔ Z correlation", fontweight="bold")

ax = fig.add_subplot(gs[2, 3])
sns.heatmap(corr_UW, ax=ax, cmap="PiYG", vmin=-1, vmax=1,
            annot=True, fmt=".2f", cbar=False, linewidths=0.5)
ax.set_title("U ↔ W correlation", fontweight="bold")

fig.suptitle("ProximalPrior — Summary Dashboard", fontsize=16,
             fontweight="bold", y=0.995)
plt.show()
print("✓ Dashboard")