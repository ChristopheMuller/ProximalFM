#%% ── 0. Imports & Style ──────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from scipy import stats

## choose the prior to investigate:

# from proximalfm.PriorLinear import ProximalPrior
# from proximalfm.Prior import ProximalPrior
from proximalfm.PriorLinearConfounding import ProximalPrior

# ── Aesthetics ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE   = sns.color_palette("tab10")
C_X       = PALETTE[0]
C_Z       = PALETTE[1]
C_W       = PALETTE[2]
C_U       = PALETTE[4]
C_A0      = PALETTE[7]
C_A1      = PALETTE[3]
C_ITE     = PALETTE[5]

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

from proximalfm.Structure   import ProximalPriorConfig

cfg = ProximalPriorConfig(
    d_x            = 1,
    d_u            = 2,
    d_z            = 2,
    d_w            = 2,
    noise_scale_w  = 1,
    noise_scale_z  = 1,
    noise_scale_x  = 1,
    temperature=2,
    confounding_strength=1
)

prior = ProximalPrior(config=cfg, seed=np.random.randint(0, 9999))

N_OBS = 5_000   # observations per episode
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
Prop_np = episode.propensity.cpu().numpy()
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
print(f"Diff in means Y|A=1 - Y|A=0 : {Y_np[A_np==1].mean() - Y_np[A_np==0].mean():.4f}")
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
    # (z_pc,  A_np, "Z (PC1)", "A",        "Z ↔ A  [expected: strong]", C_Z),
    (z_pc,  Prop_np, "Z (PC1)", "Prop",        "Z ↔ Prop(A)  [expected: strong]", C_Z),
    (w_pc,  Y_np, "W (PC1)", "Y",        "W ↔ Y  [expected: strong]", C_W),
    # (u_pc,  A_np, "U (PC1)", "A",        "U ↔ A  [expected: strong]", C_U),
    (u_pc,  Prop_np, "U (PC1)", "Prop",        "U ↔ Prop(A)  [expected: strong]", C_U),
    (u_pc,  Y_np, "U (PC1)", "Y",        "U ↔ Y  [expected: strong]", C_U),
    (z_pc,  Y_np, "Z (PC1)", "Y",        "Z ↔ Y  [expected: weaker]", C_Z),
    # (w_pc,  A_np, "W (PC1)", "A",        "W ↔ A  [expected: weaker]", C_W),
    (w_pc,  Prop_np, "W (PC1)", "Prop",        "W ↔ Prop(A)  [expected: weaker]", C_W),
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
if cfg.d_x == 1:
    axes = [axes]
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
try:
    ax.hist(ITE_np, bins=40, color=C_ITE, edgecolor="white", alpha=0.85)
    ax.axvline(ITE_np.mean(), color="k", linestyle="--",
               label=f"ATE={ITE_np.mean():.3f}")
    ax.set_title("ITE Distribution", fontweight="bold")
    ax.set_xlabel("Y(1)−Y(0)"); ax.legend(fontsize=8, frameon=False)
except Exception as e:
    print(f"Error plotting ITE Distribution: {e}")
    sns.kdeplot(ITE_np, ax=ax, color=C_ITE, fill=True, alpha=0.3)


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

# %%===========================================================================
#  CHAPTER 14 — Basic CATE Estimation Strategies
#
#  Four estimators of increasing sophistication, all evaluated against the
#  oracle ITE = Y(1) - Y(0) stored in the episode.
#
#  Ground truth available because ProximalPrior gives us both potential outcomes.
#
#  Estimators
#  ──────────
#  M1  Difference-in-means          → constant ATE, ignores confounding & X
#  M2  OLS: Y ~ X + A               → constant causal coefficient, no proxy
#  M3  OLS: Y ~ X + A + W + Z       → naïve inclusion of proxies as controls
#  M4  P2SLS (Proximal 2SLS)        → principled linear proximal estimator
#         Stage 1 : Â  = OLS(A  ~ X, Z, W)   — Z instruments, W controls
#         Stage 2 : Y  = OLS(Y  ~ Â, X, W)   — coefficient on Â is causal
#
#  Metrics reported per estimator
#  ──────────────────────────────
#  • ATE bias   = estimated_ATE − true_ATE
#  • RMSE(CATE) = sqrt( mean( (ĈATEᵢ − ITEᵢ)² ) )
#  • R²(CATE)   = 1 − Var(ĈATEᵢ − ITEᵢ) / Var(ITEᵢ)
# ============================================================================

# ── 14.0  Prepare numpy arrays from the episode ─────────────────────────────
#
#   We reuse the `episode` object generated in Chapter 1.
#   All arrays are (N,) or (N, d_*) float64 numpy for sklearn compatibility.
# ─────────────────────────────────────────────────────────────────────────────
 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
 
# ── Observed variables (what a learner actually sees) ──────────────────────
X_np  = episode.X.cpu().float().numpy()          # (N, d_x)
Z_np  = episode.Z.cpu().float().numpy()          # (N, d_z)
W_np  = episode.W.cpu().float().numpy()          # (N, d_w)
A_np  = episode.A.cpu().float().numpy()          # (N,)
Y_np  = episode.Y.cpu().float().numpy()          # (N,)
 
# ── Oracle quantities (not available to a real learner) ────────────────────
Y0_np  = episode.Y0.cpu().float().numpy()        # (N,)
Y1_np  = episode.Y1.cpu().float().numpy()        # (N,)
ITE_np = Y1_np - Y0_np                           # (N,)   ground truth
ATE_true = float(ITE_np.mean())
 
print(f"True ATE  : {ATE_true:+.4f}")
print(f"True CATE std (heterogeneity) : {ITE_np.std():.4f}")
print(f"N observations : {len(Y_np)}")
 
 
#%% ── 14.1  M1 — Difference in Means ─────────────────────────────────────────
#
#   Simplest possible estimator: compare average outcomes between arms.
#   Assumes no confounding.  In the presence of U this will be biased.
#
#   ATE_DM  = E[Y | A=1] - E[Y | A=0]
#   CATE(i) = ATE_DM   for all i   (constant, no personalisation)
# ─────────────────────────────────────────────────────────────────────────────
 
mask1 = A_np == 1.0
mask0 = A_np == 0.0
 
ate_dm   = float(Y_np[mask1].mean() - Y_np[mask0].mean())
cate_dm  = np.full(len(Y_np), ate_dm)    # constant prediction
 
print("── M1: Difference in Means ───────────────────────────────────────────")
print(f"  Estimated ATE : {ate_dm:+.4f}")
print(f"  True ATE      : {ATE_true:+.4f}")
print(f"  Bias          : {ate_dm - ATE_true:+.4f}")
 
 
#%% ── 14.2  M2 — OLS: Y ~ X + A ──────────────────────────────────────────────
#
#   Linear regression adjusting for X but ignoring U.
#   The coefficient β_A is taken as a (constant) ATE estimate.
#   CATE(i) is obtained by predicting Ŷ(x_i, 1) − Ŷ(x_i, 0).
#
#   Note: this is the standard "regression adjustment" estimator.
#         With unobserved confounding it will still be biased.
# ─────────────────────────────────────────────────────────────────────────────
 
A_col = A_np.reshape(-1, 1)
XA    = np.hstack([X_np, A_col])            # (N, d_x + 1)
 
ols_xa = LinearRegression().fit(XA, Y_np)
 
# Predict potential outcomes by setting A column to 0 / 1
XA1 = XA.copy(); XA1[:, -1] = 1.0
XA0 = XA.copy(); XA0[:, -1] = 0.0
 
cate_ols_xa = ols_xa.predict(XA1) - ols_xa.predict(XA0)
ate_ols_xa  = float(cate_ols_xa.mean())
 
# The coefficient on A is exactly the ATE in a linear model with A binary
coef_A_m2 = ols_xa.coef_[-1]
 
print("── M2: OLS  Y ~ X + A ────────────────────────────────────────────────")
print(f"  β_A (coef on treatment) : {coef_A_m2:+.4f}")
print(f"  Estimated ATE           : {ate_ols_xa:+.4f}")
print(f"  True ATE                : {ATE_true:+.4f}")
print(f"  Bias                    : {ate_ols_xa - ATE_true:+.4f}")
 
 
#%% ── 14.3  M3 — OLS: Y ~ X + A + W + Z ─────────────────────────────────────
#
#   Naïvely add both proxies as additional controls.
#   This is NOT principled: including Z and W as plain covariates can
#   introduce collider bias or attenuate the causal effect.
#   Included here as a "what happens if you just throw everything in" baseline.
# ─────────────────────────────────────────────────────────────────────────────
 
XAWZ  = np.hstack([X_np, A_col, W_np, Z_np])   # (N, d_x + 1 + d_w + d_z)
 
ols_full = LinearRegression().fit(XAWZ, Y_np)
 
# Again: predict at A=1 vs A=0, holding X, W, Z fixed
n_feat   = XAWZ.shape[1]
a_idx    = X_np.shape[1]   # column index of A in XAWZ
 
XAWZ1    = XAWZ.copy(); XAWZ1[:, a_idx] = 1.0
XAWZ0    = XAWZ.copy(); XAWZ0[:, a_idx] = 0.0
 
cate_ols_full  = ols_full.predict(XAWZ1) - ols_full.predict(XAWZ0)
ate_ols_full   = float(cate_ols_full.mean())
coef_A_m3      = ols_full.coef_[a_idx]
 
print("── M3: OLS  Y ~ X + A + W + Z ───────────────────────────────────────")
print(f"  β_A (coef on treatment) : {coef_A_m3:+.4f}")
print(f"  Estimated ATE           : {ate_ols_full:+.4f}")
print(f"  True ATE                : {ATE_true:+.4f}")
print(f"  Bias                    : {ate_ols_full - ATE_true:+.4f}")
 
 
#%% ── 14.4  M4 — P2SLS: Proximal Two-Stage Least Squares ─────────────────────
#
#   Grounded in the linear proximal causal learning framework.
#
#   Key idea (linear case)
#   ──────────────────────
#   We have two proxy "channels" for the unobserved confounder U:
#     • Z  — treatment-side proxy:  correlated with U, used as IV for A
#     • W  — outcome-side proxy:    correlated with U, acts as bridge control
#
#   Stage 1  (first-stage IV regression)
#   ─────────────────────────────────────
#     Â = OLS( A ~ X, Z, W )
#
#     Including W alongside Z in Stage 1 helps purge the part of A that is
#     driven by the confounded path U→A, while Z provides the exogenous
#     variation needed to identify the causal effect.
#
#   Stage 2  (structural equation)
#   ──────────────────────────────
#     Y = OLS( Y ~ Â, X, W )
#
#     W in Stage 2 absorbs residual confounding via the outcome proxy channel.
#     The coefficient on Â is the linear proximal ATE estimate.
#
#   CATE(i) is still constant (linear model), but the ATE is asymptotically
#   unbiased under the completeness conditions verified in Chapter 7.
#
#   Reference: Miao & Tchetgen Tchetgen (2018); Mastouri et al. (2021)
# ─────────────────────────────────────────────────────────────────────────────
 
# ── Stage 1 ────────────────────────────────────────────────────────────────
# Regress the outcome proxy W on A, X, and the treatment proxy Z
A_col_reshaped = A_np.reshape(-1, 1)
stage1_features = np.hstack([A_col_reshaped, X_np, Z_np])     # (N, 1 + d_x + d_z)

# This handles multi-output regression seamlessly if W has >1 dimension
stage1_model = LinearRegression().fit(stage1_features, W_np)
W_hat        = stage1_model.predict(stage1_features)          # (N, d_w) predicted W

# ── Stage 2 ────────────────────────────────────────────────────────────────
# Regress Y on A, X, and the predicted outcome proxy W_hat
stage2_features = np.hstack([A_col_reshaped, X_np, W_hat])    # (N, 1 + d_x + d_w)
stage2_model    = LinearRegression().fit(stage2_features, Y_np)

# Causal effect = coefficient on A (first column)
ate_p2sls    = float(stage2_model.coef_[0])
cate_p2sls   = np.full(len(Y_np), ate_p2sls)

print("── M4: P2SLS (Corrected) ──────────────────────────────────────────────")
print("  Stage 1 features : A, X, Z → Ŵ")
print("  Stage 2 features : A, X, Ŵ → Y")
s2_names  = ["Â (causal)"] + [f"X_{i}" for i in range(X_np.shape[1])] \
            + [f"W_{i}" for i in range(W_np.shape[1])]
for name, coef in zip(s2_names, stage2_model.coef_):
    print(f"        {name:15s} : {coef:+.4f}")


print(f"  β_A (causal ATE) : {ate_p2sls:+.4f}")
print(f"  True ATE         : {ATE_true:+.4f}")
print(f"  Bias             : {ate_p2sls - ATE_true:+.4f}")
print()

 
 
#%% ── 14.5  Compare Estimators — ATE Bias Bar Chart ─────────────────────────
 
estimators = {
    "M1  DiM"         : (ate_dm,        cate_dm),
    "M2  OLS(X,A)"    : (ate_ols_xa,    cate_ols_xa),
    "M3  OLS(X,A,W,Z)": (ate_ols_full,  cate_ols_full),
    "M4  P2SLS"       : (ate_p2sls,     cate_p2sls),
}
 
colors_est = [PALETTE[6], PALETTE[0], PALETTE[1], PALETTE[2]]
 
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
 
# ── Panel A: ATE estimate vs truth ─────────────────────────────────────────
ate_names  = list(estimators.keys())
ate_values = [v[0] for v in estimators.values()]
biases     = [v - ATE_true for v in ate_values]
 
bars = axes[0].barh(ate_names, biases,
                    color=[c + (0.8,) if isinstance(c, tuple) else c
                           for c in colors_est],
                    edgecolor="white", height=0.5)
axes[0].axvline(0, color="black", linewidth=1.5, linestyle="--", label="zero bias")
for bar, bias in zip(bars, biases):
    axes[0].text(
        bias + (0.003 if bias >= 0 else -0.003),
        bar.get_y() + bar.get_height() / 2,
        f"{bias:+.4f}", va="center", ha="left" if bias >= 0 else "right", fontsize=9,
    )
axes[0].set_title("ATE Bias  (estimated − true)", fontweight="bold")
axes[0].set_xlabel("Bias")
axes[0].legend(frameon=False, fontsize=9)
 
# ── Panel B: RMSE of CATE estimate ─────────────────────────────────────────
rmse_vals = []
r2_vals   = []
for name, (ate_est, cate_est) in estimators.items():
    rmse = float(np.sqrt(np.mean((cate_est - ITE_np) ** 2)))
    r2   = float(r2_score(ITE_np, cate_est))
    rmse_vals.append(rmse)
    r2_vals.append(r2)
 
axes[1].bar(ate_names, rmse_vals, color=colors_est, edgecolor="white", alpha=0.85)
for i, val in enumerate(rmse_vals):
    axes[1].text(i, val + 0.002, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
axes[1].set_title("RMSE( ĈATEᵢ , ITEᵢ )", fontweight="bold")
axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis="x", rotation=15)
 
# ── Panel C: R² of CATE estimate ───────────────────────────────────────────
axes[2].bar(ate_names, r2_vals, color=colors_est, edgecolor="white", alpha=0.85)
axes[2].axhline(0, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
for i, val in enumerate(r2_vals):
    axes[2].text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
axes[2].set_title("R²( ĈATEᵢ , ITEᵢ )", fontweight="bold")
axes[2].set_ylabel("R²")
axes[2].tick_params(axis="x", rotation=15)
 
fig.suptitle("Estimator Comparison — ATE Bias, CATE RMSE, CATE R²",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
 
 
#%% ── 14.6  CATE Scatter — Estimated vs Oracle ITE ──────────────────────────
#
#   For a perfect estimator all points would lie on the y = x diagonal.
#   Constant estimators (M1–M4 linear) produce a vertical line at the ATE.
#   Colour encodes the true ITE to reveal where each estimator fails.
# ─────────────────────────────────────────────────────────────────────────────
 
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
vmin, vmax = ITE_np.min(), ITE_np.max()
 
for ax, (name, (ate_est, cate_est)), color in zip(axes, estimators.items(), colors_est):
    sc = ax.scatter(
        ITE_np, cate_est,
        c=ITE_np, cmap="coolwarm", vmin=vmin, vmax=vmax,
        alpha=0.3, s=8, rasterized=True,
    )
    lim = [min(ITE_np.min(), cate_est.min()) - 0.1,
           max(ITE_np.max(), cate_est.max()) + 0.1]
    ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.6, label="perfect")
    ax.axhline(ate_est, color=color, linewidth=1.5, linestyle="-.",
               label=f"est. ATE={ate_est:.3f}")
    ax.axhline(ATE_true, color="black", linewidth=1.0, linestyle=":",
               alpha=0.5, label=f"true ATE={ATE_true:.3f}")
    rmse_i = float(np.sqrt(np.mean((cate_est - ITE_np) ** 2)))
    ax.set_title(f"{name}\nRMSE={rmse_i:.4f}", fontweight="bold", fontsize=9)
    ax.set_xlabel("True ITE")
    ax.set_ylabel("Estimated CATE")
    ax.legend(fontsize=7, frameon=False)
 
plt.colorbar(sc, ax=axes[-1], label="True ITE", fraction=0.04)
fig.suptitle("Estimated CATE vs Oracle ITE  (ideal = diagonal)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
 
 
#%% ── 14.7  Residual Analysis — Where Does Each Estimator Fail? ──────────────
#
#   Residual = estimated_CATE − true_ITE.
#   We examine whether residuals correlate with X, propensity, or U_PC1,
#   which would reveal systematic biases the estimator cannot capture.
# ─────────────────────────────────────────────────────────────────────────────
 
ps_np = episode.propensity.cpu().numpy()
u_pc1 = first_pc(episode.U.cpu().float().numpy())   # defined in chapter 8
 
fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharey="row")
 
covariates_diag = {
    "X_0":         X_np[:, 0],
    "Propensity":  ps_np,
    "U (PC1)":     u_pc1,
}
 
for row_idx, (est_name, (_, cate_est)) in enumerate(estimators.items()):
    residuals = cate_est - ITE_np
 
    for col_idx, (cov_name, cov_vals) in enumerate(covariates_diag.items()):
        ax = axes[row_idx, col_idx]
        ax.scatter(cov_vals, residuals, alpha=0.2, s=6,
                   color=colors_est[row_idx], rasterized=True)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        r, p = stats.pearsonr(cov_vals, residuals)
        m, b = np.polyfit(cov_vals, residuals, 1)
        xline = np.linspace(cov_vals.min(), cov_vals.max(), 100)
        ax.plot(xline, m * xline + b, "k-", linewidth=1.2, alpha=0.6)
        ax.set_title(f"{est_name}\nvs {cov_name}  (r={r:.3f})", fontsize=8, fontweight="bold")
        ax.set_xlabel(cov_name, fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Residual\n(ĈATEᵢ − ITEᵢ)", fontsize=8)
 
fig.suptitle("Residual Analysis — Systematic Bias per Estimator\n"
             "(non-zero slope → the estimator misses that source of heterogeneity)",
             fontweight="bold", y=1.01, fontsize=12)
plt.tight_layout()
plt.show()
 
 
#%% ── 14.8  Cross-Episode Stability of ATE Estimates ────────────────────────
#
#   Generate K episodes and track each estimator's ATE across them.
#   This shows variance and bias stability, not just single-episode performance.
# ─────────────────────────────────────────────────────────────────────────────
 
K_est = 30
 
cross_ep_records = []
for k in range(K_est):
    ep_k   = prior.generate_episode(batch_size=N_OBS, episode_seed=1000 + k)
 
    X_k   = ep_k.X.cpu().float().numpy()
    Z_k   = ep_k.Z.cpu().float().numpy()
    W_k   = ep_k.W.cpu().float().numpy()
    A_k   = ep_k.A.cpu().float().numpy()
    Y_k   = ep_k.Y.cpu().float().numpy()
    ITE_k = (ep_k.Y1 - ep_k.Y0).cpu().float().numpy()
    true_ate_k = float(ITE_k.mean())
 
    # M1
    m1 = float(Y_k[A_k == 1].mean() - Y_k[A_k == 0].mean())
 
    # M2
    XA_k  = np.hstack([X_k, A_k.reshape(-1, 1)])
    m2    = LinearRegression().fit(XA_k, Y_k).coef_[-1]
 
    # M3
    XAWZ_k = np.hstack([X_k, A_k.reshape(-1, 1), W_k, Z_k])
    m3     = LinearRegression().fit(XAWZ_k, Y_k).coef_[X_k.shape[1]]
 
    # M4 P2SLS
    A_col_reshaped = A_k.reshape(-1, 1)
    stage1_features = np.hstack([A_col_reshaped, X_k, Z_k])     # (N, 1 + d_x + d_z)
    stage1_model = LinearRegression().fit(stage1_features, W_k)
    W_hat        = stage1_model.predict(stage1_features)          # (N, d_w) predicted W
    stage2_features = np.hstack([A_col_reshaped, X_k, W_hat])    # (N, 1 + d_x + d_w)
    stage2_model    = LinearRegression().fit(stage2_features, Y_k)
    m4    = float(stage2_model.coef_[0])

 
    cross_ep_records.append({
        "episode": k, "true_ATE": true_ate_k,
        "M1_DiM": m1, "M2_OLS_XA": float(m2),
        "M3_OLS_full": float(m3), "M4_P2SLS": float(m4),
    })
 
ced = pd.DataFrame(cross_ep_records)
 
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
method_cols  = ["M1_DiM", "M2_OLS_XA", "M3_OLS_full", "M4_P2SLS"]
method_names = ["M1  DiM", "M2  OLS(X,A)", "M3  OLS(X,A,W,Z)", "M4  P2SLS"]
 
for ax, col, name, color in zip(axes.ravel(), method_cols, method_names, colors_est):
    bias_series = ced[col] - ced["true_ATE"]
 
    ax.plot(ced["episode"], ced["true_ATE"], color="black",
            linewidth=1.0, linestyle=":", alpha=0.6, label="true ATE")
    ax.plot(ced["episode"], ced[col], color=color, linewidth=1.5, label=name)
 
    mean_bias = bias_series.mean()
    std_bias  = bias_series.std()
    ax.set_title(
        f"{name}\nMean bias={mean_bias:+.4f}  ±{std_bias:.4f}",
        fontweight="bold", fontsize=10,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("ATE estimate")
    ax.legend(fontsize=8, frameon=False)
 
fig.suptitle(f"ATE Estimate Stability Across {K_est} Episodes\n"
             "(black dotted = oracle true ATE per episode)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
 
print(ced[["true_ATE"] + method_cols].describe().round(4).to_string())
 
 
#%% ── 14.9  Summary Table ────────────────────────────────────────────────────
 
summary_rows = []
for (name, (ate_est, cate_est)), col in zip(estimators.items(), method_cols):
    bias_series_full = ced[col] - ced["true_ATE"]
    summary_rows.append({
        "Estimator"       : name,
        "ATE (single ep)" : f"{ate_est:+.4f}",
        "Bias (single ep)": f"{ate_est - ATE_true:+.4f}",
        "RMSE CATE"       : f"{np.sqrt(np.mean((cate_est - ITE_np)**2)):.4f}",
        "R² CATE"         : f"{r2_score(ITE_np, cate_est):.4f}",
        "Cross-ep bias μ" : f"{bias_series_full.mean():+.4f}",
        "Cross-up median bias" : f"{bias_series_full.median():+.4f}",
        "Cross-ep bias σ" : f"{bias_series_full.std():.4f}",
    })
 
summary_df = pd.DataFrame(summary_rows).set_index("Estimator")
print("\n" + "=" * 80)
print("CATE ESTIMATOR SUMMARY")
print("=" * 80)
print(summary_df.to_string())
print("=" * 80)
print(f"\nOracle true ATE (single episode) : {ATE_true:+.4f}")
print("Note: All methods are LINEAR → CATE is constant (= ATE estimate).")
print("      Non-zero R²(CATE) requires heterogeneous treatment effect models.")