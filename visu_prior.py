#%% ── 0. Imports & Style ──────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from scipy import stats

# remove futurewarnings:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ── Use the new BNN prior ──────────────────────────────────────────────────
from proximalfm.prior import BNNPriorComplete as ProximalPrior

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
#   We instantiate ProximalPriorConfig with explicit NetworkSpecs so you can
#   control the complexity (depth/width) of the BNN generating the data.
# ─────────────────────────────────────────────────────────────────────────────

from proximalfm.Structure import ProximalPriorConfig, NetworkSpec

# Define network complexity. Increase max_layers or hidden_size for wilder manifolds.
default_spec = NetworkSpec(
    # hidden_size=16, 
    min_layers=2, 
    max_layers=2, 
    # weight_scale=1.0, 
    # bias_scale=0.5,
)

cfg = ProximalPriorConfig(
    d_x            = 5,
    d_u            = 2,
    d_z            = 3,
    d_w            = 3,
    noise_scale_w  = 1,
    noise_scale_z  = 1,
    noise_scale_x  = 1,
    temperature    = 2,
    confounding_strength=1,
    
    # Injecting the BNN specs
    x_spec         = default_spec,
    u_spec         = default_spec,
    z_spec         = default_spec,
    w_spec         = default_spec,
    treatment_spec = default_spec,
    outcome_spec   = default_spec,
)

prior = ProximalPrior(config=cfg, seed=42)

N_OBS = 5_000   
episode = prior.generate_episode(batch_size=N_OBS, episode_seed=100)

print("Episode generated successfully.")
print(f"  X  : {tuple(episode.X.shape)}")
print(f"  Z  : {tuple(episode.Z.shape)}")
print(f"  W  : {tuple(episode.W.shape)}")
print(f"  U  : {tuple(episode.U.shape)}")
print(f"  A  : {tuple(episode.A.shape)}")
print(f"  Y  : {tuple(episode.Y.shape)}")
print(f"  propensity: {tuple(episode.propensity.shape)}")


#%% ── 2. Tensor-Level Inspection (dtypes, ranges, NaNs) ──────────────────────

def describe_tensor(name: str, t: torch.Tensor) -> pd.Series:
    arr = t.cpu().float().numpy().ravel()
    return pd.Series({
        "shape"  : str(tuple(t.shape)),
        "NaN"    : int(np.isnan(arr).sum()),
        "min"    : float(arr.min()),
        "max"    : float(arr.max()),
        "mean"   : float(arr.mean()),
        "std"    : float(arr.std()),
    }, name=name)

tensors = {
    "X": episode.X, "Z": episode.Z, "W": episode.W,
    "U": episode.U, "A": episode.A, "Y": episode.Y,
    "ITE": episode.Y1 - episode.Y0,
}
summary = pd.DataFrame([describe_tensor(k, v) for k, v in tensors.items()])
print(summary.to_string())


#%% ── 3. Univariate Distributions — Notice the Mixed Noise! ──────────────────
#
#   BNNPriorComplete uses a mix of Gaussian, Uniform, and Laplace noise.
#   Look for sharp peaks (Laplace), bounded edges (Uniform), or standard bells.
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_kde(ax, tensor: torch.Tensor, label: str, color):
    arr = tensor.cpu().float().numpy()
    for d in range(arr.shape[1]):
        sns.kdeplot(arr[:, d], ax=ax, color=color, alpha=0.5, linewidth=1.6, label=f"{label}_{d}")
    ax.set_title(label, fontweight="bold")
    ax.set_ylabel("density")

groups = [("X", episode.X, C_X), ("Z", episode.Z, C_Z),
          ("W", episode.W, C_W), ("U", episode.U, C_U)]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, tensor, color) in zip(axes, groups):
    plot_group_kde(ax, tensor, name, color)

fig.suptitle("Univariate KDE — Mixed Noise Distributions (Laplace/Gaussian/Uniform)", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()


#%% ── 3b. NEW: BNN Functional Diversity Across Episodes ──────────────────────
#
#   This is the defining feature of BNNPriorComplete. 
#   Because it samples a NEW neural network for every episode, the manifold 
#   mapping U to Z (or U to W) completely changes shape every time you call it.
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False, sharex=False)

for i in range(4):
    ep_temp = prior.generate_episode(batch_size=2000, episode_seed=999 + i)
    u_vals = ep_temp.U.cpu().numpy()[:, 0]  # Just look at first dimension
    z_vals = ep_temp.Z.cpu().numpy()[:, 0]
    
    axes[i].scatter(u_vals, z_vals, alpha=0.4, s=5, color=C_Z)
    axes[i].set_title(f"Episode {999+i}", fontweight="bold")
    axes[i].set_xlabel("Latent U_0")
    if i == 0: axes[i].set_ylabel("Proxy Z_0")

fig.suptitle("BNN Functional Diversity: Manifold shape completely changes per episode", y=1.05, fontweight="bold")
plt.tight_layout()
plt.show()


#%% ── 4. Treatment Distribution & Propensity ─────────────────────────────────

A_np   = episode.A.cpu().numpy()
ps_np  = episode.propensity.cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(ps_np, bins=40, color=C_A1, edgecolor="white", alpha=0.8)
axes[0].set_title("Propensity Score e(X, U, Z)", fontweight="bold")

treat_counts = pd.Series(A_np).value_counts().sort_index()
axes[1].bar(["Control (A=0)", "Treated (A=1)"], treat_counts.values, color=[C_A0, C_A1])
axes[1].set_title("Treatment Assignment", fontweight="bold")

for a_val, label, color in [(0, "A=0", C_A0), (1, "A=1", C_A1)]:
    mask = A_np == a_val
    sns.kdeplot(ps_np[mask], ax=axes[2], label=label, color=color, fill=True, alpha=0.3)
axes[2].set_title("Propensity Overlap by Arm", fontweight="bold")
axes[2].legend(frameon=False)

fig.suptitle("Treatment & Propensity (BNN Logits)", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()


#%% ── 5. Outcome Distributions ───────────────────────────────────────────────

Y0_np  = episode.Y0.cpu().numpy()
Y1_np  = episode.Y1.cpu().numpy()
ITE_np = (episode.Y1 - episode.Y0).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for arr, label, color in [(Y0_np, "Y(0)", C_A0), (Y1_np, "Y(1)", C_A1)]:
    sns.kdeplot(arr, ax=axes[0], label=label, color=color, fill=True, alpha=0.3)
axes[0].set_title("Potential Outcomes Y(0) and Y(1)", fontweight="bold")
axes[0].legend(frameon=False)

sns.kdeplot(ITE_np, ax=axes[1], color=C_ITE, fill=True, alpha=0.5)
axes[1].axvline(ITE_np.mean(), color="black", linestyle="--")
axes[1].set_title("Individual Treatment Effect (ITE)", fontweight="bold")

ite_sign = (ITE_np >= 0).astype(int)
axes[2].scatter(Y0_np, Y1_np, c=ite_sign, cmap="bwr_r", alpha=0.3, s=8, rasterized=True)
lims = [min(Y0_np.min(), Y1_np.min()), max(Y0_np.max(), Y1_np.max())]
axes[2].plot(lims, lims, "k--", linewidth=1, alpha=0.5)
axes[2].set_title("Y(0) vs Y(1)", fontweight="bold")

fig.suptitle("Outcome Analysis (Complex BNN Structural Equations)", y=1.02, fontweight="bold")
plt.tight_layout()
plt.show()


#%% ── 6 & 7. Pairwise Correlations & Proxy Quality ───────────────────────────
#
#   NOTE: Because BNNs create highly non-linear mappings, Pearson's r might 
#   be close to 0 even if two variables are deterministically linked! 
#   We rely more on the scatter plots here.
# ─────────────────────────────────────────────────────────────────────────────

U_np = episode.U.cpu().numpy()
Z_np = episode.Z.cpu().numpy()
W_np = episode.W.cpu().numpy()

def scatter_panel(fig, gs_row, tensor_a, tensor_b, name_a, name_b, color):
    d_a, d_b = tensor_a.shape[1], tensor_b.shape[1]
    for i in range(d_a):
        for j in range(d_b):
            ax = fig.add_subplot(gs_row[i, j])
            ax.scatter(tensor_a[:, i], tensor_b[:, j], alpha=0.25, s=6, color=color, rasterized=True)
            r, _ = stats.pearsonr(tensor_a[:, i], tensor_b[:, j])
            ax.set_title(f"Linear r={r:.2f}", fontsize=8) # Emphasize "Linear"
            if j == 0: ax.set_ylabel(f"{name_a}_{i}", fontsize=8)
            if i == d_a - 1: ax.set_xlabel(f"{name_b}_{j}", fontsize=8)
            ax.tick_params(labelsize=6)

fig = plt.figure(figsize=(15, 8))
gs_top = gridspec.GridSpec(U_np.shape[1], Z_np.shape[1], figure=fig, top=0.9, bottom=0.55)
gs_bot = gridspec.GridSpec(U_np.shape[1], W_np.shape[1], figure=fig, top=0.4, bottom=0.05)

scatter_panel(fig, gs_top, U_np, Z_np, "U", "Z", C_Z)
scatter_panel(fig, gs_bot, U_np, W_np, "U", "W", C_W)

fig.text(0.5, 0.95, "Latent U vs Proxy Z (Notice non-linear clusters/curves)", fontweight="bold", fontsize=13, ha="center")
fig.text(0.5, 0.45, "Latent U vs Proxy W (Notice non-linear clusters/curves)", fontweight="bold", fontsize=13, ha="center")
plt.show()


#%% ── 8. Causal Graph Empirical Checks (Non-Linear View) ─────────────────────

def first_pc(arr: np.ndarray) -> np.ndarray:
    centered = arr - arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[0]

z_pc = first_pc(Z_np)
w_pc = first_pc(W_np)
u_pc = first_pc(U_np)
Y_np = episode.Y.cpu().numpy()

pairs = [
    (z_pc,  ps_np, "Z (PC1)", "Prop", "Z ↔ Prop(A) [Proxy to Assignment]", C_Z),
    (w_pc,  Y_np,  "W (PC1)", "Y",    "W ↔ Y [Proxy to Outcome]", C_W),
    (u_pc,  ps_np, "U (PC1)", "Prop", "U ↔ Prop(A) [Confounder to Assignment]", C_U),
    (u_pc,  Y_np,  "U (PC1)", "Y",    "U ↔ Y [Confounder to Outcome]", C_U),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (x, y, xlab, ylab, title, color) in zip(axes, pairs):
    ax.scatter(x, y, alpha=0.3, s=5, color=color, rasterized=True)
    
    # Fit a simple polynomial instead of just a straight line to show BNN trends
    try:
        p = np.poly1d(np.polyfit(x, y.ravel(), 3))
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, p(xline), "k--", linewidth=2, alpha=0.8, label="Cubic Trend")
    except:
        pass
        
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

fig.suptitle("Causal Graph Checks (Curves indicate non-linear BNN leakage/dependency)", fontweight="bold", y=1.05)
plt.tight_layout()
plt.show()


#%% ── 10. PRIOR CHARACTERIZATION ─────────────────────────────────────────────

K        = 100
N_SMALL  = 500   

records = []
for k in range(K):
    ep = prior.generate_episode(batch_size=N_SMALL, episode_seed=k)
    ite = (ep.Y1 - ep.Y0).cpu().numpy()
    ps  = ep.propensity.cpu().numpy()
    
    records.append({
        "treat_rate"   : float(ep.A.cpu().numpy().mean()),
        "ps_std"       : float(ps.std()),
        "ATE"          : float(ite.mean()),
        "ITE_std"      : float(ite.std()),
        "Y_std"        : float(ep.Y.cpu().numpy().std()),
    })

edf = pd.DataFrame(records)

fig, axes = plt.subplots(1, 5, figsize=(18, 3))
for ax, col, color, title in zip(
    axes, 
    ["treat_rate", "ps_std", "ATE", "ITE_std", "Y_std"],
    ["#378ADD", "#378ADD", "#1D9E75", "#1D9E75", "#888780"],
    ["Treatment Rate", "Propensity Spread", "ATE", "ITE Heterogeneity", "Outcome Std"]
):
    data = edf[col].dropna().values
    ax.hist(data, bins=20, color=color, alpha=0.75, edgecolor="white", density=True)
    mu, sd = data.mean(), data.std()
    ax.axvline(mu, color="black", linestyle="--")
    ax.set_title(title, fontsize=10, fontweight="bold")

fig.suptitle(f"Prior Characterization over {K} Episodes (Tracking BNN Volatility)", y=1.1, fontweight="bold")
plt.tight_layout()
plt.show()
print("✓ Prior Characterization Done")


#%% ── 10. Predictability of CATE (Information Progression) ───────────────────
#
#   How "hard" is the causal inference task on this specific manifold?
#   We evaluate predictability by progressively adding more information:
#   1. Naive S-Learner (OLS): Predict Y from X and A, ignoring confounding.
#   2. Oracle OLS (X): Linear limit of predicting True ITE.
#   3. Oracle RF (X): Non-linear limit of predicting True ITE.
#   4. Oracle RF (X, W, Z): Predictability when including proxies.
#   5. Oracle RF (X, W, Z, U): The absolute upper bound including latents.
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Extract all relevant tensors
X_np = episode.X.cpu().numpy()
W_np = episode.W.cpu().numpy()
Z_np = episode.Z.cpu().numpy()
U_np = episode.U.cpu().numpy()
A_np = episode.A.cpu().numpy()
Y_np = episode.Y.cpu().numpy()
ITE_np = (episode.Y1 - episode.Y0).cpu().numpy()

# ── 1. Naive S-Learner OLS (Predict Y from X and A)
XA_np = np.column_stack((X_np, A_np))
s_learner = LinearRegression().fit(XA_np, Y_np)
XA1_np = np.column_stack((X_np, np.ones_like(A_np)))
XA0_np = np.column_stack((X_np, np.zeros_like(A_np)))
pred_slearner = s_learner.predict(XA1_np) - s_learner.predict(XA0_np)
r2_slearner = r2_score(ITE_np, pred_slearner)

# ── 2. Oracle Linear Model (X -> ITE)
olin = LinearRegression().fit(X_np, ITE_np)
pred_olin = olin.predict(X_np)
r2_olin = r2_score(ITE_np, pred_olin)

# ── 3. Oracle Random Forest (X -> ITE)
rf_x = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42).fit(X_np, ITE_np)
pred_x = rf_x.predict(X_np)
r2_x = r2_score(ITE_np, pred_x)

# ── 4. Oracle Random Forest with Proxies (X, W, Z -> ITE)
XWZ_np = np.column_stack((X_np, W_np, Z_np))
rf_xwz = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42).fit(XWZ_np, ITE_np)
pred_xwz = rf_xwz.predict(XWZ_np)
r2_xwz = r2_score(ITE_np, pred_xwz)

# ── 5. Oracle Random Forest with Latents (X, W, Z, U -> ITE)
XWZU_np = np.column_stack((X_np, W_np, Z_np, U_np))
rf_xwzu = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42).fit(XWZU_np, ITE_np)
pred_xwzu = rf_xwzu.predict(XWZU_np)
r2_xwzu = r2_score(ITE_np, pred_xwzu)

# ── Plotting ──
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True, sharex=True)

plots = [
    (pred_slearner, r2_slearner, "1. Naive OLS S-Learner\n(Observed X, A → Y)", C_A1),
    (pred_olin, r2_olin, "2. Oracle OLS\n(X → True ITE)", C_ITE),
    (pred_x, r2_x, "3. Oracle RF\n(X → True ITE)", "#1D9E75"),
    (pred_xwz, r2_xwz, "4. Oracle RF + Proxies\n(X, W, Z → True ITE)", "#7F77DD"),
    (pred_xwzu, r2_xwzu, "5. Oracle RF + Latents\n(X, W, Z, U → True ITE)", "#D85A30")
]

lims = [ITE_np.min(), ITE_np.max()]

for ax, (preds, r2, title, color) in zip(axes, plots):
    ax.scatter(ITE_np, preds, alpha=0.3, s=10, color=color, rasterized=True)
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect Prediction")
    ax.set_title(f"{title}\nCATE R² = {r2:.3f}", fontweight="bold", fontsize=11)
    ax.set_xlabel("True ITE")
    if ax == axes[0]:
        ax.set_ylabel("Predicted ITE")
    ax.legend(fontsize=9, frameon=False, loc="upper left")

fig.suptitle("Predictability Progression: How much information is required to learn the CATE manifold?", fontweight="bold", fontsize=14, y=1.08)
plt.tight_layout()
plt.show()

print("\n── CATE Predictability Progression (R²) ──")
print(f"1. Naive S-Learner (X, A)     : {r2_slearner:.3f}  (Baseline biased by confounding)")
print(f"2. Oracle OLS (X)             : {r2_olin:.3f}  (Best possible linear fit)")
print(f"3. Oracle RF (X)              : {r2_x:.3f}  (Best possible non-linear fit on X alone)")
print(f"4. Oracle RF (X, W, Z)        : {r2_xwz:.3f}  (Information gain from adding proxies)")
print(f"5. Oracle RF (X, W, Z, U)     : {r2_xwzu:.3f}  (Absolute upper bound with latent U)")