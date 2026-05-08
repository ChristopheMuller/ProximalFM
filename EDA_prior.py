#%%

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

# Assuming proximalfm is in your path
from proximalfm.prior import PriorDataset

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")  # Suppress

#%% --- 1. Instantiate the Prior Dataset ---
dataset = PriorDataset(
    batch_size=10,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_W=1,
    max_dim_W=1,
    min_dim_Z=1,
    max_dim_Z=1,
    min_dim_X=2,
    max_dim_X=2,
    seq_len_per_gp=False,
    max_seq_len=300,
    prior_type="mlp_scm",
    device="cpu",
)

#%% --- 2. Generate and Store All Datasets (Flattened) ---
n_batches = 10
iterator = iter(dataset)

print(f"Generating {n_batches * dataset.batch_size} datasets...")

# Store flattened, individual datasets as numpy arrays
all_datasets = []
for _ in tqdm(range(n_batches), desc="Sampling Batches"):
    batch = next(iterator)

    # Iterate through the batch dimension to extract individual datasets
    for i in range(dataset.batch_size):
        # We only need the first 8 tensors: U, X, W, Z, A, Y_obs, Y0, Y1
        ds = tuple(tensor[i].cpu().numpy() for tensor in batch[:8])
        all_datasets.append(ds)

#%% --- 3. Helper Functions ---
def safe_target(y):
    """Safely ravel targets only if they are 1D to prevent sklearn warnings/errors."""
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    return y


def test_conditional_independence(target, base_features, test_feature):
    """
    Fits a model with base features, then base + test feature.
    Returns the difference in R^2. A negligible difference implies conditional independence.
    """
    rf = RandomForestRegressor(n_estimators=30, max_depth=5, n_jobs=-1, random_state=42)
    target = safe_target(target)

    # Model 1: Conditioned only on base
    rf.fit(base_features, target)
    score_base = r2_score(target, rf.predict(base_features))

    # Model 2: Conditioned on base + test_feature
    augmented_features = np.concatenate([base_features, test_feature], axis=1)
    rf.fit(augmented_features, target)
    score_aug = r2_score(target, rf.predict(augmented_features))

    diff = score_aug - score_base
    return score_base, score_aug, diff


def print_summary_stats(name, values):
    """Prints Mean, Min, Max, and Quartiles for a given list of metrics."""
    values = np.array(values)
    print(f"\n--- {name} ---")
    print(f"  Mean : {np.mean(values): .4f}")
    print(f"  Min  : {np.min(values): .4f}")
    print(f"  25%  : {np.percentile(values, 25): .4f}")
    print(f"  50%  : {np.median(values): .4f}")
    print(f"  75%  : {np.percentile(values, 75): .4f}")
    print(f"  Max  : {np.max(values): .4f}")


#%% --- 4. Evaluate Per-Dataset Loop ---
# Dictionaries to store our metrics across all datasets
metrics = {
    "Signal: U -> Z (R^2)": [],
    "Signal: U -> W (R^2)": [],
    "Marginal Signal: A -> Y (R^2)": [],
    "Leakage Delta: Z _|_ W | (X, U)": [],
    "Leakage Delta: Y _|_ Z | (A, X, U)": [],
    "Leakage Delta: W _|_ (A, Z) | (X, U)": [],
}

print(
    f"\nEvaluating structural integrity across {len(all_datasets)} stored datasets..."
)

for ds in tqdm(all_datasets, desc="Evaluating Datasets"):
    # Unpack the pre-flattened numpy arrays
    u, x, w, z, a, y_obs, y0, y1 = ds

    # A and Y are often 1D, ensure they are properly reshaped for concatenation/features
    a_2d = a.reshape(-1, 1) if a.ndim == 1 else a

    # Force 2D for feature arrays
    z_2d = z.reshape(-1, 1) if z.ndim == 1 else z

    # --- 4.1: Targeted Signal Strength ---
    rf_z = RandomForestRegressor(
        n_estimators=30, max_depth=5, n_jobs=-1, random_state=42
    ).fit(u, safe_target(z))
    metrics["Signal: U -> Z (R^2)"].append(r2_score(safe_target(z), rf_z.predict(u)))

    rf_w = RandomForestRegressor(
        n_estimators=30, max_depth=5, n_jobs=-1, random_state=42
    ).fit(u, safe_target(w))
    metrics["Signal: U -> W (R^2)"].append(r2_score(safe_target(w), rf_w.predict(u)))

    rf_y = RandomForestRegressor(
        n_estimators=30, max_depth=5, n_jobs=-1, random_state=42
    ).fit(a_2d, safe_target(y_obs))
    metrics["Marginal Signal: A -> Y (R^2)"].append(
        r2_score(safe_target(y_obs), rf_y.predict(a_2d))
    )

    # --- 4.2: Strict Proxy Exclusion (Leakage Deltas) ---
    base_XU = np.concatenate([x, u], axis=1)
    base_AXU = np.concatenate([a_2d, x, u], axis=1)

    # Check 1: Z _|_ W | (X, U)
    _, _, delta_zw = test_conditional_independence(
        target=w, base_features=base_XU, test_feature=z_2d
    )
    metrics["Leakage Delta: Z _|_ W | (X, U)"].append(delta_zw)

    # Check 2: Y _|_ Z | (A, X, U)
    _, _, delta_yz = test_conditional_independence(
        target=y_obs, base_features=base_AXU, test_feature=z_2d
    )
    metrics["Leakage Delta: Y _|_ Z | (A, X, U)"].append(delta_yz)

    # Check 3: W _|_ (A, Z) | (X, U)
    az_concat = np.concatenate([a_2d, z_2d], axis=1)
    _, _, delta_waz = test_conditional_independence(
        target=w, base_features=base_XU, test_feature=az_concat
    )
    metrics["Leakage Delta: W _|_ (A, Z) | (X, U)"].append(delta_waz)


#%% --- 5. Print Summary Reports & Visualizations ---
print("\n" + "=" * 50)
print("  PROXIMAL PRIOR EVALUATION REPORT")
print("  Total Datasets Evaluated:", len(all_datasets))
print("=" * 50)

for metric_name, values in metrics.items():
    print_summary_stats(metric_name, values)

# --- Plotting the Distributions ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[idx]

    # Calculate the dynamic lower limit
    lower_limit = max(-1, min(values) - 0.1)

    # Change to histogram and optionally restrict the bins to the view area
    sns.histplot(values, ax=ax, color="royalblue", binrange=(lower_limit, 1), bins=15)

    # Vertical dashed lines for reference
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.6)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6)

    ax.set_title(metric_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("R² Score" if "Signal" in metric_name else "R² Delta")
    ax.set_ylabel("Count")

    # Apply the requested x-axis limits
    ax.set_xlim(lower_limit, 1.1)

plt.tight_layout()
plt.show()


#%% --- 6. Plot Individual Datasets ---

K_to_plot = 15
fig2, axes2 = plt.subplots(K_to_plot, 4, figsize=(24, 5 * K_to_plot))

if K_to_plot == 1:
    axes2 = [axes2]

# Because all_datasets is flattened, we can just slice the first K elements!
for idx, ds in enumerate(all_datasets[:K_to_plot]):

    u, x, w, z, a_samp, y_obs_samp, y0_samp, y1_samp = ds

    ax_scatter = axes2[idx][0]
    ax_true = axes2[idx][1]
    ax_obs = axes2[idx][2]
    ax_cate = axes2[idx][3]  # New axis for Y(1) - Y(0)

    # Flatten just for the scatter and KDE plots (handles 1D variables gracefully)
    y0_flat = y0_samp.ravel()
    y1_flat = y1_samp.ravel()
    y_obs_flat = y_obs_samp.ravel()
    a_flat = a_samp.ravel()

    # Calculate true Individual Treatment Effects (CATE)
    cate_flat = y1_flat - y0_flat

    # 1. Scatter of Y0 vs Y1
    ax_scatter.scatter(y0_flat, y1_flat, alpha=0.5, color="purple", s=15)

    # Plot the y=x line for reference (where treatment has zero effect)
    min_val = min(min(y0_flat), min(y1_flat))
    max_val = max(max(y0_flat), max(y1_flat))
    ax_scatter.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        alpha=0.7,
        label="y = x (No Effect)",
    )

    ax_scatter.set_title(f"DGP {idx+1}: True Potential Outcomes")
    ax_scatter.set_xlabel("Y(0)")
    ax_scatter.set_ylabel("Y(1)")
    ax_scatter.legend()

    # 2. True Causal Shift (Marginal Y0 vs Y1)
    sns.kdeplot(y0_flat, fill=True, ax=ax_true, label="Y(0)", color="blue")
    sns.kdeplot(y1_flat, fill=True, ax=ax_true, label="Y(1)", color="orange")
    ax_true.set_title(f"DGP {idx+1}: True Causal Marginals")
    ax_true.set_xlabel("Potential Outcome Value")
    ax_true.legend()

    # 3. Observational Shift (Y_obs | A=0 vs Y_obs | A=1)
    y_obs_a0 = y_obs_flat[a_flat == 0]
    y_obs_a1 = y_obs_flat[a_flat == 1]

    sns.kdeplot(
        y_obs_a0,
        fill=True,
        ax=ax_obs,
        label="Y_obs | A=0",
        color="blue",
        linestyle="--",
    )
    sns.kdeplot(
        y_obs_a1,
        fill=True,
        ax=ax_obs,
        label="Y_obs | A=1",
        color="orange",
        linestyle="--",
    )
    ax_obs.set_title(f"DGP {idx+1}: Confounded Observational Marginals")
    ax_obs.set_xlabel("Observed Outcome Value")
    ax_obs.legend()

    # 4. True Treatment Effect Distribution (Y1 - Y0)
    sns.kdeplot(cate_flat, fill=True, ax=ax_cate, color="mediumseagreen")

    # Reference line for zero effect
    ax_cate.axvline(x=0, color="red", linestyle="--", alpha=0.6, label="Zero Effect")

    # Reference line for Average Treatment Effect (ATE)
    ate = np.mean(cate_flat)
    ax_cate.axvline(
        x=ate, color="black", linestyle="-", alpha=0.8, label=f"ATE = {ate:.2f}"
    )

    ax_cate.set_title(f"DGP {idx+1}: True Treatment Effect (CATE)")
    ax_cate.set_xlabel("Y(1) - Y(0)")
    ax_cate.set_ylabel("Density")
    ax_cate.legend()

plt.tight_layout()
plt.show()

#%%
