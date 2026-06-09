# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from proximalfm.prior import PriorDataset
from proximalfm._model import ProximalFM

# %%
# 1. Configuration & Toggles
CHECKPOINT_DIR = "check9/"
TEST_SAMPLE_IDX = 0
PLOT_MEAN = True
PLOT_NORMALIZED = True

# --- NEW PARAMETER ---
Y_LIM = (-3.0, 3.0)

prior = PriorDataset(
    batch_size=1,
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
    max_seq_len=2048,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.8,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    device="cpu",
)

# Fetch a single batch
batch = prior.get_batch(batch_size=2)
U, X, W, Z, A, A_probs, Y_obs, Y0, Y1, d_U, d_X, d_W, d_Z, seq_len, train_sizes = batch

# Ensure batch dimension exists
if len(Y0.shape) == 1:
    Y0 = Y0.unsqueeze(0)
if len(Y1.shape) == 1:
    Y1 = Y1.unsqueeze(0)
if len(A.shape) == 1:
    A = A.unsqueeze(0)
if len(Y_obs.shape) == 1:
    Y_obs = Y_obs.unsqueeze(0)

split_idx = train_sizes[0].item()
A_input = A.unsqueeze(-1).float()

# --- NEW NORMALIZATION STRATEGY ---
y_obs_train_raw = Y_obs[:, :split_idx]

# Calculate Context Stats
y_context_mean = y_obs_train_raw.mean(dim=1, keepdim=True)
y_context_std = y_obs_train_raw.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

# Standardize Context
y_obs_train_norm = (y_obs_train_raw - y_context_mean) / y_context_std

# Extract Raw Test Ground Truths
y0_test_raw = Y0[:, split_idx:]
y1_test_raw = Y1[:, split_idx:]

true_y0_val_raw = y0_test_raw[0, TEST_SAMPLE_IDX].item()
true_y1_val_raw = y1_test_raw[0, TEST_SAMPLE_IDX].item()
true_cate_val_raw = true_y1_val_raw - true_y0_val_raw

# Route based on flag
if PLOT_NORMALIZED:
    y_obs_in = y_obs_train_norm

    # Standardize the true values so the horizontal lines match the plot
    mean_val = y_context_mean[0, 0].item()
    std_val = y_context_std[0, 0].item()

    true_y0_val = (true_y0_val_raw - mean_val) / std_val
    true_y1_val = (true_y1_val_raw - mean_val) / std_val
    true_cate_val = true_y1_val - true_y0_val
else:
    y_obs_in = y_obs_train_raw

    true_y0_val = true_y0_val_raw
    true_y1_val = true_y1_val_raw
    true_cate_val = true_cate_val_raw

# %%
# 2. Gather All Checkpoints
raw_checkpoints = [
    f
    for f in os.listdir(CHECKPOINT_DIR)
    if f.startswith("step-") and f.endswith(".ckpt")
]


def get_step(ckpt_name):
    return int(ckpt_name.split("-")[1].split(".")[0])


raw_checkpoints.sort(key=get_step)
steps = [get_step(ckpt) for ckpt in raw_checkpoints]

if not raw_checkpoints:
    raise ValueError(f"No checkpoints found in {CHECKPOINT_DIR}")

# %%
# 3. Predict the Distribution Across All Checkpoints
history_weights_y0 = []
history_weights_y1 = []
history_weights_cate = []

vals_y0, vals_y1, vals_cate = None, None, None

first_ckpt_path = os.path.join(CHECKPOINT_DIR, raw_checkpoints[0])
first_ckpt = torch.load(first_ckpt_path, map_location="cpu", weights_only=True)
model = ProximalFM(**first_ckpt["config"])

print(f"Tracking evolution across {len(raw_checkpoints)} checkpoints...")

for ckpt_name in raw_checkpoints:
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, ckpt_name), map_location="cpu", weights_only=True
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict)
    model.eval()

    with torch.inference_mode():
        # Predict uses internal covariate scaling and tracks y_obs_in scale automatically
        out_dist = model.predict(
            X=X,
            Z=Z,
            W=W,
            A=A_input,
            y_train=y_obs_in,
            output_type="distribution",
            target="cate",
        )

    weights_y0 = out_dist["y0"]["weights"][0, TEST_SAMPLE_IDX].cpu().numpy()
    weights_y1 = out_dist["y1"]["weights"][0, TEST_SAMPLE_IDX].cpu().numpy()

    # --- CALCULATE CATE DISTRIBUTION (CONVOLUTION) ---
    weights_cate = np.convolve(weights_y1, weights_y0[::-1], mode="full")

    # Capture the bin values once
    if vals_y0 is None:
        vals_y0 = out_dist["y0"]["values"][0, 0].cpu().numpy()
        vals_y1 = out_dist["y1"]["values"][0, 0].cpu().numpy()

        # Calculate the new value grid for CATE
        min_cate = vals_y1[0] - vals_y0[-1]
        max_cate = vals_y1[-1] - vals_y0[0]
        vals_cate = np.linspace(min_cate, max_cate, len(weights_cate))

    history_weights_y0.append(weights_y0)
    history_weights_y1.append(weights_y1)
    history_weights_cate.append(weights_cate)

heatmap_y0 = np.array(history_weights_y0).T
heatmap_y1 = np.array(history_weights_y1).T
heatmap_cate = np.array(history_weights_cate).T

# %%
# 4. Plotting the Evolution Heatmaps
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))


def plot_evolution(ax, heatmap, vals, step_arr, true_val, title, y_lim=None):
    # IMPORTANT: Calculate the mathematical mean of the distribution BEFORE we
    # truncate the heatmap arrays. Otherwise the mean will be wrong!
    if PLOT_MEAN:
        step_means = np.sum(heatmap * vals[:, None], axis=0) / np.sum(heatmap, axis=0)

    # Clean dimension-safe slicing
    if y_lim is not None:
        # Find the bin indices that fall within our requested limits
        valid_idx = np.where((vals >= y_lim[0]) & (vals <= y_lim[1]))[0]

        if len(valid_idx) > 0:
            start_idx = valid_idx[0]
            end_idx = valid_idx[-1] + 1

            # Slice BOTH the y-values and the heatmap to keep dimensions identical
            vals = vals[start_idx:end_idx]
            heatmap = heatmap[start_idx:end_idx, :]

    # Now that the dims are safe and matched, create the meshgrid
    X_mesh, Y_mesh = np.meshgrid(step_arr, vals)

    # Log scale to make small probabilities visible
    # We add a tiny eps (1e-6) so we don't crash the LogNorm on truncated arrays
    norm = mcolors.LogNorm(vmin=1e-4, vmax=max(heatmap.max(), 1e-3))
    c = ax.pcolormesh(
        X_mesh, Y_mesh, heatmap, cmap="viridis", shading="nearest", norm=norm
    )
    fig.colorbar(c, ax=ax, label="Probability Mass (Log)")

    # Overlay the true value
    ax.axhline(
        y=true_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"True Value: {true_val:.2f}",
    )

    # Plot the pre-calculated mean
    if PLOT_MEAN:
        ax.plot(
            step_arr,
            step_means,
            color="white",
            marker=".",
            markersize=8,
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            label="Predicted Mean",
        )

    # Formatting
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Training Step")
    unit = "Normalized Value" if PLOT_NORMALIZED else "Raw Value"
    ax.set_ylabel(f"Predicted {unit}")

    # Enforce the limits on the matplotlib axis visually
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    ax.legend(loc="upper left")


# Plot Y0
plot_evolution(
    ax1,
    heatmap_y0,
    vals_y0,
    steps,
    true_y0_val,
    f"Y(0) Evolution\nSample {TEST_SAMPLE_IDX}",
    y_lim=Y_LIM,
)

# Plot Y1
plot_evolution(
    ax2,
    heatmap_y1,
    vals_y1,
    steps,
    true_y1_val,
    f"Y(1) Evolution\nSample {TEST_SAMPLE_IDX}",
    y_lim=Y_LIM,
)

# Plot CATE
# Note: Because CATE is a convolution, its min/max domain is naturally twice as wide.
# If you want to see the whole CATE distribution, you might want to multiply Y_LIM by 2 here.
plot_evolution(
    ax3,
    heatmap_cate,
    vals_cate,
    steps,
    true_cate_val,
    f"CATE [Y(1) - Y(0)] Evolution\nSample {TEST_SAMPLE_IDX}",
    y_lim=Y_LIM,
)

plt.tight_layout()
plt.show()
