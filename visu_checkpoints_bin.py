#%%

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from proximalfm.prior import PriorDataset
from proximalfm._model import TabICL

#%%
# 1. Configuration & Toggles
CHECKPOINT_DIR = "check_2/"
TEST_SAMPLE_IDX = 0
PLOT_MEAN = True

prior = PriorDataset(
    batch_size=1,
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
    max_seq_len=512,
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
U, X, W, Z, A, Y_obs, Y0, Y1, d_U, d_X, d_W, d_Z, seq_len, train_sizes = batch

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
y_obs_train = Y_obs[:, :split_idx]
A_input = A.unsqueeze(-1).float()

# Extract Test Ground Truths
y0_test = Y0[:, split_idx:]
y1_test = Y1[:, split_idx:]

# Extract the specific true values for the chosen sample
true_y0_val = y0_test[0, TEST_SAMPLE_IDX].item()
true_y1_val = y1_test[0, TEST_SAMPLE_IDX].item()

# Compute the true CATE value
true_cate_val = true_y1_val - true_y0_val

#%%
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

#%%
# 3. Predict the Distribution Across All Checkpoints
history_weights_y0 = []
history_weights_y1 = []
history_weights_cate = []

vals_y0, vals_y1, vals_cate = None, None, None

first_ckpt_path = os.path.join(CHECKPOINT_DIR, raw_checkpoints[0])
first_ckpt = torch.load(first_ckpt_path, map_location="cpu", weights_only=True)
model = TabICL(**first_ckpt["config"])

print(f"Tracking evolution across {len(raw_checkpoints)} checkpoints...")

for ckpt_name in raw_checkpoints:
    checkpoint = torch.load(
        os.path.join(CHECKPOINT_DIR, ckpt_name), map_location="cpu", weights_only=True
    )

    state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    with torch.inference_mode():
        out_dist = model.predict(
            X=X,
            Z=Z,
            W=W,
            A=A_input,
            y_train=y_obs_train,
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
        # The minimum possible difference is min(Y1) - max(Y0)
        # The maximum possible difference is max(Y1) - min(Y0)
        min_cate = vals_y1[0] - vals_y0[-1]
        max_cate = vals_y1[-1] - vals_y0[0]
        vals_cate = np.linspace(min_cate, max_cate, len(weights_cate))

    history_weights_y0.append(weights_y0)
    history_weights_y1.append(weights_y1)
    history_weights_cate.append(weights_cate)

heatmap_y0 = np.array(history_weights_y0).T
heatmap_y1 = np.array(history_weights_y1).T
heatmap_cate = np.array(history_weights_cate).T

#%%
# 4. Plotting the Evolution Heatmaps
# Expanded to 3 columns to fit the new CATE plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

keep_only_quartiles = True


def plot_evolution(ax, heatmap, vals, step_arr, true_val, title):
    X_mesh, Y_mesh = np.meshgrid(step_arr, vals)

    print(X_mesh.shape, Y_mesh.shape, heatmap.shape)

    if keep_only_quartiles:
        X_mesh = X_mesh[X_mesh.shape[0] // 4 : 3 * X_mesh.shape[0] // 4, :]
        Y_mesh = Y_mesh[Y_mesh.shape[0] // 4 : 3 * Y_mesh.shape[0] // 4, :]
        heatmap = heatmap[heatmap.shape[0] // 4 : 3 * heatmap.shape[0] // 4, :]
        vals = vals[vals.shape[0] // 4 : 3 * vals.shape[0] // 4]

    print(X_mesh.shape, Y_mesh.shape, heatmap.shape)

    # Log scale to make small probabilities visible
    norm = mcolors.LogNorm(vmin=1e-4, vmax=heatmap.max())
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

    # Calculate and plot the mean
    if PLOT_MEAN:
        step_means = np.sum(heatmap * vals[:, None], axis=0) / np.sum(heatmap, axis=0)
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
    ax.set_ylabel("Predicted Value")
    ax.legend(loc="upper left")

    # Zoom in
    active_bins = np.any(heatmap > 1e-4, axis=1)
    if np.any(active_bins):
        active_indices = np.where(active_bins)[0]
        min_idx = max(0, active_indices[0] - 50)
        max_idx = min(len(vals) - 1, active_indices[-1] + 50)
        ax.set_ylim(vals[min_idx], vals[max_idx])


# Plot Y0
plot_evolution(
    ax1,
    heatmap_y0,
    vals_y0,
    steps,
    true_y0_val,
    f"Y(0) Evolution\nSample {TEST_SAMPLE_IDX}",
)

# Plot Y1
plot_evolution(
    ax2,
    heatmap_y1,
    vals_y1,
    steps,
    true_y1_val,
    f"Y(1) Evolution\nSample {TEST_SAMPLE_IDX}",
)

# Plot CATE
plot_evolution(
    ax3,
    heatmap_cate,
    vals_cate,
    steps,
    true_cate_val,
    f"CATE [Y(1) - Y(0)] Evolution\nSample {TEST_SAMPLE_IDX}",
)

plt.tight_layout()
plt.show()
