#%%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from proximalfm.prior import PriorDataset
from proximalfm._model import TabICL

#%%
# 1. Configuration & Generation
checkpoint_path = "check_1/"

KEEP_CHECK = None
REMOVE_CHECK = []
PLOT_NORMALIZED = False
REMOVE_OUTLIERS = True

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

# Fetch batch
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

# Extract train split
train_idx = train_sizes[0].item()
y_obs_train = Y_obs[:, :train_idx]
A_input = A.unsqueeze(-1).float()

y0_test = Y0[:, train_idx:]
y1_test = Y1[:, train_idx:]
true_cate_test = y1_test - y0_test
true_target = true_cate_test[0]  # Focus on the first batch item

#%%
# 2. Locate and Filter Checkpoints
raw_checkpoints = [
    f
    for f in os.listdir(checkpoint_path)
    if f.startswith("step-") and f.endswith(".ckpt")
]

all_checkpoints = []
for ckpt in raw_checkpoints:
    step = int(ckpt.split("-")[1].split(".")[0])

    # Apply filtering logic
    if REMOVE_CHECK and step in REMOVE_CHECK:
        continue
    if KEEP_CHECK is not None and step not in KEEP_CHECK:
        continue

    all_checkpoints.append((step, ckpt))

# Sort by step number
all_checkpoints.sort(key=lambda x: x[0])
all_checkpoints = [ckpt for step, ckpt in all_checkpoints]

if not all_checkpoints:
    raise ValueError("No checkpoints found after applying filters.")

# Initialize model using the config from the first valid checkpoint
first_checkpoint = torch.load(
    os.path.join(checkpoint_path, all_checkpoints[0]),
    map_location="cpu",
    weights_only=True,
)
model = TabICL(**first_checkpoint["config"])
model.eval()

#%%
# 3. Standardization (if requested)
scale = 1.0
if PLOT_NORMALIZED:
    scale = y_obs_train.std().item() + 1e-8
    true_target = true_target / scale

#%%
# 4. Setup Plotting Grids
num_checkpoints = len(all_checkpoints)
num_cols = min(4, num_checkpoints)
num_rows = (num_checkpoints + num_cols - 1) // num_cols

# Figure 1: Scatter Plots (True CATE vs Predicted)
fig1, axes1 = plt.subplots(
    num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), squeeze=False
)
axes1 = axes1.flatten()

# 5. Evaluate and Plot
for i, checkp in enumerate(all_checkpoints):
    ckpt_full_path = os.path.join(checkpoint_path, checkp)

    # Catch Race Condition Errors
    try:
        checkpoint = torch.load(ckpt_full_path, map_location="cpu", weights_only=True)
    except RuntimeError:
        print(f"Skipping {checkp}: File incomplete or corrupted.")
        continue

    step = checkpoint["curr_step"]
    print(f"Plotting Checkpoint: {checkp}, Step: {step}")

    # Strip 'module.' prefix if saved via DDP
    state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    with torch.inference_mode():
        # 1. Get Mean CATE
        out_mean = model.predict(
            X=X,
            Z=Z,
            W=W,
            A=A_input,
            y_train=y_obs_train,
            output_type="mean",
            target="cate",
        )[0]

        # 2. Get Distribution Weights
        out_dist = model.predict(
            X=X,
            Z=Z,
            W=W,
            A=A_input,
            y_train=y_obs_train,
            output_type="distribution",
            target="cate",
        )

    # Normalization check
    if PLOT_NORMALIZED:
        out_mean = out_mean / scale

    # Convert to numpy
    true_np = true_target.cpu().numpy()
    out_np = out_mean.cpu().numpy()
    mse = np.mean((true_np - out_np) ** 2)

    # --- Plot 1: Scatter (CATE) ---
    ax1 = axes1[i]
    ax1.scatter(true_np, out_np, alpha=0.7, edgecolors="none")

    if REMOVE_OUTLIERS and PLOT_NORMALIZED:
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        line_min, line_max = -3, 3
    else:
        line_min = min(true_np.min(), out_np.min())
        line_max = max(true_np.max(), out_np.max())

    ax1.plot([line_min, line_max], [line_min, line_max], color="red", linestyle="--")

    ax1.set_title(f"Step {step} | MSE: {mse:.3f}")
    unit = "(Normalized)" if PLOT_NORMALIZED else "(Raw)"
    ax1.set_xlabel(f"True CATE {unit}")
    ax1.set_ylabel(f"Predicted CATE {unit}")
    ax1.grid(True, alpha=0.3)


# Clean up empty subplots on both grids
for j in range(num_checkpoints, len(axes1)):
    fig1.delaxes(axes1[j])

fig1.tight_layout()
plt.show()
