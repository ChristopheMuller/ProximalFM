# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from proximalfm.prior import PriorDataset
from proximalfm._model import TabICL

# %%
# 1. Configuration & Generation
TARGET = "y1"
checkpoint_path = "check_3/"

KEEP_CHECK = None
REMOVE_CHECK = None
PLOT_NORMALIZED = True
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

true_test_target_raw = y0_test_raw[0] if TARGET == "y0" else y1_test_raw[0]

# Route based on flag
if PLOT_NORMALIZED:
    y_obs_in = y_obs_train_norm

    # Standardize the true values so the plot axes match
    mean_val = y_context_mean[0, 0].item()
    std_val = y_context_std[0, 0].item()
    true_test_target = (true_test_target_raw - mean_val) / std_val
else:
    y_obs_in = y_obs_train_raw
    true_test_target = true_test_target_raw

# %%
# 2. Locate and Filter Checkpoints
raw_checkpoints = [
    f
    for f in os.listdir(checkpoint_path)
    if f.startswith("step-") and f.endswith(".ckpt")
]

all_checkpoints = []
for ckpt in raw_checkpoints:
    step = int(ckpt.split("-")[1].split(".")[0])

    if REMOVE_CHECK and step in REMOVE_CHECK:
        continue
    if KEEP_CHECK is not None and step not in KEEP_CHECK:
        continue

    all_checkpoints.append((step, ckpt))

all_checkpoints.sort(key=lambda x: x[0])
all_checkpoints = [ckpt for step, ckpt in all_checkpoints]

if not all_checkpoints:
    raise ValueError("No checkpoints found after applying filters.")

first_checkpoint = torch.load(
    os.path.join(checkpoint_path, all_checkpoints[0]),
    map_location="cpu",
    weights_only=False,
)
model = TabICL(**first_checkpoint["config"])
model.eval()

# %%
# 3. Setup Plotting Grid
num_checkpoints = len(all_checkpoints)
num_cols = min(4, num_checkpoints)
num_rows = (num_checkpoints + num_cols - 1) // num_cols

fig, axes = plt.subplots(
    num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), squeeze=False
)
axes = axes.flatten()

# 4. Evaluate and Plot
for i, checkp in enumerate(all_checkpoints):
    ckpt_full_path = os.path.join(checkpoint_path, checkp)

    try:
        checkpoint = torch.load(ckpt_full_path, map_location="cpu", weights_only=True)
    except RuntimeError as e:
        print(
            f"Skipping {checkp}: File incomplete or corrupted (likely being written right now)."
        )
        continue

    step = checkpoint["curr_step"]
    print(f"Plotting Checkpoint: {checkp}, Step: {step}")

    # Strip 'module.' prefix if saved via DDP
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    with torch.inference_mode():
        # Predict natively dictates the output scale based on y_obs_in!
        out = model.predict(
            X=X,  # Handled internally by predict
            Z=Z,
            W=W,
            A=A_input,
            y_train=y_obs_in,
            output_type="mean",
            target=TARGET,
        )[0]

    # Convert to numpy
    true_np = true_test_target.cpu().numpy()
    out_np = out.cpu().numpy()

    # Calculate R2 (and MSE for reference)
    mse = np.mean((true_np - out_np) ** 2)
    variance = np.var(true_np)

    if variance > 1e-8:
        r2 = 1.0 - (mse / variance)
    else:
        r2 = 1.0 if mse < 1e-8 else 0.0

    ax = axes[i]
    ax.scatter(true_np, out_np, alpha=0.7, edgecolors="none")

    # Determine bounds for the y=x reference line
    if REMOVE_OUTLIERS and PLOT_NORMALIZED:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        line_min, line_max = -3, 3
    else:
        line_min = min(true_np.min(), out_np.min())
        line_max = max(true_np.max(), out_np.max())

    ax.plot([line_min, line_max], [line_min, line_max], color="red", linestyle="--")

    # Formatting
    ax.set_title(f"Step {step} | R²: {r2:.3f}")
    unit_label = "(Normalized)" if PLOT_NORMALIZED else "(Raw)"
    ax.set_xlabel(f"True {TARGET.upper()} {unit_label}")
    ax.set_ylabel(f"Predicted {TARGET.upper()} {unit_label}")
    ax.grid(True, alpha=0.3)

# Clean up any empty subplots
for j in range(num_checkpoints, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
