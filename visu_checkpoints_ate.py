# %%
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from proximalfm.prior import PriorDataset
from proximalfm._model import ProximalFM

# %%
# 1. Configuration & Toggles
checkpoint_path = "mc_baseline/"

KEEP_CHECK = None
REMOVE_CHECK = []
PLOT_NORMALIZED = True
REMOVE_OUTLIERS = True

# %%
# 2. Generate MANY datasets at once
prior = PriorDataset(
    batch_size=1,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_W=1,
    max_dim_W=1,
    min_dim_Z=1,
    max_dim_Z=1,
    min_dim_X=1,
    max_dim_X=1,
    min_seq_len=None,
    max_seq_len=65,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.8,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    device="cpu",
)

num_datasets = 2
batch = prior.get_batch(batch_size=num_datasets)
(
    U,
    X,
    W,
    Z,
    A,
    A_probs,
    Y_obs,
    Y0,
    Y1,
    MU0,
    MU1,
    d_U,
    d_X,
    d_W,
    d_Z,
    seq_len,
    train_sizes,
) = batch

# Ensure batch dimension exists
if len(MU0.shape) == 1:
    MU0 = MU0.unsqueeze(0)
if len(MU1.shape) == 1:
    MU1 = MU1.unsqueeze(0)
if len(A.shape) == 1:
    A = A.unsqueeze(0)
if len(Y_obs.shape) == 1:
    Y_obs = Y_obs.unsqueeze(0)

# We use the train split boundary from the first dataset for the whole batch
split_idx = train_sizes[0].item()
A_input = A.unsqueeze(-1).float()

# --- NEW NORMALIZATION STRATEGY ---
# 1. Extract context
y_obs_train_raw = Y_obs[:, :split_idx]

# 2. Calculate Context Stats
y_context_mean = y_obs_train_raw.mean(dim=1, keepdim=True)
y_context_std = y_obs_train_raw.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-5)

# 3. Standardize Context
y_obs_train_norm = (y_obs_train_raw - y_context_mean) / y_context_std

# 4. Calculate Raw True ATE
# CATE(x) = mu1(x) - mu0(x) is the noise-free target the model is trained on
# (see _run.py), so ground truth comes from MU0/MU1, not the noisy Y0/Y1 (ITE).
mu0_test_raw = MU0[:, split_idx:]
mu1_test_raw = MU1[:, split_idx:]
true_cate_test_raw = mu1_test_raw - mu0_test_raw
true_ate_raw = true_cate_test_raw.mean(dim=1)  # Shape: (num_datasets,)

# 5. Calculate Normalized True ATE (Using Context Stats)
mu0_test_norm = (mu0_test_raw - y_context_mean) / y_context_std
mu1_test_norm = (mu1_test_raw - y_context_mean) / y_context_std
true_cate_test_norm = mu1_test_norm - mu0_test_norm
true_ate_norm = true_cate_test_norm.mean(dim=1)  # Shape: (num_datasets,)

# Route the inputs based on PLOT_NORMALIZED flag
if PLOT_NORMALIZED:
    y_obs_in = y_obs_train_norm
    true_target = true_ate_norm
else:
    y_obs_in = y_obs_train_raw
    true_target = true_ate_raw

# Note: We pass X, Z, and W as raw tensors because the updated `model.predict()`
# natively standardizes covariates inside the method!

# %%
# 3. Locate and Filter Checkpoints
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
    weights_only=True,
)
model = ProximalFM(**first_checkpoint["config"])
model.eval()

# %%
# 4. Setup Plotting Grid
num_checkpoints = len(all_checkpoints)
num_cols = min(4, num_checkpoints)
num_rows = (num_checkpoints + num_cols - 1) // num_cols

fig, axes = plt.subplots(
    num_rows, num_cols, figsize=(4 * num_cols, 5 * num_rows), squeeze=False
)
axes = axes.flatten()

# 5. Evaluate and Plot
for i, checkp in enumerate(all_checkpoints):
    ckpt_full_path = os.path.join(checkpoint_path, checkp)

    try:
        checkpoint = torch.load(ckpt_full_path, map_location="cpu", weights_only=True)
    except RuntimeError:
        print(f"Skipping {checkp}: File incomplete or corrupted.")
        continue

    step = checkpoint["curr_step"]
    print(f"Plotting Checkpoint: {checkp}, Step: {step}")

    state_dict = {
        k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    with torch.inference_mode():
        # Predict over the entire batch of datasets
        out = model.predict(
            X=X,  # model handles covariate standardization internally
            Z=Z,
            W=W,
            A=A_input,
            y_train=y_obs_in,  # Routes raw or normalized context seamlessly
            output_type="mean",
            target="cate",
            d_X=d_X,
            d_Z=d_Z,
            d_W=d_W,
        )

        # Average the predicted CATEs over the test set points to get Predicted ATE
        pred_ate = out.mean(dim=1)  # Shape: (num_datasets,)

    # Convert to numpy
    true_np = true_target.cpu().numpy()
    pred_np = pred_ate.cpu().numpy()

    # Calculate R-squared instead of just MSE
    mse = np.mean((true_np - pred_np) ** 2)
    variance = np.var(true_np)

    # Guard against division by zero if a synthetic batch happens to have zero variance
    if variance > 1e-8:
        r2 = 1.0 - (mse / variance)
    else:
        r2 = 1.0 if mse < 1e-8 else 0.0

    # Plot True ATE vs Predicted ATE
    ax = axes[i]
    ax.scatter(true_np, pred_np, alpha=0.7, edgecolors="none")

    if REMOVE_OUTLIERS and PLOT_NORMALIZED:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        line_min, line_max = -3, 3
    else:
        line_min = min(true_np.min(), pred_np.min())
        line_max = max(true_np.max(), pred_np.max())

    ax.plot([line_min, line_max], [line_min, line_max], color="red", linestyle="--")

    # Formatting
    unit = "(Normalized)" if PLOT_NORMALIZED else "(Raw)"

    # Update title to show R2
    ax.set_title(f"Step {step} | R²: {r2:.3f}")
    ax.set_xlabel(f"True ATE {unit}")
    ax.set_ylabel(f"Predicted ATE {unit}")
    ax.grid(True, alpha=0.3)

# Clean up empty subplots
for j in range(num_checkpoints, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
