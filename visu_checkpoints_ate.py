#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
from proximalfm.prior import PriorDataset

#%%

prior = PriorDataset(
    batch_size=1,
    batch_size_per_gp=1,
    min_dim_U=1,
    max_dim_U=1,
    min_dim_W=1,
    max_dim_W=2,
    min_dim_Z=1,
    max_dim_Z=2,
    min_dim_X=1,
    max_dim_X=3,
    min_seq_len=None,
    max_seq_len=256,
    log_seq_len=False,
    seq_len_per_gp=False,
    min_train_size=0.8,
    max_train_size=0.95,
    replay_small=False,
    prior_type="mlp_scm",
    device="cpu",
)

# 1. Generate MANY datasets at once
num_datasets = 50
batch = prior.get_batch(batch_size=num_datasets)

U, X, W, Z, A, Y_obs, Y0, Y1, d_U, d_X, d_W, d_Z, seq_len, train_sizes = batch

if len(Y0.shape) == 1:
    Y0 = Y0.unsqueeze(0)
if len(Y1.shape) == 1:
    Y1 = Y1.unsqueeze(0)
if len(A.shape) == 1:
    A = A.unsqueeze(0)
if len(Y_obs.shape) == 1:
    Y_obs = Y_obs.unsqueeze(0)

# We use the train split boundary from the first dataset for the whole batch
split_idx = train_sizes[0]

y_obs_train = Y_obs[:, :split_idx]

# 2. Calculate the True ATE for each dataset by averaging True CATE (Y1 - Y0) on the test set
y0_test = Y0[:, split_idx:]
y1_test = Y1[:, split_idx:]
true_cate_test = y1_test - y0_test
true_ate = true_cate_test.mean(dim=1)  # Shape: (num_datasets,)

#%%

checkpoint_path = "check_1/"

all_checkpoints = [
    f
    for f in os.listdir(checkpoint_path)
    if f.startswith("step-") and f.endswith(".ckpt")
]

all_checkpoints.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))

for checkp in all_checkpoints:
    print(os.path.join(checkpoint_path, checkp))

first_checkpoint = torch.load(
    os.path.join(checkpoint_path, all_checkpoints[0]), map_location="cpu"
)

#%%
from proximalfm._model import TabICL

model = TabICL(**first_checkpoint["config"])

# figures: 4 columns, and enough rows to fit all checkpoints
num_checkpoints = len(all_checkpoints)
num_cols = 4
num_rows = (num_checkpoints + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

# Ensure axes is always iterable even if there's only 1 checkpoint
if num_checkpoints == 1:
    axes = np.array([axes])
else:
    axes = axes.flatten()

for i, checkp in enumerate(all_checkpoints):
    checkpoint = torch.load(os.path.join(checkpoint_path, checkp), map_location="cpu")
    print(f"Checkpoint: {checkp}, Step: {checkpoint['curr_step']}")

    model.load_state_dict(checkpoint["state_dict"])

    # Predict over the entire batch of datasets
    out = model.predict(
        X,
        W,
        Z,
        A.unsqueeze(-1).float(),
        y_train=y_obs_train,
        output_type="mean",
        target="cate",
    )  # Remove the [0] index so we keep all datasets. Shape: (num_datasets, test_len)

    # 3. Average the predicted CATEs over the test set points to get Predicted ATE
    pred_ate = out.mean(dim=1)  # Shape: (num_datasets,)

    # 4. Plot True ATE vs Predicted ATE
    axes[i].scatter(true_ate.cpu().numpy(), pred_ate.detach().cpu().numpy())

    # Calculate limits for the ideal diagonal line
    min_val = min(true_ate.min().item(), pred_ate.min().item())
    max_val = max(true_ate.max().item(), pred_ate.max().item())

    axes[i].plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    axes[i].set_title(f"{checkp}")
    axes[i].set_xlabel("True ATE (Average CATE)")
    axes[i].set_ylabel("Predicted ATE")

plt.tight_layout()
plt.show()

#%%
